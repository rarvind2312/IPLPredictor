"""SQLite persistence for parsed matches and learned signals."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import audit_profile
import config
import utils

logger = logging.getLogger(__name__)
_SCHEMA_BOOTSTRAP_LOCK = threading.Lock()
_SCHEMA_BOOTSTRAP_SIG: Optional[tuple[str, int, int, int]] = None


def db_runtime_signature() -> tuple[str, int, int, int]:
    """
    Cheap signature for in-process read caches.

    Returns (db_path, exists_flag, size_bytes, mtime_ns).
    """
    p = Path(config.DB_PATH).resolve()
    try:
        st = p.stat()
        return (str(p), 1, int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
    except OSError:
        return (str(p), 0, 0, 0)


def _migrate_match_results_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(match_results)").fetchall()}
    if "canonical_match_key" not in cols:
        try:
            conn.execute("ALTER TABLE match_results ADD COLUMN canonical_match_key TEXT")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_match_results_canonical_key ON match_results(canonical_match_key)"
    )


def _backfill_canonical_history_keys_if_needed(conn: sqlite3.Connection) -> None:
    """
    One-time migration: recompute ``team_key`` / ``player_key`` on history tables using the same
    normalization as current squad + Cricsheet ingest, then rebuild ``player_franchise_features``.
    """
    uv_row = conn.execute("PRAGMA user_version").fetchone()
    if int(uv_row[0] if uv_row is not None else 0) >= 2:
        return

    import ipl_teams
    import learner
    import matchup_features

    def _tk_for_team_name(tn: str) -> str:
        lab = ipl_teams.franchise_label_for_storage(tn) or (tn or "").strip()
        return ipl_teams.canonical_team_key_for_franchise(lab)[:80]

    def _pk_for_player(pn: str) -> str:
        return learner.normalize_player_key(pn)

    for table in ("team_match_xi", "player_match_stats", "player_batting_positions", "player_phase_usage"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row:
            continue
        cols_t = {c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        has_canon = "canonical_team_key" in cols_t and "canonical_player_key" in cols_t
        rows = conn.execute(
            f"SELECT id, team_name, player_name FROM {table} WHERE id IS NOT NULL"
        ).fetchall()
        for r in rows:
            tid = int(r["id"])
            tk = _tk_for_team_name(str(r["team_name"] or ""))
            pk = _pk_for_player(str(r["player_name"] or ""))
            if has_canon:
                conn.execute(
                    f"UPDATE {table} SET team_key = ?, player_key = ?, "
                    f"canonical_team_key = ?, canonical_player_key = ? WHERE id = ?",
                    (tk, pk, tk, pk, tid),
                )
            else:
                conn.execute(
                    f"UPDATE {table} SET team_key = ?, player_key = ? WHERE id = ?",
                    (tk, pk, tid),
                )

    sm = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='team_match_summary'"
    ).fetchone()
    if sm:
        for r in conn.execute("SELECT id, team_name FROM team_match_summary").fetchall():
            tk = _tk_for_team_name(str(r["team_name"] or ""))
            conn.execute("UPDATE team_match_summary SET team_key = ? WHERE id = ?", (tk, int(r["id"])))

    conn.execute("DELETE FROM player_franchise_features")
    tkeys = [
        str(x[0]).strip()[:80]
        for x in conn.execute(
            "SELECT DISTINCT team_key FROM player_match_stats WHERE team_key IS NOT NULL AND trim(team_key) != ''"
        ).fetchall()
    ]
    for tk in sorted(set(tkeys)):
        if tk:
            matchup_features.refresh_franchise_features(conn, tk)

    conn.execute("PRAGMA user_version = 2")
    logger.info("SQLite PRAGMA user_version set to 2 (canonical history key backfill applied).")


def _migrate_player_batting_positions_innings(conn: sqlite3.Connection) -> None:
    """
    Add per-innings batting rows (super overs) via ``innings_number`` and a stricter UNIQUE key.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_batting_positions'"
    ).fetchone()
    if not row:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_batting_positions)").fetchall()}
    if "innings_number" in cols:
        return
    conn.execute("ALTER TABLE player_batting_positions RENAME TO player_batting_positions_legacy")
    conn.execute(
        """
        CREATE TABLE player_batting_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            batting_position REAL,
            season TEXT,
            innings_number INTEGER NOT NULL DEFAULT 1,
            UNIQUE(match_id, team_name, player_key, innings_number),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO player_batting_positions (
            match_id, team_name, team_key, player_name, player_key,
            canonical_team_key, canonical_player_key,
            batting_position, season, innings_number
        )
        SELECT match_id, team_name, team_key, player_name, player_key,
               team_key, player_key,
               batting_position, season, 1
        FROM player_batting_positions_legacy
        """
    )
    conn.execute("DROP TABLE player_batting_positions_legacy")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pbp_match_innings ON player_batting_positions(match_id, innings_number)"
    )


def _migrate_matches_ingest_metadata_columns(conn: sqlite3.Connection) -> None:
    """Add Cricsheet / ingest metadata columns to ``matches`` (idempotent)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='matches'"
    ).fetchone()
    if not row:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    additions = [
        ("cricsheet_match_id", "TEXT"),
        ("city", "TEXT"),
        ("season", "TEXT"),
        ("toss_winner", "TEXT"),
        ("toss_decision", "TEXT"),
        ("winner", "TEXT"),
        ("result_text", "TEXT"),
        ("match_format", "TEXT"),
    ]
    for col, sqlt in additions:
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {sqlt}")
            except sqlite3.OperationalError:
                pass
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_matches_cricsheet_id ON matches(cricsheet_match_id)"
    )


def _migrate_player_recent_form_summaries_and_cache(conn: sqlite3.Connection) -> None:
    """Archive batting/bowling summaries, role-phase usage, and global T20 recent-form cache."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_batting_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_key TEXT NOT NULL,
            team_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            position INTEGER,
            runs INTEGER,
            balls INTEGER,
            fours INTEGER,
            sixes INTEGER,
            strike_rate REAL,
            dismissal_type TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
            UNIQUE(match_id, team_name, player_name)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmbat_match ON player_match_batting_summary(match_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmbat_player ON player_match_batting_summary(player_key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_bowling_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_key TEXT NOT NULL,
            team_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            overs_bowled REAL,
            maidens INTEGER,
            wickets INTEGER,
            runs_conceded INTEGER,
            economy REAL,
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
            UNIQUE(match_id, team_name, player_name)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmbowl_match ON player_match_bowling_summary(match_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmbowl_player ON player_match_bowling_summary(player_key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_role_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL,
            phase TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            wickets INTEGER NOT NULL DEFAULT 0,
            vs_spin_balls INTEGER NOT NULL DEFAULT 0,
            vs_pace_balls INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
            UNIQUE(match_id, team_name, player_name, role, phase)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmrole_match ON player_match_role_usage(match_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pmrole_player ON player_match_role_usage(player_key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_recent_form_cache (
            player_key TEXT PRIMARY KEY,
            last_updated REAL NOT NULL,
            reference_as_of_date TEXT,
            t20_matches_in_window INTEGER NOT NULL DEFAULT 0,
            batting_recent_form REAL,
            bowling_recent_form REAL,
            combined_recent_form REAL,
            last_t20_match_date TEXT,
            competitions_json TEXT,
            matches_last_30d INTEGER,
            matches_last_60d INTEGER,
            matches_last_150d INTEGER,
            recent_batting_position_ema REAL,
            bowling_pp_ball_share REAL,
            bowling_middle_ball_share REAL,
            bowling_death_ball_share REAL,
            sample_confidence REAL,
            debug_json TEXT
        )
        """
    )


def _migrate_drop_prediction_runtime_summary_tables(conn: sqlite3.Connection) -> None:
    """Remove optional runtime-summary tables added in a later experiment (idempotent)."""
    conn.execute("DROP TABLE IF EXISTS player_venue_summary")
    conn.execute("DROP TABLE IF EXISTS player_ipl_role_summary")
    conn.execute("DROP TABLE IF EXISTS player_ipl_history_summary")
    conn.execute("DROP INDEX IF EXISTS idx_matches_match_date_id")


def _migrate_prediction_aggregate_summary_tables(conn: sqlite3.Connection) -> None:
    """Prediction-time aggregate summaries to avoid raw-table GROUP BY during inference."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_xi_counts (
            player_name TEXT PRIMARY KEY,
            xi_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_batting (
            player_name TEXT PRIMARY KEY,
            avg_position REAL NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_bowling (
            player_name TEXT PRIMARY KEY,
            avg_balls REAL NOT NULL DEFAULT 0,
            match_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_venue_team_player_xi (
            venue TEXT NOT NULL,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            xi_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (venue, team, player_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_match_team_venue (
            match_id INTEGER NOT NULL,
            venue TEXT NOT NULL,
            team TEXT NOT NULL,
            PRIMARY KEY (match_id, team)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_match_meta (
            match_id INTEGER PRIMARY KEY,
            winner TEXT,
            team_a TEXT,
            team_b TEXT,
            venue TEXT,
            batting_first TEXT,
            created_at REAL,
            match_date TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_global_xi (
            player_key TEXT PRIMARY KEY,
            tmx_rows INTEGER NOT NULL DEFAULT 0,
            distinct_matches INTEGER NOT NULL DEFAULT 0,
            distinct_teams INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_team_xi (
            player_key TEXT NOT NULL,
            team_key TEXT NOT NULL,
            tmx_rows INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (player_key, team_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_player_global_slot (
            player_key TEXT PRIMARY KEY,
            slot_ema REAL NOT NULL DEFAULT 0,
            slot_samples INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_summary_rebuild_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_rebuilt_at REAL NOT NULL DEFAULT 0,
            match_xi_rows INTEGER NOT NULL DEFAULT 0,
            match_batting_rows INTEGER NOT NULL DEFAULT 0,
            match_results_rows INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ps_match_meta_date_id "
        "ON prediction_summary_match_meta(match_date DESC, match_id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ps_match_meta_teams "
        "ON prediction_summary_match_meta(team_a, team_b)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ps_player_team_xi_team "
        "ON prediction_summary_player_team_xi(team_key)"
    )


def _migrate_player_profiles_derive_schema(conn: sqlite3.Connection) -> None:
    """Rebuild ``player_profiles`` for Stage 2 composite PK (player + franchise) if legacy schema."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_profiles'"
    ).fetchone()
    if not row:
        conn.execute(
            """
            CREATE TABLE player_profiles (
                player_key TEXT NOT NULL,
                franchise_team_key TEXT NOT NULL,
                display_name_hint TEXT,
                xi_selection_frequency REAL,
                batting_position_ema REAL,
                opener_likelihood REAL,
                middle_order_likelihood REAL,
                finisher_likelihood REAL,
                powerplay_bowler_likelihood REAL,
                middle_overs_bowler_likelihood REAL,
                death_bowler_likelihood REAL,
                batting_aggressor_score REAL,
                bowling_control_score REAL,
                batting_vs_spin_tendency REAL,
                batting_vs_pace_tendency REAL,
                venue_fit_score REAL,
                role_stability_score REAL,
                recent_usage_score REAL,
                h2h_basis_json TEXT,
                profile_confidence REAL,
                sample_matches INTEGER NOT NULL DEFAULT 0,
                last_updated REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (player_key, franchise_team_key)
            )
            """
        )
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_profiles)").fetchall()}
    if "franchise_team_key" in cols and "xi_selection_frequency" in cols:
        return
    conn.execute("ALTER TABLE player_profiles RENAME TO player_profiles__legacy_derive")
    conn.execute(
        """
        CREATE TABLE player_profiles (
            player_key TEXT NOT NULL,
            franchise_team_key TEXT NOT NULL,
            display_name_hint TEXT,
            xi_selection_frequency REAL,
            batting_position_ema REAL,
            opener_likelihood REAL,
            middle_order_likelihood REAL,
            finisher_likelihood REAL,
            powerplay_bowler_likelihood REAL,
            middle_overs_bowler_likelihood REAL,
            death_bowler_likelihood REAL,
            batting_aggressor_score REAL,
            bowling_control_score REAL,
            batting_vs_spin_tendency REAL,
            batting_vs_pace_tendency REAL,
            venue_fit_score REAL,
            role_stability_score REAL,
            recent_usage_score REAL,
            h2h_basis_json TEXT,
            profile_confidence REAL,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (player_key, franchise_team_key)
        )
        """
    )
    conn.execute("DROP TABLE player_profiles__legacy_derive")


def _migrate_stage1_canonical_alias_columns(conn: sqlite3.Connection) -> None:
    """
    Add ``canonical_team_key`` / ``canonical_player_key`` (mirrors ``team_key`` / ``player_key``)
    for Stage 1 audit / linkage clarity.
    """
    specs: list[tuple[str, tuple[str, str]]] = [
        ("team_match_xi", ("team_key", "player_key")),
        ("player_match_stats", ("team_key", "player_key")),
        ("player_batting_positions", ("team_key", "player_key")),
        ("player_phase_usage", ("team_key", "player_key")),
    ]
    for table, (_tk, _pk) in specs:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row:
            continue
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "canonical_team_key" not in cols:
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN canonical_team_key TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass
        if "canonical_player_key" not in cols:
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN canonical_player_key TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass
        try:
            conn.execute(
                f"""
                UPDATE {table}
                SET canonical_team_key = team_key,
                    canonical_player_key = player_key
                """
            )
        except sqlite3.OperationalError:
            pass


def _migrate_player_aliases_table(conn: sqlite3.Connection) -> None:
    """Create ``player_aliases`` on older DB files that predate the table in ``executescript``."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_aliases'"
    ).fetchone()
    if row:
        return
    conn.execute(
        """
        CREATE TABLE player_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            franchise_team_key TEXT NOT NULL,
            squad_full_name TEXT NOT NULL,
            normalized_full_name_key TEXT NOT NULL,
            resolved_history_key TEXT,
            resolution_type TEXT NOT NULL,
            confidence REAL,
            ambiguous_candidates_json TEXT,
            updated_at REAL NOT NULL,
            UNIQUE(franchise_team_key, normalized_full_name_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_player_aliases_franchise ON player_aliases(franchise_team_key)"
    )


def _migrate_team_selection_derive_columns(conn: sqlite3.Connection) -> None:
    """Add Stage 2 columns to ``team_selection_patterns`` when missing."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='team_selection_patterns'"
    ).fetchone()
    if not row:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(team_selection_patterns)").fetchall()}
    for col, sqlt in (
        ("preferred_xi_core_json", "TEXT"),
        ("keeper_consistency", "REAL"),
        ("opener_stability", "REAL"),
        ("finisher_stability", "REAL"),
        ("bowling_composition_json", "TEXT"),
        ("chase_defend_json", "TEXT"),
    ):
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE team_selection_patterns ADD COLUMN {col} {sqlt}")
            except sqlite3.OperationalError:
                pass


def _migrate_player_metadata_and_matchup_tables(conn: sqlite3.Connection) -> None:
    """Create Phase-2 player metadata + matchup summary tables (idempotent)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_metadata (
            player_key TEXT PRIMARY KEY,
            display_name TEXT,
            batting_hand TEXT NOT NULL DEFAULT 'unknown',
            bowling_style_raw TEXT NOT NULL DEFAULT '',
            bowling_type_bucket TEXT NOT NULL DEFAULT 'unknown',
            primary_role TEXT NOT NULL DEFAULT 'batter',
            secondary_role TEXT NOT NULL DEFAULT '',
            likely_batting_band TEXT NOT NULL DEFAULT 'unknown',
            likely_bowling_phases TEXT NOT NULL DEFAULT 'unknown',
            source TEXT NOT NULL DEFAULT 'derived_history',
            confidence REAL NOT NULL DEFAULT 0.0,
            last_updated REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_player_metadata_role ON player_metadata(primary_role, likely_batting_band)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batter_bowler_matchup_summary (
            batter_key TEXT NOT NULL,
            bowler_key TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            dismissals INTEGER NOT NULL DEFAULT 0,
            strike_rate REAL NOT NULL DEFAULT 0.0,
            dot_ball_pct REAL NOT NULL DEFAULT 0.0,
            boundary_pct REAL NOT NULL DEFAULT 0.0,
            innings_count INTEGER NOT NULL DEFAULT 0,
            match_count INTEGER NOT NULL DEFAULT 0,
            last_match_date TEXT,
            sample_size_confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'direct_delivery_history',
            last_updated REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (batter_key, bowler_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bbms_bowler ON batter_bowler_matchup_summary(bowler_key, sample_size_confidence DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batter_vs_bowling_type_summary (
            batter_key TEXT NOT NULL,
            bowling_type_bucket TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            dismissals REAL NOT NULL DEFAULT 0.0,
            strike_rate REAL NOT NULL DEFAULT 0.0,
            dot_ball_pct REAL NOT NULL DEFAULT 0.0,
            boundary_pct REAL NOT NULL DEFAULT 0.0,
            sample_size_confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'derived_history',
            last_updated REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (batter_key, bowling_type_bucket)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bvbt_bucket ON batter_vs_bowling_type_summary(bowling_type_bucket, sample_size_confidence DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batter_vs_phase_summary (
            batter_key TEXT NOT NULL,
            bowling_phase TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            dismissals INTEGER NOT NULL DEFAULT 0,
            strike_rate REAL NOT NULL DEFAULT 0.0,
            dot_ball_pct REAL NOT NULL DEFAULT 0.0,
            sample_size_confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'direct_phase_history',
            last_updated REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (batter_key, bowling_phase)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bvps_phase ON batter_vs_phase_summary(bowling_phase, sample_size_confidence DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bowler_vs_batting_hand_summary (
            bowler_key TEXT NOT NULL,
            batting_hand TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            dismissals INTEGER NOT NULL DEFAULT 0,
            economy REAL NOT NULL DEFAULT 0.0,
            strike_rate_against REAL NOT NULL DEFAULT 0.0,
            dot_ball_pct REAL NOT NULL DEFAULT 0.0,
            sample_size_confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'direct_delivery_history',
            last_updated REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (bowler_key, batting_hand)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bvhs_hand ON bowler_vs_batting_hand_summary(batting_hand, sample_size_confidence DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batter_vs_spin_pace_summary (
            batter_key TEXT NOT NULL,
            pace_spin_bucket TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            dismissals REAL NOT NULL DEFAULT 0.0,
            strike_rate REAL NOT NULL DEFAULT 0.0,
            dot_ball_pct REAL NOT NULL DEFAULT 0.0,
            sample_size_confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'derived_history',
            last_updated REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (batter_key, pace_spin_bucket)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS matchup_summary_rebuild_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            player_metadata_updated_at REAL NOT NULL DEFAULT 0.0,
            matchup_summaries_updated_at REAL NOT NULL DEFAULT 0.0,
            direct_delivery_rows_seen INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        )
        """
    )


def ensure_data_dir() -> None:
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    global _SCHEMA_BOOTSTRAP_SIG
    ensure_data_dir()
    sig = db_runtime_signature()
    with _SCHEMA_BOOTSTRAP_LOCK:
        needs_bootstrap = _SCHEMA_BOOTSTRAP_SIG != sig
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    if needs_bootstrap:
        conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            team_a TEXT,
            team_b TEXT,
            venue TEXT,
            match_date TEXT,
            winner TEXT,
            toss_winner TEXT,
            toss_decision TEXT,
            batting_first TEXT,
            margin TEXT,
            raw_payload TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS match_xi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            bat_order INTEGER,
            is_playing_xi INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (match_id) REFERENCES match_results(id)
        );
        CREATE TABLE IF NOT EXISTS match_batting (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position INTEGER,
            runs INTEGER,
            balls INTEGER,
            FOREIGN KEY (match_id) REFERENCES match_results(id)
        );
        CREATE TABLE IF NOT EXISTS match_bowling (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            overs REAL,
            maidens INTEGER,
            runs INTEGER,
            wickets INTEGER,
            FOREIGN KEY (match_id) REFERENCES match_results(id)
        );
        CREATE TABLE IF NOT EXISTS learned_player (
            player_key TEXT PRIMARY KEY,
            matches_in_db INTEGER NOT NULL DEFAULT 0,
            xi_appearances INTEGER NOT NULL DEFAULT 0,
            batting_runs INTEGER NOT NULL DEFAULT 0,
            batting_balls INTEGER NOT NULL DEFAULT 0,
            wickets INTEGER NOT NULL DEFAULT 0,
            balls_bowled INTEGER NOT NULL DEFAULT 0,
            impact_ema REAL NOT NULL DEFAULT 0.5,
            last_updated REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS learned_venue_team (
            venue_key TEXT NOT NULL,
            team_key TEXT NOT NULL,
            matches INTEGER NOT NULL DEFAULT 0,
            wins_bat_first INTEGER NOT NULL DEFAULT 0,
            wins_bowl_first INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (venue_key, team_key)
        );
        CREATE INDEX IF NOT EXISTS idx_match_url ON match_results(url);

        CREATE TABLE IF NOT EXISTS match_context (
            match_id INTEGER PRIMARY KEY,
            start_hour_local INTEGER,
            is_night INTEGER,
            dew_proxy REAL,
            overseas_team_a INTEGER,
            overseas_team_b INTEGER
        );

        CREATE TABLE IF NOT EXISTS learned_overseas_mix (
            venue_key TEXT NOT NULL,
            team_key TEXT NOT NULL,
            n_overseas INTEGER NOT NULL,
            tally INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (venue_key, team_key, n_overseas)
        );

        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            competition TEXT NOT NULL DEFAULT 'IPL',
            match_date TEXT,
            venue TEXT,
            team_a TEXT,
            team_b TEXT,
            result TEXT,
            scorecard_url TEXT,
            source TEXT,
            batting_first TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (id) REFERENCES match_results(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS team_match_xi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            selected_in_xi INTEGER NOT NULL DEFAULT 1,
            batting_position REAL,
            is_keeper INTEGER NOT NULL DEFAULT 0,
            overs_bowled REAL,
            batting_order_index INTEGER,
            bowling_type TEXT,
            role_bucket TEXT,
            is_impact_used INTEGER NOT NULL DEFAULT 0,
            overseas INTEGER NOT NULL DEFAULT 0,
            UNIQUE(match_id, team_name, player_name),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_tmx_team_key ON team_match_xi(team_key);
        CREATE INDEX IF NOT EXISTS idx_tmx_player_key ON team_match_xi(player_key);
        CREATE INDEX IF NOT EXISTS idx_tmx_match_id ON team_match_xi(match_id);
        CREATE INDEX IF NOT EXISTS idx_tmx_team_player ON team_match_xi(team_key, player_key);

        CREATE TABLE IF NOT EXISTS player_match_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            season TEXT,
            selected_in_xi INTEGER NOT NULL DEFAULT 0,
            batting_position INTEGER,
            runs INTEGER,
            balls INTEGER,
            fours INTEGER,
            sixes INTEGER,
            strike_rate REAL,
            overs_bowled REAL,
            wickets INTEGER,
            runs_conceded INTEGER,
            economy REAL,
            dismissal_type TEXT,
            vs_spin_balls_faced INTEGER NOT NULL DEFAULT 0,
            vs_pace_balls_faced INTEGER NOT NULL DEFAULT 0,
            UNIQUE(match_id, team_name, player_name),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_pms_match ON player_match_stats(match_id);
        CREATE INDEX IF NOT EXISTS idx_pms_player_key ON player_match_stats(player_key);
        CREATE INDEX IF NOT EXISTS idx_pms_team_player ON player_match_stats(team_key, player_key);

        CREATE TABLE IF NOT EXISTS player_phase_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL,
            phase TEXT NOT NULL,
            balls INTEGER NOT NULL DEFAULT 0,
            runs INTEGER NOT NULL DEFAULT 0,
            wickets INTEGER NOT NULL DEFAULT 0,
            vs_spin_balls INTEGER NOT NULL DEFAULT 0,
            vs_pace_balls INTEGER NOT NULL DEFAULT 0,
            UNIQUE(match_id, team_name, player_name, role, phase),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_ppu_match ON player_phase_usage(match_id);
        CREATE INDEX IF NOT EXISTS idx_ppu_player ON player_phase_usage(player_key);

        CREATE TABLE IF NOT EXISTS match_ball_by_ball (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            innings_number INTEGER NOT NULL DEFAULT 1,
            batting_team_key TEXT NOT NULL,
            bowling_team_key TEXT NOT NULL,
            over_number INTEGER NOT NULL DEFAULT 0,
            ball_in_over INTEGER NOT NULL DEFAULT 0,
            phase TEXT NOT NULL DEFAULT 'middle',
            batter_name TEXT NOT NULL,
            batter_key TEXT NOT NULL,
            bowler_name TEXT NOT NULL,
            bowler_key TEXT NOT NULL,
            runs_batter INTEGER NOT NULL DEFAULT 0,
            runs_total INTEGER NOT NULL DEFAULT 0,
            is_legal_ball INTEGER NOT NULL DEFAULT 1,
            is_dot_ball INTEGER NOT NULL DEFAULT 0,
            is_boundary INTEGER NOT NULL DEFAULT 0,
            is_dismissal INTEGER NOT NULL DEFAULT 0,
            dismissal_kind TEXT,
            batter_out_key TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_mbbb_match ON match_ball_by_ball(match_id, innings_number);
        CREATE INDEX IF NOT EXISTS idx_mbbb_batter ON match_ball_by_ball(batter_key, match_id);
        CREATE INDEX IF NOT EXISTS idx_mbbb_bowler ON match_ball_by_ball(bowler_key, match_id);
        CREATE INDEX IF NOT EXISTS idx_mbbb_phase ON match_ball_by_ball(phase, bowler_key);

        CREATE TABLE IF NOT EXISTS player_batting_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_key TEXT NOT NULL,
            canonical_team_key TEXT NOT NULL DEFAULT '',
            canonical_player_key TEXT NOT NULL DEFAULT '',
            batting_position REAL,
            season TEXT,
            innings_number INTEGER NOT NULL DEFAULT 1,
            UNIQUE(match_id, team_name, player_key, innings_number),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS player_franchise_features (
            player_key TEXT NOT NULL,
            franchise_team_key TEXT NOT NULL,
            batting_position_ema REAL,
            batting_slot_samples INTEGER NOT NULL DEFAULT 0,
            batting_positions_tail_json TEXT,
            pp_overs_bowled REAL NOT NULL DEFAULT 0,
            middle_overs_bowled REAL NOT NULL DEFAULT 0,
            death_overs_bowled REAL NOT NULL DEFAULT 0,
            pp_bowl_ball_share REAL NOT NULL DEFAULT 0,
            middle_bowl_ball_share REAL NOT NULL DEFAULT 0,
            death_bowl_ball_share REAL NOT NULL DEFAULT 0,
            phase_bowl_rate_pp REAL NOT NULL DEFAULT 0,
            phase_bowl_rate_middle REAL NOT NULL DEFAULT 0,
            phase_bowl_rate_death REAL NOT NULL DEFAULT 0,
            vs_spin_balls INTEGER NOT NULL DEFAULT 0,
            vs_pace_balls INTEGER NOT NULL DEFAULT 0,
            vs_spin_tendency REAL NOT NULL DEFAULT 0,
            vs_pace_tendency REAL NOT NULL DEFAULT 0,
            batting_aggressor_score REAL NOT NULL DEFAULT 0,
            bowling_control_score REAL NOT NULL DEFAULT 0,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (player_key, franchise_team_key)
        );
        CREATE INDEX IF NOT EXISTS idx_pff_franchise ON player_franchise_features(franchise_team_key);

        CREATE TABLE IF NOT EXISTS player_profiles (
            player_key TEXT NOT NULL,
            franchise_team_key TEXT NOT NULL,
            display_name_hint TEXT,
            xi_selection_frequency REAL,
            batting_position_ema REAL,
            opener_likelihood REAL,
            middle_order_likelihood REAL,
            finisher_likelihood REAL,
            powerplay_bowler_likelihood REAL,
            middle_overs_bowler_likelihood REAL,
            death_bowler_likelihood REAL,
            batting_aggressor_score REAL,
            bowling_control_score REAL,
            batting_vs_spin_tendency REAL,
            batting_vs_pace_tendency REAL,
            venue_fit_score REAL,
            role_stability_score REAL,
            recent_usage_score REAL,
            h2h_basis_json TEXT,
            profile_confidence REAL,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (player_key, franchise_team_key)
        );

        CREATE TABLE IF NOT EXISTS team_selection_patterns (
            team_key TEXT NOT NULL,
            venue_key TEXT NOT NULL,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            xi_frequency_json TEXT,
            overseas_combo_json TEXT,
            preferred_xi_core_json TEXT,
            keeper_consistency REAL,
            opener_stability REAL,
            finisher_stability REAL,
            bowling_composition_json TEXT,
            chase_defend_json TEXT,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (team_key, venue_key)
        );

        CREATE TABLE IF NOT EXISTS venue_team_patterns (
            venue_key TEXT NOT NULL,
            team_key TEXT NOT NULL,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            xi_frequency_json TEXT,
            overseas_combo_json TEXT,
            preferred_xi_core_json TEXT,
            keeper_consistency REAL,
            opener_stability REAL,
            finisher_stability REAL,
            bowling_composition_json TEXT,
            chase_defend_json TEXT,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (venue_key, team_key)
        );

        CREATE TABLE IF NOT EXISTS team_derived_summary (
            team_key TEXT PRIMARY KEY,
            preferred_xi_core_json TEXT,
            preferred_overseas_combinations_json TEXT,
            keeper_consistency REAL,
            opener_stability REAL,
            finisher_stability REAL,
            bowling_composition_patterns_json TEXT,
            chase_vs_defend_json TEXT,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            last_updated REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS head_to_head_patterns (
            team_a_key TEXT NOT NULL,
            team_b_key TEXT NOT NULL,
            sample_matches INTEGER NOT NULL DEFAULT 0,
            weight_sum REAL NOT NULL DEFAULT 0,
            team_a_wins_weighted REAL NOT NULL DEFAULT 0,
            team_b_wins_weighted REAL NOT NULL DEFAULT 0,
            head_to_head_basis_json TEXT,
            last_updated REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (team_a_key, team_b_key)
        );

        CREATE TABLE IF NOT EXISTS team_match_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_key TEXT NOT NULL,
            playing_xi_json TEXT,
            batting_order_json TEXT,
            bowlers_used_json TEXT,
            overseas_combo_json TEXT,
            impact_player_json TEXT,
            UNIQUE(match_id, team_name),
            FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS player_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            franchise_team_key TEXT NOT NULL,
            squad_full_name TEXT NOT NULL,
            normalized_full_name_key TEXT NOT NULL,
            resolved_history_key TEXT,
            resolution_type TEXT NOT NULL,
            confidence REAL,
            ambiguous_candidates_json TEXT,
            updated_at REAL NOT NULL,
            UNIQUE(franchise_team_key, normalized_full_name_key)
        );
        CREATE INDEX IF NOT EXISTS idx_player_aliases_franchise ON player_aliases(franchise_team_key);
            """
        )
        _migrate_match_results_schema(conn)
        _migrate_player_batting_positions_innings(conn)
        _backfill_canonical_history_keys_if_needed(conn)
        # Must run after migration: old DBs had player_batting_positions without innings_number;
        # CREATE INDEX on that column in executescript would fail before migrate could rebuild the table.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pbp_match_innings ON player_batting_positions(match_id, innings_number)"
        )
        _migrate_matches_ingest_metadata_columns(conn)
        _migrate_player_recent_form_summaries_and_cache(conn)
        _migrate_drop_prediction_runtime_summary_tables(conn)
        _migrate_prediction_aggregate_summary_tables(conn)
        _migrate_player_profiles_derive_schema(conn)
        _migrate_team_selection_derive_columns(conn)
        _migrate_stage1_canonical_alias_columns(conn)
        _migrate_player_aliases_table(conn)
        _migrate_player_metadata_and_matchup_tables(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tmx_team_player ON team_match_xi(team_key, player_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pms_team_player ON player_match_stats(team_key, player_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pbp_team_player_match "
            "ON player_batting_positions(team_key, player_key, match_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pff_franchise_player "
            "ON player_franchise_features(franchise_team_key, player_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ppu_team_player_match "
            "ON player_phase_usage(team_key, player_key, match_id)"
        )
        # Runtime-query indexes for prediction hot paths.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_matches_match_date_id ON matches(match_date DESC, id DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_match_results_team_ab ON match_results(team_a, team_b)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_match_xi_match_team_player "
            "ON match_xi(match_id, team, player_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_match_xi_player_name ON match_xi(player_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_team_match_summary_team_key ON team_match_summary(team_key)"
        )
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    if needs_bootstrap:
        with _SCHEMA_BOOTSTRAP_LOCK:
            _SCHEMA_BOOTSTRAP_SIG = db_runtime_signature()
    return conn


@contextmanager
def connection() -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection()
    if audit_profile.audit_enabled() and audit_profile.sql_capture_active():
        conn = audit_profile.wrap_sqlite_connection(conn)  # type: ignore[assignment]
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Ensure SQLite file and tables exist (also runs on every `get_connection()`)."""
    conn = get_connection()
    conn.close()


def rebuild_prediction_summary_tables() -> dict[str, Any]:
    """
    Recompute prediction aggregate summary tables from raw history tables.

    Intended to run after large ingest batches so prediction avoids expensive runtime GROUP BY/JOIN scans.
    """
    t0 = time.perf_counter()
    with connection() as conn:
        conn.execute("DELETE FROM prediction_summary_player_xi_counts")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_xi_counts (player_name, xi_count)
            SELECT player_name, COUNT(*) AS c
            FROM match_xi
            GROUP BY player_name
            """
        )

        conn.execute("DELETE FROM prediction_summary_player_batting")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_batting (player_name, avg_position, sample_count)
            SELECT player_name, AVG(position) AS av, COUNT(*) AS n
            FROM match_batting
            WHERE position IS NOT NULL AND position > 0
            GROUP BY player_name
            """
        )

        conn.execute("DELETE FROM prediction_summary_player_bowling")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_bowling (player_name, avg_balls, match_count)
            SELECT player_name,
                   (
                       SUM(CASE WHEN overs IS NOT NULL THEN overs * 6.0 ELSE 0 END)
                       / CAST(COUNT(DISTINCT match_id) AS REAL)
                   ) AS avg_balls,
                   COUNT(DISTINCT match_id) AS n
            FROM match_bowling
            GROUP BY player_name
            """
        )

        conn.execute("DELETE FROM prediction_summary_venue_team_player_xi")
        conn.execute(
            """
            INSERT INTO prediction_summary_venue_team_player_xi (venue, team, player_name, xi_count)
            SELECT r.venue, m.team, m.player_name, COUNT(*) AS c
            FROM match_xi m
            JOIN match_results r ON r.id = m.match_id
            WHERE r.venue IS NOT NULL AND trim(r.venue) != ''
            GROUP BY r.venue, m.team, m.player_name
            """
        )

        conn.execute("DELETE FROM prediction_summary_match_team_venue")
        conn.execute(
            """
            INSERT INTO prediction_summary_match_team_venue (match_id, venue, team)
            SELECT DISTINCT m.match_id, COALESCE(r.venue, ''), m.team
            FROM match_xi m
            JOIN match_results r ON r.id = m.match_id
            """
        )

        conn.execute("DELETE FROM prediction_summary_match_meta")
        conn.execute(
            """
            INSERT INTO prediction_summary_match_meta (
                match_id, winner, team_a, team_b, venue, batting_first, created_at, match_date
            )
            SELECT mr.id, mr.winner, mr.team_a, mr.team_b,
                   COALESCE(NULLIF(trim(m.venue), ''), NULLIF(trim(mr.venue), ''), '') AS venue,
                   mr.batting_first, mr.created_at, m.match_date
            FROM match_results mr
            LEFT JOIN matches m ON m.id = mr.id
            WHERE mr.team_a IS NOT NULL AND trim(mr.team_a) != ''
              AND mr.team_b IS NOT NULL AND trim(mr.team_b) != ''
            """
        )

        conn.execute("DELETE FROM prediction_summary_player_global_xi")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_global_xi (
                player_key, tmx_rows, distinct_matches, distinct_teams
            )
            SELECT player_key,
                   COUNT(*) AS tmx_rows,
                   COUNT(DISTINCT match_id) AS distinct_matches,
                   COUNT(DISTINCT team_key) AS distinct_teams
            FROM team_match_xi
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
            GROUP BY player_key
            """
        )

        conn.execute("DELETE FROM prediction_summary_player_team_xi")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_team_xi (player_key, team_key, tmx_rows)
            SELECT player_key, team_key, COUNT(*) AS tmx_rows
            FROM team_match_xi
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
              AND team_key IS NOT NULL AND trim(team_key) != ''
            GROUP BY player_key, team_key
            """
        )

        conn.execute("DELETE FROM prediction_summary_player_global_slot")
        conn.execute(
            """
            INSERT INTO prediction_summary_player_global_slot (player_key, slot_ema, slot_samples)
            SELECT player_key,
                   AVG(batting_position) AS slot_ema,
                   COUNT(*) AS slot_samples
            FROM player_batting_positions
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
              AND batting_position IS NOT NULL AND batting_position > 0
            GROUP BY player_key
            """
        )

        match_xi_rows = int(conn.execute("SELECT COUNT(*) AS c FROM match_xi").fetchone()["c"] or 0)
        match_batting_rows = int(conn.execute("SELECT COUNT(*) AS c FROM match_batting").fetchone()["c"] or 0)
        match_results_rows = int(conn.execute("SELECT COUNT(*) AS c FROM match_results").fetchone()["c"] or 0)
        conn.execute(
            """
            INSERT INTO prediction_summary_rebuild_state (
                id, last_rebuilt_at, match_xi_rows, match_batting_rows, match_results_rows
            ) VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_rebuilt_at = excluded.last_rebuilt_at,
                match_xi_rows = excluded.match_xi_rows,
                match_batting_rows = excluded.match_batting_rows,
                match_results_rows = excluded.match_results_rows
            """,
            (time.time(), match_xi_rows, match_batting_rows, match_results_rows),
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": True,
        "elapsed_ms": round(elapsed_ms, 2),
        "match_xi_rows": match_xi_rows,
        "match_batting_rows": match_batting_rows,
        "match_results_rows": match_results_rows,
    }


def _confidence_from_samples(n: float, *, full: float = 80.0) -> float:
    try:
        x = float(n)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(1.0, x / max(1.0, float(full))))


def _normalize_small_key(v: Any) -> str:
    return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")


def _load_curated_player_metadata_file(raw_path: str) -> dict[str, dict[str, Any]]:
    raw = str(raw_path or "").strip()
    if not raw:
        return {}
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / raw
    try:
        if not p.is_file():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("player metadata curated file unreadable: %s", p)
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in obj.items():
        nk = str(k or "").strip()[:80]
        if nk and isinstance(v, dict):
            out[nk] = dict(v)
    return out


def _load_curated_player_metadata_with_priority() -> dict[str, dict[str, Any]]:
    """
    Merge metadata sources with explicit priority:
    1) Manual curated
    2) Cricinfo curated
    """
    manual = _load_curated_player_metadata_file(
        str(getattr(config, "PLAYER_METADATA_CURATED_PATH", "data/player_metadata_curated.json") or "")
    )
    cricinfo = _load_curated_player_metadata_file(
        str(getattr(config, "PLAYER_METADATA_CRICINFO_PATH", "data/player_metadata_cricinfo.json") or "")
    )
    merged = dict(cricinfo)
    merged.update(manual)
    return merged


def _bowling_bucket_from_style(raw_style: str) -> str:
    s = _normalize_small_key(raw_style)
    if not s:
        return "unknown"
    if "mystery" in s:
        return "mystery_spin"
    if "orthodox" in s:
        return "left_arm_orthodox"
    if "wrist" in s or "legbreak" in s or "leg_spin" in s:
        return "wrist_spin"
    if "offbreak" in s or "off_spin" in s or "finger" in s:
        return "finger_spin"
    if any(x in s for x in ("fast", "medium", "seam", "pace", "swing")):
        return "pace"
    if "spin" in s:
        return "finger_spin"
    return "unknown"


def rebuild_player_metadata() -> dict[str, Any]:
    """
    Build/update ``player_metadata`` from curated file + SQLite usage/history.
    """
    t0 = time.perf_counter()
    curated = _load_curated_player_metadata_with_priority()
    now = time.time()
    with connection() as conn:
        # latest display names + role/bowling hints
        keys = conn.execute(
            """
            SELECT DISTINCT player_key
            FROM team_match_xi
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
            """
        ).fetchall()
        upserts = 0
        for r in keys:
            pk = str(r["player_key"] or "").strip()[:80]
            if not pk:
                continue
            latest_name_row = conn.execute(
                """
                SELECT player_name FROM team_match_xi
                WHERE player_key = ?
                ORDER BY id DESC LIMIT 1
                """,
                (pk,),
            ).fetchone()
            display_name = str((latest_name_row["player_name"] if latest_name_row else pk) or pk).strip()
            role_rows = conn.execute(
                """
                SELECT role_bucket, COUNT(*) AS c
                FROM team_match_xi
                WHERE player_key = ? AND role_bucket IS NOT NULL AND trim(role_bucket) != ''
                GROUP BY role_bucket
                ORDER BY c DESC
                LIMIT 2
                """,
                (pk,),
            ).fetchall()
            primary_role = "batter"
            secondary_role = ""
            role_map = {
                "Batter": "batter",
                "WK-Batter": "wicketkeeper_batter",
                "All-Rounder": "batting_allrounder",
                "Bowler": "bowler",
            }
            if role_rows:
                primary_role = role_map.get(str(role_rows[0]["role_bucket"] or "").strip(), "batter")
                if len(role_rows) > 1:
                    secondary_role = role_map.get(str(role_rows[1]["role_bucket"] or "").strip(), "")

            style_row = conn.execute(
                """
                SELECT bowling_type, COUNT(*) AS c
                FROM team_match_xi
                WHERE player_key = ? AND bowling_type IS NOT NULL AND trim(bowling_type) != ''
                GROUP BY bowling_type
                ORDER BY c DESC
                LIMIT 1
                """,
                (pk,),
            ).fetchone()
            bowling_style_raw = str((style_row["bowling_type"] if style_row else "") or "").strip()
            bowling_type_bucket = _bowling_bucket_from_style(bowling_style_raw)

            bat_row = conn.execute(
                """
                SELECT AVG(batting_position) AS av, COUNT(*) AS n
                FROM player_batting_positions
                WHERE player_key = ? AND batting_position IS NOT NULL AND batting_position > 0
                """,
                (pk,),
            ).fetchone()
            bat_avg = float((bat_row["av"] if bat_row else 0.0) or 0.0)
            bat_n = int((bat_row["n"] if bat_row else 0) or 0)
            if bat_n < 2:
                likely_batting_band = "unknown"
            elif bat_avg <= 2.2:
                likely_batting_band = "opener"
            elif bat_avg <= 4.0:
                likely_batting_band = "top_order"
            elif bat_avg <= 6.0:
                likely_batting_band = "middle_order"
            elif bat_avg <= 8.0:
                likely_batting_band = "finisher"
            else:
                likely_batting_band = "tail"

            ph_rows = conn.execute(
                """
                SELECT phase, SUM(balls) AS b
                FROM player_phase_usage
                WHERE player_key = ? AND role = 'bowl'
                GROUP BY phase
                """,
                (pk,),
            ).fetchall()
            by_ph = {str(x["phase"] or ""): int(x["b"] or 0) for x in ph_rows}
            nz = [k for k, v in by_ph.items() if v > 0]
            if not nz:
                likely_bowling_phases = "unknown"
            elif len(nz) > 1:
                likely_bowling_phases = "multiple"
            else:
                one = nz[0]
                likely_bowling_phases = one if one in ("powerplay", "middle", "death") else "unknown"

            batting_hand = "unknown"
            source = "derived_history"
            conf = _confidence_from_samples(bat_n, full=20.0)
            if pk in curated:
                c = curated[pk]
                display_name = str(c.get("display_name") or c.get("player_name") or display_name).strip() or display_name
                batting_hand = str(c.get("batting_hand") or "unknown").strip().lower()
                bowling_style_raw = str(c.get("bowling_style_raw") or bowling_style_raw).strip()
                bowling_type_bucket = str(c.get("bowling_type_bucket") or bowling_type_bucket).strip().lower()
                primary_role = str(c.get("primary_role") or primary_role).strip().lower()
                secondary_role = str(c.get("secondary_role") or secondary_role).strip().lower()
                likely_batting_band = str(c.get("likely_batting_band") or likely_batting_band).strip().lower()
                likely_bowling_phases = str(c.get("likely_bowling_phases") or likely_bowling_phases).strip().lower()
                source = str(c.get("source") or "curated_manual").strip() or "curated_manual"
                conf = max(conf, float(c.get("confidence") or 0.95))

            conn.execute(
                """
                INSERT INTO player_metadata (
                    player_key, display_name, batting_hand, bowling_style_raw, bowling_type_bucket,
                    primary_role, secondary_role, likely_batting_band, likely_bowling_phases,
                    source, confidence, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_key) DO UPDATE SET
                    display_name = excluded.display_name,
                    batting_hand = excluded.batting_hand,
                    bowling_style_raw = excluded.bowling_style_raw,
                    bowling_type_bucket = excluded.bowling_type_bucket,
                    primary_role = excluded.primary_role,
                    secondary_role = excluded.secondary_role,
                    likely_batting_band = excluded.likely_batting_band,
                    likely_bowling_phases = excluded.likely_bowling_phases,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    last_updated = excluded.last_updated
                """,
                (
                    pk,
                    display_name,
                    batting_hand if batting_hand in ("right", "left", "unknown") else "unknown",
                    bowling_style_raw,
                    bowling_type_bucket
                    if bowling_type_bucket
                    in ("pace", "finger_spin", "wrist_spin", "left_arm_orthodox", "mystery_spin", "unknown")
                    else "unknown",
                    primary_role
                    if primary_role
                    in (
                        "batter",
                        "wk_batter",
                        "all_rounder",
                        "bowler",
                        "wicketkeeper_batter",
                        "batting_allrounder",
                        "bowling_allrounder",
                    )
                    else "batter",
                    secondary_role,
                    likely_batting_band
                    if likely_batting_band in ("opener", "top_order", "middle_order", "finisher", "tail", "unknown")
                    else "unknown",
                    likely_bowling_phases
                    if likely_bowling_phases in ("powerplay", "middle", "death", "multiple", "unknown")
                    else "unknown",
                    source,
                    max(0.0, min(1.0, conf)),
                    now,
                ),
            )
            upserts += 1
        conn.execute(
            """
            INSERT INTO matchup_summary_rebuild_state (id, player_metadata_updated_at)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET player_metadata_updated_at = excluded.player_metadata_updated_at
            """,
            (now,),
        )
    return {"ok": True, "rows_upserted": upserts, "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2)}


def _delivery_facts_from_raw_payload(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], int]:
    """
    Parse direct batter-bowler delivery facts from ``match_results.raw_payload`` when present.
    Returns (delivery_rows, payload_rows_seen_with_overs).
    """
    rows = conn.execute(
        "SELECT id, match_date, raw_payload FROM match_results WHERE raw_payload IS NOT NULL AND trim(raw_payload) != ''"
    ).fetchall()
    out: list[dict[str, Any]] = []
    seen_payloads = 0
    for r in rows:
        raw = str(r["raw_payload"] or "")
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        innings = obj.get("innings")
        if not isinstance(innings, list) or not innings:
            continue
        seen_payloads += 1
        mdate = str(r["match_date"] or "")
        for inn in innings:
            if not isinstance(inn, dict):
                continue
            overs = inn.get("overs")
            if not isinstance(overs, list):
                continue
            for ov in overs:
                if not isinstance(ov, dict):
                    continue
                deliveries = ov.get("deliveries")
                if not isinstance(deliveries, list):
                    continue
                for d in deliveries:
                    if not isinstance(d, dict):
                        continue
                    batter = str(((d.get("batter") if isinstance(d.get("batter"), str) else None) or "")).strip()
                    bowler = str(((d.get("bowler") if isinstance(d.get("bowler"), str) else None) or "")).strip()
                    if not batter or not bowler:
                        continue
                    batter_key = batter.lower().strip()[:80]
                    bowler_key = bowler.lower().strip()[:80]
                    runs_block = d.get("runs") if isinstance(d.get("runs"), dict) else {}
                    bruns = int((runs_block.get("batter") if isinstance(runs_block, dict) else 0) or 0)
                    total = int((runs_block.get("total") if isinstance(runs_block, dict) else bruns) or bruns)
                    is_wicket = 0
                    wk = d.get("wickets")
                    if isinstance(wk, list) and wk:
                        for w in wk:
                            if isinstance(w, dict) and str(w.get("player_out") or "").strip().lower() == batter.lower():
                                is_wicket = 1
                                break
                    out.append(
                        {
                            "batter_key": batter_key,
                            "bowler_key": bowler_key,
                            "runs_batter": bruns,
                            "runs_total": total,
                            "is_dot": 1 if total == 0 else 0,
                            "is_boundary": 1 if bruns >= 4 else 0,
                            "dismissal": is_wicket,
                            "match_date": mdate,
                            "match_id": int(r["id"] or 0),
                        }
                    )
    return out, seen_payloads


def rebuild_matchup_summaries() -> dict[str, Any]:
    """
    Recompute matchup summary tables from SQLite-ingested history.
    """
    t0 = time.perf_counter()
    now = time.time()
    with connection() as conn:
        conn.execute("DELETE FROM batter_bowler_matchup_summary")
        conn.execute("DELETE FROM batter_vs_bowling_type_summary")
        conn.execute("DELETE FROM batter_vs_phase_summary")
        conn.execute("DELETE FROM bowler_vs_batting_hand_summary")
        conn.execute("DELETE FROM batter_vs_spin_pace_summary")
        delivery_rows = conn.execute(
            """
            SELECT bb.match_id, m.match_date, bb.phase,
                   bb.batter_key, bb.bowler_key,
                   bb.runs_batter, bb.runs_total,
                   bb.is_dot_ball, bb.is_boundary, bb.is_dismissal
            FROM match_ball_by_ball bb
            LEFT JOIN matches m ON m.id = bb.match_id
            WHERE bb.batter_key IS NOT NULL AND trim(bb.batter_key) != ''
              AND bb.bowler_key IS NOT NULL AND trim(bb.bowler_key) != ''
              AND bb.is_legal_ball = 1
            """
        ).fetchall()

        # Metadata maps for hand/type joins.
        hand_map = {
            str(x["player_key"] or "").strip()[:80]: str(x["batting_hand"] or "unknown").strip().lower()
            for x in conn.execute("SELECT player_key, batting_hand FROM player_metadata").fetchall()
        }
        bowl_type_map = {
            str(x["player_key"] or "").strip()[:80]: str(x["bowling_type_bucket"] or "unknown").strip().lower()
            for x in conn.execute("SELECT player_key, bowling_type_bucket FROM player_metadata").fetchall()
        }

        bb: dict[tuple[str, str], dict[str, Any]] = {}
        bphase: dict[tuple[str, str], dict[str, Any]] = {}
        bvtype: dict[tuple[str, str], dict[str, Any]] = {}
        bvh: dict[tuple[str, str], dict[str, Any]] = {}
        bspinpace: dict[tuple[str, str], dict[str, Any]] = {}
        for r in delivery_rows:
            batter_key = str(r["batter_key"] or "").strip()[:80]
            bowler_key = str(r["bowler_key"] or "").strip()[:80]
            phase = str(r["phase"] or "middle").strip().lower()
            if phase not in ("powerplay", "middle", "death"):
                phase = "middle"
            runs_batter = int(r["runs_batter"] or 0)
            runs_total = int(r["runs_total"] or 0)
            is_dot = int(r["is_dot_ball"] or 0)
            is_boundary = int(r["is_boundary"] or 0)
            is_dismissal = int(r["is_dismissal"] or 0)
            mdate = str(r["match_date"] or "")
            mid = int(r["match_id"] or 0)

            # 1) Direct batter vs bowler.
            k_bb = (batter_key, bowler_key)
            cur_bb = bb.setdefault(
                k_bb,
                {
                    "balls": 0,
                    "runs": 0,
                    "dismissals": 0,
                    "dots": 0,
                    "bounds": 0,
                    "match_ids": set(),
                    "last_match_date": "",
                },
            )
            cur_bb["balls"] += 1
            cur_bb["runs"] += runs_batter
            cur_bb["dismissals"] += is_dismissal
            cur_bb["dots"] += is_dot
            cur_bb["bounds"] += is_boundary
            cur_bb["match_ids"].add(mid)
            if mdate and mdate > str(cur_bb["last_match_date"]):
                cur_bb["last_match_date"] = mdate

            # 2) Batter vs phase.
            k_phase = (batter_key, phase)
            cur_phase = bphase.setdefault(k_phase, {"balls": 0, "runs": 0, "dismissals": 0, "dots": 0, "bounds": 0})
            cur_phase["balls"] += 1
            cur_phase["runs"] += runs_batter
            cur_phase["dismissals"] += is_dismissal
            cur_phase["dots"] += is_dot
            cur_phase["bounds"] += is_boundary

            # 3) Batter vs bowling type and spin/pace fallback buckets.
            btype = bowl_type_map.get(bowler_key, "unknown")
            if btype not in ("pace", "finger_spin", "wrist_spin", "left_arm_orthodox", "mystery_spin", "unknown"):
                btype = "unknown"
            k_type = (batter_key, btype)
            cur_type = bvtype.setdefault(k_type, {"balls": 0, "runs": 0, "dismissals": 0, "dots": 0, "bounds": 0})
            cur_type["balls"] += 1
            cur_type["runs"] += runs_batter
            cur_type["dismissals"] += is_dismissal
            cur_type["dots"] += is_dot
            cur_type["bounds"] += is_boundary

            ps_bucket = "pace" if btype == "pace" else ("spin" if btype in ("finger_spin", "wrist_spin", "left_arm_orthodox", "mystery_spin") else "unknown")
            k_ps = (batter_key, ps_bucket)
            cur_ps = bspinpace.setdefault(k_ps, {"balls": 0, "runs": 0, "dismissals": 0, "dots": 0})
            cur_ps["balls"] += 1
            cur_ps["runs"] += runs_batter
            cur_ps["dismissals"] += is_dismissal
            cur_ps["dots"] += is_dot

            # 4) Bowler vs batting hand.
            hand = hand_map.get(batter_key, "unknown")
            if hand not in ("right", "left", "unknown"):
                hand = "unknown"
            k_hand = (bowler_key, hand)
            cur_hand = bvh.setdefault(k_hand, {"balls": 0, "runs": 0, "dismissals": 0, "dots": 0})
            cur_hand["balls"] += 1
            cur_hand["runs"] += runs_total
            cur_hand["dismissals"] += is_dismissal
            cur_hand["dots"] += is_dot

        for (batter_key, bowler_key), cur in bb.items():
            balls = int(cur["balls"])
            runs = int(cur["runs"])
            dismissals = int(cur["dismissals"])
            match_count = len(cur["match_ids"])
            conn.execute(
                """
                INSERT INTO batter_bowler_matchup_summary (
                    batter_key, bowler_key, balls, runs, dismissals, strike_rate,
                    dot_ball_pct, boundary_pct, innings_count, match_count, last_match_date,
                    sample_size_confidence, source, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batter_key,
                    bowler_key,
                    balls,
                    runs,
                    dismissals,
                    (100.0 * runs / balls) if balls > 0 else 0.0,
                    (float(cur["dots"]) / balls) if balls > 0 else 0.0,
                    (float(cur["bounds"]) / balls) if balls > 0 else 0.0,
                    match_count,
                    match_count,
                    cur["last_match_date"] or None,
                    _confidence_from_samples(balls, full=90.0),
                    "direct_ball_by_ball",
                    now,
                ),
            )

        for (batter_key, phase), cur in bphase.items():
            balls = int(cur["balls"])
            runs = int(cur["runs"])
            conn.execute(
                """
                INSERT INTO batter_vs_phase_summary (
                    batter_key, bowling_phase, balls, runs, dismissals, strike_rate,
                    dot_ball_pct, sample_size_confidence, source, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batter_key,
                    phase,
                    balls,
                    runs,
                    int(cur["dismissals"]),
                    (100.0 * runs / balls) if balls > 0 else 0.0,
                    (float(cur["dots"]) / balls) if balls > 0 else 0.0,
                    _confidence_from_samples(balls, full=120.0),
                    "direct_ball_by_ball",
                    now,
                ),
            )

        for (batter_key, btype), cur in bvtype.items():
            balls = int(cur["balls"])
            runs = int(cur["runs"])
            conn.execute(
                """
                INSERT INTO batter_vs_bowling_type_summary (
                    batter_key, bowling_type_bucket, balls, runs, dismissals,
                    strike_rate, dot_ball_pct, boundary_pct, sample_size_confidence, source, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batter_key,
                    btype,
                    balls,
                    runs,
                    int(cur["dismissals"]),
                    (100.0 * runs / balls) if balls > 0 else 0.0,
                    (float(cur["dots"]) / balls) if balls > 0 else 0.0,
                    (float(cur["bounds"]) / balls) if balls > 0 else 0.0,
                    _confidence_from_samples(balls, full=120.0),
                    "direct_ball_by_ball_with_metadata",
                    now,
                ),
            )

        for (batter_key, ps_bucket), cur in bspinpace.items():
            balls = int(cur["balls"])
            runs = int(cur["runs"])
            conn.execute(
                """
                INSERT INTO batter_vs_spin_pace_summary (
                    batter_key, pace_spin_bucket, balls, runs, dismissals,
                    strike_rate, dot_ball_pct, sample_size_confidence, source, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batter_key,
                    ps_bucket,
                    balls,
                    runs,
                    int(cur["dismissals"]),
                    (100.0 * runs / balls) if balls > 0 else 0.0,
                    (float(cur["dots"]) / balls) if balls > 0 else 0.0,
                    _confidence_from_samples(balls, full=120.0),
                    "direct_ball_by_ball_with_metadata",
                    now,
                ),
            )

        for (bowler_key, hand), cur in bvh.items():
            balls = int(cur["balls"])
            runs = int(cur["runs"])
            conn.execute(
                """
                INSERT INTO bowler_vs_batting_hand_summary (
                    bowler_key, batting_hand, balls, runs, dismissals, economy,
                    strike_rate_against, dot_ball_pct, sample_size_confidence, source, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bowler_key,
                    hand,
                    balls,
                    runs,
                    int(cur["dismissals"]),
                    (6.0 * runs / balls) if balls > 0 else 0.0,
                    (100.0 * runs / balls) if balls > 0 else 0.0,
                    (float(cur["dots"]) / balls) if balls > 0 else 0.0,
                    _confidence_from_samples(balls, full=90.0),
                    "direct_ball_by_ball_with_metadata",
                    now,
                ),
            )

        conn.execute(
            """
            INSERT INTO matchup_summary_rebuild_state (
                id, matchup_summaries_updated_at, direct_delivery_rows_seen, notes
            ) VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                matchup_summaries_updated_at = excluded.matchup_summaries_updated_at,
                direct_delivery_rows_seen = excluded.direct_delivery_rows_seen,
                notes = excluded.notes
            """,
            (
                now,
                int(len(delivery_rows)),
                "summaries populated from match_ball_by_ball (direct) plus metadata joins",
            ),
        )
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "direct_delivery_rows_seen": int(len(delivery_rows)),
        "payload_rows_with_overs": 0,
    }


def rebuild_player_metadata_and_matchup_summaries() -> dict[str, Any]:
    """One-call Phase-2 rebuild entrypoint (safe to rerun)."""
    m = rebuild_player_metadata()
    s = rebuild_matchup_summaries()
    return {"ok": bool(m.get("ok") and s.get("ok")), "metadata": m, "matchups": s}
def fetch_player_metadata_batch(player_keys: list[str]) -> dict[str, dict[str, Any]]:
    """
    Batch fetch from ``player_metadata`` keyed by ``player_key``.

    Returns a mapping keyed by normalized (lowercased) player_key.
    """
    keys = [str(k).strip().lower() for k in (player_keys or []) if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        try:
            rows = conn.execute(
                f"""
                SELECT lower(player_key) AS player_key,
                       display_name,
                       batting_hand,
                       bowling_style_raw,
                       bowling_type_bucket,
                       primary_role,
                       secondary_role,
                       likely_batting_band,
                       likely_bowling_phases,
                       source,
                       confidence,
                       last_updated
                FROM player_metadata
                WHERE lower(player_key) IN ({qm})
                """,
                keys,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    out: dict[str, dict[str, Any]] = {}
    for r in rows or []:
        pk = str(r[0] or "").strip().lower()
        if not pk:
            continue
        out[pk] = {
            "player_key": pk,
            "display_name": r[1],
            "batting_hand": r[2],
            "bowling_style_raw": r[3],
            "bowling_type_bucket": r[4],
            "primary_role": r[5],
            "secondary_role": r[6],
            "likely_batting_band": r[7],
            "likely_bowling_phases": r[8],
            "source": r[9],
            "confidence": r[10],
            "last_updated": r[11],
        }
    return out


def fetch_player_batting_position_profile_batch(
    franchise_team_key: str, player_keys: list[str]
) -> dict[str, dict[str, Any]]:
    """
    Per-player batting-position profile derived from ``player_batting_positions``.
    """
    tk = str(franchise_team_key or "").strip()
    keys = [str(k).strip().lower() for k in (player_keys or []) if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not tk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        try:
            rows = conn.execute(
                f"""
                SELECT lower(player_key) AS player_key, batting_position
                FROM player_batting_positions
                WHERE team_key = ? AND lower(player_key) IN ({qm}) AND batting_position IS NOT NULL
                """,
                [tk, *keys],
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    positions_by_pk: dict[str, list[float]] = {}
    for r in rows or []:
        pk = str(r[0] or "").strip().lower()
        if not pk:
            continue
        try:
            pos = float(r[1])
        except (TypeError, ValueError):
            continue
        if pos <= 0 or pos > 15:
            continue
        positions_by_pk.setdefault(pk, []).append(pos)
    out: dict[str, dict[str, Any]] = {}
    for pk, vals in positions_by_pk.items():
        if not vals:
            continue
        s = sorted(vals)
        dominant = float(round(s[len(s) // 2], 2))
        top12_share = sum(1 for x in vals if x <= 2.0) / max(1, len(vals))
        dist: dict[str, int] = {}
        for x in vals:
            k = str(int(round(x)))
            dist[k] = dist.get(k, 0) + 1
        out[pk] = {
            "dominant_position": dominant,
            "top12_share": float(round(top12_share, 4)),
            "distribution": dist,
        }
    return out


def fetch_bowler_phase_summary_batch(
    franchise_team_key: str, player_keys: list[str]
) -> dict[str, dict[str, Any]]:
    """
    Bowling phase usage summary from ``player_phase_usage`` for a franchise slice.
    """
    tk = str(franchise_team_key or "").strip()
    keys = [str(k).strip().lower() for k in (player_keys or []) if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not tk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        try:
            rows = conn.execute(
                f"""
                SELECT lower(player_key) AS player_key, phase,
                       SUM(balls) AS balls, SUM(wickets) AS wickets
                FROM player_phase_usage
                WHERE team_key = ? AND lower(player_key) IN ({qm}) AND role IN ('bowler','bowl','all')
                GROUP BY lower(player_key), phase
                """,
                [tk, *keys],
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    phase_map: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows or []:
        pk = str(r[0] or "").strip().lower()
        ph = str(r[1] or "").strip().lower()
        if not pk or ph not in ("powerplay", "middle", "death"):
            continue
        phase_map.setdefault(pk, {})[ph] = {"balls": float(r[2] or 0.0), "wickets": float(r[3] or 0.0)}
    out: dict[str, dict[str, Any]] = {}
    for pk, phs in phase_map.items():
        pp = phs.get("powerplay", {})
        md = phs.get("middle", {})
        dt = phs.get("death", {})
        pp_b = float(pp.get("balls") or 0.0)
        md_b = float(md.get("balls") or 0.0)
        dt_b = float(dt.get("balls") or 0.0)
        total = pp_b + md_b + dt_b
        if total <= 0:
            continue
        out[pk] = {
            "total_balls": total,
            "powerplay_share": pp_b / total,
            "middle_share": md_b / total,
            "death_share": dt_b / total,
            "powerplay_wickets_per_ball": (float(pp.get("wickets") or 0.0) / pp_b) if pp_b > 0 else 0.0,
            "death_wickets_per_ball": (float(dt.get("wickets") or 0.0) / dt_b) if dt_b > 0 else 0.0,
        }
    return out
def remove_sqlite_database_files() -> dict[str, Any]:
    """
    Delete the configured SQLite database file and any **-wal** / **-shm** sidecars.

    Use when the DB is corrupted or you want a clean rebuild (IPL + all_json re-ingest, then
    Stage 2 derive, then recent-form cache). **Irreversible.** Close other connections first
    (stop duplicate Streamlit tabs/processes using this DB).

    After calling, run :func:`init_schema` to recreate an empty schema.
    """
    base = Path(config.DB_PATH).resolve()
    removed: list[str] = []
    candidates = [
        base,
        Path(str(base) + "-wal"),
        Path(str(base) + "-shm"),
    ]
    for p in candidates:
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError as exc:
            logger.warning("remove_sqlite_database_files: could not remove %s: %s", p, exc)
    with _SCHEMA_BOOTSTRAP_LOCK:
        global _SCHEMA_BOOTSTRAP_SIG
        _SCHEMA_BOOTSTRAP_SIG = None
    return {"db_path": str(base), "removed_paths": removed}


def _scorecard_url_variants(url: str) -> list[str]:
    """Normalize URL variants so discovery + manual ingest dedupe consistently."""
    u = (url or "").strip().split("?")[0].split("#")[0].rstrip("/")
    if not u:
        return []
    out: set[str] = {u, u + "/"}
    if u.startswith("https://"):
        out.add("http://" + u[8:])
    elif u.startswith("http://"):
        out.add("https://" + u[7:])
    return list(out)


def existing_cricsheet_match_ids() -> set[str]:
    """Numeric Cricsheet match ids already ingested (URL and/or ``matches.cricsheet_match_id``)."""
    out: set[str] = set()
    with connection() as conn:
        rows = conn.execute(
            "SELECT url FROM match_results WHERE url LIKE 'cricsheet://ipl/%'"
        ).fetchall()
        for r in rows:
            u = str(r["url"] or "").strip().rstrip("/")
            if u.startswith("cricsheet://ipl/"):
                tail = u.split("/")[-1]
                if tail.isdigit():
                    out.add(tail)
        try:
            mrows = conn.execute(
                "SELECT cricsheet_match_id FROM matches WHERE cricsheet_match_id IS NOT NULL AND trim(cricsheet_match_id) != ''"
            ).fetchall()
        except sqlite3.OperationalError:
            mrows = []
        for r in mrows:
            mid = str(r[0] or "").strip()
            if mid.isdigit():
                out.add(mid)
    return out


def sql_cricsheet_match_result_ids(conn: sqlite3.Connection) -> list[int]:
    """SQLite ``match_results.id`` for rows ingested from ``cricsheet://ipl/{id}`` URLs."""
    rows = conn.execute(
        "SELECT id FROM match_results WHERE url LIKE 'cricsheet://ipl/%' ORDER BY id"
    ).fetchall()
    return [int(r[0]) for r in rows]


def delete_cricsheet_history_for_match_ids(conn: sqlite3.Connection, ids: list[int]) -> int:
    """
    Remove Stage-1-style rows for the given ``match_results`` ids (Cricsheet and same-shaped data).

    Deletes dependent history rows then ``match_results`` rows. Returns how many ``match_results``
    rows were deleted (last chunk only — same as total if one chunk).
    """
    if not ids:
        return 0
    deleted = 0
    chunk_size = 400
    for i in range(0, len(ids), chunk_size):
        part = ids[i : i + chunk_size]
        qm = ",".join("?" * len(part))
        conn.execute(f"DELETE FROM player_phase_usage WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM player_batting_positions WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM player_match_stats WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM team_match_xi WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM team_match_summary WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM matches WHERE id IN ({qm})", part)
        conn.execute(f"DELETE FROM match_context WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM match_xi WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM match_batting WHERE match_id IN ({qm})", part)
        conn.execute(f"DELETE FROM match_bowling WHERE match_id IN ({qm})", part)
        cur = conn.execute(f"DELETE FROM match_results WHERE id IN ({qm})", part)
        deleted += cur.rowcount if cur.rowcount is not None else len(part)
    return deleted


def delete_all_cricsheet_ingested_matches() -> dict[str, Any]:
    """Delete every Cricsheet-ingested fixture and dependent raw rows (admin rebuild)."""
    with connection() as conn:
        ids = sql_cricsheet_match_result_ids(conn)
        n = len(ids)
        deleted = delete_cricsheet_history_for_match_ids(conn, ids)
        return {
            "cricsheet_match_results_found": n,
            "match_results_deleted": deleted,
        }


def match_exists_by_url(url: str) -> bool:
    variants = _scorecard_url_variants(url)
    if not variants:
        return False
    with connection() as conn:
        qmarks = ",".join("?" * len(variants))
        row = conn.execute(
            f"SELECT 1 FROM match_results WHERE url IN ({qmarks}) LIMIT 1",
            variants,
        ).fetchone()
        return row is not None


def _clean_name(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _canonical_key_from_payload(payload: dict[str, Any]) -> Optional[str]:
    teams = list(payload.get("teams") or [])
    if len(teams) < 2:
        return None
    meta = payload.get("meta") or {}
    key = utils.canonical_match_identity_key(teams[0], teams[1], meta.get("date"))
    return key or None


def _find_existing_match_id_for_canonical(conn: sqlite3.Connection, key: str) -> Optional[int]:
    if not key:
        return None
    row = conn.execute(
        "SELECT id FROM match_results WHERE canonical_match_key = ? LIMIT 1",
        (key,),
    ).fetchone()
    if row:
        return int(row[0])
    tail = key.rsplit("|", 2)
    if len(tail) < 3:
        return None
    date_clause = tail[-1]
    if len(date_clause) < 4:
        return None
    lp = len(date_clause)
    like_pat = date_clause + "%"
    rows = conn.execute(
        """
        SELECT id, team_a, team_b, match_date FROM match_results
        WHERE match_date IS NOT NULL AND (
            substr(match_date, 1, ?) = ? OR match_date LIKE ?
        )
        LIMIT 800
        """,
        (lp, date_clause, like_pat),
    ).fetchall()
    for r in rows:
        k2 = utils.canonical_match_identity_key(r["team_a"], r["team_b"], r["match_date"])
        if k2 == key:
            return int(r["id"])
    return None


def insert_parsed_match(
    payload: dict[str, Any],
    *,
    skip_derived_aggregates: bool = False,
    resync_on_duplicate_match: bool = False,
) -> tuple[int, str]:
    """
    Store a parsed scorecard (partial OK).

    Returns ``(match_id, status)`` where status is ``inserted``, ``duplicate_url``,
    or ``duplicate_match`` (same teams+date as an existing row, different URL).

    When ``skip_derived_aggregates`` is True (bulk Cricsheet **ingest** stage), extended
    tables are still written, but ``player_franchise_features`` refresh is skipped — that
    belongs to a later **derive** stage.

    When ``resync_on_duplicate_match`` is True and a row already exists for the same
    canonical teams+date, **re-applies** ``matches`` / extended tables from this payload
    (e.g. enriching ``match_format`` from the full Cricsheet archive after an IPL-only row).
    """
    now = time.time()
    raw_json = json.dumps(payload, ensure_ascii=False)
    meta = payload.get("meta") or {}
    teams = payload.get("teams") or []
    team_a = teams[0] if len(teams) > 0 else None
    team_b = teams[1] if len(teams) > 1 else None
    url = (meta.get("url") or "").strip()

    with connection() as conn:
        if url:
            variants = _scorecard_url_variants(url)
            if variants:
                qmarks = ",".join("?" * len(variants))
                row = conn.execute(
                    f"SELECT id FROM match_results WHERE url IN ({qmarks}) LIMIT 1",
                    variants,
                ).fetchone()
                if row:
                    return int(row[0]), "duplicate_url"

        ckey = _canonical_key_from_payload(payload)
        if ckey:
            existing = _find_existing_match_id_for_canonical(conn, ckey)
            if existing is not None:
                if resync_on_duplicate_match:
                    try:
                        conn.execute(
                            """
                            UPDATE match_results SET raw_payload = ?, source = ?
                            WHERE id = ?
                            """,
                            (raw_json, meta.get("source", "unknown"), existing),
                        )
                        _sync_history_match_tables(conn, existing, payload, now)
                        _sync_extended_player_tables(
                            conn,
                            existing,
                            payload,
                            skip_derived_aggregates=skip_derived_aggregates,
                        )
                        logger.info(
                            "insert_parsed_match resynced_duplicate match_id=%s url=%s",
                            existing,
                            (meta.get("url") or "")[:100],
                        )
                        return existing, "resynced_duplicate"
                    except Exception:
                        logger.exception(
                            "resync_on_duplicate_match failed match_id=%s", existing
                        )
                        return existing, "duplicate_match"
                return existing, "duplicate_match"

        cur = conn.execute(
            """
            INSERT INTO match_results (
                url, source, team_a, team_b, venue, match_date, winner,
                toss_winner, toss_decision, batting_first, margin, raw_payload, created_at,
                canonical_match_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.get("url", ""),
                meta.get("source", "unknown"),
                team_a,
                team_b,
                meta.get("venue"),
                meta.get("date"),
                meta.get("winner"),
                meta.get("toss_winner"),
                meta.get("toss_decision"),
                meta.get("batting_first"),
                meta.get("margin"),
                raw_json,
                now,
                ckey,
            ),
        )
        match_id = int(cur.lastrowid)

        for side in payload.get("playing_xi") or []:
            tname = side.get("team") or ""
            pos = 0
            for name in side.get("players") or []:
                pn = _clean_name(name)
                if not pn:
                    continue
                pos += 1
                conn.execute(
                    """
                    INSERT INTO match_xi (match_id, team, player_name, bat_order, is_playing_xi)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (match_id, tname, pn, pos),
                )

        for inn in payload.get("batting") or []:
            tname = inn.get("team") or ""
            for row in inn.get("rows") or []:
                pn = _clean_name(row.get("player"))
                if not pn:
                    continue
                conn.execute(
                    """
                    INSERT INTO match_batting
                    (match_id, team, player_name, position, runs, balls)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        tname,
                        pn,
                        row.get("position"),
                        row.get("runs"),
                        row.get("balls"),
                    ),
                )

        for inn in payload.get("bowling") or []:
            tname = inn.get("team") or ""
            for row in inn.get("rows") or []:
                pn = _clean_name(row.get("player"))
                if not pn:
                    continue
                conn.execute(
                    """
                    INSERT INTO match_bowling
                    (match_id, team, player_name, overs, maidens, runs, wickets)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        tname,
                        pn,
                        row.get("overs"),
                        row.get("maidens"),
                        row.get("runs"),
                        row.get("wickets"),
                    ),
                )

        _write_match_context(conn, match_id, meta)
        try:
            _sync_history_match_tables(conn, match_id, payload, now)
        except Exception:
            logger.exception("history match sync failed for match_id=%s", match_id)
        try:
            _sync_extended_player_tables(
                conn, match_id, payload, skip_derived_aggregates=skip_derived_aggregates
            )
        except Exception:
            logger.exception("extended player tables sync failed for match_id=%s", match_id)

        logger.info(
            "insert_parsed_match match_id=%s status=inserted url=%s canonical_key=%s parser=%s",
            match_id,
            (meta.get("url") or "")[:120],
            ckey or "",
            meta.get("source", ""),
        )
        return match_id, "inserted"


def _write_match_context(conn: sqlite3.Connection, match_id: int, meta: dict[str, Any]) -> None:
    """Optional start hour / overseas counts for learning (all nullable)."""
    sh_raw = meta.get("start_hour_local")
    start_hour: Optional[int] = None
    is_night: Optional[int] = None
    dew_p: Optional[float] = None
    if sh_raw is not None and str(sh_raw).strip() != "":
        try:
            h = int(float(sh_raw))
            if 0 <= h <= 23:
                start_hour = h
                is_night = 1 if h >= config.NIGHT_START_HOUR_LOCAL else 0
        except (TypeError, ValueError):
            pass
    dew_raw = meta.get("dew_proxy_ingest")
    if dew_raw is not None and str(dew_raw).strip() != "":
        try:
            dew_p = float(dew_raw)
        except (TypeError, ValueError):
            dew_p = None
    if dew_p is None and is_night is not None:
        dew_p = float(config.DEW_PROXY_NIGHT if is_night else config.DEW_PROXY_DAY)

    def _oi(key: str) -> Optional[int]:
        v = meta.get(key)
        if v is None or str(v).strip() == "":
            return None
        try:
            x = int(float(v))
            return max(0, min(11, x))
        except (TypeError, ValueError):
            return None

    oa = _oi("overseas_in_xi_team_a")
    ob = _oi("overseas_in_xi_team_b")

    conn.execute(
        """
        INSERT OR REPLACE INTO match_context
        (match_id, start_hour_local, is_night, dew_proxy, overseas_team_a, overseas_team_b)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (match_id, start_hour, is_night, dew_p, oa, ob),
    )


def _keeper_name_heuristic(name: str) -> bool:
    n = (name or "").lower()
    return "wk)" in n or "(wk" in n or "wicket" in n or "keeper" in n


def _infer_role_bucket_row(
    *,
    is_keeper: bool,
    batting_position: Optional[float],
    overs_bowled: float,
) -> str:
    """Conservative labels for analytics rows — avoid labelling everyone All-Rounder on 1 over."""
    if is_keeper:
        return "WK-Batter"
    bp = batting_position if batting_position is not None else 99.0
    ov = float(overs_bowled or 0)
    if ov >= 2.0 and 3.0 <= bp <= 7.0:
        return "All-Rounder"
    if ov >= 3.0 and bp <= 6.0:
        return "All-Rounder"
    if ov >= 1.0 and bp >= 8.0:
        return "Bowler"
    if ov >= 2.0 and bp <= 7.0:
        return "All-Rounder"
    if ov > 0:
        return "Bowler"
    return "Batter"


def _batting_position_map(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    """
    Map (team_label, player_display_name) -> primary batting slot (1..11).

    Uses storage franchise labels (archive names → current IPL label) and merges
    ball-by-ball ``innings_batting_orders`` when scorecard rows omit positions.
    """
    import ipl_teams

    out: dict[tuple[str, str], float] = {}
    for inn in payload.get("batting") or []:
        t_raw = str(inn.get("team") or "").strip()
        t_store = ipl_teams.franchise_label_for_storage(t_raw) or t_raw
        for row in inn.get("rows") or []:
            pn = _clean_name(row.get("player"))
            if not pn:
                continue
            pos = row.get("position")
            if pos is None:
                continue
            try:
                fp = float(pos)
            except (TypeError, ValueError):
                continue
            out[(t_store, pn)] = fp
            out.setdefault((t_raw, pn), fp)
            clab = ipl_teams.canonical_franchise_label(t_raw)
            if clab and clab != t_store:
                out.setdefault((clab, pn), fp)

    locked: set[str] = set()
    blocks = sorted(
        list(payload.get("innings_batting_orders") or []),
        key=lambda b: int(b.get("innings_number") or 999),
    )
    for block in blocks:
        tm_raw = str(block.get("team") or "").strip()
        t_store = ipl_teams.franchise_label_for_storage(tm_raw) or tm_raw
        if t_store in locked:
            continue
        locked.add(t_store)
        for pos, pname in enumerate(block.get("order") or [], start=1):
            pn = _clean_name(pname)
            if not pn:
                continue
            fp = float(pos)
            if (t_store, pn) not in out:
                out[(t_store, pn)] = fp
            out.setdefault((tm_raw, pn), fp)

    return out


def _resolve_batting_position_for_xi(
    pos_map: dict[tuple[str, str], float],
    *,
    team_name_side: str,
    canon_label: str,
    team_key: str,
    player_name: str,
) -> Optional[float]:
    """Resolve slot for a playing-XI row using storage labels + normalized player keys."""
    import ipl_teams
    import learner as lr

    pname = _clean_name(player_name)
    if not pname:
        return None
    t_side = (team_name_side or "").strip()
    canon = (canon_label or "").strip()
    tk = (team_key or "").strip()
    t_store = ipl_teams.franchise_label_for_storage(t_side) or t_side

    for tlab in (t_store, canon, t_side):
        if not tlab:
            continue
        v = pos_map.get((tlab, pname))
        if v is not None:
            return float(v)

    pk_n = lr.normalize_player_key(pname)
    if not pk_n:
        return None

    for (t_k, p_k), val in pos_map.items():
        if lr.normalize_player_key(str(p_k)) != pk_n:
            continue
        t_res = ipl_teams.franchise_label_for_storage(str(t_k))
        if lr.normalize_player_key(t_res) == tk:
            return float(val)
        if t_res.lower() == canon.lower():
            return float(val)
        ctl = ipl_teams.canonical_franchise_label(str(t_k)) or str(t_k)
        if lr.normalize_player_key(ctl) == tk:
            return float(val)
    return None


def _overs_bowled_map(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    import ipl_teams

    out: dict[tuple[str, str], float] = {}
    for inn in payload.get("bowling") or []:
        t_raw = str(inn.get("team") or "").strip()
        t_store = ipl_teams.franchise_label_for_storage(t_raw) or t_raw
        for row in inn.get("rows") or []:
            pn = _clean_name(row.get("player"))
            if not pn:
                continue
            try:
                ov = float(row.get("overs") or 0)
            except (TypeError, ValueError):
                ov = 0.0
            for tlab in {t_store, t_raw}:
                key = (tlab, pn)
                out[key] = out.get(key, 0.0) + ov
    return out


def _bowling_type_from_row(row: dict[str, Any]) -> Optional[str]:
    # Optional: parsers may add style later
    return None


def _phase_from_over_number(over_number: int) -> str:
    # Stored over_number is 0-based in Cricsheet payload.
    ov = int(over_number)
    if ov <= 5:
        return "powerplay"
    if ov <= 14:
        return "middle"
    return "death"


def _insert_match_ball_by_ball_rows(
    conn: sqlite3.Connection,
    match_id: int,
    payload: dict[str, Any],
) -> int:
    import ipl_teams
    import learner

    events = payload.get("delivery_events") or []
    if not isinstance(events, list) or not events:
        return 0
    n = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        bteam_raw = str(ev.get("batting_team") or "").strip()
        fteam_raw = str(ev.get("bowling_team") or "").strip()
        batter = str(ev.get("batter") or "").strip()
        bowler = str(ev.get("bowler") or "").strip()
        if not bteam_raw or not fteam_raw or not batter or not bowler:
            continue
        bteam = ipl_teams.franchise_label_for_storage(bteam_raw) or bteam_raw
        fteam = ipl_teams.franchise_label_for_storage(fteam_raw) or fteam_raw
        batter_key = str(learner.normalize_player_key(batter) or "")[:80]
        bowler_key = str(learner.normalize_player_key(bowler) or "")[:80]
        if not batter_key or not bowler_key:
            continue
        try:
            over_n = int(ev.get("over_number") or 0)
            ball_n = int(ev.get("ball_in_over") or 0)
            runs_batter = int(ev.get("runs_batter") or 0)
            runs_total = int(ev.get("runs_total") or 0)
            innings_n = int(ev.get("innings_number") or 1)
            legal = 1 if int(ev.get("is_legal_ball") or 0) == 1 else 0
            dismissal = 1 if int(ev.get("is_dismissal") or 0) == 1 else 0
        except (TypeError, ValueError):
            continue
        phase = str(ev.get("phase") or _phase_from_over_number(over_n)).strip().lower()
        if phase not in ("powerplay", "middle", "death"):
            phase = _phase_from_over_number(over_n)
        conn.execute(
            """
            INSERT INTO match_ball_by_ball (
                match_id, innings_number, batting_team_key, bowling_team_key,
                over_number, ball_in_over, phase,
                batter_name, batter_key, bowler_name, bowler_key,
                runs_batter, runs_total,
                is_legal_ball, is_dot_ball, is_boundary, is_dismissal,
                dismissal_kind, batter_out_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                innings_n,
                str(ipl_teams.canonical_team_key_for_franchise(bteam) or "")[:80],
                str(ipl_teams.canonical_team_key_for_franchise(fteam) or "")[:80],
                over_n,
                ball_n,
                phase,
                batter,
                batter_key,
                bowler,
                bowler_key,
                runs_batter,
                runs_total,
                legal,
                1 if legal and runs_total == 0 else 0,
                1 if runs_batter >= 4 else 0,
                dismissal,
                str(ev.get("dismissal_kind") or "").strip().lower() or None,
                str(learner.normalize_player_key(str(ev.get("batter_out") or "").strip()) or "")[:80] or None,
            ),
        )
        n += 1
    return n


def _sync_history_match_tables(
    conn: sqlite3.Connection,
    match_result_id: int,
    payload: dict[str, Any],
    created_at: float,
) -> None:
    """Populate IPL `matches`, `team_match_xi`, `team_match_summary` (1:1 with match_results id)."""
    import ipl_teams
    import learner

    meta = payload.get("meta") or {}
    teams = list(payload.get("teams") or [])
    team_a = teams[0] if len(teams) > 0 else None
    team_b = teams[1] if len(teams) > 1 else None
    result_bits = []
    if meta.get("winner"):
        result_bits.append(f"Winner: {meta.get('winner')}")
    if meta.get("margin"):
        result_bits.append(str(meta.get("margin")))
    result_str = " | ".join(result_bits) if result_bits else ""

    cs_mid = meta.get("cricsheet_match_id")
    cs_mid_s = str(cs_mid).strip() if cs_mid is not None and str(cs_mid).strip() else None
    result_text = meta.get("result_text")
    if not (result_text and str(result_text).strip()):
        result_text = meta.get("margin")

    conn.execute(
        """
        INSERT OR REPLACE INTO matches (
            id, competition, match_date, venue, team_a, team_b, result,
            scorecard_url, source, batting_first, created_at,
            cricsheet_match_id, city, season, toss_winner, toss_decision, winner, result_text,
            match_format
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_result_id,
            str(meta.get("competition") or config.IPL_COMPETITION_LABEL),
            meta.get("date"),
            meta.get("venue"),
            team_a,
            team_b,
            result_str,
            meta.get("url", ""),
            meta.get("source", "unknown"),
            meta.get("batting_first"),
            created_at,
            cs_mid_s,
            meta.get("city"),
            meta.get("season"),
            meta.get("toss_winner"),
            meta.get("toss_decision"),
            meta.get("winner"),
            result_text,
            meta.get("match_format"),
        ),
    )

    conn.execute("DELETE FROM team_match_xi WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM team_match_summary WHERE match_id = ?", (match_result_id,))

    pos_map = _batting_position_map(payload)
    overs_map = _overs_bowled_map(payload)
    batting_order = payload.get("batting_order") or []
    bowlers_used = payload.get("bowlers_used") or []

    oa = meta.get("overseas_in_xi_team_a")
    ob = meta.get("overseas_in_xi_team_b")
    overseas_combo: dict[str, Any] = {}
    if oa is not None and str(oa).strip() != "":
        try:
            overseas_combo["team_a_overseas"] = int(float(oa))
        except (TypeError, ValueError):
            pass
    if ob is not None and str(ob).strip() != "":
        try:
            overseas_combo["team_b_overseas"] = int(float(ob))
        except (TypeError, ValueError):
            pass

    impact_blob: dict[str, Any] = {}
    if meta.get("impact_players"):
        impact_blob["impact_players"] = meta.get("impact_players")
    if meta.get("impact_substitute"):
        impact_blob["impact_substitute"] = meta.get("impact_substitute")

    for side in payload.get("playing_xi") or []:
        tname = str(side.get("team") or "").strip()
        if not tname:
            continue
        canon_label = ipl_teams.franchise_label_for_storage(tname)
        tk = ipl_teams.canonical_team_key_for_franchise(canon_label)[:80]
        players = [str(p).strip() for p in (side.get("players") or []) if str(p).strip()]
        bo_for_team: list[str] = []
        for block in batting_order:
            bt = str(block.get("team") or "").strip()
            if not bt:
                continue
            bcan = ipl_teams.canonical_franchise_label(bt) or bt
            if bcan.lower() == canon_label.lower() or bt.lower() == tname.lower():
                bo_for_team = [str(x) for x in (block.get("order") or []) if x]
                break
        bow_for_team: list[str] = []
        for block in bowlers_used:
            bt = str(block.get("team") or "").strip()
            if not bt:
                continue
            bcan = ipl_teams.canonical_franchise_label(bt) or bt
            if bcan.lower() == canon_label.lower() or bt.lower() == tname.lower():
                bow_for_team = [str(x) for x in (block.get("bowlers") or []) if x]
                break

        conn.execute(
            """
            INSERT INTO team_match_summary (
                match_id, team_name, team_key, playing_xi_json, batting_order_json,
                bowlers_used_json, overseas_combo_json, impact_player_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_result_id,
                canon_label,
                tk,
                json.dumps(players, ensure_ascii=False),
                json.dumps(bo_for_team, ensure_ascii=False),
                json.dumps(bow_for_team, ensure_ascii=False),
                json.dumps(overseas_combo, ensure_ascii=False),
                json.dumps(impact_blob, ensure_ascii=False),
            ),
        )

        for idx, pname in enumerate(players[:11], start=1):
            pk = learner.normalize_player_key(pname)
            if not pk:
                continue
            bp = _resolve_batting_position_for_xi(
                pos_map,
                team_name_side=tname,
                canon_label=canon_label,
                team_key=tk,
                player_name=pname,
            )
            ov = overs_map.get((canon_label, pname))
            if ov is None:
                ov = overs_map.get((tname, pname))
            if ov is None:
                ov = 0.0
            if ov == 0.0:
                pk_n = learner.normalize_player_key(pname)
                for alt_t, alt_p in list(overs_map.keys()):
                    if learner.normalize_player_key(alt_p) != pk_n:
                        continue
                    t_res = ipl_teams.franchise_label_for_storage(str(alt_t))
                    if t_res.lower() == canon_label.lower() or ipl_teams.canonical_team_key_for_franchise(
                        t_res
                    ) == tk:
                        ov = overs_map[(alt_t, alt_p)]
                        break
            is_k = 1 if _keeper_name_heuristic(pname) else 0
            rb = _infer_role_bucket_row(
                is_keeper=bool(is_k),
                batting_position=bp,
                overs_bowled=float(ov or 0),
            )
            conn.execute(
                """
                INSERT INTO team_match_xi (
                    match_id, team_name, player_name, team_key, player_key,
                    canonical_team_key, canonical_player_key,
                    selected_in_xi, batting_position, is_keeper, overs_bowled,
                    batting_order_index, bowling_type, role_bucket,
                    is_impact_used, overseas
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, 0, 0)
                """,
                (
                    match_result_id,
                    canon_label,
                    pname,
                    tk,
                    pk,
                    tk,
                    pk,
                    bp,
                    is_k,
                    float(ov or 0),
                    idx,
                    _bowling_type_from_row({}),
                    rb,
                ),
            )

    logger.info(
        "history sync match_id=%s teams=%s xi_players=%d",
        match_result_id,
        teams,
        sum(len((s.get("players") or [])[:11]) for s in (payload.get("playing_xi") or [])),
    )


def _sync_extended_player_tables(
    conn: sqlite3.Connection,
    match_result_id: int,
    payload: dict[str, Any],
    *,
    skip_derived_aggregates: bool = False,
) -> None:
    """Optional Cricsheet-derived rows: ``player_match_stats``, ``player_phase_usage``, positions."""
    import ipl_teams
    import learner

    conn.execute("DELETE FROM player_match_stats WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM player_phase_usage WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM player_batting_positions WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM player_match_batting_summary WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM player_match_bowling_summary WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM player_match_role_usage WHERE match_id = ?", (match_result_id,))
    conn.execute("DELETE FROM match_ball_by_ball WHERE match_id = ?", (match_result_id,))
    bb_rows_inserted = _insert_match_ball_by_ball_rows(conn, match_result_id, payload)

    for row in payload.get("player_stats_extended") or []:
        pn = _clean_name(row.get("player_name"))
        if not pn:
            continue
        tn = _clean_name(row.get("team_name")) or ""
        tn_s = ipl_teams.franchise_label_for_storage(tn) or tn
        tk_row = ipl_teams.canonical_team_key_for_franchise(tn_s)[:80]
        pk_row = str(row.get("player_key") or learner.normalize_player_key(pn))[:80]
        conn.execute(
            """
            INSERT INTO player_match_stats (
                match_id, team_name, team_key, player_name, player_key,
                canonical_team_key, canonical_player_key,
                season,
                selected_in_xi, batting_position, runs, balls, fours, sixes, strike_rate,
                overs_bowled, wickets, runs_conceded, economy, dismissal_type,
                vs_spin_balls_faced, vs_pace_balls_faced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_result_id,
                tn_s,
                tk_row,
                pn,
                pk_row,
                tk_row,
                pk_row,
                _clean_name(row.get("season")),
                int(row.get("selected_in_xi") or 0),
                row.get("batting_position"),
                row.get("runs"),
                row.get("balls"),
                row.get("fours"),
                row.get("sixes"),
                row.get("strike_rate"),
                row.get("overs_bowled"),
                row.get("wickets"),
                row.get("runs_conceded"),
                row.get("economy"),
                _clean_name(row.get("dismissal_type")),
                int(row.get("vs_spin_balls_faced") or 0),
                int(row.get("vs_pace_balls_faced") or 0),
            ),
        )
        br = row.get("runs")
        bb = row.get("balls")
        if (br is not None and int(br or 0) > 0) or (bb is not None and int(bb or 0) > 0):
            conn.execute(
                """
                INSERT INTO player_match_batting_summary (
                    match_id, team_key, team_name, player_key, player_name,
                    canonical_team_key, canonical_player_key,
                    position, runs, balls, fours, sixes, strike_rate, dismissal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_result_id,
                    tk_row,
                    tn_s,
                    pk_row,
                    pn,
                    tk_row,
                    pk_row,
                    row.get("batting_position"),
                    row.get("runs"),
                    row.get("balls"),
                    row.get("fours"),
                    row.get("sixes"),
                    row.get("strike_rate"),
                    _clean_name(row.get("dismissal_type")),
                ),
            )
        try:
            ob = float(row.get("overs_bowled") or 0)
        except (TypeError, ValueError):
            ob = 0.0
        if ob > 0:
            conn.execute(
                """
                INSERT INTO player_match_bowling_summary (
                    match_id, team_key, team_name, player_key, player_name,
                    canonical_team_key, canonical_player_key,
                    overs_bowled, maidens, wickets, runs_conceded, economy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_result_id,
                    tk_row,
                    tn_s,
                    pk_row,
                    pn,
                    tk_row,
                    pk_row,
                    ob,
                    0,
                    row.get("wickets"),
                    row.get("runs_conceded"),
                    row.get("economy"),
                ),
            )

    meta_season = _clean_name((payload.get("meta") or {}).get("season"))
    n_pbp = 0
    pbp_debug: list[dict[str, Any]] = []
    for block in payload.get("innings_batting_orders") or []:
        inn_n = int(block.get("innings_number") or 1)
        tm_raw = str(block.get("team") or "").strip()
        if not tm_raw:
            continue
        tname_canon = ipl_teams.franchise_label_for_storage(tm_raw)
        tk = ipl_teams.canonical_team_key_for_franchise(tname_canon)[:80]
        order = list(block.get("order") or [])
        for pos, pname in enumerate(order, start=1):
            pn = _clean_name(pname)
            if not pn:
                continue
            pk = str(learner.normalize_player_key(pn))[:80]
            conn.execute(
                """
                INSERT OR REPLACE INTO player_batting_positions (
                    match_id, team_name, team_key, player_name, player_key,
                    canonical_team_key, canonical_player_key,
                    batting_position, season, innings_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_result_id,
                    tname_canon,
                    tk,
                    pn,
                    pk,
                    tk,
                    pk,
                    float(pos),
                    meta_season,
                    inn_n,
                ),
            )
            n_pbp += 1
            if len(pbp_debug) < 20:
                pbp_debug.append(
                    {
                        "innings": inn_n,
                        "team_name": tname_canon,
                        "team_key": tk,
                        "batting_position": pos,
                        "player_name": pn,
                        "player_key": pk,
                    }
                )
    if n_pbp:
        logger.info(
            "player_batting_positions match_id=%s rows_inserted=%s sample=%s",
            match_result_id,
            n_pbp,
            pbp_debug,
        )
    if bb_rows_inserted:
        logger.info(
            "match_ball_by_ball match_id=%s rows_inserted=%s",
            match_result_id,
            bb_rows_inserted,
        )

    for row in payload.get("player_phase_extended") or []:
        pn = _clean_name(row.get("player_name"))
        if not pn:
            continue
        tn = _clean_name(row.get("team_name")) or ""
        tn_s = ipl_teams.franchise_label_for_storage(tn) or tn
        tk_row = ipl_teams.canonical_team_key_for_franchise(tn_s)[:80]
        pk_ph = str(row.get("player_key") or learner.normalize_player_key(pn))[:80]
        conn.execute(
            """
            INSERT INTO player_phase_usage (
                match_id, team_name, team_key, player_name, player_key,
                canonical_team_key, canonical_player_key,
                role, phase, balls, runs, wickets, vs_spin_balls, vs_pace_balls
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_result_id,
                tn_s,
                tk_row,
                pn,
                pk_ph,
                tk_row,
                pk_ph,
                str(row.get("role") or ""),
                str(row.get("phase") or ""),
                int(row.get("balls") or 0),
                int(row.get("runs") or 0),
                int(row.get("wickets") or 0),
                int(row.get("vs_spin_balls") or 0),
                int(row.get("vs_pace_balls") or 0),
            ),
        )
        conn.execute(
            """
            INSERT INTO player_match_role_usage (
                match_id, team_name, team_key, player_name, player_key,
                canonical_team_key, canonical_player_key,
                role, phase, balls, runs, wickets, vs_spin_balls, vs_pace_balls
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_result_id,
                tn_s,
                tk_row,
                pn,
                pk_ph,
                tk_row,
                pk_ph,
                str(row.get("role") or ""),
                str(row.get("phase") or ""),
                int(row.get("balls") or 0),
                int(row.get("runs") or 0),
                int(row.get("wickets") or 0),
                int(row.get("vs_spin_balls") or 0),
                int(row.get("vs_pace_balls") or 0),
            ),
        )

    if not skip_derived_aggregates:
        import matchup_features

        team_keys = {
            str(r.get("team_key") or "").strip()[:80]
            for r in (payload.get("player_stats_extended") or [])
        }
        team_keys.discard("")
        for tk in sorted(team_keys):
            try:
                matchup_features.refresh_franchise_features(conn, tk)
            except Exception:
                logger.exception("refresh_franchise_features failed team_key=%s", tk)


def player_phase_bowl_rates(
    player_key: str,
    franchise_team_key: str,
    *,
    limit_matches: int = 40,
) -> dict[str, float]:
    """
    Fraction of recent franchise matches (distinct, capped) where the player bowled
    at least one legal ball in each phase (powerplay / middle / death).
    """
    pk = (player_key or "").strip()
    fk = (franchise_team_key or "").strip()
    if not pk or not fk:
        return {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
    lim = max(5, min(120, int(limit_matches)))
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT p.match_id
            FROM player_phase_usage p
            JOIN matches m ON m.id = p.match_id
            WHERE p.team_key = ? AND p.player_key = ? AND p.role = 'bowl'
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
            """,
            (fk, pk, lim * 3),
        ).fetchall()
        mids = [int(r[0]) for r in rows][:lim]
        if not mids:
            return {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
        qm = ",".join("?" * len(mids))
        hits = conn.execute(
            f"""
            SELECT phase, COUNT(DISTINCT match_id) AS c
            FROM player_phase_usage
            WHERE player_key = ? AND team_key = ? AND role = 'bowl'
              AND balls > 0 AND match_id IN ({qm})
            GROUP BY phase
            """,
            [pk, fk] + mids,
        ).fetchall()
    by_ph = {str(r["phase"]): int(r["c"]) for r in hits}
    n = float(len(mids))
    return {
        "powerplay": by_ph.get("powerplay", 0) / n,
        "middle": by_ph.get("middle", 0) / n,
        "death": by_ph.get("death", 0) / n,
    }


def player_spin_pace_faced_share(player_key: str, franchise_team_key: str, *, limit_rows: int = 80) -> dict[str, float]:
    """Aggregate batter balls faced vs heuristic spin/pace from recent ``player_match_stats`` rows."""
    pk = (player_key or "").strip()
    fk = (franchise_team_key or "").strip()
    if not pk or not fk:
        return {"spin_share": 0.0, "pace_share": 0.0, "samples": 0}
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT s.vs_spin_balls_faced, s.vs_pace_balls_faced
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.player_key = ? AND s.team_key = ?
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
            """,
            (pk, fk, max(10, min(400, int(limit_rows)))),
        ).fetchall()
    sp = pp = 0
    for r in rows:
        sp += int(r["vs_spin_balls_faced"] or 0)
        pp += int(r["vs_pace_balls_faced"] or 0)
    tot = sp + pp
    if tot <= 0:
        return {"spin_share": 0.0, "pace_share": 0.0, "samples": 0}
    return {
        "spin_share": sp / tot,
        "pace_share": pp / tot,
        "samples": tot,
    }


def batch_player_phase_bowl_rates(
    player_keys: list[str],
    franchise_team_key: str,
    *,
    limit_matches: int = 40,
) -> dict[str, dict[str, float]]:
    """Batch variant of ``player_phase_bowl_rates`` with a single DB connection."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    lim = max(5, min(120, int(limit_matches)))
    out: dict[str, dict[str, float]] = {}
    with connection() as conn:
        for pk in keys:
            rows = conn.execute(
                """
                SELECT DISTINCT p.match_id
                FROM player_phase_usage p
                JOIN matches m ON m.id = p.match_id
                WHERE p.team_key = ? AND p.player_key = ? AND p.role = 'bowl'
                ORDER BY m.match_date DESC NULLS LAST, m.id DESC
                LIMIT ?
                """,
                (fk, pk, lim * 3),
            ).fetchall()
            mids = [int(r[0]) for r in rows][:lim]
            if not mids:
                out[pk] = {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
                continue
            qm = ",".join("?" * len(mids))
            hits = conn.execute(
                f"""
                SELECT phase, COUNT(DISTINCT match_id) AS c
                FROM player_phase_usage
                WHERE player_key = ? AND team_key = ? AND role = 'bowl'
                  AND balls > 0 AND match_id IN ({qm})
                GROUP BY phase
                """,
                [pk, fk] + mids,
            ).fetchall()
            by_ph = {str(r["phase"]): int(r["c"]) for r in hits}
            n = float(len(mids))
            out[pk] = {
                "powerplay": by_ph.get("powerplay", 0) / n,
                "middle": by_ph.get("middle", 0) / n,
                "death": by_ph.get("death", 0) / n,
            }
    return out


def batch_player_spin_pace_faced_share(
    player_keys: list[str],
    franchise_team_key: str,
    *,
    limit_rows: int = 80,
) -> dict[str, dict[str, float]]:
    """Batch variant of ``player_spin_pace_faced_share`` with a single DB connection."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    lim = max(10, min(400, int(limit_rows)))
    out: dict[str, dict[str, float]] = {}
    with connection() as conn:
        for pk in keys:
            rows = conn.execute(
                """
                SELECT s.vs_spin_balls_faced, s.vs_pace_balls_faced
                FROM player_match_stats s
                JOIN matches m ON m.id = s.match_id
                WHERE s.player_key = ? AND s.team_key = ?
                ORDER BY m.match_date DESC NULLS LAST, m.id DESC
                LIMIT ?
                """,
                (pk, fk, lim),
            ).fetchall()
            sp = pp = 0
            for r in rows:
                sp += int(r["vs_spin_balls_faced"] or 0)
                pp += int(r["vs_pace_balls_faced"] or 0)
            tot = sp + pp
            if tot <= 0:
                out[pk] = {"spin_share": 0.0, "pace_share": 0.0, "samples": 0}
            else:
                out[pk] = {
                    "spin_share": sp / tot,
                    "pace_share": pp / tot,
                    "samples": tot,
                }
    return out


def refresh_all_player_franchise_features(min_season_year: Optional[int] = None) -> int:
    """
    Recompute ``player_franchise_features`` for every ``team_key`` present in
    ``player_match_stats`` (e.g. after a DB restore or schema upgrade).

    Optional ``min_season_year`` limits rows to ``matches.match_date`` years in range
    (used by Stage 2 derive).
    """
    import matchup_features

    n = 0
    with connection() as conn:
        keys = [
            str(r[0]).strip()[:80]
            for r in conn.execute(
                "SELECT DISTINCT team_key FROM player_match_stats WHERE team_key IS NOT NULL AND trim(team_key) != ''"
            ).fetchall()
        ]
        for tk in sorted(set(keys)):
            n += matchup_features.refresh_franchise_features(conn, tk, min_season_year=min_season_year)
    return n


def get_player_franchise_features(
    player_key: str,
    franchise_team_key: str,
) -> Optional[dict[str, Any]]:
    """Latest recomputed Cricsheet-driven matchup row for (player, franchise)."""
    pk = (player_key or "").strip()[:80]
    fk = (franchise_team_key or "").strip()[:80]
    if not pk or not fk:
        return None
    with connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM player_franchise_features
            WHERE player_key = ? AND franchise_team_key = ?
            LIMIT 1
            """,
            (pk, fk),
        ).fetchone()
    return dict(row) if row else None


def batch_get_player_franchise_features(
    player_keys: list[str],
    franchise_team_key: str,
) -> dict[str, dict[str, Any]]:
    """One round-trip for many ``player_franchise_features`` rows (same franchise)."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [str(k).strip()[:80] for k in player_keys if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM player_franchise_features
            WHERE franchise_team_key = ? AND player_key IN ({qm})
            """,
            [fk] + keys,
        ).fetchall()
    return {str(r["player_key"]): dict(r) for r in rows}


def batch_fetch_primary_pbp_slots_for_franchise(
    canonical_franchise_label: str,
    sql_key_to_match_ids: dict[str, list[int]],
) -> dict[str, dict[int, float]]:
    """
    Batting slot from first innings per (player_key/sql_key, match_id), one query for the whole squad.

    ``sql_key_to_match_ids`` maps history lookup / ``player_key`` → match ids to resolve (union fetched in SQL).
    """
    import ipl_teams

    if not sql_key_to_match_ids:
        return {}
    lab = (canonical_franchise_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    if not ck:
        return {}
    all_mids: set[int] = set()
    ukeys: list[str] = []
    for sk, mids in sql_key_to_match_ids.items():
        pk = str(sk).strip()[:80]
        if not pk:
            continue
        ukeys.append(pk)
        for m in mids:
            try:
                all_mids.add(int(m))
            except (TypeError, ValueError):
                continue
    ukeys = list(dict.fromkeys(ukeys))
    umids = sorted(all_mids)
    if not ukeys or not umids:
        return {}
    qk = ",".join("?" * len(ukeys))
    qm = ",".join("?" * len(umids))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key, match_id, innings_number, batting_position
            FROM player_batting_positions
            WHERE team_key = ? AND player_key IN ({qk}) AND match_id IN ({qm})
            ORDER BY player_key ASC, match_id ASC, innings_number ASC
            """,
            [ck] + ukeys + umids,
        ).fetchall()
    by_pk_mid: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()[:80]
        mid = int(r["match_id"])
        inn = int(r["innings_number"] or 1)
        try:
            bp = float(r["batting_position"])
        except (TypeError, ValueError):
            continue
        by_pk_mid.setdefault(pk, {}).setdefault(mid, []).append((inn, bp))
    out: dict[str, dict[int, float]] = {}
    for pk, mids_map in by_pk_mid.items():
        inner: dict[int, float] = {}
        for mid, pairs in mids_map.items():
            pairs.sort(key=lambda x: x[0])
            inner[mid] = pairs[0][1]
        out[pk] = inner
    return out


def fetch_primary_pbp_slot_by_match_for_player(
    canonical_franchise_label: str,
    player_key: str,
    match_ids: list[int],
) -> dict[int, float]:
    """
    For each match_id, batting slot from **first innings** that has a
    ``player_batting_positions`` row for this player + franchise.
    """
    import ipl_teams

    if not match_ids:
        return {}
    lab = (canonical_franchise_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    pk = (player_key or "").strip()[:80]
    if not pk:
        return {}
    qm = ",".join("?" * len(match_ids))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT match_id, innings_number, batting_position
            FROM player_batting_positions
            WHERE player_key = ?
              AND team_key = ?
              AND match_id IN ({qm})
            ORDER BY match_id ASC, innings_number ASC
            """,
            [pk, ck] + match_ids,
        ).fetchall()
    by_mid: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        mid = int(r["match_id"])
        inn = int(r["innings_number"] or 1)
        try:
            bp = float(r["batting_position"])
        except (TypeError, ValueError):
            continue
        by_mid.setdefault(mid, []).append((inn, bp))
    out: dict[int, float] = {}
    for mid, pairs in by_mid.items():
        pairs.sort(key=lambda x: x[0])
        out[mid] = pairs[0][1]
    return out


def count_player_batting_positions_for_franchise(canonical_label: str) -> int:
    """Row count in ``player_batting_positions`` for this franchise (team_key / name)."""
    import ipl_teams

    lab = (canonical_label or "").strip()
    if not lab:
        return 0
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    with connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM player_batting_positions
            WHERE team_key = ?
            """,
            (ck,),
        ).fetchone()
    return int(row[0]) if row else 0


def batting_positions_sqlite_pipeline_summary(*, top_team_keys: int = 12, top_player_keys: int = 20) -> dict[str, Any]:
    """Debug: table size, per-team / per-player-key counts for the batting-slot pipeline."""
    with connection() as conn:
        total = int(
            conn.execute("SELECT COUNT(*) FROM player_batting_positions").fetchone()[0]
        )
        by_team = conn.execute(
            """
            SELECT team_key, team_name, COUNT(*) AS n
            FROM player_batting_positions
            GROUP BY team_key, team_name
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(top_team_keys),),
        ).fetchall()
        by_player = conn.execute(
            """
            SELECT player_key, COUNT(*) AS n
            FROM player_batting_positions
            GROUP BY player_key
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(top_player_keys),),
        ).fetchall()
    return {
        "player_batting_positions_total_rows": total,
        "rows_per_team_key_sample": [dict(r) for r in by_team],
        "rows_per_player_key_sample": [dict(r) for r in by_player],
    }


def squad_pbp_coverage_for_franchise(
    canonical_label: str,
    squad_player_keys: set[str],
) -> dict[str, Any]:
    """How many current-squad player_keys have at least one PBP row for this franchise."""
    import ipl_teams

    lab = (canonical_label or "").strip()
    keys = {str(k).strip()[:80] for k in squad_player_keys if str(k).strip()}
    if not lab or not keys:
        return {"franchise": lab, "squad_keys_checked": 0, "keys_with_pbp_rows": 0, "missing_keys": []}
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    with connection() as conn:
        ph = ",".join("?" * len(keys))
        rows = conn.execute(
            f"""
            SELECT DISTINCT player_key FROM player_batting_positions
            WHERE team_key = ?
              AND player_key IN ({ph})
            """,
            [ck] + list(keys),
        ).fetchall()
    have = {str(r["player_key"]) for r in rows}
    missing = sorted(keys - have)
    return {
        "franchise": lab,
        "canonical_team_key": ck,
        "squad_keys_checked": len(keys),
        "keys_with_pbp_rows": len(have & keys),
        "missing_keys_sample": missing[:30],
    }


def franchise_history_snapshot(canonical_label: str) -> dict[str, Any]:
    """
    Distinct completed-match coverage for a franchise in local SQLite.

    Read-only SQLite snapshot for UI / sufficiency hints (no internet; load data via ingest stage).
    only (Cricsheet backfill or manual ingest is expected to populate rows).
    """
    import ipl_teams

    lab = (canonical_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    cur_year = int(getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    stale_days = float(getattr(config, "HISTORY_SYNC_STALE_DAYS", 10.0))

    with connection() as conn:
        xi_row = conn.execute(
            "SELECT COUNT(*) AS c FROM team_match_xi WHERE team_key = ?",
            (ck,),
        ).fetchone()
        xi_row_count = int(xi_row["c"] or 0) if xi_row else 0
        sm_row = conn.execute(
            "SELECT COUNT(*) AS c FROM team_match_summary WHERE team_key = ?",
            (ck,),
        ).fetchone()
        summary_row_count = int(sm_row["c"] or 0) if sm_row else 0
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS distinct_match_count,
                SUM(
                    CASE
                        WHEN m.match_date IS NOT NULL
                         AND trim(m.match_date) != ''
                         AND CAST(substr(trim(m.match_date), 1, 4) AS INTEGER) < ?
                        THEN 1 ELSE 0
                    END
                ) AS prior_season_match_count,
                MAX(COALESCE(m.created_at, 0)) AS newest_created_at
            FROM matches m
            JOIN (
                SELECT DISTINCT match_id
                FROM team_match_xi
                WHERE team_key = ?
            ) t ON t.match_id = m.id
            """,
            (cur_year, ck),
        ).fetchone()
    distinct_count = int(agg["distinct_match_count"] or 0) if agg else 0
    prior_season = int(agg["prior_season_match_count"] or 0) if agg else 0
    newest_created: Optional[float] = None
    if agg is not None:
        try:
            newest_created = float(agg["newest_created_at"]) if agg["newest_created_at"] is not None else None
        except (TypeError, ValueError):
            newest_created = None

    stale_local = False
    if newest_created is not None:
        stale_local = (time.time() - newest_created) / 86400.0 > stale_days

    target_recent = int(getattr(config, "HISTORY_SYNC_TARGET_RECENT_MATCHES", 10))
    recent_window_n = min(distinct_count, target_recent)

    return {
        "canonical_label": lab,
        "team_key": ck,
        "distinct_match_count": distinct_count,
        "xi_row_count": xi_row_count,
        "team_match_summary_row_count": summary_row_count,
        "recent_window_distinct_count": recent_window_n,
        "prior_season_match_count": prior_season,
        "newest_created_at": newest_created,
        "days_since_newest_created": (
            (time.time() - newest_created) / 86400.0 if newest_created is not None else None
        ),
        "stale_local_cache": stale_local,
        "ipl_current_season_year": cur_year,
    }


def get_cached_match_count_for_franchise(canonical_label: str) -> int:
    """Distinct matches in SQLite for this franchise (``team_match_xi`` coverage)."""
    import ipl_teams

    lab = (canonical_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    if not ck:
        return 0
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT match_id) AS c FROM team_match_xi WHERE team_key = ?",
            (ck,),
        ).fetchone()
    return int(row["c"] or 0) if row else 0


def franchise_recent_match_summaries(canonical_label: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Recent stored matches for debug / UI (teams, date, URL, source)."""
    import ipl_teams

    lab = (canonical_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.match_date, m.team_a, m.team_b, m.scorecard_url, m.source,
                   m.result, m.competition, m.created_at
            FROM matches m
            WHERE m.id IN (
                SELECT DISTINCT x.match_id FROM team_match_xi x
                WHERE x.team_key = ?
            )
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
            """,
            (ck, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def history_team_xi_rows(team_key: str, *, limit: int = 500) -> list[dict[str, Any]]:
    """Legacy: rows for a single stored team_key only. Prefer ``history_team_xi_rows_for_franchise``."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT x.match_id, x.team_name, x.player_name, x.team_key, x.player_key,
                   x.batting_position, x.is_keeper, x.overs_bowled, x.batting_order_index,
                   x.role_bucket,
                   m.match_date, m.venue, m.batting_first, m.team_a, m.team_b, m.created_at,
                   mc.overseas_team_a, mc.overseas_team_b
            FROM team_match_xi x
            JOIN matches m ON m.id = x.match_id
            LEFT JOIN match_context mc ON mc.match_id = m.id
            WHERE x.team_key = ?
            ORDER BY m.created_at DESC, m.id DESC, x.batting_order_index ASC
            LIMIT ?
            """,
            (team_key, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def history_team_xi_rows_for_franchise(canonical_label: str, *, limit: int = 650) -> list[dict[str, Any]]:
    """
    Strict franchise history: same canonical team_key **or** exact team_name match (legacy rows).

    Rows are filtered again in Python so cross-team keys never leak in.
    """
    import ipl_teams

    lab = ipl_teams.franchise_label_for_storage(canonical_label) or (canonical_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    sig = db_runtime_signature()
    return [dict(r) for r in _history_team_xi_rows_for_franchise_cached(sig, lab, ck, int(limit))]


@lru_cache(maxsize=48)
def _history_team_xi_rows_for_franchise_cached(
    _sig: tuple[str, int, int, int],
    lab: str,
    ck: str,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    import ipl_teams

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT x.match_id, x.team_name, x.player_name, x.team_key, x.player_key,
                   x.batting_position, x.is_keeper, x.overs_bowled, x.batting_order_index,
                   x.role_bucket,
                   m.match_date, m.venue, m.batting_first, m.team_a, m.team_b, m.created_at,
                   mc.overseas_team_a, mc.overseas_team_b
            FROM team_match_xi x
            JOIN matches m ON m.id = x.match_id
            LEFT JOIN match_context mc ON mc.match_id = m.id
            WHERE x.team_key = ?
            ORDER BY m.created_at DESC, m.id DESC, x.batting_order_index ASC
            LIMIT ?
            """,
            (ck, limit),
        ).fetchall()
    strict: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if not ipl_teams.franchise_row_matches_canonical(
            stored_team_name=str(d.get("team_name") or ""),
            stored_team_key=str(d.get("team_key") or ""),
            canonical_label=lab,
        ):
            continue
        strict.append(d)
    logger.info(
        "history_team_xi_rows_for_franchise label=%s key=%s raw_sql=%d strict=%d",
        lab,
        ck,
        len(rows),
        len(strict),
    )
    return tuple(strict)


def backfill_history_tables_from_results(*, limit: int = 400) -> int:
    """Populate `matches` / team_* for older rows that only had match_results. Returns rows processed."""
    n_done = 0
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.raw_payload, r.created_at
            FROM match_results r
            WHERE r.id NOT IN (SELECT id FROM matches)
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        for r in rows:
            try:
                payload = json.loads(r["raw_payload"])
            except json.JSONDecodeError:
                continue
            _sync_history_match_tables(conn, int(r["id"]), payload, float(r["created_at"]))
            n_done += 1
    logger.info("backfill_history_tables: processed %d matches", n_done)
    return n_done


def fetch_recent_matches(limit: int = 200) -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, url, source, team_a, team_b, venue, winner,
                   batting_first, raw_payload, created_at
            FROM match_results
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "url": r["url"],
                "source": r["source"],
                "team_a": r["team_a"],
                "team_b": r["team_b"],
                "venue": r["venue"],
                "winner": r["winner"],
                "batting_first": r["batting_first"],
                "raw_payload": r["raw_payload"],
                "created_at": r["created_at"],
            }
        )
    return out


def get_learned_players() -> dict[str, sqlite3.Row]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM learned_player").fetchall()
    return {r["player_key"]: r for r in rows}


def upsert_learned_players(rows: Iterable[tuple[str, dict[str, Any]]]) -> None:
    """rows: (player_key, fields)"""
    now = time.time()
    with connection() as conn:
        for key, f in rows:
            conn.execute(
                """
                INSERT INTO learned_player (
                    player_key, matches_in_db, xi_appearances,
                    batting_runs, batting_balls, wickets, balls_bowled, impact_ema, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_key) DO UPDATE SET
                    matches_in_db = excluded.matches_in_db,
                    xi_appearances = excluded.xi_appearances,
                    batting_runs = excluded.batting_runs,
                    batting_balls = excluded.batting_balls,
                    wickets = excluded.wickets,
                    balls_bowled = excluded.balls_bowled,
                    impact_ema = excluded.impact_ema,
                    last_updated = excluded.last_updated
                """,
                (
                    key,
                    int(f.get("matches_in_db", 0)),
                    int(f.get("xi_appearances", 0)),
                    int(f.get("batting_runs", 0)),
                    int(f.get("batting_balls", 0)),
                    int(f.get("wickets", 0)),
                    int(f.get("balls_bowled", 0)),
                    float(f.get("impact_ema", 0.5)),
                    now,
                ),
            )


def get_venue_team_stats() -> list[sqlite3.Row]:
    with connection() as conn:
        return conn.execute("SELECT * FROM learned_venue_team").fetchall()


def upsert_venue_team(
    venue_key: str,
    team_key: str,
    *,
    bat_first_win_delta: Optional[int] = None,
    bowl_first_win_delta: Optional[int] = None,
) -> None:
    """Increment match outcome stats for (venue, team). Exactly one delta should be 1 per call."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM learned_venue_team WHERE venue_key = ? AND team_key = ?",
            (venue_key, team_key),
        ).fetchone()
        m = int(row["matches"]) if row else 0
        wb = int(row["wins_bat_first"]) if row else 0
        wl = int(row["wins_bowl_first"]) if row else 0
        m += 1
        if bat_first_win_delta:
            wb += 1
        elif bowl_first_win_delta:
            wl += 1
        conn.execute(
            """
            INSERT INTO learned_venue_team (venue_key, team_key, matches, wins_bat_first, wins_bowl_first)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(venue_key, team_key) DO UPDATE SET
                matches = excluded.matches,
                wins_bat_first = excluded.wins_bat_first,
                wins_bowl_first = excluded.wins_bowl_first
            """,
            (venue_key, team_key, m, wb, wl),
        )


def count_stored_matches() -> int:
    with connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM match_results").fetchone()
    return int(row["c"]) if row else 0


def xi_pick_counts_raw() -> list[tuple[str, int]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT player_name, xi_count AS c FROM prediction_summary_player_xi_counts"
        ).fetchall()
    return [(str(r["player_name"]), int(r["c"])) for r in rows]


def max_xi_pick_count() -> int:
    with connection() as conn:
        row = conn.execute(
            "SELECT MAX(xi_count) AS m FROM prediction_summary_player_xi_counts"
        ).fetchone()
    return int(row["m"] or 0) if row else 0


def avg_batting_position_raw() -> list[tuple[str, float, int]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT player_name, avg_position AS av, sample_count AS n FROM prediction_summary_player_batting"
        ).fetchall()
    return [(str(r["player_name"]), float(r["av"]), int(r["n"])) for r in rows]


def bowling_usage_raw() -> list[tuple[str, float, int]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT player_name, avg_balls AS balls, match_count AS n FROM prediction_summary_player_bowling"
        ).fetchall()
    out: list[tuple[str, float, int]] = []
    for r in rows:
        balls = float(r["balls"] or 0.0)
        n = int(r["n"] or 0)
        out.append((str(r["player_name"]), balls, n))
    return out


def venue_team_xi_raw() -> list[tuple[str, str, str, int]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT venue, team, player_name, xi_count AS c
            FROM prediction_summary_venue_team_player_xi
            """
        ).fetchall()
    return [(str(r["venue"] or ""), str(r["team"] or ""), str(r["player_name"]), int(r["c"])) for r in rows]


def match_xi_team_venue_rows() -> list[tuple[int, str, str]]:
    """Distinct (match_id, venue, team) from XI rows."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT match_id, venue, team FROM prediction_summary_match_team_venue"
        ).fetchall()
    return [(int(r["match_id"]), str(r["venue"] or ""), str(r["team"] or "")) for r in rows]


def night_day_xi_raw() -> list[tuple[str, int, int]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT m.player_name, mc.is_night, COUNT(*) AS c
            FROM match_xi m
            JOIN match_context mc ON mc.match_id = m.match_id
            WHERE mc.is_night IS NOT NULL
            GROUP BY m.player_name, mc.is_night
            """
        ).fetchall()
    return [(str(r["player_name"]), int(r["is_night"]), int(r["c"])) for r in rows]


def learned_overseas_mix_raw() -> list[tuple[str, str, int, int]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT venue_key, team_key, n_overseas, tally FROM learned_overseas_mix"
        ).fetchall()
    return [(str(r["venue_key"]), str(r["team_key"]), int(r["n_overseas"]), int(r["tally"])) for r in rows]


def learned_venue_team_chase_rollup() -> list[tuple[str, int, int]]:
    """Per venue_key: sum bat-first wins, bowl-first (chase) wins."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT venue_key,
                   SUM(wins_bat_first) AS wb,
                   SUM(wins_bowl_first) AS wl
            FROM learned_venue_team
            GROUP BY venue_key
            """
        ).fetchall()
    return [(str(r["venue_key"]), int(r["wb"] or 0), int(r["wl"] or 0)) for r in rows]


def fetch_match_results_meta(limit: int = 400) -> list[dict[str, Any]]:
    """Recent matches for head-to-head, venue form, and chase/defend splits (needs batting_first)."""
    return [dict(r) for r in _fetch_match_results_meta_cached(db_runtime_signature(), int(limit))]


@lru_cache(maxsize=8)
def _fetch_match_results_meta_cached(
    _sig: tuple[str, int, int, int],
    limit: int,
) -> tuple[dict[str, Any], ...]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT winner, team_a, team_b, venue, batting_first, created_at, match_id AS id, match_date
            FROM prediction_summary_match_meta
            ORDER BY match_date DESC NULLS LAST, match_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "winner": str(r["winner"] or ""),
                "team_a": str(r["team_a"] or ""),
                "team_b": str(r["team_b"] or ""),
                "venue": str(r["venue"] or ""),
                "batting_first": str(r["batting_first"] or ""),
                "created_at": r["created_at"],
                "id": r["id"],
                "match_date": r["match_date"],
            }
        )
    return tuple(out)


def h2h_fixtures_between_franchises(
    label_a: str,
    label_b: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Recent fixtures between two franchises (order-independent), newest first.

    Restricted to the last ``CRICSHEET_HISTORY_SEASON_COUNT`` IPL seasons (by match year).

    Each dict: ``match_id``, ``match_date``, ``venue``, ``team_a``, ``team_b``.
    """
    import config as _cfg
    import h2h_history
    import ipl_teams

    ca = ipl_teams.canonical_franchise_label(label_a) or (label_a or "").strip()
    cb = ipl_teams.canonical_franchise_label(label_b) or (label_b or "").strip()
    cur_y = int(getattr(_cfg, "IPL_CURRENT_SEASON_YEAR", 2026))
    span = int(getattr(_cfg, "CRICSHEET_HISTORY_SEASON_COUNT", 5))
    min_year = cur_y - span + 1
    out: list[dict[str, Any]] = []
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, team_a, team_b, match_date, venue
            FROM matches
            ORDER BY match_date DESC NULLS LAST, id DESC
            LIMIT 900
            """,
        ).fetchall()
    for r in rows:
        if not h2h_history.rows_are_h2h(ca, cb, str(r["team_a"] or ""), str(r["team_b"] or "")):
            continue
        y = h2h_history.year_from_match_row(
            {
                "match_date": r["match_date"],
                "created_at": None,
            }
        )
        if y is not None and y < min_year:
            continue
        out.append(
            {
                "match_id": int(r["id"]),
                "match_date": r["match_date"],
                "venue": str(r["venue"] or ""),
                "team_a": str(r["team_a"] or ""),
                "team_b": str(r["team_b"] or ""),
            }
        )
        if len(out) >= int(limit):
            break
    return out


def h2h_match_ids_between_franchises(
    label_a: str,
    label_b: str,
    *,
    limit: int = 100,
) -> list[int]:
    return [int(x["match_id"]) for x in h2h_fixtures_between_franchises(label_a, label_b, limit=limit)]


def batch_team_match_xi_counts(player_keys: list[str], franchise_team_key: str) -> dict[str, int]:
    """Row counts in ``team_match_xi`` per ``player_key`` for this franchise ``team_key`` only."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key, COUNT(*) AS c
            FROM team_match_xi
            WHERE team_key = ? AND player_key IN ({qm})
            GROUP BY player_key
            """,
            [fk] + keys,
        ).fetchall()
    return {str(r["player_key"]): int(r["c"] or 0) for r in rows}


def batch_team_match_xi_latest_dates(
    player_keys: list[str], franchise_team_key: str
) -> dict[str, Optional[str]]:
    """Latest ``matches.match_date`` per squad player from ``team_match_xi`` (canonical ``team_key`` only)."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT x.player_key AS player_key, MAX(m.match_date) AS latest
            FROM team_match_xi x
            JOIN matches m ON m.id = x.match_id
            WHERE x.team_key = ? AND x.player_key IN ({qm})
            GROUP BY x.player_key
            """,
            [fk] + keys,
        ).fetchall()
    out: dict[str, Optional[str]] = {}
    for r in rows:
        pk = str(r["player_key"] or "")
        lv = r["latest"]
        out[pk] = str(lv) if lv is not None and str(lv).strip() else None
    return out


def batch_team_match_xi_h2h_counts(
    player_keys: list[str],
    franchise_team_key: str,
    h2h_match_ids: list[int],
) -> dict[str, int]:
    """``team_match_xi`` row counts restricted to H2H ``match_id``s."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    mids = [int(x) for x in h2h_match_ids if int(x) > 0][:200]
    if not fk or not keys or not mids:
        return {}
    qm_k = ",".join("?" * len(keys))
    qm_m = ",".join("?" * len(mids))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key, COUNT(*) AS c
            FROM team_match_xi
            WHERE team_key = ? AND player_key IN ({qm_k}) AND match_id IN ({qm_m})
            GROUP BY player_key
            """,
            [fk] + keys + mids,
        ).fetchall()
    return {str(r["player_key"]): int(r["c"] or 0) for r in rows}


def batch_player_batting_positions_counts(player_keys: list[str], franchise_team_key: str) -> dict[str, int]:
    """Row counts in ``player_batting_positions`` per ``player_key`` for this franchise."""
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key, COUNT(*) AS c
            FROM player_batting_positions
            WHERE team_key = ? AND player_key IN ({qm})
            GROUP BY player_key
            """,
            [fk] + keys,
        ).fetchall()
    return {str(r["player_key"]): int(r["c"] or 0) for r in rows}


def batch_global_team_match_xi_stats(player_keys: list[str]) -> dict[str, dict[str, int]]:
    """
    Per ``player_key``, all-franchise aggregates from ``team_match_xi`` (IPL-wide presence).
    """
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key,
                   tmx_rows,
                   distinct_matches,
                   distinct_teams
            FROM prediction_summary_player_global_xi
            WHERE player_key IN ({qm})
            """,
            keys,
        ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        out[pk] = {
            "tmx_rows": int(r["tmx_rows"] or 0),
            "distinct_matches": int(r["distinct_matches"] or 0),
            "distinct_teams": int(r["distinct_teams"] or 0),
        }
    return out


def batch_global_player_batting_slot_ema(player_keys: list[str]) -> dict[str, tuple[float, int]]:
    """Mean batting position across all stored rows (all franchises) per ``player_key``."""
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key,
                   slot_ema AS ema,
                   slot_samples AS n
            FROM prediction_summary_player_global_slot
            WHERE player_key IN ({qm})
            """,
            keys,
        ).fetchall()
    out: dict[str, tuple[float, int]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        try:
            ema = float(r["ema"] or 0.0)
        except (TypeError, ValueError):
            ema = 0.0
        out[pk] = (ema, int(r["n"] or 0))
    return out


def batch_player_other_franchise_tmx_counts(
    player_keys: list[str],
    exclude_team_key: str,
) -> dict[str, int]:
    """
    Count ``team_match_xi`` rows per ``player_key`` where ``team_key`` ≠ ``exclude_team_key``
    (IPL history on other franchises).
    """
    fk = (exclude_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        totals = conn.execute(
            f"""
            SELECT player_key, tmx_rows
            FROM prediction_summary_player_global_xi
            WHERE player_key IN ({qm})
            """,
            keys,
        ).fetchall()
        rows = conn.execute(
            f"""
            SELECT player_key, tmx_rows AS c
            FROM prediction_summary_player_team_xi
            WHERE player_key IN ({qm})
              AND team_key = ?
            """,
            keys + [fk],
        ).fetchall()
    own = {str(r["player_key"]): int(r["c"] or 0) for r in rows}
    out: dict[str, int] = {}
    for r in totals:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        all_n = int(r["tmx_rows"] or 0)
        out[pk] = max(0, all_n - int(own.get(pk, 0)))
    return out


def batch_global_player_profile_aggregates(player_keys: list[str]) -> dict[str, dict[str, float]]:
    """
    Best-effort maxima of derive-time ``player_profiles`` across all franchises per player.
    """
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key,
                   MAX(COALESCE(sample_matches, 0)) AS max_samples,
                   MAX(COALESCE(xi_selection_frequency, 0)) AS max_xi_freq,
                   MAX(COALESCE(profile_confidence, 0)) AS max_profile_confidence,
                   MAX(COALESCE(opener_likelihood, 0)) AS max_opener_likelihood,
                   MAX(COALESCE(recent_usage_score, 0)) AS max_recent_usage
            FROM player_profiles
            WHERE player_key IN ({qm})
            GROUP BY player_key
            """,
            keys,
        ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        out[pk] = {
            "max_samples": float(r["max_samples"] or 0),
            "max_xi_freq": float(r["max_xi_freq"] or 0),
            "max_profile_confidence": float(r["max_profile_confidence"] or 0),
            "max_opener_likelihood": float(r["max_opener_likelihood"] or 0),
            "max_recent_usage": float(r["max_recent_usage"] or 0),
        }
    return out


def batch_player_profiles_for_franchise(
    player_keys: list[str],
    franchise_team_key: str,
) -> dict[str, dict[str, Any]]:
    """
    Stage-2 ``player_profiles`` rows for (player_key, franchise_team_key).

    Used at prediction time only — reads SQLite derive output, not raw Cricsheet JSON.
    """
    fk = (franchise_team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    cols = (
        "player_key, franchise_team_key, xi_selection_frequency, batting_position_ema, "
        "opener_likelihood, middle_order_likelihood, finisher_likelihood, "
        "powerplay_bowler_likelihood, middle_overs_bowler_likelihood, death_bowler_likelihood, "
        "venue_fit_score, role_stability_score, recent_usage_score, profile_confidence, sample_matches"
    )
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {cols}
            FROM player_profiles
            WHERE franchise_team_key = ? AND player_key IN ({qm})
            """,
            [fk] + keys,
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        out[pk] = dict(r)
    return out


def fetch_team_selection_pattern(
    team_key: str,
    venue_key_candidates: list[str],
) -> Optional[dict[str, Any]]:
    """First matching ``team_selection_patterns`` row across venue key candidates."""
    tk = (team_key or "").strip()[:80]
    if not tk:
        return None
    seen: set[str] = set()
    ordered: list[str] = []
    for vk in venue_key_candidates:
        s = str(vk or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    with connection() as conn:
        for vk in ordered:
            v = (vk or "").strip()[:80]
            if not v:
                continue
            row = conn.execute(
                """
                SELECT * FROM team_selection_patterns
                WHERE team_key = ? AND venue_key = ?
                LIMIT 1
                """,
                (tk, v),
            ).fetchone()
            if row and int(row["sample_matches"] or 0) > 0:
                return dict(row)
    return None


def team_selection_pattern_join_explain(
    team_key: str,
    venue_key_candidates: list[str],
) -> dict[str, Any]:
    """
    Debug-only: why ``fetch_team_selection_pattern`` may miss, and which venue keys exist in SQLite.
    """
    tk = (team_key or "").strip()[:80]
    seen: set[str] = set()
    tried: list[str] = []
    for vk in venue_key_candidates:
        s = str(vk or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            tried.append(s)
    out: dict[str, Any] = {
        "team_key": tk,
        "venue_pattern_keys_tried": tried,
        "matched_venue_key": None,
        "pattern_rows_for_team": 0,
        "distinct_venue_keys_in_db_for_team": [],
        "miss_reason": "",
    }
    if not tk:
        out["miss_reason"] = "empty_team_key"
        return out
    if not tried:
        out["miss_reason"] = "no_venue_key_candidates"
        return out
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM team_selection_patterns WHERE team_key = ?",
            (tk,),
        ).fetchone()
        out["pattern_rows_for_team"] = int(row["c"] if row is not None else 0)
        vrows = conn.execute(
            """
            SELECT DISTINCT venue_key FROM team_selection_patterns
            WHERE team_key = ? ORDER BY venue_key LIMIT 25
            """,
            (tk,),
        ).fetchall()
        out["distinct_venue_keys_in_db_for_team"] = [
            str(r["venue_key"] or "") for r in vrows if str(r["venue_key"] or "").strip()
        ]
    hit = fetch_team_selection_pattern(tk, tried)
    if hit:
        out["matched_venue_key"] = str(hit.get("venue_key") or "").strip() or None
        out["miss_reason"] = ""
        return out
    if out["pattern_rows_for_team"] == 0:
        out["miss_reason"] = (
            "no team_selection_patterns rows for this franchise team_key "
            "(run Stage 2 derive for team_selection_patterns)."
        )
    else:
        out["miss_reason"] = (
            "franchise has pattern rows but none of the tried venue keys matched "
            "(Stage 3 uses canonical/hashed venue keys from derive — compare "
            "venue_pattern_keys_tried to distinct_venue_keys_in_db_for_team)."
        )
    return out


def _matches_table_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(matches)").fetchall()}


def match_row_is_t20_family(competition: Optional[str], match_format: Optional[str]) -> bool:
    """
    True if a row should count toward **T20-family recent form** for IPL prediction.

    Prefer explicit ``match_format`` when present (from ingest metadata); otherwise infer from
    ``competition`` using config substring lists.
    """
    mf = (str(match_format).strip().lower() if match_format is not None else "") or ""
    if mf in ("t20", "t20i", "it20"):
        return True
    if mf in ("test", "odi", "oda", "list a", "first-class", "fc"):
        return False
    comp = (competition or "").strip().lower()
    if not comp:
        return False
    if any(s in comp for s in config.T20_EXCLUDE_COMPETITION_SUBSTRINGS):
        if not any(s in comp for s in config.T20_FAMILY_COMPETITION_SUBSTRINGS):
            return False
    return any(s in comp for s in config.T20_FAMILY_COMPETITION_SUBSTRINGS)


def fetch_recent_pms_rows_for_squad_players(
    franchise_team_key: str,
    player_keys: list[str],
    *,
    max_rows: int = 8000,
) -> list[dict[str, Any]]:
    """
    Recent ``player_match_stats`` joined to ``matches`` for a franchise squad slice.

    Rows are returned **newest first** (by ``match_date`` / id). Caller filters T20-family
    via :func:`match_row_is_t20_family`.
    """
    fk = (franchise_team_key or "").strip()[:80]
    keys = [str(k).strip()[:80] for k in player_keys if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not fk or not keys:
        return []
    qm = ",".join(["?"] * len(keys))
    with connection() as conn:
        cols = _matches_table_columns(conn)
        mf_expr = "m.match_format" if "match_format" in cols else "NULL AS match_format"
        rows = conn.execute(
            f"""
            SELECT s.player_key AS player_key,
                   s.match_id AS match_id,
                   m.match_date AS match_date,
                   m.competition AS competition,
                   {mf_expr},
                   s.runs AS runs, s.balls AS balls, s.fours AS fours, s.sixes AS sixes,
                   s.strike_rate AS strike_rate,
                   s.overs_bowled AS overs_bowled, s.wickets AS wickets,
                   s.runs_conceded AS runs_conceded, s.economy AS economy
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.team_key = ? AND s.player_key IN ({qm})
              AND m.match_date IS NOT NULL AND trim(m.match_date) != ''
            ORDER BY m.match_date DESC, s.match_id DESC
            LIMIT ?
            """,
            (fk, *keys, int(max_rows)),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_last_team_match_player_signals(
    franchise_team_key: str,
    player_keys: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Latest stored match signals for a franchise squad slice.

    Returns per ``player_key`` signals from the most recent ``team_match_xi`` match:
    - was in last XI
    - batting position / overs bowled / keeper flag
    - basic batting and bowling contribution fields (if summary rows exist)
    """
    fk = (franchise_team_key or "").strip()[:80]
    keys = [str(k).strip()[:80] for k in player_keys if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not fk or not keys:
        return {}
    qm = ",".join(["?"] * len(keys))
    with connection() as conn:
        latest = conn.execute(
            """
            SELECT x.match_id AS match_id, m.match_date AS match_date
            FROM team_match_xi x
            JOIN matches m ON m.id = x.match_id
            WHERE x.team_key = ?
              AND m.match_date IS NOT NULL
              AND trim(m.match_date) != ''
            ORDER BY m.match_date DESC, x.match_id DESC
            LIMIT 1
            """,
            (fk,),
        ).fetchone()
        if latest is None:
            return {}
        match_id = int(latest["match_id"])
        match_date = str(latest["match_date"] or "")
        rows = conn.execute(
            f"""
            SELECT x.player_key AS player_key,
                   x.selected_in_xi AS selected_in_xi,
                   x.batting_position AS batting_position,
                   x.overs_bowled AS overs_bowled,
                   x.is_keeper AS is_keeper,
                   b.runs AS bat_runs,
                   b.balls AS bat_balls,
                   b.strike_rate AS bat_strike_rate,
                   bw.wickets AS bowl_wickets,
                   bw.runs_conceded AS bowl_runs_conceded,
                   bw.economy AS bowl_economy
            FROM team_match_xi x
            LEFT JOIN player_match_batting_summary b
              ON b.match_id = x.match_id
             AND b.team_key = x.team_key
             AND b.player_key = x.player_key
            LEFT JOIN player_match_bowling_summary bw
              ON bw.match_id = x.match_id
             AND bw.team_key = x.team_key
             AND bw.player_key = x.player_key
            WHERE x.team_key = ?
              AND x.match_id = ?
              AND x.player_key IN ({qm})
            """,
            (fk, match_id, *keys),
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        pk = str(r["player_key"] or "").strip()
        if not pk:
            continue
        out[pk] = {
            "last_match_id": match_id,
            "last_match_date": match_date,
            "was_in_last_match_xi": bool(int(r["selected_in_xi"] or 0) > 0),
            "last_match_batting_position": r["batting_position"],
            "last_match_overs_bowled": r["overs_bowled"],
            "last_match_is_keeper": bool(int(r["is_keeper"] or 0) > 0),
            "last_match_batting_runs": int(r["bat_runs"] or 0),
            "last_match_batting_balls": int(r["bat_balls"] or 0),
            "last_match_batting_strike_rate": float(r["bat_strike_rate"] or 0.0),
            "last_match_bowling_wickets": int(r["bowl_wickets"] or 0),
            "last_match_bowling_runs_conceded": int(r["bowl_runs_conceded"] or 0),
            "last_match_bowling_economy": float(r["bowl_economy"] or 0.0),
        }
    return out


def fetch_player_recent_form_cache_batch(player_keys: list[str]) -> dict[str, dict[str, Any]]:
    """
    Rows from ``player_recent_form_cache`` for the given global ``player_key`` values.

    Used at prediction time so recent-form scoring does not scan ``player_match_stats``.
    """
    keys = [str(k).strip()[:80] for k in player_keys if str(k).strip()]
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        chk = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_recent_form_cache'"
        ).fetchone()
        if not chk:
            return {}
        rows = conn.execute(
            f"SELECT * FROM player_recent_form_cache WHERE player_key IN ({qm})",
            keys,
        ).fetchall()
    return {str(r["player_key"]): dict(r) for r in rows}


def fetch_head_to_head_derived(team_key_a: str, team_key_b: str) -> Optional[dict[str, Any]]:
    """
    Row from ``head_to_head_patterns`` (team keys sorted lexicographically, same as Stage 2 derive).
    """
    a = (team_key_a or "").strip()[:80]
    b = (team_key_b or "").strip()[:80]
    if not a or not b or a == b:
        return None
    k1, k2 = (a, b) if a < b else (b, a)
    with connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM head_to_head_patterns
            WHERE team_a_key = ? AND team_b_key = ?
            LIMIT 1
            """,
            (k1, k2),
        ).fetchone()
    return dict(row) if row else None


def batch_player_match_stats_counts(player_keys: list[str], team_key: str) -> dict[str, int]:
    """Row counts in ``player_match_stats`` per player_key for this franchise team_key."""
    fk = (team_key or "").strip()[:80]
    keys = [(k or "").strip()[:80] for k in player_keys if (k or "").strip()]
    if not fk or not keys:
        return {}
    qm = ",".join("?" * len(keys))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT player_key, COUNT(*) AS c
            FROM player_match_stats
            WHERE team_key = ? AND player_key IN ({qm})
            GROUP BY player_key
            """,
            [fk] + keys,
        ).fetchall()
    return {str(r["player_key"]): int(r["c"] or 0) for r in rows}


def sample_stored_player_name_for_key(franchise_team_key: str, player_key: str) -> Optional[str]:
    """
    Latest raw ``player_name`` seen in history tables for this franchise + ``player_key``.

    For debug only — **never** use this as the squad display name.
    """
    fk = (franchise_team_key or "").strip()[:80]
    pk = (player_key or "").strip()[:80]
    if not fk or not pk:
        return None
    with connection() as conn:
        row = conn.execute(
            """
            SELECT player_name FROM team_match_xi
            WHERE team_key = ? AND player_key = ?
            ORDER BY id DESC LIMIT 1
            """,
            (fk, pk),
        ).fetchone()
        if row and str(row["player_name"] or "").strip():
            return str(row["player_name"]).strip()
        row = conn.execute(
            """
            SELECT player_name FROM player_match_stats
            WHERE team_key = ? AND player_key = ?
            ORDER BY id DESC LIMIT 1
            """,
            (fk, pk),
        ).fetchone()
        if row and str(row["player_name"] or "").strip():
            return str(row["player_name"]).strip()
    return None


def fetch_team_derived_summary(team_key: str) -> Optional[dict[str, Any]]:
    """Single-row Stage-2 team summary for substitution / tactical heuristics (may be missing)."""
    fk = (team_key or "").strip()[:80]
    if not fk:
        return None
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM team_derived_summary WHERE team_key = ? LIMIT 1",
            (fk,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def franchise_distinct_history_player_keys(franchise_team_key: str) -> frozenset[str]:
    """
    Distinct ``player_key`` values for one franchise across raw history tables
    (Stage 1 linkage / alias resolver index).
    """
    fk = (franchise_team_key or "").strip()[:80]
    if not fk:
        return frozenset()
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT player_key FROM team_match_xi WHERE team_key = ?
            UNION
            SELECT player_key FROM player_match_stats WHERE team_key = ?
            UNION
            SELECT player_key FROM player_batting_positions WHERE team_key = ?
            """,
            (fk, fk, fk),
        ).fetchall()
    return frozenset(str(r["player_key"]).strip() for r in rows if str(r["player_key"]).strip())


def global_distinct_history_player_keys() -> frozenset[str]:
    """
    Distinct ``player_key`` across all franchises in raw history tables (IPL-wide alias index).

    Used when franchise-scoped resolution returns ``no_match`` to resolve Cricsheet-style keys
    (e.g. ``sv samson``, ``ra jadeja``) from full display names.
    """
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT player_key FROM team_match_xi
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
            UNION
            SELECT player_key FROM player_match_stats
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
            UNION
            SELECT player_key FROM player_batting_positions
            WHERE player_key IS NOT NULL AND trim(player_key) != ''
            """
        ).fetchall()
    return frozenset(str(r["player_key"]).strip() for r in rows if str(r["player_key"]).strip())


def recent_franchise_history_player_keys(
    franchise_team_key: str,
    *,
    limit_matches: int = 18,
) -> frozenset[str]:
    """
    Distinct player keys seen in the most recent stored matches for a franchise.

    Used as supporting evidence for initials-style linkage when franchise-wide alias matching is weak.
    """
    fk = (franchise_team_key or "").strip()[:80]
    lim = max(3, min(80, int(limit_matches)))
    if not fk:
        return frozenset()
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT x.player_key
            FROM team_match_xi x
            JOIN (
                SELECT DISTINCT t.match_id
                FROM team_match_xi t
                JOIN matches m ON m.id = t.match_id
                WHERE t.team_key = ?
                ORDER BY m.match_date DESC NULLS LAST, t.match_id DESC
                LIMIT ?
            ) recent ON recent.match_id = x.match_id
            WHERE x.team_key = ?
              AND x.player_key IS NOT NULL
              AND trim(x.player_key) != ''
            """,
            (fk, lim, fk),
        ).fetchall()
    return frozenset(str(r["player_key"]).strip() for r in rows if str(r["player_key"]).strip())


def sqlite_matches_temporal_audit(*, cricsheet_derived_only: bool = True) -> dict[str, Any]:
    """
    Calendar coverage in ``matches`` for ingest/debug (earliest/latest ISO-ish dates, distinct years).

    ``cricsheet_derived_only`` limits rows to those with ``cricsheet_match_id`` or source mentioning cricsheet.
    """
    with connection() as conn:
        where = ""
        if cricsheet_derived_only:
            where = """WHERE (
                (cricsheet_match_id IS NOT NULL AND trim(cricsheet_match_id) != '')
                OR lower(coalesce(source, '')) LIKE '%cricsheet%'
                OR lower(coalesce(scorecard_url, '')) LIKE '%cricsheet%'
            )"""
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM matches {where}",
        ).fetchone()
        n = int(row["c"] or 0) if row else 0
        dates = conn.execute(
            f"""
            SELECT match_date FROM matches
            {where}
              AND match_date IS NOT NULL
              AND trim(match_date) != ''
              AND length(trim(match_date)) >= 10
              AND substr(trim(match_date), 5, 1) = '-'
              AND substr(trim(match_date), 8, 1) = '-'
            """,
        ).fetchall()
    iso_dates: list[str] = []
    years: set[int] = set()
    for r in dates:
        md = str(r["match_date"] or "").strip()[:10]
        if len(md) >= 10 and md[4] == "-" and md[7] == "-":
            try:
                y = int(md[:4])
                if 1990 <= y <= 2100:
                    iso_dates.append(md)
                    years.add(y)
            except ValueError:
                continue
    iso_dates.sort()
    earliest = iso_dates[0] if iso_dates else None
    latest = iso_dates[-1] if iso_dates else None
    year_span = (max(years) - min(years) + 1) if years else 0
    # Heuristic: full archive if we see pre-2020 IPL history or very wide span.
    likely_full_archive = bool(years and (min(years) < 2020 or year_span >= 12))
    mode = "likely_full_archive" if likely_full_archive else "likely_sliding_recent_window"
    return {
        "matches_rows_considered": n,
        "iso_parseable_match_dates": len(iso_dates),
        "earliest_match_date": earliest,
        "latest_match_date": latest,
        "distinct_years": sorted(years),
        "distinct_year_count": len(years),
        "year_span_inclusive": year_span,
        "ingest_coverage_hint": mode,
        "cricsheet_derived_only": bool(cricsheet_derived_only),
    }


def upsert_player_alias_resolution(
    franchise_team_key: str,
    squad_full_name: str,
    normalized_full_name_key: str,
    resolved_history_key: Optional[str],
    resolution_type: str,
    confidence: Optional[float],
    ambiguous_candidates_json: Optional[str],
) -> None:
    """Cache last squad→history resolution for debugging (optional; safe to call frequently)."""
    fk = (franchise_team_key or "").strip()[:80]
    nk = (normalized_full_name_key or "").strip()[:80]
    if not fk or not nk:
        return
    now = time.time()
    rk = (resolved_history_key or "").strip()[:80] or None
    rt = (resolution_type or "no_match").strip() or "no_match"
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO player_aliases (
                franchise_team_key, squad_full_name, normalized_full_name_key,
                resolved_history_key, resolution_type, confidence,
                ambiguous_candidates_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(franchise_team_key, normalized_full_name_key) DO UPDATE SET
                squad_full_name = excluded.squad_full_name,
                resolved_history_key = excluded.resolved_history_key,
                resolution_type = excluded.resolution_type,
                confidence = excluded.confidence,
                ambiguous_candidates_json = excluded.ambiguous_candidates_json,
                updated_at = excluded.updated_at
            """,
            (
                fk,
                (squad_full_name or "").strip(),
                nk,
                rk,
                rt,
                float(confidence) if confidence is not None else None,
                ambiguous_candidates_json,
                now,
            ),
        )


def player_phase_bowl_rates_in_match_ids(
    player_key: str,
    franchise_team_key: str,
    match_ids: list[int],
) -> dict[str, float]:
    """Fraction of the given matches where the player bowled in each phase (any legal ball)."""
    pk = (player_key or "").strip()[:80]
    fk = (franchise_team_key or "").strip()[:80]
    if not pk or not fk or not match_ids:
        return {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
    mids = [int(x) for x in match_ids[:120]]
    qm = ",".join("?" * len(mids))
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT phase, COUNT(DISTINCT match_id) AS c
            FROM player_phase_usage
            WHERE player_key = ? AND team_key = ? AND role = 'bowl'
              AND balls > 0 AND match_id IN ({qm})
            GROUP BY phase
            """,
            [pk, fk] + mids,
        ).fetchall()
    by_ph = {str(r["phase"]): int(r["c"]) for r in rows}
    n = float(len(set(mids)))
    return {
        "powerplay": by_ph.get("powerplay", 0) / n,
        "middle": by_ph.get("middle", 0) / n,
        "death": by_ph.get("death", 0) / n,
    }


def bump_overseas_mix(venue_key: str, team_key: str, n_overseas: int, delta: int = 1) -> None:
    if n_overseas < 0 or n_overseas > 11:
        return
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO learned_overseas_mix (venue_key, team_key, n_overseas, tally)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(venue_key, team_key, n_overseas) DO UPDATE SET
                tally = learned_overseas_mix.tally + excluded.tally
            """,
            (venue_key, team_key, n_overseas, delta),
        )
