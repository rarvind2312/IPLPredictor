"""
Stage 1 — audit reports: raw SQLite population, Cricsheet readme vs DB, batting slots, canonical keys.

Reads SQLite and local readme/JSON listing only (no prediction). Use from Streamlit admin expanders.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import config
import cricsheet_ingest
import cricsheet_readme
import db
import history_linkage
import ipl_teams
import predictor


def _scalar(conn: sqlite3.Connection, q: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(q, params).fetchone()
    if not row:
        return 0
    return int(row[0] or 0)


def raw_history_table_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Global row counts for Stage 1 raw tables."""
    return {
        "matches": _scalar(conn, "SELECT COUNT(*) FROM matches"),
        "team_match_summary": _scalar(conn, "SELECT COUNT(*) FROM team_match_summary"),
        "team_match_xi": _scalar(conn, "SELECT COUNT(*) FROM team_match_xi"),
        "player_match_stats": _scalar(conn, "SELECT COUNT(*) FROM player_match_stats"),
        "player_batting_positions": _scalar(conn, "SELECT COUNT(*) FROM player_batting_positions"),
        "player_phase_usage": _scalar(conn, "SELECT COUNT(*) FROM player_phase_usage"),
        "match_results_cricsheet_urls": _scalar(
            conn, "SELECT COUNT(*) FROM match_results WHERE url LIKE 'cricsheet://ipl/%'"
        ),
    }


def raw_counts_by_official_franchise_labels(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """
    Per official IPL label (from ``ipl_teams``): counts in ``team_match_xi`` / ``player_batting_positions``.

    Join key is ``team_name`` / ``franchise_label_for_storage`` alignment used at ingest.
    """
    out: dict[str, dict[str, int]] = {}
    for _slug, label in ipl_teams.IPL_TEAMS:
        tmx = _scalar(
            conn,
            "SELECT COUNT(*) FROM team_match_xi WHERE team_name = ?",
            (label,),
        )
        pbp = _scalar(
            conn,
            "SELECT COUNT(*) FROM player_batting_positions WHERE team_name = ?",
            (label,),
        )
        pms = _scalar(
            conn,
            "SELECT COUNT(*) FROM player_match_stats WHERE team_name = ?",
            (label,),
        )
        ppu = _scalar(
            conn,
            "SELECT COUNT(*) FROM player_phase_usage WHERE team_name = ?",
            (label,),
        )
        out[label] = {
            "team_match_xi_rows": tmx,
            "player_batting_positions_rows": pbp,
            "player_match_stats_rows": pms,
            "player_phase_usage_rows": ppu,
        }
    return out


def _readme_window_rows(
    *,
    readme_path: Optional[Path] = None,
    current_season_year: Optional[int] = None,
    n_seasons: Optional[int] = None,
) -> list[cricsheet_readme.CricsheetReadmeRow]:
    cur = int(current_season_year or getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    n = int(n_seasons or getattr(config, "CRICSHEET_HISTORY_SEASON_COUNT", 5))
    rows = cricsheet_readme.load_readme_rows(readme_path)
    return cricsheet_readme.filter_last_n_seasons(rows, current_season_year=cur, n_seasons=n)


def cricsheet_readme_vs_sqlite_report(
    *,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    current_season_year: Optional[int] = None,
    n_seasons: Optional[int] = None,
) -> dict[str, Any]:
    """
    Compare readme IPL rows (last N seasons) to JSON on disk and Cricsheet rows in SQLite.

    Returns expected ids, inserted ids, missing, extras, duplicate URL tails, missing JSON files.
    """
    jdir = Path(json_dir or config.CRICSHEET_JSON_DIR)
    window = _readme_window_rows(
        readme_path=readme_path,
        current_season_year=current_season_year,
        n_seasons=n_seasons,
    )
    expected_ids = sorted({r.match_id for r in window}, key=lambda x: int(x))
    on_disk = set(cricsheet_ingest.discover_json_match_ids(jdir)) if jdir.is_dir() else set()

    with db.connection() as conn:
        in_db = set()
        for r in conn.execute(
            "SELECT url FROM match_results WHERE url LIKE 'cricsheet://ipl/%'"
        ).fetchall():
            u = str(r[0] or "").strip().rstrip("/")
            if u.startswith("cricsheet://ipl/"):
                tail = u.split("/")[-1]
                if tail.isdigit():
                    in_db.add(tail)
        try:
            for r in conn.execute(
                "SELECT DISTINCT cricsheet_match_id FROM matches WHERE cricsheet_match_id IS NOT NULL AND trim(cricsheet_match_id) != ''"
            ).fetchall():
                mid = str(r[0] or "").strip()
                if mid.isdigit():
                    in_db.add(mid)
        except sqlite3.OperationalError:
            pass

        tail_counts: Counter[str] = Counter()
        for r in conn.execute(
            "SELECT url FROM match_results WHERE url LIKE 'cricsheet://ipl/%'"
        ).fetchall():
            u = str(r[0] or "").strip().rstrip("/")
            if u.startswith("cricsheet://ipl/"):
                tail = u.split("/")[-1]
                if tail.isdigit():
                    tail_counts[tail] += 1
        duplicates = [
            {"cricsheet_match_id": mid, "row_count": c} for mid, c in tail_counts.items() if c > 1
        ]

    exp_set = set(expected_ids)
    missing_in_sqlite = sorted(exp_set - in_db, key=int)
    missing_json_on_disk = sorted(exp_set - on_disk, key=int)
    extra_in_sqlite_not_in_readme_window = sorted(in_db - exp_set, key=int)

    per_team: dict[str, Any] = {}
    for _slug, label in ipl_teams.IPL_TEAMS:
        sub = [r for r in window if cricsheet_readme.row_involves_team_name(r, label, canonical=True)]
        te = {r.match_id for r in sub}
        per_team[label] = {
            "readme_matches_in_window": len(te),
            "in_sqlite": len(te & in_db),
            "missing_from_sqlite": sorted(te - in_db, key=int)[:80],
            "missing_json": sorted(te - on_disk, key=int)[:80],
        }

    return {
        "window_season_years": sorted(
            cricsheet_readme.season_years_window(
                int(current_season_year or config.IPL_CURRENT_SEASON_YEAR),
                int(n_seasons or config.CRICSHEET_HISTORY_SEASON_COUNT),
            )
        ),
        "readme_rows_in_window": len(window),
        "expected_distinct_match_ids": len(expected_ids),
        "json_files_numeric_on_disk": len(on_disk),
        "cricsheet_distinct_ids_in_sqlite": len(in_db),
        "missing_match_ids_readme_expected_but_not_sqlite": missing_in_sqlite[:500],
        "missing_match_ids_count_truncated": max(0, len(missing_in_sqlite) - 500),
        "readme_expected_but_json_missing_on_disk": missing_json_on_disk[:200],
        "sqlite_cricsheet_ids_outside_readme_window": extra_in_sqlite_not_in_readme_window[:200],
        "duplicate_cricsheet_ids_in_match_results": duplicates,
        "per_official_franchise": per_team,
    }


def batting_position_ingest_sample(
    conn: sqlite3.Connection,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Recent Cricsheet matches: teams, batting_order JSON per side, ``player_batting_positions`` row counts."""
    rows = conn.execute(
        """
        SELECT m.id AS match_id,
               m.cricsheet_match_id,
               m.team_a,
               m.team_b,
               m.match_date
        FROM matches m
        WHERE m.cricsheet_match_id IS NOT NULL AND trim(m.cricsheet_match_id) != ''
        ORDER BY m.match_date DESC NULLS LAST, m.id DESC
        LIMIT ?
        """,
        (max(1, min(50, int(limit))),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        mid = int(r["match_id"])
        pbp_n = _scalar(conn, "SELECT COUNT(*) FROM player_batting_positions WHERE match_id = ?", (mid,))
        tmx_n = _scalar(conn, "SELECT COUNT(*) FROM team_match_xi WHERE match_id = ?", (mid,))
        summaries = conn.execute(
            """
            SELECT team_name, batting_order_json, playing_xi_json
            FROM team_match_summary
            WHERE match_id = ?
            """,
            (mid,),
        ).fetchall()
        sides: list[dict[str, Any]] = []
        for s in summaries:
            bo_raw = s["batting_order_json"]
            try:
                bo = json.loads(bo_raw or "[]")
            except json.JSONDecodeError:
                bo = []
            sides.append(
                {
                    "team_name": str(s["team_name"] or ""),
                    "batting_order_from_deliveries": bo if isinstance(bo, list) else [],
                    "batting_order_len": len(bo) if isinstance(bo, list) else 0,
                }
            )
        out.append(
            {
                "sqlite_match_id": mid,
                "cricsheet_match_id": str(r["cricsheet_match_id"] or ""),
                "team_a": str(r["team_a"] or ""),
                "team_b": str(r["team_b"] or ""),
                "match_date": str(r["match_date"] or ""),
                "team_match_xi_rows": tmx_n,
                "player_batting_positions_rows": pbp_n,
                "per_team_summary": sides,
            }
        )
    return out


def canonical_key_consistency(conn: sqlite3.Connection) -> dict[str, Any]:
    """Rows where ``canonical_*`` diverges from ``team_key`` / ``player_key`` (should be zero)."""
    tables = ("team_match_xi", "player_match_stats", "player_batting_positions", "player_phase_usage")
    result: dict[str, Any] = {}
    for t in tables:
        try:
            n = _scalar(
                conn,
                f"""
                SELECT COUNT(*) FROM {t}
                WHERE team_key != canonical_team_key OR player_key != canonical_player_key
                """,
            )
            empty_t = _scalar(
                conn,
                f"""
                SELECT COUNT(*) FROM {t}
                WHERE trim(canonical_team_key) = '' OR trim(canonical_player_key) = ''
                """,
            )
            result[t] = {"mismatch_team_or_player_key": n, "empty_canonical_columns": empty_t}
        except sqlite3.OperationalError as e:
            result[t] = {"error": str(e)}
    return result


def canonical_key_sample_rows(conn: sqlite3.Connection, *, per_table: int = 4) -> dict[str, list[dict[str, Any]]]:
    """Sample raw rows showing ``team_key`` / ``player_key`` vs ``canonical_*``."""
    lim = max(1, min(20, int(per_table)))
    out: dict[str, list[dict[str, Any]]] = {}
    for t in ("team_match_xi", "player_match_stats", "player_batting_positions"):
        try:
            rows = conn.execute(
                f"""
                SELECT team_name, player_name, team_key, player_key,
                       canonical_team_key, canonical_player_key, match_id
                FROM {t}
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            out[t] = [dict(row) for row in rows]
        except sqlite3.OperationalError as e:
            out[t] = [{"error": str(e)}]
    return out


def squad_raw_history_linkage_for_team(
    squad_text: str,
    franchise_display_label: str,
    *,
    opponent_label: Optional[str] = None,
) -> dict[str, Any]:
    """
    Per-player raw-table counts + latest match date (same keys as ``history_linkage``, SQLite only).
    """
    players = predictor.parse_squad_text(squad_text or "")
    return history_linkage.link_current_squad_to_history(
        players,
        franchise_display_label,
        opponent_canonical_label=opponent_label,
    )


def full_stage1_audit_bundle(
    *,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Single dict for UI JSON dump (global counts + coverage + samples)."""
    with db.connection() as conn:
        counts = raw_history_table_counts(conn)
        by_team = raw_counts_by_official_franchise_labels(conn)
        bat = batting_position_ingest_sample(conn, limit=8)
        canon_c = canonical_key_consistency(conn)
        canon_s = canonical_key_sample_rows(conn, per_table=4)
    coverage = cricsheet_readme_vs_sqlite_report(json_dir=json_dir, readme_path=readme_path)
    return {
        "raw_table_counts": counts,
        "raw_counts_by_official_franchise": by_team,
        "cricsheet_coverage": coverage,
        "batting_position_recent_sample": bat,
        "canonical_key_consistency": canon_c,
        "canonical_key_samples": canon_s,
    }


__all__ = [
    "batting_position_ingest_sample",
    "canonical_key_consistency",
    "canonical_key_sample_rows",
    "cricsheet_readme_vs_sqlite_report",
    "full_stage1_audit_bundle",
    "raw_counts_by_official_franchise_labels",
    "raw_history_table_counts",
    "squad_raw_history_linkage_for_team",
]
