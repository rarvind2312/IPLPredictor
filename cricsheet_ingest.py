"""
Stage 1 — **INGEST**: local Cricsheet IPL bundle (``data/readme.txt`` + ``data/ipl_json/*.json``) → SQLite.

Only this stage reads raw Cricsheet JSON. Use ``run_initial_cricsheet_backfill`` or
``run_sync_new_cricsheet_matches`` from the UI or tooling.

Bulk ingest skips ``learner.ingest_payload`` and ``player_franchise_features`` refresh;
those belong to later derive / predict wiring.

Team-scoped ``ingest_local_history_for_teams`` remains for optional filtered loads but
uses the same DB path and **does not** run learning aggregates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import config
import cricsheet_convert
import cricsheet_readme
import db
import ipl_teams

logger = logging.getLogger(__name__)


@dataclass
class CricsheetIngestSummary:
    """Counters for one ingest run (matches + row inserts from payloads)."""

    matches_inserted: int = 0
    matches_skipped_duplicate: int = 0
    matches_skipped_malformed: int = 0
    readme_rows_total: int = 0
    readme_rows_missing_json: int = 0
    json_files_on_disk: int = 0
    player_stats_rows_inserted: int = 0
    batting_position_rows_inserted: int = 0
    phase_rows_inserted: int = 0
    warnings: list[str] = field(default_factory=list)


def apply_readme_row_to_payload(
    payload: dict[str, Any],
    row: Optional[cricsheet_readme.CricsheetReadmeRow],
) -> None:
    """Attach readme index metadata (and optional competition hint) before DB insert."""
    if row is None:
        return
    meta = payload.setdefault("meta", {})
    meta["readme_index"] = row.as_dict()
    if row.competition:
        meta.setdefault("competition", str(row.competition))


def _count_player_stats_rows(payload: dict[str, Any]) -> int:
    return len(payload.get("player_stats_extended") or [])


def _count_batting_position_rows(payload: dict[str, Any]) -> int:
    n = 0
    for block in payload.get("innings_batting_orders") or []:
        n += len(block.get("order") or [])
    return n


def _count_phase_rows(payload: dict[str, Any]) -> int:
    return len(payload.get("player_phase_extended") or [])


def discover_json_match_ids(json_dir: Path) -> list[str]:
    """Sorted numeric stems of ``*.json`` under ``json_dir``."""
    jdir = Path(json_dir)
    ids: list[str] = []
    for p in jdir.glob("*.json"):
        stem = p.stem.strip()
        if stem.isdigit():
            ids.append(stem)
    return sorted(set(ids), key=lambda x: int(x))


def _ingest_one_file(
    path: Path,
    cricsheet_match_id: str,
    readme_row: Optional[cricsheet_readme.CricsheetReadmeRow],
    summary: CricsheetIngestSummary,
) -> None:
    try:
        payload = cricsheet_convert.load_cricsheet_payload(path, cricsheet_match_id=cricsheet_match_id)
    except Exception as exc:  # noqa: BLE001
        summary.matches_skipped_malformed += 1
        msg = f"{cricsheet_match_id}: {type(exc).__name__}: {exc}"
        summary.warnings.append(msg)
        logger.warning("cricsheet ingest skip malformed: %s", msg)
        return

    apply_readme_row_to_payload(payload, readme_row)

    _sql_mid, status = db.insert_parsed_match(payload, skip_derived_aggregates=True)
    if status == "inserted":
        summary.matches_inserted += 1
        summary.player_stats_rows_inserted += _count_player_stats_rows(payload)
        summary.batting_position_rows_inserted += _count_batting_position_rows(payload)
        summary.phase_rows_inserted += _count_phase_rows(payload)
    elif status in ("duplicate_url", "duplicate_match"):
        summary.matches_skipped_duplicate += 1
    else:
        summary.warnings.append(f"{cricsheet_match_id}: unexpected status {status}")


def run_cricsheet_folder_ingest(
    *,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    report_readme_gaps: bool = False,
) -> CricsheetIngestSummary:
    """
    Scan ``json_dir`` for ``*.json``; ingest each numeric id not already in SQLite.

    When ``report_readme_gaps`` is True (initial backfill), counts readme index rows whose
    JSON file is missing from the folder.
    """
    summary = CricsheetIngestSummary()
    jdir = Path(json_dir or config.CRICSHEET_JSON_DIR)
    if not jdir.is_dir():
        summary.warnings.append(f"JSON directory does not exist: {jdir}")
        return summary

    readme_rows = cricsheet_readme.load_readme_rows(readme_path)
    summary.readme_rows_total = len(readme_rows)
    by_id: dict[str, cricsheet_readme.CricsheetReadmeRow] = {r.match_id: r for r in readme_rows}

    if report_readme_gaps:
        for mid, _row in by_id.items():
            if not (jdir / f"{mid}.json").is_file():
                summary.readme_rows_missing_json += 1

    ids = discover_json_match_ids(jdir)
    summary.json_files_on_disk = len(ids)
    already = db.existing_cricsheet_match_ids()

    for mid in ids:
        if mid in already:
            summary.matches_skipped_duplicate += 1
            continue
        path = jdir / f"{mid}.json"
        _ingest_one_file(path, mid, by_id.get(mid), summary)
        already.add(mid)

    if summary.matches_inserted > 0:
        try:
            db.rebuild_prediction_summary_tables()
        except Exception as exc:  # noqa: BLE001
            summary.warnings.append(f"prediction summary rebuild failed: {type(exc).__name__}: {exc}")
            logger.exception("prediction summary rebuild failed after folder ingest")

    return summary


def run_initial_cricsheet_backfill(
    *,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
) -> CricsheetIngestSummary:
    """UI: full pass over the archive + readme gap report."""
    return run_cricsheet_folder_ingest(
        json_dir=json_dir,
        readme_path=readme_path,
        report_readme_gaps=True,
    )


def run_sync_new_cricsheet_matches(
    *,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
) -> CricsheetIngestSummary:
    """UI: ingest JSON files whose ``match_id`` is not yet in SQLite (no readme gap report)."""
    return run_cricsheet_folder_ingest(
        json_dir=json_dir,
        readme_path=readme_path,
        report_readme_gaps=False,
    )


@dataclass
class RawCricsheetRebuildSummary:
    """Admin rebuild of Cricsheet raw rows (optional wipe + re-ingest readme window)."""

    clear_first: bool = False
    cleared_match_results: int = 0
    readme_rows_in_window: int = 0
    readme_rows_total_before_filter: int = 0
    ingest_mode: str = "sliding_recent_window"
    season_years_in_window: Optional[list[int]] = None
    full_archive_ingest: bool = False
    matches_missing_json_on_disk: int = 0
    matches_inserted: int = 0
    matches_skipped_duplicate: int = 0
    matches_skipped_malformed: int = 0
    player_stats_rows_from_payloads: int = 0
    batting_position_rows_from_payloads: int = 0
    phase_rows_from_payloads: int = 0
    warnings: list[str] = field(default_factory=list)
    sqlite_totals_after: dict[str, int] = field(default_factory=dict)
    sqlite_calendar_audit: dict[str, Any] = field(default_factory=dict)


def _sqlite_raw_totals() -> dict[str, int]:
    with db.connection() as conn:
        def c(q: str) -> int:
            return int(conn.execute(q).fetchone()[0])

        return {
            "match_results_cricsheet": c(
                "SELECT COUNT(*) FROM match_results WHERE url LIKE 'cricsheet://ipl/%'"
            ),
            "matches": c("SELECT COUNT(*) FROM matches"),
            "team_match_xi": c("SELECT COUNT(*) FROM team_match_xi"),
            "team_match_summary": c("SELECT COUNT(*) FROM team_match_summary"),
            "player_match_stats": c("SELECT COUNT(*) FROM player_match_stats"),
            "player_batting_positions": c("SELECT COUNT(*) FROM player_batting_positions"),
            "player_phase_usage": c("SELECT COUNT(*) FROM player_phase_usage"),
        }


def run_rebuild_raw_cricsheet_ingest(
    *,
    clear_first: bool = True,
    json_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    current_season_year: Optional[int] = None,
    n_seasons: Optional[int] = None,
    full_archive_ingest: Optional[bool] = None,
) -> RawCricsheetRebuildSummary:
    """
    Optional wipe of all Cricsheet-derived ``match_results`` (and dependent raw rows), then re-ingest
    IPL readme rows when ``{match_id}.json`` exists.

    Default: last ``n_seasons`` calendar years (see ``CRICSHEET_HISTORY_SEASON_COUNT``).
    With ``full_archive_ingest=True`` (or ``config.CRICSHEET_FULL_ARCHIVE_INGEST``): all IPL readme rows.

    Does **not** run Stage 2 derive (``skip_derived_aggregates=True`` on insert).
    """
    use_full = (
        bool(full_archive_ingest)
        if full_archive_ingest is not None
        else bool(getattr(config, "CRICSHEET_FULL_ARCHIVE_INGEST", False))
    )
    out = RawCricsheetRebuildSummary(
        clear_first=bool(clear_first),
        full_archive_ingest=use_full,
        ingest_mode="full_archive" if use_full else "sliding_recent_window",
    )
    jdir = Path(json_dir or config.CRICSHEET_JSON_DIR)
    cur = int(current_season_year or getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    n = int(n_seasons or getattr(config, "CRICSHEET_HISTORY_SEASON_COUNT", 5))

    if clear_first:
        cleared = db.delete_all_cricsheet_ingested_matches()
        out.cleared_match_results = int(
            cleared.get("cricsheet_match_results_found") or cleared.get("match_results_deleted") or 0
        )

    rows = cricsheet_readme.load_readme_rows(readme_path)
    out.readme_rows_total_before_filter = len(rows)
    if use_full:
        filtered = list(rows)
        out.season_years_in_window = None
    else:
        filtered = cricsheet_readme.filter_last_n_seasons(
            rows, current_season_year=cur, n_seasons=n
        )
        out.season_years_in_window = sorted(
            cricsheet_readme.season_years_window(cur, n),
            reverse=True,
        )
    out.readme_rows_in_window = len(filtered)
    by_id = {r.match_id: r for r in filtered}
    ing = CricsheetIngestSummary()

    for mid in sorted(by_id.keys(), key=lambda x: int(x)):
        path = jdir / f"{mid}.json"
        if not path.is_file():
            out.matches_missing_json_on_disk += 1
            continue
        _ingest_one_file(path, mid, by_id.get(mid), ing)

    out.matches_inserted = ing.matches_inserted
    out.matches_skipped_duplicate = ing.matches_skipped_duplicate
    out.matches_skipped_malformed = ing.matches_skipped_malformed
    out.player_stats_rows_from_payloads = ing.player_stats_rows_inserted
    out.batting_position_rows_from_payloads = ing.batting_position_rows_inserted
    out.phase_rows_from_payloads = ing.phase_rows_inserted
    out.warnings = list(ing.warnings)
    out.sqlite_totals_after = _sqlite_raw_totals()
    try:
        out.sqlite_calendar_audit = db.sqlite_matches_temporal_audit(cricsheet_derived_only=True)
    except Exception as exc:  # noqa: BLE001
        out.sqlite_calendar_audit = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def ingest_local_history_for_teams(
    canonical_team_a: str,
    canonical_team_b: str,
    *,
    current_season_year: Optional[int] = None,
    season_count: Optional[int] = None,
    json_dir: Optional[Path] = None,
    full_archive_ingest: Optional[bool] = None,
) -> dict[str, Any]:
    """
    **Ingest stage only** — union of matches involving either franchise (filtered readme window).

    Does **not** call ``learner.ingest_payload`` or refresh franchise feature aggregates
    (stage 1 raw ingest only).
    """
    use_full = (
        bool(full_archive_ingest)
        if full_archive_ingest is not None
        else bool(getattr(config, "CRICSHEET_FULL_ARCHIVE_INGEST", False))
    )
    cur = int(current_season_year or getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    n_seasons = int(season_count or getattr(config, "CRICSHEET_HISTORY_SEASON_COUNT", 5))
    jdir = Path(json_dir or getattr(config, "CRICSHEET_JSON_DIR", config.DATA_DIR / "ipl_json"))

    canon_a = ipl_teams.franchise_label_for_storage(canonical_team_a) or (canonical_team_a or "").strip()
    canon_b = ipl_teams.franchise_label_for_storage(canonical_team_b) or (canonical_team_b or "").strip()

    debug: dict[str, Any] = {
        "readme_path": None,
        "json_dir": str(jdir),
        "ipl_current_season_year": cur,
        "ingest_mode": "full_archive" if use_full else "sliding_recent_window",
        "season_years_used": None
        if use_full
        else sorted(
            cricsheet_readme.season_years_window(cur, n_seasons),
            reverse=True,
        ),
        "readme_rows_total_ipl_male": 0,
        "readme_rows_after_season_filter": 0,
        "matches_union_ids": 0,
        "matches_for_team_a_only_count": 0,
        "matches_for_team_b_only_count": 0,
        "json_files_opened": 0,
        "json_files_missing": 0,
        "inserted": 0,
        "skipped_duplicate_url": 0,
        "skipped_duplicate_match": 0,
        "player_stats_rows_inserted": 0,
        "batting_position_rows_inserted": 0,
        "phase_rows_inserted": 0,
        "parse_errors": [],
        "used_current_season_in_index": False,
        "used_prior_season_in_index": False,
    }

    readme_path = cricsheet_readme.resolve_readme_path()
    if readme_path is None:
        debug["error"] = "readme_not_found"
        debug["hint"] = (
            "Place the Cricsheet index at data/readme.txt or under data/ipl_json/ "
            "(see config.CRICSHEET_README_CANDIDATES)."
        )
        return debug

    debug["readme_path"] = str(readme_path)
    all_rows = cricsheet_readme.parse_cricsheet_readme(readme_path)
    debug["readme_rows_total_ipl_male"] = len(all_rows)

    if use_full:
        filtered = list(all_rows)
    else:
        years = cricsheet_readme.season_years_window(cur, n_seasons)
        filtered = cricsheet_readme.filter_rows_by_seasons(all_rows, years)
    debug["readme_rows_after_season_filter"] = len(filtered)

    for r in filtered:
        y = cricsheet_readme.row_season_year(r)
        if y == cur:
            debug["used_current_season_in_index"] = True
        if y < cur:
            debug["used_prior_season_in_index"] = True

    rows_a = cricsheet_readme.rows_involving_franchises(filtered, [canon_a])
    rows_b = cricsheet_readme.rows_involving_franchises(filtered, [canon_b])
    ids_a = {r.match_id for r in rows_a}
    ids_b = {r.match_id for r in rows_b}
    debug["matches_for_team_a_only_count"] = len(ids_a)
    debug["matches_for_team_b_only_count"] = len(ids_b)

    union_ids = sorted(ids_a | ids_b, key=lambda x: int(x), reverse=True)
    debug["matches_union_ids"] = len(union_ids)

    by_id = {r.match_id: r for r in filtered}
    already = db.existing_cricsheet_match_ids()
    for mid in union_ids:
        path = jdir / f"{mid}.json"
        if not path.is_file():
            debug["json_files_missing"] += 1
            continue
        if mid in already:
            debug["skipped_duplicate_url"] += 1
            continue
        debug["json_files_opened"] += 1
        try:
            payload = cricsheet_convert.load_cricsheet_payload(path, cricsheet_match_id=str(mid))
        except Exception as exc:  # noqa: BLE001
            msg = f"{mid}: {type(exc).__name__}: {exc}"
            logger.warning("cricsheet parse failed %s", msg)
            debug["parse_errors"].append(msg)
            continue

        apply_readme_row_to_payload(payload, by_id.get(mid))

        _mid_sql, status = db.insert_parsed_match(payload, skip_derived_aggregates=True)
        if status == "inserted":
            debug["inserted"] += 1
            debug["player_stats_rows_inserted"] = int(debug.get("player_stats_rows_inserted") or 0) + _count_player_stats_rows(
                payload
            )
            debug["batting_position_rows_inserted"] = int(
                debug.get("batting_position_rows_inserted") or 0
            ) + _count_batting_position_rows(payload)
            debug["phase_rows_inserted"] = int(debug.get("phase_rows_inserted") or 0) + _count_phase_rows(payload)
            already.add(mid)
        elif status == "duplicate_url":
            debug["skipped_duplicate_url"] += 1
        elif status == "duplicate_match":
            debug["skipped_duplicate_match"] += 1

    return debug


__all__ = [
    "CricsheetIngestSummary",
    "RawCricsheetRebuildSummary",
    "apply_readme_row_to_payload",
    "discover_json_match_ids",
    "ingest_local_history_for_teams",
    "run_cricsheet_folder_ingest",
    "run_initial_cricsheet_backfill",
    "run_rebuild_raw_cricsheet_ingest",
    "run_sync_new_cricsheet_matches",
]
