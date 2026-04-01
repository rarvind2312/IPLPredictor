"""
Ingest the full multi-competition Cricsheet JSON archive (``data/all_json``).

**Ingest-only**: reads JSON from disk; prediction and selection use SQLite only.
Use :func:`run_cricsheet_all_archive_ingest` then :func:`recent_form_cache.rebuild_player_recent_form_cache`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
import cricsheet_convert
import db

logger = logging.getLogger(__name__)


@dataclass
class CricsheetAllArchiveSummary:
    json_files_seen: int = 0
    matches_inserted: int = 0
    matches_resynced_duplicate: int = 0
    matches_skipped_duplicate: int = 0
    matches_skipped_duplicate_url: int = 0
    matches_skipped_malformed: int = 0
    warnings: list[str] = field(default_factory=list)


def run_cricsheet_all_archive_ingest(
    *,
    json_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> CricsheetAllArchiveSummary:
    """
    Walk ``*.json`` numeric stems under ``json_dir`` (default ``config.CRICSHEET_ALL_JSON_DIR``).

    Each file is parsed with ``url_scheme='all'`` (real competition + ``match_format`` from Cricsheet
    ``info``). ``resync_on_duplicate_match=True`` refreshes rows when the same teams+date already exist
    (e.g. after an IPL-only ingest).
    """
    jdir = Path(json_dir or config.CRICSHEET_ALL_JSON_DIR)
    summary = CricsheetAllArchiveSummary()
    if not jdir.is_dir():
        summary.warnings.append(f"not a directory: {jdir}")
        return summary

    paths = sorted(
        [p for p in jdir.glob("*.json") if p.stem.strip().isdigit()],
        key=lambda p: int(p.stem.strip()),
    )
    summary.json_files_seen = len(paths)
    for i, path in enumerate(paths):
        if limit is not None and i >= limit:
            break
        stem = path.stem.strip()
        try:
            payload = cricsheet_convert.load_cricsheet_payload(
                path, cricsheet_match_id=stem, url_scheme="all"
            )
        except Exception as exc:  # noqa: BLE001
            summary.matches_skipped_malformed += 1
            msg = f"{stem}: {type(exc).__name__}: {exc}"
            summary.warnings.append(msg)
            logger.warning("cricsheet all_json skip malformed: %s", msg)
            continue

        try:
            _mid, status = db.insert_parsed_match(
                payload,
                skip_derived_aggregates=True,
                resync_on_duplicate_match=True,
            )
        except Exception as exc:  # noqa: BLE001
            summary.matches_skipped_malformed += 1
            summary.warnings.append(f"{stem}: db {type(exc).__name__}: {exc}")
            logger.exception("cricsheet all_json db error id=%s", stem)
            continue

        if status == "inserted":
            summary.matches_inserted += 1
        elif status == "resynced_duplicate":
            summary.matches_resynced_duplicate += 1
        elif status == "duplicate_url":
            summary.matches_skipped_duplicate_url += 1
        elif status == "duplicate_match":
            summary.matches_skipped_duplicate += 1
        else:
            summary.warnings.append(f"{stem}: unexpected status {status}")

    if summary.matches_inserted > 0 or summary.matches_resynced_duplicate > 0:
        try:
            db.rebuild_prediction_summary_tables()
        except Exception as exc:  # noqa: BLE001
            summary.warnings.append(f"prediction summary rebuild failed: {type(exc).__name__}: {exc}")
            logger.exception("prediction summary rebuild failed after all_json ingest")

    return summary


__all__ = ["CricsheetAllArchiveSummary", "run_cricsheet_all_archive_ingest"]
