"""Recent Cricsheet sync + query helpers for latest match details."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cricsheet_convert
import db
import utils

logger = logging.getLogger(__name__)

CRICSHEET_RECENT_ZIP_URL = "https://cricsheet.org/downloads/recently_added_2_json.zip"
CRICSHEET_RECENT_SOURCE_KIND = "recently_added_2_json"


@dataclass
class CricsheetRecentSyncSummary:
    source_url: str
    sync_scope: str = CRICSHEET_RECENT_SOURCE_KIND
    downloaded_files: int = 0
    parsed_matches: int = 0
    inserted_new: int = 0
    updated_existing: int = 0
    skipped_duplicates: int = 0
    failed_parses: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    audit_id: Optional[int] = None
    failures: list[str] = field(default_factory=list)

    def finalize(self) -> dict[str, Any]:
        self.finished_at = time.time()
        notes = {"failures": self.failures[:100]}
        row = {
            "sync_scope": self.sync_scope,
            "source_url": self.source_url,
            "downloaded_files": self.downloaded_files,
            "parsed_matches": self.parsed_matches,
            "inserted_new": self.inserted_new,
            "updated_existing": self.updated_existing,
            "skipped_duplicates": self.skipped_duplicates,
            "failed_parses": self.failed_parses,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": notes,
        }
        self.audit_id = db.insert_cricsheet_sync_audit(row)
        out = asdict(self)
        out["notes"] = notes
        return out


def _download_recent_zip(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IPLPredictor/1.0; +https://cricsheet.org/)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _competition_matches_filter(competition_label: Optional[str], competition: Optional[str]) -> bool:
    if not competition:
        return True
    comp = str(competition or "").strip().lower()
    label = str(competition_label or "").strip().lower()
    if comp == "ipl":
        return "ipl" in label or "indian premier league" in label
    return label == comp


def _series_name_from_info(info: dict[str, Any]) -> Optional[str]:
    event = info.get("event")
    if isinstance(event, dict):
        name = str(event.get("name") or "").strip()
        if name:
            return name[:160]
    events = info.get("events")
    if isinstance(events, list):
        for item in events:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    return name[:160]
    return None


def _fallback_identity_key(team_a: Optional[str], team_b: Optional[str], match_date: Optional[str], venue: Optional[str]) -> str:
    base = utils.canonical_match_identity_key(team_a, team_b, match_date) or ""
    venue_part = str(venue or "").strip().lower()
    return f"{base}|{venue_part}" if venue_part else base


def _upsert_one_recent_match(
    payload: dict[str, Any],
    *,
    source_match_key: str,
    source_url: str,
    source_file: str,
    source_kind: str,
    raw_json_ref: Optional[str] = None,
) -> str:
    match_id, status = db.insert_parsed_match(
        payload,
        skip_derived_aggregates=True,
        resync_on_duplicate_match=True,
    )
    meta = payload.get("meta") or {}
    teams = list(payload.get("teams") or [])
    team_a = teams[0] if len(teams) > 0 else None
    team_b = teams[1] if len(teams) > 1 else None
    db.upsert_cricsheet_match_catalog(
        {
            "source_match_key": source_match_key,
            "match_id": match_id or db.find_match_id_for_cricsheet_source_key(source_match_key),
            "canonical_match_key": utils.canonical_match_identity_key(team_a, team_b, meta.get("date")),
            "fallback_identity_key": _fallback_identity_key(
                team_a, team_b, meta.get("date"), meta.get("venue")
            ),
            "competition": meta.get("competition"),
            "match_type": meta.get("match_format"),
            "gender": meta.get("gender"),
            "series_name": meta.get("series_name"),
            "match_date": meta.get("date"),
            "venue": meta.get("venue"),
            "city": meta.get("city"),
            "team_a": team_a,
            "team_b": team_b,
            "toss_winner": meta.get("toss_winner"),
            "toss_decision": meta.get("toss_decision"),
            "winner": meta.get("winner"),
            "result_text": meta.get("result_text") or meta.get("margin"),
            "source_url": source_url,
            "source_kind": source_kind,
            "source_file": source_file,
            "raw_json_ref": raw_json_ref,
            "updated_at": time.time(),
        }
    )
    return status


def sync_recent_matches_from_zip_path(
    zip_path: str | Path,
    *,
    source_url: str = CRICSHEET_RECENT_ZIP_URL,
    competition_filter: Optional[str] = None,
) -> dict[str, Any]:
    summary = CricsheetRecentSyncSummary(source_url=source_url)
    archive = Path(zip_path)
    with tempfile.TemporaryDirectory(prefix="cricsheet_recent_") as tmpdir:
        with zipfile.ZipFile(archive) as zf:
            members = sorted(
                name for name in zf.namelist() if name.lower().endswith(".json") and not name.endswith("/")
            )
            summary.downloaded_files = len(members)
            zf.extractall(tmpdir)
        for member in members:
            member_path = Path(tmpdir) / member
            source_match_key = member_path.stem.strip()
            try:
                raw = json.loads(member_path.read_text(encoding="utf-8"))
                info = raw.get("info") or {}
                payload = cricsheet_convert.cricsheet_json_to_payload(
                    raw,
                    cricsheet_match_id=source_match_key,
                    url_scheme="all",
                )
                payload_meta = payload.get("meta") or {}
                payload_meta["gender"] = str(info.get("gender") or "").strip() or None
                payload_meta["series_name"] = _series_name_from_info(info)
                payload["meta"] = payload_meta
                if not _competition_matches_filter(payload_meta.get("competition"), competition_filter):
                    continue
                summary.parsed_matches += 1
                status = _upsert_one_recent_match(
                    payload,
                    source_match_key=source_match_key,
                    source_url=source_url,
                    source_file=f"{archive.name}:{member}",
                    source_kind=CRICSHEET_RECENT_SOURCE_KIND,
                    raw_json_ref=f"{archive.name}:{member}",
                )
                if status == "inserted":
                    summary.inserted_new += 1
                elif status == "resynced_duplicate":
                    summary.updated_existing += 1
                else:
                    summary.skipped_duplicates += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("recent cricsheet parse failed member=%s", member)
                summary.failed_parses += 1
                summary.failures.append(f"{member}: {type(exc).__name__}: {exc}")
    return summary.finalize()


def sync_latest_cricsheet_matches(
    *,
    download_url: str = CRICSHEET_RECENT_ZIP_URL,
    competition_filter: Optional[str] = None,
    timeout: int = 45,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="cricsheet_recent_", suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(_download_recent_zip(download_url, timeout=timeout))
        tmp.flush()
    try:
        return sync_recent_matches_from_zip_path(
            tmp_path,
            source_url=download_url,
            competition_filter=competition_filter,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("failed to clean temp cricsheet zip %s", tmp_path)


def post_sync_latest(*, competition_filter: Optional[str] = None) -> dict[str, Any]:
    """API-ready POST /api/cricsheet/sync-latest equivalent."""
    return sync_latest_cricsheet_matches(competition_filter=competition_filter)


def get_recent_matches(*, competition: Optional[str] = "ipl", days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    """API-ready GET /api/cricsheet/recent-matches equivalent."""
    return db.fetch_cricsheet_recent_matches(competition=competition, days=days, limit=limit)


def get_match(match_key: str) -> Optional[dict[str, Any]]:
    """API-ready GET /api/cricsheet/match/{match_key} equivalent."""
    return db.fetch_cricsheet_match(match_key)


def get_team_recent_matches(
    team_name: str,
    *,
    competition: Optional[str] = "ipl",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """API-ready GET /api/cricsheet/team/{team_name}/recent-matches equivalent."""
    return db.fetch_cricsheet_recent_matches(
        competition=competition,
        days=3650,
        limit=limit,
        team_name=team_name,
    )
