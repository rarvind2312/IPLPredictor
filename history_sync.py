"""
Local SQLite history context for XI prediction (**read-only** in the prediction path).

**Prediction** uses only what is already in SQLite (``matches``, ``team_match_xi``, etc.).

Historical rows are populated by a separate **ingest** stage (local Cricsheet archive, manual
**Parse & store**, or other loaders) — not during **Run prediction**.

What remains here:
- ``local_history_debug_for_prediction``: SQLite franchise snapshot + optional squad↔history join
  (no network, no JSON ingest).
- ``fetch_and_store_scorecard``: one-off URL fetch → parse → SQLite (sidebar **Parse & store** only).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

import config
import db
import ipl_teams
import learner
import player_alias_resolve
import utils
from parsers import router
from parsers.schema import has_storable_content

logger = logging.getLogger(__name__)

HISTORY_MISSING_USER_MESSAGE = (
    "Historical data must be loaded from local Cricsheet archive using the ingest stage."
)


def raw_stage1_tables_near_empty_from_snapshot(snap: dict[str, Any]) -> bool:
    """True when franchise raw history is empty or near-empty (ingest reminder justified)."""
    dm = int(snap.get("distinct_match_count") or 0)
    xi = int(snap.get("xi_row_count") or 0)
    thr_m = int(getattr(config, "STAGE_1_RAW_NEAR_EMPTY_MATCHES", 3))
    thr_x = int(getattr(config, "STAGE_1_RAW_NEAR_EMPTY_XI_ROWS", 40))
    return dm < thr_m or xi < thr_x


def raw_stage1_tables_near_empty(block: dict[str, Any]) -> bool:
    """Same as snapshot helper, using fields already copied onto a ``local_history_debug`` team block."""
    if block.get("raw_stage1_near_empty") is not None:
        return bool(block["raw_stage1_near_empty"])
    dm = int(block.get("distinct_matches_local_db") or 0)
    xi = int(block.get("rows_loaded_local_db_xi") or 0)
    thr_m = int(getattr(config, "STAGE_1_RAW_NEAR_EMPTY_MATCHES", 3))
    thr_x = int(getattr(config, "STAGE_1_RAW_NEAR_EMPTY_XI_ROWS", 40))
    return dm < thr_m or xi < thr_x


def build_squad_vs_history_report(
    canonical_label: str,
    squad_player_names: list[str],
) -> dict[str, Any]:
    """Per squad player: join stats against ``team_match_xi`` (alias-resolved ``player_key``)."""
    all_rows = db.history_team_xi_rows_for_franchise(canonical_label, limit=900)
    lab = ipl_teams.franchise_label_for_storage(canonical_label) or (canonical_label or "").strip()
    ck = ipl_teams.canonical_team_key_for_franchise(lab)
    franchise_keys = db.franchise_distinct_history_player_keys(ck)
    snap = db.franchise_history_snapshot(lab)
    distinct_m = int(snap.get("distinct_match_count") or 0)
    xi_rows = int(snap.get("xi_row_count") or 0)
    fk_n = len(franchise_keys)
    per_player: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for raw_name in squad_player_names:
        nm = (raw_name or "").strip()
        if not nm:
            continue
        canon_pk = learner.normalize_player_key(nm)
        res = player_alias_resolve.resolve_player_to_history_key(nm, franchise_keys)
        hk = player_alias_resolve.history_lookup_key_from_resolution(res)
        hstat = player_alias_resolve.history_status_from_resolution(res)
        matched_hist_nm = db.sample_stored_player_name_for_key(ck, hk) if hk else None
        matched = [r for r in all_rows if hk and (r.get("player_key") or "") == hk]
        if not matched:
            unmatched.append(nm)
        matched_sorted = sorted(
            matched,
            key=lambda r: (float(r.get("created_at") or 0), int(r.get("match_id") or 0)),
            reverse=True,
        )
        seen_mid: list[int] = []
        for r in matched_sorted:
            mid = int(r["match_id"])
            if mid not in seen_mid:
                seen_mid.append(mid)
        top10 = set(seen_mid[:10])
        rows_in_last_10_distinct = sum(1 for r in matched if int(r["match_id"]) in top10)
        recent_positions: list[Any] = []
        for r in matched_sorted[:20]:
            bp = r.get("batting_position")
            if bp is None:
                continue
            try:
                recent_positions.append(float(bp))
            except (TypeError, ValueError):
                continue
        distinct_xi = len({int(r["match_id"]) for r in matched})
        rolled = player_alias_resolve.rolled_up_history_interpretation(
            res,
            distinct_franchise_matches=distinct_m,
            franchise_xi_row_count=xi_rows,
            franchise_key_count=fk_n,
            role_bucket=None,
            usable_history_rows=len(matched),
        )
        layer_debug = {
            "resolution_layer_used": res.resolution_layer_used,
            "surname_bucket_size": res.surname_bucket_size,
            "surname_candidates_checked": list(res.surname_candidates_checked),
            "relaxed_unique_surname_rule_applied": res.layer_d_branch == "relaxed_unique_surname",
            "layer_d_branch": res.layer_d_branch or None,
            "layer_d_reason": res.layer_d_reason or None,
        }
        per_player.append(
            {
                "squad_display_name": nm,
                "canonical_squad_key": canon_pk,
                "normalized_full_name_key": res.normalized_full_name_key,
                "resolved_history_key": res.resolved_history_key,
                "history_lookup_key": hk,
                "resolution_type": res.resolution_type,
                "surname_bucket_size": res.surname_bucket_size,
                "history_status": hstat,
                "rolled_up_interpretation": rolled,
                "resolution_layer_debug": layer_debug,
                "matched_history_player_name": matched_hist_nm,
                "alias_confidence": round(float(res.confidence), 4),
                "confidence": round(float(res.confidence), 4),
                "ambiguous_candidates": list(res.ambiguous_candidates),
                "team_match_xi_rows_found": len(matched),
                "player_match_stats_rows_found": None,
                "player_batting_positions_rows_found": None,
                "latest_match_date_found": matched_sorted[0].get("match_date") if matched_sorted else None,
                "matched_history_row_count": len(matched),
                "distinct_matches_with_player_in_xi": distinct_xi,
                "xi_rows_in_last_10_stored_matches": rows_in_last_10_distinct,
                "latest_batting_positions_sample": recent_positions[:10],
                "sample_recent_match_ids": seen_mid[:10],
            }
        )
    resolve_keys: list[str] = []
    for row in per_player:
        hk = (row.get("history_lookup_key") or "").strip()
        resolve_keys.append(hk or str(row.get("canonical_squad_key") or ""))
    uq = list(dict.fromkeys(k for k in resolve_keys if k))
    glob_xi = db.batch_global_team_match_xi_stats(uq)
    glob_other = db.batch_player_other_franchise_tmx_counts(uq, ck)
    fc_min_g = int(getattr(config, "FIRST_CHOICE_GLOBAL_MIN_DISTINCT_MATCHES", 2))
    for row, qk in zip(per_player, resolve_keys):
        qk = (qk or "").strip()
        gx = glob_xi.get(qk) or {}
        gdm = int(gx.get("distinct_matches") or 0)
        other_n = int(glob_other.get(qk, 0))
        dmx = int(row.get("distinct_matches_with_player_in_xi") or 0)
        g_pres = gdm >= 1
        row["global_ipl_history_presence"] = g_pres
        row["selected_franchise_history_presence"] = dmx > 0
        row["history_for_other_franchises_presence"] = other_n > 0
        row["valid_current_squad_new_to_franchise_squad_report"] = bool(
            g_pres
            and dmx < fc_min_g
            and (other_n > 0 or gdm >= fc_min_g)
        )
    return {
        "franchise_canonical": canonical_label,
        "canonical_team_key": ck,
        "franchise_distinct_history_player_keys": len(franchise_keys),
        "current_squad_count": len([n for n in squad_player_names if str(n).strip()]),
        "franchise_total_history_xi_rows_loaded": len(all_rows),
        "per_player": per_player,
        "unmatched_current_squad_names": unmatched,
    }


def _snapshot_sufficiency(
    snap: dict[str, Any],
    *,
    min_recent: int,
    min_prior: int,
) -> tuple[bool, bool, bool, bool]:
    n = int(snap.get("distinct_match_count") or 0)
    prior = int(snap.get("prior_season_match_count") or 0)
    stale = bool(snap.get("stale_local_cache"))
    recent_ok = n >= min_recent and not stale
    prior_ok = prior >= min_prior
    history_ok = recent_ok and prior_ok
    return history_ok, recent_ok, prior_ok, stale


def get_cached_match_count(team_name: str) -> int:
    """Count distinct historical matches stored for this franchise (local SQLite)."""
    canon = ipl_teams.canonical_franchise_label(team_name) or (team_name or "").strip()
    return db.get_cached_match_count_for_franchise(canon)


def _payload_ok_for_history(payload: dict[str, Any]) -> bool:
    if not has_storable_content(payload):
        return False
    ing = payload.get("ingestion") or {}
    if not ing.get("fetch_ok"):
        return False
    for side in payload.get("playing_xi") or []:
        pl = [str(x).strip() for x in (side.get("players") or []) if str(x).strip()]
        if len(pl) >= 11:
            return True
    return False


def fetch_and_store_scorecard(
    scorecard_url: str,
    source: Optional[str] = None,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """
    Fetch a scorecard URL, parse, insert into SQLite. Not invoked during **Run prediction**.

    Prefer Cricsheet backfill for bulk history; use Streamlit **Parse & store** or this helper for
    occasional manual URLs.
    """
    sess = session or requests.Session()
    nu = utils.normalize_scorecard_url(scorecard_url)
    payload = router.parse_scorecard(nu, session=sess)
    meta = payload.get("meta") or {}
    parser_source = str(meta.get("source") or source or "unknown")
    ing = payload.get("ingestion") or {}

    if not _payload_ok_for_history(payload):
        logger.info(
            "fetch_and_store_scorecard rejected url=%s parser=%s fetch_ok=%s",
            nu[:120],
            parser_source,
            ing.get("fetch_ok"),
        )
        return {
            "ok": False,
            "url": nu,
            "parser": parser_source,
            "status": "rejected",
            "ingestion": ing,
            "match_id": None,
            "fetch_ok": bool(ing.get("fetch_ok")),
        }

    mid, st = db.insert_parsed_match(payload)
    if st in ("inserted", "resynced_duplicate"):
        learner.ingest_payload(payload)
        try:
            db.rebuild_prediction_summary_tables()
        except Exception:
            logger.exception("prediction summary rebuild failed after fetch_and_store_scorecard")
        try:
            db.rebuild_player_metadata_and_matchup_summaries()
        except Exception:
            logger.exception("metadata/matchup rebuild failed after fetch_and_store_scorecard")
        logger.info(
            "fetch_and_store_scorecard inserted match_id=%s url=%s parser=%s",
            mid,
            nu[:120],
            parser_source,
        )
    else:
        logger.info(
            "fetch_and_store_scorecard skip match_id=%s status=%s url=%s parser=%s",
            mid,
            st,
            nu[:120],
            parser_source,
        )

    return {
        "ok": True,
        "match_id": mid,
        "status": st,
        "parser": parser_source,
        "url": nu,
        "fetch_ok": bool(ing.get("fetch_ok")),
    }


def local_history_debug_for_prediction(
    canonical_label: str,
    *,
    squad_player_names: Optional[list[str]] = None,
    include_squad_report: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Read-only: local SQLite franchise snapshot + optional squad↔history join (no network).

    Replaces the old ``ensure_team_history_for_prediction`` internet sync pipeline.
    """
    canon = (canonical_label or "").strip()
    slug = ipl_teams.slug_for_canonical_label(canon)
    min_r = int(getattr(config, "HISTORY_SYNC_MIN_RECENT_MATCHES", 5))
    min_p = int(getattr(config, "HISTORY_SYNC_MIN_PRIOR_SEASON_MATCHES", 2))
    min_warn = int(getattr(config, "LOCAL_HISTORY_MIN_DISTINCT_MATCHES_WARN", 2))

    snap = db.franchise_history_snapshot(canon)
    cached_before = int(snap.get("distinct_match_count") or 0)
    hist_ok, recent_ok, prior_ok, stale = _snapshot_sufficiency(snap, min_recent=min_r, min_prior=min_p)

    names = [n for n in (squad_player_names or []) if str(n).strip()]
    if include_squad_report is None:
        # Default to include the report when called directly (tests/debug). Prediction-time callers
        # explicitly pass ``include_squad_report=False`` to avoid the extra work.
        include_squad_report = True
    squad_report = build_squad_vs_history_report(canon, names) if (include_squad_report and names) else None

    alias_warns: list[str] = []
    if squad_report:
        for row in squad_report.get("per_player") or []:
            nm = row.get("squad_display_name") or ""
            rt = row.get("resolution_type") or ""
            rolled = row.get("rolled_up_interpretation") or ""
            if rt == "ambiguous_alias":
                cands = row.get("ambiguous_candidates") or []
                alias_warns.append(
                    f"Ambiguous history alias for {nm!r} ({rolled}): {len(cands)} SQLite key candidate(s); not auto-linked."
                )
            elif rt == "no_match" and int(row.get("matched_history_row_count") or 0) == 0:
                if bool(row.get("global_ipl_history_presence")) and (
                    bool(row.get("history_for_other_franchises_presence"))
                    or bool(row.get("valid_current_squad_new_to_franchise_squad_report"))
                ):
                    alias_warns.append(
                        f"{nm!r}: current squad player new to franchise in stored history; "
                        "global IPL fallback prior applies (not a hard linkage failure by itself)."
                    )
                elif rolled == "likely_alias_miss":
                    alias_warns.append(
                        f"No SQLite key linked yet for {nm!r} ({rolled}); check ``resolution_layer_debug`` in JSON."
                    )
                else:
                    alias_warns.append(
                        f"No stored XI/stats key matched for {nm!r} ({rolled}); using role/heuristic fallbacks."
                    )

    n_matches = int(snap.get("distinct_match_count") or 0)
    xi_n = int(snap.get("xi_row_count") or 0)
    near_empty = raw_stage1_tables_near_empty_from_snapshot(snap)
    notes: list[str] = []
    if n_matches < min_warn:
        if near_empty:
            notes.append(
                f"SQLite has very few stored matches for this franchise ({n_matches} distinct fixtures, "
                f"{xi_n} ``team_match_xi`` rows). " + HISTORY_MISSING_USER_MESSAGE
            )
        else:
            notes.append(
                f"SQLite has {n_matches} distinct stored match(es) for this franchise (≥{min_warn} recommended "
                "for richer history signals). Prediction still uses available rows."
            )
    if stale:
        days = float(getattr(config, "HISTORY_SYNC_STALE_DAYS", 10.0))
        notes.append(
            f"Newest stored match is older than ~{days:.0f} days; recency-weighted history may be stale."
        )

    return {
        "canonical_label": canon,
        "slug": slug,
        "history_source": "local_sqlite_only",
        "deprecated_prematch_internet_sync_removed": True,
        "get_cached_match_count_before": cached_before,
        "rows_loaded_local_db_xi": int(snap.get("xi_row_count") or 0),
        "distinct_matches_local_db": int(snap.get("distinct_match_count") or 0),
        "prior_season_matches_local_db": int(snap.get("prior_season_match_count") or 0),
        "team_match_summary_rows_local_db": int(snap.get("team_match_summary_row_count") or 0),
        "history_sufficient_local": hist_ok,
        "recent_sufficient_local": recent_ok,
        "prior_season_sufficient_local": prior_ok,
        "stale_local_cache": stale,
        "days_since_newest_created": snap.get("days_since_newest_created"),
        "local_history_notes": notes,
        "squad_vs_history_match_report": squad_report,
        "player_alias_resolution_warnings": alias_warns,
        "get_cached_match_count_after": cached_before,
        "history_xi_rows_used": int(snap.get("xi_row_count") or 0),
        "history_rows_used": int(snap.get("xi_row_count") or 0),
        "errors": [],
        "raw_stage1_near_empty": near_empty,
        "warnings": list(notes) + alias_warns[:8],
        "pipeline": [
            "1_ingest_stage_loads_sqlite_separately",
            "2_read_local_sqlite_franchise_snapshot",
            "3_optional_squad_vs_history_join",
            "4_predict_using_stored_rows_only",
        ],
    }


def ensure_team_history_for_prediction(
    canonical_label: str,
    *,
    session: Optional[requests.Session] = None,
    opponent_canonical: Optional[str] = None,
    squad_player_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Backward-compatible name: **no longer touches the network**. Same as
    ``local_history_debug_for_prediction`` (``session`` / ``opponent_canonical`` ignored).
    """
    _ = session
    _ = opponent_canonical
    return local_history_debug_for_prediction(canonical_label, squad_player_names=squad_player_names)


def sync_team_history_if_needed(
    team_name: str,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """
    DEPRECATED: replaced by ingest → derive → predict. No pre-match network sync.
    Returns ``local_history_debug_for_prediction`` only.
    """
    _ = session
    canon = ipl_teams.canonical_franchise_label(team_name) or (team_name or "").strip()
    return local_history_debug_for_prediction(canon, squad_player_names=None)


def failsafe_history_debug(canonical_label: str, exc: Optional[Exception] = None) -> dict[str, Any]:
    """Minimal snapshot when local history read fails mid-prediction."""
    canon = (canonical_label or "").strip()
    snap = db.franchise_history_snapshot(canon)
    cached = int(snap.get("distinct_match_count") or 0)
    block: dict[str, Any] = {
        "canonical_label": canon,
        "failsafe": True,
        "history_source": "local_sqlite_only",
        "get_cached_match_count": cached,
        "sample_recent_matches": db.franchise_recent_match_summaries(canon, limit=5),
        "distinct_matches_local_db": int(snap.get("distinct_match_count") or 0),
        "rows_loaded_local_db_xi": int(snap.get("xi_row_count") or 0),
        "raw_stage1_near_empty": raw_stage1_tables_near_empty_from_snapshot(snap),
        **snap,
    }
    if exc is not None:
        block["exception"] = f"{type(exc).__name__}: {exc}"
    return block


def debug_local_snapshot(team_name: str) -> dict[str, Any]:
    """Read-only snapshot for logging/UI without network."""
    canon = ipl_teams.canonical_franchise_label(team_name) or (team_name or "").strip()
    snap = db.franchise_history_snapshot(canon)
    min_r = int(getattr(config, "HISTORY_SYNC_MIN_RECENT_MATCHES", 5))
    min_p = int(getattr(config, "HISTORY_SYNC_MIN_PRIOR_SEASON_MATCHES", 2))
    hist_ok, recent_ok, prior_ok, stale = _snapshot_sufficiency(snap, min_recent=min_r, min_prior=min_p)
    return {
        "canonical_label": canon,
        "slug": ipl_teams.slug_for_canonical_label(canon),
        **snap,
        "history_sufficient_local": hist_ok,
        "recent_sufficient_local": recent_ok,
        "prior_season_sufficient_local": prior_ok,
    }
