"""Playing XI, subs, toss effects, and win probability from multi-perspective scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, FrozenSet, Optional

import audit_profile
import canonical_keys
import config
import db
import history_rules
import history_sync
import history_xi
import impact_subs_engine
import ipl_squad
import ipl_teams
import learner
import player_alias_resolve
import time_utils
import win_probability_engine
from history_context import HistoryContext, build_history_context, venue_lookup_keys
from ipl_squad import (
    ALL_ROUNDER,
    BATTER,
    BOWLER,
    WK_BATTER,
    role_bucket_to_predictor_role,
)
from player_role_classifier import classify_player, role_counts
import rules_xi
from venues import VenueProfile, venue_conditions_summary

logger = logging.getLogger(__name__)
_perf_logger = logging.getLogger("ipl_predictor.perf")
_CURATED_PLAYER_META_FALLBACK: Optional[dict[str, dict[str, Any]]] = None
_CURATED_PLAYER_META_FALLBACK_MTIME: Optional[tuple[float, float]] = None


def _load_curated_marquee_overrides() -> dict[str, dict[str, Any]]:
    raw_path = str(getattr(config, "PLAYER_MARQUEE_OVERRIDES_PATH", "") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / raw_path
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("marquee_overrides: failed reading %s (%s)", path, exc)
        return {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict) and isinstance(raw.get("players"), dict):
        src = raw.get("players") or {}
    elif isinstance(raw, dict):
        src = raw
    else:
        src = {}
    for k, v in src.items():
        nk = learner.normalize_player_key(str(k or ""))
        if not nk or not isinstance(v, dict):
            continue
        tier = str(v.get("marquee_tier") or "").strip().lower()
        if tier not in ("tier_1", "tier_2", "tier_3"):
            continue
        out[nk] = {
            "marquee_tier": tier,
            "marquee_reason": str(v.get("marquee_reason") or "curated_override").strip() or "curated_override",
        }
    return out


def _load_curated_player_metadata_fallback() -> dict[str, dict[str, Any]]:
    """
    Lightweight JSON fallback for runtime metadata propagation.
    Priority: manual curated > cricinfo curated (manual fixes override/supplement).
    """
    global _CURATED_PLAYER_META_FALLBACK
    global _CURATED_PLAYER_META_FALLBACK_MTIME
    curated_path_s = str(getattr(config, "PLAYER_METADATA_CURATED_PATH", "") or "").strip()
    cricinfo_path_s = str(getattr(config, "PLAYER_METADATA_CRICINFO_PATH", "") or "").strip()
    curated_p = Path(curated_path_s) if curated_path_s else None
    cricinfo_p = Path(cricinfo_path_s) if cricinfo_path_s else None
    if curated_p and not curated_p.is_absolute():
        curated_p = Path(__file__).resolve().parent / curated_path_s
    if cricinfo_p and not cricinfo_p.is_absolute():
        cricinfo_p = Path(__file__).resolve().parent / cricinfo_path_s

    mt_a = curated_p.stat().st_mtime if curated_p and curated_p.is_file() else 0.0
    mt_b = cricinfo_p.stat().st_mtime if cricinfo_p and cricinfo_p.is_file() else 0.0
    cur_mtime = (float(mt_a), float(mt_b))
    if _CURATED_PLAYER_META_FALLBACK is not None and _CURATED_PLAYER_META_FALLBACK_MTIME == cur_mtime:
        return _CURATED_PLAYER_META_FALLBACK

    out: dict[str, dict[str, Any]] = {}
    # Load cricinfo first, then overlay manual curated (higher priority).
    for path in (cricinfo_p, curated_p):
        if not path:
            continue
        if not path.exists():
            continue
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("player_metadata_fallback: failed reading %s", path)
            continue
        # Support both dict keyed by player_key and list[rows].
        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, list):
            items = ((str(x.get("player_name") or x.get("player_key") or ""), x) for x in raw if isinstance(x, dict))
        else:
            items = []
        for k, v in items:
            if not isinstance(v, dict):
                continue
            nk = learner.normalize_player_key(str(k or ""))
            if not nk:
                nk = learner.normalize_player_key(str(v.get("player_name") or v.get("display_name") or ""))
            if not nk:
                continue
            out[nk] = dict(v)
    _CURATED_PLAYER_META_FALLBACK = out
    _CURATED_PLAYER_META_FALLBACK_MTIME = cur_mtime
    return out


def _slim_prediction_layer_debug(pld: dict[str, Any]) -> dict[str, Any]:
    """Drop the largest lists (full bench impact + per-player history usage) for lighter JSON/UI."""
    slimmed: dict[str, Any] = {
        k: v for k, v in (pld or {}).items() if k not in ("team_a", "team_b")
    }
    for side in ("team_a", "team_b"):
        block = dict(pld.get(side) or {})
        block["impact_sub_ranking"] = []
        block["history_usage_per_player"] = []
        block["_light_debug_omitted_large_lists"] = True
        slimmed[side] = block
    return slimmed


def _metadata_dependency_report(
    *,
    xi_a: list[SquadPlayer],
    xi_b: list[SquadPlayer],
) -> dict[str, Any]:
    def _json_status(raw_path: str) -> dict[str, Any]:
        rp = str(raw_path or "").strip()
        if not rp:
            return {"active": False, "exists": False, "path": "", "keys": 0}
        p = Path(rp)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / rp
        if not p.is_file():
            return {"active": False, "exists": bool(p.exists()), "path": str(p), "keys": 0}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"active": False, "exists": True, "path": str(p), "keys": 0}
        if isinstance(raw, dict):
            keys = len(raw.get("players") or {}) if isinstance(raw.get("players"), dict) else len(raw)
        elif isinstance(raw, list):
            keys = len(raw)
        else:
            keys = 0
        return {"active": keys > 0, "exists": True, "path": str(p), "keys": int(keys)}

    def _xi_meta_sources(xi: list[SquadPlayer]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in xi or []:
            hd = getattr(p, "history_debug", None) or {}
            meta = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
            out[p.name] = {
                "runtime_source": hd.get("player_metadata_source_runtime") or "",
                "metadata_source": (meta.get("source") if isinstance(meta, dict) else "") or "",
                "metadata_lookup_key": hd.get("metadata_lookup_key") or "",
            }
        return out

    marquee = _json_status(str(getattr(config, "PLAYER_MARQUEE_OVERRIDES_PATH", "") or ""))
    curated = _json_status(str(getattr(config, "PLAYER_METADATA_CURATED_PATH", "") or ""))
    cricinfo = _json_status(str(getattr(config, "PLAYER_METADATA_CRICINFO_PATH", "") or ""))
    alias = _json_status(str(getattr(config, "PLAYER_ALIAS_OVERRIDES_PATH", "") or ""))
    alias["active"] = bool(alias.get("active") and player_alias_resolve.alias_overrides_active())
    # Predictor loads + applies marquee overrides via _load_curated_marquee_overrides().
    marquee["active"] = bool(marquee.get("active") and _load_curated_marquee_overrides())

    return {
        "alias_overrides": alias,
        "marquee_overrides": marquee,
        "cricinfo_metadata_json": cricinfo,
        "curated_metadata_json": curated,
        "team_a_xi_metadata_sources": _xi_meta_sources(xi_a),
        "team_b_xi_metadata_sources": _xi_meta_sources(xi_b),
    }

# Toss keys for UI ↔ engine (Team B bowls first ≡ Team A bats first in T20).
TOSS_SCENARIO_OPTIONS: tuple[tuple[str, str], ...] = (
    ("unknown", "Unknown toss"),
    ("a_bats_first", "Team A bats first"),
    ("a_bowls_first", "Team A bowls first"),
    ("b_bats_first", "Team B bats first"),
    ("b_bowls_first", "Team B bowls first"),
)


def resolve_a_bats_first_toss(toss_key: str) -> Optional[bool]:
    """Return True if Team A bats first, False if Team B bats first, None if unknown."""
    k = (toss_key or "unknown").strip().lower()
    if k in ("unknown", "", "none"):
        return None
    if k in ("a_bats_first", "b_bowls_first"):
        return True
    if k in ("a_bowls_first", "b_bats_first"):
        return False
    return None


def franchise_xi_scenario_branch(for_team_a: bool, a_bats_first: Optional[bool]) -> str:
    """SQLite history_xi scenario key used to rank this franchise's XI."""
    if a_bats_first is None:
        return "if_team_bats_first"
    a_first = bool(a_bats_first)
    this_bats_first = a_first if for_team_a else not a_first
    return "if_team_bats_first" if this_bats_first else "if_team_bowls_first"


def _franchise_name_match(a: str, b: str) -> bool:
    na, nb = learner.normalize_player_key(a), learner.normalize_player_key(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def chase_context_for_team(
    team_name: str,
    team_a_name: str,
    team_b_name: str,
    a_bats_first: Optional[bool],
) -> Optional[bool]:
    """
    True if this team is expected to **chase** (bat second); False if bat first; None if toss unknown.
    """
    if a_bats_first is None:
        return None
    if _franchise_name_match(team_name, team_a_name):
        return not a_bats_first
    if _franchise_name_match(team_name, team_b_name):
        return a_bats_first
    return None


def _batting_order_diagnostic_source(p: SquadPlayer) -> str:
    """
    Coarse UI label: direct_history | prior_season | h2h_history | role_fallback.
    Derived from stored scorecard EMA path (``batting_order_source``) and unknown-slot sentinel.
    """
    hd = getattr(p, "history_debug", None) or {}
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    ema = float(getattr(p, "history_batting_ema", unk))
    src = str(hd.get("batting_order_source") or "")
    if ema >= unk - 1e-6:
        return "role_fallback"
    low = src.lower()
    if "h2h" in low:
        return "h2h_history"
    if "prior_season" in src:
        return "prior_season"
    return "direct_history"


def _batting_order_sources_summary(xi: list[SquadPlayer]) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for p in xi:
        c[_batting_order_diagnostic_source(p)] += 1
    return dict(c)


def _batting_order_team_lineup_summary(xi: list[SquadPlayer]) -> str:
    if not xi:
        return "No XI to summarise."
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    anchored = sum(1 for p in xi if float(getattr(p, "history_batting_ema", unk)) < unk - 1e-6)
    derive_touch = sum(
        1
        for p in xi
        if isinstance((getattr(p, "history_debug", None) or {}).get("derive_player_profile"), dict)
    )
    return (
        f"{anchored}/{len(xi)} batters anchored on scorecard EMA; "
        f"{derive_touch}/{len(xi)} picked up Stage-2 batting-role hints; "
        "remainder ordered via role-bucket fallback when EMA is missing."
    )


def _xi_batting_order_diagnostics_rows(
    xi: list[SquadPlayer], order: list[str]
) -> list[dict[str, Any]]:
    by_name = {p.name: p for p in xi}
    rows: list[dict[str, Any]] = []
    for i, n in enumerate(order):
        p = by_name.get(n)
        if p is None:
            continue
        hd = getattr(p, "history_debug", None) or {}
        rows.append(
            {
                "name": n,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "batting_order_position": i + 1,
                "batting_positions_history": hd.get("batting_positions_history"),
                "batting_position_ema": hd.get("batting_slot_ema"),
                "batting_order_source": hd.get("batting_order_source"),
                "batting_order_diagnostic_source": hd.get("batting_order_diagnostic_source"),
                "batting_order_final": hd.get("batting_order_final"),
                "batting_order_reason_summary": hd.get("batting_order_reason_summary"),
                "role_band": hd.get("role_band"),
                "batting_position_history_basis": hd.get("batting_position_history_basis"),
                "final_order_reason": hd.get("final_order_reason"),
                "batting_order_signal_source_ranked": hd.get("batting_order_signal_source_ranked"),
            }
        )
    return rows


@dataclass
class SquadPlayer:
    name: str
    role: str  # bat, bowl, all, wk — scoring / history
    is_overseas: bool
    player_key: str = ""  # canonical identity (truncated); mirrors ``canonical_player_key``
    team_display_name: str = ""
    canonical_team_key: str = ""
    canonical_player_key: str = ""  # same normalization as ``player_key`` when populated
    role_bucket: str = "Batter"  # IPL: Batter, WK-Batter, All-Rounder, Bowler
    bat_skill: float = 0.55
    bowl_skill: float = 0.45
    is_wicketkeeper: bool = False
    batting_roles: list[str] = field(default_factory=list)
    bowling_type: Optional[str] = None
    is_opener_candidate: bool = False
    is_finisher_candidate: bool = False
    history_xi_score: float = 0.0
    history_batting_ema: float = 99.0
    history_debug: dict[str, Any] = field(default_factory=dict)
    selection_score: float = 0.0
    perspectives: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    history_delta: float = 0.0
    history_notes: list[str] = field(default_factory=list)


def _annotate_squad_canonical_keys(squad: list[SquadPlayer], franchise_canonical_label: str) -> None:
    lab = (franchise_canonical_label or "").strip()
    ctk = ipl_teams.canonical_team_key_for_franchise(lab)
    for p in squad:
        p.team_display_name = lab
        p.canonical_team_key = ctk
        pk = (p.player_key or "").strip() or learner.normalize_player_key(p.name)
        p.player_key = pk
        p.canonical_player_key = pk


def _micro_noise(name: str) -> float:
    """Deterministic tiny tie-breaker; not a popularity prior (stable across Python runs)."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10007 / 10007.0) * 0.012


def _role_skills(role: str) -> tuple[float, float, bool]:
    r = role.lower().strip()
    if r == "wk":
        return 0.62, 0.22, True
    if r == "bowl":
        return 0.30, 0.84, False
    if r == "all":
        return 0.58, 0.62, False
    if r == "bat":
        return 0.80, 0.24, False
    return 0.55, 0.40, False


def _squad_player_from_ipl(
    name: str,
    role_bucket: str,
    *,
    overseas: bool = False,
) -> Optional[SquadPlayer]:
    ok, _reason = ipl_squad.validate_clean_name(name)
    if not ok:
        return None
    pk = learner.normalize_player_key(name)
    role = role_bucket_to_predictor_role(role_bucket)
    bat, bowl, is_wk = _role_skills(role)
    bat += _micro_noise(name)
    bowl += _micro_noise(name[::-1])
    br = ipl_squad.default_batting_roles_for_bucket(
        role_bucket, is_keeper=(role_bucket == WK_BATTER)
    )
    return SquadPlayer(
        name=name,
        role=role,
        role_bucket=role_bucket,
        is_overseas=overseas,
        player_key=pk,
        canonical_player_key=pk,
        bat_skill=max(0.05, min(0.98, bat)),
        bowl_skill=max(0.05, min(0.98, bowl)),
        is_wicketkeeper=is_wk or (role_bucket == WK_BATTER),
        batting_roles=list(br),
        bowling_type=None,
    )


def _log_role_bucket_distribution(squad: list[SquadPlayer], tag: str) -> dict[str, int]:
    from collections import Counter

    c = Counter(p.role_bucket for p in squad)
    dist = dict(c)
    logger.info("role_bucket_distribution %s: %s (n=%d)", tag, dist, len(squad))
    ar = c.get(ALL_ROUNDER, 0)
    if len(squad) >= 12 and ar > max(7, int(len(squad) * 0.32)):
        logger.warning(
            "%s: unusually high All-Rounder count=%d of %d — check IPL source roles",
            tag,
            ar,
            len(squad),
        )
    return dist


def _squad_player_key_set(squad: list[SquadPlayer]) -> set[str]:
    """Normalized keys for every player in the current fetched squad."""
    return {learner.normalize_player_key(p.name) for p in squad if learner.normalize_player_key(p.name)}


def _validate_xi_in_current_squad(
    xi: list[SquadPlayer], squad: list[SquadPlayer], side: str
) -> tuple[bool, list[str]]:
    """Hard gate: XI must be a subset of the current structured squad (inner-join semantics)."""
    sk = _squad_player_key_set(squad)
    bad: list[str] = []
    for p in xi:
        pk = (getattr(p, "player_key", None) or "").strip() or learner.normalize_player_key(p.name)
        if pk not in sk:
            bad.append(p.name)
    logger.info(
        "%s strict XI⊆squad: squad_keys=%d xi=%d subset_ok=%s xi_names=%s offenders=%s",
        side,
        len(sk),
        len(xi),
        len(bad) == 0,
        [p.name for p in xi],
        bad,
    )
    return (len(bad) == 0, bad)


def _batting_order_strict_names_for_xi(
    xi: list[SquadPlayer], candidate: list[str]
) -> tuple[list[str], list[str]]:
    """
    Batting order may only reference **selected XI** names (no historical or foreign names).

    Returns ``(order, warnings)``. Missing XI members are appended using bat_skill order.
    """
    xi_set = {p.name for p in xi}
    warnings: list[str] = []
    seen: set[str] = set()
    out: list[str] = []
    for n in candidate:
        if n in xi_set and n not in seen:
            out.append(n)
            seen.add(n)
    dropped = [n for n in candidate if n not in xi_set]
    if dropped:
        msg = (
            f"Batting order ignored {len(dropped)} name(s) not in the selected XI "
            f"(strict scope): {', '.join(dropped[:8])}"
            + (" …" if len(dropped) > 8 else "")
        )
        warnings.append(msg)
        logger.warning("batting_order strict: dropped non-XI names team context dropped=%s", dropped)
    missing = [p.name for p in xi if p.name not in seen]
    if missing:
        rest = sorted(
            [p for p in xi if p.name in missing],
            key=lambda q: (-q.bat_skill, q.name),
        )
        for p in rest:
            out.append(p.name)
            seen.add(p.name)
        warnings.append(
            "Batting order was repaired to include every XI player (by bat_skill): "
            + ", ".join(missing[:10])
            + (" …" if len(missing) > 10 else "")
        )
    if len(xi) >= 11 and len(out) != 11:
        warnings.append(f"Batting order length {len(out)} after strict XI repair (expected 11).")
    return out, warnings


def _collect_strict_validation_warnings(
    *,
    team_a_label: str,
    team_b_label: str,
    xi_a: list[SquadPlayer],
    xi_b: list[SquadPlayer],
    scored_squad_a: list[SquadPlayer],
    scored_squad_b: list[SquadPlayer],
    history_sync_debug: dict[str, Any],
    batting_order_warnings: list[str],
) -> list[str]:
    """User-visible warnings: sparse local SQLite history, batting-order repairs."""
    out: list[str] = []
    thr = int(getattr(config, "HISTORY_VALIDATION_SPARSE_ROWS_WARN", 2))
    frac = float(getattr(config, "HISTORY_VALIDATION_SPARSE_XI_FRACTION", 0.35))
    db_thr = int(getattr(config, "HISTORY_VALIDATION_SPARSE_REQUIRE_DB_MATCHES_BELOW", 14))

    def _franchise_db_matches(key: str) -> int:
        block = (history_sync_debug or {}).get(key) or {}
        return int(block.get("distinct_matches_local_db") or 0)

    def sparse_block(label: str, xi: list[SquadPlayer], side_key: str) -> None:
        if not xi:
            return
        block = (history_sync_debug or {}).get(side_key) or {}
        if not history_sync.raw_stage1_tables_near_empty(block):
            return
        db_n = _franchise_db_matches(side_key)
        if db_n >= db_thr:
            return
        sparse_n = sum(
            1
            for p in xi
            if int((getattr(p, "history_debug", None) or {}).get("history_rows_found") or 0) < thr
        )
        if sparse_n / max(1, len(xi)) >= frac:
            out.append(
                f"{label}: Thin SQLite history — {db_n} distinct stored matches for this franchise "
                f"(warning threshold: <{db_thr} matches), and {sparse_n}/{len(xi)} XI players have fewer than "
                f"{thr} team_match_xi rows. {history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )

    sparse_block(team_a_label, xi_a, "team_a")
    sparse_block(team_b_label, xi_b, "team_b")

    for label, squad in (
        (team_a_label, scored_squad_a),
        (team_b_label, scored_squad_b),
    ):
        if not squad:
            continue
        summ = (getattr(squad[0], "history_debug", None) or {}).get("history_linkage_team_summary") or {}
        health = (summ or {}).get("stage_f_team_health") or ""
        detail = (summ or {}).get("stage_f_team_health_detail") or {}
        if health == "major_linkage_failure":
            out.append(
                f"{label}: Stage F major_linkage_failure — unresolved core players "
                f"{detail.get('core_unresolved', '?')}/{detail.get('core_players', '?')} "
                f"(Batter/WK/AR). Inspect history_usage_per_player and resolution_layer_debug."
            )
        elif health == "partial_linkage_issue":
            out.append(
                f"{label}: Stage F partial_linkage_issue — {detail.get('core_unresolved', '?')} core "
                "player(s) still lack a SQLite key; others linked OK."
            )
        elif health == "healthy_with_sparse_new_players":
            out.append(
                f"{label}: Stage F healthy_with_sparse_new_players — linkage healthy; remaining gaps "
                "skew toward newer/non-core squad names."
            )

    def batting_slot_coverage_warn(label: str, xi: list[SquadPlayer], side_key: str) -> None:
        if len(xi) != 11:
            return
        block = (history_sync_debug or {}).get(side_key) or {}
        if not history_sync.raw_stage1_tables_near_empty(block):
            return
        n_missing = 0
        for p in xi:
            hd = getattr(p, "history_debug", None) or {}
            n_slot = int(
                hd.get("batting_position_rows_found")
                or hd.get("batting_positions_found")
                or (hd.get("history_usage_debug") or {}).get("batting_position_rows_found")
                or 0
            )
            if n_slot < 1:
                n_missing += 1
        if n_missing / 11.0 > 0.5:
            out.append(
                f"{label}: {n_missing}/11 XI players have no stored batting-slot rows in SQLite. "
                f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )

    batting_slot_coverage_warn(team_a_label, xi_a, "team_a")
    batting_slot_coverage_warn(team_b_label, xi_b, "team_b")

    def alias_resolution_core_warn(label: str, xi: list[SquadPlayer], squad_scored: list[SquadPlayer]) -> None:
        """Only escalate when Stage F health is already partial/major (avoid noisy warnings)."""
        if not xi or not squad_scored:
            return
        summ = (getattr(squad_scored[0], "history_debug", None) or {}).get("history_linkage_team_summary") or {}
        health = summ.get("stage_f_team_health") or ""
        if health not in ("major_linkage_failure", "partial_linkage_issue"):
            return
        core_roles = (BATTER, WK_BATTER, ALL_ROUNDER)
        bad: list[str] = []
        for p in xi:
            if p.role_bucket not in core_roles:
                continue
            hd = getattr(p, "history_debug", None) or {}
            lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
            rt = lk.get("resolution_type")
            n_tmx = int(lk.get("team_match_xi_rows", 0) or 0)
            rolled = lk.get("rolled_up_interpretation") or ""
            if rt == "ambiguous_alias":
                bad.append(f"{p.name} (ambiguous_alias / {rolled})")
            elif rt == "no_match" and n_tmx == 0:
                if bool(hd.get("valid_current_squad_new_to_franchise")) and bool(
                    hd.get("global_ipl_history_presence")
                ):
                    continue
                bad.append(f"{p.name} (no_match / {rolled})")
        if bad:
            out.append(
                f"{label}: Core players still unresolved under Stage F ({health}) — "
                + ", ".join(bad[:8])
                + (" …" if len(bad) > 8 else "")
            )

    alias_resolution_core_warn(team_a_label, xi_a, scored_squad_a)
    alias_resolution_core_warn(team_b_label, xi_b, scored_squad_b)

    def wrong_side_squad_warn(label: str, squad: list[SquadPlayer]) -> None:
        leaked = [
            p.name
            for p in squad
            if bool((getattr(p, "history_debug", None) or {}).get("wrong_side_squad_assignment"))
        ]
        if leaked:
            out.append(
                f"{label}: Wrong-side or stale squad leak suspected (player in opposite official fetch "
                f"or stale cache vs selected franchise): {', '.join(leaked[:14])}"
                + (" …" if len(leaked) > 14 else "")
            )

    wrong_side_squad_warn(team_a_label, scored_squad_a)
    wrong_side_squad_warn(team_b_label, scored_squad_b)

    def new_to_franchise_note(label: str, squad: list[SquadPlayer]) -> None:
        names = [
            p.name
            for p in squad
            if bool((getattr(p, "history_debug", None) or {}).get("valid_current_squad_new_to_franchise"))
        ]
        if names:
            out.append(
                f"{label}: Current squad player(s) new to franchise in stored history; "
                f"global IPL fallback prior applied: {', '.join(names[:10])}"
                + (" …" if len(names) > 10 else "")
            )

    new_to_franchise_note(team_a_label, scored_squad_a)
    new_to_franchise_note(team_b_label, scored_squad_b)

    for label, key in ((team_a_label, "team_a"), (team_b_label, "team_b")):
        block = (history_sync_debug or {}).get(key) or {}
        if block.get("failsafe"):
            out.append(
                f"{label}: Local SQLite history snapshot failed; continuing with heuristics. "
                f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )
            continue
        for note in (block.get("local_history_notes") or [])[:4]:
            if isinstance(note, str) and note.strip():
                out.append(f"{label}: {note.strip()}")

    out.extend(batting_order_warnings)
    return out


def parse_squad_text(text: str) -> list[SquadPlayer]:
    """
    IPL squad lines:

    - Preferred: ``Name | Batter`` or ``Name | WK-Batter | overseas``
    - Legacy: ``Name, bat`` / ``Name, bowl`` / ``Name, wk`` / ``Name, all`` (+ optional overseas)
    """
    out: list[SquadPlayer] = []
    seen_player_keys: set[str] = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if not parts:
                continue
            base, emb = ipl_squad.split_embedded_role_from_name(parts[0])
            name = (base or parts[0]).strip()
            role_bucket = BATTER
            if len(parts) >= 2:
                nb = ipl_squad.normalize_role_bucket_label(parts[1])
                if nb:
                    role_bucket = nb
                else:
                    low = parts[1].lower().replace("_", " ")
                    if low in ("bat", "batsman", "batting"):
                        role_bucket = BATTER
                    elif low in ("bowl", "bowler", "bowling"):
                        role_bucket = BOWLER
                    elif low in ("wk", "keeper", "wk-batter", "wk batter", "wkbat"):
                        role_bucket = WK_BATTER
                    elif low in ("all", "allrounder", "all-rounder", "all rounder", "ar"):
                        role_bucket = ALL_ROUNDER
            if emb:
                role_bucket = emb
            overseas = any(p.lower() in ("overseas", "os", "foreign", "int", "intl") for p in parts[2:])
            sp = _squad_player_from_ipl(name, role_bucket, overseas=overseas)
            if sp and sp.player_key and sp.player_key not in seen_player_keys:
                seen_player_keys.add(sp.player_key)
                out.append(sp)
            continue

        parts = [p.strip() for p in re.split(r"[,|]", line) if p.strip()]
        if not parts:
            continue
        base, emb = ipl_squad.split_embedded_role_from_name(parts[0])
        name = (base or parts[0]).strip()
        role_bucket = BATTER
        overseas = False
        for p in parts[1:]:
            low = p.lower()
            if low in ("bat", "batsman", "batting", "batter"):
                role_bucket = BATTER
            elif low in ("bowl", "bowler", "bowling"):
                role_bucket = BOWLER
            elif low in ("all", "ar", "allrounder", "all-rounder"):
                role_bucket = ALL_ROUNDER
            elif low in ("wk", "keeper", "wicketkeeper", "wkbat", "wk-batter"):
                role_bucket = WK_BATTER
            elif low in ("overseas", "os", "foreign", "int", "intl"):
                overseas = True
        if emb:
            role_bucket = emb
        sp = _squad_player_from_ipl(name, role_bucket, overseas=overseas)
        if sp and sp.player_key and sp.player_key not in seen_player_keys:
            seen_player_keys.add(sp.player_key)
            out.append(sp)

    logger.info("parse_squad_text: loaded %d structured players (deduped by player_key)", len(out))
    return out


def _normalize_name_set(text: str) -> set[str]:
    return {learner.normalize_player_key(x) for x in text.splitlines() if x.strip()}


def _expanded_unavailable_keys(text: str) -> set[str]:
    """
    Expand unavailable names through explicit alias overrides so one unavailable entry blocks
    display-name, player_key, and curated alias variants before any XI/sub selection starts.
    """
    banned = _normalize_name_set(text)
    if not banned:
        return set()
    expanded: set[str] = set()
    for key in banned:
        if not key:
            continue
        expanded.add(key)
        canon = player_alias_resolve.canonicalize_via_alias_overrides(key)
        if canon:
            expanded.add(canon)
        alias_candidates = getattr(player_alias_resolve, "_alias_override_candidates", None)
        if callable(alias_candidates):
            for cand in alias_candidates(key) or []:
                if cand:
                    expanded.add(str(cand).strip().lower())
            if canon:
                for cand in alias_candidates(canon) or []:
                    if cand:
                        expanded.add(str(cand).strip().lower())
    return expanded


def filter_unavailable(squad: list[SquadPlayer], unavailable_blob: str) -> list[SquadPlayer]:
    banned = _expanded_unavailable_keys(unavailable_blob)
    if not banned:
        return list(squad)
    out: list[SquadPlayer] = []
    for p in squad:
        keys = {
            learner.normalize_player_key(p.name),
            learner.normalize_player_key(getattr(p, "player_key", "") or ""),
        }
        keys = {k for k in keys if k}
        if not keys:
            continue
        expanded_keys: set[str] = set(keys)
        alias_candidates = getattr(player_alias_resolve, "_alias_override_candidates", None)
        for key in list(keys):
            canon = player_alias_resolve.canonicalize_via_alias_overrides(key)
            if canon:
                expanded_keys.add(canon)
            if callable(alias_candidates):
                for cand in alias_candidates(key) or []:
                    if cand:
                        expanded_keys.add(str(cand).strip().lower())
                if canon:
                    for cand in alias_candidates(canon) or []:
                        if cand:
                            expanded_keys.add(str(cand).strip().lower())
        if expanded_keys & banned:
            continue
        out.append(p)
    return out


def _squad_shape(squad: list[SquadPlayer]) -> dict[str, float]:
    n = max(1, len(squad))
    wk = sum(1 for p in squad if p.is_wicketkeeper)
    bow = sum(
        1
        for p in squad
        if p.role_bucket in (BOWLER, ALL_ROUNDER)
        or p.role in ("bowl", "all")
        or p.bowl_skill >= 0.55
    )
    bat = sum(
        1
        for p in squad
        if p.role_bucket in (BATTER, WK_BATTER)
        or p.role == "bat"
        or p.bat_skill >= 0.65
    )
    os = sum(1 for p in squad if p.is_overseas)
    return {
        "wk_density": wk / n,
        "bowl_depth": bow / n,
        "bat_depth": bat / n,
        "overseas_density": os / n,
    }


def _score_player(
    p: SquadPlayer,
    *,
    self_shape: dict[str, float],
    opp_shape: dict[str, float],
    conditions: dict[str, Any],
    learned_map: dict[str, dict[str, Any]],
    franchise_canonical: Optional[str] = None,
) -> SquadPlayer:
    bf = float(conditions["batting_friendliness"])
    pace_bias = float(conditions["pace_bias"])
    dew = float(conditions["dew_risk"])
    swing = float(conditions["swing_seam_proxy"])
    spin_f = float(conditions["spin_friendliness"])
    rain = float(conditions["rain_disruption_risk"])
    boundary = float(conditions["boundary_size"])

    learned = learner.learned_boost_for_player(p.name, learned_map)

    # Coach: fit to venue + squad balance needs
    need_bowling = max(0.0, 0.55 - self_shape["bowl_depth"])
    need_bat = max(0.0, 0.52 - self_shape["bat_depth"])
    coach = (
        0.34 * (p.bat_skill * bf + p.bowl_skill * (1.0 - bf))
        + 0.22 * (p.bowl_skill * need_bowling * 2.2 + p.bat_skill * need_bat * 2.2)
        + 0.18 * (p.is_wicketkeeper * 0.25 + (1.0 if p.is_wicketkeeper else 0.0) * (0.65 - self_shape["wk_density"]))
        + 0.16 * (p.bowl_skill * pace_bias * swing + p.bowl_skill * (1.0 - pace_bias) * spin_f)
        + 0.10 * (1.0 - rain) * (p.bat_skill * 0.55 + p.bowl_skill * 0.45)
    )
    coach = max(0.0, min(1.0, coach))

    # Player (self): matchup vs opposition squad shape
    player = (
        0.45 * (p.bat_skill * (0.55 + opp_shape["bowl_depth"]) + p.bowl_skill * (0.45 + opp_shape["bat_depth"]))
        + 0.25 * (p.bowl_skill * swing * pace_bias + p.bat_skill * (1.0 - swing) * 0.35)
        + 0.18 * (p.bat_skill * (1.0 - boundary) * 0.4 + p.bowl_skill * boundary * 0.35)
        + 0.12 * (p.bowl_skill * dew * 0.45)
    )
    player = max(0.0, min(1.0, player))

    # Analyst: conditions-heavy read
    analyst = (
        0.30 * (p.bat_skill * bf + p.bowl_skill * (1.1 - bf) * 0.85)
        + 0.22 * (p.bowl_skill * (spin_f * (1.0 - pace_bias) + swing * pace_bias))
        + 0.20 * (p.bat_skill * (1.0 - 0.55 * rain))
        + 0.16 * (p.bowl_skill * (0.5 + 0.5 * dew))
        + 0.12 * learned
    )
    analyst = max(0.0, min(1.0, analyst))

    # Opposition: how much the opponent would dislike this pick (mirror shapes)
    opposition = (
        0.40 * (p.bowl_skill * (0.45 + opp_shape["bat_depth"]) + p.bat_skill * (0.5 + opp_shape["bowl_depth"]))
        + 0.28 * (p.bat_skill * pace_bias * (1.0 - boundary) + p.bowl_skill * boundary)
        + 0.18 * (p.bowl_skill * (1.0 - opp_shape["bowl_depth"]) * 1.15)
        + 0.14 * (p.bat_skill * spin_f * 0.45 + p.bowl_skill * (1.0 - spin_f) * 0.35)
    )
    opposition = max(0.0, min(1.0, opposition))

    composite = (
        config.WEIGHT_COACH * coach
        + config.WEIGHT_PLAYER * player
        + config.WEIGHT_ANALYST * analyst
        + config.WEIGHT_OPPOSITION * opposition
        + config.WEIGHT_LEARNED * learned
    )
    p.perspectives = {
        "coach": coach,
        "player": player,
        "analyst": analyst,
        "opposition": opposition,
        "learned": learned,
    }
    p.composite = composite
    if franchise_canonical:
        lab = ipl_teams.canonical_franchise_label(franchise_canonical) or franchise_canonical.strip()
        fk = ipl_teams.canonical_team_key_for_franchise(lab)
        mfeat = db.get_player_franchise_features(learner.normalize_player_key(p.name), fk)
        if mfeat:
            agg = float(mfeat.get("batting_aggressor_score") or 0)
            ctrl = float(mfeat.get("bowling_control_score") or 0)
            spin_t = float(mfeat.get("vs_spin_tendency") or 0)
            mu = (
                0.012 * agg * float(p.bat_skill)
                + 0.012 * ctrl * float(p.bowl_skill)
                + 0.006 * spin_t * float(p.bat_skill)
            )
            p.composite = max(0.0, min(1.0, float(p.composite) + mu))
    return p


def _history_adjust_for_player(
    p: SquadPlayer,
    team_name: str,
    shape: dict[str, float],
    venue_keys: list[str],
    is_night: bool,
    dew_risk: float,
    ctx: HistoryContext,
) -> None:
    """Optional small composite bump from legacy history rules (disabled when history-primary XI)."""
    bump_scale = float(getattr(config, "HISTORY_COMPOSITE_HISTORY_BUMP_SCALE", 1.0))
    if getattr(config, "HISTORY_PRIMARY_XI_SELECTION", False) and bump_scale <= 0:
        p.history_delta = 0.0
        p.history_notes = []
        return
    pk = learner.normalize_player_key(p.name)
    tk = learner.normalize_player_key(team_name)[:80]
    proxy_slot = int(config.HISTORY_PROXY_BAT_SLOT.get(p.role, 6))

    t_xi = history_rules.xi_frequency_term(
        ctx.xi_by_player.get(pk, 0),
        max(1, ctx.max_xi_picks),
        ctx.db_match_count,
    )

    slot_row = ctx.avg_slot_by_player.get(pk)
    t_slot = history_rules.batting_slot_term(
        slot_row[0] if slot_row else None,
        proxy_slot,
        slot_row[1] if slot_row else 0,
    )

    bowl_row = ctx.bowl_balls_avg_by_player.get(pk)
    t_bowl = history_rules.bowling_usage_term(
        bowl_row[0] if bowl_row else 0.0,
        p.role,
        bowl_row[1] if bowl_row else 0,
    )

    picks_v, vtm = 0, 0
    for vk in venue_keys:
        key3 = (vk, tk, pk)
        if key3 in ctx.venue_team_player_xi:
            picks_v = ctx.venue_team_player_xi[key3]
            vtm = ctx.venue_team_matches.get((vk, tk), 0)
            break
    t_vt = history_rules.venue_team_xi_term(picks_v, vtm)

    os_guess = min(
        config.MAX_OVERSEAS,
        max(0, int(round(shape["overseas_density"] * 11 + 0.001))),
    )
    mix = ctx.pick_overseas_mix(venue_keys, tk)
    t_os = history_rules.overseas_mix_term(os_guess, mix)

    nx = ctx.night_xi_by_player.get(pk, 0)
    dx = ctx.day_xi_by_player.get(pk, 0)
    t_nd = history_rules.night_day_term(is_night, nx, dx)
    t_dew = history_rules.dew_context_term(dew_risk, nx, dx, p.role)

    terms = [t_xi, t_slot, t_bowl, t_vt, t_os, t_nd, t_dew]
    weights = [
        config.LEARN_WEIGHT_XI_FREQUENCY,
        config.LEARN_WEIGHT_BATTING_SLOT,
        config.LEARN_WEIGHT_BOWLING_USAGE,
        config.LEARN_WEIGHT_VENUE_TEAM_XI,
        config.LEARN_WEIGHT_OVERSEAS_MIX,
        config.LEARN_WEIGHT_DAY_NIGHT,
        config.LEARN_WEIGHT_DEW_CONTEXT,
    ]
    d, lines = history_rules.blend_history_deltas(terms, weights)
    p.history_delta = float(d * bump_scale)
    p.history_notes = lines[:8]
    p.composite = max(0.001, p.composite + float(config.HISTORY_BLEND_SCALE) * d * bump_scale)


def _resolve_chase_prior(
    venue_keys: list[str], ctx: HistoryContext
) -> tuple[float, int]:
    for vk in venue_keys:
        row = ctx.chase_share_by_venue.get(vk)
        if row:
            return row[0], row[1]
    return 0.5, 0


def _linkage_confidence_adjustments(
    scored_a: list[SquadPlayer],
    scored_b: list[SquadPlayer],
) -> dict[str, Any]:
    core_roles = frozenset({"Batter", "WK-Batter", "All-Rounder"})
    alias_quality_hits = 0
    global_fb = 0
    collision_n = 0
    sparse_core = 0
    ambiguous_n = 0
    no_match_core = 0

    for p in scored_a + scored_b:
        hd = getattr(p, "history_debug", None) or {}
        smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        bbd = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        rt = str(lk.get("resolution_type") or "")
        coll = str(lk.get("collision_resolution_outcome") or "")
        if coll == "lost_collision" or rt == "ambiguous_alias_collision":
            collision_n += 1
        if rt == "ambiguous_alias":
            ambiguous_n += 1
        rolled = str(lk.get("rolled_up_interpretation") or "")
        if bool(lk.get("used_global_resolved_key_for_prior")) or "global_alias" in rolled:
            global_fb += 1
        if (p.role_bucket or "") in core_roles:
            if rt == "no_match":
                no_match_core += 1
            if str(lk.get("rolled_up_interpretation") or "") == "likely_new_or_sparse":
                sparse_core += 1
        if rt in ("exact_match", "alias_match") and coll != "lost_collision":
            alias_quality_hits += 1

    npl = max(1, len(scored_a) + len(scored_b))
    alias_quality_component = min(1.0, alias_quality_hits / float(npl))
    global_fallback_penalty = min(0.22, 0.028 * global_fb)
    collision_penalty = min(0.28, 0.065 * collision_n)
    sparse_core_penalty = min(0.2, 0.035 * sparse_core)
    ambiguous_penalty = min(0.14, 0.022 * ambiguous_n)
    no_match_core_penalty = min(0.18, 0.04 * no_match_core)
    total_pen = min(
        0.55,
        global_fallback_penalty
        + collision_penalty
        + sparse_core_penalty
        + ambiguous_penalty
        + no_match_core_penalty,
    )
    return {
        "alias_quality_component": round(alias_quality_component, 4),
        "global_fallback_penalty": round(global_fallback_penalty, 4),
        "collision_penalty": round(collision_penalty, 4),
        "sparse_core_penalty": round(sparse_core_penalty, 4),
        "ambiguous_alias_penalty": round(ambiguous_penalty, 4),
        "no_match_core_penalty": round(no_match_core_penalty, 4),
        "linkage_penalty_total": round(total_pen, 4),
        "counts": {
            "collision_losers": collision_n,
            "global_fallback_signals": global_fb,
            "ambiguous_alias_players": ambiguous_n,
            "no_match_core_players": no_match_core,
            "likely_sparse_core_players": sparse_core,
        },
    }


def _prediction_confidence(
    xi_a: list[SquadPlayer],
    xi_b: list[SquadPlayer],
    hctx: HistoryContext,
    str_a: float,
    str_b: float,
    *,
    scored_a: Optional[list[SquadPlayer]] = None,
    scored_b: Optional[list[SquadPlayer]] = None,
) -> dict[str, Any]:
    n = hctx.db_match_count
    db_s = min(1.0, n / max(1, config.CONF_MIN_MATCHES_FOR_FULL_DB))

    all_xi = list(xi_a) + list(xi_b)

    def xi_cov(xi: list[SquadPlayer]) -> float:
        if len(xi) != 11:
            return 0.0
        hit = sum(
            1 for p in xi if hctx.xi_by_player.get(learner.normalize_player_key(p.name), 0) > 0
        )
        return min(1.0, (hit / 11.0) / max(0.05, config.CONF_IDEAL_XI_HISTORY_COVERAGE))

    cov_s = (xi_cov(xi_a) + xi_cov(xi_b)) / 2.0
    history_coverage_s = max(0.0, min(1.0, 0.65 * cov_s + 0.35 * db_s))

    # Linkage quality as contribution (instead of large post-hoc subtraction).
    link_adj: dict[str, Any] = {}
    linkage_s = 0.55
    if scored_a is not None and scored_b is not None:
        link_adj = _linkage_confidence_adjustments(scored_a, scored_b)
        alias_q = float(link_adj.get("alias_quality_component") or 0.0)
        pen = float(link_adj.get("linkage_penalty_total") or 0.0)
        linkage_s = max(0.0, min(1.0, 0.15 + 0.85 * alias_q - 0.45 * pen))

    # Recent-form coverage & confidence.
    rf_vals: list[float] = []
    for p in all_xi:
        hd = getattr(p, "history_debug", None) or {}
        smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        rfd = smd.get("recent_form_detail") if isinstance(smd.get("recent_form_detail"), dict) else {}
        source = str(rfd.get("recent_form_source") or "")
        conf = float(rfd.get("sample_confidence") or 0.0)
        if source == "player_recent_form_cache":
            rf_vals.append(max(0.45, conf))
        elif source:
            rf_vals.append(max(0.25, 0.65 * conf))
        else:
            rf_vals.append(0.2)
    recent_form_s = sum(rf_vals) / max(1, len(rf_vals))

    # Batting-order stability from slot coverage and in-band placement.
    bo_stab_vals: list[float] = []
    for p in all_xi:
        hd = getattr(p, "history_debug", None) or {}
        rows = float(hd.get("batting_position_rows_found") or 0.0)
        row_s = min(1.0, rows / 10.0)
        in_band = 0.0 if bool(hd.get("moved_outside_band")) else 1.0
        bo_stab_vals.append(0.55 * row_s + 0.45 * in_band)
    batting_order_stability_s = sum(bo_stab_vals) / max(1, len(bo_stab_vals))

    # Role stability from derive / selection debug (fallback to role-band confidence proxy).
    rs_vals: list[float] = []
    for p in all_xi:
        hd = getattr(p, "history_debug", None) or {}
        smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        bbd = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
        rs = float(bbd.get("stable_role_identity_score") or 0.0)
        if rs <= 1e-6:
            ds = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
            rs = float(ds.get("role_stability_score") or 0.0)
        if rs <= 1e-6:
            rb = str(hd.get("role_band") or "")
            rs = 0.36 if rb else 0.18
        rs_vals.append(max(0.0, min(1.0, rs)))
    role_stability_s = sum(rs_vals) / max(1, len(rs_vals))

    # Structure validity.
    ok_a, _ = _validate_xi(xi_a)
    ok_b, _ = _validate_xi(xi_b)
    structure_validity_s = 1.0 if (ok_a and ok_b and len(xi_a) == 11 and len(xi_b) == 11) else 0.45

    # Team separation (fix prior scaling bug; avoid near-zero collapse).
    sep = abs(str_a - str_b) / max(1e-6, 11.0)
    sep_target = max(1e-6, float(getattr(config, "CONF_SEPARATION_TARGET", 0.045)))
    team_separation_s = min(1.0, sep / sep_target)

    # Matchup coverage from precomputed summary tables (lightweight).
    matchup_coverage_s = 0.35
    try:
        xi_keys = [str(getattr(p, "player_key", "") or "").strip() for p in all_xi]
        xi_keys = [k for k in xi_keys if k]
        if xi_keys:
            qm = ",".join("?" * len(xi_keys))
            with db.connection() as conn:
                q1 = conn.execute(
                    f"SELECT COUNT(DISTINCT batter_key) AS n FROM batter_vs_bowling_type_summary WHERE batter_key IN ({qm})",
                    xi_keys,
                ).fetchone()
                q2 = conn.execute(
                    f"SELECT COUNT(DISTINCT bowler_key) AS n FROM bowler_vs_batting_hand_summary WHERE bowler_key IN ({qm})",
                    xi_keys,
                ).fetchone()
            bat_cov = float((q1["n"] if q1 else 0) or 0) / max(1.0, float(len(xi_keys)))
            bowl_cov = float((q2["n"] if q2 else 0) or 0) / max(1.0, float(len(xi_keys)))
            matchup_coverage_s = max(0.0, min(1.0, 0.5 * bat_cov + 0.5 * bowl_cov))
    except Exception:
        matchup_coverage_s = 0.35

    components = {
        "linkage_contribution": linkage_s,
        "history_coverage_contribution": history_coverage_s,
        "recent_form_contribution": recent_form_s,
        "batting_order_stability_contribution": batting_order_stability_s,
        "role_stability_contribution": role_stability_s,
        "structure_contribution": structure_validity_s,
        "team_separation_contribution": team_separation_s,
        "matchup_coverage_contribution": matchup_coverage_s,
    }

    weights = {
        "linkage_contribution": 0.18,
        "history_coverage_contribution": 0.18,
        "recent_form_contribution": 0.14,
        "batting_order_stability_contribution": 0.10,
        "role_stability_contribution": 0.10,
        "structure_contribution": 0.12,
        "team_separation_contribution": 0.12,
        "matchup_coverage_contribution": 0.06,
    }
    score = 0.0
    for k, w in weights.items():
        score += w * float(components.get(k) or 0.0)
    score = max(0.0, min(1.0, score))

    # Guardrail floor when data quality is healthy.
    healthy_data = (
        linkage_s >= 0.62
        and history_coverage_s >= 0.55
        and recent_form_s >= 0.45
        and structure_validity_s >= 0.95
    )
    if healthy_data:
        score = max(score, 0.46)

    out: dict[str, Any] = {
        "type": "model_confidence",
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "score_pct": round(100.0 * float(max(0.0, min(1.0, score))), 2),
        "components": {k: round(float(v), 4) for k, v in components.items()},
        "weights": weights,
        "raw": {
            "stored_matches": n,
            "strength_gap_per_slot": round(sep, 6),
            "healthy_data_floor_applied": bool(healthy_data and score >= 0.46),
        },
    }
    if link_adj:
        out["linkage_adjustments"] = link_adj
    return out


def _is_bowling_option(p: SquadPlayer) -> bool:
    return classify_player(p).is_bowling_option


def _bowling_style_bucket(p: SquadPlayer) -> str:
    f = classify_player(p)
    if f.is_spinner:
        return "spin"
    if f.is_pacer:
        return "pace"
    return "none"


def _is_pace_bowler_candidate(p: SquadPlayer) -> bool:
    return classify_player(p).is_pacer


def _is_spinner_candidate(p: SquadPlayer) -> bool:
    return classify_player(p).is_spinner


def _is_proper_batter(p: SquadPlayer) -> bool:
    return p.role_bucket in (BATTER, WK_BATTER)


def _annotate_phase_bowling_signals(players: list[SquadPlayer], franchise_team_key: str) -> None:
    def _lookup_key(p: SquadPlayer) -> str:
        hd = getattr(p, "history_debug", None) or {}
        hud = hd.get("history_usage_debug") if isinstance(hd.get("history_usage_debug"), dict) else {}
        return (
            str(hud.get("history_lookup_key") or hud.get("resolved_history_key") or getattr(p, "canonical_player_key", "") or getattr(p, "player_key", "") or "")
            .strip()
            .lower()
        )

    key_by_name = {p.name: _lookup_key(p) for p in players}
    pks = [k for k in key_by_name.values() if k]
    phase_map = db.fetch_bowler_phase_summary_batch(franchise_team_key, pks)
    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        pk = key_by_name.get(p.name, "")
        ph = phase_map.get(pk, {})
        if ph:
            hd["bowler_phase_summary"] = ph
            hd["phase_lookup_key"] = pk


def _annotate_player_metadata(players: list[SquadPlayer]) -> None:
    def _lookup_key(p: SquadPlayer) -> str:
        hd = getattr(p, "history_debug", None) or {}
        hud = hd.get("history_usage_debug") if isinstance(hd.get("history_usage_debug"), dict) else {}
        return (
            str(hud.get("history_lookup_key") or hud.get("resolved_history_key") or getattr(p, "canonical_player_key", "") or getattr(p, "player_key", "") or "")
            .strip()
            .lower()
        )

    def _lookup_candidates(p: SquadPlayer) -> list[str]:
        hd = getattr(p, "history_debug", None) or {}
        hud = hd.get("history_usage_debug") if isinstance(hd.get("history_usage_debug"), dict) else {}
        name_norm = learner.normalize_player_key(str(getattr(p, "name", "") or ""))
        name_tokens = [t for t in name_norm.split() if t]
        first_last = ""
        if len(name_tokens) >= 3:
            first_last = f"{name_tokens[0]} {name_tokens[-1]}"
        cands = [
            str(hud.get("history_lookup_key") or "").strip().lower(),
            str(hud.get("resolved_history_key") or "").strip().lower(),
            str(getattr(p, "canonical_player_key", "") or "").strip().lower(),
            str(getattr(p, "player_key", "") or "").strip().lower(),
            name_norm,
            first_last,
        ]
        out: list[str] = []
        for c in cands:
            if c and c not in out:
                out.append(c)
        # Also try curated alias canonicalization for metadata keys (does not affect display names).
        # This fixes cases where squad truth uses a variant ("mohammad shami") but metadata is stored
        # under a curated canonical key ("mohammed shami").
        canon_extra: list[str] = []
        for c in out:
            canon = player_alias_resolve.canonicalize_via_alias_overrides(c)
            if canon and canon not in out and canon not in canon_extra:
                canon_extra.append(canon)
        out.extend(canon_extra)
        return out

    def _runtime_bowling_type_from_metadata(m: dict[str, Any]) -> str:
        raw = str(m.get("bowling_style_raw") or "").strip().lower()
        btb = str(m.get("bowling_type_bucket") or "").strip().lower()
        # Prefer explicit raw style parsing so pace is not left blank.
        if ("right" in raw and ("fast medium" in raw or "medium fast" in raw)):
            return "right_arm_fast_medium"
        if ("left" in raw and ("fast medium" in raw or "medium fast" in raw)):
            return "left_arm_fast_medium"
        if "right" in raw and "fast" in raw:
            return "right_arm_fast"
        if "left" in raw and "fast" in raw:
            return "left_arm_fast"
        if "offbreak" in raw or "off break" in raw:
            return "finger_spin"
        if any(x in raw for x in ("legbreak", "leg break", "googly", "chinaman", "left-arm wrist")):
            return "wrist_spin"
        if any(x in raw for x in ("left-arm orthodox", "slow left-arm orthodox")):
            return "left_arm_orthodox"
        if "mystery" in raw:
            return "mystery_spin"
        # Bucket fallback.
        if btb in (
            "right_arm_fast",
            "right_arm_fast_medium",
            "left_arm_fast",
            "left_arm_fast_medium",
            "finger_spin",
            "wrist_spin",
            "left_arm_orthodox",
            "mystery_spin",
            "unknown",
        ):
            return btb
        if btb == "pace":
            # Arm/speed missing in metadata; keep non-blank pace style.
            return "right_arm_fast_medium"
        return "unknown"

    def _meta_quality(m: dict[str, Any]) -> tuple[int, int]:
        btb = str(m.get("bowling_type_bucket") or "").strip().lower()
        raw = str(m.get("bowling_style_raw") or "").strip().lower()
        bucket_quality = 0 if btb in ("", "unknown") else 1
        raw_quality = 0 if raw in ("", "unknown") else 1
        return (bucket_quality, raw_quality)

    key_cands_by_name = {p.name: _lookup_candidates(p) for p in players}
    pks: list[str] = []
    for cands in key_cands_by_name.values():
        for c in cands:
            if c not in pks:
                pks.append(c)
    meta = db.fetch_player_metadata_batch(pks)
    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        m = None
        chosen_pk = ""
        curated = _load_curated_player_metadata_fallback()
        options: list[tuple[str, dict[str, Any], str]] = []
        for cand in key_cands_by_name.get(p.name, []):
            if cand in meta:
                options.append((cand, dict(meta[cand]), "db"))
            if cand in curated:
                options.append((cand, dict(curated[cand]), "curated_fallback"))
        if options:
            options.sort(key=lambda x: (_meta_quality(x[1]), 1 if x[2] == "curated_fallback" else 0), reverse=True)
            chosen_pk, m, m_src = options[0]
            if not isinstance(getattr(p, "history_debug", None), dict):
                p.history_debug = {}
            p.history_debug["player_metadata_source_runtime"] = m_src
        if not m:
            continue
        hd["player_metadata"] = m
        hd["metadata_lookup_key"] = chosen_pk
        hd["bowling_type_from_metadata"] = _runtime_bowling_type_from_metadata(m)
        # Metadata is authoritative for bowling_type in runtime/export **only** when it provides
        # a non-unknown style. Do not clobber existing squad bowling_type with "unknown".
        bt_meta = str(hd.get("bowling_type_from_metadata") or "").strip()
        if bt_meta and bt_meta.lower() not in ("unknown", "none", "na", "n/a"):
            p.bowling_type = bt_meta


def _batting_band_from_profile(dominant_position: float, top12_share: float) -> str:
    dp = float(dominant_position or 0.0)
    if dp <= 2.0:
        return "opener"
    if dp <= 3.0 or (top12_share >= 0.65 and dp <= 4.0):
        return "top_order"
    if dp <= 5.0:
        return "middle"
    if dp <= 7.0:
        return "finisher"
    return "tail"


def _annotate_batting_position_profiles(players: list[SquadPlayer], franchise_team_key: str) -> None:
    def _lookup_key(p: SquadPlayer) -> str:
        hd = getattr(p, "history_debug", None) or {}
        hud = hd.get("history_usage_debug") if isinstance(hd.get("history_usage_debug"), dict) else {}
        return (
            str(hud.get("history_lookup_key") or hud.get("resolved_history_key") or getattr(p, "canonical_player_key", "") or getattr(p, "player_key", "") or "")
            .strip()
            .lower()
        )

    key_by_name = {p.name: _lookup_key(p) for p in players}
    pks = [k for k in key_by_name.values() if k]
    prof = db.fetch_player_batting_position_profile_batch(franchise_team_key, pks)
    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        pk = key_by_name.get(p.name, "")
        pr = prof.get(pk)
        if not pr:
            continue
        dp = float(pr.get("dominant_position") or 0.0)
        t12 = float(pr.get("top12_share") or 0.0)
        band = _batting_band_from_profile(dp, t12)
        hd["dominant_position"] = dp
        hd["batting_band"] = band
        hd["batting_position_distribution"] = pr.get("distribution") or {}
        hd["top12_share"] = t12
        hd["overwhelming_top_order_history"] = bool(t12 >= 0.65)
        hd["batting_profile_lookup_key"] = pk


def _set_player_ipl_flags(p: SquadPlayer) -> None:
    """After scoring: opener / finisher flags for XI validation and batting order."""
    hd = getattr(p, "history_debug", None) or {}
    band = str(hd.get("role_band") or "")
    bat_band = str(hd.get("batting_band") or "")
    p.is_opener_candidate = band in ("opener", "top_order") or bat_band in ("opener", "top_order") or p.role_bucket in (BATTER, WK_BATTER)
    p.is_finisher_candidate = (
        band in ("finisher", "batting_allrounder")
        or p.role_bucket == ALL_ROUNDER
        or (p.bat_skill >= 0.54 and p.bowl_skill >= 0.48)
        or (p.role_bucket == BOWLER and p.bat_skill >= 0.43)
    )


def _batting_position_signal_for_role_band(p: SquadPlayer) -> float:
    hd = getattr(p, "history_debug", None) or {}
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    try:
        ema = float(getattr(p, "history_batting_ema", unk))
    except (TypeError, ValueError):
        ema = unk
    if ema < unk - 1e-6:
        return ema
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
    try:
        d_ema = float(prof.get("batting_position_ema") or 0.0)
    except (TypeError, ValueError):
        d_ema = 0.0
    if d_ema > 0.5:
        return d_ema
    return 7.5


def _derive_role_band_for_player(p: SquadPlayer) -> str:
    hd = getattr(p, "history_debug", None) or {}
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
    phs = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    meta = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    rb = str(getattr(p, "role_bucket", "") or "")
    slot = _batting_position_signal_for_role_band(p)
    try:
        ol = float(prof.get("opener_likelihood") or 0.0)
        fl = float(prof.get("finisher_likelihood") or 0.0)
        ppl = float(prof.get("powerplay_bowler_likelihood") or 0.0)
        dth = float(prof.get("death_bowler_likelihood") or 0.0)
    except (TypeError, ValueError):
        ol, fl, ppl, dth = 0.0, 0.0, 0.0, 0.0
    try:
        pp_share = float(phs.get("powerplay_share") or 0.0)
        md_share = float(phs.get("middle_share") or 0.0)
        dt_share = float(phs.get("death_share") or 0.0)
        pp_wpb = float(phs.get("powerplay_wickets_per_ball") or 0.0)
        dt_wpb = float(phs.get("death_wickets_per_ball") or 0.0)
        ph_total_balls = float(phs.get("total_balls") or 0.0)
    except (TypeError, ValueError):
        pp_share, md_share, dt_share, pp_wpb, dt_wpb, ph_total_balls = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    btb = str(meta.get("bowling_type_bucket") or "").strip().lower()
    spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist")) or btb in (
        "finger_spin",
        "wrist_spin",
        "left_arm_orthodox",
        "mystery_spin",
    )
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    lmd = sm.get("last_match_detail") if isinstance(sm.get("last_match_detail"), dict) else {}
    try:
        last_pos = float(lmd.get("last_match_batting_position") or 0.0)
    except (TypeError, ValueError):
        last_pos = 0.0

    if rb == WK_BATTER:
        bband = str(hd.get("batting_band") or "")
        if bband == "opener":
            return "top_order"
        if bband == "top_order":
            return "top_order"
        if bband == "middle":
            return "wicketkeeper_batter"
        if bband == "finisher":
            return "finisher"
        if slot >= 5.5 or fl >= 0.56:
            return "finisher"
        if (last_pos > 0 and last_pos <= 3.0) and ol >= 0.54:
            return "top_order"
        if slot <= 3.0 and ol >= 0.56:
            return "top_order"
        return "wicketkeeper_batter"
    if rb == BATTER:
        bband = str(hd.get("batting_band") or "")
        if bband == "opener":
            return "opener"
        if bband == "top_order":
            return "top_order"
        if bband == "middle":
            return "middle_order"
        if bband == "finisher":
            return "finisher"
        if (last_pos > 0 and last_pos <= 2.2) or (slot <= 2.6) or (ol >= 0.62):
            return "opener"
        if slot <= 4.3:
            return "top_order"
        if slot <= 6.2:
            return "middle_order"
        return "finisher"
    if rb == ALL_ROUNDER:
        if ph_total_balls >= 120:
            if dt_share >= 0.30 and (dt_wpb >= (pp_wpb * 0.9) or dt_share + 0.02 >= pp_share):
                return "death_bowler"
            if dt_share >= pp_share and dt_share >= md_share and dt_share >= 0.24:
                return "death_bowler"
            if pp_share >= dt_share and pp_share >= md_share and pp_share >= 0.24:
                return "powerplay_bowler"
            if md_share >= pp_share and md_share >= dt_share and md_share >= 0.30 and spin_like:
                return "middle_overs_spinner"
        if dt_share >= 0.33 or (dt_share >= 0.20 and dt_wpb >= 0.02):
            return "death_bowler"
        if pp_share >= 0.30 or (pp_share >= 0.20 and pp_wpb >= 0.02):
            return "powerplay_bowler"
        if md_share >= 0.36 and spin_like:
            return "middle_overs_spinner"
        if dth >= 0.62:
            return "death_bowler"
        if ppl >= 0.62:
            return "powerplay_bowler"
        if p.bowl_skill >= p.bat_skill + 0.08:
            if spin_like:
                return "middle_overs_spinner"
            return "bowling_allrounder"
        if p.bat_skill >= p.bowl_skill + 0.05:
            return "batting_allrounder"
        if spin_like:
            return "middle_overs_spinner"
        return "batting_allrounder"
    if rb == BOWLER:
        if ph_total_balls >= 120:
            if dt_share >= 0.30 and (dt_wpb >= (pp_wpb * 0.9) or dt_share + 0.02 >= pp_share):
                return "death_bowler"
            if dt_share >= pp_share and dt_share >= md_share and dt_share >= 0.24:
                return "death_bowler"
            if pp_share >= dt_share and pp_share >= md_share and pp_share >= 0.24:
                return "powerplay_bowler"
            if md_share >= pp_share and md_share >= dt_share and md_share >= 0.30 and spin_like:
                return "middle_overs_spinner"
        if dt_share >= 0.33 or (dt_share >= 0.20 and dt_wpb >= 0.02):
            return "death_bowler"
        if pp_share >= 0.30 or (pp_share >= 0.20 and pp_wpb >= 0.02):
            return "powerplay_bowler"
        if md_share >= 0.36 and spin_like:
            return "middle_overs_spinner"
        if dth >= 0.58:
            return "death_bowler"
        if ppl >= 0.58:
            return "powerplay_bowler"
        if spin_like:
            return "middle_overs_spinner"
        return "utility_bowler"
    return "middle_order"


def _annotate_role_bands(players: list[SquadPlayer]) -> None:
    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        band = _derive_role_band_for_player(p)
        hd["role_band"] = band
        hd["batting_position_history_basis"] = round(_batting_position_signal_for_role_band(p), 3)


def _role_band(p: SquadPlayer) -> str:
    return str((getattr(p, "history_debug", None) or {}).get("role_band") or "")


def _is_powerplay_bowler_candidate(p: SquadPlayer) -> bool:
    if _role_band(p) in ("powerplay_bowler",):
        return True
    hd = getattr(p, "history_debug", None) or {}
    ph = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    try:
        pp_share = float(ph.get("powerplay_share") or 0.0)
        pp_wpb = float(ph.get("powerplay_wickets_per_ball") or 0.0)
        dt_share = float(ph.get("death_share") or 0.0)
        total_balls = float(ph.get("total_balls") or 0.0)
    except (TypeError, ValueError):
        pp_share, pp_wpb, dt_share, total_balls = 0.0, 0.0, 0.0, 0.0
    if total_balls >= 120 and (pp_share + 0.08) < dt_share:
        return False
    if pp_share >= 0.23 or (pp_share >= 0.17 and pp_wpb >= 0.02):
        return True
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
    try:
        ppl = float(prof.get("powerplay_bowler_likelihood") or 0.0)
    except (TypeError, ValueError):
        ppl = 0.0
    if ppl >= 0.45:
        return True
    if _is_bowling_composition_band(_role_band(p)) and float(getattr(p, "bowl_skill", 0.0) or 0.0) >= 0.67:
        return True
    return False


def _is_death_bowler_candidate(p: SquadPlayer) -> bool:
    if _role_band(p) in ("death_bowler",):
        return True
    hd = getattr(p, "history_debug", None) or {}
    ph = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    try:
        dt_share = float(ph.get("death_share") or 0.0)
        dt_wpb = float(ph.get("death_wickets_per_ball") or 0.0)
        pp_share = float(ph.get("powerplay_share") or 0.0)
        total_balls = float(ph.get("total_balls") or 0.0)
    except (TypeError, ValueError):
        dt_share, dt_wpb, pp_share, total_balls = 0.0, 0.0, 0.0, 0.0
    if total_balls >= 120 and (dt_share + 0.08) < pp_share:
        return False
    if dt_share >= 0.24 or (dt_share >= 0.16 and dt_wpb >= 0.022):
        return True
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
    try:
        dth = float(prof.get("death_bowler_likelihood") or 0.0)
    except (TypeError, ValueError):
        dth = 0.0
    if dth >= 0.45:
        return True
    if _is_bowling_composition_band(_role_band(p)) and float(getattr(p, "bowl_skill", 0.0) or 0.0) >= 0.72:
        return True
    return False


def _is_opener_candidate_strict(p: SquadPlayer) -> bool:
    return classify_player(p).is_top_order_batter


def _is_primary_strike_bowler(p: SquadPlayer) -> bool:
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    ipl = float(bb.get("ipl_history_and_role_score") or 0.0)
    return _is_death_bowler_candidate(p) and _is_bowling_option(p) and float(getattr(p, "bowl_skill", 0.0)) >= 0.68 and ipl >= 0.28


def _is_structural_allrounder(p: SquadPlayer) -> bool:
    return classify_player(p).is_structural_all_rounder


def _wk_designation_priority(p: SquadPlayer) -> tuple[float, float, float, float]:
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    continuity = float(bb.get("last_match_continuity_score") or 0.0)
    role_stability = float(bb.get("stable_role_identity_score") or 0.0)
    experience = float(bb.get("ipl_history_and_role_score") or 0.0)
    pm = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    meta_conf = float(pm.get("confidence") or 0.0)
    return (continuity, role_stability, experience, meta_conf)


def _assign_designated_keeper(xi: list[SquadPlayer]) -> Optional[str]:
    chosen_name = rules_xi.assign_designated_keeper_name(xi)
    for p in xi:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        p.history_debug["is_wk_role"] = bool(getattr(p, "is_wicketkeeper", False))
        p.history_debug["designated_keeper"] = bool(chosen_name and p.name == chosen_name)
    return chosen_name


def _must_lock_in_base_xi(p: SquadPlayer) -> bool:
    """Hard core lock for obvious starters unless constraints become impossible."""
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    cont = float(bb.get("last_match_continuity_score") or 0.0)
    ipl = float(bb.get("ipl_history_and_role_score") or 0.0)
    role_st = float(bb.get("stable_role_identity_score") or 0.0)
    rf = float(bb.get("recent_form_score") or 0.0)
    band = _role_band(p)
    marquee_tier = str(hd.get("marquee_tier") or "").strip().lower()
    marquee_source = str(hd.get("marquee_source") or "").strip().lower()
    marquee_score = float(hd.get("marquee_suggested_score") or 0.0)
    if marquee_tier == "tier_1":
        return True
    if marquee_tier == "tier_2" and marquee_source == "curated" and marquee_score >= 0.62:
        return True
    if _is_primary_strike_bowler(p):
        return True
    if band in ("opener", "top_order") and _elite_player_signal(p) >= 0.52:
        return True
    if bool(hd.get("captain_selected_for_team")) and (ipl >= 0.28 or cont >= 0.54):
        return True
    if bool(hd.get("wicketkeeper_selected_for_team")) and (band in ("wicketkeeper_batter", "top_order", "finisher")):
        return True
    if band == "wicketkeeper_batter" and _elite_player_signal(p) >= 0.52:
        return True
    return (cont >= 0.62 and role_st >= 0.3) or (ipl >= 0.34 and rf >= 0.32 and role_st >= 0.28)


def _xi_role_validation_counts(xi: list[SquadPlayer]) -> dict[str, Any]:
    """
    Counts on the **selected XI** for selection-debug UI (derive likelihood thresholds are heuristic).
    """
    if not xi:
        return {
            "xi_size": 0,
            "wicketkeeper_count": 0,
            "bowling_options_count": 0,
            "powerplay_bowlers_count": 0,
            "death_bowlers_count": 0,
            "opener_candidates_count": 0,
            "finisher_candidates_count": 0,
            "overseas_count": 0,
        }

    designated_keeper_name = _assign_designated_keeper(xi)
    pp_n = sum(1 for p in xi if _is_powerplay_bowler_candidate(p))
    de_n = sum(1 for p in xi if _is_death_bowler_candidate(p))
    wk_role_players = [p for p in xi if bool(getattr(p, "is_wicketkeeper", False))]
    wk_role_names = [p.name for p in wk_role_players]
    wk_used = 1 if designated_keeper_name else 0
    raw_ars = [p for p in xi if p.role_bucket == ALL_ROUNDER]
    structural_ars = [p for p in xi if classify_player(p).is_structural_all_rounder]
    treated_as_batter_for_balance = [
        p.name
        for p in xi
        if (p.name in wk_role_names and p.name != designated_keeper_name)
        or (p.role_bucket == ALL_ROUNDER and not _is_structural_allrounder(p))
    ]
    soft_preference_flags: list[str] = []
    if len(wk_role_players) > 2:
        soft_preference_flags.append(f"soft_wk_preference_exceeded:{len(wk_role_players)}")
    if len(structural_ars) > 3:
        soft_preference_flags.append(f"soft_structural_allrounder_preference_exceeded:{len(structural_ars)}")

    c = role_counts(xi)
    return {
        "xi_size": len(xi),
        "wicketkeeper_count": wk_used,
        "wk_role_players_count": len(wk_role_players),
        "wk_count_used_for_structure": wk_used,
        "designated_keeper_name": designated_keeper_name,
        "wk_role_players": wk_role_names,
        "bowling_options_count": int(c["bowling_options"]),
        "pacers_count": int(c["pacers"]),
        "spinners_count": int(c["spinners"]),
        "proper_batters_count": sum(1 for p in xi if _is_proper_batter(p)),
        "all_rounders_count": len(structural_ars),
        "all_rounder_players_count_raw": len(raw_ars),
        "structural_all_rounder_count": len(structural_ars),
        "treated_as_batter_for_balance": treated_as_batter_for_balance,
        "soft_preference_flags": soft_preference_flags,
        "powerplay_bowlers_count": pp_n,
        "death_bowlers_count": de_n,
        "opener_candidates_count": sum(1 for p in xi if _is_opener_candidate_strict(p)),
        "finisher_candidates_count": sum(1 for p in xi if getattr(p, "is_finisher_candidate", False)),
        "overseas_count": sum(1 for p in xi if p.is_overseas),
        "overseas_count_selected": sum(1 for p in xi if p.is_overseas),
        "powerplay_death_threshold_note": "PP/death counts use role band + derive likelihood + ball-by-ball phase shares",
    }


def _validate_xi(
    xi: list[SquadPlayer],
    conditions: Optional[dict[str, Any]] = None,
    squad: Optional[list[SquadPlayer]] = None,
) -> tuple[bool, list[str]]:
    res = rules_xi.validate_xi(xi, conditions=conditions, squad=squad)
    return (res.hard_ok, [v.message for v in res.violations])


def _semi_hard_structure_errors(xi: list[SquadPlayer]) -> list[str]:
    res = rules_xi.validate_xi(xi)
    return [w.message for w in res.warnings]


def _scenario_xi_rank_value(p: SquadPlayer, scenario_branch: Optional[str]) -> float:
    """Rank key for XI build: scenario-adjusted when branch is set, else base selection_score."""
    if scenario_branch not in ("if_team_bats_first", "if_team_bowls_first"):
        return float(getattr(p, "selection_score", 0.0))
    sx = (getattr(p, "history_debug", None) or {}).get("scenario_xi") or {}
    sub = sx.get(scenario_branch) or {}
    v = sub.get("scenario_selection_score")
    if v is not None:
        return float(v)
    return float(getattr(p, "selection_score", 0.0))


def _core_anchor_strength(p: SquadPlayer) -> float:
    """Score how strongly this player should remain in the base XI."""
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    rf = float(bb.get("recent_form_score") or 0.0)
    ih = float(bb.get("ipl_history_and_role_score") or 0.0)
    tb = float(bb.get("team_balance_fit_score") or 0.0)
    cont = float(bb.get("last_match_continuity_score") or 0.0)
    base = float(bb.get("base_weighted_sum") or getattr(p, "selection_score", 0.0))
    prior_fc = float(hd.get("probable_first_choice_prior") or 0.0)
    used_g = bool(hd.get("used_global_fallback_prior"))
    gsf = float(hd.get("global_selection_frequency") or 0.0)
    s = 0.0
    if base >= 0.38:
        s += 0.05
    if rf >= 0.36:
        s += 0.04
    if ih >= 0.34:
        s += 0.04
    if tb >= 0.56:
        s += 0.02
    if cont >= 0.62:
        s += 0.04
    elif cont >= 0.5:
        s += 0.02
    if bool(hd.get("global_ipl_history_presence")):
        s += 0.02
    if prior_fc >= 0.55:
        s += 0.06 if used_g else 0.04
    elif prior_fc >= 0.45:
        s += 0.03
    if gsf >= 0.8:
        s += 0.02
    if bool(hd.get("captain_selected_for_team")):
        s += 0.03
    if bool(hd.get("wicketkeeper_selected_for_team")):
        s += 0.03
    if bool(hd.get("valid_current_squad_new_to_franchise")) and prior_fc >= 0.45:
        s += 0.03
    if p.role_bucket in (BATTER, WK_BATTER):
        try:
            ol = float((hd.get("derive_player_profile") or {}).get("opener_likelihood") or 0.0)
        except (TypeError, ValueError):
            ol = 0.0
        if ol >= 0.55:
            s += 0.02
    return max(0.0, min(0.28, s))


def _elite_player_signal(p: SquadPlayer) -> float:
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    cont = float(bb.get("last_match_continuity_score") or 0.0)
    rf = float(bb.get("recent_form_score") or 0.0)
    ipl = float(bb.get("ipl_history_and_role_score") or 0.0)
    role_st = float(bb.get("stable_role_identity_score") or 0.0)
    prior_fc = float(hd.get("probable_first_choice_prior") or 0.0)
    gsf = float(hd.get("global_selection_frequency") or 0.0)
    has_global = bool(hd.get("global_ipl_history_presence"))
    core = 0.15 + 0.26 * cont + 0.20 * rf + 0.20 * ipl + 0.10 * role_st + 0.11 * prior_fc + 0.10 * gsf
    if has_global:
        core += 0.05
    if bool(hd.get("captain_selected_for_team")):
        core += 0.05
    if bool(hd.get("wicketkeeper_selected_for_team")):
        core += 0.05
    rb = str(hd.get("role_band") or "")
    if rb in ("opener", "top_order", "death_bowler", "powerplay_bowler"):
        core += 0.03
    return max(0.0, min(1.0, core))


def _marquee_role_identity_boost(p: SquadPlayer, base_role_identity: float) -> float:
    """
    Boost role identity for serious first-XI archetypes so Tier 2 is not underused.
    """
    hd = getattr(p, "history_debug", None) or {}
    rb = str(hd.get("role_band") or "").strip().lower()
    boost = 0.0

    if rb == "opener":
        boost += 0.22
    elif rb == "top_order":
        boost += 0.20
    elif rb == "wicketkeeper_batter":
        boost += 0.18
    elif rb == "death_bowler":
        boost += 0.22
    elif rb == "powerplay_bowler":
        boost += 0.20
    elif rb == "middle_overs_spinner":
        boost += 0.15
    elif rb == "bowling_allrounder":
        boost += 0.14

    # Strengthen first-choice wicketkeeper signal.
    if bool(hd.get("wicketkeeper_selected_for_team")):
        boost += 0.08

    # Strike-bowler and premium all-rounder pattern.
    if p.role_bucket == BOWLER and float(getattr(p, "bowl_skill", 0.0) or 0.0) >= 0.72:
        boost += 0.08
    if (
        p.role_bucket == ALL_ROUNDER
        and float(getattr(p, "bat_skill", 0.0) or 0.0) >= 0.68
        and float(getattr(p, "bowl_skill", 0.0) or 0.0) >= 0.65
    ):
        boost += 0.10

    return max(0.0, min(1.0, base_role_identity + boost))


def _suggested_marquee_score(p: SquadPlayer) -> tuple[float, dict[str, float]]:
    """
    Transparent marquee suggestion score (not final authority; curated overrides win).
    Signals: squad truth, continuity, IPL history, role identity, elite sanity, transfer-core boost.
    """
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    continuity = float(bb.get("last_match_continuity_score") or 0.0)
    ipl_hist = float(bb.get("ipl_history_and_role_score") or 0.0)
    role_identity = float(bb.get("stable_role_identity_score") or 0.0)
    # Role identity fallback when derive role stability is missing/zero.
    derive_snap = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
    role_identity = max(role_identity, float(derive_snap.get("role_stability_score") or 0.0))
    rb = str(hd.get("role_band") or "")
    if role_identity <= 1e-6 and rb:
        if rb in ("opener", "top_order", "wicketkeeper_batter", "death_bowler", "powerplay_bowler", "middle_overs_spinner"):
            role_identity = 0.42
        elif rb in ("middle_order", "finisher", "bowling_allrounder"):
            role_identity = 0.34
        else:
            role_identity = 0.26
    role_identity_pre_boost = max(0.0, min(1.0, role_identity))
    role_identity = _marquee_role_identity_boost(p, role_identity_pre_boost)
    elite = float(_elite_player_signal(p))
    squad_truth = 1.0  # scored players are already in the active current squad.
    transfer_core = 0.0
    if bool(hd.get("valid_current_squad_new_to_franchise")):
        transfer_core = max(
            float(hd.get("probable_first_choice_prior") or 0.0),
            float(hd.get("global_selection_frequency") or 0.0),
        )
    components = {
        "squad_truth": squad_truth,
        "last_xi_continuity": continuity,
        "strong_ipl_history": ipl_hist,
        "stable_role_identity_pre_boost": role_identity_pre_boost,
        "stable_role_identity_role_boost": max(0.0, role_identity - role_identity_pre_boost),
        "stable_role_identity": role_identity,
        "elite_player_sanity": elite,
        "transfer_core_boost": transfer_core,
    }
    score = (
        0.08 * components["squad_truth"]
        + 0.24 * components["last_xi_continuity"]
        + 0.27 * components["strong_ipl_history"]
        + 0.19 * components["stable_role_identity"]
        + 0.16 * components["elite_player_sanity"]
        + 0.06 * components["transfer_core_boost"]
    )
    return max(0.0, min(1.0, score)), components


def _suggested_marquee_tier(
    score: float,
    components: Optional[dict[str, float]] = None,
    *,
    rank_pct: float = 0.0,
) -> str:
    comps = components or {}
    role_identity = float(comps.get("stable_role_identity") or 0.0)
    transfer_core = float(comps.get("transfer_core_boost") or 0.0)
    elite = float(comps.get("elite_player_sanity") or 0.0)
    ipl_hist = float(comps.get("strong_ipl_history") or 0.0)

    # Tier 1 must remain rare.
    if score >= 0.78 and rank_pct >= 0.92:
        return "tier_1"
    # Promote obvious transferred core players when evidence is strong.
    if score >= 0.72 and transfer_core >= 0.82 and elite >= 0.72 and rank_pct >= 0.85:
        return "tier_1"

    # Tier 2 should represent strong first-XI core.
    if score >= 0.58 and rank_pct >= 0.58:
        return "tier_2"
    if score >= 0.54 and rank_pct >= 0.52 and role_identity >= 0.62 and (elite >= 0.58 or ipl_hist >= 0.60):
        return "tier_2"
    if score >= 0.54 and rank_pct >= 0.52 and transfer_core >= 0.72 and (elite >= 0.58 or ipl_hist >= 0.55):
        return "tier_2"

    # Tier 3 for useful role players; avoid catch-all crowding.
    if score >= 0.49 and rank_pct >= 0.40:
        return "tier_3"
    return ""


def _annotate_marquee_tags(players: list[SquadPlayer]) -> None:
    overrides = _load_curated_marquee_overrides()
    suggested: list[tuple[SquadPlayer, float, dict[str, float]]] = []
    for p in players:
        sug_score, sug_components = _suggested_marquee_score(p)
        suggested.append((p, sug_score, sug_components))

    # Rank-based shaping prevents compressed suggested scores from collapsing tiers.
    ordered = sorted(
        suggested,
        key=lambda x: (-float(x[1]), learner.normalize_player_key(getattr(x[0], "name", "") or "")),
    )
    n = len(ordered)
    rank_map: dict[str, float] = {}
    for idx, (p, _s, _c) in enumerate(ordered):
        nk = learner.normalize_player_key(getattr(p, "player_key", "") or getattr(p, "name", ""))
        if n <= 1:
            rp = 1.0
        else:
            rp = 1.0 - (float(idx) / float(n - 1))
        rank_map[nk] = max(0.0, min(1.0, rp))

    for p, sug_score_raw, sug_components in suggested:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        nk_name = learner.normalize_player_key(p.name)
        nk_key = learner.normalize_player_key(getattr(p, "player_key", "") or "")
        ov = overrides.get(nk_key) or overrides.get(nk_name) or {}
        rank_pct = float(rank_map.get(nk_key) or rank_map.get(nk_name) or 0.0)
        sug_score = max(0.0, min(1.0, 0.60 * float(sug_score_raw) + 0.40 * rank_pct))
        sug_tier = _suggested_marquee_tier(sug_score, sug_components, rank_pct=rank_pct)
        if ov:
            tier = str(ov.get("marquee_tier") or "").strip().lower()
            source = "curated"
            reason = str(ov.get("marquee_reason") or "curated_override").strip()
        elif sug_tier:
            tier = sug_tier
            source = "suggested"
            reason = (
                "suggested_from_squad_truth_continuity_ipl_role_elite_transfer "
                f"(score={round(sug_score, 3)})"
            )
        else:
            tier = ""
            source = "suggested"
            reason = f"below_marquee_threshold(score={round(sug_score, 3)})"
        hd["marquee_tier"] = tier
        hd["marquee_source"] = source
        hd["marquee_reason"] = reason
        hd["marquee_suggested_score"] = round(sug_score, 5)
        hd["marquee_suggested_score_raw"] = round(float(sug_score_raw), 5)
        hd["marquee_suggested_rank_pct"] = round(rank_pct, 5)
        hd["marquee_suggested_components"] = {k: round(float(v), 5) for k, v in sug_components.items()}


def _role_identity_score(p: SquadPlayer) -> float:
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    return float(bb.get("stable_role_identity_score") or 0.0)


def _is_bowling_composition_band(b: str) -> bool:
    return b in (
        "powerplay_bowler",
        "middle_overs_spinner",
        "death_bowler",
        "utility_bowler",
        "bowling_allrounder",
    )


def _base_xi_rank_value(p: SquadPlayer) -> float:
    """
    Priority-ordered base rank:
      - availability handled earlier
      - marquee/core tier strongly biases selection (but is still structure-repairable)
      - then continuity/form/history/role identity/elite sanity
    """
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
    cont = float(bb.get("last_match_continuity_score") or 0.0)
    rf = float(bb.get("recent_form_score") or 0.0)
    ipl = float(bb.get("ipl_history_and_role_score") or 0.0)
    role_id = _role_identity_score(p)
    elite = _elite_player_signal(p)
    base = 0.30 * cont + 0.24 * rf + 0.21 * ipl + 0.13 * role_id + 0.12 * elite

    tier = str(hd.get("marquee_tier") or "").strip().lower()
    tier_bonus = 0.0
    if tier == "tier_1":
        tier_bonus = 0.28
    elif tier == "tier_2":
        tier_bonus = 0.18
    elif tier == "tier_3":
        tier_bonus = 0.07
    else:
        tier_bonus = -0.02

    return float(max(0.0, min(1.0, base + _core_anchor_strength(p) + tier_bonus)))


def _tier_priority_order_bonus(p: SquadPlayer) -> float:
    """
    Small explicit ordering bonus so XI construction prefers curated core tiers before
    fringe untiered options when cricketing scores are otherwise close.
    """
    tier = str(((getattr(p, "history_debug", None) or {}).get("marquee_tier") or "")).strip().lower()
    if tier == "tier_1":
        return 0.14
    if tier == "tier_2":
        return 0.08
    if tier == "tier_3":
        return 0.025
    return -0.03


def _allowed_condition_swaps(conditions: dict[str, Any], scenario_branch: Optional[str]) -> int:
    if scenario_branch not in ("if_team_bats_first", "if_team_bowls_first"):
        return 0
    spin_f = float(conditions.get("spin_friendliness", 0.5))
    pace_b = float(conditions.get("pace_bias", 0.5))
    dew = float(conditions.get("dew_risk", 0.5))
    rain = float(conditions.get("rain_disruption_risk", 0.0))
    extreme = (
        abs(spin_f - 0.5) >= 0.15
        or abs(pace_b - 0.5) >= 0.15
        or dew >= 0.67
        or rain >= 0.45
    )
    if extreme:
        return int(getattr(config, "XI_MAX_CONDITION_SWAPS_EXTREME", 2))
    return int(getattr(config, "XI_MAX_CONDITION_SWAPS_NORMAL", 1))


def _try_build_xi(
    sorted_pool: list[SquadPlayer],
    full_pool: list[SquadPlayer],
    penalties: Optional[dict[str, float]] = None,
    *,
    scenario_branch: Optional[str] = None,
    rank_fn: Optional[Callable[[SquadPlayer], float]] = None,
    conditions: Optional[dict[str, Any]] = None,
) -> Optional[list[SquadPlayer]]:
    pen = penalties or {}
    rs: Callable[[SquadPlayer], float] = rank_fn or (
        lambda x: _scenario_xi_rank_value(x, scenario_branch)
    )
    xi: list[SquadPlayer] = []
    used = set()
    for p in sorted_pool:
        if len(xi) >= 11:
            break
        xi.append(p)
        used.add(p.name)
    ok, _ = _validate_xi(xi, conditions=conditions)
    if ok:
        return xi

    # Repair loop
    pool_by_name = {q.name: q for q in full_pool}
    order = sorted(
        full_pool,
        key=lambda x: (
            rs(x) - pen.get(x.name, 0.0),
            _xi_selection_tier(x),
            x.composite,
        ),
        reverse=True,
    )

    def rebuild_from(names: list[str]) -> list[SquadPlayer]:
        return [pool_by_name[n] for n in names if n in pool_by_name]

    def _is_protected_top_order(p: SquadPlayer) -> bool:
        hd = getattr(p, "history_debug", None) or {}
        band = str(hd.get("role_band") or "")
        if band in ("opener", "top_order"):
            return True
        if _elite_player_signal(p) >= 0.5 and band in ("wicketkeeper_batter", "middle_order"):
            return True
        return False

    names = [p.name for p in xi]

    for _ in range(240):
        cur = rebuild_from(names)
        ok, errs = _validate_xi(cur, conditions=conditions)
        if ok:
            return cur
        # Remove weakest fixable player and try replacements
        if any("Overseas" in e for e in errs):
            os_players = [p for p in cur if p.is_overseas]
            if not os_players:
                break
            drop = min(os_players, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name]
            nxt = next((q.name for q in order if q.name not in names and not q.is_overseas), None)
            if nxt:
                names.append(nxt)
            continue
        if any("wicketkeeper" in e for e in errs):
            wks = [q for q in order if q.is_wicketkeeper and q.name not in names]
            if not wks:
                break
            add = wks[0].name
            non_essential = [p for p in cur if not p.is_wicketkeeper]
            if not non_essential:
                break
            drop = min(non_essential, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name] + [add]
            continue
        if any("Bowling options" in e for e in errs):
            bow_candidates = [q for q in order if _is_bowling_option(q) and q.name not in names]
            if not bow_candidates:
                break
            add = bow_candidates[0].name
            non_bowlers = [
                p
                for p in cur
                if not _is_bowling_option(p) and not p.is_wicketkeeper and not _is_protected_top_order(p)
            ]
            if not non_bowlers:
                non_bowlers = [p for p in cur if not _is_bowling_option(p) and not _is_protected_top_order(p)]
            if not non_bowlers:
                non_bowlers = [p for p in cur if not _is_bowling_option(p)]
            if not non_bowlers:
                break
            drop = min(non_bowlers, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name] + [add]
            continue
        if any("Pace options" in e for e in errs):
            adds = [q for q in order if _is_pace_bowler_candidate(q) and q.name not in names]
            if not adds:
                break
            add = adds[0].name
            drops = [p for p in cur if _is_spinner_candidate(p) and not _must_lock_in_base_xi(p)]
            if not drops:
                drops = [p for p in cur if not _is_pace_bowler_candidate(p) and not _must_lock_in_base_xi(p)]
            if not drops:
                break
            drop = min(drops, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Spinner options" in e for e in errs):
            adds = [q for q in order if _is_spinner_candidate(q) and q.name not in names]
            if not adds:
                break
            add = adds[0].name
            drops = [p for p in cur if _is_pace_bowler_candidate(p) and not _must_lock_in_base_xi(p)]
            if not drops:
                drops = [p for p in cur if not _is_spinner_candidate(p) and not _must_lock_in_base_xi(p)]
            if not drops:
                break
            drop = min(drops, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Proper batters" in e for e in errs):
            adds = [q for q in order if _is_proper_batter(q) and q.name not in names]
            drops = [p for p in cur if p.role_bucket == BOWLER and not _must_lock_in_base_xi(p)]
            if adds and drops:
                names = [n for n in names if n != min(drops, key=lambda x: rs(x)).name] + [adds[0].name]
                continue
            break
        if any("All-rounders" in e and "<" in e for e in errs):
            adds = [q for q in order if q.role_bucket == ALL_ROUNDER and q.name not in names]
            drops = [p for p in cur if p.role_bucket == BOWLER and not _must_lock_in_base_xi(p)]
            if adds and drops:
                names = [n for n in names if n != min(drops, key=lambda x: rs(x)).name] + [adds[0].name]
                continue
            break
        if any("All-rounders" in e and ">" in e for e in errs):
            drops = [p for p in cur if p.role_bucket == ALL_ROUNDER and not _must_lock_in_base_xi(p)]
            adds = [q for q in order if q.role_bucket != ALL_ROUNDER and q.name not in names]
            if adds and drops:
                names = [n for n in names if n != min(drops, key=lambda x: rs(x)).name] + [adds[0].name]
                continue
            break
        if any("Powerplay options" in e for e in errs):
            pool = []
            for q in order:
                if q.name in names:
                    continue
                qh = getattr(q, "history_debug", None) or {}
                qd = qh.get("derive_player_profile") if isinstance(qh.get("derive_player_profile"), dict) else {}
                try:
                    ppl = float(qd.get("powerplay_bowler_likelihood") or 0.0)
                except (TypeError, ValueError):
                    ppl = 0.0
                if ppl >= float(getattr(config, "HISTORY_PHASE_SPECIALIST_THRESHOLD", 0.48)):
                    pool.append(q)
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if not _is_bowling_option(p) and not _is_protected_top_order(p)]
            if not drop_candidates:
                drop_candidates = [p for p in cur if not _is_bowling_option(p)]
            if not drop_candidates:
                drop_candidates = [p for p in cur]
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Death options" in e for e in errs):
            pool = []
            for q in order:
                if q.name in names:
                    continue
                qh = getattr(q, "history_debug", None) or {}
                qd = qh.get("derive_player_profile") if isinstance(qh.get("derive_player_profile"), dict) else {}
                try:
                    dth = float(qd.get("death_bowler_likelihood") or 0.0)
                except (TypeError, ValueError):
                    dth = 0.0
                if dth >= float(getattr(config, "HISTORY_PHASE_SPECIALIST_THRESHOLD", 0.48)):
                    pool.append(q)
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if not _is_bowling_option(p) and not _is_protected_top_order(p)]
            if not drop_candidates:
                drop_candidates = [p for p in cur if not _is_bowling_option(p)]
            if not drop_candidates:
                drop_candidates = [p for p in cur]
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Batting depth" in e for e in errs):
            bowlers_in = [p for p in cur if p.role_bucket == BOWLER]
            nb_outside = [q for q in order if q.role_bucket != BOWLER and q.name not in names]
            if bowlers_in and nb_outside:
                drop = min(bowlers_in, key=lambda x: rs(x)).name
                add = max(nb_outside, key=lambda x: rs(x)).name
                names = [n for n in names if n != drop] + [add]
            continue
        if any("Opener candidates" in e for e in errs):
            pool = [q for q in order if q.role_bucket in (BATTER, WK_BATTER) and q.name not in names]
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if p.role_bucket == BOWLER]
            if not drop_candidates:
                drop_candidates = [p for p in cur if p.role_bucket not in (BATTER, WK_BATTER)]
            if not drop_candidates:
                break
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Finisher:" in e for e in errs):
            pool = [q for q in order if q.role_bucket == ALL_ROUNDER and q.name not in names]
            if not pool:
                pool = [
                    q
                    for q in order
                    if q.role_bucket == BOWLER and q.bat_skill >= 0.42 and q.name not in names
                ]
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if p.role_bucket == BATTER]
            if not drop_candidates:
                drop_candidates = [p for p in cur if p.role_bucket not in (ALL_ROUNDER, BOWLER)]
            if not drop_candidates:
                break
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("XI size" in e for e in errs):
            if len(names) < 11:
                for q in order:
                    if q.name not in names:
                        names.append(q.name)
                        break
            else:
                names = names[:11]
            continue
        break

    cur = rebuild_from(names)
    ok, _ = _validate_xi(cur, conditions=conditions)
    return cur if ok and len(cur) == 11 else None


def _build_xi_with_hard_role_quotas(
    full_pool: list[SquadPlayer],
    rank_fn: Callable[[SquadPlayer], float],
    *,
    conditions: Optional[dict[str, Any]] = None,
) -> Optional[list[SquadPlayer]]:
    """Deterministic hard-constraint XI constructor when greedy repair cannot find a valid XI."""
    if len(full_pool) < 11:
        return None
    order = sorted(full_pool, key=lambda p: (rank_fn(p), _xi_selection_tier(p), p.composite), reverse=True)
    by_name = {p.name: p for p in order}
    chosen: list[str] = []

    def _add_from(cands: list[SquadPlayer], k: int) -> None:
        for p in cands:
            if len(chosen) >= 11:
                break
            if p.name in chosen:
                continue
            chosen.append(p.name)
            if len([n for n in chosen if n in {x.name for x in cands}]) >= k:
                break

    wk_pool = [p for p in order if p.is_wicketkeeper]
    open_pool = [p for p in order if _is_opener_candidate_strict(p)]
    pp_pool = [p for p in order if _is_powerplay_bowler_candidate(p)]
    de_pool = [p for p in order if _is_death_bowler_candidate(p)]
    _add_from(wk_pool, 1)
    _add_from(open_pool, 2)
    _add_from(pp_pool, 2)
    _add_from(de_pool, 2)
    for p in order:
        if len(chosen) >= 11:
            break
        if p.name not in chosen:
            chosen.append(p.name)
    xi = [by_name[n] for n in chosen[:11] if n in by_name]
    ok, _errs = _validate_xi(xi, conditions=conditions)
    if ok:
        return xi
    # local swap attempts to satisfy any remaining hard constraints
    for _ in range(140):
        ok, errs = _validate_xi(xi, conditions=conditions)
        if ok:
            return xi
        xi_names = {p.name for p in xi}
        add_pool = [p for p in order if p.name not in xi_names]
        replaced = False
        if any("Powerplay options" in e for e in errs):
            adds = [p for p in add_pool if _is_powerplay_bowler_candidate(p)]
            drops = [p for p in xi if not _is_powerplay_bowler_candidate(p) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Pace options" in e for e in errs):
            adds = [p for p in add_pool if _is_pace_bowler_candidate(p)]
            drops = [p for p in xi if (not _is_pace_bowler_candidate(p)) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Spinner options" in e for e in errs):
            adds = [p for p in add_pool if _is_spinner_candidate(p)]
            drops = [p for p in xi if (not _is_spinner_candidate(p)) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Death options" in e for e in errs):
            adds = [p for p in add_pool if _is_death_bowler_candidate(p)]
            drops = [p for p in xi if not _is_death_bowler_candidate(p) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Proper batters" in e for e in errs):
            adds = [p for p in add_pool if _is_proper_batter(p)]
            drops = [p for p in xi if p.role_bucket == BOWLER and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("All-rounders" in e and "<" in e for e in errs):
            adds = [p for p in add_pool if p.role_bucket == ALL_ROUNDER]
            drops = [p for p in xi if p.role_bucket == BOWLER and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("All-rounders" in e and ">" in e for e in errs):
            adds = [p for p in add_pool if p.role_bucket != ALL_ROUNDER]
            drops = [p for p in xi if p.role_bucket == ALL_ROUNDER and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Opener candidates" in e for e in errs):
            adds = [p for p in add_pool if _is_opener_candidate_strict(p)]
            drops = [p for p in xi if not _is_opener_candidate_strict(p) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced and any("Wicketkeepers" in e for e in errs):
            drops = [p for p in xi if p.is_wicketkeeper and _elite_player_signal(p) < 0.5]
            if drops:
                xi.remove(min(drops, key=rank_fn))
                replaced = True
        if not replaced and any("No wicketkeeper" in e for e in errs):
            adds = [p for p in add_pool if p.is_wicketkeeper]
            drops = [p for p in xi if (not p.is_wicketkeeper) and not _must_lock_in_base_xi(p)]
            if adds and drops:
                xi.remove(min(drops, key=rank_fn))
                xi.append(max(adds, key=rank_fn))
                replaced = True
        if not replaced:
            break
    ok, _errs = _validate_xi(xi, conditions=conditions)
    return xi if ok else None


def _xi_selection_tier(p: SquadPlayer) -> int:
    """Prefer specialists for initial fill; all-rounders next; bowlers last (IPL balance)."""
    if p.role_bucket == BATTER:
        return 3
    if p.role_bucket == WK_BATTER:
        return 3
    if p.role_bucket == ALL_ROUNDER:
        return 2
    return 1


def _derive_pattern_venue_keys(venue: VenueProfile, base_keys: list[str]) -> list[str]:
    """Match Stage-2 ``team_selection_patterns.venue_key`` (canonical hash of venue strings)."""
    out: list[str] = []
    seen: set[str] = set()
    for k in base_keys:
        s = str(k or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for attr in ("key", "display_name", "city"):
        v = getattr(venue, attr, None)
        if v and str(v).strip():
            ck = canonical_keys.canonical_player_key(str(v).strip())[:80]
            if ck and ck not in seen:
                seen.add(ck)
                out.append(ck)
    return out


def _precompute_xi_build_penalties(players: list[SquadPlayer]) -> dict[str, float]:
    """
    Soft penalties before greedy XI fill: dull the 5th+ similar bowler type so specialists
    win realistic construction ties without breaking overseas / WK repairs.
    """
    from collections import defaultdict

    rate = float(getattr(config, "STAGE3_SIMILAR_BOWLER_PENALTY", 0.026))
    by_type: dict[str, list[SquadPlayer]] = defaultdict(list)
    for p in players:
        if p.role_bucket != BOWLER:
            continue
        bt = (p.bowling_type or "pace_like").lower().strip() or "pace_like"
        by_type[bt].append(p)
    pen: dict[str, float] = {}
    for group in by_type.values():
        group.sort(key=lambda x: x.selection_score, reverse=True)
        for i, p in enumerate(group):
            if i >= 4:
                pen[p.name] = pen.get(p.name, 0.0) + rate * float(i - 3)
    return pen


def _refine_opener_finisher_from_derive(players: list[SquadPlayer]) -> None:
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        if not prof:
            continue
        try:
            ol = float(prof.get("opener_likelihood") or 0.0)
            fl = float(prof.get("finisher_likelihood") or 0.0)
        except (TypeError, ValueError):
            continue
        if p.role_bucket in (BATTER, WK_BATTER) and ol >= 0.55:
            p.is_opener_candidate = True
        if fl >= 0.57:
            p.is_finisher_candidate = True


def _annotate_bench_xi_margins(squad: list[SquadPlayer], xi: list[SquadPlayer]) -> None:
    xi_set = {p.name for p in xi}
    if len(xi) != 11:
        return
    xi_sorted = sorted(xi, key=lambda p: p.selection_score)
    worst = float(xi_sorted[0].selection_score) if xi_sorted else 0.0
    for p in squad:
        if p.name in xi_set:
            continue
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        p.history_debug["bench_near_xi_margin"] = float(p.selection_score - worst)


def _batting_order_signal_source_ranked(p: SquadPlayer) -> list[str]:
    hd = getattr(p, "history_debug", None) or {}
    src = str(hd.get("batting_order_source") or "").lower()
    final = str(hd.get("batting_order_final") or "")
    ranks: list[str] = []
    if final == "historical_ema_primary":
        ranks.append("batting_position_ema")
    if "h2h" in src:
        ranks.append("head_to_head_slot_blend")
    if "global" in src or "all_franchise" in src:
        ranks.append("global_batting_slot_history")
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
    if prof:
        try:
            if float(prof.get("opener_likelihood") or 0) >= 0.55:
                ranks.append("derive_opener_likelihood")
            if float(prof.get("finisher_likelihood") or 0) >= 0.55:
                ranks.append("derive_finisher_likelihood")
            sm = int(prof.get("sample_matches") or 0)
            bpe = float(prof.get("batting_position_ema") or 0)
            if sm >= 3 and bpe > 0.5:
                ranks.append("derive_batting_position_ema")
        except (TypeError, ValueError):
            pass
    if final in ("role_fallback_proxy", "role_bucket_only"):
        ranks.append("role_bucket_fallback")
    if not ranks:
        ranks.append("role_bucket_fallback")
    dedup: list[str] = []
    seen: set[str] = set()
    for r in ranks:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    return dedup


def _batting_order_reason_summary_for_player(p: SquadPlayer, ranked: list[str]) -> str:
    hd = getattr(p, "history_debug", None) or {}
    ema = float(getattr(p, "history_batting_ema", float(config.HISTORY_BAT_SLOT_UNKNOWN)))
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    parts = [
        f"Ordered using: {' > '.join(ranked)}.",
    ]
    if ema < unk - 1e-6:
        parts.append(f"Primary numeric slot signal ≈ {ema:.2f} (lower bats earlier).")
    else:
        parts.append("No stable scorecard EMA for this XI player; derive opener/finisher hints and role buckets fill gaps.")
    if isinstance(hd.get("derive_player_profile"), dict):
        prof = hd["derive_player_profile"]
        try:
            ol = float(prof.get("opener_likelihood") or 0)
            fl = float(prof.get("finisher_likelihood") or 0)
            if ol >= 0.55 or fl >= 0.55:
                parts.append(
                    f"Stage-2 profile suggests opener={ol:.2f}, finisher={fl:.2f} (0–1 scale)."
                )
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def select_playing_xi(
    scored: list[SquadPlayer],
    *,
    scenario_branch: Optional[str] = None,
    conditions: Optional[dict[str, Any]] = None,
) -> list[SquadPlayer]:
    pen = _precompute_xi_build_penalties(scored)
    rs = lambda x: _scenario_xi_rank_value(x, scenario_branch) + _tier_priority_order_bonus(x)
    order = sorted(
        scored,
        key=lambda x: (
            rs(x) - pen.get(x.name, 0.0),
            _xi_selection_tier(x),
            x.composite,
        ),
        reverse=True,
    )
    xi = _try_build_xi(order, scored, pen, scenario_branch=scenario_branch, conditions=conditions)
    if xi:
        sk = _squad_player_key_set(scored)
        for p in xi:
            pk = (getattr(p, "player_key", None) or "").strip() or learner.normalize_player_key(p.name)
            if pk not in sk:
                logger.error(
                    "select_playing_xi: strict invariant broken — %r not in current squad keys",
                    p.name,
                )
        logger.info(
            "select_playing_xi: selected XI names=%s role_buckets=%s",
            [p.name for p in xi],
            [p.role_bucket for p in xi],
        )
        return xi
    if len(scored) < 11:
        fb = sorted(scored, key=lambda x: (rs(x), x.composite), reverse=True)
        logger.warning("select_playing_xi: short squad len=%d", len(scored))
        return fb
    fb = _build_xi_with_hard_role_quotas(scored, rs, conditions=conditions)
    if fb:
        return fb
    top = order[:11]
    _, errs = _validate_xi(top, conditions=conditions)
    logger.error("select_playing_xi: unable to build valid XI under hard constraints errs=%s", errs)
    return top


def select_base_playing_xi(
    scored: list[SquadPlayer],
    *,
    conditions: Optional[dict[str, Any]] = None,
) -> list[SquadPlayer]:
    """Stage 1: strongest cricket-logical base XI from squad + linked history."""
    pen = _precompute_xi_build_penalties(scored)
    order = sorted(
        scored,
        key=lambda x: (
            _base_xi_rank_value(x) - pen.get(x.name, 0.0),
            _xi_selection_tier(x),
            x.composite,
        ),
        reverse=True,
    )
    xi = _try_build_xi(
        order,
        scored,
        pen,
        scenario_branch=None,
        rank_fn=_base_xi_rank_value,
        conditions=conditions,
    )
    if not xi:
        xi = select_playing_xi(scored, scenario_branch=None, conditions=conditions)
    by_name = {p.name: p for p in scored}
    xi_names = {p.name for p in xi}

    # Hard core lock: include obvious starters (strike bowlers, top-order anchors, captain/WK-significant).
    locked = [p for p in scored if _must_lock_in_base_xi(p)]
    locked.sort(key=lambda p: _base_xi_rank_value(p), reverse=True)
    for lk in locked:
        if lk.name in xi_names:
            continue
        lk_band = _role_band(lk)
        drop_candidates = [
            x
            for x in xi
            if not _must_lock_in_base_xi(x)
            and (
                _is_bowling_composition_band(_role_band(x)) == _is_bowling_composition_band(lk_band)
                or _elite_player_signal(x) < 0.42
            )
        ]
        if not drop_candidates:
            drop_candidates = [x for x in xi if not _must_lock_in_base_xi(x)]
        drop_candidates.sort(key=lambda p: (_elite_player_signal(p), _base_xi_rank_value(p)))
        for d in drop_candidates:
            trial_names = [n for n in xi_names if n != d.name] + [lk.name]
            xi_trial = [by_name[n] for n in trial_names if n in by_name]
            ok, _errs = _validate_xi(xi_trial, conditions=conditions)
            if ok:
                xi = xi_trial
                xi_names = {p.name for p in xi}
                break

    # Last-XI continuity: carry forward at least 7 (prefer 8) when available.
    last_xi_all = [p for p in scored if bool(((getattr(p, "history_debug", None) or {}).get("selection_model_debug") or {}).get("last_match_detail", {}).get("was_in_last_match_xi"))]
    continuity_available = len(last_xi_all)
    continuity_floor = min(8, continuity_available) if continuity_available >= 7 else continuity_available
    def _continuity_in_xi(cur: list[SquadPlayer]) -> int:
        return sum(
            1
            for p in cur
            if bool(((getattr(p, "history_debug", None) or {}).get("selection_model_debug") or {}).get("last_match_detail", {}).get("was_in_last_match_xi"))
        )
    if _continuity_in_xi(xi) < continuity_floor:
        need = continuity_floor - _continuity_in_xi(xi)
        candidates = [p for p in last_xi_all if p.name not in xi_names]
        candidates.sort(key=lambda p: _base_xi_rank_value(p), reverse=True)
        for add in candidates:
            if need <= 0:
                break
            drop_candidates = [
                x
                for x in xi
                if not bool(((getattr(x, "history_debug", None) or {}).get("selection_model_debug") or {}).get("last_match_detail", {}).get("was_in_last_match_xi"))
                and not _must_lock_in_base_xi(x)
            ]
            drop_candidates.sort(key=lambda p: (_elite_player_signal(p), _base_xi_rank_value(p)))
            for d in drop_candidates:
                trial_names = [n for n in xi_names if n != d.name] + [add.name]
                xi_trial = [by_name[n] for n in trial_names if n in by_name]
                ok, _errs = _validate_xi(xi_trial, conditions=conditions)
                if ok:
                    xi = xi_trial
                    xi_names = {p.name for p in xi}
                    need -= 1
                    break

    # Elite sanity guardrail: obvious core players should not drop without strong reason.
    elite_pool = [p for p in scored if _elite_player_signal(p) >= 0.48]
    elite_pool.sort(key=lambda p: _base_xi_rank_value(p), reverse=True)
    for ep in elite_pool:
        if ep.name in xi_names:
            continue
        ep_band = str((getattr(ep, "history_debug", None) or {}).get("role_band") or "")
        drop_candidates = [x for x in xi if _elite_player_signal(x) < 0.46]
        if _is_bowling_composition_band(ep_band):
            drop_candidates = [
                x
                for x in drop_candidates
                if _is_bowling_composition_band(
                    str((getattr(x, "history_debug", None) or {}).get("role_band") or "")
                )
            ] or drop_candidates
        else:
            drop_candidates = [
                x
                for x in drop_candidates
                if not _is_bowling_composition_band(
                    str((getattr(x, "history_debug", None) or {}).get("role_band") or "")
                )
            ] or drop_candidates
        drop_candidates.sort(key=lambda p: _base_xi_rank_value(p))
        swapped = False
        for d in drop_candidates:
            trial_names = [n for n in xi_names if n != d.name] + [ep.name]
            xi_trial = [by_name[n] for n in trial_names if n in by_name]
            ok, _errs = _validate_xi(xi_trial, conditions=conditions)
            if ok:
                xi = xi_trial
                xi_names = {p.name for p in xi}
                swapped = True
                break
        if not swapped:
            continue

    # Structure sanity: preserve at least two strong top-order core options in XI when available.
    top_core_all = [
        p
        for p in scored
        if str((getattr(p, "history_debug", None) or {}).get("role_band") or "") in ("opener", "top_order")
        and _elite_player_signal(p) >= 0.44
    ]
    top_core_all.sort(key=lambda p: _base_xi_rank_value(p), reverse=True)
    top_cur = [
        p
        for p in xi
        if str((getattr(p, "history_debug", None) or {}).get("role_band") or "") in ("opener", "top_order")
        and _elite_player_signal(p) >= 0.44
    ]
    for tc in top_core_all:
        if len(top_cur) >= 2:
            break
        if tc.name in xi_names:
            continue
        drop_candidates = [
            p
            for p in xi
            if str((getattr(p, "history_debug", None) or {}).get("role_band") or "")
            not in ("opener", "top_order", "wicketkeeper_batter")
            and _elite_player_signal(p) < 0.5
        ]
        drop_candidates.sort(key=lambda p: _base_xi_rank_value(p))
        for d in drop_candidates:
            trial_names = [n for n in xi_names if n != d.name] + [tc.name]
            xi_trial = [by_name[n] for n in trial_names if n in by_name]
            ok, _errs = _validate_xi(xi_trial, conditions=conditions)
            if ok:
                xi = xi_trial
                xi_names = {p.name for p in xi}
                top_cur = [
                    p
                    for p in xi
                    if str((getattr(p, "history_debug", None) or {}).get("role_band") or "") in ("opener", "top_order")
                    and _elite_player_signal(p) >= 0.44
                ]
                break
    for p in scored:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        hd["base_xi_score"] = round(_base_xi_rank_value(p), 5)
        hd["base_xi_selected"] = p.name in xi_names
        hd["core_anchor_strength"] = round(_core_anchor_strength(p), 5)
        hd["elite_core_signal"] = round(_elite_player_signal(p), 5)
        if p.name in xi_names:
            hd["base_xi_reason"] = "Selected in Stage-1 strongest XI from recent form + IPL history + team balance."
        else:
            hd["base_xi_reason"] = "Outside Stage-1 strongest XI score/rank after role-balance repairs."
    return xi


def _apply_condition_adjustments_from_base(
    scored: list[SquadPlayer],
    base_xi: list[SquadPlayer],
    *,
    scenario_branch: Optional[str],
    conditions: dict[str, Any],
) -> tuple[list[SquadPlayer], list[dict[str, Any]]]:
    """Stage 2: allow only small tactical changes from base XI."""
    def _is_condition_bowling_candidate(p: SquadPlayer) -> bool:
        cls = classify_player(p)
        return bool(
            cls.is_bowling_option
            and (cls.is_spinner or cls.is_pacer or cls.is_structural_all_rounder)
        )

    def _scenario_bonus(p: SquadPlayer) -> tuple[float, str]:
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
        band = _role_band(p)
        cls = classify_player(p)
        spin_f = float(conditions.get("spin_friendliness", 0.5))
        pace_b = float(conditions.get("pace_bias", 0.5))
        dew = float(conditions.get("dew_risk", 0.5))
        bf = float(conditions.get("batting_friendliness", 0.5))
        spin_edge = spin_f - pace_b
        pace_edge = pace_b - spin_f
        b = 0.0
        reason = "neutral"
        if cls.is_spinner and cls.is_bowling_option:
            b += 0.075 * (spin_f - 0.5)
            reason = "spin_friendly_boost"
        if cls.is_spinner and _is_condition_bowling_candidate(p) and spin_f >= 0.55 and spin_edge >= 0.05:
            b += 0.02 + 0.08 * min(0.2, spin_edge)
            reason = "strong_spin_tactical_slot"
        if cls.is_pacer and cls.is_bowling_option:
            b += 0.07 * (pace_b - 0.5)
            if cls.is_pacer and pace_b >= 0.55 and pace_edge >= 0.05:
                b += 0.02 + 0.08 * min(0.2, pace_edge)
                reason = "strong_pace_tactical_slot"
            if dew >= 0.62:
                b += 0.035
                reason = "dew_seam_bias"
            elif reason == "neutral":
                reason = "pace_bias"
        elif cls.is_pacer and _is_condition_bowling_candidate(p) and pace_b >= 0.55 and pace_edge >= 0.05:
            b += 0.018 + 0.06 * min(0.2, pace_edge)
            reason = "strong_pace_tactical_slot"
        if band in ("batting_allrounder", "finisher") and bf >= 0.62 and dew >= 0.58:
            b += 0.03
            reason = "batting_depth_dew"
        if scenario_branch == "if_team_bowls_first" and band in ("powerplay_bowler", "death_bowler"):
            b += 0.02
            reason = "toss_bowls_first_bowler"
        if scenario_branch == "if_team_bats_first" and band in ("opener", "top_order"):
            b += 0.012
            reason = "toss_bats_first_top_order"
        # small anchor damping: core players are less scenario-volatile.
        b *= (1.0 - min(0.45, _elite_player_signal(p) * 0.5))
        return float(b), reason

    max_swaps = max(0, _allowed_condition_swaps(conditions, scenario_branch))
    if max_swaps == 0:
        return base_xi, []
    base_names = {p.name for p in base_xi}
    adds = [p for p in scored if p.name not in base_names]
    drops = [p for p in base_xi]
    if not adds or not drops:
        return base_xi, []

    by_name = {p.name: p for p in scored}
    final_names = list(base_names)
    extreme_ctx = max_swaps >= int(getattr(config, "XI_MAX_CONDITION_SWAPS_EXTREME", 2))
    min_gain = float(getattr(config, "XI_CONDITION_SWAP_MIN_GAIN", 0.012))
    if extreme_ctx:
        min_gain = min(min_gain, 0.002)
    adds_sorted = sorted(
        adds,
        key=lambda p: (
            (_scenario_xi_rank_value(p, scenario_branch) - _base_xi_rank_value(p)) + _scenario_bonus(p)[0]
        ),
        reverse=True,
    )
    changes: list[dict[str, Any]] = []
    used_adds: set[str] = set()
    swaps_done = 0
    for add in adds_sorted:
        if swaps_done >= max_swaps:
            break
        # Conditions should mostly tune bowling composition, not top-order batting personnel.
        if not _is_condition_bowling_candidate(add):
            continue
        add_bonus, add_driver = _scenario_bonus(add)
        add_gain = (_scenario_xi_rank_value(add, scenario_branch) - _base_xi_rank_value(add)) + add_bonus
        drop_pool = [d for d in drops if d.name in final_names and d.name not in used_adds]
        drop_pool = sorted(
            drop_pool,
            key=lambda p: (
                _scenario_xi_rank_value(p, scenario_branch) - _base_xi_rank_value(p),
                _core_anchor_strength(p),
                _scenario_xi_rank_value(p, scenario_branch),
            ),
        )
        chosen_drop: Optional[SquadPlayer] = None
        for d in drop_pool:
            if not _is_condition_bowling_candidate(d):
                continue
            anchor = _core_anchor_strength(d)
            # Core players need a larger tactical delta to be displaced.
            if anchor >= 0.11 and add_gain < (min_gain + 0.028):
                continue
            if _elite_player_signal(d) >= 0.54:
                continue
            drop_bonus, _drop_driver = _scenario_bonus(d)
            scen_gain = (
                _scenario_xi_rank_value(add, scenario_branch)
                - _scenario_xi_rank_value(d, scenario_branch)
                + add_bonus
                - drop_bonus
            )
            if scen_gain < min_gain:
                continue
            trial = [n for n in final_names if n != d.name] + [add.name]
            xi_trial = [by_name[n] for n in trial if n in by_name]
            ok, _errs = _validate_xi(xi_trial)
            if ok:
                chosen_drop = d
                break
        if chosen_drop is None:
            continue
        final_names = [n for n in final_names if n != chosen_drop.name] + [add.name]
        used_adds.add(add.name)
        swaps_done += 1
        changes.append(
            {
                "out": chosen_drop.name,
                "in": add.name,
                "reason": (
                    f"Condition swap ({scenario_branch}) adjusts bowling composition only (spinner/pacer/AR balance) "
                    f"while keeping XI constraints; "
                    f"driver={add_driver}; limited change budget {swaps_done}/{max_swaps}."
                ),
                "scenario_gain": round(
                    _scenario_xi_rank_value(add, scenario_branch)
                    - _scenario_xi_rank_value(chosen_drop, scenario_branch),
                    5,
                ),
            }
        )

    if not changes:
        bench_bowl = [p for p in scored if p.name not in final_names and _is_condition_bowling_candidate(p)]
        xi_bowl = [by_name[n] for n in final_names if n in by_name and _is_condition_bowling_candidate(by_name[n])]
        bench_bowl.sort(key=lambda p: (_scenario_bonus(p)[0], _scenario_xi_rank_value(p, scenario_branch)), reverse=True)
        xi_bowl.sort(key=lambda p: (_scenario_bonus(p)[0], _scenario_xi_rank_value(p, scenario_branch)))
        min_forced_gain = 0.0 if extreme_ctx else 0.008
        for add in bench_bowl[:3]:
            for drop in xi_bowl[:3]:
                if _must_lock_in_base_xi(drop) or _elite_player_signal(drop) >= 0.54:
                    continue
                raw_gain = (
                    (_scenario_bonus(add)[0] - _scenario_bonus(drop)[0])
                    + _scenario_xi_rank_value(add, scenario_branch)
                    - _scenario_xi_rank_value(drop, scenario_branch)
                )
                if raw_gain < min_forced_gain:
                    continue
                trial = [n for n in final_names if n != drop.name] + [add.name]
                xi_trial = [by_name[n] for n in trial if n in by_name]
                ok, _errs = _validate_xi(xi_trial)
                if ok:
                    final_names = trial
                    changes.append(
                        {
                            "out": drop.name,
                            "in": add.name,
                            "reason": (
                                f"{'Extreme' if extreme_ctx else 'Scenario'} forced bowling-balance tweak "
                                f"({scenario_branch}); spinner/pacer role fit improvement."
                            ),
                            "scenario_gain": round(raw_gain, 5),
                        }
                    )
                    break
            if changes:
                break

    final_xi = [by_name[n] for n in final_names if n in by_name]
    if len(final_xi) != 11:
        return base_xi, []

    return final_xi, changes


def _reconcile_condition_changes_and_annotate(
    scored: list[SquadPlayer],
    base_xi: list[SquadPlayer],
    final_xi: list[SquadPlayer],
    *,
    scenario_branch: Optional[str],
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_names = {p.name for p in base_xi}
    final_names = {p.name for p in final_xi}
    surviving: list[dict[str, Any]] = []
    for raw in changes or []:
        add_name = str(raw.get("in") or "").strip()
        drop_name = str(raw.get("out") or "").strip()
        if add_name and drop_name and add_name in final_names and drop_name not in final_names:
            surviving.append(dict(raw))

    by_name = {p.name: p for p in scored}
    for p in scored:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        hd["included_from_base_xi"] = p.name in base_names
        hd["included_in_final_xi"] = p.name in final_names
        hd["condition_adjustment_reason"] = ""
        hd["included_due_to_conditions"] = False
        hd["excluded_due_to_conditions"] = False

    for c in surviving:
        in_p = by_name.get(str(c.get("in") or ""))
        out_p = by_name.get(str(c.get("out") or ""))
        if in_p:
            in_p.history_debug["condition_adjustment_reason"] = str(c.get("reason") or "")
            in_p.history_debug["included_due_to_conditions"] = True
        if out_p:
            out_p.history_debug["condition_adjustment_reason"] = (
                f"Excluded in Stage-2 condition swap ({scenario_branch}) after base XI selection."
            )
            out_p.history_debug["excluded_due_to_conditions"] = True
    return surviving


def _repair_xi_if_needed(
    scored: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    conditions: Optional[dict[str, Any]] = None,
) -> tuple[list[SquadPlayer], bool, list[dict[str, Any]], dict[str, Any]]:
    res0 = rules_xi.validate_xi(xi, conditions=conditions, squad=scored)
    if res0.hard_ok and not res0.warnings:
        return xi, False, [], {
            "hard_constraints_satisfied": True,
            "failed_constraints": [],
            "repair_failure_reason": "",
            "semi_hard_failed": [],
        }
    repaired = list(xi)
    final_errs = [v.message for v in res0.violations]
    final_semi_errs = [w.message for w in res0.warnings]
    applied_rule_labels: list[str] = []

    def _q_rank(p: SquadPlayer) -> float:
        return 0.7 * float(getattr(p, "selection_score", 0.0) or 0.0) + 0.3 * _base_xi_rank_value(p)

    def _drop_safe(p: SquadPlayer, *, allow_locked: bool = False) -> bool:
        if (not allow_locked) and _must_lock_in_base_xi(p):
            return False
        if _elite_player_signal(p) >= 0.68:
            return False
        if _core_anchor_strength(p) >= 0.16:
            return False
        rb = str(((getattr(p, "history_debug", None) or {}).get("role_band") or "")).strip()
        if rb in ("opener", "top_order") and _elite_player_signal(p) >= 0.5:
            return False
        return True

    def _best_swap(
        cur_xi: list[SquadPlayer],
        adds: list[SquadPlayer],
        drops: list[SquadPlayer],
        hard_before: set[str],
        semi_before: set[str],
        *,
        min_gain: float,
        max_quality_drop: float,
    ) -> Optional[tuple[list[SquadPlayer], SquadPlayer, SquadPlayer, list[str], list[str]]]:
        if not adds or not drops:
            return None
        best: Optional[
            tuple[
                float,
                float,
                float,
                str,
                str,
                list[SquadPlayer],
                SquadPlayer,
                SquadPlayer,
                list[str],
                list[str],
            ]
        ] = None
        adds_sorted = sorted(adds, key=_q_rank, reverse=True)[:18]
        drops_sorted = sorted(drops, key=_q_rank)[:18]
        for add in adds_sorted:
            for drop in drops_sorted:
                if add.name == drop.name:
                    continue
                gain = _q_rank(add) - _q_rank(drop)
                if gain < min_gain:
                    continue
                if -gain > max_quality_drop:
                    continue
                trial = [p for p in cur_xi if p.name != drop.name] + [add]
                res_t = rules_xi.validate_xi(trial, conditions=conditions, squad=scored)
                hard_t = {v.code for v in res_t.violations}
                semi_t = {w.code for w in res_t.warnings}

                # Conflict-safe: do not "fix" one hard rule by breaking a different hard rule.
                if not hard_t.issubset(hard_before):
                    continue

                hard_improve = len(hard_before) - len(hard_t)
                semi_improve = len(semi_before) - len(semi_t)
                if hard_improve < 0:
                    continue
                if hard_improve == 0 and semi_improve <= 0:
                    continue
                if hard_before and hard_improve <= 0:
                    continue

                rank_key = (
                    float(hard_improve),
                    float(semi_improve),
                    float(gain),
                    str(add.name),
                    str(drop.name),
                )
                if best is None or rank_key > (best[0], best[1], best[2], best[3], best[4]):
                    best = (
                        rank_key[0],
                        rank_key[1],
                        rank_key[2],
                        rank_key[3],
                        rank_key[4],
                        trial,
                        add,
                        drop,
                        [v.message for v in res_t.violations],
                        [w.message for w in res_t.warnings],
                    )
                    if res_t.hard_ok and not res_t.warnings and gain >= 0:
                        break
            if best is not None and best[0] > 0 and best[2] >= 0:
                break
        if best is None:
            return None
        return (best[5], best[6], best[7], best[8], best[9])
    # Local repair first: preserve core/locked players and swap fringe players only.
    for iter_idx in range(120):
        res_i = rules_xi.validate_xi(repaired, conditions=conditions, squad=scored)
        hard_codes = {v.code for v in res_i.violations}
        semi_codes = {w.code for w in res_i.warnings}
        final_errs = [v.message for v in res_i.violations]
        final_semi_errs = [w.message for w in res_i.warnings]
        if res_i.hard_ok and not res_i.warnings:
            break
        rep_names = {p.name for p in repaired}
        add_pool = [p for p in scored if p.name not in rep_names]
        strict_phase = iter_idx < 80
        min_gain = -0.015 if strict_phase else -0.12
        max_quality_drop = 0.05 if strict_phase else 0.2
        replaced = False
        if "designated_keeper" in hard_codes:
            adds = [p for p in add_pool if classify_player(p).is_wk_role_player]
            drops = [p for p in repaired if (not classify_player(p).is_wk_role_player) and _drop_safe(p)]
            if not drops:
                drops = [
                    p
                    for p in repaired
                    if (not classify_player(p).is_wk_role_player) and _drop_safe(p, allow_locked=True)
                ]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"designated_keeper:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and "bowling_options_min" in hard_codes:
            adds = [p for p in add_pool if classify_player(p).is_bowling_option]
            drops = [p for p in repaired if (not classify_player(p).is_bowling_option) and _drop_safe(p)]
            if not drops:
                drops = [
                    p
                    for p in repaired
                    if (not classify_player(p).is_bowling_option) and _drop_safe(p, allow_locked=True)
                ]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"bowling_options:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and "pacers_min" in hard_codes:
            adds = [p for p in add_pool if classify_player(p).is_pacer]
            drops = [p for p in repaired if (not classify_player(p).is_pacer) and _drop_safe(p)]
            if not drops:
                drops = [
                    p for p in repaired if (not classify_player(p).is_pacer) and _drop_safe(p, allow_locked=True)
                ]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"pacers_min:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and "spinners_min" in hard_codes:
            adds = [p for p in add_pool if classify_player(p).is_spinner]
            drops = [p for p in repaired if (not classify_player(p).is_spinner) and _drop_safe(p)]
            if not drops:
                drops = [
                    p
                    for p in repaired
                    if (not classify_player(p).is_spinner) and _drop_safe(p, allow_locked=True)
                ]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"spinners_min:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and ("overseas_min" in hard_codes or "overseas_max" in hard_codes):
            if "overseas_min" in hard_codes:
                adds = [p for p in add_pool if p.is_overseas]
                drops = [p for p in repaired if (not p.is_overseas) and _drop_safe(p)]
                if not drops:
                    drops = [p for p in repaired if (not p.is_overseas) and _drop_safe(p, allow_locked=True)]
            else:
                adds = [p for p in add_pool if not p.is_overseas]
                drops = [p for p in repaired if p.is_overseas and _drop_safe(p)]
                if not drops:
                    drops = [p for p in repaired if p.is_overseas and _drop_safe(p, allow_locked=True)]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"overseas_bounds:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and "wk_role_players_cap" in semi_codes:
            designated_keeper_name = rules_xi.assign_designated_keeper_name(repaired)
            drops = [
                p
                for p in repaired
                if classify_player(p).is_wk_role_player
                and p.name != designated_keeper_name
                and str(((getattr(p, "history_debug", None) or {}).get("marquee_tier") or "").strip().lower())
                not in ("tier_1", "tier_2")
            ]
            adds = [p for p in add_pool if not classify_player(p).is_wk_role_player]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"semi_wk_cap:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced and "structural_all_rounders_cap" in semi_codes:
            drops = [p for p in repaired if classify_player(p).is_structural_all_rounder and _drop_safe(p)]
            adds = [p for p in add_pool if not classify_player(p).is_structural_all_rounder]
            pick = _best_swap(repaired, adds, drops, hard_codes, semi_codes, min_gain=min_gain, max_quality_drop=max_quality_drop)
            if pick is not None:
                repaired, add_p, drop_p, final_errs, final_semi_errs = pick
                applied_rule_labels.append(f"semi_ar_cap:{drop_p.name}->{add_p.name}")
                replaced = True
        if not replaced:
            break
    res2 = rules_xi.validate_xi(repaired, conditions=conditions, squad=scored)
    if not res2.hard_ok:
        repaired = select_playing_xi(scored, conditions=conditions)
    if len(repaired) != 11:
        return repaired, False, [], {
            "hard_constraints_satisfied": False,
            "failed_constraints": ["XI size != 11"],
            "repair_failure_reason": "repair_exhausted_with_incomplete_xi",
            "semi_hard_failed": [w.code for w in res2.warnings],
        }
    res3 = rules_xi.validate_xi(repaired, conditions=conditions, squad=scored)
    if not res3.hard_ok:
        return repaired, False, [], {
            "hard_constraints_satisfied": False,
            "failed_constraints": [v.message for v in res3.violations],
            "repair_failure_reason": "repair_exhausted_no_constraint_safe_swaps_remaining",
            "semi_hard_failed": [w.code for w in res3.warnings],
        }
    before = {p.name for p in xi}
    after = {p.name for p in repaired}
    outs = sorted(before - after)
    ins = sorted(after - before)
    swaps: list[dict[str, Any]] = []
    for i in range(max(len(outs), len(ins))):
        swaps.append(
            {
                "out": outs[i] if i < len(outs) else None,
                "in": ins[i] if i < len(ins) else None,
                "reason": (
                    "Post-selection structure repair (quality-first, minimal swaps). "
                    f"rules={'; '.join(applied_rule_labels[:6])}"
                ),
            }
        )
    return repaired, True, swaps, {
        "hard_constraints_satisfied": True,
        "failed_constraints": [],
        "repair_failure_reason": "",
        "semi_hard_failed": [w.code for w in res3.warnings],
    }


def _optimize_overseas_preference(
    scored: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    conditions: Optional[dict[str, Any]] = None,
    overseas_min_required: int = 3,
    overseas_target: int = 4,
) -> tuple[list[SquadPlayer], dict[str, Any]]:
    """
    Overseas preference + minimum target:
    - keep hard cap <=4
    - if XI has < overseas_min_required, try to reach minimum first
    - then try to reach overseas_target when realistic and continuity-safe
    """
    cur_xi = list(xi)
    overseas_selected = sum(1 for p in cur_xi if p.is_overseas)
    squad_overseas_available = sum(1 for p in scored if p.is_overseas)
    target = max(0, min(int(getattr(config, "MAX_OVERSEAS", 4)), int(overseas_target), squad_overseas_available))
    min_required = max(0, min(target, int(overseas_min_required), squad_overseas_available))
    dbg: dict[str, Any] = {
        "overseas_count_selected": overseas_selected,
        "overseas_target": target,
        "overseas_min_required": min_required,
        "squad_overseas_available": squad_overseas_available,
        "overseas_target_preference_applied": False,
        "overseas_repair_applied": False,
        "best_excluded_overseas_candidates": [],
        "why_4th_overseas_selected_or_not": "",
        "overseas_swaps": [],
    }
    if overseas_selected >= target:
        dbg["why_4th_overseas_selected_or_not"] = "already_at_overseas_cap"
        return cur_xi, dbg

    def _rank(p: SquadPlayer) -> float:
        return 0.72 * float(getattr(p, "selection_score", 0.0) or 0.0) + 0.28 * _base_xi_rank_value(p)

    # Continuity-aware threshold: if recent usage suggests 3 overseas, require larger gain.
    last_xi_overseas = sum(
        1
        for p in scored
        if p.is_overseas
        and bool(
            (
                (
                    (getattr(p, "history_debug", None) or {}).get("selection_model_debug")
                    if isinstance((getattr(p, "history_debug", None) or {}).get("selection_model_debug"), dict)
                    else {}
                ).get("last_match_detail")
                or {}
            ).get("was_in_last_match_xi")
        )
    )
    min_gain_for_target = 0.035 if last_xi_overseas <= 3 else 0.015

    def _candidate_lists(xi_now: list[SquadPlayer]) -> tuple[list[SquadPlayer], list[SquadPlayer]]:
        xi_names_now = {p.name for p in xi_now}
        excluded_os = [p for p in scored if p.name not in xi_names_now and p.is_overseas]
        excluded_os.sort(key=lambda p: (_rank(p), _base_xi_rank_value(p)), reverse=True)
        designated_keeper_name = _assign_designated_keeper(xi_now)
        drops = [
            p
            for p in xi_now
            if (not p.is_overseas)
            and (not _must_lock_in_base_xi(p))
            and (
                str(((getattr(p, "history_debug", None) or {}).get("marquee_tier") or "").strip().lower())
                not in ("tier_1", "tier_2")
            )
            and (not (designated_keeper_name and p.name == designated_keeper_name))
        ]
        if not drops:
            # If curated marquee protection blocks reaching the minimum overseas requirement, allow dropping
            # tier-2 marquee before giving up (tier-1 remains protected here).
            drops = [
                p
                for p in xi_now
                if (not p.is_overseas)
                and (not _must_lock_in_base_xi(p))
                and (
                    str(((getattr(p, "history_debug", None) or {}).get("marquee_tier") or "").strip().lower())
                    != "tier_1"
                )
                and (not (designated_keeper_name and p.name == designated_keeper_name))
            ]
        drops.sort(key=lambda p: (_rank(p), _base_xi_rank_value(p)))
        return excluded_os, drops

    def _try_one_swap(
        xi_now: list[SquadPlayer], min_gain: float
    ) -> tuple[list[SquadPlayer], Optional[dict[str, Any]], str]:
        excluded_os, drops = _candidate_lists(xi_now)
        if not excluded_os:
            return xi_now, None, "no_excluded_overseas_candidates"
        if not drops:
            return xi_now, None, "no_safe_non_core_indian_drop_candidate"
        if not dbg["best_excluded_overseas_candidates"]:
            dbg["best_excluded_overseas_candidates"] = [
                {
                    "name": p.name,
                    "selection_score": round(float(getattr(p, "selection_score", 0.0) or 0.0), 5),
                    "base_rank": round(_base_xi_rank_value(p), 5),
                    "effective_rank": round(_rank(p), 5),
                }
                for p in excluded_os[:5]
            ]
        add = excluded_os[0]
        for drop in drops:
            gain = _rank(add) - _rank(drop)
            if gain < min_gain:
                continue
            xi_trial = [p for p in xi_now if p.name != drop.name] + [add]
            ok, _errs = _validate_xi(xi_trial, conditions=conditions)
            if not ok:
                continue
            return (
                xi_trial,
                {
                    "in": add.name,
                    "out": drop.name,
                    "gain": round(gain, 5),
                    "min_gain_required": round(min_gain, 5),
                },
                "applied_swap",
            )
        return xi_now, None, "no_constraint_safe_upgrade_met_gain_threshold"

    # Step 1: reach overseas minimum first (when available).
    while sum(1 for p in cur_xi if p.is_overseas) < min_required:
        # Minimum stage allows slight negative gain if needed for realistic overseas composition.
        xi_next, swap_meta, status = _try_one_swap(cur_xi, min_gain=-0.03)
        if swap_meta is None:
            dbg["why_4th_overseas_selected_or_not"] = f"could_not_reach_min_overseas:{status}"
            dbg["last_xi_overseas_count"] = last_xi_overseas
            dbg["overseas_count_selected"] = sum(1 for p in cur_xi if p.is_overseas)
            return cur_xi, dbg
        dbg["overseas_target_preference_applied"] = True
        dbg["overseas_repair_applied"] = True
        dbg["overseas_swaps"].append({**swap_meta, "stage": "reach_minimum"})
        cur_xi = xi_next

    # Step 2: prefer target overseas if realistic.
    if sum(1 for p in cur_xi if p.is_overseas) < target:
        xi_next, swap_meta, status = _try_one_swap(cur_xi, min_gain=min_gain_for_target)
        if swap_meta is not None:
            dbg["overseas_target_preference_applied"] = True
            dbg["overseas_swaps"].append(
                {**swap_meta, "stage": "prefer_target", "last_xi_overseas_count": last_xi_overseas}
            )
            cur_xi = xi_next
            dbg["why_4th_overseas_selected_or_not"] = "applied_realistic_upgrade_to_reach_target_overseas"
        else:
            dbg["why_4th_overseas_selected_or_not"] = (
                f"kept_current_overseas_count_due_to_realism_or_continuity:{status}"
            )
    else:
        dbg["why_4th_overseas_selected_or_not"] = "already_at_overseas_cap"

    dbg["last_xi_overseas_count"] = last_xi_overseas
    dbg["min_gain_required_for_target"] = round(min_gain_for_target, 5)
    dbg["overseas_count_selected"] = sum(1 for p in cur_xi if p.is_overseas)
    if dbg["overseas_swaps"]:
        dbg["overseas_swap"] = dbg["overseas_swaps"][-1]
    else:
        dbg["overseas_swap"] = None
    return cur_xi, dbg


def _build_batting_order_role_fallback(xi: list[SquadPlayer], conditions: dict[str, Any]) -> list[str]:
    """
    Legacy IPL role-bucket batting stack: openers → top batters → batting AR → WK slot
    → lower AR / finishers → bowlers last. Used when scorecard history is missing per player.
    """
    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])

    if len(xi) != 11:
        out = [p.name for p in sorted(xi, key=lambda x: (-x.bat_skill, x.name))]
        logger.warning("batting_order: incomplete XI len=%d", len(xi))
        return out

    specialists = [p for p in xi if p.role_bucket == BATTER]
    wks = [p for p in xi if p.role_bucket == WK_BATTER]
    ars = [p for p in xi if p.role_bucket == ALL_ROUNDER]
    bows = [p for p in xi if p.role_bucket == BOWLER]

    openers: list[SquadPlayer] = []
    used: set[str] = set()

    def _opener_rank_key(p: SquadPlayer) -> tuple[float, float, str]:
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        try:
            ol = float(prof.get("opener_likelihood") or 0.0) if prof else 0.0
        except (TypeError, ValueError):
            ol = 0.0
        return (-ol, -p.bat_skill, p.name)

    if specialists:
        openers = sorted(specialists, key=_opener_rank_key)[:2]
    for p in openers:
        used.add(p.name)

    if len(openers) < 2:
        for p in sorted([x for x in wks if x.name not in used], key=lambda q: q.bat_skill, reverse=True):
            if len(openers) >= 2:
                break
            openers.append(p)
            used.add(p.name)

    if len(openers) < 2 and not specialists:
        openers = sorted(wks, key=lambda p: p.bat_skill, reverse=True)[:2]
        used = {p.name for p in openers}

    rest_spec = sorted(
        [p for p in specialists if p.name not in used],
        key=lambda p: p.bat_skill,
        reverse=True,
    )

    ar_middle = sorted(
        [p for p in ars if p.bat_skill >= p.bowl_skill + 0.04],
        key=lambda p: p.bat_skill,
        reverse=True,
    )
    ar_lower = [p for p in ars if p not in ar_middle]

    rest_wk = sorted([p for p in wks if p.name not in used], key=lambda p: p.bat_skill, reverse=True)

    middle_line: list[SquadPlayer] = []
    middle_line.extend(openers)
    middle_line.extend(rest_spec)
    middle_line.extend(ar_middle)

    insert_at = min(max(3, int(3 + bf)), len(middle_line))
    for w in rest_wk:
        idx = min(insert_at, len(middle_line))
        middle_line.insert(idx, w)
        insert_at = idx + 2

    def fin_key(p: SquadPlayer) -> float:
        return p.bat_skill * (0.55 + 0.2 * dew) + p.bowl_skill * (0.45 + 0.15 * (1.0 - dew))

    ar_tail = sorted(ar_lower, key=fin_key, reverse=True)
    bow_tail = sorted(bows, key=lambda p: p.bowl_skill, reverse=True)

    ordered_players = middle_line + ar_tail + bow_tail

    logger.info(
        "batting_order: groups openers=%s spec_mid=%s ar_mid=%s ar_tail=%s bowlers=%s",
        [p.name for p in openers],
        [p.name for p in rest_spec],
        [p.name for p in ar_middle],
        [p.name for p in ar_tail],
        [p.name for p in bow_tail],
    )

    seen: set[str] = set()
    final: list[str] = []
    for p in ordered_players:
        if p.name not in seen:
            seen.add(p.name)
            final.append(p.name)
    for p in xi:
        if p.name not in seen:
            final.append(p.name)
            seen.add(p.name)
    return final


def _batting_order_for_short_xi(
    xi: list[SquadPlayer],
    *,
    team_name: str,
    out_warnings: Optional[list[str]] = None,
) -> list[str]:
    names = [p.name for p in sorted(xi, key=lambda x: (-x.bat_skill, x.name))]
    strict, w = _batting_order_strict_names_for_xi(xi, names)
    if out_warnings is not None:
        out_warnings.extend(w)
    for i, n in enumerate(strict):
        p = next((q for q in xi if q.name == n), None)
        if p is None:
            continue
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        p.history_debug["batting_order_strict_xi_scope"] = True
        p.history_debug["batting_order_rank_final"] = i + 1
        p.history_debug["batting_order_diagnostic_source"] = "role_fallback"
    logger.warning("batting_order: incomplete XI len=%d team=%s", len(xi), team_name)
    return strict


def build_batting_order(
    xi: list[SquadPlayer],
    conditions: dict[str, Any],
    *,
    team_name: str,
    venue_keys: list[str],
    out_warnings: Optional[list[str]] = None,
) -> list[str]:
    """
    Primary sort key is historical batting slot (EMA from stored scorecards) for **XI players only**.

    Output names are strictly a permutation of the selected XI (no non-squad / historical-only names).
    """
    if len(xi) != 11:
        return _batting_order_for_short_xi(xi, team_name=team_name, out_warnings=out_warnings)

    for p in xi:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        if not p.history_debug.get("role_band"):
            p.history_debug["role_band"] = _derive_role_band_for_player(p)

    band_rank = {
        "opener": 0,
        "top_order": 1,
        "wicketkeeper_batter": 2,
        "middle_order": 3,
        "batting_allrounder": 4,
        "finisher": 5,
        "bowling_allrounder": 6,
        "powerplay_bowler": 7,
        "middle_overs_spinner": 7,
        "death_bowler": 7,
        "utility_bowler": 8,
    }

    def _ol(p: SquadPlayer) -> float:
        d = (getattr(p, "history_debug", None) or {}).get("derive_player_profile")
        if not isinstance(d, dict):
            return 0.0
        try:
            return float(d.get("opener_likelihood") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _slot(p: SquadPlayer) -> float:
        return float(_batting_position_signal_for_role_band(p))

    def _band(p: SquadPlayer) -> str:
        hd = getattr(p, "history_debug", None) or {}
        raw = str(hd.get("batting_band") or hd.get("role_band") or "middle_order")
        # Normalize to canonical band tokens used by guardrails.
        if raw in ("middle_order", "middle"):
            return "middle"
        if raw in ("batting_allrounder", "wicketkeeper_batter"):
            return "middle"
        return raw

    open_pool = [p for p in xi if _band(p) in ("opener", "top_order")]
    if len(open_pool) < 2:
        open_pool.extend(
            [
                p
                for p in xi
                if _band(p) in ("top_order", "middle")
            ]
        )
    open_pool.sort(
        key=lambda p: (
            _slot(p) - 0.62 * _ol(p) - 0.38 * _elite_player_signal(p),
            -p.bat_skill,
            p.name,
        )
    )
    opener_pick = open_pool[:2]
    opener_names = {p.name for p in opener_pick}

    rem = [p for p in xi if p.name not in opener_names]
    early_bands = {"top_order", "wicketkeeper_batter", "middle_order", "batting_allrounder"}
    late_bands = {"finisher", "bowling_allrounder", "powerplay_bowler", "middle_overs_spinner", "death_bowler", "utility_bowler"}
    early = [p for p in rem if str((p.history_debug or {}).get("role_band") or "") in early_bands]
    late = [p for p in rem if p not in early]
    early.sort(
        key=lambda p: (
            band_rank.get(str((p.history_debug or {}).get("role_band") or ""), 9),
            _slot(p),
            -p.bat_skill,
            p.name,
        )
    )
    late.sort(
        key=lambda p: (
            band_rank.get(str((p.history_debug or {}).get("role_band") or ""), 9),
            _slot(p),
            -p.bowl_skill,
            p.name,
        )
    )
    ordered = opener_pick + early + late
    candidate = [p.name for p in ordered]
    strict_names, bo_w = _batting_order_strict_names_for_xi(xi, candidate)
    if out_warnings is not None:
        out_warnings.extend(bo_w)

    # Elite top-order sanity: keep clear top-order cores inside top 4 when selected.
    by_name = {p.name: p for p in xi}
    top_core = [
        p.name
        for p in xi
        if str((getattr(p, "history_debug", None) or {}).get("role_band") or "") in ("opener", "top_order")
        and _elite_player_signal(p) >= 0.5
    ]
    for nm in top_core:
        if nm not in strict_names:
            continue
        cur_idx = strict_names.index(nm)
        if cur_idx <= 3:
            continue
        swap_idx = None
        for i in range(4):
            cand = strict_names[i]
            p_i = by_name.get(cand)
            if p_i is None:
                continue
            b = str((getattr(p_i, "history_debug", None) or {}).get("role_band") or "")
            if b not in ("opener", "top_order", "wicketkeeper_batter") and _elite_player_signal(p_i) < 0.48:
                swap_idx = i
                break
        if swap_idx is not None:
            strict_names[cur_idx], strict_names[swap_idx] = strict_names[swap_idx], strict_names[cur_idx]

    def _strong_batting_history_for_bowling_ar(p: SquadPlayer) -> bool:
        hd = getattr(p, "history_debug", None) or {}
        if str(hd.get("role_band") or "") != "bowling_allrounder":
            return False
        rows = float(hd.get("batting_position_rows_found") or 0.0)
        dom = int(hd.get("dominant_position") or 99)
        return rows >= 10 and dom <= 5 and float(getattr(p, "bat_skill", 0.0) or 0.0) >= 0.65

    # Canonical batting-order hard guardrails (single source of truth).
    def _allowed_range(p: SquadPlayer) -> tuple[int, int, str]:
        f = classify_player(p)
        rb = str((getattr(p, "history_debug", None) or {}).get("role_band") or "")
        b = _band(p)

        if f.is_specialist_bowler:
            return (8, 11, "specialist_bowler")
        if b == "opener":
            return (1, 3, "opener")
        if b == "top_order" or f.is_top_order_batter:
            return (1, 5, "top_order")
        if b == "finisher" or f.is_finisher:
            return (5, 8, "finisher")
        if rb == "bowling_allrounder" and not _strong_batting_history_for_bowling_ar(p):
            return (6, 8, "bowling_allrounder")
        return (3, 7, "middle_order")

    def _violates_band(p: SquadPlayer, pos: int) -> bool:
        lo, hi, _b = _allowed_range(p)
        return pos < lo or pos > hi

    for _ in range(20):
        changed = False
        for idx, nm in enumerate(list(strict_names)):
            p = by_name.get(nm)
            if p is None:
                continue
            pos = idx + 1
            if not _violates_band(p, pos):
                continue
            # Pull into valid slot by swapping with the first candidate violating less.
            target = None
            for j, cand_nm in enumerate(strict_names):
                if j == idx:
                    continue
                cand = by_name.get(cand_nm)
                if cand is None:
                    continue
                if _violates_band(p, j + 1):
                    continue
                if _violates_band(cand, pos):
                    continue
                target = j
                break
            if target is not None:
                strict_names[idx], strict_names[target] = strict_names[target], strict_names[idx]
                changed = True
        if not changed:
            break

    current = [by_name[n] for n in strict_names if n in by_name]
    grp_bat: list[SquadPlayer] = []
    grp_bowling_ar: list[SquadPlayer] = []
    grp_special: list[SquadPlayer] = []
    for p in current:
        f = classify_player(p)
        rb = str((getattr(p, "history_debug", None) or {}).get("role_band") or "")
        if f.is_specialist_bowler:
            grp_special.append(p)
        elif rb == "bowling_allrounder" and not _strong_batting_history_for_bowling_ar(p):
            grp_bowling_ar.append(p)
        else:
            grp_bat.append(p)
    strict_names = [p.name for p in (grp_bat + grp_bowling_ar + grp_special)]

    # Enforce specialist-bowler tail (8–11) deterministically via swaps.
    def _enforce_specialist_bowler_tail(order: list[str]) -> list[str]:
        if len(order) != 11:
            return order
        by_nm = {p.name: p for p in xi}
        out = list(order)
        # Scan left-to-right for specialist bowlers placed too early; swap with the earliest non-specialist in the tail.
        for i in range(7):  # positions 1..7 (0-index 0..6)
            p = by_nm.get(out[i])
            if p is None or not classify_player(p).is_specialist_bowler:
                continue
            swap_j = None
            for j in range(7, 11):
                q = by_nm.get(out[j])
                if q is None:
                    continue
                if classify_player(q).is_specialist_bowler:
                    continue
                # Do not move the tail player into an illegal spot.
                if _violates_band(q, i + 1):
                    continue
                swap_j = j
                break
            if swap_j is not None:
                out[i], out[swap_j] = out[swap_j], out[i]
        return out

    strict_names = _enforce_specialist_bowler_tail(strict_names)

    n_role_fb = 0
    for i, name in enumerate(strict_names):
        p = by_name.get(name)
        if p is None:
            continue
        unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
        ema = float(getattr(p, "history_batting_ema", unk))
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        if ema < unk - 1e-6:
            p.history_debug["batting_order_final"] = "role_band_plus_historical_slot"
        elif _batting_position_signal_for_role_band(p) < 7.4:
            p.history_debug["batting_order_final"] = "role_band_plus_derive_slot"
        else:
            p.history_debug["batting_order_final"] = "role_band_fallback"
        p.history_debug["batting_order_rank_final"] = i + 1
        p.history_debug["batting_order_strict_xi_scope"] = True
        diag = _batting_order_diagnostic_source(p)
        p.history_debug["batting_order_diagnostic_source"] = diag
        ranked = _batting_order_signal_source_ranked(p)
        p.history_debug["batting_order_signal_source_ranked"] = ranked
        p.history_debug["batting_order_reason_summary"] = _batting_order_reason_summary_for_player(p, ranked)
        p.history_debug["final_order_reason"] = (
            f"role_band={p.history_debug.get('role_band')} with batting_slot_signal="
            f"{round(_batting_position_signal_for_role_band(p), 2)}"
        )
        p.history_debug["dominant_position"] = p.history_debug.get("dominant_position")
        p.history_debug["batting_band"] = p.history_debug.get("batting_band") or p.history_debug.get("role_band")
        p.history_debug["final_position"] = i + 1
        lo, hi, band_used = _allowed_range(p)
        p.history_debug["batting_band"] = band_used
        p.history_debug["batting_allowed_min"] = lo
        p.history_debug["batting_allowed_max"] = hi
        p.history_debug["moved_outside_band"] = _violates_band(p, i + 1)
        rb_now = str((getattr(p, "history_debug", None) or {}).get("role_band") or "")
        bowler_guardrail_applied = False
        if classify_player(p).is_specialist_bowler and (i + 1) >= 8:
            bowler_guardrail_applied = True
        if rb_now == "bowling_allrounder" and 6 <= (i + 1) <= 8 and not _strong_batting_history_for_bowling_ar(p):
            bowler_guardrail_applied = True
        p.history_debug["bowler_order_guardrail_applied"] = bowler_guardrail_applied
        if diag == "role_fallback":
            n_role_fb += 1

    # Conflict detection: if guardrails could not be satisfied, log explicitly.
    band_conflicts: list[dict[str, Any]] = []
    for i, name in enumerate(strict_names):
        p = by_name.get(name)
        if p is None:
            continue
        lo = int((getattr(p, "history_debug", None) or {}).get("batting_allowed_min") or 0)
        hi = int((getattr(p, "history_debug", None) or {}).get("batting_allowed_max") or 99)
        if lo and hi and ((i + 1) < lo or (i + 1) > hi):
            band_conflicts.append({"name": name, "pos": i + 1, "allowed": [lo, hi]})
    if band_conflicts:
        logger.error(
            "rule_conflict: batting_order_guardrails_unsatisfied team=%s conflicts=%s order=%s",
            team_name,
            band_conflicts[:8],
            strict_names,
        )

    logger.info(
        "batting_order: team=%s venue_keys=%s strict_order=%s role_fallback_xi_players=%d",
        team_name,
        (venue_keys or [])[:3],
        strict_names,
        n_role_fb,
    )
    if n_role_fb >= 6:
        msg = (
            f"{team_name}: {n_role_fb}/11 XI players lack stored batting-slot history (role_fallback). "
            "SQLite ``team_match_xi`` slot coverage is thin for this lineup. "
            f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
        )
        if out_warnings is not None:
            out_warnings.append(msg)
        logger.warning("batting_order: %s", msg)
    return strict_names


def impact_subs(
    squad: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    is_chasing: Optional[bool] = None,
    conditions: Optional[dict[str, Any]] = None,
    team_bats_first: Optional[bool] = None,
    team_display_name: str = "",
    canonical_team_key: str = "",
    venue_key: str = "",
    venue_key_candidates: Optional[list[str]] = None,
) -> tuple[list[SquadPlayer], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Bench ranking for impact subs: team patterns, game-state scenarios, tactical role cases,
    plus history/H2H/phase signals (see ``impact_subs_engine``).

    ``team_bats_first``: True if this **squad's** team bats first in the hypothetical fixture.

    Returns ``(top5_players, top5_debug_rows, all_bench_debug_rows)``.
    """
    subs, dbg_rows, dbg_all = impact_subs_engine.rank_impact_sub_candidates(
        squad,
        xi,
        team_display_name=team_display_name,
        canonical_team_key=canonical_team_key or "",
        venue_key=venue_key or "",
        venue_key_candidates=venue_key_candidates,
        is_chasing=is_chasing,
        conditions=conditions or {},
        team_bats_first=team_bats_first,
    )
    xi_names = {p.name for p in xi}
    bench_all = [p for p in squad if p.name not in xi_names]
    candidate_pool_size = len(bench_all)
    protected_bench_names = {
        p.name
        for p in squad
        if p.name not in xi_names
        and (
            _must_lock_in_base_xi(p)
            or _is_primary_strike_bowler(p)
            or (_role_band(p) in ("opener", "top_order") and _elite_player_signal(p) >= 0.46)
        )
    }
    excluded_reason_map: dict[str, str] = {}
    for n in protected_bench_names:
        excluded_reason_map[n] = "protected_bench_core_or_strike_role"

    ranked = [p for p in subs if p.name not in protected_bench_names]
    ranked_names = {p.name for p in ranked}
    for p in bench_all:
        if p.name not in ranked_names and p.name not in excluded_reason_map:
            excluded_reason_map[p.name] = "below_rank_cut_or_role_mix_filter"

    # Diversity-aware initial pick: batter + bowler + all-rounder where possible.
    def _bucket(p: SquadPlayer) -> str:
        if p.role_bucket == ALL_ROUNDER:
            return "all_rounder"
        if p.role_bucket == BOWLER:
            return "bowler"
        return "batter"

    picked: list[SquadPlayer] = []
    picked_names: set[str] = set()
    by_bucket: dict[str, list[SquadPlayer]] = {"batter": [], "bowler": [], "all_rounder": []}
    for p in ranked:
        by_bucket.setdefault(_bucket(p), []).append(p)
    for b in ("batter", "bowler", "all_rounder"):
        if by_bucket.get(b):
            p = by_bucket[b][0]
            if p.name not in picked_names:
                picked.append(p)
                picked_names.add(p.name)
    for p in ranked:
        if len(picked) >= 5:
            break
        if p.name in picked_names:
            continue
        picked.append(p)
        picked_names.add(p.name)

    fallback_used = False
    if len(picked) < 5:
        fallback_used = True
        remaining = [p for p in bench_all if p.name not in picked_names and p.name not in xi_names]
        remaining.sort(key=lambda p: (float(getattr(p, "selection_score", 0.0) or 0.0), _base_xi_rank_value(p)), reverse=True)
        for p in remaining:
            if len(picked) >= 5:
                break
            picked.append(p)
            picked_names.add(p.name)
            excluded_reason_map.pop(p.name, None)

    picked = picked[:5]
    dbg_map_all = {str(r.get("name") or ""): dict(r) for r in dbg_all}
    dbg_rows_out: list[dict[str, Any]] = []
    for p in picked:
        row = dict(dbg_map_all.get(p.name, {}))
        row["name"] = p.name
        dbg_rows_out.append(row)
    dbg_all_out = []
    for p in bench_all:
        row = dict(dbg_map_all.get(p.name, {}))
        row["name"] = p.name
        if p.name in picked_names:
            row["impact_selection_status"] = "selected_top_5"
            row["impact_excluded_reason"] = ""
        else:
            row["impact_selection_status"] = "excluded"
            row["impact_excluded_reason"] = excluded_reason_map.get(p.name, "not_selected_after_ranking")
        dbg_all_out.append(row)

    if dbg_rows_out:
        dbg_rows_out[0]["total_impact_subs_selected"] = len(picked)
        dbg_rows_out[0]["candidate_pool_size"] = candidate_pool_size
        dbg_rows_out[0]["fallback_used"] = fallback_used

    return picked, dbg_rows_out, dbg_all_out


def _impact_subs_payload(
    subs: list[SquadPlayer], dbg: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, p in enumerate(subs):
        row: dict[str, Any] = {
            "name": p.name,
            "role": p.role,
            "role_bucket": p.role_bucket,
            "composite": round(p.composite, 4),
            "is_overseas": p.is_overseas,
            "selection_score": round(p.selection_score, 5),
        }
        if i < len(dbg):
            row.update(dbg[i])
        out.append(row)
    return out


def _merge_player_match_stats_into_debug(players: list[SquadPlayer], team_key: str) -> None:
    fk = (team_key or "").strip()
    if not fk:
        return
    qkeys: list[str] = []
    for p in players:
        pk = learner.normalize_player_key(p.name)
        hd = getattr(p, "history_debug", None) or {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        hk = lk.get("history_lookup_key") or lk.get("resolved_history_key")
        qk = str(hk).strip() if hk else pk
        qkeys.append(qk)
    uniq = list(dict.fromkeys([k for k in qkeys if k]))
    counts = db.batch_player_match_stats_counts(uniq, fk)
    for p, qk in zip(players, qkeys):
        pk = learner.normalize_player_key(p.name)
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        n = int(counts.get(qk, 0))
        hu = hd.get("history_usage_debug")
        if isinstance(hu, dict):
            hu2 = dict(hu)
            hu2["player_match_stats_row_count"] = n
            hd["history_usage_debug"] = hu2
        else:
            hd["player_match_stats_row_count"] = n


def _squad_post_linkage_history_rows(players: list[SquadPlayer]) -> list[dict[str, Any]]:
    """
    Post–Stage F / post-scoring squad: merges ``history_usage_debug`` with final
    ``history_linkage`` and top-level linkage flags (same objects attached after
    global fallback and collision handling).
    """
    rows: list[dict[str, Any]] = []
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        hu = hd.get("history_usage_debug")
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        row: dict[str, Any] = {}
        if isinstance(hu, dict):
            row.update(dict(hu))
        row["squad_display_name"] = p.name
        row["player_name"] = p.name
        row["canonical_player_key"] = p.player_key or learner.normalize_player_key(p.name)
        for k in (
            "history_lookup_key",
            "collision_resolution_outcome",
            "collided_history_key",
            "collision_group_members",
            "rolled_up_interpretation",
            "history_status",
            "resolution_type",
        ):
            if k in lk:
                row[k] = lk.get(k)
        row["global_resolved_history_key"] = lk.get("global_resolved_history_key") or hd.get(
            "global_resolved_history_key"
        )
        row["used_global_resolved_key_for_prior"] = lk.get("used_global_resolved_key_for_prior")
        if row.get("used_global_resolved_key_for_prior") is None:
            row["used_global_resolved_key_for_prior"] = hd.get("used_global_resolved_key_for_prior")
        row["collision_resolution_winner_name"] = lk.get("collision_winner_player_name")
        row["likely_first_ipl_player"] = lk.get("likely_first_ipl_player")
        if row.get("likely_first_ipl_player") is None:
            row["likely_first_ipl_player"] = hd.get("likely_first_ipl_player")
        row["debutant_alias_suppression_applied"] = lk.get("debutant_alias_suppression_applied")
        if row.get("debutant_alias_suppression_applied") is None:
            row["debutant_alias_suppression_applied"] = hd.get("debutant_alias_suppression_applied")
        row["debutant_alias_rejection_reason"] = lk.get("debutant_alias_rejection_reason")
        if row.get("debutant_alias_rejection_reason") is None:
            row["debutant_alias_rejection_reason"] = hd.get("debutant_alias_rejection_reason")
        rows.append(row)
    return rows


def _history_linkage_squad_rollup(players: list[SquadPlayer]) -> dict[str, Any]:
    from collections import Counter

    res_t = Counter()
    coll_o = Counter()
    for p in players:
        lk = (getattr(p, "history_debug", None) or {}).get("history_linkage")
        if not isinstance(lk, dict):
            continue
        rt = str(lk.get("resolution_type") or "unknown")
        res_t[rt] += 1
        co = str(lk.get("collision_resolution_outcome") or "unknown")
        coll_o[co] += 1
    summ = None
    if players:
        hd0 = getattr(players[0], "history_debug", None) or {}
        summ = hd0.get("history_linkage_team_summary")
    return {
        "player_count": len(players),
        "resolution_type_counts": dict(res_t),
        "collision_resolution_outcome_counts": dict(coll_o),
        "history_linkage_team_summary": summ,
    }


def _squad_scoring_breakdown_rows(players: list[SquadPlayer]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        sb = hd.get("scoring_breakdown") if isinstance(hd.get("scoring_breakdown"), dict) else {}
        ssc = hd.get("selection_score_components") if isinstance(hd.get("selection_score_components"), dict) else {}
        row: dict[str, Any] = {
            "squad_display_name": p.name,
            "player_name": p.name,
            "canonical_player_key": p.player_key or learner.normalize_player_key(p.name),
            "bowling_type": p.bowling_type,
        }
        row.update(sb)
        row["final_selection_score"] = row.get("final_selection_score", ssc.get("selection_score"))
        row["history_weight_applied"] = ssc.get("history_weight_applied")
        row["has_usable_history"] = ssc.get("has_usable_sqlite_or_cricsheet_history")
        row["probable_first_choice_prior"] = hd.get("probable_first_choice_prior")
        row["global_ipl_history_presence"] = hd.get("global_ipl_history_presence")
        row["global_selection_frequency"] = hd.get("global_selection_frequency")
        row["used_global_fallback_prior"] = hd.get("used_global_fallback_prior")
        row["captain_boost_applied"] = (sb.get("captain_boost_applied") if isinstance(sb, dict) else None) or hd.get(
            "captain_boost_applied"
        )
        row["wicketkeeper_boost_applied"] = (
            (sb.get("wicketkeeper_boost_applied") if isinstance(sb, dict) else None)
            or hd.get("wicketkeeper_boost_applied")
        )
        row["valid_current_squad_new_to_franchise"] = hd.get("valid_current_squad_new_to_franchise")
        row["wrong_side_squad_assignment"] = hd.get("wrong_side_squad_assignment")
        row["selection_reason_summary"] = hd.get("selection_reason_summary")
        row["history_source_used"] = hd.get("history_source_used")
        row["fallback_used"] = hd.get("fallback_used")
        row["derive_player_profile"] = hd.get("derive_player_profile")
        row["marquee_tier"] = hd.get("marquee_tier")
        row["marquee_source"] = hd.get("marquee_source")
        row["marquee_reason"] = hd.get("marquee_reason")
        row["marquee_suggested_score"] = hd.get("marquee_suggested_score")
        row["marquee_suggested_score_raw"] = hd.get("marquee_suggested_score_raw")
        row["marquee_suggested_rank_pct"] = hd.get("marquee_suggested_rank_pct")
        sm_dbg = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else None
        if sm_dbg is not None:
            row["tactical_adjustment_total"] = sm_dbg.get("tactical_adjustment_total")
            rf_det = sm_dbg.get("recent_form_detail") if isinstance(sm_dbg.get("recent_form_detail"), dict) else {}
            comps = rf_det.get("competitions_used")
            if isinstance(comps, list) and comps:
                row["recent_form_competitions_used"] = ", ".join(str(c) for c in comps[:20])
            elif isinstance(comps, list):
                row["recent_form_competitions_used"] = ""
            ex = sm_dbg.get("explainability") if isinstance(sm_dbg.get("explainability"), dict) else {}
            line = " | ".join(
                str(x).strip()
                for x in (
                    ex.get("role_reason"),
                    ex.get("recent_form_reason"),
                    ex.get("ipl_role_history_reason"),
                    ex.get("team_balance_reason"),
                    ex.get("venue_reason"),
                )
                if x
            )
            row["selection_model_explain_line"] = (line[:420] + "…") if len(line) > 420 else line
        out.append(row)
    return out


def _omitted_xi_report(
    scored: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    scenario_branch: Optional[str] = None,
    xi_baseline: Optional[list[SquadPlayer]] = None,
    top_n: int = 14,
) -> list[dict[str, Any]]:
    """
    Omitted players ranked with the same effective key as ``select_playing_xi``:
    scenario branch score minus precomputed XI-build penalties.
    """
    xi_names = {p.name for p in xi}
    pen = _precompute_xi_build_penalties(scored)

    def eff_rank(p: SquadPlayer) -> float:
        return _scenario_xi_rank_value(p, scenario_branch) - pen.get(p.name, 0.0)

    order = sorted(
        scored,
        key=lambda x: (eff_rank(x), _xi_selection_tier(x), x.composite),
        reverse=True,
    )
    rank_by_name = {p.name: i + 1 for i, p in enumerate(order)}
    cut_eff = eff_rank(order[10]) if len(order) >= 11 else 0.0
    first11_scen = {p.name for p in order[:11]}
    base_xi_names = {p.name for p in xi_baseline} if xi_baseline else set()
    pool = [p for p in scored if p.name not in xi_names]
    pool.sort(key=lambda p: (eff_rank(p), p.composite), reverse=True)
    out: list[dict[str, Any]] = []
    for p in pool[:top_n]:
        hd = getattr(p, "history_debug", None) or {}
        smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        bbd = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        rt = lk.get("resolution_type") or ""
        rk = rank_by_name.get(p.name)
        scen_gap = float(cut_eff - eff_rank(p)) if len(order) >= 11 else 0.0
        omitted_due_to_constraint = (p.name in first11_scen) and (p.name not in xi_names)
        omitted_due_to_scenario_rank = (
            scenario_branch in ("if_team_bats_first", "if_team_bowls_first")
            and bool(base_xi_names)
            and p.name in base_xi_names
            and p.name not in xi_names
        )
        reasons: list[str] = []
        if rk is not None and rk > 11:
            reasons.append(f"squad_selection_rank_{rk}")
        if len(order) >= 11 and scen_gap > 1e-7:
            reasons.append(f"scenario_eff_rank_gap_vs_rank11_{scen_gap:.5f}")
        if omitted_due_to_constraint:
            reasons.append("omitted_due_to_xi_constraint_repair")
        if omitted_due_to_scenario_rank:
            reasons.append("omitted_due_to_scenario_branch_vs_baseline_xi")
        if int(hd.get("history_rows_found") or 0) < 2:
            reasons.append("low_team_match_xi_rows")
        if int(hd.get("franchise_history_distinct_matches") or 0) <= 2:
            reasons.append("low_franchise_history_distinct_matches")
        if rt in ("no_match", "ambiguous_alias"):
            reasons.append("stage_f_alias_miss_or_ambiguous")
        if p.is_overseas:
            reasons.append("overseas_player_constraints_may_apply")
        if p.role_bucket == BOWLER and sum(1 for x in xi if x.role_bucket == BOWLER) >= 5:
            reasons.append("bowling_slot_competition")
        c_anchor = _core_anchor_strength(p)
        if c_anchor >= 0.11 and p.name not in xi_names:
            reasons.append("omitted_despite_core_anchor_due_to_balance_or_condition_limits")
        p_prior = float(hd.get("probable_first_choice_prior") or 0.0)
        if p_prior >= 0.48 and bool(hd.get("used_global_fallback_prior")):
            reasons.append("notable_strong_probable_first_choice_prior_still_omitted")
        if bool(hd.get("captain_selected_for_team")):
            reasons.append("captain_omitted_despite_manual_selection")
        if bool(hd.get("wicketkeeper_selected_for_team")):
            reasons.append("wicketkeeper_omitted_despite_manual_selection")
        omitted_reason_summary = "; ".join(reasons) if reasons else "not_in_top_11_by_xi_model"
        out.append(
            {
                "name": p.name,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "in_current_squad": True,
                "role_bucket": p.role_bucket,
                "selection_score": round(p.selection_score, 5),
                "scenario_rank_used": round(_scenario_xi_rank_value(p, scenario_branch), 5),
                "effective_xi_rank_score": round(eff_rank(p), 5),
                "composite": round(p.composite, 5),
                "squad_selection_rank": rk,
                "scenario_score_gap_vs_cutoff": round(scen_gap, 5),
                "score_gap_vs_cutoff": round(scen_gap, 5),
                "omitted_due_to_constraint": omitted_due_to_constraint,
                "omitted_due_to_scenario_rank": omitted_due_to_scenario_rank,
                "omitted_reason_summary": omitted_reason_summary,
                "omission_reason": omitted_reason_summary,
                "probable_first_choice_prior": round(p_prior, 5),
                "base_xi_score": round(float(hd.get("base_xi_score") or _base_xi_rank_value(p)), 5),
                "base_xi_reason": hd.get("base_xi_reason"),
                "core_anchor_strength": round(c_anchor, 5),
                "condition_adjustment_reason": hd.get("condition_adjustment_reason"),
                "was_in_last_match_xi": (smd.get("last_match_detail") or {}).get("was_in_last_match_xi"),
                "last_match_continuity_score": bbd.get("last_match_continuity_score"),
                "recent_form_score": bbd.get("recent_form_score"),
                "ipl_history_and_role_score": bbd.get("ipl_history_and_role_score"),
                "stable_role_identity_score": bbd.get("stable_role_identity_score"),
                "team_balance_fit_score": bbd.get("team_balance_fit_score"),
                "core_player_signal": round(c_anchor, 5),
                "condition_adjustment": smd.get("tactical_adjustment_total"),
                "final_xi_decision_reason": omitted_reason_summary,
                "global_ipl_history_presence": hd.get("global_ipl_history_presence"),
                "global_selection_frequency": hd.get("global_selection_frequency"),
                "used_global_fallback_prior": bool(hd.get("used_global_fallback_prior")),
                "marquee_tier": hd.get("marquee_tier"),
                "marquee_source": hd.get("marquee_source"),
                "marquee_reason": hd.get("marquee_reason"),
                "marquee_suggested_score": hd.get("marquee_suggested_score"),
                "marquee_suggested_score_raw": hd.get("marquee_suggested_score_raw"),
                "marquee_suggested_rank_pct": hd.get("marquee_suggested_rank_pct"),
                "reasons_not_in_playing_xi": reasons,
            }
        )
    return out


def _xi_strength(xi: list[SquadPlayer]) -> float:
    if not xi:
        return 0.0
    w = float(getattr(config, "STAGE3_XI_STRENGTH_SELECTION_BLEND", 0.34))
    w = max(0.0, min(0.78, w))
    acc = 0.0
    for p in xi:
        comp = float(p.composite)
        sel = float(getattr(p, "selection_score", comp))
        acc += (1.0 - w) * comp + w * sel
    return acc / max(1, len(xi)) * 11.0


def toss_matchup_effects(
    team_a: str,
    team_b: str,
    *,
    strength_a: float,
    strength_b: float,
    conditions: dict[str, Any],
    venue_key: str,
) -> dict[str, Any]:
    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])
    rain = float(conditions["rain_disruption_risk"])

    edge_a = learner.venue_toss_edge(venue_key, team_a)
    edge_b = learner.venue_toss_edge(venue_key, team_b)

    # Relative chase/defend strength (unitless)
    bat_first_boost = 0.18 * (bf - 0.5) - 0.12 * dew - 0.22 * rain
    chase_boost = -bat_first_boost + 0.14 * dew

    rel = (strength_a - strength_b) / 11.0
    bat_first_a = rel + bat_first_boost + edge_a["bat_first_logit"] - edge_b["bowl_first_logit"]
    bowl_first_a = -rel + chase_boost + edge_a["bowl_first_logit"] - edge_b["bat_first_logit"]

    return {
        "bat_first_edge_team_a": float(bat_first_a),
        "bowl_first_edge_team_a": float(bowl_first_a),
        "venue_bat_first_prior": bat_first_boost,
        "dew_chase_factor": chase_boost,
        "team_a_bat_first_samples": edge_a["sample_bat"],
        "team_a_bowl_first_samples": edge_a["sample_bowl"],
    }


def win_probability(
    strength_a: float,
    strength_b: float,
    conditions: dict[str, Any],
    toss_effects: dict[str, Any],
    *,
    assume_team_a_bats_first: Optional[bool] = None,
    chase_boost_logit: float = 0.0,
) -> dict[str, Any]:
    """
    If assume_team_a_bats_first is None, marginalize toss with 0.5/0.5.
    chase_boost_logit: venue chase prior applied when A chases; half applied when toss unknown.
    """
    rain = float(conditions["rain_disruption_risk"])
    base = (strength_a - strength_b) / config.WIN_MODEL_TEMPERATURE
    base -= 0.35 * rain * math.copysign(1.0, strength_a - strength_b)

    if assume_team_a_bats_first is None:
        e_bf = float(toss_effects["bat_first_edge_team_a"])
        e_ch = float(toss_effects["bowl_first_edge_team_a"])
        logit = base + 0.5 * e_bf + 0.5 * e_ch + 0.5 * chase_boost_logit
    elif assume_team_a_bats_first:
        logit = base + float(toss_effects["bat_first_edge_team_a"])
    else:
        logit = base + float(toss_effects["bowl_first_edge_team_a"]) + chase_boost_logit

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    p = sigmoid(logit)
    return {
        "team_a_win": float(p),
        "team_b_win": float(1.0 - p),
        "logit": float(logit),
    }


def run_prediction(
    team_a_name: str,
    team_b_name: str,
    squad_a_text: str,
    squad_b_text: str,
    unavailable_text: str,
    venue: VenueProfile,
    match_time: datetime,
    weather: dict[str, Any],
    *,
    toss_scenario_key: str = "unknown",
    team_a_captain_display_name: str = "",
    team_b_captain_display_name: str = "",
    team_a_wicketkeeper_display_name: str = "",
    team_b_wicketkeeper_display_name: str = "",
    team_a_fetched_squad_player_keys: Optional[set[str]] = None,
    team_b_fetched_squad_player_keys: Optional[set[str]] = None,
    team_a_stale_cached_player_keys: Optional[set[str]] = None,
    team_b_stale_cached_player_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    _audit_par = audit_profile.PredictionRunAudit() if audit_profile.audit_enabled() else None
    try:
        return _run_prediction_inner(
            team_a_name,
            team_b_name,
            squad_a_text,
            squad_b_text,
            unavailable_text,
            venue,
            match_time,
            weather,
            toss_scenario_key=toss_scenario_key,
            team_a_captain_display_name=team_a_captain_display_name,
            team_b_captain_display_name=team_b_captain_display_name,
            team_a_wicketkeeper_display_name=team_a_wicketkeeper_display_name,
            team_b_wicketkeeper_display_name=team_b_wicketkeeper_display_name,
            team_a_fetched_squad_player_keys=team_a_fetched_squad_player_keys,
            team_b_fetched_squad_player_keys=team_b_fetched_squad_player_keys,
            team_a_stale_cached_player_keys=team_a_stale_cached_player_keys,
            team_b_stale_cached_player_keys=team_b_stale_cached_player_keys,
            _audit_par=_audit_par,
        )
    except BaseException:
        if _audit_par:
            _audit_par.close_failure()
        raise


def _run_prediction_inner(
    team_a_name: str,
    team_b_name: str,
    squad_a_text: str,
    squad_b_text: str,
    unavailable_text: str,
    venue: VenueProfile,
    match_time: datetime,
    weather: dict[str, Any],
    *,
    toss_scenario_key: str = "unknown",
    team_a_captain_display_name: str = "",
    team_b_captain_display_name: str = "",
    team_a_wicketkeeper_display_name: str = "",
    team_b_wicketkeeper_display_name: str = "",
    team_a_fetched_squad_player_keys: Optional[set[str]] = None,
    team_b_fetched_squad_player_keys: Optional[set[str]] = None,
    team_a_stale_cached_player_keys: Optional[set[str]] = None,
    team_b_stale_cached_player_keys: Optional[set[str]] = None,
    _audit_par: Any = None,
) -> dict[str, Any]:
    _run_t0 = time.perf_counter()
    prediction_timing_ms: dict[str, float] = {}

    def _ap_phase(name: str, t0: float) -> None:
        if audit_profile.audit_enabled():
            audit_profile.record_prediction_phase(name, (time.perf_counter() - t0) * 1000.0)

    def _fetched_key_set(keys: Optional[set[str]]) -> FrozenSet[str]:
        if not keys:
            return frozenset()
        out: set[str] = set()
        for k in keys:
            s = str(k).strip()
            if s:
                out.add(s)
        return frozenset(out)

    _t_early = time.perf_counter()
    conditions = venue_conditions_summary(venue, weather)

    sa = filter_unavailable(parse_squad_text(squad_a_text), unavailable_text)
    sb = filter_unavailable(parse_squad_text(squad_b_text), unavailable_text)
    role_dist_a = _log_role_bucket_distribution(sa, "team_a")
    role_dist_b = _log_role_bucket_distribution(sb, "team_b")
    canon_a = ipl_teams.franchise_label_for_storage(team_a_name) or team_a_name.strip()
    canon_b = ipl_teams.franchise_label_for_storage(team_b_name) or team_b_name.strip()
    slug_a = (ipl_teams.slug_for_canonical_label(canon_a) or "").strip()
    slug_b = (ipl_teams.slug_for_canonical_label(canon_b) or "").strip()
    fetch_sel_a = _fetched_key_set(team_a_fetched_squad_player_keys)
    fetch_sel_b = _fetched_key_set(team_b_fetched_squad_player_keys)
    fetch_opp_a = fetch_sel_b
    fetch_opp_b = fetch_sel_a
    stale_a = _fetched_key_set(team_a_stale_cached_player_keys)
    stale_b = _fetched_key_set(team_b_stale_cached_player_keys)
    _annotate_squad_canonical_keys(sa, canon_a)
    _annotate_squad_canonical_keys(sb, canon_b)
    # Bulk Cricsheet JSON → SQLite runs in the **ingest** stage only (see ``cricsheet_ingest``).
    # ``run_prediction`` reads existing SQLite rows via ``history_xi`` / ``db`` — no on-the-fly JSON ingest.
    history_sync_debug: dict[str, Any] = {
        "local_history_warning": None,
        "team_a": {},
        "team_b": {},
        "h2h_layer": {},
    }

    try:
        history_sync_debug["team_a"] = history_sync.local_history_debug_for_prediction(
            canon_a,
            squad_player_names=[p.name for p in sa],
            include_squad_report=False,
        )
        history_sync_debug["team_b"] = history_sync.local_history_debug_for_prediction(
            canon_b,
            squad_player_names=[p.name for p in sb],
            include_squad_report=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Local history snapshot failed; continuing with heuristics: %s", exc)
        history_sync_debug["local_history_warning"] = (
            "Could not read local SQLite history metadata; prediction continues using squad heuristics. "
            + history_sync.HISTORY_MISSING_USER_MESSAGE
        )
        history_sync_debug["sync_exception"] = f"{type(exc).__name__}: {exc}"
        history_sync_debug["team_a"] = history_sync.failsafe_history_debug(canon_a, exc)
        history_sync_debug["team_b"] = history_sync.failsafe_history_debug(canon_b, exc)

    thin_msgs: list[str] = []
    for side in ("team_a", "team_b"):
        block = history_sync_debug.get(side) or {}
        for w in block.get("warnings") or []:
            if isinstance(w, str) and w.strip():
                thin_msgs.append(f"{side}: {w.strip()}")
    if thin_msgs and not history_sync_debug.get("local_history_warning"):
        a_ne = history_sync.raw_stage1_tables_near_empty(history_sync_debug.get("team_a") or {})
        b_ne = history_sync.raw_stage1_tables_near_empty(history_sync_debug.get("team_b") or {})
        if a_ne or b_ne:
            history_sync_debug["local_history_warning"] = history_sync.HISTORY_MISSING_USER_MESSAGE

    _ap_phase("parse_squads_franchise_slug_and_history_sync_ms", _t_early)

    _t_learn = time.perf_counter()
    learned_map = learner.load_learned_map()
    logger.info(
        "run_prediction: structured squads team_a=%d team_b=%d canonical=(%s,%s) preview_a=%s",
        len(sa),
        len(sb),
        canon_a,
        canon_b,
        [(p.name, p.role_bucket) for p in sa[:6]],
    )
    shape_a = _squad_shape(sa)
    shape_b = _squad_shape(sb)

    scored_a: list[SquadPlayer] = []
    for p in sa:
        sp = _score_player(
            p,
            self_shape=shape_a,
            opp_shape=shape_b,
            conditions=conditions,
            learned_map=learned_map,
            franchise_canonical=canon_a,
        )
        _set_player_ipl_flags(sp)
        scored_a.append(sp)
    scored_b = []
    for p in sb:
        sp = _score_player(
            p,
            self_shape=shape_b,
            opp_shape=shape_a,
            conditions=conditions,
            learned_map=learned_map,
            franchise_canonical=canon_b,
        )
        _set_player_ipl_flags(sp)
        scored_b.append(sp)

    _ap_phase("learned_map_load_shape_and_score_players_both_teams_ms", _t_learn)

    _t_hctx = time.perf_counter()
    hctx = build_history_context()
    vkeys = venue_lookup_keys(venue)
    pattern_vkeys = _derive_pattern_venue_keys(venue, vkeys)
    is_night = time_utils.ist_hour(match_time) >= config.NIGHT_START_HOUR_LOCAL
    dew_risk = float(conditions["dew_risk"])
    a_bats_first = resolve_a_bats_first_toss(toss_scenario_key)
    chase_ctx_a = chase_context_for_team(team_a_name, team_a_name, team_b_name, a_bats_first)
    chase_ctx_b = chase_context_for_team(team_b_name, team_a_name, team_b_name, a_bats_first)

    for p in scored_a:
        _history_adjust_for_player(p, team_a_name, shape_a, vkeys, is_night, dew_risk, hctx)
    for p in scored_b:
        _history_adjust_for_player(p, team_b_name, shape_b, vkeys, is_night, dew_risk, hctx)

    _ap_phase("build_history_context_and_history_rules_bump_ms", _t_hctx)

    h2h_layer: dict[str, Any] = {}
    _t_hist = time.perf_counter()
    history_xi.attach_primary_history_to_squad(
        scored_a,
        team_a_name,
        vkeys,
        shape=shape_a,
        chase_context=chase_ctx_a,
        opponent_canonical_label=canon_b,
        h2h_explain=h2h_layer,
        h2h_explain_scope="team_a",
        fetched_team_slug=slug_a,
        selected_fetched_player_keys=fetch_sel_a if fetch_sel_a else None,
        opposite_fetched_player_keys=fetch_opp_a,
        stale_cached_player_keys=stale_a,
        captain_display_name=team_a_captain_display_name or "",
        wicketkeeper_display_name=team_a_wicketkeeper_display_name or "",
    )
    fixture_ctx_a = {"is_night": is_night, "reference_iso_date": match_time.date().isoformat()}
    fixture_ctx_b = {"is_night": is_night, "reference_iso_date": match_time.date().isoformat()}
    if a_bats_first is not None:
        fixture_ctx_a["xi_scenario_branch_for_tactical"] = (
            "if_team_bats_first" if a_bats_first else "if_team_bowls_first"
        )
        fixture_ctx_b["xi_scenario_branch_for_tactical"] = (
            "if_team_bats_first" if not a_bats_first else "if_team_bowls_first"
        )
    history_xi.compute_selection_scores(
        scored_a,
        conditions=conditions,
        venue_key_candidates=pattern_vkeys,
        fixture_context=fixture_ctx_a,
    )
    _refine_opener_finisher_from_derive(scored_a)
    tk_a = ipl_teams.canonical_team_key_for_franchise(canon_a)
    _merge_player_match_stats_into_debug(scored_a, tk_a)
    _annotate_player_metadata(scored_a)
    _annotate_batting_position_profiles(scored_a, tk_a)
    _annotate_phase_bowling_signals(scored_a, tk_a)
    _annotate_role_bands(scored_a)
    for sp in scored_a:
        _set_player_ipl_flags(sp)
    _annotate_marquee_tags(scored_a)
    history_xi.attach_primary_history_to_squad(
        scored_b,
        team_b_name,
        vkeys,
        shape=shape_b,
        chase_context=chase_ctx_b,
        opponent_canonical_label=canon_a,
        h2h_explain=h2h_layer,
        h2h_explain_scope="team_b",
        fetched_team_slug=slug_b,
        selected_fetched_player_keys=fetch_sel_b if fetch_sel_b else None,
        opposite_fetched_player_keys=fetch_opp_b,
        stale_cached_player_keys=stale_b,
        captain_display_name=team_b_captain_display_name or "",
        wicketkeeper_display_name=team_b_wicketkeeper_display_name or "",
    )
    history_xi.compute_selection_scores(
        scored_b,
        conditions=conditions,
        venue_key_candidates=pattern_vkeys,
        fixture_context=fixture_ctx_b,
    )
    _refine_opener_finisher_from_derive(scored_b)
    tk_b = ipl_teams.canonical_team_key_for_franchise(canon_b)
    _merge_player_match_stats_into_debug(scored_b, tk_b)
    _annotate_player_metadata(scored_b)
    _annotate_batting_position_profiles(scored_b, tk_b)
    _annotate_phase_bowling_signals(scored_b, tk_b)
    _annotate_role_bands(scored_b)
    for sp in scored_b:
        _set_player_ipl_flags(sp)
    _annotate_marquee_tags(scored_b)
    _ap_phase("history_xi_attach_selection_scores_merge_pms_both_teams_ms", _t_hist)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["history_xi_and_selection_both_teams_ms"] = round(
            (time.perf_counter() - _t_hist) * 1000.0, 2
        )
    h2h_layer.pop("_h2h_layer_meta_done", None)
    history_sync_debug["h2h_layer"] = h2h_layer

    _t_xi_pick = time.perf_counter()
    br_a = franchise_xi_scenario_branch(True, a_bats_first)
    br_b = franchise_xi_scenario_branch(False, a_bats_first)
    squad_os_a = sum(1 for p in scored_a if p.is_overseas)
    squad_os_b = sum(1 for p in scored_b if p.is_overseas)
    os_min_a = min(3, squad_os_a, int(getattr(config, "MAX_OVERSEAS", 4)))
    os_min_b = min(3, squad_os_b, int(getattr(config, "MAX_OVERSEAS", 4)))
    os_target_a = min(4, squad_os_a, int(getattr(config, "MAX_OVERSEAS", 4)))
    os_target_b = min(4, squad_os_b, int(getattr(config, "MAX_OVERSEAS", 4)))
    xi_a_base = select_base_playing_xi(scored_a, conditions=conditions)
    xi_a, xi_a_condition_changes = _apply_condition_adjustments_from_base(
        scored_a,
        xi_a_base,
        scenario_branch=br_a,
        conditions=conditions,
    )
    xi_a, overseas_dbg_a_pre = _optimize_overseas_preference(
        scored_a,
        xi_a,
        conditions=conditions,
        overseas_min_required=os_min_a,
        overseas_target=os_target_a,
    )
    overseas_dbg_a = dict(overseas_dbg_a_pre)
    xi_a, xi_a_repaired, xi_a_repair_swaps, xi_a_repair_enforce = _repair_xi_if_needed(
        scored_a,
        xi_a,
        conditions=conditions,
    )
    xi_a_condition_changes = _reconcile_condition_changes_and_annotate(
        scored_a,
        xi_a_base,
        xi_a,
        scenario_branch=br_a,
        changes=xi_a_condition_changes,
    )
    _delta_scen_a = {p.name for p in xi_a}.symmetric_difference({p.name for p in xi_a_base})
    for p in scored_a:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd_a = p.history_debug
        hd_a["xi_scenario_branch_for_xi_build"] = br_a
        hd_a["scenario_rank_used"] = round(_scenario_xi_rank_value(p, br_a), 5)
        hd_a["selection_changed_due_to_scenario"] = p.name in _delta_scen_a
        hd_a["base_xi_rank_used"] = round(_base_xi_rank_value(p), 5)

    xi_b_base = select_base_playing_xi(scored_b, conditions=conditions)
    xi_b, xi_b_condition_changes = _apply_condition_adjustments_from_base(
        scored_b,
        xi_b_base,
        scenario_branch=br_b,
        conditions=conditions,
    )
    xi_b, overseas_dbg_b_pre = _optimize_overseas_preference(
        scored_b,
        xi_b,
        conditions=conditions,
        overseas_min_required=os_min_b,
        overseas_target=os_target_b,
    )
    overseas_dbg_b = dict(overseas_dbg_b_pre)
    xi_b, xi_b_repaired, xi_b_repair_swaps, xi_b_repair_enforce = _repair_xi_if_needed(
        scored_b,
        xi_b,
        conditions=conditions,
    )
    xi_b_condition_changes = _reconcile_condition_changes_and_annotate(
        scored_b,
        xi_b_base,
        xi_b,
        scenario_branch=br_b,
        changes=xi_b_condition_changes,
    )
    _delta_scen_b = {p.name for p in xi_b}.symmetric_difference({p.name for p in xi_b_base})
    for p in scored_b:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd_b = p.history_debug
        hd_b["xi_scenario_branch_for_xi_build"] = br_b
        hd_b["scenario_rank_used"] = round(_scenario_xi_rank_value(p, br_b), 5)
        hd_b["selection_changed_due_to_scenario"] = p.name in _delta_scen_b
        hd_b["base_xi_rank_used"] = round(_base_xi_rank_value(p), 5)

    xi_a_if_bats_first: Optional[list[SquadPlayer]] = None
    xi_a_if_bowls_first: Optional[list[SquadPlayer]] = None
    xi_b_if_bats_first: Optional[list[SquadPlayer]] = None
    xi_b_if_bowls_first: Optional[list[SquadPlayer]] = None
    if a_bats_first is None:
        xi_a_if_bats_first, _ = _apply_condition_adjustments_from_base(
            scored_a,
            xi_a_base,
            scenario_branch="if_team_bats_first",
            conditions=conditions,
        )
        xi_a_if_bats_first, _dbg = _optimize_overseas_preference(
            scored_a,
            xi_a_if_bats_first,
            conditions=conditions,
            overseas_min_required=os_min_a,
            overseas_target=os_target_a,
        )
        xi_a_if_bats_first, _repaired, _swaps, _enf = _repair_xi_if_needed(
            scored_a,
            xi_a_if_bats_first,
            conditions=conditions,
        )
        xi_a_if_bowls_first, _ = _apply_condition_adjustments_from_base(
            scored_a,
            xi_a_base,
            scenario_branch="if_team_bowls_first",
            conditions=conditions,
        )
        xi_a_if_bowls_first, _dbg = _optimize_overseas_preference(
            scored_a,
            xi_a_if_bowls_first,
            conditions=conditions,
            overseas_min_required=os_min_a,
            overseas_target=os_target_a,
        )
        xi_a_if_bowls_first, _repaired, _swaps, _enf = _repair_xi_if_needed(
            scored_a,
            xi_a_if_bowls_first,
            conditions=conditions,
        )
        xi_b_if_bats_first, _ = _apply_condition_adjustments_from_base(
            scored_b,
            xi_b_base,
            scenario_branch="if_team_bats_first",
            conditions=conditions,
        )
        xi_b_if_bats_first, _dbg = _optimize_overseas_preference(
            scored_b,
            xi_b_if_bats_first,
            conditions=conditions,
            overseas_min_required=os_min_b,
            overseas_target=os_target_b,
        )
        xi_b_if_bats_first, _repaired, _swaps, _enf = _repair_xi_if_needed(
            scored_b,
            xi_b_if_bats_first,
            conditions=conditions,
        )
        xi_b_if_bowls_first, _ = _apply_condition_adjustments_from_base(
            scored_b,
            xi_b_base,
            scenario_branch="if_team_bowls_first",
            conditions=conditions,
        )
        xi_b_if_bowls_first, _dbg = _optimize_overseas_preference(
            scored_b,
            xi_b_if_bowls_first,
            conditions=conditions,
            overseas_min_required=os_min_b,
            overseas_target=os_target_b,
        )
        xi_b_if_bowls_first, _repaired, _swaps, _enf = _repair_xi_if_needed(
            scored_b,
            xi_b_if_bowls_first,
            conditions=conditions,
        )

    sub_ok_a, sub_err_a = _validate_xi_in_current_squad(xi_a, sa, "team_a")
    sub_ok_b, sub_err_b = _validate_xi_in_current_squad(xi_b, sb, "team_b")
    if not sub_ok_a:
        raise ValueError(
            f"Team A predicted XI includes player(s) not in the current squad (inner-join violation): {sub_err_a}"
        )
    if not sub_ok_b:
        raise ValueError(
            f"Team B predicted XI includes player(s) not in the current squad (inner-join violation): {sub_err_b}"
        )

    v_ok_a, v_err_a = _validate_xi(xi_a, conditions=conditions, squad=scored_a)
    v_ok_b, v_err_b = _validate_xi(xi_b, conditions=conditions, squad=scored_b)
    if not bool(xi_a_repair_enforce.get("hard_constraints_satisfied")):
        logger.error(
            "team_a hard constraints unsatisfied after repair failed=%s reason=%s",
            xi_a_repair_enforce.get("failed_constraints"),
            xi_a_repair_enforce.get("repair_failure_reason"),
        )
        raise ValueError(
            "Team A hard constraints unsatisfied after repair: "
            f"{xi_a_repair_enforce.get('failed_constraints')} | "
            f"reason={xi_a_repair_enforce.get('repair_failure_reason')}"
        )
    if not bool(xi_b_repair_enforce.get("hard_constraints_satisfied")):
        logger.error(
            "team_b hard constraints unsatisfied after repair failed=%s reason=%s",
            xi_b_repair_enforce.get("failed_constraints"),
            xi_b_repair_enforce.get("repair_failure_reason"),
        )
        raise ValueError(
            "Team B hard constraints unsatisfied after repair: "
            f"{xi_b_repair_enforce.get('failed_constraints')} | "
            f"reason={xi_b_repair_enforce.get('repair_failure_reason')}"
        )
    xi_counts_a = _xi_role_validation_counts(xi_a)
    xi_counts_b = _xi_role_validation_counts(xi_b)
    xi_rules_a = rules_xi.validate_xi(xi_a, conditions=conditions, squad=scored_a)
    xi_rules_b = rules_xi.validate_xi(xi_b, conditions=conditions, squad=scored_b)
    logger.info(
        "xi_rule_summary: team=%s summary=%s hard_ok=%s warnings=%s violations=%s",
        team_a_name,
        xi_rules_a.summary,
        xi_rules_a.hard_ok,
        [w.code for w in xi_rules_a.warnings],
        [v.code for v in xi_rules_a.violations],
    )
    logger.info(
        "xi_rule_summary: team=%s summary=%s hard_ok=%s warnings=%s violations=%s",
        team_b_name,
        xi_rules_b.summary,
        xi_rules_b.hard_ok,
        [w.code for w in xi_rules_b.warnings],
        [v.code for v in xi_rules_b.violations],
    )
    if not v_ok_a:
        logger.warning("team_a XI constraints failed: %s", v_err_a)
    if not v_ok_b:
        logger.warning("team_b XI constraints failed: %s", v_err_b)
    batting_order_warnings: list[str] = []
    order_a = (
        build_batting_order(
            xi_a,
            conditions,
            team_name=team_a_name,
            venue_keys=vkeys,
            out_warnings=batting_order_warnings,
        )
        if len(xi_a) == 11
        else _batting_order_for_short_xi(xi_a, team_name=team_a_name, out_warnings=batting_order_warnings)
    )
    order_b = (
        build_batting_order(
            xi_b,
            conditions,
            team_name=team_b_name,
            venue_keys=vkeys,
            out_warnings=batting_order_warnings,
        )
        if len(xi_b) == 11
        else _batting_order_for_short_xi(xi_b, team_name=team_b_name, out_warnings=batting_order_warnings)
    )
    # Freeze XI membership before batting order: order may be assigned, but membership may not mutate.
    if len(xi_a) == 11 and set(order_a) != {p.name for p in xi_a}:
        logger.error(
            "rule_conflict: batting_order_not_permutation_of_xi team=%s xi=%s order=%s",
            team_a_name,
            [p.name for p in xi_a],
            order_a,
        )
    if len(xi_b) == 11 and set(order_b) != {p.name for p in xi_b}:
        logger.error(
            "rule_conflict: batting_order_not_permutation_of_xi team=%s xi=%s order=%s",
            team_b_name,
            [p.name for p in xi_b],
            order_b,
        )

    strict_validation_warnings = _collect_strict_validation_warnings(
        team_a_label=team_a_name,
        team_b_label=team_b_name,
        xi_a=xi_a,
        xi_b=xi_b,
        scored_squad_a=scored_a,
        scored_squad_b=scored_b,
        history_sync_debug=history_sync_debug,
        batting_order_warnings=batting_order_warnings,
    )
    _annotate_bench_xi_margins(scored_a, xi_a)
    _annotate_bench_xi_margins(scored_b, xi_b)

    _ap_phase("xi_scenario_select_batting_order_validate_and_strict_checks_ms", _t_xi_pick)

    _t_imp = time.perf_counter()
    subs_a, impact_dbg_a, impact_dbg_all_a = impact_subs(
        scored_a,
        xi_a,
        is_chasing=chase_ctx_a,
        conditions=conditions,
        team_bats_first=a_bats_first,
        team_display_name=team_a_name,
        canonical_team_key=tk_a,
        venue_key=venue.key,
        venue_key_candidates=pattern_vkeys,
    )
    subs_b, impact_dbg_b, impact_dbg_all_b = impact_subs(
        scored_b,
        xi_b,
        is_chasing=chase_ctx_b,
        conditions=conditions,
        team_bats_first=(not a_bats_first) if a_bats_first is not None else None,
        team_display_name=team_b_name,
        canonical_team_key=tk_b,
        venue_key=venue.key,
        venue_key_candidates=pattern_vkeys,
    )
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["impact_subs_both_teams_ms"] = round(
            (time.perf_counter() - _t_imp) * 1000.0, 2
        )
    _ap_phase("impact_subs_both_teams_ms", _t_imp)

    _t_win_tail = time.perf_counter()
    str_a = _xi_strength(xi_a) if len(xi_a) == 11 else _xi_strength(xi_a) * 11 / max(1, len(xi_a))
    str_b = _xi_strength(xi_b) if len(xi_b) == 11 else _xi_strength(xi_b) * 11 / max(1, len(xi_b))

    toss = toss_matchup_effects(
        team_a_name,
        team_b_name,
        strength_a=str_a,
        strength_b=str_b,
        conditions=conditions,
        venue_key=venue.key,
    )
    chase_share, chase_n = _resolve_chase_prior(vkeys, hctx)
    chase_boost_logit = 0.0
    if chase_n >= config.LEARN_MIN_SAMPLES_CHASE:
        chase_boost_logit = (
            config.LEARN_WEIGHT_CHASE_BIAS_LOGIT * (chase_share - 0.5) * 2.0
        )
    toss["venue_chase_win_share"] = float(chase_share)
    toss["venue_chase_sample_wins"] = int(chase_n)
    toss["chase_boost_logit_applied"] = float(chase_boost_logit)

    win_marginal_logistic = win_probability(
        str_a,
        str_b,
        conditions,
        toss,
        assume_team_a_bats_first=a_bats_first,
        chase_boost_logit=chase_boost_logit,
    )
    win_a_bf_logistic = win_probability(
        str_a, str_b, conditions, toss, assume_team_a_bats_first=True, chase_boost_logit=0.0
    )
    win_a_ch_logistic = win_probability(
        str_a,
        str_b,
        conditions,
        toss,
        assume_team_a_bats_first=False,
        chase_boost_logit=chase_boost_logit,
    )

    _t_win = time.perf_counter()
    match_rows = db.fetch_match_results_meta(450)
    win_eng = win_probability_engine.compute_win_probability(
        team_a_name,
        team_b_name,
        xi_a,
        xi_b,
        order_a,
        order_b,
        venue,
        conditions,
        venue_keys=vkeys,
        match_rows=match_rows,
        toss_scenario_key=toss_scenario_key,
        a_bats_first_selected=a_bats_first,
        chase_share_by_venue=hctx.chase_share_by_venue,
        is_night_fixture=is_night,
    )
    eng_d = win_eng.to_dict()
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["win_engine_fetch_matches_and_compute_ms"] = round(
            (time.perf_counter() - _t_win) * 1000.0, 2
        )
    _ap_phase("win_engine_fetch_match_meta_and_compute_ms", _t_win)
    headline_a = eng_d["team_a_win_pct_selected_toss"] / 100.0
    neutral_a = eng_d["marginal_team_a_win_pct"] / 100.0
    sf0 = eng_d.get("scenario_factors") or {}
    fac_a = sf0.get("a_bats_first") or {}
    xi_f = fac_a.get("xi_strength") or {}
    vn_f = fac_a.get("venue") or {}
    h2_f = fac_a.get("head_to_head") or {}
    cw_f = fac_a.get("conditions_weather") or {}
    gap_xi = abs(float(xi_f.get("team_a", 0)) - float(xi_f.get("team_b", 0)))
    tilt_v = float(vn_f.get("team_a", 0)) - float(vn_f.get("team_b", 0))
    tilt_h = float(h2_f.get("team_a", 0)) - float(h2_f.get("team_b", 0))
    tilt_cw = float(cw_f.get("team_a", 0)) - float(cw_f.get("team_b", 0))
    dew_c = float(conditions.get("dew_risk", 0.5))
    driver_extra = (
        f" Weight readout (A bats first scenario slice): XI strength separation≈{gap_xi:.2f}; "
        f"venue history tilt A−B≈{tilt_v:.2f}; head-to-head tilt≈{tilt_h:.2f}; "
        f"conditions/weather tilt≈{tilt_cw:.2f} (fixture dew_risk≈{dew_c:.2f})."
    )
    win_marginal = {
        "team_a_win": float(headline_a),
        "team_b_win": float(1.0 - headline_a),
        "model": "rule_engine",
        "uses_selected_toss": a_bats_first is not None,
        "neutral_toss_team_a_win": float(neutral_a),
        "driver_summary": (str(eng_d.get("explanation") or "").strip() + driver_extra).strip()[:520],
    }
    win_a_bf = {
        "team_a_win": float(eng_d["team_a_win_pct_if_a_bats_first"] / 100.0),
        "team_b_win": float(eng_d["team_b_win_pct_if_a_bats_first"] / 100.0),
        "model": "rule_engine",
    }
    win_a_ch = {
        "team_a_win": float(eng_d["team_a_win_pct_if_b_bats_first"] / 100.0),
        "team_b_win": float(eng_d["team_b_win_pct_if_b_bats_first"] / 100.0),
        "model": "rule_engine",
    }

    confidence = _prediction_confidence(
        xi_a, xi_b, hctx, str_a, str_b, scored_a=scored_a, scored_b=scored_b
    )
    _ap_phase(
        "toss_logistic_win_probability_marginals_and_confidence_ms",
        _t_win_tail,
    )

    def _structured_squad_rows(squad: list[SquadPlayer]) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "squad_display_name": p.name,
                "player_name": p.name,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "canonical_player_key": p.canonical_player_key or p.player_key or learner.normalize_player_key(p.name),
                "team_name": p.team_display_name,
                "canonical_team_key": p.canonical_team_key,
                "role_bucket": p.role_bucket,
                "overseas": p.is_overseas,
                "role": p.role,
            }
            for p in squad
        ]

    def xi_payload(
        xi: list[SquadPlayer], order: list[str], scenario_branch: str
    ) -> list[dict[str, Any]]:
        pos = {n: i + 1 for i, n in enumerate(order)}
        rows = []
        for p in xi:
            hd = dict(getattr(p, "history_debug", {}) or {})
            sx_br = (hd.get("scenario_xi") or {}).get(scenario_branch) or {}
            smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
            bbd = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
            lmd = smd.get("last_match_detail") if isinstance(smd.get("last_match_detail"), dict) else {}
            rows.append(
                {
                    "name": p.name,
                    "squad_display_name": p.name,
                    "player_key": p.player_key or learner.normalize_player_key(p.name),
                    "in_current_squad": hd.get("in_current_squad", True),
                    "role": p.role,
                    "role_bucket": p.role_bucket,
                    "is_wk_role": bool(getattr(p, "is_wicketkeeper", False)),
                    "designated_keeper": bool(hd.get("designated_keeper")),
                    "batting_roles": list(p.batting_roles),
                    "bowling_type": p.bowling_type,
                    "overseas": p.is_overseas,
                    "bat_order": pos.get(p.name),
                    "batting_order_rank_final": hd.get("batting_order_rank_final"),
                    "composite": round(p.composite, 4),
                    "selection_score": round(float(getattr(p, "selection_score", 0.0)), 5),
                    "history_xi_score": round(float(getattr(p, "history_xi_score", 0.0)), 5),
                    "history_rows_found": hd.get("history_rows_found"),
                    "history_distinct_matches": hd.get("history_distinct_matches"),
                    "batting_positions_history": hd.get("batting_positions_history"),
                    "batting_position_rows_found": hd.get("batting_position_rows_found"),
                    "batting_position_ema": hd.get("batting_position_ema"),
                    "pbp_primary_slot_matches": hd.get("pbp_primary_slot_matches"),
                    "canonical_player_key": hd.get("canonical_player_key"),
                    "selection_score_components": hd.get("selection_score_components"),
                    "history_batting_ema": round(float(getattr(p, "history_batting_ema", 99.0)), 3),
                    "history_delta": round(p.history_delta, 5),
                    "history_notes": p.history_notes[:4],
                    "batting_order_source": hd.get("batting_order_source", ""),
                    "batting_order_final": hd.get("batting_order_final", ""),
                    "history_enrichment_current_squad_only": hd.get(
                        "history_enrichment_current_squad_only"
                    ),
                    "batting_order_strict_xi_scope": hd.get("batting_order_strict_xi_scope"),
                    "used_current_season_history": hd.get("used_current_season_history"),
                    "used_prior_season_fallback": hd.get("used_prior_season_fallback"),
                    "used_venue_history": hd.get("used_venue_history"),
                    "fallback_heuristics_only": hd.get("fallback_heuristics_only"),
                    "xi_selection_tier": hd.get("xi_selection_tier", ""),
                    "xi_used_prior_season_rows": hd.get("xi_used_prior_season_rows"),
                    "recent5_xi_rate": hd.get("recent5_xi_rate"),
                    "venue_xi_rate": hd.get("venue_xi_rate"),
                    "prior_season_xi_rate": hd.get("prior_season_xi_rate"),
                    "h2h_weighted_xi_rate": hd.get("h2h_weighted_xi_rate"),
                    "h2h_venue_xi_rate": hd.get("h2h_venue_xi_rate"),
                    "h2h_fixtures_in_layer": hd.get("h2h_fixtures_in_layer"),
                    "h2h_batting_order_used_opponent_history": hd.get(
                        "h2h_batting_order_used_opponent_history"
                    ),
                    "h2h_phase_blend_applied": hd.get("h2h_phase_blend_applied"),
                    "weights_h2h_xi_venue": hd.get("weights_h2h_xi_venue"),
                    "scoring_breakdown": hd.get("scoring_breakdown"),
                    "probable_first_choice_prior": hd.get("probable_first_choice_prior"),
                    "global_ipl_history_presence": hd.get("global_ipl_history_presence"),
                    "global_selection_frequency": hd.get("global_selection_frequency"),
                    "global_batting_position_pattern": hd.get("global_batting_position_pattern"),
                    "global_role_strength": hd.get("global_role_strength"),
                    "used_global_fallback_prior": hd.get("used_global_fallback_prior"),
                    "first_choice_prior_debug": hd.get("first_choice_prior_debug"),
                    "history_usage_debug": hd.get("history_usage_debug"),
                    "batting_order_diagnostic_source": hd.get("batting_order_diagnostic_source"),
                    "fetched_from_team_slug": hd.get("fetched_from_team_slug"),
                    "in_selected_team_fetched_squad": hd.get("in_selected_team_fetched_squad"),
                    "in_opposite_team_fetched_squad": hd.get("in_opposite_team_fetched_squad"),
                    "wrong_side_squad_assignment": hd.get("wrong_side_squad_assignment"),
                    "stale_cached_entry_detected": hd.get("stale_cached_entry_detected"),
                    "valid_current_squad_new_to_franchise": hd.get("valid_current_squad_new_to_franchise"),
                    "selected_franchise_history_presence": hd.get("selected_franchise_history_presence"),
                    "history_for_other_franchises_presence": hd.get("history_for_other_franchises_presence"),
                    "captain_selected_for_team": hd.get("captain_selected_for_team"),
                    "wicketkeeper_selected_for_team": hd.get("wicketkeeper_selected_for_team"),
                    "captain_boost_applied": hd.get("captain_boost_applied"),
                    "wicketkeeper_boost_applied": hd.get("wicketkeeper_boost_applied"),
                    "selection_reason_summary": hd.get("selection_reason_summary"),
                    "base_xi_score": hd.get("base_xi_score"),
                    "base_xi_reason": hd.get("base_xi_reason"),
                    "base_xi_selected": hd.get("base_xi_selected"),
                    "core_anchor_strength": hd.get("core_anchor_strength"),
                    "condition_adjustment_reason": hd.get("condition_adjustment_reason"),
                    "included_due_to_conditions": hd.get("included_due_to_conditions"),
                    "excluded_due_to_conditions": hd.get("excluded_due_to_conditions"),
                    "history_source_used": hd.get("history_source_used"),
                    "fallback_used": hd.get("fallback_used"),
                    "derive_player_profile": hd.get("derive_player_profile"),
                    "marquee_tier": hd.get("marquee_tier"),
                    "marquee_source": hd.get("marquee_source"),
                    "marquee_reason": hd.get("marquee_reason"),
                    "marquee_suggested_score": hd.get("marquee_suggested_score"),
                    "marquee_suggested_score_raw": hd.get("marquee_suggested_score_raw"),
                    "marquee_suggested_rank_pct": hd.get("marquee_suggested_rank_pct"),
                    "batting_order_reason_summary": hd.get("batting_order_reason_summary"),
                    "role_band": hd.get("role_band"),
                    "dominant_position": hd.get("dominant_position"),
                    "batting_band": hd.get("batting_band"),
                    "final_position": hd.get("final_position"),
                    "moved_outside_band": hd.get("moved_outside_band"),
                    "bowler_order_guardrail_applied": hd.get("bowler_order_guardrail_applied"),
                    "bowler_phase_summary": hd.get("bowler_phase_summary"),
                    "is_powerplay_bowler_candidate": _is_powerplay_bowler_candidate(p),
                    "is_death_bowler_candidate": _is_death_bowler_candidate(p),
                    "batting_position_history_basis": hd.get("batting_position_history_basis"),
                    "final_order_reason": hd.get("final_order_reason"),
                    "batting_order_signal_source_ranked": hd.get("batting_order_signal_source_ranked"),
                    "xi_selection_frequency": (hd.get("selection_score_components") or {}).get("xi_selection_frequency"),
                    "recent_usage_score": (hd.get("selection_score_components") or {}).get("recent_usage_score"),
                    "role_stability_score": (hd.get("selection_score_components") or {}).get("role_stability_score"),
                    "venue_team_pattern_boost": (hd.get("selection_score_components") or {}).get(
                        "venue_team_pattern_boost"
                    ),
                    "venue_fit_conditions_align": (hd.get("selection_score_components") or {}).get(
                        "venue_fit_conditions_align"
                    ),
                    "h2h_boost_rate": (hd.get("selection_score_components") or {}).get("h2h_boost_rate"),
                    "weather_score_proxy": (hd.get("selection_score_components") or {}).get("weather_score_proxy"),
                    "perspectives": {k: round(v, 4) for k, v in p.perspectives.items()},
                    "selection_model_debug": hd.get("selection_model_debug"),
                    "was_in_last_match_xi": lmd.get("was_in_last_match_xi"),
                    "last_match_continuity_score": bbd.get("last_match_continuity_score"),
                    "recent_form_score": bbd.get("recent_form_score"),
                    "ipl_history_and_role_score": bbd.get("ipl_history_and_role_score"),
                    "stable_role_identity_score": bbd.get("stable_role_identity_score"),
                    "core_player_signal": hd.get("core_anchor_strength"),
                    "team_balance_fit_score": bbd.get("team_balance_fit_score"),
                    "condition_adjustment": smd.get("tactical_adjustment_total"),
                    "final_xi_decision_reason": (
                        hd.get("condition_adjustment_reason")
                        or hd.get("base_xi_reason")
                        or hd.get("selection_reason_summary")
                    ),
                    "scenario_xi_branch_used": scenario_branch,
                    "scenario_selection_score": sx_br.get("scenario_selection_score"),
                    "scenario_adjustment_total": sx_br.get("scenario_adjustment_total"),
                    "scenario_adjustment_breakdown": sx_br.get("scenario_adjustment_breakdown"),
                    "selected_for_batting_first_reason": sx_br.get(
                        "selected_for_batting_first_reason"
                    ),
                    "selected_for_bowling_first_reason": sx_br.get(
                        "selected_for_bowling_first_reason"
                    ),
                    "scenario_xi_if_team_bats_first": (hd.get("scenario_xi") or {}).get(
                        "if_team_bats_first"
                    ),
                    "scenario_xi_if_team_bowls_first": (hd.get("scenario_xi") or {}).get(
                        "if_team_bowls_first"
                    ),
                    "scenario_rank_used": hd.get("scenario_rank_used"),
                    "selection_changed_due_to_scenario": hd.get(
                        "selection_changed_due_to_scenario"
                    ),
                    "xi_scenario_branch_for_xi_build": hd.get("xi_scenario_branch_for_xi_build"),
                }
            )
        rows.sort(key=lambda r: r["bat_order"] or 99)
        return rows

    xi_scenario_alternates: Optional[dict[str, Any]] = None
    if a_bats_first is None and xi_a_if_bats_first is not None:
        w_alt: list[str] = []

        def _alt_bo(xi: list[SquadPlayer], tnm: str) -> list[str]:
            if len(xi) != 11:
                return [p.name for p in xi]
            return build_batting_order(
                xi, conditions, team_name=tnm, venue_keys=vkeys, out_warnings=w_alt
            )

        xi_scenario_alternates = {
            "team_a": {
                "xi_if_team_bats_first": xi_payload(
                    xi_a_if_bats_first,
                    _alt_bo(xi_a_if_bats_first, team_a_name),
                    "if_team_bats_first",
                ),
                "xi_if_team_bowls_first": xi_payload(
                    xi_a_if_bowls_first,
                    _alt_bo(xi_a_if_bowls_first, team_a_name),
                    "if_team_bowls_first",
                ),
            },
            "team_b": {
                "xi_if_team_bats_first": xi_payload(
                    xi_b_if_bats_first,
                    _alt_bo(xi_b_if_bats_first, team_b_name),
                    "if_team_bats_first",
                ),
                "xi_if_team_bowls_first": xi_payload(
                    xi_b_if_bowls_first,
                    _alt_bo(xi_b_if_bowls_first, team_b_name),
                    "if_team_bowls_first",
                ),
            },
        }

    os_squad_a = sum(1 for p in sa if p.is_overseas)
    os_squad_b = sum(1 for p in sb if p.is_overseas)
    os_xi_a = sum(1 for p in xi_a if p.is_overseas)
    os_xi_b = sum(1 for p in xi_b if p.is_overseas)

    omitted_a = _omitted_xi_report(
        scored_a, xi_a, scenario_branch=br_a, xi_baseline=xi_a_base
    )
    omitted_b = _omitted_xi_report(
        scored_b, xi_b, scenario_branch=br_b, xi_baseline=xi_b_base
    )

    def _rule_trace_for_team(
        *,
        team_label: str,
        scored_squad: list[SquadPlayer],
        xi: list[SquadPlayer],
        order: list[str],
        xi_rules: Any,
        repair_swaps: list[dict[str, Any]],
        omitted_rows: list[dict[str, Any]],
        batting_order_warnings: list[str],
    ) -> dict[str, Any]:
        in_by_repair = {str(s.get("in") or "") for s in (repair_swaps or []) if s.get("in")}
        out_by_repair = {str(s.get("out") or "") for s in (repair_swaps or []) if s.get("out")}
        by_name = {p.name: p for p in scored_squad}

        def _selected_because(p: SquadPlayer) -> list[str]:
            hd = getattr(p, "history_debug", None) or {}
            smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
            bbd = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
            lmd = smd.get("last_match_detail") if isinstance(smd.get("last_match_detail"), dict) else {}
            out: list[str] = []
            tier = str(hd.get("marquee_tier") or "").strip().lower()
            if tier in ("tier_1", "tier_2"):
                out.append("marquee")
            if bool(lmd.get("was_in_last_match_xi")) or float(bbd.get("last_match_continuity_score") or 0.0) >= 0.55:
                out.append("continuity")
            if float(bbd.get("recent_form_score") or 0.0) >= 0.35:
                out.append("form")
            if float(bbd.get("ipl_history_and_role_score") or 0.0) >= 0.3:
                out.append("history")
            if float(bbd.get("stable_role_identity_score") or 0.0) >= 0.3:
                out.append("role")
            if bool(hd.get("included_due_to_conditions")) or bool(hd.get("condition_adjustment_reason")):
                out.append("conditions")
            if p.name in in_by_repair:
                out.append("repair")
            if not out:
                out.append("model_rank")
            return out

        selected = []
        for p in xi:
            hd = getattr(p, "history_debug", None) or {}
            selected.append(
                {
                    "name": p.name,
                    "selected_because": _selected_because(p),
                    "final_xi_decision_reason": (
                        hd.get("condition_adjustment_reason")
                        or hd.get("base_xi_reason")
                        or hd.get("selection_reason_summary")
                        or ""
                    ),
                }
            )

        dropped = []
        xi_names = {p.name for p in xi}
        for row in (omitted_rows or []):
            nm = str(row.get("name") or "")
            if nm and nm not in xi_names:
                dropped.append(
                    {
                        "name": nm,
                        "dropped_because": str(row.get("omitted_reason_summary") or row.get("omission_reason") or ""),
                        "was_repaired_out": nm in out_by_repair,
                    }
                )

        batting_trace = []
        for pos, nm in enumerate(order, start=1):
            p = by_name.get(nm)
            if p is None:
                continue
            hd = getattr(p, "history_debug", None) or {}
            batting_trace.append(
                {
                    "name": nm,
                    "batting_band": hd.get("batting_band"),
                    "allowed_range": [hd.get("batting_allowed_min"), hd.get("batting_allowed_max")],
                    "final_position": hd.get("final_position") or pos,
                    "guardrail_applied": bool(hd.get("bowler_order_guardrail_applied")),
                    "moved_outside_band": bool(hd.get("moved_outside_band")),
                }
            )

        return {
            "final_xi_rule_summary": xi_rules.summary,
            "hard_constraints_satisfied": bool(xi_rules.hard_ok),
            "hard_violations": [v.code for v in xi_rules.violations],
            "semi_hard_warnings": [w.code for w in xi_rules.warnings],
            "selected_players": selected,
            "dropped_players_top": dropped[:18],
            "batting_order": batting_trace,
            "batting_order_warnings": list(batting_order_warnings or [])[:6],
            "xi_player_metadata_sources": {
                p.name: (getattr(p, "history_debug", None) or {}).get("player_metadata_source_runtime")
                for p in (xi or [])
            },
        }

    rule_trace_a = _rule_trace_for_team(
        team_label=team_a_name,
        scored_squad=scored_a,
        xi=xi_a,
        order=order_a,
        xi_rules=xi_rules_a,
        repair_swaps=xi_a_repair_swaps,
        omitted_rows=omitted_a,
        batting_order_warnings=batting_order_warnings,
    )
    rule_trace_b = _rule_trace_for_team(
        team_label=team_b_name,
        scored_squad=scored_b,
        xi=xi_b,
        order=order_b,
        xi_rules=xi_rules_b,
        repair_swaps=xi_b_repair_swaps,
        omitted_rows=omitted_b,
        batting_order_warnings=batting_order_warnings,
    )

    prediction_layer_debug: dict[str, Any] = {
        "metadata_dependency_report": _metadata_dependency_report(xi_a=xi_a, xi_b=xi_b),
        "team_a": {
            "history_usage_per_player": _squad_post_linkage_history_rows(scored_a),
            "history_linkage_squad_rollup": _history_linkage_squad_rollup(scored_a),
            "scoring_breakdown_per_player": _squad_scoring_breakdown_rows(scored_a),
            "omitted_from_playing_xi": omitted_a,
            "base_to_final_condition_changes": xi_a_condition_changes,
            "xi_batting_order_diagnostics": _xi_batting_order_diagnostics_rows(xi_a, order_a),
            "impact_sub_ranking": impact_dbg_all_a,
            "impact_sub_ranking_top5": impact_dbg_a,
            "rule_trace": rule_trace_a,
        },
        "team_b": {
            "history_usage_per_player": _squad_post_linkage_history_rows(scored_b),
            "history_linkage_squad_rollup": _history_linkage_squad_rollup(scored_b),
            "scoring_breakdown_per_player": _squad_scoring_breakdown_rows(scored_b),
            "omitted_from_playing_xi": omitted_b,
            "base_to_final_condition_changes": xi_b_condition_changes,
            "xi_batting_order_diagnostics": _xi_batting_order_diagnostics_rows(xi_b, order_b),
            "impact_sub_ranking": impact_dbg_all_b,
            "impact_sub_ranking_top5": impact_dbg_b,
            "rule_trace": rule_trace_b,
        },
    }

    if not getattr(config, "PREDICTION_FULL_DEBUG_PAYLOAD", False):
        prediction_layer_debug = _slim_prediction_layer_debug(prediction_layer_debug)

    _t_asm = time.perf_counter()
    _timing_out = bool(
        getattr(config, "PREDICTION_TIMING_LOG", False) or audit_profile.audit_enabled()
    )

    _out = {
        "conditions": conditions,
        "weather": weather,
        "squad_debug": {
            "team_a_canonical_franchise": canon_a,
            "team_b_canonical_franchise": canon_b,
            "team_a_role_bucket_counts": role_dist_a,
            "team_b_role_bucket_counts": role_dist_b,
            "structured_squad_team_a": _structured_squad_rows(sa),
            "structured_squad_team_b": _structured_squad_rows(sb),
        },
        "history_sync_debug": history_sync_debug,
        "prediction_layer_debug": prediction_layer_debug,
        "prediction_timing_ms": prediction_timing_ms if _timing_out else None,
        "xi_validation": {
            "team_a_in_squad": sub_ok_a,
            "team_b_in_squad": sub_ok_b,
            "team_a_xi_names": [p.name for p in xi_a],
            "team_b_xi_names": [p.name for p in xi_b],
            "constraints_team_a_ok": v_ok_a and len(xi_a) == 11,
            "constraints_team_b_ok": v_ok_b and len(xi_b) == 11,
            "strict_xi_subset_of_fetched_squad": bool(sub_ok_a and sub_ok_b),
            "history_enrichment_limited_to_fetched_squad_only": True,
            "batting_order_names_subset_of_selected_xi": True,
            "strict_validation_warnings": strict_validation_warnings,
            "team_a_repaired": bool(xi_a_repaired),
            "team_b_repaired": bool(xi_b_repaired),
            "team_a_repair_swaps": xi_a_repair_swaps,
            "team_b_repair_swaps": xi_b_repair_swaps,
            "team_a_hard_constraints_satisfied": bool(xi_a_repair_enforce.get("hard_constraints_satisfied")),
            "team_b_hard_constraints_satisfied": bool(xi_b_repair_enforce.get("hard_constraints_satisfied")),
            "team_a_hard_constraint_failures": xi_a_repair_enforce.get("failed_constraints") or [],
            "team_b_hard_constraint_failures": xi_b_repair_enforce.get("failed_constraints") or [],
            "team_a_repair_failure_reason": xi_a_repair_enforce.get("repair_failure_reason") or "",
            "team_b_repair_failure_reason": xi_b_repair_enforce.get("repair_failure_reason") or "",
        },
        "selection_debug": {
            "team_a": {
                "xi_validation": {
                    **xi_counts_a,
                    "final_xi_valid": bool(v_ok_a and len(xi_a) == 11),
                    "failed_rules": list(v_err_a),
                    "repair_pass_ran": bool(xi_a_repaired),
                    "hard_constraints_satisfied": bool(xi_a_repair_enforce.get("hard_constraints_satisfied")),
                    "repair_failure_reason": xi_a_repair_enforce.get("repair_failure_reason") or "",
                    "semi_hard_failed": xi_a_repair_enforce.get("semi_hard_failed") or [],
                },
                "base_xi_names": [p.name for p in xi_a_base],
                "condition_change_count": len(xi_a_condition_changes),
                "condition_changes": xi_a_condition_changes,
                "xi_repaired": bool(xi_a_repaired),
                "repair_swaps": xi_a_repair_swaps,
                "overseas_target_preference_applied": bool(overseas_dbg_a.get("overseas_target_preference_applied")),
                "overseas_repair_applied": bool(
                    overseas_dbg_a_pre.get("overseas_repair_applied") or overseas_dbg_a.get("overseas_repair_applied")
                ),
                "overseas_target": overseas_dbg_a.get("overseas_target"),
                "overseas_min_required": overseas_dbg_a.get("overseas_min_required"),
                "best_excluded_overseas_candidates": overseas_dbg_a.get("best_excluded_overseas_candidates") or [],
                "why_4th_overseas_selected_or_not": overseas_dbg_a.get("why_4th_overseas_selected_or_not") or "",
                "overseas_preference_swap": overseas_dbg_a.get("overseas_swap"),
                "overseas_preference_swaps": overseas_dbg_a.get("overseas_swaps") or [],
                "total_impact_subs_selected": len(subs_a),
                "impact_candidate_pool_size": len([p for p in scored_a if p.name not in {x.name for x in xi_a}]),
                "impact_fallback_used": bool((impact_dbg_a or [{}])[0].get("fallback_used")) if impact_dbg_a else False,
                "condition_change_profile": (
                    "no_change"
                    if not xi_a_condition_changes
                    else "bowling_balance_tweak_only"
                ),
            },
            "team_b": {
                "xi_validation": {
                    **xi_counts_b,
                    "final_xi_valid": bool(v_ok_b and len(xi_b) == 11),
                    "failed_rules": list(v_err_b),
                    "repair_pass_ran": bool(xi_b_repaired),
                    "hard_constraints_satisfied": bool(xi_b_repair_enforce.get("hard_constraints_satisfied")),
                    "repair_failure_reason": xi_b_repair_enforce.get("repair_failure_reason") or "",
                    "semi_hard_failed": xi_b_repair_enforce.get("semi_hard_failed") or [],
                },
                "base_xi_names": [p.name for p in xi_b_base],
                "condition_change_count": len(xi_b_condition_changes),
                "condition_changes": xi_b_condition_changes,
                "xi_repaired": bool(xi_b_repaired),
                "repair_swaps": xi_b_repair_swaps,
                "overseas_target_preference_applied": bool(overseas_dbg_b.get("overseas_target_preference_applied")),
                "overseas_repair_applied": bool(
                    overseas_dbg_b_pre.get("overseas_repair_applied") or overseas_dbg_b.get("overseas_repair_applied")
                ),
                "overseas_target": overseas_dbg_b.get("overseas_target"),
                "overseas_min_required": overseas_dbg_b.get("overseas_min_required"),
                "best_excluded_overseas_candidates": overseas_dbg_b.get("best_excluded_overseas_candidates") or [],
                "why_4th_overseas_selected_or_not": overseas_dbg_b.get("why_4th_overseas_selected_or_not") or "",
                "overseas_preference_swap": overseas_dbg_b.get("overseas_swap"),
                "overseas_preference_swaps": overseas_dbg_b.get("overseas_swaps") or [],
                "total_impact_subs_selected": len(subs_b),
                "impact_candidate_pool_size": len([p for p in scored_b if p.name not in {x.name for x in xi_b}]),
                "impact_fallback_used": bool((impact_dbg_b or [{}])[0].get("fallback_used")) if impact_dbg_b else False,
                "condition_change_profile": (
                    "no_change"
                    if not xi_b_condition_changes
                    else "bowling_balance_tweak_only"
                ),
            },
        },
        "toss_scenario": {
            "key": toss_scenario_key,
            "team_a_bats_first": a_bats_first,
            "chase_context_team_a": chase_ctx_a,
            "chase_context_team_b": chase_ctx_b,
            "team_a_xi_scenario_branch_used": br_a,
            "team_b_xi_scenario_branch_used": br_b,
            "team_a_condition_change_count": len(xi_a_condition_changes),
            "team_b_condition_change_count": len(xi_b_condition_changes),
        },
        "xi_scenario_alternates": xi_scenario_alternates,
        "overseas_counts": {
            "squad_team_a": os_squad_a,
            "squad_team_b": os_squad_b,
            "xi_team_a": os_xi_a,
            "xi_team_b": os_xi_b,
        },
        "batting_order_summary": {
            "team_a": {
                **_batting_order_sources_summary(xi_a),
                "lineup_summary": _batting_order_team_lineup_summary(xi_a),
            },
            "team_b": {
                **_batting_order_sources_summary(xi_b),
                "lineup_summary": _batting_order_team_lineup_summary(xi_b),
            },
        },
        "chase_defend_bias": eng_d.get("chase_defend_context") or {},
        "team_a": {
            "name": team_a_name,
            "base_xi": [p.name for p in xi_a_base],
            "xi": xi_payload(xi_a, order_a, br_a),
            "batting_order": order_a,
            "impact_subs": _impact_subs_payload(subs_a, impact_dbg_a),
            "squad_size": len(sa),
            "squad_overseas": os_squad_a,
            "xi_overseas": os_xi_a,
            "valid_xi": v_ok_a and len(xi_a) == 11,
            "xi_constraint_errors": v_err_a if not v_ok_a else [],
        },
        "team_b": {
            "name": team_b_name,
            "base_xi": [p.name for p in xi_b_base],
            "xi": xi_payload(xi_b, order_b, br_b),
            "batting_order": order_b,
            "impact_subs": _impact_subs_payload(subs_b, impact_dbg_b),
            "squad_size": len(sb),
            "squad_overseas": os_squad_b,
            "xi_overseas": os_xi_b,
            "valid_xi": v_ok_b and len(xi_b) == 11,
            "xi_constraint_errors": v_err_b if not v_ok_b else [],
        },
        "toss_effects": toss,
        "win_probability": win_marginal,
        "win_probability_engine": eng_d,
        "win_probability_logistic": {
            "marginal": win_marginal_logistic,
            "if_team_a_bats_first": win_a_bf_logistic,
            "if_team_a_bowls_first": win_a_ch_logistic,
        },
        "win_if_team_a_bats_first": win_a_bf,
        "win_if_team_a_bowls_first": win_a_ch,
        "strength": {"team_a": str_a, "team_b": str_b},
        "prediction_confidence": confidence,
        "learning_context": {
            "stored_matches": hctx.db_match_count,
            "venue_keys_tried": vkeys,
            "derive_team_selection_pattern_keys_tried": pattern_vkeys,
            "fixture_night": is_night,
            "toss_scenario_used": toss_scenario_key,
            "history_weights_note": (
                "History bump = weighted blend of XI frequency, slot, bowling load, "
                "venue–team picks, overseas mix, day/night, dew (see config.py)."
            ),
            "stage3_note": (
                "Selection_score blends franchise SQLite history, Stage-2 player_profiles, "
                "and venue team_selection_patterns (when keys match derive)."
            ),
        },
    }
    _ap_phase("prediction_result_dict_assembly_ms", _t_asm)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["run_prediction_total_ms"] = round(
            (time.perf_counter() - _run_t0) * 1000.0, 2
        )
        _perf_logger.info("run_prediction timing_ms=%s", prediction_timing_ms)
    if _audit_par:
        _audit_par.close_success(_out)
    return _out
