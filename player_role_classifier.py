from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Optional

import config
import db
import learner
import player_alias_resolve
from ipl_squad import ALL_ROUNDER, BATTER, BOWLER, WK_BATTER


_SPIN_BUCKETS = {
    "finger_spin",
    "wrist_spin",
    "left_arm_orthodox",
    "mystery_spin",
}

_PACE_BUCKETS = {
    "pace",
    "right_arm_fast",
    "right_arm_fast_medium",
    "left_arm_fast",
    "left_arm_fast_medium",
}

_SPIN_TOKENS = (
    "spin",
    "wrist",
    "offbreak",
    "off break",
    "legbreak",
    "leg break",
    "orthodox",
    "slow",
    "googly",
    "chinaman",
)

_PACE_TOKENS = (
    "fast",
    "medium",
    "seam",
    "swing",
    "pace",
)


def _meta_dict(p: Any) -> dict[str, Any]:
    hd = getattr(p, "history_debug", None) or {}
    m = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    return m if isinstance(m, dict) else {}


def _role_string_indicates_keeper(raw: str) -> bool:
    """
    True when structured role text clearly denotes a wicketkeeper.

    Avoid naive substring checks: e.g. ``"wk" in "awkward"`` or ``"wk" in "hawk"`` are false positives
    for XI wicketkeeper caps.
    """
    s = "_".join(str(raw or "").strip().lower().replace("-", " ").split())
    if not s:
        return False
    if s in ("wk", "wk_batter", "wicketkeeper_batter", "wicketkeeper"):
        return True
    if "wicketkeeper" in s:
        return True
    parts = [t for t in s.split("_") if t]
    if not parts:
        return False
    if "wk" in parts:
        return True
    if any(p == "keeper" or p.startswith("wicketkeeper") for p in parts):
        return True
    return False


def _meta_role_indicates_keeper(m: dict[str, Any]) -> bool:
    pr = str(m.get("primary_role") or "").strip().lower()
    sr = str(m.get("secondary_role") or "").strip().lower()
    return _role_string_indicates_keeper(pr) or _role_string_indicates_keeper(sr)


def _primary_role_indicates_keeper(m: dict[str, Any]) -> bool:
    pr = str(m.get("primary_role") or "").strip().lower()
    return _role_string_indicates_keeper(pr)


def _secondary_only_keeper_meta(meta: dict[str, Any]) -> bool:
    """
    Primary role is non-keeper (e.g. batter) but secondary lists wicketkeeper — IPL squad flex.
    These players should remain eligible as designated-keeper *candidates* but must not each
    consume a ``wk_max`` slot.
    """
    pr = str(meta.get("primary_role") or "").strip().lower()
    sr = str(meta.get("secondary_role") or "").strip().lower()
    if not pr:
        return False
    if _role_string_indicates_keeper(pr):
        return False
    return _role_string_indicates_keeper(sr)


def _wk_backup_wk_batter_excluded_from_wk_cap(p: Any, meta: dict[str, Any], hd: dict[str, Any]) -> bool:
    """
    Tier-2/3 WK-Batter listings without recent keeper usage: treat as squad backup / depth,
    not an additional hard-cap wicketkeeper slot.
    """
    role_bucket = str(getattr(p, "role_bucket", "") or "")
    if role_bucket != WK_BATTER:
        return False
    tier = str(hd.get("marquee_tier") or "").strip().lower()
    if tier not in ("tier_2", "tier_3"):
        return False
    smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    lmd = smd.get("last_match_detail") if isinstance(smd.get("last_match_detail"), dict) else {}
    if bool(lmd.get("last_match_is_keeper") or lmd.get("is_keeper")):
        return False
    return True


def _broad_designated_keeper_candidate(p: Any, meta: dict[str, Any], role_bucket: str) -> bool:
    return (
        bool(getattr(p, "is_wicketkeeper", False))
        or role_bucket == WK_BATTER
        or _meta_role_indicates_keeper(meta)
    )


def _counts_toward_wk_max_cap(p: Any, meta: dict[str, Any], hd: dict[str, Any], role_bucket: str) -> bool:
    if not _broad_designated_keeper_candidate(p, meta, role_bucket):
        return False
    if _secondary_only_keeper_meta(meta):
        return False
    if _wk_backup_wk_batter_excluded_from_wk_cap(p, meta, hd):
        return False
    if role_bucket == BATTER and bool(getattr(p, "is_wicketkeeper", False)):
        pr = str(meta.get("primary_role") or "").strip()
        if pr and not _primary_role_indicates_keeper(meta):
            return False
    return True


def wk_cap_exclusion_reason(p: Any) -> str:
    """Non-empty when broad keeper signals exist but the player is excluded from ``wk_max`` counting."""
    role_bucket = str(getattr(p, "role_bucket", "") or "")
    hd = getattr(p, "history_debug", None) or {}
    if not isinstance(hd, dict):
        hd = {}
    meta = _meta_dict(p)
    if not _broad_designated_keeper_candidate(p, meta, role_bucket):
        return ""
    if _counts_toward_wk_max_cap(p, meta, hd, role_bucket):
        return ""
    if _secondary_only_keeper_meta(meta):
        return "wk_cap_excluded_secondary_only_keeper_in_meta"
    if role_bucket == WK_BATTER and _wk_backup_wk_batter_excluded_from_wk_cap(p, meta, hd):
        return "wk_cap_excluded_tier_2_3_wk_batter_no_recent_keeper_signal"
    if role_bucket == BATTER and bool(getattr(p, "is_wicketkeeper", False)):
        pr = str(meta.get("primary_role") or "").strip()
        if pr and not _primary_role_indicates_keeper(meta):
            return "wk_cap_excluded_batter_bucket_without_primary_keeper_meta"
    return "wk_cap_excluded"


def _meta_role_indicates_all_rounder(m: dict[str, Any]) -> bool:
    pr = str(m.get("primary_role") or "").strip().lower()
    sr = str(m.get("secondary_role") or "").strip().lower()
    return pr == "all_rounder" or sr == "all_rounder"


def _text_has_any_token(raw: str, tokens: tuple[str, ...] | list[str] | set[str]) -> bool:
    s = str(raw or "").strip().lower()
    if not s:
        return False
    return any(t in s for t in tokens)


def _infer_spin_pace_evidence(p: Any) -> tuple[bool, bool]:
    """
    Determine spinner/pacer evidence from most-reliable to least-reliable signals.

    Important: do NOT default unknown bowling types to pacer.
    If evidence is missing, return (False, False) to represent 'unknown'.
    """
    hd = getattr(p, "history_debug", None) or {}
    role_band = str(hd.get("role_band") or "")
    if role_band == "middle_overs_spinner":
        return (True, False)

    m = _meta_dict(p)
    btb = str(m.get("bowling_type_bucket") or "").strip().lower()
    bs_raw = str(m.get("bowling_style_raw") or "").strip().lower()

    bt = str(getattr(p, "bowling_type", "") or "").strip().lower()

    # 1) Explicit metadata buckets.
    if btb in _SPIN_BUCKETS:
        return (True, False)
    if btb in _PACE_BUCKETS:
        return (False, True)

    # 2) Text parsing from bowling_type / bowling_style_raw.
    spin_like = _text_has_any_token(bt, _SPIN_TOKENS) or _text_has_any_token(bs_raw, _SPIN_TOKENS)
    pace_like = _text_has_any_token(bt, _PACE_TOKENS) or _text_has_any_token(bs_raw, _PACE_TOKENS)
    if spin_like and not pace_like:
        return (True, False)
    if pace_like and not spin_like:
        return (False, True)
    if spin_like and pace_like:
        # Ambiguous strings are treated as unknown rather than guessing.
        return (False, False)

    # 3) Very cautious fallback: phase profile (spinner-typical middle overs dominance).
    ph = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    try:
        total = float(ph.get("total_balls") or 0.0)
        pp = float(ph.get("powerplay_share") or 0.0)
        md = float(ph.get("middle_share") or 0.0)
        dt = float(ph.get("death_share") or 0.0)
    except (TypeError, ValueError):
        total, pp, md, dt = 0.0, 0.0, 0.0, 0.0

    rbucket = str(getattr(p, "role_bucket", "") or "")

    # Spinners: infer from phases when there is enough volume and a clear middle-overs dominance.
    #
    # - Bowlers: lower volume threshold is OK.
    # - All-rounders: require larger sample to avoid misclassifying part-time bowlers.
    is_all_rounder = rbucket == ALL_ROUNDER
    if rbucket == BOWLER and total >= 120:
        if md >= 0.56 and pp <= 0.26 and dt <= 0.24:
            return (True, False)
    if is_all_rounder and total >= 600:
        if md >= 0.56 and pp <= 0.26 and dt <= 0.24:
            return (True, False)

    # Pacers are more likely to have pronounced powerplay skew, or a death skew coupled with
    # non-trivial powerplay usage. Avoid inferring pacer from death skew alone (spinners can bowl late).
    if rbucket == BOWLER and total >= 240:
        if (pp >= 0.34 or (dt >= 0.45 and pp >= 0.18)) and md <= 0.58:
            return (False, True)

    return (False, False)


def _last_match_overs_bowled(p: Any) -> float:
    hd = getattr(p, "history_debug", None) or {}
    sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    lmd = sm.get("last_match_detail") if isinstance(sm.get("last_match_detail"), dict) else {}
    try:
        return float(lmd.get("last_match_overs_bowled") or 0.0)
    except (TypeError, ValueError):
        return 0.0


@lru_cache(maxsize=1)
def _global_bowling_usage_summary() -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for name, avg_balls, match_count in db.bowling_usage_raw():
        nk = learner.normalize_player_key(str(name or ""))
        if not nk:
            continue
        out[nk] = (float(avg_balls or 0.0), int(match_count or 0))
    return out


def _global_bowling_usage_evidence(p: Any) -> tuple[float, int]:
    usage = _global_bowling_usage_summary()
    raw_name = str(getattr(p, "name", "") or "")
    candidates = [
        learner.normalize_player_key(raw_name),
        learner.normalize_player_key(player_alias_resolve.canonicalize_via_alias_overrides(raw_name) or ""),
        learner.normalize_player_key(str(getattr(p, "canonical_player_key", "") or "")),
        learner.normalize_player_key(str(getattr(p, "player_key", "") or "")),
    ]
    for cand in candidates:
        if cand and cand in usage:
            return usage[cand]
    query = learner.normalize_player_key(raw_name)
    if not query:
        return (0.0, 0)
    q_parts = [x for x in query.split() if x]
    q_surname = q_parts[-1] if q_parts else ""
    q_first = q_parts[0] if q_parts else ""
    best_key = ""
    best_score = 0.0
    for key in usage.keys():
        k_parts = [x for x in key.split() if x]
        if not k_parts:
            continue
        k_surname = k_parts[-1]
        if q_surname and k_surname != q_surname:
            continue
        k_first = k_parts[0]
        ratio = SequenceMatcher(None, query, key).ratio()
        if q_first and k_first and q_first[0] == k_first[0]:
            ratio += 0.08
        if ratio > best_score:
            best_score = ratio
            best_key = key
    if best_key and best_score >= 0.84:
        return usage[best_key]
    return (0.0, 0)


def _all_rounder_has_meaningful_bowling_usage(p: Any) -> bool:
    """
    Real structural bowling option gate for all-rounders.

    A player can still have spin/pace type metadata without counting as a
    dependable XI bowling option. We only count all-rounders here when there is
    actual bowling-usage evidence.
    """
    hd = getattr(p, "history_debug", None) or {}
    ph = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    try:
        total_balls = float(ph.get("total_balls") or 0.0)
    except (TypeError, ValueError):
        total_balls = 0.0
    if total_balls >= 72.0:
        return True

    if _last_match_overs_bowled(p) >= 1.5:
        return True

    avg_balls, bowling_matches = _global_bowling_usage_evidence(p)
    if bowling_matches >= 8 and avg_balls >= 12.0:
        return True

    return False


def _all_rounder_has_any_bowling_usage_signal(p: Any) -> bool:
    hd = getattr(p, "history_debug", None) or {}
    ph = hd.get("bowler_phase_summary") if isinstance(hd.get("bowler_phase_summary"), dict) else {}
    try:
        total_balls = float(ph.get("total_balls") or 0.0)
    except (TypeError, ValueError):
        total_balls = 0.0
    if total_balls > 0:
        return True
    if _last_match_overs_bowled(p) > 0:
        return True
    avg_balls, bowling_matches = _global_bowling_usage_evidence(p)
    return bowling_matches > 0 or avg_balls > 0


@dataclass(frozen=True)
class PlayerRoleFlags:
    is_designated_keeper_candidate: bool
    is_wk_role_player: bool
    is_structural_all_rounder: bool
    is_bowling_option: bool
    is_pacer: bool
    is_spinner: bool
    is_top_order_batter: bool
    is_finisher: bool
    is_specialist_bowler: bool


def pace_only_override_from_conditions(conditions: Optional[dict[str, Any]]) -> bool:
    """
    Canonical pace-only override for the spinner-min constraint.

    This is intentionally narrow: the spinner rule is only relaxed when conditions
    strongly favor pace and are not spin-friendly.
    """
    c = conditions or {}
    try:
        pace_bias = float(c.get("pace_bias", 0.5) or 0.5)
        spin_friendly = float(c.get("spin_friendliness", 0.5) or 0.5)
    except (TypeError, ValueError):
        pace_bias, spin_friendly = 0.5, 0.5
    return pace_bias >= 0.72 and spin_friendly <= 0.42


def _spin_like(p: Any) -> bool:
    hd = getattr(p, "history_debug", None) or {}
    rb = str(hd.get("role_band") or "")
    if rb == "middle_overs_spinner":
        return True

    bt = str(getattr(p, "bowling_type", "") or "").lower()
    if any(x in bt for x in ("spin", "wrist", "offbreak", "legbreak", "orthodox", "slow")):
        return True

    meta = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    btb = str(meta.get("bowling_type_bucket") or "").strip().lower()
    return btb in _SPIN_BUCKETS


def classify_player(p: Any) -> PlayerRoleFlags:
    """
    Canonical derived-role classifier.

    All XI validation/repair and batting-order guardrails must use ONLY this classifier.
    The classifier is deterministic and side-effect free.
    """
    role_bucket = str(getattr(p, "role_bucket", "") or "")
    role = str(getattr(p, "role", "") or "")
    hd = getattr(p, "history_debug", None) or {}
    role_band = str(hd.get("role_band") or "")
    batting_band = str(hd.get("batting_band") or "")

    # Note: upstream uses canonical role_bucket labels from ipl_squad.py (e.g. "All-Rounder", "WK-Batter").
    meta = _meta_dict(p)
    effective_all_rounder = bool(role_bucket == ALL_ROUNDER or _meta_role_indicates_all_rounder(meta))
    # Broad: who may be chosen as designated keeper. Narrow: who consumes a wk_max hard-cap slot.
    is_designated_keeper_candidate = _broad_designated_keeper_candidate(p, meta, role_bucket)
    is_wk_role_player = _counts_toward_wk_max_cap(p, meta, hd, role_bucket)

    # Spin/pace type remains metadata/history-driven enrichment, but XI structure counts
    # only use it when the player also qualifies as a real bowling option.
    spin_like, pace_like = _infer_spin_pace_evidence(p)
    known_bowling_type = bool(spin_like or pace_like)
    is_bowling_option = (
        role_bucket == BOWLER
        or role == "bowl"
        or (
            role_bucket == ALL_ROUNDER
            and known_bowling_type
            and (
                _all_rounder_has_meaningful_bowling_usage(p)
                or not _all_rounder_has_any_bowling_usage_signal(p)
            )
        )
        or (
            effective_all_rounder
            and role_bucket != ALL_ROUNDER
            and known_bowling_type
            and (
                _all_rounder_has_meaningful_bowling_usage(p)
                or not _all_rounder_has_any_bowling_usage_signal(p)
            )
        )
    )
    is_spinner = bool(is_bowling_option and spin_like)
    is_pacer = bool(is_bowling_option and (pace_like and not spin_like))

    # Structural all-rounder means the player occupies an all-rounder slot in team structure.
    # Use the squad-truth role bucket first; role bands are secondary.
    is_structural_all_rounder = bool(role_bucket == ALL_ROUNDER)

    is_top_order_batter = (
        role_band in ("opener", "top_order")
        or batting_band in ("opener", "top_order")
        or bool(getattr(p, "is_opener_candidate", False))
    )
    is_finisher = role_band == "finisher" or batting_band == "finisher" or bool(getattr(p, "is_finisher_candidate", False))

    # Specialist bowlers are primarily bowlers; don't accidentally classify all-rounders as specialists
    # just because their derived role_band is bowling-heavy.
    is_specialist_bowler = role_bucket == BOWLER

    return PlayerRoleFlags(
        is_designated_keeper_candidate=is_designated_keeper_candidate,
        is_wk_role_player=is_wk_role_player,
        is_structural_all_rounder=is_structural_all_rounder,
        is_bowling_option=is_bowling_option,
        is_pacer=is_pacer,
        is_spinner=is_spinner,
        is_top_order_batter=is_top_order_batter,
        is_finisher=is_finisher,
        is_specialist_bowler=is_specialist_bowler,
    )


def role_counts(players: list[Any]) -> dict[str, int]:
    out = {
        "designated_keeper_candidates": 0,
        "wk_role_players": 0,
        "structural_all_rounders": 0,
        "bowling_options": 0,
        "pacers": 0,
        "spinners": 0,
        "top_order_batters": 0,
        "finishers": 0,
        "specialist_bowlers": 0,
        "proper_batters": 0,
    }
    for p in players:
        f = classify_player(p)
        out["designated_keeper_candidates"] += int(f.is_designated_keeper_candidate)
        out["wk_role_players"] += int(f.is_wk_role_player)
        out["structural_all_rounders"] += int(f.is_structural_all_rounder)
        out["bowling_options"] += int(f.is_bowling_option)
        out["pacers"] += int(f.is_pacer)
        out["spinners"] += int(f.is_spinner)
        out["top_order_batters"] += int(f.is_top_order_batter)
        out["finishers"] += int(f.is_finisher)
        out["specialist_bowlers"] += int(f.is_specialist_bowler)
        out["proper_batters"] += int(getattr(p, 'role_bucket', '') in (BATTER, WK_BATTER))
    return out


def wicketkeeper_xi_debug_rows(xi: list[Any]) -> list[dict[str, Any]]:
    """
    Per-player wicketkeeper signals for diagnosing wk_max / repair failures (temporary diagnostics).
    """
    rows: list[dict[str, Any]] = []
    for p in xi:
        hd = getattr(p, "history_debug", None) or {}
        if not isinstance(hd, dict):
            hd = {}
        smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        lmd = smd.get("last_match_detail") if isinstance(smd.get("last_match_detail"), dict) else {}
        meta = _meta_dict(p)
        flags = classify_player(p)
        rows.append(
            {
                "name": str(getattr(p, "name", "") or ""),
                "role": str(getattr(p, "role", "") or ""),
                "role_bucket": str(getattr(p, "role_bucket", "") or ""),
                "marquee_tier": str(hd.get("marquee_tier") or ""),
                "is_wk_role": bool(hd.get("is_wk_role", getattr(p, "is_wicketkeeper", False))),
                "is_keeper": bool(lmd.get("last_match_is_keeper") or lmd.get("is_keeper")),
                "designated_keeper": bool(hd.get("designated_keeper")),
                "squad_is_wicketkeeper_attr": bool(getattr(p, "is_wicketkeeper", False)),
                "role_bucket_is_wk_batter": str(getattr(p, "role_bucket", "") or "") == WK_BATTER,
                "meta_primary_role": str(meta.get("primary_role") or ""),
                "meta_secondary_role": str(meta.get("secondary_role") or ""),
                "meta_indicates_keeper": bool(_meta_role_indicates_keeper(meta)),
                "classify_is_designated_keeper_candidate": bool(flags.is_designated_keeper_candidate),
                "classify_counts_toward_wk_max_cap": bool(flags.is_wk_role_player),
                # Back-compat: ``is_wk_role_player`` on flags now means wk_max cap slot only.
                "classify_is_wk_role_player": bool(flags.is_wk_role_player),
                "wk_cap_exclusion_reason": wk_cap_exclusion_reason(p),
            }
        )
    return rows
