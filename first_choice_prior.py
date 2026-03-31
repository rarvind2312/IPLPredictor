"""
Heuristic **probable first-choice** prior for XI ordering (not a hardcoded XI).

Uses franchise-thin detection plus **global** IPL SQLite aggregates (all teams) only as a
fallback signal — never a substitute for franchise-specific ``history_xi_score`` terms.
"""

from __future__ import annotations

import math
from typing import Any

import config


def compute_probable_first_choice_prior(
    *,
    player: Any,
    franchise_distinct_matches: int,
    franchise_team_match_xi_rows: int,
    global_stats: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Return ``(prior 0..1, debug_dict)`` for debug / selection_score blending.

    ``global_stats`` should include keys from ``db.batch_global_*`` merges:
    ``distinct_matches``, ``distinct_teams``, ``tmx_rows``, ``global_batting_slot_ema``,
    ``global_batting_slot_samples``, and ``profile`` (per-player aggregate dict).
    """
    gx = dict(global_stats or {})
    dm = int(gx.get("distinct_matches") or 0)
    teams = int(gx.get("distinct_teams") or 0)
    tmx = int(gx.get("tmx_rows") or 0)
    slot_ema = float(gx.get("global_batting_slot_ema") or 0.0)
    slot_n = int(gx.get("global_batting_slot_samples") or 0)
    prof = gx.get("profile") if isinstance(gx.get("profile"), dict) else {}

    presence = dm >= 1
    sel_freq = min(1.0, math.log1p(dm) / math.log1p(48.0)) if presence else 0.0

    top_order = 0.0
    if slot_ema > 0 and slot_n >= 2:
        top_order = max(0.0, min(1.0, (8.5 - slot_ema) / 7.5))

    pxi = float(prof.get("max_xi_freq") or 0.0)
    pconf = float(prof.get("max_profile_confidence") or 0.0)
    pop = float(prof.get("max_opener_likelihood") or 0.0)
    pru = float(prof.get("max_recent_usage") or 0.0)
    role_strength = min(1.0, 0.42 * pxi + 0.28 * pconf + 0.18 * pop + 0.12 * pru)

    psamples = int(prof.get("max_samples") or 0)
    sample_bonus = min(1.0, math.log1p(psamples) / math.log1p(60.0)) if psamples else 0.0

    bucket = str(getattr(player, "role_bucket", "") or "")
    role_bonus = 0.0
    if bucket == "WK-Batter":
        role_bonus = 0.15
    elif bucket == "All-Rounder":
        role_bonus = 0.12
    elif bucket == "Batter":
        role_bonus = 0.07
    elif bucket == "Bowler":
        if float(getattr(player, "bowl_skill", 0)) >= 0.58:
            role_bonus = 0.09
    if bool(getattr(player, "is_opener_candidate", False)):
        role_bonus += 0.045

    comp = float(getattr(player, "composite", 0.5))
    cw = float(getattr(config, "FIRST_CHOICE_PRIOR_COMPOSITE_WEIGHT", 0.1))

    prior = (
        0.26 * sel_freq
        + 0.17 * role_strength
        + 0.14 * min(1.0, teams / 9.0)
        + 0.12 * top_order
        + 0.10 * sample_bonus
        + role_bonus
        + cw * comp
    )
    if not presence:
        prior *= 0.32
    prior = max(0.0, min(1.0, prior))

    dbg: dict[str, Any] = {
        "global_ipl_history_presence": presence,
        "global_selection_frequency": round(sel_freq, 4),
        "global_batting_position_pattern": round(slot_ema, 3) if slot_ema > 0 else None,
        "global_batting_slot_samples_global": slot_n,
        "global_role_strength": round(role_strength, 4),
        "distinct_matches_all_franchises": dm,
        "distinct_teams_all_franchises": teams,
        "team_match_xi_rows_all_franchises": tmx,
        "franchise_distinct_matches_input": int(franchise_distinct_matches),
        "franchise_team_match_xi_rows_input": int(franchise_team_match_xi_rows),
    }
    return prior, dbg


__all__ = ["compute_probable_first_choice_prior"]
