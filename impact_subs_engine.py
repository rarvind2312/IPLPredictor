"""
Impact substitution ranking: team derive patterns, game-state scenarios, tactical role cases.

Pre–Stage 3: avoids generic all-rounder spam by scoring specialist tactical fits first.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import learner
from ipl_squad import ALL_ROUNDER, BATTER, BOWLER, WK_BATTER

import db
import history_xi


def _marquee_tier_val(p: Any) -> int:
    """Align with predictor._tier_val: tier_1 > tier_2 > tier_3 > other (bench impact ordering)."""
    hd = getattr(p, "history_debug", None) or {}
    tier = str(hd.get("marquee_tier") or "").lower()
    if tier == "tier_1":
        return 3
    if tier == "tier_2":
        return 2
    if tier == "tier_3":
        return 1
    return 0


def _safe_json_obj(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_venue_pattern_context(
    team_key: str,
    venue_key: str,
    venue_key_candidates: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Resolve ``team_selection_patterns`` using the same multi-key strategy as Stage 3
    (primary venue key + derived pattern candidates).
    """
    fk = (team_key or "").strip()[:80]
    seen: set[str] = set()
    ordered: list[str] = []
    for vk in [venue_key, *(venue_key_candidates or [])]:
        s = str(vk or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    base: dict[str, Any] = {
        "weights": {},
        "pattern_src": "no_venue_key",
        "venue_pattern_keys_tried": list(ordered),
        "venue_pattern_key_matched": None,
        "team_selection_pattern_found": False,
        "team_selection_pattern_source": "",
        "pattern_miss_detail": "",
        "distinct_venue_keys_in_db_for_team": [],
    }
    if not fk:
        base["pattern_src"] = "empty_team_key"
        base["pattern_miss_detail"] = "empty canonical_team_key for impact pattern join"
        return base
    if not ordered:
        base["pattern_src"] = "no_venue_key"
        base["pattern_miss_detail"] = "no venue_key or venue_key_candidates provided"
        ex = db.team_selection_pattern_join_explain(fk, [])
        base["distinct_venue_keys_in_db_for_team"] = ex.get("distinct_venue_keys_in_db_for_team", [])
        return base
    row = db.fetch_team_selection_pattern(fk, ordered)
    if row:
        w = history_xi._parse_team_selection_xi_freq_weights(row.get("xi_frequency_json"))
        vk_m = str(row.get("venue_key") or "").strip()
        base.update(
            {
                "weights": w,
                "pattern_src": "team_selection_pattern_sqlite",
                "venue_pattern_key_matched": vk_m or None,
                "team_selection_pattern_found": True,
                "team_selection_pattern_source": "team_selection_patterns",
                "pattern_miss_detail": "",
            }
        )
        return base
    ex = db.team_selection_pattern_join_explain(fk, ordered)
    base.update(
        {
            "pattern_src": "team_selection_pattern_miss",
            "venue_pattern_key_matched": None,
            "team_selection_pattern_found": False,
            "team_selection_pattern_source": "",
            "pattern_miss_detail": str(ex.get("miss_reason") or "team_selection_pattern_miss"),
            "distinct_venue_keys_in_db_for_team": ex.get(
                "distinct_venue_keys_in_db_for_team", []
            ),
        }
    )
    return base


def _impact_rejected_because(
    idx: int,
    total: float,
    dbg: dict[str, Any],
    p: Any,
    scored: list[tuple[float, dict[str, Any], Any]],
    pattern_ctx: dict[str, Any],
) -> str:
    if idx < 5:
        return "selected_top_5"
    top5 = scored[:5]
    fifth_total = float(top5[4][0]) if len(top5) >= 5 else float(top5[-1][0] if top5 else 0.0)
    reasons: list[str] = []
    if total + 1e-9 < fifth_total:
        reasons.append(f"below_top5_total_gap_{round(fifth_total - total, 4)}")
    tact = float(dbg.get("tactical_scenario_block") or 0.0)
    max_tact = max(float(t[1].get("tactical_scenario_block") or 0.0) for t in top5) if top5 else 0.0
    if tact + 0.02 < max_tact:
        reasons.append("lower_tactical_fit_than_top5")
    scen_fit = float(dbg.get("impact_role_fit_score") or 0.0)
    max_sf = max(float(t[1].get("impact_role_fit_score") or 0.0) for t in top5) if top5 else 0.0
    if scen_fit + 0.04 < max_sf:
        reasons.append("low_scenario_utility_vs_top5")
    vw = float(dbg.get("venue_pattern_weight_hit") or 0.0)
    if not pattern_ctx.get("team_selection_pattern_found") and vw < 1e-6:
        reasons.append("weak_venue_team_pattern_support")
    elif vw < 0.02 and max(float(t[1].get("venue_pattern_weight_hit") or 0.0) for t in top5) >= 0.04:
        reasons.append("lower_venue_pattern_weight_than_top5")
    near = float(dbg.get("bench_near_xi_boost") or 0.0)
    max_near = max(float(t[1].get("bench_near_xi_boost") or 0.0) for t in top5) if top5 else 0.0
    if near < 1e-5 and max_near > 0.02:
        reasons.append("low_bench_near_xi_margin_vs_top5")
    rb = str(getattr(p, "role_bucket", "") or "")
    top5_buckets = [str(getattr(t[2], "role_bucket", "") or "") for t in top5]
    if rb and top5_buckets.count(rb) >= 3:
        reasons.append("role_duplication_vs_top5_mix")
    if bool(getattr(p, "is_overseas", False)):
        os_n = sum(1 for t in top5 if bool(getattr(t[2], "is_overseas", False)))
        if os_n >= 4:
            reasons.append("overseas_mix_friction_vs_top5")
    ar_pen = float(dbg.get("all_rounder_penalty_applied") or 0.0)
    if ar_pen > 0.06:
        reasons.append("all_rounder_tactical_penalty")
    if not reasons:
        reasons.append("below_top5_aggregate_ranking")
    return "; ".join(reasons)


def _team_sub_patterns(team_key: str) -> dict[str, Any]:
    row = db.fetch_team_derived_summary(team_key)
    if not row:
        return {
            "sample_matches": 0,
            "extra_batter_xi_tendency": 0.5,
            "extra_bowler_xi_tendency": 0.5,
            "batting_sub_after_pressure": 0.5,
            "bowling_sub_defending": 0.5,
        }
    chase = _safe_json_obj(row.get("chase_vs_defend_json"))
    bowl_pat = _safe_json_obj(row.get("bowling_composition_patterns_json"))
    n = int(row.get("sample_matches") or 0)
    bat_sub = float(chase.get("defend_batting_sub_signal", chase.get("chase_share", 0.5)) or 0.5)
    bowl_sub = float(chase.get("defend_bowl_sub_signal", 0.5) or 0.5)
    return {
        "sample_matches": n,
        "extra_batter_xi_tendency": float(bowl_pat.get("extra_batter_carry", 0.5) or 0.5),
        "extra_bowler_xi_tendency": float(bowl_pat.get("extra_bowler_carry", 0.5) or 0.5),
        "batting_sub_after_pressure": min(0.92, max(0.35, bat_sub)),
        "bowling_sub_defending": min(0.92, max(0.35, bowl_sub)),
        "chase_vs_defend_raw": chase,
        "bowling_composition_raw": bowl_pat,
    }


def _infer_scenarios(
    *,
    xi: list[Any],
    is_chasing: Optional[bool],
    conditions: dict[str, Any],
    team_bats_first: Optional[bool],
) -> dict[str, Any]:
    cond = conditions or {}
    bf = float(cond.get("batting_friendliness", 0.5))
    spin_f = float(cond.get("spin_friendliness", 0.5))
    pace_b = float(cond.get("pace_bias", 0.5))
    rain = float(cond.get("rain_disruption_risk", 0.0))
    dew = float(cond.get("dew_risk", 0.5))

    top4_sel = [
        float(getattr(p, "selection_score", 0.0)) for p in xi[: min(4, len(xi))]
    ]
    mean_top4 = sum(top4_sel) / max(1, len(top4_sel))
    # Early batting pressure: batting first but XI top looks weak vs field (tactical rescue window).
    top_order_collapse = bool(team_bats_first is True and mean_top4 < 0.46)

    stable_bat_first = bool(team_bats_first is True and not top_order_collapse)

    defending_small_total = bool(is_chasing is False and team_bats_first is False and bf < 0.52)

    chase_strong_start = bool(is_chasing is True and dew >= 0.55 and mean_top4 >= 0.48)

    spin_venue_bias = bool(spin_f >= 0.56 and pace_b <= 0.52)
    pace_venue_bias = bool(pace_b >= 0.56 and spin_f <= 0.52)

    return {
        "top_order_collapse": top_order_collapse,
        "stable_bat_first": stable_bat_first,
        "defending_small_total": defending_small_total,
        "chase_strong_start": chase_strong_start,
        "spin_venue_bias": spin_venue_bias,
        "pace_venue_bias": pace_venue_bias,
        "mean_top4_selection_score": round(mean_top4, 4),
        "batting_friendliness": bf,
        "rain_disruption_risk": rain,
    }


def _classify_impact_role_case(
    p: Any,
    scenarios: dict[str, Any],
    patterns: dict[str, Any],
) -> tuple[str, str]:
    rb = str(getattr(p, "role_bucket", "") or "")
    bat = float(getattr(p, "bat_skill", 0.5))
    bowl = float(getattr(p, "bowl_skill", 0.5))
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    spin_like = "spin" in btype or "slow" in btype or "orthodox" in btype

    if scenarios.get("top_order_collapse") and rb in (BATTER, WK_BATTER) and bat >= 0.48:
        return (
            "batting_rescue",
            "Top-order pressure while batting first; prefer specialist batter/WK stabiliser "
            "(tactical rescue, e.g. early-collapse batting impact).",
        )
    if rb in (BATTER, WK_BATTER) and (
        bool(getattr(p, "is_finisher_candidate", False)) or (rb == BATTER and bat >= 0.57)
    ):
        if scenarios.get("chase_strong_start") or float(
            patterns.get("batting_sub_after_pressure", 0) or 0
        ) > 0.55:
            return (
                "finisher_boost",
                "Finisher / acceleration profile when chasing or team often subs batting under pressure.",
            )
    if scenarios.get("spin_venue_bias") and rb == BOWLER and spin_like:
        return ("spin_defend", "Spin-friendly venue; defensive spin bowling cover.")
    if scenarios.get("pace_venue_bias") and rb == BOWLER and not spin_like:
        return ("pace_defend", "Pace-friendly venue; seam bowling cover while defending.")
    if scenarios.get("defending_small_total") and rb == BOWLER and bowl >= 0.52:
        return ("bowling_cover", "Defending a moderate total; extra bowling option for control/variants.")
    if rb == BOWLER and scenarios.get("stable_bat_first") and bowl >= 0.55:
        return (
            "bowling_cover",
            "Batting first with stable top order; middle/death insurance bowler.",
        )
    if rb == ALL_ROUNDER and bat >= 0.5 and bowl >= 0.48:
        return (
            "flexibility_cover",
            "Bench all-rounder — lower priority than a clear specialist tactical sub.",
        )
    if rb == ALL_ROUNDER:
        return ("flexibility_cover", "Utility bench cover.")
    if rb == BOWLER:
        return ("bowling_cover", "General bowling depth.")
    return ("batting_rescue", "Batting bench depth / partial rescue.")


def _scenario_fit_scores(
    role_case: str,
    scenarios: dict[str, Any],
    patterns: dict[str, Any],
) -> dict[str, float]:
    def clip(x: float) -> float:
        return max(0.0, min(1.0, x))

    t_oc = 0.85 if scenarios.get("top_order_collapse") else 0.22
    st_bf = 0.8 if scenarios.get("stable_bat_first") else 0.3
    d_small = 0.82 if scenarios.get("defending_small_total") else 0.25
    ch_str = 0.78 if scenarios.get("chase_strong_start") else 0.3

    pmap = float(patterns.get("batting_sub_after_pressure", 0.5) or 0.5)
    if role_case == "batting_rescue":
        t_oc = clip(t_oc + 0.1 * pmap)
    elif role_case == "finisher_boost":
        ch_str = clip(ch_str + 0.12 * pmap)
    elif role_case in ("pace_defend", "spin_defend", "bowling_cover"):
        d_small = clip(
            d_small + 0.1 * float(patterns.get("bowling_sub_defending", 0.5) or 0.5)
        )

    return {
        "top_order_collapse_fit": round(t_oc, 4),
        "stable_bat_first_fit": round(st_bf, 4),
        "defending_small_total_fit": round(d_small, 4),
        "chase_strong_start_fit": round(ch_str, 4),
    }


def _likelihood_flags(
    role_case: str,
    scenarios: dict[str, Any],
    p: Any,
) -> dict[str, float]:
    rb = str(getattr(p, "role_bucket", "") or "")
    bat = float(getattr(p, "bat_skill", 0.5))
    bowl = float(getattr(p, "bowl_skill", 0.5))

    lf_bat_first = 0.45
    lf_bowl_first = 0.45
    lf_collapse = 0.35
    lf_def_small = 0.35

    if role_case == "batting_rescue":
        lf_bat_first = 0.72
        lf_collapse = 0.88
        lf_bowl_first = 0.38
    elif role_case == "finisher_boost":
        lf_bat_first = 0.55
        lf_bowl_first = 0.62
        lf_collapse = 0.42
        lf_def_small = 0.5
    elif role_case in ("pace_defend", "spin_defend", "bowling_cover"):
        lf_bowl_first = 0.78
        lf_def_small = 0.74
        lf_bat_first = 0.4
    elif role_case == "flexibility_cover":
        lf_bat_first = 0.5
        lf_bowl_first = 0.5

    if scenarios.get("top_order_collapse"):
        lf_collapse = min(0.95, lf_collapse + 0.05)
    if scenarios.get("defending_small_total"):
        lf_def_small = min(0.92, lf_def_small + 0.06)
    if rb == BOWLER:
        lf_bowl_first = min(0.94, lf_bowl_first + 0.04 * bowl)
    if rb in (BATTER, WK_BATTER):
        lf_bat_first = min(0.92, lf_bat_first + 0.04 * bat)

    return {
        "likely_used_if_batting_first": round(lf_bat_first, 4),
        "likely_used_if_bowling_first": round(lf_bowl_first, 4),
        "likely_used_if_top_order_collapse": round(lf_collapse, 4),
        "likely_used_if_defending_small_total": round(lf_def_small, 4),
    }


def rank_impact_sub_candidates(
    squad: list[Any],
    xi: list[Any],
    *,
    team_display_name: str,
    canonical_team_key: str,
    venue_key: str,
    venue_key_candidates: Optional[list[str]] = None,
    is_chasing: Optional[bool],
    conditions: dict[str, Any],
    team_bats_first: Optional[bool],
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Rank bench players for likely impact sub use.

    Uses SQLite ``team_selection_patterns`` when any candidate venue key matches derive keys, plus
    team_derived_summary patterns. Scoring splits partially by batting-first vs bowling-first.

    Returns ``(top5_players, top5_debug_rows, all_bench_debug_rows_sorted)``.
    """
    cond = conditions or {}
    xi_names = {getattr(p, "name", "") for p in xi}
    rest = [p for p in squad if getattr(p, "name", "") not in xi_names]

    patterns = _team_sub_patterns(canonical_team_key)
    pattern_ctx = _resolve_venue_pattern_context(
        canonical_team_key, venue_key, venue_key_candidates
    )
    venue_weights = pattern_ctx["weights"]
    pattern_src = str(pattern_ctx["pattern_src"])
    scenarios = _infer_scenarios(
        xi=xi,
        is_chasing=is_chasing,
        conditions=cond,
        team_bats_first=team_bats_first,
    )

    bf = float(cond.get("batting_friendliness", 0.5))
    spin_f = float(cond.get("spin_friendliness", 0.5))
    pace_b = float(cond.get("pace_bias", 0.5))
    rain = float(cond.get("rain_disruption_risk", 0.0))
    dew = float(cond.get("dew_risk", 0.5))

    scored: list[tuple[float, dict[str, Any], Any]] = []

    for p in rest:
        hd = getattr(p, "history_debug", None) or {}
        try:
            bench_margin = float(hd.get("bench_near_xi_margin") or -99.0)
        except (TypeError, ValueError):
            bench_margin = -99.0
        role_case, reason_summary = _classify_impact_role_case(p, scenarios, patterns)
        scen_scores = _scenario_fit_scores(role_case, scenarios, patterns)
        like = _likelihood_flags(role_case, scenarios, p)
        pk_imp = str(
            getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))
        ).strip()[:80]
        venue_w = float(venue_weights.get(pk_imp, 0.0)) if venue_weights and pk_imp else 0.0

        sel = float(getattr(p, "selection_score", 0.0))
        hx = float(getattr(p, "history_xi_score", 0.0))
        h2h = float(hd.get("h2h_weighted_xi_rate") or 0.0)
        vnr = float(hd.get("venue_xi_rate") or 0.0)
        r5 = float(hd.get("recent5_xi_rate") or 0.0)
        cd = float(hd.get("chase_defend_xi_rate") or 0.0)

        phh = hd.get("phase_bowl_rates_h2h_matches") or {}
        ph_mix = hd.get("phase_bowl_blend_with_h2h")
        if ph_mix is None:
            ph_mix = hd.get("phase_bowl_blend_general")
        ph_mix = float(ph_mix or 0.0)
        pp = float(phh.get("powerplay") or 0.0) if isinstance(phh, dict) else 0.0
        death = float(phh.get("death") or 0.0) if isinstance(phh, dict) else 0.0
        mid = float(phh.get("middle") or 0.0) if isinstance(phh, dict) else 0.0

        rb = str(getattr(p, "role_bucket", "") or "")
        btype = str(getattr(p, "bowling_type", "") or "").lower()
        spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
        seam_like = (not spin_like) and rb == BOWLER
        prof_d = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        try:
            finisher_l = float(prof_d.get("finisher_likelihood") or 0.0) if prof_d else 0.0
        except (TypeError, ValueError):
            finisher_l = 0.0
        try:
            pp_l = float(prof_d.get("powerplay_bowler_likelihood") or 0.0) if prof_d else 0.0
        except (TypeError, ValueError):
            pp_l = 0.0
        try:
            death_l = float(prof_d.get("death_bowler_likelihood") or 0.0) if prof_d else 0.0
        except (TypeError, ValueError):
            death_l = 0.0
        bat_skill = float(getattr(p, "bat_skill", 0.5) or 0.5)

        hist_line = (
            0.22 * sel
            + 0.16 * hx
            + 0.13 * h2h
            + 0.08 * vnr
            + 0.07 * r5
            + 0.04
            * (
                cd
                if is_chasing is True
                else ((1.0 - cd) if is_chasing is False else 0.5)
            )
        )
        near_xi_boost = 0.0
        if bench_margin > -0.14:
            near_xi_boost = 0.052 * max(0.0, min(1.0, 1.0 + bench_margin * 10.0))
        hist_line += near_xi_boost
        bowl_line = 0.1 * ph_mix + 0.06 * (pp + death + mid) * float(getattr(p, "bowl_skill", 0.0))

        tactical = 0.0
        bat_first_layer = 1.0 if team_bats_first is True else 0.55 if team_bats_first is None else 0.35
        bowl_first_layer = 1.0 if team_bats_first is False else 0.55 if team_bats_first is None else 0.35
        if role_case == "batting_rescue":
            tactical += 0.14 * scen_scores["top_order_collapse_fit"] * bat_first_layer
        elif role_case == "finisher_boost":
            tactical += 0.11 * scen_scores["chase_strong_start_fit"] * bowl_first_layer
        elif role_case in ("spin_defend", "pace_defend"):
            tactical += 0.12 * scen_scores["defending_small_total_fit"] * bowl_first_layer
            tactical += 0.06 * (spin_f if role_case == "spin_defend" else pace_b) * bowl_first_layer
        elif role_case == "bowling_cover":
            tactical += 0.1 * scen_scores["defending_small_total_fit"] * bowl_first_layer
            tactical += 0.05 * scen_scores["stable_bat_first_fit"] * bat_first_layer
        else:
            tactical += 0.04

        team_adj = 0.0
        if role_case in ("batting_rescue", "finisher_boost"):
            team_adj += (
                0.07
                * (float(patterns.get("batting_sub_after_pressure", 0.5)) - 0.5)
                * (bat_first_layer if role_case == "batting_rescue" else bowl_first_layer)
            )
        if role_case in ("bowling_cover", "pace_defend", "spin_defend"):
            team_adj += (
                0.08
                * (float(patterns.get("bowling_sub_defending", 0.5)) - 0.5)
                * bowl_first_layer
            )

        cond_style = 0.05 * float(getattr(p, "bat_skill", 0.5)) * bf
        cond_style += 0.045 * float(getattr(p, "bowl_skill", 0.5)) * (1.0 - bf)
        cond_style *= max(0.38, 1.0 - 0.42 * rain)

        ar_penalty = 0.0
        if rb == ALL_ROUNDER:
            specialist_signal = 1.0 if role_case != "flexibility_cover" else 0.35
            ar_penalty = 0.14 * (1.0 - specialist_signal)
            if role_case == "flexibility_cover":
                ar_penalty += 0.08

        toss_chase = 0.0
        if team_bats_first is True:
            toss_chase += 0.055 * death * float(getattr(p, "bowl_skill", 0.0)) * (1.0 - 0.32 * rain)
            toss_chase += 0.035 * r5 * float(getattr(p, "bat_skill", 0.0))
        elif team_bats_first is False:
            toss_chase += 0.055 * r5 * float(getattr(p, "bat_skill", 0.0))
            toss_chase += 0.045 * float(getattr(p, "bat_skill", 0.5)) * (0.52 + 0.48 * dew)
        else:
            toss_chase += 0.03 * (pp + death) * float(getattr(p, "bowl_skill", 0.0))

        if is_chasing is True:
            toss_chase += 0.04 * float(getattr(p, "bat_skill", 0.5)) * bf
        elif is_chasing is False:
            toss_chase += 0.038 * pp * float(getattr(p, "bowl_skill", 0.0))

        venue_bonus = 0.048 * venue_w * (0.6 + 0.4 * bf) if venue_w > 0 else 0.0
        extra_seam = 0.0
        if scenarios.get("pace_venue_bias") and seam_like and team_bats_first is False:
            extra_seam = 0.034 * pace_b * float(getattr(p, "bowl_skill", 0.0))
        extra_spin = 0.0
        if scenarios.get("spin_venue_bias") and spin_like:
            extra_spin = 0.032 * spin_f * float(getattr(p, "bowl_skill", 0.0))
        pp_reserve = 0.0
        if team_bats_first is False and max(pp, pp_l) >= 0.52 and seam_like:
            pp_reserve = 0.028 * max(pp, pp_l) * float(getattr(p, "bowl_skill", 0.0)) * pace_b
        death_reserve = 0.0
        if (is_chasing is False or team_bats_first is True) and max(death, death_l) >= 0.5:
            death_reserve = 0.026 * max(death, death_l) * float(getattr(p, "bowl_skill", 0.0))
        fin_res = 0.0
        if team_bats_first is False and finisher_l >= 0.55 and rb in (BATTER, WK_BATTER, ALL_ROUNDER):
            fin_res = 0.03 * finisher_l * bat_skill

        role_fit = (
            0.28 * scen_scores.get("top_order_collapse_fit", 0.0)
            + 0.26 * scen_scores.get("chase_strong_start_fit", 0.0)
            + 0.24 * scen_scores.get("defending_small_total_fit", 0.0)
            + 0.22 * scen_scores.get("stable_bat_first_fit", 0.0)
        )
        scen_reason = reason_summary
        if team_bats_first is True:
            scen_reason += " [bat-first emphasis: collapse rescue / set-total depth]."
        elif team_bats_first is False:
            scen_reason += " [bowl-first emphasis: chase spine / PP–death bowling]."
        else:
            scen_reason += " [toss open — blended bat-first and bowl-first impact weights]."

        tier = str(hd.get("marquee_tier") or "").lower()
        tier_impact = 0.0
        if tier == "tier_1": tier_impact = 0.45
        elif tier == "tier_2": tier_impact = 0.22
        elif tier == "tier_3": tier_impact = -0.25
        else: tier_impact = -0.35

        total = (
            hist_line
            + bowl_line
            + tactical
            + team_adj
            + cond_style
            + toss_chase
            + venue_bonus
            + extra_seam
            + extra_spin
            + pp_reserve
            + death_reserve
            + fin_res
            + tier_impact
            - ar_penalty
        )

        xi_proj_parts: list[str] = []
        if bench_margin < -0.02:
            xi_proj_parts.append(
                f"starting_XI_margin_vs_weakest_pick={round(bench_margin, 4)}_so_not_auto_XI"
            )
        if role_case in ("pace_defend", "spin_defend", "bowling_cover") and bf >= 0.56 and rb == BOWLER:
            xi_proj_parts.append("high_scoring_venue_marginal_bowler_as_flexible_impact_cover")
        if role_case == "finisher_boost" and bf >= 0.57:
            xi_proj_parts.append("finisher_accelerator_slot_suits_impact_in_chase_scenarios")
        if seam_like and pace_b >= 0.56 and team_bats_first is False:
            xi_proj_parts.append("extra_pace_option_for_defence_PP_death")
        if spin_like and spin_f >= 0.56:
            xi_proj_parts.append("extra_spin_option_when_venue_spin_bias_high")
        if role_case == "batting_rescue":
            xi_proj_parts.append("batting_rescue_depth_if_top_order_pressure")
        impact_xi_projection_explanation = (
            "; ".join(xi_proj_parts)
            if xi_proj_parts
            else "impact_rank_uses_tactical_fit_history_venue_patterns_not_plain_XI_selection_order"
        )

        scenario_public = {
            k: v
            for k, v in scenarios.items()
            if isinstance(v, (bool, int, float, str))
        }

        dbg = {
            "impact_total_score": round(total, 5),
            "impact_role_case": role_case,
            "impact_reason_summary": reason_summary,
            "impact_pattern_source": pattern_src,
            "impact_scenario_reason": scen_reason,
            "impact_role_fit_score": round(role_fit, 4),
            "impact_team_venue_bonus": round(venue_bonus, 5),
            "impact_team_pattern_adjustment": round(team_adj, 5),
            "impact_venue_key_used": (venue_key or "")[:80],
            "venue_pattern_keys_tried": list(pattern_ctx.get("venue_pattern_keys_tried") or []),
            "venue_pattern_key_matched": pattern_ctx.get("venue_pattern_key_matched"),
            "team_selection_pattern_found": bool(
                pattern_ctx.get("team_selection_pattern_found")
            ),
            "team_selection_pattern_source": pattern_ctx.get("team_selection_pattern_source")
            or "",
            "team_selection_pattern_miss_detail": pattern_ctx.get("pattern_miss_detail") or "",
            "distinct_venue_keys_in_db_for_team": pattern_ctx.get(
                "distinct_venue_keys_in_db_for_team", []
            ),
            "impact_team_context": (team_display_name or "")[:120],
            "impact_rejected_because": "",
            "scenario_fit_scores": scen_scores,
            "game_state_scenarios": scenario_public,
            "team_pattern_snapshot": {
                "sample_matches": patterns.get("sample_matches"),
                "batting_sub_after_pressure": patterns.get("batting_sub_after_pressure"),
                "bowling_sub_defending": patterns.get("bowling_sub_defending"),
            },
            "history_block": round(hist_line, 5),
            "bench_near_xi_boost": round(near_xi_boost, 5),
            "bowling_phase_block": round(bowl_line, 5),
            "tactical_scenario_block": round(tactical, 5),
            "team_pattern_block": round(team_adj, 5),
            "venue_weather_style_block": round(cond_style, 5),
            "toss_chase_scenario_block": round(toss_chase, 5),
            "venue_pattern_weight_hit": round(venue_w, 5),
            "tactical_extras": {
                "extra_seam_venue": round(extra_seam, 5),
                "extra_spin_venue": round(extra_spin, 5),
                "pp_bowler_reserve": round(pp_reserve, 5),
                "death_bowler_reserve": round(death_reserve, 5),
                "finisher_reserve": round(fin_res, 5),
            },
            "all_rounder_penalty_applied": round(ar_penalty, 5),
            "impact_xi_projection_explanation": impact_xi_projection_explanation,
            **like,
        }
        scored.append((total, dbg, p))

    scored.sort(
        key=lambda x: (
            _marquee_tier_val(x[2]),
            x[0],
            getattr(x[2], "composite", 0),
            getattr(x[2], "name", ""),
        ),
        reverse=True,
    )
    dbg_all: list[dict[str, Any]] = []
    for i, (total, d, p) in enumerate(scored):
        d = dict(d)
        d["impact_sub_rank"] = i + 1
        d["impact_rejected_because"] = _impact_rejected_because(
            i, float(total), d, p, scored, pattern_ctx
        )
        dbg_all.append(
            {
                "name": getattr(p, "name", ""),
                "player_key": getattr(p, "player_key", None)
                or learner.normalize_player_key(getattr(p, "name", "")),
                **d,
            }
        )
    top = scored[:5]
    dbg_rows = dbg_all[:5]
    return [p for _, _, p in top], dbg_rows, dbg_all


__all__ = ["rank_impact_sub_candidates"]
