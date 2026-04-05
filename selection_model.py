"""
Modular Stage-3 selection scoring for IPL XI / impact ranking.

Base score (each term in [0, 1]):
  40% recent_form_score   — global T20 recent form from SQLite ``player_recent_form_cache`` only;
                            if a player has no cached T20 window, IPL derive usage/composite fallback.
  30% ipl_history_role_score — normalized franchise history + Stage-2 derive role signals
  20% team_balance_fit_score — two-pass gap fill vs provisional top-11
  10% venue_experience_score — venue XI rate, derive venue fit, team pattern weights

Then additive tactical modifiers (capped): pitch, weather, toss, opponent, squad_need.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Optional

import audit_profile
import config
import db
import ipl_squad
import learner
import matchup_features

_perf_logger = logging.getLogger("ipl_predictor.perf")

BATTER = ipl_squad.BATTER
WK_BATTER = ipl_squad.WK_BATTER
ALL_ROUNDER = ipl_squad.ALL_ROUNDER
BOWLER = ipl_squad.BOWLER


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s or not str(s).strip():
        return None
    raw = str(s).strip()[:10]
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _recent_form_role_weights(p: Any) -> tuple[float, float]:
    rb = str(getattr(p, "role_bucket", "") or "")
    bat_skill = float(getattr(p, "bat_skill", 0.5) or 0.5)
    bowl_skill = float(getattr(p, "bowl_skill", 0.5) or 0.5)
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
    if rb == BOWLER and not spin_like:
        return 0.12, 0.88
    if rb == BOWLER and spin_like:
        return 0.18, 0.82
    if rb in (BATTER, WK_BATTER):
        return 0.88, 0.12
    if rb == ALL_ROUNDER:
        s = bat_skill + bowl_skill + 1e-6
        w_bat = 0.35 + 0.45 * (bat_skill / s)
        return w_bat, 1.0 - w_bat
    return 0.55, 0.45


def _recent_form_score_player(
    p: Any,
    cache_row: Optional[dict[str, Any]],
    derive_snap: Optional[dict[str, Any]],
    *,
    reference_date: Optional[date] = None,
) -> tuple[float, dict[str, Any]]:
    n_m = int(getattr(config, "SELECTION_RECENT_FORM_LAST_N_MATCHES", 5))
    n_months = int(getattr(config, "SELECTION_RECENT_FORM_MONTHS", 5))
    w_bat, w_bowl = _recent_form_role_weights(p)
    ds = derive_snap or {}
    ru = float(ds.get("recent_usage_score") or 0.0)
    comp = float(getattr(p, "composite", 0.5) or 0.5)

    n_cached = int((cache_row or {}).get("t20_matches_in_window") or 0)
    if cache_row is not None and n_cached > 0:
        bat = _clamp01(float(cache_row.get("batting_recent_form") or 0.48))
        bowl = _clamp01(float(cache_row.get("bowling_recent_form") or 0.48))
        combined = _clamp01(w_bat * bat + w_bowl * bowl)
        comps_raw = cache_row.get("competitions_json") or "[]"
        try:
            comps_list = json.loads(comps_raw) if isinstance(comps_raw, str) else comps_raw
        except Exception:
            comps_list = []
        dbg_json = cache_row.get("debug_json")
        try:
            parsed_dbg = json.loads(dbg_json) if isinstance(dbg_json, str) else dbg_json
        except Exception:
            parsed_dbg = {}
        dbg = {
            "recent_form_source": "player_recent_form_cache",
            "t20_row_count_used": n_cached,
            "t20_union_last_n_matches": n_m,
            "t20_union_months": n_months,
            "batting_recent_form": round(bat, 5),
            "bowling_recent_form": round(bowl, 5),
            "recent_form_role_weights": {"batting": round(w_bat, 4), "bowling": round(w_bowl, 4)},
            "last_t20_match_date": (cache_row.get("last_t20_match_date") or "") or "",
            "competitions_used": comps_list if isinstance(comps_list, list) else [],
            "matches_last_30d": int(cache_row.get("matches_last_30d") or 0),
            "matches_last_60d": int(cache_row.get("matches_last_60d") or 0),
            "matches_last_150d": int(cache_row.get("matches_last_150d") or 0),
            "sample_confidence": round(float(cache_row.get("sample_confidence") or 0.0), 5),
            "cache_debug": parsed_dbg if isinstance(parsed_dbg, dict) else {},
        }
        return combined, dbg

    neutral = _clamp01(w_bat * 0.48 + w_bowl * 0.48)
    combined = _clamp01(0.62 * ru + 0.28 * comp + 0.1 * neutral)
    dbg = {
        "recent_form_source": "ipl_derive_fallback_no_t20_cache",
        "t20_row_count_used": 0,
        "t20_union_last_n_matches": n_m,
        "t20_union_months": n_months,
        "batting_recent_form": round(0.48, 5),
        "bowling_recent_form": round(0.48, 5),
        "recent_form_role_weights": {"batting": round(w_bat, 4), "bowling": round(w_bowl, 4)},
        "reference_date_hint": (reference_date.isoformat() if reference_date else ""),
    }
    return combined, dbg


def _ipl_history_role_score(
    hn_mixed: float,
    derive_snap: Optional[dict[str, Any]],
    hd: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Franchise IPL history + derive role stability (0..1)."""
    ds = derive_snap or {}
    xi_freq = float(ds.get("xi_selection_frequency") or 0.0)
    role_stab = float(ds.get("role_stability_score") or 0.0)
    ol = float(ds.get("opener_likelihood") or 0.0)
    fl = float(ds.get("finisher_likelihood") or 0.0)
    pp = float(ds.get("powerplay_bowler_likelihood") or 0.0)
    dt = float(ds.get("death_bowler_likelihood") or 0.0)
    role_shape = _clamp01(0.22 * ol + 0.22 * fl + 0.28 * pp + 0.28 * dt)
    hist = _clamp01(float(hn_mixed))
    score = _clamp01(0.55 * hist + 0.25 * xi_freq + 0.12 * role_stab + 0.08 * role_shape)
    prior_fc = float(hd.get("probable_first_choice_prior") or 0.0)
    used_g = bool(hd.get("used_global_fallback_prior"))
    if prior_fc > 0.22:
        score = _clamp01(score + 0.06 * prior_fc * (1.15 if used_g else 0.85))
    dbg = {
        "history_normalized_component": round(hist, 5),
        "xi_selection_frequency": round(xi_freq, 5),
        "role_stability_score": round(role_stab, 5),
        "role_shape_score": round(role_shape, 5),
    }
    return score, dbg


def _venue_experience_score(
    p: Any,
    hd: dict[str, Any],
    derive_snap: Optional[dict[str, Any]],
    venue_weights: dict[str, float],
    pk: str,
    conditions: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    ds = derive_snap or {}
    vnr = float(hd.get("venue_xi_rate") or 0.0)
    vfit = float(ds.get("venue_fit_score") or 0.5)
    pat = float(venue_weights.get(pk, 0.0)) if pk and venue_weights else 0.0
    bf = float(conditions.get("batting_friendliness", 0.5))
    spin_f = float(conditions.get("spin_friendliness", 0.5))
    pace_b = float(conditions.get("pace_bias", 0.5))
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
    seam_like = (getattr(p, "role_bucket", "") == BOWLER) and not spin_like
    align = 0.0
    if spin_like:
        align += 0.06 * (spin_f - 0.5)
    if seam_like:
        align += 0.06 * (pace_b - 0.5)
    if getattr(p, "role_bucket", "") in (BATTER, WK_BATTER, ALL_ROUNDER):
        align += 0.05 * (bf - 0.5) * float(getattr(p, "bat_skill", 0.5) or 0.5)
    score = _clamp01(
        0.38 * _clamp01(vnr)
        + 0.28 * _clamp01(vfit)
        + 0.22 * _clamp01(pat * 2.5)
        + 0.12 * (0.5 + align)
    )
    dbg = {
        "venue_xi_rate": round(vnr, 5),
        "derive_venue_fit": round(vfit, 5),
        "team_pattern_weight": round(pat, 5),
        "venue_type_alignment": round(align, 5),
    }
    return score, dbg


def _gap_snapshot(top11: list[Any]) -> dict[str, Any]:
    openers = 0
    finishers = 0
    wk = 0
    seam = 0
    spin = 0
    pp_b = 0
    death_b = 0
    bat_depth = 0
    overseas = 0
    for p in top11:
        hd = getattr(p, "history_debug", None) or {}
        ds = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
        ol = float(ds.get("opener_likelihood") or 0.0)
        fl = float(ds.get("finisher_likelihood") or 0.0)
        ppl = float(ds.get("powerplay_bowler_likelihood") or 0.0)
        dth = float(ds.get("death_bowler_likelihood") or 0.0)
        if ol >= 0.54 or bool(getattr(p, "is_opener_candidate", False)):
            openers += 1
        if fl >= 0.52 or bool(getattr(p, "is_finisher_candidate", False)):
            finishers += 1
        if getattr(p, "role_bucket", "") == WK_BATTER or bool(getattr(p, "is_wicketkeeper", False)):
            wk += 1
        rb = str(getattr(p, "role_bucket", "") or "")
        btype = str(getattr(p, "bowling_type", "") or "").lower()
        spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
        if rb == BOWLER:
            if spin_like:
                spin += 1
            else:
                seam += 1
        if ppl >= 0.48:
            pp_b += 1
        if dth >= 0.48:
            death_b += 1
        if rb in (BATTER, WK_BATTER, ALL_ROUNDER):
            bat_depth += 1
        if bool(getattr(p, "is_overseas", False)):
            overseas += 1
    return {
        "openers": openers,
        "finishers": finishers,
        "wk": wk,
        "seam": seam,
        "spin": spin,
        "pp_bowlers": pp_b,
        "death_bowlers": death_b,
        "batting_depth": bat_depth,
        "overseas": overseas,
    }


def _team_balance_fit_scores(players: list[Any], prelim_for_tb: dict[str, float]) -> dict[str, float]:
    ordered = sorted(players, key=lambda p: prelim_for_tb.get(p.name, 0.0), reverse=True)
    top11 = ordered[:11]
    gaps = _gap_snapshot(top11)
    need_open = max(0, 2 - int(gaps["openers"]))
    need_fin = max(0, 1 - int(gaps["finishers"]))
    need_wk = max(0, 1 - int(gaps["wk"]))
    need_seam = max(0, 3 - int(gaps["seam"]))
    need_spin = max(0, 2 - int(gaps["spin"]))
    need_pp = max(0, 2 - int(gaps["pp_bowlers"]))
    need_death = max(0, 2 - int(gaps["death_bowlers"]))
    need_bat = max(0, 5 - int(gaps["batting_depth"]))
    out: dict[str, float] = {}
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        ds = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else {}
        ol = float(ds.get("opener_likelihood") or 0.0)
        fl = float(ds.get("finisher_likelihood") or 0.0)
        ppl = float(ds.get("powerplay_bowler_likelihood") or 0.0)
        dth = float(ds.get("death_bowler_likelihood") or 0.0)
        s = 0.46
        if need_open > 0 and (ol >= 0.5 or bool(getattr(p, "is_opener_candidate", False))):
            s += 0.11 * min(1.0, ol + 0.15)
        if need_fin > 0 and (fl >= 0.5 or bool(getattr(p, "is_finisher_candidate", False))):
            s += 0.09 * min(1.0, fl + 0.1)
        if need_wk > 0 and (
            getattr(p, "role_bucket", "") == WK_BATTER or bool(getattr(p, "is_wicketkeeper", False))
        ):
            s += 0.16
        btype = str(getattr(p, "bowling_type", "") or "").lower()
        spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
        seam_like = getattr(p, "role_bucket", "") == BOWLER and not spin_like
        if need_seam > 0 and seam_like:
            s += 0.07 * float(getattr(p, "bowl_skill", 0.5) or 0.5)
        if need_spin > 0 and spin_like:
            s += 0.07 * float(getattr(p, "bowl_skill", 0.5) or 0.5)
        if need_pp > 0 and ppl >= 0.45:
            s += 0.06 * ppl
        if need_death > 0 and dth >= 0.45:
            s += 0.06 * dth
        if need_bat > 0 and getattr(p, "role_bucket", "") in (BATTER, WK_BATTER, ALL_ROUNDER):
            s += 0.04 * float(getattr(p, "bat_skill", 0.5) or 0.5)
        if bool(hd.get("captain_selected_for_team")):
            s += 0.05
        if bool(hd.get("wicketkeeper_selected_for_team")):
            s += 0.06
        out[p.name] = _clamp01(s)
    return out


def _tactical_modifiers(
    p: Any,
    conditions: dict[str, Any],
    fixture_context: dict[str, Any],
    hd: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    bf = float(conditions.get("batting_friendliness", 0.5))
    spin_f = float(conditions.get("spin_friendliness", 0.5))
    pace_b = float(conditions.get("pace_bias", 0.5))
    dew = float(conditions.get("dew_risk", 0.5))
    rain = float(conditions.get("rain_disruption_risk", 0.0))
    swing_p = float(conditions.get("swing_seam_proxy", 0.5))
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
    seam_like = getattr(p, "role_bucket", "") == BOWLER and not spin_like
    bat_s = float(getattr(p, "bat_skill", 0.5) or 0.5)
    bowl_s = float(getattr(p, "bowl_skill", 0.5) or 0.5)

    pitch_spin = 0.055 * (spin_f - 0.5) * bowl_s if spin_like else 0.0

    # 1. TOP ORDER PRIORITY (Boost)
    top_order_boost = 0.0
    pm = hd.get("player_metadata") or {}
    role_desc = str(pm.get("role_description") or "").lower()
    if "top_order" in role_desc or "opener" in role_desc:
        top_order_boost = 0.045

    # 2. CORE PLAYER LOCK (Strong bias)
    core_bias = 0.0
    is_cap = bool(pm.get("is_captain")) or bool(hd.get("captain_selected_for_team"))
    is_wk = bool(pm.get("is_wicketkeeper")) or bool(hd.get("wicketkeeper_selected_for_team"))
    tier1 = str(hd.get("marquee_tier") or "").lower() == "tier_1"
    if is_cap or is_wk or tier1:
        core_bias = 0.08

    # 7. IMPACT WEIGHT adjustment (recent XI presence)
    recent_xi_boost = 0.0
    lmd = (hd.get("selection_model_debug") or {}).get("last_match_detail") or {}
    if bool(lmd.get("was_in_last_match_xi")):
        recent_xi_boost = 0.0
    pitch_pace = 0.052 * (pace_b - 0.5) * bowl_s * (0.65 + 0.35 * swing_p) if seam_like else 0.0
    pitch_bat = 0.04 * (bf - 0.5) * bat_s if getattr(p, "role_bucket", "") in (BATTER, WK_BATTER, ALL_ROUNDER) else 0.0

    weather_dew = -0.035 * (dew - 0.55) * bowl_s if spin_like else 0.01 * (dew - 0.5) * bat_s
    weather_rain = -0.028 * rain * (bowl_s if seam_like else 0.35 * bat_s)

    toss_note = str(fixture_context.get("xi_scenario_branch_for_tactical") or "").strip()
    toss_adj = 0.0
    if toss_note == "if_team_bats_first" and getattr(p, "role_bucket", "") in (BATTER, WK_BATTER):
        toss_adj += 0.018 * bat_s
    elif toss_note == "if_team_bowls_first" and getattr(p, "role_bucket", "") == BOWLER:
        toss_adj += 0.022 * bowl_s

    h2h = float(hd.get("h2h_weighted_xi_rate") or 0.0)
    opp = 0.034 * (h2h - 0.35)

    squad = 0.0
    if bool(hd.get("captain_selected_for_team")):
        squad += 0.022
    if bool(hd.get("wicketkeeper_selected_for_team")):
        squad += 0.028

    parts = {
        "pitch_spin_assist": round(pitch_spin, 5),
        "pitch_pace_assist": round(pitch_pace, 5),
        "pitch_batting_friendly": round(pitch_bat, 5),
        "weather_dew": round(weather_dew, 5),
        "weather_rain": round(weather_rain, 5),
        "toss_scenario": round(toss_adj, 5),
        "opponent_matchup": round(opp, 5),
        "squad_need_manual": round(squad, 5),
        "top_order_priority": round(top_order_boost, 5),
        "core_player_lock": round(core_bias, 5),
        "recent_xi_presence": round(recent_xi_boost, 5),
    }
    raw = sum(parts.values())
    cap = float(getattr(config, "SELECTION_TACTICAL_ADJUST_CAP", 0.11))
    if raw > cap:
        raw = cap
    elif raw < -cap:
        raw = -cap
    return float(raw), parts


def apply_selection_model(
    players: list[Any],
    *,
    conditions: dict[str, Any],
    franchise_team_key: str,
    profiles: dict[str, dict[str, Any]],
    venue_weights: dict[str, float],
    pattern_row: Optional[dict[str, Any]],
    fixture_context: Optional[dict[str, Any]] = None,
    hn_by_player: dict[str, float],
    history_weights_by_pk: dict[str, float],
    composite_by_player: dict[str, float],
) -> None:
    """
    Mutates each player: ``selection_score``, ``history_debug['selection_model_debug']``,
    and refreshes ``scenario_xi`` / breakdown fields that depend on the final score.

    ``hn_by_player`` maps ``player_key`` → blended normalized history (pre-venue-pattern), ~[0,1].
    """
    if not players:
        return
    fc = fixture_context or {}
    w_rf = float(getattr(config, "SELECTION_WEIGHT_RECENT_FORM", 0.40))
    w_ipl = float(getattr(config, "SELECTION_WEIGHT_IPL_HISTORY_ROLE", 0.30))
    w_tb = float(getattr(config, "SELECTION_WEIGHT_TEAM_BALANCE_FIT", 0.20))
    w_ve = float(getattr(config, "SELECTION_WEIGHT_VENUE_EXPERIENCE", 0.10))

    pkeys_for_cache = [
        str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
        for p in players
    ]
    pkeys_for_cache = list(dict.fromkeys([k for k in pkeys_for_cache if k]))
    _t_rf = time.perf_counter()
    cache_by_pk = db.fetch_player_recent_form_cache_batch(pkeys_for_cache)
    _rf_ms = (time.perf_counter() - _t_rf) * 1000.0
    if audit_profile.audit_enabled():
        audit_profile.record_prediction_phase("selection_recent_form_cache_fetch_ms", _rf_ms)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "selection_model recent_form_cache_fetch_ms=%.2f squad_keys=%d cache_hits=%d",
            _rf_ms,
            len(pkeys_for_cache),
            len(cache_by_pk),
        )
    ref_d = _parse_iso_date((fixture_context or {}).get("reference_iso_date"))

    _t_sm = time.perf_counter()
    partial: list[tuple[Any, dict[str, Any]]] = []
    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        pk = str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
        prof = profiles.get(pk)
        derive_snap = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        if not derive_snap and prof:
            derive_snap = {
                "xi_selection_frequency": float(prof.get("xi_selection_frequency") or 0.0),
                "recent_usage_score": float(prof.get("recent_usage_score") or 0.0),
                "role_stability_score": float(prof.get("role_stability_score") or 0.0),
                "venue_fit_score": float(prof.get("venue_fit_score") or 0.0),
                "batting_position_ema": float(prof.get("batting_position_ema") or 0.0),
                "opener_likelihood": float(prof.get("opener_likelihood") or 0.0),
                "finisher_likelihood": float(prof.get("finisher_likelihood") or 0.0),
                "powerplay_bowler_likelihood": float(prof.get("powerplay_bowler_likelihood") or 0.0),
                "death_bowler_likelihood": float(prof.get("death_bowler_likelihood") or 0.0),
            }
        hn = float(hn_by_player.get(pk, 0.5))
        cr = cache_by_pk.get(pk)
        rf, rf_dbg = _recent_form_score_player(p, cr, derive_snap, reference_date=ref_d)
        ipl_s, ipl_dbg = _ipl_history_role_score(hn, derive_snap, hd)
        ven_s, ven_dbg = _venue_experience_score(p, hd, derive_snap, venue_weights, pk, conditions)
        prelim_tb = w_rf * rf + w_ipl * ipl_s + w_ve * ven_s + w_tb * 0.5
        partial.append(
            (
                p,
                {
                    "pk": pk,
                    "rf": rf,
                    "rf_dbg": rf_dbg,
                    "ipl_s": ipl_s,
                    "ipl_dbg": ipl_dbg,
                    "ven_s": ven_s,
                    "ven_dbg": ven_dbg,
                    "prelim_tb": prelim_tb,
                    "derive_snap": derive_snap,
                    "hn": hn,
                },
            )
        )

    prelim_map = {p.name: float(d["prelim_tb"]) for p, d in partial}
    tb_map = _team_balance_fit_scores(players, prelim_map)

    import history_xi

    for p, d in partial:
        hd = p.history_debug
        tb = float(tb_map.get(p.name, 0.5))
        base = w_rf * d["rf"] + w_ipl * d["ipl_s"] + w_tb * tb + w_ve * d["ven_s"]
        tact, tact_parts = _tactical_modifiers(p, conditions, fc, hd)
        hw = float(history_weights_by_pk.get(d["pk"], 0.82))
        legacy_blend = hw * d["hn"] + (1.0 - hw) * float(
            composite_by_player.get(p.name, float(getattr(p, "composite", 0.5)))
        )
        sel = _clamp01(base + tact)
        p.selection_score = float(sel)

        role_reason = (
            f"Role bucket {getattr(p, 'role_bucket', '')}; "
            f"opener/finisher/PP/death likelihoods drive IPL role fit."
        )
        rf_reason = (
            f"Recent T20 form from ``player_recent_form_cache`` "
            f"(union last-{config.SELECTION_RECENT_FORM_LAST_N_MATCHES} + {config.SELECTION_RECENT_FORM_MONTHS}m), "
            f"bat={d['rf_dbg'].get('batting_recent_form')}, bowl={d['rf_dbg'].get('bowling_recent_form')}."
        )
        if str(d["rf_dbg"].get("recent_form_source") or "") != "player_recent_form_cache":
            rf_reason += " No global T20 cache row — IPL derive recent_usage + composite fallback."

        ipl_reason = (
            f"IPL history blend uses normalized franchise XI signal ({d['ipl_dbg'].get('history_normalized_component')}) "
            f"with xi_frequency={d['ipl_dbg'].get('xi_selection_frequency')} and "
            f"role_stability={d['ipl_dbg'].get('role_stability_score')}."
        )
        bal_reason = (
            f"Team-balance vs provisional top-11 gaps (openers/WK/seam/spin/PP/death); score={round(tb, 4)}."
        )
        ven_reason = (
            f"Venue XI rate / derive venue fit / pattern weight; alignment to spin/pace/batting_friendliness."
        )
        tact_reason = ", ".join(f"{k}={v}" for k, v in tact_parts.items() if abs(float(v)) > 1e-6)

        hd["selection_model_debug"] = {
            "base_score_breakdown": {
                "recent_form_score": round(d["rf"], 5),
                "ipl_history_and_role_score": round(d["ipl_s"], 5),
                "team_balance_fit_score": round(tb, 5),
                "venue_experience_score": round(d["ven_s"], 5),
                "weights": {"recent": w_rf, "ipl": w_ipl, "team_balance": w_tb, "venue": w_ve},
                "base_weighted_sum": round(
                    w_rf * d["rf"] + w_ipl * d["ipl_s"] + w_tb * tb + w_ve * d["ven_s"], 5
                ),
            },
            "tactical_modifiers": tact_parts,
            "tactical_adjustment_total": round(tact, 5),
            "final_selection_score": round(sel, 5),
            "legacy_blend_selection_score": round(legacy_blend, 5),
            "recent_form_detail": d["rf_dbg"],
            "ipl_history_detail": d["ipl_dbg"],
            "venue_experience_detail": d["ven_dbg"],
            "explainability": {
                "role_reason": role_reason,
                "recent_form_reason": rf_reason,
                "ipl_role_history_reason": ipl_reason,
                "team_balance_reason": bal_reason,
                "venue_reason": ven_reason,
                "tactical_modifiers_reason": tact_reason or "neutral",
            },
        }

        is_nf = bool(fc.get("is_night", False))
        scen_pkg = history_xi._scenario_xi_package_for_player(
            p, float(sel), d.get("derive_snap"), conditions, is_night=is_nf
        )
        hd["scenario_xi"] = scen_pkg

        sb = hd.get("scoring_breakdown")
        if isinstance(sb, dict):
            sb = dict(sb)
            sb["final_selection_score"] = round(sel, 5)
            sb["selection_model_base"] = hd["selection_model_debug"]["base_score_breakdown"]
            sb["selection_model_tactical"] = tact_parts
            hd["scoring_breakdown"] = sb

        # Full ``selection_score_components`` merge happens in ``history_xi.compute_selection_scores``.

    _loop_ms = (time.perf_counter() - _t_sm) * 1000.0
    if audit_profile.audit_enabled():
        audit_profile.record_prediction_phase("selection_model_apply_player_loop_ms", _loop_ms)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "selection_model scoring_loops_ms=%.2f n_players=%d",
            _loop_ms,
            len(players),
        )


__all__ = ["apply_selection_model"]
