"""Render stored prediction results in Streamlit (no prediction engine calls)."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

import audit_profile
import config
import db
import h2h_history
import history_sync
import ipl_teams
import ipl_squad
import learner
import predictor
import rules_xi
import win_probability_engine
import player_role_classifier
import batting_order_whatif
from history_context import build_history_context
from venues import VenueProfile, resolve_venue, venue_conditions_summary

_perf_logger = logging.getLogger("ipl_predictor.perf")
_logger = logging.getLogger(__name__)


def _coach_signature(r: dict[str, Any]) -> str:
    # Stored payloads may encode team blocks as dicts or strings; only XI (list[dict]) is assumed structured.
    name_a = _team_name_from_stored_result(r, "team_a")
    name_b = _team_name_from_stored_result(r, "team_b")
    team_a_block = r.get("team_a") if isinstance(r.get("team_a"), dict) else {}
    team_b_block = r.get("team_b") if isinstance(r.get("team_b"), dict) else {}
    payload = "|".join(
        [
            str(name_a or ""),
            str(name_b or ""),
            ",".join(sorted(str(x.get("name") or "") for x in (team_a_block.get("xi") or []))),
            ",".join(sorted(str(x.get("name") or "") for x in (team_b_block.get("xi") or []))),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _coach_state(r: dict[str, Any]) -> dict[str, Any]:
    sig = _coach_signature(r)
    key = "coach_tools_state"
    cur = st.session_state.get(key)
    if isinstance(cur, dict) and cur.get("signature") == sig:
        _coach_merge_defaults(cur, r)
        return cur

    def _px(side: str) -> list[str]:
        return [str(x.get("name") or "") for x in (r.get(side, {}).get("xi") or []) if x.get("name")]

    def _pi(side: str) -> list[str]:
        return [
            str(x.get("name") or "")
            for x in (r.get(side, {}).get("impact_subs") or [])
            if x.get("name")
        ][:5]

    def _pbo(side: str) -> list[str]:
        return list(r.get(side, {}).get("batting_order") or [])

    state: dict[str, Any] = {
        "signature": sig,
        "team_a_predicted_xi": list(_px("team_a")),
        "team_b_predicted_xi": list(_px("team_b")),
        "team_a_final_xi": list(_px("team_a")),
        "team_b_final_xi": list(_px("team_b")),
        "team_a_impact_names": list(_pi("team_a")),
        "team_b_impact_names": list(_pi("team_b")),
        "team_a_batting_order": list(_pbo("team_a")),
        "team_b_batting_order": list(_pbo("team_b")),
        "team_a_finalised": False,
        "team_b_finalised": False,
        "team_a_finalised_xi": None,
        "team_a_finalised_batting_order": None,
        "team_a_finalised_impact_names": None,
        "team_b_finalised_xi": None,
        "team_b_finalised_batting_order": None,
        "team_b_finalised_impact_names": None,
    }
    st.session_state[key] = state
    return state


def _coach_merge_defaults(state: dict[str, Any], r: dict[str, Any]) -> None:
    for side in ("team_a", "team_b"):
        pred = [str(x.get("name") or "") for x in (r.get(side, {}).get("xi") or []) if x.get("name")]
        if f"{side}_predicted_xi" not in state:
            state[f"{side}_predicted_xi"] = list(pred)
        if f"{side}_final_xi" not in state:
            state[f"{side}_final_xi"] = list(pred)
        if f"{side}_impact_names" not in state:
            impact = [
                str(x.get("name") or "")
                for x in (r.get(side, {}).get("impact_subs") or [])
                if x.get("name")
            ]
            state[f"{side}_impact_names"] = impact[:5]
        if f"{side}_batting_order" not in state:
            state[f"{side}_batting_order"] = list(r.get(side, {}).get("batting_order") or [])
        for k, default in (
            (f"{side}_finalised", False),
            (f"{side}_finalised_xi", None),
            (f"{side}_finalised_batting_order", None),
            (f"{side}_finalised_impact_names", None),
        ):
            if k not in state:
                state[k] = default


def _effective_xi_names(state: dict[str, Any], side_key: str) -> list[str]:
    if state.get(f"{side_key}_finalised"):
        fx = state.get(f"{side_key}_finalised_xi")
        if isinstance(fx, list) and len(fx) == 11:
            return list(fx)
    return list(state.get(f"{side_key}_final_xi") or [])


def _effective_impact_names(state: dict[str, Any], side_key: str) -> list[str]:
    if state.get(f"{side_key}_finalised"):
        im = state.get(f"{side_key}_finalised_impact_names")
        if isinstance(im, list):
            return list(im)
    return list(state.get(f"{side_key}_impact_names") or [])


def _effective_batting_order(state: dict[str, Any], side_key: str) -> list[str]:
    if state.get(f"{side_key}_finalised"):
        bo = state.get(f"{side_key}_finalised_batting_order")
        if isinstance(bo, list) and bo:
            return list(bo)
    return list(state.get(f"{side_key}_batting_order") or [])


def _rebuild_batting_order_state(
    state: dict[str, Any],
    *,
    side_key: str,
    team_name: str,
    squad_map: dict[str, dict[str, Any]],
    conditions: dict[str, Any],
    venue_keys: list[str],
) -> None:
    if state.get(f"{side_key}_finalised"):
        return
    names = list(state.get(f"{side_key}_final_xi") or [])
    if len(names) != 11:
        return
    xi, errs = _validate_manual_xi(team_name, squad_map, names, conditions)
    if errs or len(xi) != 11:
        return
    try:
        order = predictor.build_batting_order(
            xi,
            conditions,
            team_name=team_name,
            venue_keys=venue_keys or [],
            out_warnings=[],
        )
        if len(order) == 11:
            state[f"{side_key}_batting_order"] = list(order)
    except Exception:
        pass


def _resolve_batting_order_for_display(
    state: dict[str, Any],
    side_key: str,
) -> list[str]:
    xi_names = list(_effective_xi_names(state, side_key))
    xi_set = set(xi_names)
    bo = _effective_batting_order(state, side_key)
    if len(bo) == 11 and set(bo) == xi_set:
        return list(bo)
    return xi_names


def _xi_rows_for_display(batting_order: list[str], squad_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, nm in enumerate(batting_order):
        base = squad_map.get(nm)
        if not base:
            continue
        row = dict(base)
        row["name"] = nm
        row["bat_order"] = i + 1
        row["final_position"] = i + 1
        rows.append(row)
    return rows


def _impact_rows_for_display(
    names: list[str],
    squad_map: dict[str, dict[str, Any]],
    impact_template: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_orig = {str(x.get("name") or ""): dict(x) for x in impact_template if x.get("name")}
    out: list[dict[str, Any]] = []
    for nm in names:
        row = dict(squad_map.get(nm) or {"name": nm})
        row["name"] = nm
        for k, v in (by_orig.get(nm) or {}).items():
            row.setdefault(k, v)
        out.append(row)
    return out


def _bench_rows_exclusive(
    xi_names: list[str],
    impact_names: list[str],
    full_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocked = set(xi_names) | set(impact_names)
    return [row for row in full_rows if str(row.get("name") or "") not in blocked]


def _team_slug_from_name(name: str) -> str:
    return ipl_teams.slug_for_canonical_label(str(name or "").strip()) or ""


def _logo_html_for_team_name(name: str, *, width: int = 26) -> str:
    slug = _team_slug_from_name(name)
    if not slug:
        return ""
    path = ipl_teams.team_logo_path_for_slug(slug)
    if not path:
        return ""
    return f'<img src="data:image/png;base64,{_img_b64(path)}" style="width:{width}px;height:{width}px;object-fit:contain;vertical-align:middle;margin-right:8px;" />'


def _img_b64(path: str) -> str:
    import base64
    from pathlib import Path

    raw = Path(path).read_bytes()
    return base64.b64encode(raw).decode("ascii")


def _role_tag_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    rb = str(row.get("role_bucket") or "")
    if bool(row.get("is_wk_role")) or rb == ipl_squad.WK_BATTER:
        tags.append(("WK", "#7c3aed"))
    if rb == ipl_squad.BATTER:
        tags.append(("BAT", "#2563eb"))
    if rb == ipl_squad.ALL_ROUNDER:
        tags.append(("AR", "#059669"))
    if rb == ipl_squad.BOWLER:
        tags.append(("BOWL", "#dc2626"))
    if bool(row.get("overseas")):
        tags.append(("Overseas", "#111827"))
    bt = str(row.get("bowling_type") or "").lower()
    meta = row.get("_meta") if isinstance(row.get("_meta"), dict) else {}
    btb = str(meta.get("bowling_type_bucket") or "").lower()
    if any(x in bt for x in ("spin", "orthodox", "wrist", "finger")) or btb in (
        "finger_spin",
        "wrist_spin",
        "left_arm_orthodox",
        "mystery_spin",
    ):
        tags.append(("Spinner", "#7c2d12"))
    if any(x in bt for x in ("fast", "medium", "pace")) or btb in (
        "pace",
        "right_arm_fast",
        "right_arm_fast_medium",
        "left_arm_fast",
        "left_arm_fast_medium",
    ):
        tags.append(("Pacer", "#0f766e"))
    return tags


def _render_badged_players(rows: list[dict[str, Any]], *, empty_text: str) -> None:
    if not rows:
        st.caption(empty_text)
        return
    cards: list[str] = []
    for row in rows:
        chips = "".join(
            f'<span style="display:inline-block;background:{color};color:white;border-radius:999px;padding:2px 8px;margin:2px 4px 0 0;font-size:11px;">{label}</span>'
            for label, color in _role_tag_row(row)
        )
        cards.append(
            '<div style="border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;margin-bottom:8px;">'
            f'<div style="font-weight:600;">{row.get("name")}</div>'
            f'<div style="margin-top:4px;">{chips}</div>'
            '</div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def _team_name_from_stored_result(r: dict[str, Any], side_key: str) -> str:
    """Stored payloads may use ``team_a`` / ``team_b`` as either a dict (with ``name``) or a bare string."""
    block = r.get(side_key)
    if isinstance(block, dict):
        return str(block.get("name") or "").strip()
    if isinstance(block, str):
        return block.strip()
    return ""


def _condition_scalar_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not str(v).strip():
        return True
    try:
        float(v)
    except (TypeError, ValueError):
        return True
    return False


# Keys required by win_probability_engine / batting_order_scores on conditions dict.
_WIN_ENGINE_CONDITION_SCALARS: tuple[str, ...] = (
    "pace_bias",
    "spin_friendliness",
    "batting_friendliness",
    "dew_risk",
    "swing_seam_proxy",
    "rain_disruption_risk",
    "boundary_size",
    "heat_fatigue",
)

_WIN_ENGINE_CONDITION_DEFAULTS: dict[str, float] = {
    "pace_bias": 0.5,
    "spin_friendliness": 0.5,
    "batting_friendliness": 0.55,
    "dew_risk": 0.5,
    "swing_seam_proxy": 0.45,
    "rain_disruption_risk": 0.0,
    "boundary_size": 0.5,
    "heat_fatigue": 0.0,
}


def _normalize_conditions_for_win_recalc(
    stored: dict[str, Any],
    *,
    venue_profile: VenueProfile,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Merge stored conditions with venue+weather-derived fields so compute_win_probability never KeyErrors.

    Returns ``(normalized_conditions, sorted_keys_before, sorted_keys_after)``.
    """
    before_keys = sorted(stored.keys())
    ws = stored.get("weather_snapshot")
    weather = ws if isinstance(ws, dict) else {}
    derived = venue_conditions_summary(venue_profile, weather)
    out: dict[str, Any] = dict(stored)
    for k, v in derived.items():
        if k == "weather_snapshot":
            continue
        if _condition_scalar_missing(out.get(k)):
            out[k] = v
    if not isinstance(out.get("weather_snapshot"), dict):
        out["weather_snapshot"] = dict(weather)
    for k in _WIN_ENGINE_CONDITION_SCALARS:
        if _condition_scalar_missing(out.get(k)):
            out[k] = float(_WIN_ENGINE_CONDITION_DEFAULTS.get(k, 0.5))
    if not str(out.get("venue") or "").strip():
        out["venue"] = venue_profile.display_name
    if not str(out.get("notes") or "").strip():
        out["notes"] = str(derived.get("notes") or venue_profile.notes or "")
    return out, before_keys, sorted(out.keys())


def _win_context_from_stored_prediction(r: dict[str, Any], cond_d: dict[str, Any]) -> dict[str, Any]:
    """Normalized venue + conditions + toss keys for win_probability_engine (shared by recalc & what-if)."""
    venue = resolve_venue(str(cond_d.get("venue") or ""))
    conditions, _, _ = _normalize_conditions_for_win_recalc(cond_d, venue_profile=venue)
    lc = r.get("learning_context")
    lc_d = lc if isinstance(lc, dict) else {}
    venue_keys = list((lc_d.get("venue_keys_tried") or []))
    ts = r.get("toss_scenario")
    ts_d = ts if isinstance(ts, dict) else {}
    return {
        "venue": venue,
        "conditions": conditions,
        "venue_keys": venue_keys,
        "toss_key": str((ts_d.get("key") or "unknown")),
        "a_bats_first": ts_d.get("team_a_bats_first"),
        "is_night": bool(lc_d.get("fixture_night")),
        "pace_bias": float(conditions["pace_bias"]),
        "spin_friendliness": float(conditions["spin_friendliness"]),
    }


def _compute_win_engine_core(
    name_a: str,
    name_b: str,
    xi_a: list[Any],
    xi_b: list[Any],
    order_a: list[str],
    order_b: list[str],
    ctx: dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Run compute_win_probability; return (team_a_pct, team_b_pct, error_message)."""
    try:
        match_rows = db.fetch_match_results_meta(450)
        hctx = build_history_context()
        win = win_probability_engine.compute_win_probability(
            name_a,
            name_b,
            xi_a,
            xi_b,
            order_a,
            order_b,
            ctx["venue"],
            ctx["conditions"],
            venue_keys=ctx["venue_keys"],
            match_rows=match_rows,
            toss_scenario_key=ctx["toss_key"],
            a_bats_first_selected=ctx["a_bats_first"],
            chase_share_by_venue=hctx.chase_share_by_venue,
            is_night_fixture=ctx["is_night"],
        )
        win_dict = win.to_dict() if hasattr(win, "to_dict") else {}
        team_a_selected = win_dict.get("team_a_win_pct_selected_toss")
        if team_a_selected is None:
            team_a_selected = getattr(win, "team_a_win_pct_selected_toss", None)
        if team_a_selected is None:
            team_a_selected = win_dict.get("team_a_win_pct_neutral_toss")
        if team_a_selected is None:
            team_a_selected = getattr(win, "team_a_win_pct_neutral_toss", None)
        if team_a_selected is None:
            team_a_selected = win_dict.get("marginal_team_a_win_pct")
        if team_a_selected is None:
            team_a_selected = getattr(win, "prob_team_a_if_a_bats_first_pct", None)
        if team_a_selected is None:
            return None, None, "Win engine did not return a usable Team A win percentage."
        team_a_final = float(team_a_selected)
        team_b_selected = win_dict.get("team_b_win_pct_selected_toss")
        if team_b_selected is None:
            team_b_selected = getattr(win, "team_b_win_pct_selected_toss", None)
        if team_b_selected is None:
            team_b_selected = 100.0 - team_a_final
        return team_a_final, float(team_b_selected), None
    except Exception as exc:
        return None, None, str(exc)


def _batting_order_valid_for_xi(order: list[str], xi: list[Any], label: str) -> Optional[str]:
    if len(xi) != 11:
        return f"{label}: XI must have 11 players."
    names = {getattr(p, "name", "") for p in xi}
    if len(order) != 11 or set(order) != names:
        return f"{label}: batting order must list all 11 XI players exactly once."
    return None


def _render_batting_order_whatif_studio(
    r: dict[str, Any],
    state: dict[str, Any],
    squad_map_a: dict[str, dict[str, Any]],
    squad_map_b: dict[str, dict[str, Any]],
    disp_a: str,
    disp_b: str,
    cond_raw: Any,
) -> None:
    """
    Batting-order-only what-if: same XIs, optional reorder on one side, win % delta + matchup notes.
    Does not mutate coach finalise state or XI lists.
    """
    st.subheader("Batting Order What-If Studio")
    st.caption(
        "Experiment with batting order only — playing XIs stay as set under **Finalise Playing XI**. "
        "Opposition bowling phases use franchise phase-usage summaries when available."
    )
    cond_d = cond_raw if isinstance(cond_raw, dict) else {}
    names_a = _effective_xi_names(state, "team_a")
    names_b = _effective_xi_names(state, "team_b")
    if len(names_a) != 11 or len(names_b) != 11:
        st.info("Set valid 11-player XIs on both sides to use the what-if studio.")
        return

    name_a = _team_name_from_stored_result(r, "team_a")
    name_b = _team_name_from_stored_result(r, "team_b")
    xi_a, errs_a = _validate_manual_xi(name_a, squad_map_a, names_a, cond_d)
    xi_b, errs_b = _validate_manual_xi(name_b, squad_map_b, names_b, cond_d)
    if errs_a or errs_b or len(xi_a) != 11 or len(xi_b) != 11:
        st.warning("Fix XI validation errors above before running what-if scenarios.")
        return

    sig = _coach_signature(r)
    bo_a_base = _effective_batting_order(state, "team_a")
    bo_b_base = _effective_batting_order(state, "team_b")
    if _batting_order_valid_for_xi(bo_a_base, xi_a, "Team A") or _batting_order_valid_for_xi(bo_b_base, xi_b, "Team B"):
        st.info("Batting order must match the current XI (11 names, no extras). Re-finalise or reset batting order.")
        return

    def _wk(sk: str) -> str:
        return f"whatif_bo_{sig}_{sk}"

    xi_set_a, xi_set_b = set(names_a), set(names_b)
    for sk, xset in (("team_a", xi_set_a), ("team_b", xi_set_b)):
        k = _wk(sk)
        cur = st.session_state.get(k)
        if not isinstance(cur, list) or len(cur) != 11 or set(cur) != xset:
            st.session_state[k] = list(_effective_batting_order(state, sk))

    try:
        wctx = _win_context_from_stored_prediction(r, cond_d)
    except Exception as exc:
        st.warning(f"Could not build venue/conditions context: {exc}")
        return

    _whatif_side_options = (f"{disp_a} (A)", f"{disp_b} (B)")
    side_label = st.radio(
        "Team to edit",
        _whatif_side_options,
        horizontal=True,
        key=f"whatif_pick_side_{sig}",
    )
    side_key = "team_a" if side_label == _whatif_side_options[0] else "team_b"
    disp_side = disp_a if side_key == "team_a" else disp_b
    opp_names = names_b if side_key == "team_a" else names_a
    opp_squad = squad_map_b if side_key == "team_a" else squad_map_a
    opp_label = disp_b if side_key == "team_a" else disp_a
    xi_side = xi_a if side_key == "team_a" else xi_b
    bo_base_side = bo_a_base if side_key == "team_a" else bo_b_base
    k_edit = _wk(side_key)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current order (session)**")
        st.markdown(
            "".join(f"{i + 1}. {nm}  \n" for i, nm in enumerate(bo_base_side))
            or "—"
        )
    with c2:
        st.markdown("**Edited order (what-if)**")
        st.markdown("".join(f"{i + 1}. {nm}  \n" for i, nm in enumerate(st.session_state[k_edit])))

    st.markdown("**Swap two players**")
    xlist = list(names_a if side_key == "team_a" else names_b)
    s1, s2, s3 = st.columns([2, 2, 1])
    with s1:
        p_a = st.selectbox("Player A", xlist, key=f"whatif_swap_a_{sig}_{side_key}")
    with s2:
        p_b = st.selectbox("Player B", xlist, key=f"whatif_swap_b_{sig}_{side_key}")
    with s3:
        st.write("")
        st.write("")
        if st.button("Swap", key=f"whatif_swap_btn_{sig}_{side_key}"):
            lst = list(st.session_state[k_edit])
            if p_a != p_b and p_a in lst and p_b in lst:
                ia, ib = lst.index(p_a), lst.index(p_b)
                lst[ia], lst[ib] = lst[ib], lst[ia]
                st.session_state[k_edit] = lst
                st.rerun()

    st.markdown("**Move one player to a slot**")
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        p_mv = st.selectbox("Player to move", xlist, key=f"whatif_mv_p_{sig}_{side_key}")
    with m2:
        slot_to = st.number_input("New position (1–11)", min_value=1, max_value=11, value=1, key=f"whatif_mv_slot_{sig}_{side_key}")
    with m3:
        st.write("")
        st.write("")
        if st.button("Move", key=f"whatif_mv_btn_{sig}_{side_key}"):
            lst = list(st.session_state[k_edit])
            if p_mv in lst:
                lst = [x for x in lst if x != p_mv]
                lst.insert(int(slot_to) - 1, p_mv)
                if len(lst) == 11:
                    st.session_state[k_edit] = lst
                    st.rerun()

    if st.button("Reset edited order to current session order", key=f"whatif_reset_bo_{sig}_{side_key}"):
        st.session_state[k_edit] = list(bo_base_side)
        st.rerun()

    plan = batting_order_whatif.project_opposition_bowling_plan(opp_names, opp_squad, opp_label)
    st.markdown("**Opposition bowling (phase projection)**")
    st.caption("From squad role + franchise phase-usage shares (powerplay / middle / death).")
    pp_n = ", ".join(t[0] for t in plan["powerplay"][:5]) or "—"
    md_n = ", ".join(t[0] for t in plan["middle"][:6]) or "—"
    dt_n = ", ".join(t[0] for t in plan["death"][:5]) or "—"
    st.markdown(f"- **Powerplay lean:** {pp_n}")
    st.markdown(f"- **Middle overs:** {md_n}")
    st.markdown(f"- **Death overs:** {dt_n}")

    if st.button("Evaluate batting order", key=f"whatif_eval_{sig}"):
        trial_a = list(st.session_state[_wk("team_a")])
        trial_b = list(st.session_state[_wk("team_b")])
        err = _batting_order_valid_for_xi(trial_a, xi_a, "Team A") or _batting_order_valid_for_xi(trial_b, xi_b, "Team B")
        if err:
            st.session_state[f"whatif_eval_result_{sig}"] = {"error": err}
        else:
            b_pa, b_pb, e0 = _compute_win_engine_core(name_a, name_b, xi_a, xi_b, bo_a_base, bo_b_base, wctx)
            n_pa, n_pb, e1 = _compute_win_engine_core(name_a, name_b, xi_a, xi_b, trial_a, trial_b, wctx)
            cards: list[dict[str, Any]] = []
            if not e0 and not e1 and b_pa is not None and n_pa is not None:
                sm_side = squad_map_a if side_key == "team_a" else squad_map_b
                cards = batting_order_whatif.build_recommendation_cards(
                    baseline_order=bo_base_side,
                    edited_order=st.session_state[k_edit],
                    batter_squad_map=sm_side,
                    opposition_xi=opp_names,
                    opposition_squad_map=opp_squad,
                    opposition_franchise_label=opp_label,
                    pace_bias=float(wctx["pace_bias"]),
                    spin_friendliness=float(wctx["spin_friendliness"]),
                )
            st.session_state[f"whatif_eval_result_{sig}"] = {
                "error": e0 or e1,
                "baseline_a": b_pa,
                "baseline_b": b_pb,
                "new_a": n_pa,
                "new_b": n_pb,
                "side_key": side_key,
                "cards": cards,
            }
        st.rerun()

    ev = st.session_state.get(f"whatif_eval_result_{sig}")
    if isinstance(ev, dict) and ev:
        if ev.get("error"):
            st.warning(str(ev["error"]))
        elif ev.get("baseline_a") is not None and ev.get("new_a") is not None:
            ba, bb = float(ev["baseline_a"]), float(ev["baseline_b"] or (100.0 - ev["baseline_a"]))
            na, nb = float(ev["new_a"]), float(ev["new_b"] or (100.0 - ev["new_a"]))
            sk_ev = str(ev.get("side_key") or "team_a")
            st.markdown("**Win probability (same XIs, edited batting order on selected side)**")
            m1, m2, m3 = st.columns(3)
            focus_disp = disp_a if sk_ev == "team_a" else disp_b
            with m1:
                st.metric(f"Baseline · {disp_a}", f"{ba:.1f}%")
                st.caption(f"{disp_b}: {bb:.1f}%")
            with m2:
                st.metric(f"After what-if · {disp_a}", f"{na:.1f}%", delta=f"{na - ba:+.1f}%")
                st.caption(f"{disp_b}: {nb:.1f}% ({nb - bb:+.1f}%)")
            with m3:
                st.metric("Focus team Δ", f"{(na - ba) if sk_ev == 'team_a' else (nb - bb):+.1f} pts", help=f"Win % change for {focus_disp}")
            st.markdown("**Recommendations (moved batters)**")
            cards = ev.get("cards") or []
            if not cards:
                st.caption("No batting-order changes vs current session order, or no moved players to analyse.")
            for card in cards:
                with st.container():
                    st.markdown(
                        f"##### {card.get('name')} · #{card.get('old_position')} → #{card.get('new_position')} "
                        f"({card.get('verdict', 'neutral')})"
                    )
                    st.markdown(card.get("recommendation") or "")
                    st.caption(
                        f"Likely bowlers: {card.get('likely_bowlers')} · "
                        f"Types: {card.get('likely_bowling_types')}"
                    )


def _build_squad_row_maps(r: dict[str, Any], side_key: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    team_block = r.get(side_key) or {}
    squad_debug = r.get("squad_debug") or {}
    pld = r.get("prediction_layer_debug") or {}
    side_dbg = pld.get(side_key) or {}
    structured = list(squad_debug.get(f"structured_squad_{side_key}") or [])
    scoring = list(side_dbg.get("scoring_breakdown_per_player") or [])
    xi_rows = list(team_block.get("xi") or [])
    omitted = list(side_dbg.get("omitted_from_playing_xi") or [])

    by_name: dict[str, dict[str, Any]] = {}
    for row in structured:
        nm = str(row.get("name") or row.get("player_name") or "")
        if nm:
            by_name[nm] = dict(row)
    for source in (scoring, omitted, xi_rows):
        for row in source:
            nm = str(row.get("name") or row.get("player_name") or row.get("squad_display_name") or "")
            if not nm:
                continue
            base = by_name.get(nm, {})
            merged = dict(base)
            merged.update(row)
            merged["name"] = nm
            by_name[nm] = merged

    keys = []
    for row in by_name.values():
        pk = str(row.get("canonical_player_key") or row.get("player_key") or "")
        if pk:
            keys.append(pk)
    meta_map = db.fetch_player_metadata_batch(list(dict.fromkeys(keys)))
    ordered = sorted(by_name.values(), key=lambda x: str(x.get("name") or "").lower())
    for row in ordered:
        pk = str(row.get("canonical_player_key") or row.get("player_key") or "")
        row["_meta"] = dict(meta_map.get(pk) or {})
        row["_team_name"] = str(team_block.get("name") or "")
        row["_canonical_team_key"] = str(row.get("canonical_team_key") or "")
        row["_batting_roles"] = list(row.get("batting_roles") or [])
        if "role" not in row:
            rb = str(row.get("role_bucket") or ipl_squad.BATTER)
            row["role"] = ipl_squad.role_bucket_to_predictor_role(rb)
    return ordered, {str(rw.get("name") or ""): rw for rw in ordered if rw.get("name")}


def _row_to_player(row: dict[str, Any]) -> Any:
    role = str(row.get("role") or ipl_squad.role_bucket_to_predictor_role(str(row.get("role_bucket") or ipl_squad.BATTER)))
    bat_skill, bowl_skill, is_wk = predictor._role_skills(role)
    rb = str(row.get("role_bucket") or ipl_squad.BATTER)
    # XI cap uses structured squad / engine WK signals only. ``is_keeper`` from DB/name heuristics
    # (see db._keeper_name_heuristic) is not authoritative for wicketkeeper role classification.
    is_wicketkeeper = bool(row.get("is_wk_role") or is_wk or rb == ipl_squad.WK_BATTER)
    player = predictor.SquadPlayer(
        name=str(row.get("name") or ""),
        role=role,
        is_overseas=bool(row.get("overseas")),
        player_key=str(row.get("player_key") or row.get("canonical_player_key") or ""),
        team_display_name=str(row.get("_team_name") or ""),
        canonical_team_key=str(row.get("_canonical_team_key") or ""),
        canonical_player_key=str(row.get("canonical_player_key") or row.get("player_key") or ""),
        role_bucket=rb,
        bat_skill=float(row.get("bat_skill") or bat_skill),
        bowl_skill=float(row.get("bowl_skill") or bowl_skill),
        is_wicketkeeper=is_wicketkeeper,
        batting_roles=list(row.get("_batting_roles") or []),
        bowling_type=row.get("bowling_type"),
        is_opener_candidate=bool(row.get("is_opener_candidate")),
        is_finisher_candidate=bool(row.get("is_finisher_candidate")),
        history_xi_score=float(row.get("history_xi_score") or 0.0),
        history_batting_ema=float(row.get("history_batting_ema") or 99.0),
        selection_score=float(row.get("selection_score") or row.get("final_selection_score") or 0.0),
        composite=float(row.get("composite") or row.get("composite_score") or 0.0),
    )
    player.history_debug = {
        "player_metadata": dict(row.get("_meta") or {}),
        "selection_model_debug": dict(row.get("selection_model_debug") or {}) if isinstance(row.get("selection_model_debug"), dict) else {},
        "derive_player_profile": dict(row.get("derive_player_profile") or {}) if isinstance(row.get("derive_player_profile"), dict) else {},
        "bowler_phase_summary": dict(row.get("bowler_phase_summary") or {}) if isinstance(row.get("bowler_phase_summary"), dict) else {},
        "role_band": row.get("role_band"),
        "batting_band": row.get("batting_band"),
        "dominant_position": row.get("dominant_position"),
        "batting_position_rows_found": row.get("batting_position_rows_found"),
        "marquee_tier": row.get("marquee_tier"),
        "selection_reason_summary": row.get("selection_reason_summary"),
        "canonical_player_key": row.get("canonical_player_key"),
        "designated_keeper": bool(row.get("designated_keeper")),
    }
    predictor._set_player_ipl_flags(player)
    return player


def _validate_manual_xi(
    team_name: str,
    squad_map: dict[str, dict[str, Any]],
    final_names: list[str],
    conditions: dict[str, Any],
) -> tuple[list[Any], list[str]]:
    errors: list[str] = []
    if len(final_names) != len(set(final_names)):
        errors.append("Duplicate players are not allowed.")
    if len(final_names) != 11:
        errors.append(f"{team_name}: XI must contain exactly 11 players.")
    missing = [nm for nm in final_names if nm not in squad_map]
    if missing:
        errors.append(f"{team_name}: player not available in squad — {', '.join(missing[:4])}")
    xi = [_row_to_player(squad_map[nm]) for nm in final_names if nm in squad_map]
    squad = [_row_to_player(row) for row in squad_map.values()]
    if len(xi) == 11:
        res = rules_xi.validate_xi(xi, conditions=conditions, squad=squad)
        if any(v.code == "wk_max" for v in res.violations):
            _logger.warning(
                "manual_xi wicketkeeper_debug team=%s players=%s",
                team_name,
                player_role_classifier.wicketkeeper_xi_debug_rows(xi),
            )
        errors.extend(v.message for v in res.violations)
    return xi, errors


def _recalculate_final_win_prediction(
    r: dict[str, Any],
    squad_map_a: dict[str, dict[str, Any]],
    squad_map_b: dict[str, dict[str, Any]],
    final_names_a: list[str],
    final_names_b: list[str],
) -> tuple[Optional[dict[str, float]], list[str], dict[str, Any]]:
    dbg: dict[str, Any] = {
        "condition_keys_before_normalization": "—",
        "condition_keys_after_normalization": "—",
    }
    name_a = _team_name_from_stored_result(r, "team_a")
    name_b = _team_name_from_stored_result(r, "team_b")
    cond_raw = r.get("conditions")
    cond_d = cond_raw if isinstance(cond_raw, dict) else {}
    xi_a, errs_a = _validate_manual_xi(name_a, squad_map_a, final_names_a, cond_d)
    xi_b, errs_b = _validate_manual_xi(name_b, squad_map_b, final_names_b, cond_d)
    if errs_a or errs_b:
        dbg["condition_keys_before_normalization"] = sorted(cond_d.keys())
        return None, [*errs_a, *errs_b], dbg
    try:
        wctx = _win_context_from_stored_prediction(r, cond_d)
        dbg["condition_keys_before_normalization"] = sorted(cond_d.keys())
        dbg["condition_keys_after_normalization"] = sorted(wctx["conditions"].keys())
        order_a = predictor.build_batting_order(
            xi_a,
            wctx["conditions"],
            team_name=name_a,
            venue_keys=wctx["venue_keys"],
            out_warnings=[],
        )
        order_b = predictor.build_batting_order(
            xi_b,
            wctx["conditions"],
            team_name=name_b,
            venue_keys=wctx["venue_keys"],
            out_warnings=[],
        )
        team_a_final, team_b_final, werr = _compute_win_engine_core(
            name_a, name_b, xi_a, xi_b, order_a, order_b, wctx
        )
        if werr:
            raise ValueError(werr)
        return (
            {
                "team_a_final_win_pct": team_a_final,
                "team_b_final_win_pct": team_b_final,
            },
            [],
            dbg,
        )
    except Exception as exc:
        dbg.setdefault("condition_keys_before_normalization", sorted(cond_d.keys()))
        return None, [f"Final win recalculation unavailable: {exc}"], dbg


def _fetch_direct_matchup(batter_key: str, bowler_key: str) -> Optional[dict[str, Any]]:
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT balls, runs, dismissals, strike_rate, dot_ball_pct, boundary_pct,
                   innings_count, match_count, last_match_date, sample_size_confidence
            FROM batter_bowler_matchup_summary
            WHERE batter_key = ? AND bowler_key = ?
            """,
            (batter_key, bowler_key),
        ).fetchone()
    return dict(row) if row else None


def _fetch_batter_type_fallback(batter_key: str, bowling_type_bucket: str) -> Optional[dict[str, Any]]:
    if not bowling_type_bucket:
        return None
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT balls, runs, dismissals, strike_rate, dot_ball_pct, boundary_pct, sample_size_confidence
            FROM batter_vs_bowling_type_summary
            WHERE batter_key = ? AND bowling_type_bucket = ?
            """,
            (batter_key, bowling_type_bucket),
        ).fetchone()
    return dict(row) if row else None


def _fetch_bowler_hand_fallback(bowler_key: str, batting_hand: str) -> Optional[dict[str, Any]]:
    if not batting_hand:
        return None
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT balls, runs, dismissals, economy, strike_rate_against, dot_ball_pct, sample_size_confidence
            FROM bowler_vs_batting_hand_summary
            WHERE bowler_key = ? AND batting_hand = ?
            """,
            (bowler_key, batting_hand),
        ).fetchone()
    return dict(row) if row else None


def _bowler_bucket_and_label_for_matchups(meta: dict[str, Any]) -> tuple[str, str]:
    """Map registry/history metadata to ``batter_vs_bowling_type_summary`` bucket + display label."""
    raw = str(meta.get("bowling_style_raw") or "").strip().lower()
    btb = str(meta.get("bowling_type_bucket") or "").strip().lower()
    if "mystery" in raw:
        return "mystery_spin", "Mystery spin"
    if ("right" in raw and ("fast medium" in raw or "medium fast" in raw)) or (
        "left" in raw and ("fast medium" in raw or "medium fast" in raw)
    ):
        return "pace", "Pace (fast-medium)"
    if "right" in raw and "fast" in raw:
        return "pace", "Right-arm pace"
    if "left" in raw and "fast" in raw:
        return "pace", "Left-arm pace"
    if "offbreak" in raw or "off break" in raw:
        return "finger_spin", "Off / finger spin"
    if any(x in raw for x in ("legbreak", "leg break", "googly", "chinaman", "left-arm wrist")):
        return "wrist_spin", "Wrist spin"
    if any(x in raw for x in ("left-arm orthodox", "slow left-arm orthodox")):
        return "left_arm_orthodox", "Left-arm orthodox"
    if btb in ("pace", "finger_spin", "wrist_spin", "left_arm_orthodox", "mystery_spin"):
        labels = {
            "pace": "Pace / seam",
            "finger_spin": "Finger spin",
            "wrist_spin": "Wrist spin",
            "left_arm_orthodox": "Left-arm orthodox",
            "mystery_spin": "Mystery spin",
        }
        return btb, labels.get(btb, btb.replace("_", " "))
    if btb == "unknown" and "spin" in raw:
        return "finger_spin", "Spin"
    return "unknown", "Bowling type (unknown)"


def _batter_archetype_phrase(meta: dict[str, Any]) -> str:
    hand = str(meta.get("batting_hand") or "").strip().lower()
    band = str(meta.get("likely_batting_band") or "").strip().lower()
    band_aliases = {"anchor": "middle_order", "aggressor": "middle_order"}
    band = band_aliases.get(band, band)
    hand_l = "Right-hand" if hand == "right" else ("Left-hand" if hand == "left" else "")
    parts = [p for p in (hand_l, band.replace("_", " ") if band and band != "unknown" else "") if p]
    return " · ".join(parts) if parts else "Profile (limited metadata)"


def _matchup_insight_tags_direct(direct: Optional[dict[str, Any]]) -> list[str]:
    if not direct:
        return ["small-sample"]
    balls = int(direct.get("balls") or 0)
    conf = float(direct.get("sample_size_confidence") or 0.0)
    sr = float(direct.get("strike_rate") or 0.0)
    dis = int(direct.get("dismissals") or 0)
    tags: list[str] = []
    if balls < 18 or conf < 0.22:
        tags.append("small-sample")
    if balls >= 12 and dis >= 2 and sr <= 118:
        tags.append("caution")
    elif balls >= 12 and sr >= 142 and dis <= 1:
        tags.append("favorable")
    elif balls >= 10 and 118 < sr < 142 and dis <= 1:
        tags.append("neutral")
    if balls >= 20 and dis >= 3 and sr < 110:
        tags.append("high-risk")
    if not tags:
        tags.append("neutral")
    # De-duplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _tag_badge_html(tags: list[str]) -> str:
    colors = {
        "favorable": "#065f46",
        "neutral": "#4b5563",
        "caution": "#b45309",
        "high-risk": "#991b1b",
        "small-sample": "#6b21a8",
    }
    parts = []
    for t in tags:
        c = colors.get(t, "#374151")
        parts.append(
            f"<span style='display:inline-block;margin:2px 6px 0 0;padding:2px 8px;"
            f"border-radius:999px;font-size:12px;font-weight:600;color:{c};"
            f"background:{c}12;border:1px solid {c}44;'>{t.replace('-', ' ')}</span>"
        )
    return "".join(parts)


def _recommendation_from_matchup(snapshot: dict[str, Any], *, mode: str) -> str:
    conf = float(snapshot.get("sample_size_confidence") or 0.0)
    if mode == "direct":
        if conf >= 0.45 and float(snapshot.get("dismissals") or 0.0) >= 2 and float(snapshot.get("balls") or 0.0) >= 18:
            return "Bowler has a meaningful historical edge."
        if conf >= 0.45 and float(snapshot.get("strike_rate") or 0.0) >= 145:
            return "Batter has scored freely in this matchup."
        if conf < 0.2:
            return "Very small sample — lean on the fallback profile."
        return "Mixed direct history — treat as balanced."
    if float(snapshot.get("dot_ball_pct") or 0.0) >= 0.42:
        return "Containment profile looks strong."
    if float(snapshot.get("strike_rate") or snapshot.get("strike_rate_against") or 0.0) >= 145:
        return "Attacking profile looks favorable."
    return "Fallback profile is fairly neutral."


def _matchup_player_key_candidates(row: dict[str, Any]) -> list[str]:
    out: list[str] = []

    def add(raw: str) -> None:
        k = str(learner.normalize_player_key(raw) or "").strip().lower()[:80]
        if k and k not in out:
            out.append(k)

    add(str(row.get("player_key") or ""))
    add(str(row.get("canonical_player_key") or ""))
    add(str(row.get("name") or ""))
    pk = str(learner.normalize_player_key(str(row.get("name") or "")) or "").strip()
    if pk:
        try:
            from player_alias_resolve import _alias_override_candidates

            for c in _alias_override_candidates(pk):
                add(str(c))
        except Exception:
            pass
    return out


def _bowler_phase_subtitle(meta: dict[str, Any]) -> str:
    raw = str(meta.get("likely_bowling_phases") or "").lower()
    bits: list[str] = []
    if "death" in raw:
        bits.append("death overs")
    if "powerplay" in raw or "power" in raw:
        bits.append("powerplay")
    if "middle" in raw:
        bits.append("middle overs")
    if bits:
        return " · ".join(bits) + " usage"
    return ""


def _cricket_batter_hand_label(meta: dict[str, Any]) -> str:
    h = str(meta.get("batting_hand") or "").strip().lower()
    if h == "left":
        return "Left-hand batter"
    if h == "right":
        return "Right-hand batter"
    return "Batter"


def _cricket_batter_slot_label(meta: dict[str, Any]) -> str:
    band = str(meta.get("likely_batting_band") or "").strip().lower()
    aliases = {"anchor": "middle order", "aggressor": "middle order", "top_order": "top order"}
    band = aliases.get(band, band.replace("_", " "))
    if band and band not in ("unknown", "tail", ""):
        return f"{band.title()} batter"
    return "Batting profile"


def _cricket_bowler_style_label(meta: dict[str, Any]) -> str:
    _, lab = _bowler_bucket_and_label_for_matchups(meta)
    return lab


def _sample_quality_label(direct: Optional[dict[str, Any]]) -> str:
    if not direct:
        return "Small sample"
    balls = int(direct.get("balls") or 0)
    conf = float(direct.get("sample_size_confidence") or 0.0)
    if balls >= 36 and conf >= 0.35:
        return "Strong sample"
    if balls >= 18 and conf >= 0.22:
        return "Moderate sample"
    return "Small sample"


def _insight_batter_vs_type(stats: dict[str, Any]) -> str:
    sr = float(stats.get("strike_rate") or 0.0)
    dis = int(float(stats.get("dismissals") or 0))
    balls = int(stats.get("balls") or 0)
    dotp = float(stats.get("dot_ball_pct") or 0.0)
    if balls < 12:
        return "Limited balls in this bucket — read the trend lightly."
    if sr >= 145 and dis <= 1:
        return "Scores freely against this bowling type in the sample."
    if sr <= 115 and dis >= 2:
        return "Generally tied down with meaningful dismissal pressure in the sample."
    if dotp >= 0.42:
        return "Dot-ball pressure shows up often against this type."
    if sr >= 130:
        return "Positive scoring tempo against this type."
    return "Balanced numbers against this bowling type in the sample."


def _insight_bowler_vs_profile(stats: dict[str, Any]) -> str:
    econ = float(stats.get("economy") or 0.0)
    sra = float(stats.get("strike_rate_against") or 0.0)
    balls = int(stats.get("balls") or 0)
    if balls < 15:
        return "Thin slice for this batter profile — treat as indicative only."
    if econ <= 7.5 and sra <= 125:
        return "Tight lines and control against this batter shape."
    if sra >= 140 or econ >= 9.0:
        return "Leaks runs against this batter profile in the sample."
    return "Fairly even contest against this batter profile."


def _badge_row_html(labels: list[tuple[str, str]]) -> str:
    """labels: (text, color_hex)"""
    parts = []
    for text, col in labels:
        parts.append(
            f"<span style='display:inline-block;margin:4px 8px 0 0;padding:3px 10px;border-radius:4px;"
            f"font-size:11px;font-weight:700;letter-spacing:0.04em;color:{col};"
            f"background:{col}18;border:1px solid {col}55;'>{text}</span>"
        )
    return "<div style='margin:8px 0 12px 0;'>" + "".join(parts) + "</div>"


def _h2h_table_html(headers: list[str], rows: list[list[str]], *, bold_col: int = 2) -> str:
    th = "".join(
        f"<th style='text-align:left;padding:8px 12px;font-size:12px;text-transform:uppercase;"
        f"letter-spacing:0.06em;color:#6b7280;border-bottom:2px solid #e5e7eb'>{h}</th>"
        for h in headers
    )
    trs: list[str] = []
    for r in rows:
        tds = []
        for i, cell in enumerate(r):
            base = "padding:10px 12px;font-size:14px;border-bottom:1px solid #f3f4f6;color:#374151"
            if i == bold_col:
                tds.append(f"<td style='{base}'><b style='color:#111827'>{cell}</b></td>")
            else:
                tds.append(f"<td style='{base}'>{cell}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<table style='width:100%;border-collapse:collapse;background:#fafafa;border-radius:8px;"
        "overflow:hidden;margin-top:6px;'>"
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    )


def _direct_insight_tags_for_badges(direct: Optional[dict[str, Any]]) -> list[tuple[str, str]]:
    tags = _matchup_insight_tags_direct(direct)
    cmap = {
        "favorable": ("FAVORABLE", "#059669"),
        "neutral": ("NEUTRAL", "#6b7280"),
        "caution": ("CAUTION", "#d97706"),
        "high-risk": ("HIGH RISK", "#dc2626"),
        "small-sample": ("SMALL SAMPLE", "#64748b"),
    }
    out: list[tuple[str, str]] = []
    for t in tags:
        if t in cmap:
            out.append(cmap[t])
    return out


def _render_matchup_player_vs_player_polished(
    left_row: dict[str, Any],
    right_row: dict[str, Any],
) -> dict[str, Any]:
    """Cricbuzz-style player matchup panel; returns debug fields for Matchup Debug expander."""
    dbg: dict[str, Any] = {
        "direct_row_count": 0,
        "batter_type_fallback_row_count": 0,
        "bowler_profile_fallback_row_count": 0,
        "normalized_batter_keys_tried": [],
        "normalized_bowler_keys_tried": [],
        "visible_block": "",
        "notes": [],
    }
    left_name = str(left_row.get("name") or "Player A")
    right_name = str(right_row.get("name") or "Player B")
    left_role = str(left_row.get("role_bucket") or "")
    right_role = str(right_row.get("role_bucket") or "")
    left_is_bowler = left_role in (ipl_squad.BOWLER, ipl_squad.ALL_ROUNDER)
    right_is_bowler = right_role in (ipl_squad.BOWLER, ipl_squad.ALL_ROUNDER)

    batter_row = left_row
    bowler_row = right_row
    if left_is_bowler and not right_is_bowler:
        batter_row, bowler_row = right_row, left_row

    batter_keys = _matchup_player_key_candidates(batter_row)
    bowler_keys = _matchup_player_key_candidates(bowler_row)
    dbg["normalized_batter_keys_tried"] = list(batter_keys)
    dbg["normalized_bowler_keys_tried"] = list(bowler_keys)
    dbg["batter_display"] = str(batter_row.get("name") or "")
    dbg["bowler_display"] = str(bowler_row.get("name") or "")

    batter_meta = batter_row.get("_meta") if isinstance(batter_row.get("_meta"), dict) else {}
    bowler_meta = bowler_row.get("_meta") if isinstance(bowler_row.get("_meta"), dict) else {}
    batting_hand = str(batter_meta.get("batting_hand") or "").strip().lower()
    bowling_type_bucket_row = str(bowler_meta.get("bowling_type_bucket") or "").strip().lower()
    sum_bucket, bowl_archetype_label = _bowler_bucket_and_label_for_matchups(bowler_meta)

    direct, pair_used = db.fetch_batter_bowler_direct_matchup_candidates(batter_keys, bowler_keys)
    if direct:
        dbg["direct_row_count"] = 1
        dbg["direct_pair_matched"] = pair_used
    else:
        dbg["notes"].append(
            "No batter_bowler_matchup_summary row with balls>0 for any candidate key cross-product."
        )
    balls_d = int(direct.get("balls") or 0) if direct else 0
    conf_d = float(direct.get("sample_size_confidence") or 0.0) if direct else 0.0
    thin_direct = direct is None or balls_d < 14 or conf_d < 0.22

    type_fb = None
    type_bucket_used = sum_bucket
    if thin_direct or not direct:
        type_fb = db.fetch_batter_vs_bowling_type_first_hit(batter_keys, sum_bucket)
        if not type_fb and bowling_type_bucket_row in (
            "pace",
            "finger_spin",
            "wrist_spin",
            "left_arm_orthodox",
            "mystery_spin",
        ):
            type_fb = db.fetch_batter_vs_bowling_type_first_hit(batter_keys, bowling_type_bucket_row)
            type_bucket_used = bowling_type_bucket_row
        if not type_fb and sum_bucket == "unknown":
            for fallb in ("pace", "finger_spin", "wrist_spin", "left_arm_orthodox", "mystery_spin"):
                type_fb = db.fetch_batter_vs_bowling_type_first_hit(batter_keys, fallb)
                if type_fb:
                    type_bucket_used = fallb
                    dbg["notes"].append(f"Type fallback used generic bucket '{fallb}' after unknown style.")
                    break
        if type_fb:
            dbg["batter_type_fallback_row_count"] = 1
            dbg["batter_type_bucket"] = type_bucket_used

    b_hand = batting_hand if batting_hand in ("right", "left") else ""
    b_band = str(batter_meta.get("likely_batting_band") or "").strip().lower()
    b_band = {"anchor": "middle_order", "aggressor": "middle_order"}.get(b_band, b_band)
    if b_band in ("unknown", "tail", ""):
        b_band = ""

    profile_fb = db.fetch_bowler_vs_batter_archetype_aggregate_multi(
        bowler_keys,
        batting_hand=b_hand,
        likely_batting_band=b_band,
    )
    if not profile_fb and b_hand and b_band:
        profile_fb = db.fetch_bowler_vs_batter_archetype_aggregate_multi(
            bowler_keys, batting_hand=b_hand
        )
    if not profile_fb and b_band:
        profile_fb = db.fetch_bowler_vs_batter_archetype_aggregate_multi(
            bowler_keys, likely_batting_band=b_band
        )
    hand_fb = None
    if not profile_fb and b_hand:
        hand_fb = db.fetch_bowler_vs_batting_hand_first_hit(bowler_keys, b_hand)
    if profile_fb:
        dbg["bowler_profile_fallback_row_count"] = 1
    elif hand_fb:
        dbg["bowler_profile_fallback_row_count"] = 1
        dbg["notes"].append("Used batting-hand summary (profile slice sparse).")

    st.markdown(
        f"<div style='font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-0.02em;'>"
        f"{left_name} <span style='color:#94a3b8;font-weight:600;'>vs</span> {right_name}</div>",
        unsafe_allow_html=True,
    )
    sub1 = f"{_cricket_batter_hand_label(batter_meta)} · {_cricket_batter_slot_label(batter_meta)}"
    sub2 = f"{_cricket_bowler_style_label(bowler_meta)}"
    ph = _bowler_phase_subtitle(bowler_meta)
    if ph:
        sub2 = f"{sub2} · {ph}"
    st.caption(f"{sub1}  \n{sub2}")

    insight_badges = _direct_insight_tags_for_badges(direct)
    if insight_badges:
        st.markdown(_badge_row_html(insight_badges), unsafe_allow_html=True)

    # --- Direct card
    if direct and not thin_direct:
        dbg["visible_block"] = "direct_primary"
        sq = _sample_quality_label(direct)
        st.markdown("##### Head-to-head")
        st.caption(sq)
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Balls", int(direct.get("balls") or 0))
        b2.metric("Runs", int(direct.get("runs") or 0))
        b3.metric("Dismissals", int(direct.get("dismissals") or 0))
        b4.metric("Strike rate", f"{float(direct.get('strike_rate') or 0.0):.1f}")
        b5.metric("Dot ball %", f"{100 * float(direct.get('dot_ball_pct') or 0.0):.1f}%")
        b6.metric("Boundary %", f"{100 * float(direct.get('boundary_pct') or 0.0):.1f}%")
        st.caption(
            f"Matches **{int(direct.get('match_count') or 0)}** · Innings **{int(direct.get('innings_count') or 0)}** · "
            f"Last matchup **{direct.get('last_match_date') or '—'}**"
        )
    elif direct and thin_direct:
        dbg["visible_block"] = "direct_thin_plus_fallback"
        st.markdown("##### Head-to-head")
        st.caption(_sample_quality_label(direct))
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Balls", int(direct.get("balls") or 0))
        b2.metric("Runs", int(direct.get("runs") or 0))
        b3.metric("Dismissals", int(direct.get("dismissals") or 0))
        b4.metric("Strike rate", f"{float(direct.get('strike_rate') or 0.0):.1f}")
        b5.metric("Dot ball %", f"{100 * float(direct.get('dot_ball_pct') or 0.0):.1f}%")
        b6.metric("Boundary %", f"{100 * float(direct.get('boundary_pct') or 0.0):.1f}%")
        st.info("Limited direct sample — type-based trends below add context.")
    else:
        dbg["visible_block"] = "fallback_only"
        st.info("No direct player-vs-player sample found. Showing closest type-based trends.")

    # --- Batter fallback card
    if type_fb and (thin_direct or not direct):
        dbg["visible_block"] = (dbg.get("visible_block") or "") + "+batter_type"
        _btitle = {
            "pace": "Pace bowling",
            "finger_spin": "Finger spin",
            "wrist_spin": "Wrist spin",
            "left_arm_orthodox": "Left-arm orthodox",
            "mystery_spin": "Mystery spin",
        }.get(str(type_bucket_used or ""), bowl_archetype_label)
        st.markdown(f"##### {batter_row.get('name')} vs {_btitle}")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Balls", int(type_fb.get("balls") or 0))
        t2.metric("Runs", int(type_fb.get("runs") or 0))
        t3.metric("Dismissals", int(float(type_fb.get("dismissals") or 0)))
        t4.metric("Strike rate", f"{float(type_fb.get('strike_rate') or 0.0):.1f}")
        t5.metric("Dot ball %", f"{100 * float(type_fb.get('dot_ball_pct') or 0.0):.1f}%")
        t6.metric("Boundary %", f"{100 * float(type_fb.get('boundary_pct') or 0.0):.1f}%")
        conf_t = float(type_fb.get("sample_size_confidence") or 0.0)
        st.caption(f"Confidence **{100 * conf_t:.0f}%** · Aggregated vs **{_btitle}** bowlers")
        st.caption(_insight_batter_vs_type(type_fb))
    elif thin_direct or not direct:
        dbg["notes"].append("No batter-vs-type bucket row for candidate keys.")

    # --- Bowler-view card
    bow_stats = profile_fb or hand_fb
    if bow_stats:
        hand_lbl = "left-hand" if batting_hand == "left" else ("right-hand" if batting_hand == "right" else "this hand")
        slot_lbl = _cricket_batter_slot_label(batter_meta).lower()
        title = f"{bowler_row.get('name')} vs {hand_lbl} {slot_lbl} batters"
        if hand_fb and not profile_fb:
            title = f"{bowler_row.get('name')} vs {hand_lbl.title()} batters"
        st.markdown(f"##### {title}")
        if profile_fb:
            p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
            p1.metric("Balls", int(profile_fb.get("balls") or 0))
            p2.metric("Runs conceded", int(profile_fb.get("runs") or 0))
            p3.metric("Dismissals", int(profile_fb.get("dismissals") or 0))
            p4.metric("Economy", f"{float(profile_fb.get('economy') or 0.0):.2f}")
            p5.metric("SR conceded", f"{float(profile_fb.get('strike_rate_against') or 0.0):.1f}")
            p6.metric("Dot ball %", f"{100 * float(profile_fb.get('dot_ball_pct') or 0.0):.1f}%")
            p7.metric("Boundary %", f"{100 * float(profile_fb.get('boundary_pct') or 0.0):.1f}%")
        else:
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Balls", int(hand_fb.get("balls") or 0))
            p2.metric("Runs conceded", int(hand_fb.get("runs") or 0))
            p3.metric("Dismissals", int(hand_fb.get("dismissals") or 0))
            p4.metric("Economy", f"{float(hand_fb.get('economy') or 0.0):.2f}")
            p5.metric("SR conceded", f"{float(hand_fb.get('strike_rate_against') or 0.0):.1f}")
            p6.metric("Dot ball %", f"{100 * float(hand_fb.get('dot_ball_pct') or 0.0):.1f}%")
        st.caption(_insight_bowler_vs_profile(bow_stats))
    else:
        dbg["notes"].append("No bowler-vs-profile or hand aggregate.")

    if not direct and not type_fb and not bow_stats:
        st.warning(
            "No matchup sample found in local DB for the selected players or their closest type-based fallback."
        )
        dbg["visible_block"] = "empty"

    return dbg


def render_stored_prediction_results(
    r: dict[str, Any],
    *,
    show_advanced_prediction_debug: bool,
    selection_debug_top15_for_side: Callable[..., Any],
) -> None:
    _t_render = time.perf_counter()
    cond = r["conditions"]
    disp_a = _team_name_from_stored_result(r, "team_a") or "Team A"
    disp_b = _team_name_from_stored_result(r, "team_b") or "Team B"

    st.divider()
    st.subheader("Weather & venue")
    w = r["weather"]
    if w.get("ok"):
        tz_note = w.get("timezone_note", "IST")
        st.success(
            f"**{w['hour_iso']}** ({tz_note}) — {w['temperature_c']:.1f}°C, "
            f"humidity {w['relative_humidity_pct']:.0f}%, "
            f"rain prob {w['precipitation_probability_pct']:.0f}%, "
            f"precip {w['precipitation_mm']:.2f} mm, "
            f"cloud {w['cloud_cover_pct']:.0f}%, wind {w['wind_kmh']:.1f} km/h."
        )
    else:
        st.warning(f"Weather fallback (API issue): {w.get('error')}")

    st.markdown(
        f"**Venue model:** {cond['venue']}  \n"
        f"- Batting friendliness: {cond['batting_friendliness']:.2f}  \n"
        f"- Pace bias: {cond['pace_bias']:.2f} · Spin friendliness: {cond['spin_friendliness']:.2f}  \n"
        f"- Effective dew risk: {cond['dew_risk']:.2f} · Swing/seam proxy: {cond['swing_seam_proxy']:.2f}  \n"
        f"- Rain disruption risk: {cond['rain_disruption_risk']:.2f}  \n"
        f"_Notes:_ {cond['notes']}"
    )

    wp = r["win_probability"]
    eng = r.get("win_probability_engine") or {}
    st.subheader("Win probability")
    team_a_win = float(wp.get("team_a_win") or 0.0)
    team_b_win = max(0.0, 1.0 - team_a_win)
    wp1, wp2 = st.columns(2)
    with wp1:
        st.metric(disp_a, f"{100 * team_a_win:.1f}%")
    with wp2:
        st.metric(disp_b, f"{100 * team_b_win:.1f}%")
    st.progress(int(round(team_a_win * 100)))
    if show_advanced_prediction_debug:
        with st.expander("Squad, history joins, XI validation & toss debug"):
            _t_sq = time.perf_counter()
            sd = r.get("squad_debug") or {}
            xv2 = r.get("xi_validation") or {}
            heavy_squad_dbg = st.checkbox(
                "Render large JSON / full squad tables (slower)",
                value=False,
                key="squad_history_debug_heavy",
            )
            st.markdown("**Canonical franchise (history join key)**")
            st.write(
                {
                    "team_a": sd.get("team_a_canonical_franchise"),
                    "team_b": sd.get("team_b_canonical_franchise"),
                }
            )
            hsd = r.get("history_sync_debug") or {}
            if hsd:
                st.markdown("**SQLite history (read-only during prediction)**")
                st.caption(
                    f"**Run prediction** does not fetch history from the internet or ingest Cricsheet JSON. "
                    f"{history_sync.HISTORY_MISSING_USER_MESSAGE} "
                    "Official IPL pages are used **only** for the current squad list."
                )
                if hsd.get("sync_exception"):
                    st.error(f"Local history read error (prediction continued): {hsd.get('sync_exception')}")
                for side in ("team_a", "team_b"):
                    block = hsd.get(side)
                    if block:
                        st.write(f"**{side}**")
                        if heavy_squad_dbg:
                            st.json(block)
                        else:
                            st.caption(
                                f"History sync block: **{len(block)}** top-level keys — enable checkbox above for JSON."
                            )
            st.markdown("**Role bucket counts (fetched structured squad)**")
            c1, c2 = st.columns(2)
            with c1:
                if heavy_squad_dbg:
                    st.json(sd.get("team_a_role_bucket_counts") or {})
                else:
                    st.caption("Team A role counts (enable large JSON to show).")
            with c2:
                if heavy_squad_dbg:
                    st.json(sd.get("team_b_role_bucket_counts") or {})
                else:
                    st.caption("Team B role counts (enable large JSON to show).")
            st.markdown("**Structured squad (name, player_key, role_bucket, overseas)**")
            sqa = sd.get("structured_squad_team_a") or []
            sqb = sd.get("structured_squad_team_b") or []
            if sqa:
                st.caption(disp_a)
                st.dataframe(pd.DataFrame(sqa), width="stretch", hide_index=True)
            if sqb:
                st.caption(disp_b)
                st.dataframe(pd.DataFrame(sqb), width="stretch", hide_index=True)
            st.markdown(
                "**XI validation (strict: XI ⊆ fetched squad; batting order ⊆ XI; history enriches squad only)**"
            )
            if heavy_squad_dbg:
                st.json(xv2)
            else:
                st.caption(
                    f"XI validation payload: **{len(xv2)}** top-level keys — enable checkbox above for full JSON."
                )
            st.markdown("**Toss / overseas / chase / batting-order summary**")
            st.write(
                {
                    "toss_scenario_key": ts.get("key"),
                    "team_a_bats_first_resolved": ts.get("team_a_bats_first"),
                    "chase_context_team_a": ts.get("chase_context_team_a"),
                    "chase_context_team_b": ts.get("chase_context_team_b"),
                    "overseas_counts": r.get("overseas_counts"),
                    "batting_order_summary": r.get("batting_order_summary"),
                    "chase_defend_bias": r.get("chase_defend_bias"),
                    "engine_toss_label": eng.get("toss_scenario_used"),
                }
            )
            if getattr(config, "PREDICTION_TIMING_LOG", False):
                _perf_logger.info(
                    "app_ui_squad_history_debug_expander_ms=%.2f",
                    (time.perf_counter() - _t_sq) * 1000.0,
                )

        pld = r.get("prediction_layer_debug") or {}
        if pld:
            with st.expander("Prediction layer: history usage, scoring, batting order, impact (SQLite)"):
                _t_pld = time.perf_counter()
                if any(
                    (pld.get(s) or {}).get("_light_debug_omitted_large_lists")
                    for s in ("team_a", "team_b")
                ):
                    st.caption(
                        "Large lists (full history usage, full impact bench ranking) were omitted in the payload "
                        "for speed. Set env **IPL_PREDICTION_FULL_DEBUG=true** and re-run prediction to restore them."
                    )
                st.caption(
                    "Per-player joins against ``team_match_xi``, ``player_match_stats``, and head-to-head "
                    "fixtures already stored in SQLite after ingest."
                )
                heavy_pld = st.checkbox(
                    "Render full prediction-layer tables (history usage, impact ranking, etc.)",
                    value=False,
                    key="prediction_layer_debug_heavy_tables",
                )
                for side, title in (("team_a", disp_a), ("team_b", disp_b)):
                    block = pld.get(side) or {}
                    if not block:
                        continue
                    st.markdown(f"**{title} ({side})**")
                    hu = block.get("history_usage_per_player") or []
                    if hu:
                        st.markdown("*History usage (all fetched squad players)*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(hu), width="stretch", hide_index=True)
                        else:
                            st.caption(
                                f"**{len(hu)}** rows — enable “Render full prediction-layer tables” above."
                            )
                    sc = block.get("scoring_breakdown_per_player") or []
                    if sc:
                        st.markdown("*Scoring breakdown (scored squad — history vs composite vs selection)*")
                        st.dataframe(pd.DataFrame(sc), width="stretch", hide_index=True)
                    omit = block.get("omitted_from_playing_xi") or []
                    if omit:
                        st.markdown("*High composite / selection rank — not in playing XI*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(omit), width="stretch", hide_index=True)
                        else:
                            st.caption(f"**{len(omit)}** rows — enable full tables above.")
                    bod = block.get("xi_batting_order_diagnostics") or []
                    if bod:
                        st.markdown("*Selected XI — batting-order diagnostics*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(bod), width="stretch", hide_index=True)
                        else:
                            st.caption(f"**{len(bod)}** rows — enable full tables above.")
                    imp = block.get("impact_sub_ranking") or []
                    if imp:
                        st.markdown("*Impact sub ranking (model components)*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(imp), width="stretch", hide_index=True)
                        else:
                            st.caption(f"**{len(imp)}** rows — enable full tables above.")
                    st.divider()
                if getattr(config, "PREDICTION_TIMING_LOG", False):
                    _perf_logger.info(
                        "app_ui_prediction_layer_debug_expander_ms=%.2f",
                        (time.perf_counter() - _t_pld) * 1000.0,
                    )

        with st.expander("Selection debug (temporary — top 15 squad by selection score)", expanded=False):
            st.caption(
                "One team at a time. **Impact candidate** = in the model’s top-5 impact sub list for that side. "
                "PP/death bowler counts on the XI use derive likelihood ≥ 0.45 (see note under metrics)."
            )
            dbg_side_pick = st.radio(
                "Team",
                ("team_a", "team_b"),
                format_func=lambda s: f"{disp_a} (A)" if s == "team_a" else f"{disp_b} (B)",
                horizontal=True,
                key="selection_debug_side",
            )
            df_dbg, xi_val = selection_debug_top15_for_side(r, dbg_side_pick)
            if df_dbg.empty:
                st.info("No scoring breakdown rows — run prediction after a full squad load.")
            else:
                st.dataframe(df_dbg, width="stretch", hide_index=True)
            if xi_val:
                st.markdown("**XI validation (selected playing XI)**")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Wicketkeepers", xi_val.get("wicketkeeper_count", "—"))
                a2.metric("Bowling options", xi_val.get("bowling_options_count", "—"))
                a3.metric("Powerplay bowlers (derive ≥0.45)", xi_val.get("powerplay_bowlers_count", "—"))
                a4.metric("Death bowlers (derive ≥0.45)", xi_val.get("death_bowlers_count", "—"))
                b1, b2, b3 = st.columns(3)
                b1.metric("Opener candidates", xi_val.get("opener_candidates_count", "—"))
                b2.metric("Finisher candidates", xi_val.get("finisher_candidates_count", "—"))
                b3.metric("Overseas in XI", xi_val.get("overseas_count", "—"))
                note = xi_val.get("powerplay_death_threshold_note")
                if note:
                    st.caption(str(note))
            else:
                st.caption("XI validation metrics missing — click **Run prediction** again to refresh payload.")

    venue_keys_bo = list(((r.get("learning_context") or {}).get("venue_keys_tried") or []))
    team_a_rows, squad_map_a = _build_squad_row_maps(r, "team_a")
    team_b_rows, squad_map_b = _build_squad_row_maps(r, "team_b")
    state = _coach_state(r)

    st.subheader("Predicted line-ups & impact subs")
    tc1, tc2 = st.columns(2)
    for col, side_key, tname in (
        (tc1, "team_a", disp_a),
        (tc2, "team_b", disp_b),
    ):
        with col:
            slug = ipl_teams.slug_for_canonical_label(r[side_key]["name"])
            logo_path = ipl_teams.team_logo_path_for_slug(slug) if slug else ""
            hc1, hc2 = st.columns([1, 9])
            with hc1:
                if logo_path:
                    st.image(logo_path, width=34)
            with hc2:
                st.markdown(f"### {r[side_key]['name']}")
            st.caption(
                f"Squad overseas: **{r[side_key].get('squad_overseas', '—')}** · "
                f"XI overseas: **{r[side_key].get('xi_overseas', '—')}** / 4 max"
            )
            if not r[side_key]["valid_xi"]:
                errs = r[side_key].get("xi_constraint_errors") or []
                detail = " ".join(errs) if errs else "See squad role buckets (Batter / WK-Batter / All-Rounder / Bowler)."
                st.error(
                    f"Could not satisfy XI rules (11 players, overseas cap, WK, bowling depth, "
                    f"batting balance, opener/finisher pool). {detail}"
                )
            squad_map_side = squad_map_a if side_key == "team_a" else squad_map_b
            bo_disp = _resolve_batting_order_for_display(state, side_key)
            xi_rows_live = _xi_rows_for_display(bo_disp, squad_map_side)
            xi_df = pd.DataFrame(xi_rows_live)
            if state.get(f"{side_key}_finalised"):
                st.success("**Finalise XI** — this side is frozen for the session.")
            pred_xi = list(state.get(f"{side_key}_predicted_xi") or [])
            eff_xi = _effective_xi_names(state, side_key)
            if pred_xi and eff_xi and pred_xi != eff_xi:
                st.caption("Session XI differs from model prediction (predicted list preserved in session).")
            if not xi_df.empty:
                display_cols = [
                    c
                    for c in (
                        "name",
                        "bat_order",
                        "role",
                        "role_bucket",
                        "allrounder_subtype",
                        "batting_band",
                        "final_position",
                        "bowling_type",
                        "overseas",
                        "designated_keeper",
                        "is_wk_role",
                        "selection_score",
                        "marquee_tier",
                        "batting_order_final",
                    )
                    if c in xi_df.columns
                ]
                xi_display_df = xi_df[display_cols] if display_cols else xi_df
                st.dataframe(xi_display_df, width="stretch", hide_index=True)
            else:
                st.caption("No XI rows to display — complete the squad and re-run prediction.")
            st.markdown("**Impact subs**")
            imp_rows = _impact_rows_for_display(
                _effective_impact_names(state, side_key),
                squad_map_side,
                list(r[side_key].get("impact_subs") or []),
            )
            st.dataframe(
                pd.DataFrame(imp_rows) if imp_rows else pd.DataFrame(),
                width="stretch",
                hide_index=True,
            )

    final_names_a = _effective_xi_names(state, "team_a")
    final_names_b = _effective_xi_names(state, "team_b")

    def _final_rows(side_key: str, final_names: list[str], squad_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in final_names:
            row = squad_map.get(name)
            if row:
                rows.append(row)
        return rows

    matchup_debug: dict[str, Any] = {
        "batter_display": "—",
        "bowler_display": "—",
        "normalized_batter_keys_tried": "—",
        "normalized_bowler_keys_tried": "—",
        "disp_a": "—",
        "disp_b": "—",
        "canonical_disp_a": "—",
        "canonical_disp_b": "—",
        "fetch_match_results_meta_total": "—",
        "h2h_rows_count": "—",
        "first_5_extracted_row_team_pairs_before_filter": "—",
        "first_5_h2h_filtered_team_pairs": "—",
        "team_vs_team_meta_limit": "—",
        "team_vs_team_row_debug": "—",
        "player_matchup_competition_scope": (
            f"{db.PLAYER_MATCHUP_DB_COMPETITION_SCOPE} — {db.PLAYER_MATCHUP_DB_COMPETITION_SCOPE_DETAIL}"
        ),
    }

    st.subheader("Matchups")
    st.caption("Head-to-head and player matchup trends from your local IPL history database.")

    top_mode = st.radio(
        "Matchups",
        ("Team vs Team", "Player vs Player"),
        horizontal=True,
        key="matchups_top_mode",
    )
    matchup_debug["top_level_mode"] = top_mode

    if top_mode == "Team vs Team":
        # Wider window: prediction_summary_match_meta mixes competitions; 450 newest rows can miss IPL H2H.
        _meta_limit = 5000
        rows_all = db.fetch_match_results_meta(_meta_limit)
        first_5_pre = [tuple(h2h_history.row_team_names_pair(r)) for r in rows_all[:5]]
        h2h_rows = h2h_history.filter_match_rows_to_h2h(rows_all, disp_a, disp_b)
        h2h_rows = h2h_history.sort_h2h_rows_recent_first(h2h_rows)
        first_5_post = [tuple(h2h_history.row_team_names_pair(r)) for r in h2h_rows[:5]]

        for r in h2h_rows:
            r["_dbg_winner_raw_summary"] = str(r.get("winner") or "")
        _h2h_ids = [int(x.get("id") or x.get("match_id") or 0) for x in h2h_rows if int(x.get("id") or x.get("match_id") or 0) > 0]
        _enrich_map = db.fetch_match_result_display_enrichment_by_ids(_h2h_ids)
        h2h_history.enrich_h2h_summary_rows_from_match_results(h2h_rows, _enrich_map)

        matchup_debug["disp_a"] = disp_a
        matchup_debug["disp_b"] = disp_b
        matchup_debug["canonical_disp_a"] = ipl_teams.canonical_franchise_label(disp_a) or str(disp_a or "").strip()
        matchup_debug["canonical_disp_b"] = ipl_teams.canonical_franchise_label(disp_b) or str(disp_b or "").strip()
        matchup_debug["fetch_match_results_meta_total"] = len(rows_all)
        matchup_debug["h2h_rows_count"] = len(h2h_rows)
        matchup_debug["first_5_extracted_row_team_pairs_before_filter"] = first_5_pre
        matchup_debug["first_5_h2h_filtered_team_pairs"] = first_5_post
        matchup_debug["sample_h2h_team_pairs"] = [
            (str(r.get("team_a") or ""), str(r.get("team_b") or "")) for r in h2h_rows[:3]
        ]
        matchup_debug["sample_h2h_venues"] = [str(r.get("venue") or "") for r in h2h_rows[:3]]
        matchup_debug["team_vs_team_meta_limit"] = _meta_limit
        matchup_debug["team_sub_mode"] = "last_5_h2h"
        meetings_5 = h2h_rows[:5]
        summ_5 = h2h_history.summarize_h2h_fixture_rows(meetings_5, disp_a, disp_b)
        matchup_debug["meetings_5_count"] = len(meetings_5)
        matchup_debug["meetings_v_count"] = "—"
        matchup_debug["expanded_venue_tokens"] = "—"
        matchup_debug["data_source"] = (
            "fetch_match_results_meta(5000) → filter_match_rows_to_h2h → sort_h2h_rows_recent_first → [:5]"
        )
        st.markdown("##### Last 5 meetings")
        matchup_debug["team_vs_team_row_debug"] = [
            {
                "match_id": int(r.get("id") or r.get("match_id") or 0),
                "winner_raw": str(r.get("_dbg_winner_raw_summary") or ""),
                "winner_label": h2h_history.winner_display_for_row(r, disp_a, disp_b),
                "margin": str(r.get("margin") or "").strip() or "—",
                "result_text": str(r.get("result_text") or "").strip() or "—",
                "score_summary": h2h_history.h2h_score_summary_display(r),
            }
            for r in meetings_5
        ]
        if not meetings_5:
            st.info("No recent meetings found between these teams in the current history window.")
        else:
            s1, s2, s3 = st.columns(3)
            wa, wb = int(summ_5.get("wins_a") or 0), int(summ_5.get("wins_b") or 0)
            la3, lb3 = int(summ_5.get("last3_wins_a") or 0), int(summ_5.get("last3_wins_b") or 0)
            avg_fi = summ_5.get("avg_first_innings_runs")
            with s1:
                st.metric("Win split", f"{wa} – {wb}")
                st.caption(f"{summ_5.get('label_a')} · {summ_5.get('label_b')}")
            with s2:
                st.metric("Recent trend (last 3)", f"{la3} – {lb3}")
                st.caption("Last three fixtures")
            with s3:
                st.metric(
                    "Avg 1st inns score",
                    f"{float(avg_fi):.1f}" if avg_fi is not None else "—",
                )
                st.caption("Not in summary feed")
            tbl_rows: list[list[str]] = []
            for m in meetings_5:
                margin_cell = str(m.get("margin") or "").strip() or "—"
                tbl_rows.append(
                    [
                        str(m.get("match_date") or "—"),
                        str(m.get("venue") or "—"),
                        h2h_history.winner_display_for_row(m, disp_a, disp_b),
                        margin_cell,
                        h2h_history.h2h_score_summary_display(m),
                    ]
                )
            st.markdown(
                _h2h_table_html(
                    ["Date", "Venue", "Winner", "Margin", "Score summary"],
                    tbl_rows,
                    bold_col=2,
                ),
                unsafe_allow_html=True,
            )

    else:
        matchup_debug["team_sub_mode"] = "—"
        matchup_debug["disp_a"] = "—"
        matchup_debug["disp_b"] = "—"
        matchup_debug["canonical_disp_a"] = "—"
        matchup_debug["canonical_disp_b"] = "—"
        matchup_debug["fetch_match_results_meta_total"] = "—"
        matchup_debug["h2h_rows_count"] = "—"
        matchup_debug["first_5_extracted_row_team_pairs_before_filter"] = "—"
        matchup_debug["first_5_h2h_filtered_team_pairs"] = "—"
        matchup_debug["team_vs_team_meta_limit"] = "—"
        matchup_debug["team_vs_team_row_debug"] = "—"
        matchup_debug["meetings_5_count"] = "—"
        matchup_debug["meetings_v_count"] = "—"
        matchup_debug["sample_h2h_team_pairs"] = "—"
        matchup_debug["sample_h2h_venues"] = "—"
        matchup_debug["expanded_venue_tokens"] = "—"
        st.markdown("##### Player vs player")
        dm1, dm2 = st.columns(2)
        team_a_names = [str(row.get("name") or "") for row in team_a_rows]
        team_b_names = [str(row.get("name") or "") for row in team_b_rows]
        with dm1:
            pick_a = st.selectbox("Team A squad player", team_a_names, key="matchup_direct_a")
        with dm2:
            pick_b = st.selectbox("Team B squad player", team_b_names, key="matchup_direct_b")
        row_a = squad_map_a.get(pick_a)
        row_b = squad_map_b.get(pick_b)
        if row_a and row_b:
            pd_dbg = _render_matchup_player_vs_player_polished(row_a, row_b)
            matchup_debug.update(pd_dbg)
            matchup_debug["data_source"] = str(pd_dbg.get("visible_block") or "player_matchup")

    with st.expander("Matchup Debug"):
        st.caption("Temporary diagnostics for verifying DB wiring and key normalization.")
        lines = [f"{k}: {v}" for k, v in matchup_debug.items()]
        st.code("\n".join(lines) if lines else "(no debug payload)", language=None)

    st.subheader("Finalise Playing XI")

    def _render_finalise_team(
        *,
        side_key: str,
        team_name: str,
        all_rows: list[dict[str, Any]],
        squad_map: dict[str, dict[str, Any]],
    ) -> None:
        final_key = f"{side_key}_final_xi"
        impact_key = f"{side_key}_impact_names"
        bo_key = f"{side_key}_batting_order"
        finalised = bool(state.get(f"{side_key}_finalised"))
        current_final_names = list(state.get(final_key) or [])
        impact_names = list(state.get(impact_key) or [])
        current_rows = _final_rows(side_key, current_final_names, squad_map)
        bench = _bench_rows_exclusive(current_final_names, impact_names, all_rows)
        logo_html = _logo_html_for_team_name(team_name, width=22)
        st.markdown(f"{logo_html}<span style='font-size:20px;font-weight:700;'>{team_name}</span>", unsafe_allow_html=True)
        current_tags = sorted({label for row in current_rows for label, _ in _role_tag_row(row)})
        bench_tags = sorted({label for row in bench for label, _ in _role_tag_row(row)})
        st.caption(
            f"Editable XI: {len(current_rows)} players · Impact: {len(impact_names)} · Bench pool: {len(bench)}"
        )
        if current_final_names:
            st.caption(", ".join(current_final_names[:11]) + (" …" if len(current_final_names) > 11 else ""))
        if current_tags or bench_tags:
            st.caption(
                f"Current tags: {', '.join(current_tags[:8]) or '—'} · Bench tags: {', '.join(bench_tags[:8]) or '—'}"
            )

        if finalised:
            st.info("**Finalise XI** — edits frozen for this session. Use **Reset to Suggested XI** to unlock.")
            if st.button("Reset to Suggested XI", key=f"{side_key}_reset_final_xi_finalised"):
                px = [str(x.get("name") or "") for x in (r.get(side_key, {}).get("xi") or []) if x.get("name")]
                pi = [
                    str(x.get("name") or "")
                    for x in (r.get(side_key, {}).get("impact_subs") or [])
                    if x.get("name")
                ][:5]
                pbo = list(r.get(side_key, {}).get("batting_order") or [])
                state[final_key] = list(px)
                state[impact_key] = list(pi)
                state[bo_key] = list(pbo)
                state[f"{side_key}_finalised"] = False
                state[f"{side_key}_finalised_xi"] = None
                state[f"{side_key}_finalised_batting_order"] = None
                state[f"{side_key}_finalised_impact_names"] = None
                st.session_state["coach_tools_state"] = state
                st.rerun()
            _, errs = _validate_manual_xi(team_name, squad_map, _effective_xi_names(state, side_key), cond)
            if errs:
                for err in errs:
                    st.error(err)
            else:
                st.success("Final XI is valid.")
            return

        swap_mode = st.radio(
            "Swap type",
            ("XI ↔ bench", "XI ↔ impact", "Impact ↔ bench"),
            horizontal=True,
            key=f"{side_key}_swap_mode",
        )
        xi_list = list(current_final_names)
        bench_names = [str(row.get("name") or "") for row in bench]
        imp_list = [n for n in impact_names if n]

        xi_pick: Optional[str] = None
        bench_pick: Optional[str] = None
        impact_pick: Optional[str] = None

        if swap_mode == "XI ↔ bench":
            c1, c2 = st.columns(2)
            with c1:
                xi_pick = st.selectbox("XI player", xi_list or [""], key=f"{side_key}_swap_xi_b_xi")
            with c2:
                bench_pick = st.selectbox("Bench player", bench_names or [""], key=f"{side_key}_swap_xi_b_bench")
        elif swap_mode == "XI ↔ impact":
            c1, c2 = st.columns(2)
            with c1:
                xi_pick = st.selectbox("XI player", xi_list or [""], key=f"{side_key}_swap_xi_i_xi")
            with c2:
                impact_pick = st.selectbox("Impact player", imp_list or [""], key=f"{side_key}_swap_xi_i_imp")
        else:
            c1, c2 = st.columns(2)
            with c1:
                impact_pick = st.selectbox("Impact player", imp_list or [""], key=f"{side_key}_swap_i_b_imp")
            with c2:
                bench_pick = st.selectbox("Bench player", bench_names or [""], key=f"{side_key}_swap_i_b_bench")

        apply_col, reset_col, fin_col = st.columns([1, 1, 1])
        with apply_col:
            if st.button("Apply swap", key=f"{side_key}_apply_swap"):
                ok = True
                if swap_mode == "XI ↔ bench":
                    if not xi_pick or not bench_pick or xi_pick not in xi_list or bench_pick not in bench_names:
                        ok = False
                    else:
                        state[final_key] = [bench_pick if n == xi_pick else n for n in xi_list]
                        state[impact_key] = [n for n in state.get(impact_key) or [] if n != bench_pick]
                elif swap_mode == "XI ↔ impact":
                    if not xi_pick or not impact_pick or xi_pick not in xi_list or impact_pick not in imp_list:
                        ok = False
                    else:
                        im = list(state.get(impact_key) or [])
                        try:
                            j = im.index(impact_pick)
                        except ValueError:
                            ok = False
                        else:
                            im[j] = xi_pick
                            state[impact_key] = im
                            state[final_key] = [impact_pick if n == xi_pick else n for n in xi_list]
                else:
                    if not impact_pick or not bench_pick or impact_pick not in imp_list or bench_pick not in bench_names:
                        ok = False
                    else:
                        im = list(state.get(impact_key) or [])
                        try:
                            j = im.index(impact_pick)
                        except ValueError:
                            ok = False
                        else:
                            im[j] = bench_pick
                            state[impact_key] = im

                if ok:
                    xi_after = list(state.get(final_key) or [])
                    imp_after = list(state.get(impact_key) or [])
                    state[impact_key] = [n for n in imp_after if n and n not in set(xi_after)]
                    if swap_mode in ("XI ↔ bench", "XI ↔ impact"):
                        _rebuild_batting_order_state(
                            state,
                            side_key=side_key,
                            team_name=team_name,
                            squad_map=squad_map,
                            conditions=cond,
                            venue_keys=venue_keys_bo,
                        )
                    st.session_state["coach_tools_state"] = state
                    st.rerun()
                else:
                    st.warning("Pick two valid players for this swap type.")

        with reset_col:
            if st.button("Reset to Suggested XI", key=f"{side_key}_reset_final_xi"):
                px = [str(x.get("name") or "") for x in (r.get(side_key, {}).get("xi") or []) if x.get("name")]
                pi = [
                    str(x.get("name") or "")
                    for x in (r.get(side_key, {}).get("impact_subs") or [])
                    if x.get("name")
                ][:5]
                pbo = list(r.get(side_key, {}).get("batting_order") or [])
                state[final_key] = list(px)
                state[impact_key] = list(pi)
                state[bo_key] = list(pbo)
                state[f"{side_key}_finalised"] = False
                state[f"{side_key}_finalised_xi"] = None
                state[f"{side_key}_finalised_batting_order"] = None
                state[f"{side_key}_finalised_impact_names"] = None
                st.session_state["coach_tools_state"] = state
                st.rerun()

        with fin_col:
            if st.button("Finalise XI", key=f"{side_key}_finalise_xi_btn"):
                names = list(state.get(final_key) or [])
                _, errs = _validate_manual_xi(team_name, squad_map, names, cond)
                if len(names) != 11 or errs:
                    for err in errs or ["XI must be valid (11 players, squad + rules)."]:
                        st.warning(err)
                else:
                    xi_set = set(names)
                    _rebuild_batting_order_state(
                        state,
                        side_key=side_key,
                        team_name=team_name,
                        squad_map=squad_map,
                        conditions=cond,
                        venue_keys=venue_keys_bo,
                    )
                    bo = list(state.get(bo_key) or [])
                    if len(bo) != 11 or set(bo) != xi_set:
                        st.warning("Batting order could not be aligned to this XI — adjust the XI or reset, then try again.")
                    else:
                        imp_fin = [n for n in (state.get(impact_key) or []) if n and n not in xi_set]
                        state[impact_key] = imp_fin
                        state[f"{side_key}_finalised"] = True
                        state[f"{side_key}_finalised_xi"] = list(names)
                        state[f"{side_key}_finalised_batting_order"] = list(bo)
                        state[f"{side_key}_finalised_impact_names"] = list(imp_fin)
                        st.session_state["coach_tools_state"] = state
                        st.rerun()

        _, errs = _validate_manual_xi(team_name, squad_map, current_final_names, cond)
        if errs:
            for err in errs:
                st.error(err)
        else:
            st.success("Final XI is valid.")

    fx1, fx2 = st.columns(2)
    with fx1:
        _render_finalise_team(
            side_key="team_a",
            team_name=disp_a,
            all_rows=team_a_rows,
            squad_map=squad_map_a,
        )
    with fx2:
        _render_finalise_team(
            side_key="team_b",
            team_name=disp_b,
            all_rows=team_b_rows,
            squad_map=squad_map_b,
        )

    _render_batting_order_whatif_studio(
        r,
        state,
        squad_map_a,
        squad_map_b,
        disp_a,
        disp_b,
        cond,
    )

    st.subheader("Final Match Prediction")
    base_team_a = 100.0 * float((r.get("win_probability") or {}).get("team_a_win") or 0.0)
    base_team_b = max(0.0, 100.0 - base_team_a)
    st.caption("Win probability")
    base1, base2 = st.columns(2)
    with base1:
        st.metric(f"Base · {disp_a}", f"{base_team_a:.1f}%")
    with base2:
        st.metric(f"Base · {disp_b}", f"{base_team_b:.1f}%")
    recalc_result = _recalculate_final_win_prediction(
        r,
        squad_map_a,
        squad_map_b,
        _effective_xi_names(state, "team_a"),
        _effective_xi_names(state, "team_b"),
    )
    if not isinstance(recalc_result, tuple) or len(recalc_result) != 3:
        final_payload, recalc_errors, recalc_dbg = None, ["Final win recalculation returned an invalid result."], {}
    else:
        final_payload, recalc_errors, recalc_dbg = recalc_result
    with st.expander("Final win recalculation (conditions debug)", expanded=False):
        st.caption("Temporary: keys present on stored conditions vs after normalization for the win engine.")
        st.code(
            f"condition_keys_before_normalization: {recalc_dbg.get('condition_keys_before_normalization')!r}\n"
            f"condition_keys_after_normalization: {recalc_dbg.get('condition_keys_after_normalization')!r}",
            language=None,
        )
    if recalc_errors:
        final1, final2 = st.columns(2)
        with final1:
            st.metric(f"Final · {disp_a}", "—")
        with final2:
            st.metric(f"Final · {disp_b}", "—")
        for err in recalc_errors:
            st.warning(err)
    elif final_payload:
        final_a_raw = final_payload.get("team_a_final_win_pct")
        final_b_raw = final_payload.get("team_b_final_win_pct")
        if final_a_raw is None or final_b_raw is None:
            final1, final2 = st.columns(2)
            with final1:
                st.metric(f"Final · {disp_a}", "—")
            with final2:
                st.metric(f"Final · {disp_b}", "—")
            st.warning("Final win recalculation did not return usable team percentages.")
        else:
            final_a = float(final_a_raw)
            final_b = float(final_b_raw)
            final1, final2 = st.columns(2)
            with final1:
                st.metric(f"Final · {disp_a}", f"{final_a:.1f}%", delta=f"{final_a - base_team_a:+.1f}%")
            with final2:
                st.metric(f"Final · {disp_b}", f"{final_b:.1f}%", delta=f"{final_b - base_team_b:+.1f}%")
            st.progress(int(round(final_a)))
    audit_profile.append_session_audit_event(
        "post_prediction_render",
        "predict_ui_render.render_stored_prediction_results",
        (time.perf_counter() - _t_render) * 1000.0,
        extra={"advanced_debug": show_advanced_prediction_debug},
    )


def render_prediction_admin_debug(
    r: dict[str, Any],
    *,
    selection_debug_top15_for_side: Callable[..., Any],
) -> None:
    disp_a = r["team_a"]["name"]
    disp_b = r["team_b"]["name"]
    st.subheader("Prediction debug")
    st.caption("Technical prediction details moved off the main page.")

    hlw = (r.get("history_sync_debug") or {}).get("local_history_warning")
    if hlw:
        st.warning(str(hlw))
    xv = r.get("xi_validation") or {}
    warn_rows = [str(w) for w in (xv.get("strict_validation_warnings") or []) if str(w).strip()]
    if warn_rows:
        with st.expander("Linkage / validation warnings", expanded=False):
            for row in warn_rows:
                st.write(row)

    with st.expander("Confidence, timing, and internals", expanded=False):
        st.markdown("**model_confidence**")
        st.json(r.get("prediction_confidence") or {})
        st.markdown("**prediction_timing_ms**")
        st.json(r.get("prediction_timing_ms") or {})
        st.markdown("**audit_prediction**")
        st.json(r.get("audit_prediction") or {})
        st.markdown("**toss_effects**")
        st.json(r.get("toss_effects") or {})
        st.markdown("**batting_order_summary**")
        st.json(r.get("batting_order_summary") or {})
        st.markdown("**win_probability_engine**")
        st.json(r.get("win_probability_engine") or {})
        st.markdown("**scenario win splits**")
        st.json(
            {
                "team_a_if_bats_first": r.get("win_if_team_a_bats_first") or {},
                "team_a_if_bowls_first": r.get("win_if_team_a_bowls_first") or {},
            }
        )

    with st.expander("Batting order raw text", expanded=False):
        team_a_block = r.get("team_a") if isinstance(r.get("team_a"), dict) else {}
        team_b_block = r.get("team_b") if isinstance(r.get("team_b"), dict) else {}
        st.write(f"{disp_a}: " + " -> ".join(team_a_block.get("batting_order") or []))
        st.write(f"{disp_b}: " + " -> ".join(team_b_block.get("batting_order") or []))

    with st.expander("Selection top-15", expanded=False):
        dbg_side_pick = st.radio(
            "Team",
            ("team_a", "team_b"),
            format_func=lambda s: f"{disp_a} (A)" if s == "team_a" else f"{disp_b} (B)",
            horizontal=True,
            key="admin_selection_debug_side",
        )
        df_dbg, xi_val = selection_debug_top15_for_side(r, dbg_side_pick)
        if isinstance(df_dbg, pd.DataFrame) and not df_dbg.empty:
            st.dataframe(df_dbg, width="stretch", hide_index=True)
        else:
            st.caption("No top-15 rows available in the stored payload.")
        if xi_val:
            st.json(xi_val)

    with st.expander("Raw prediction payloads", expanded=False):
        st.markdown("**history_sync_debug**")
        st.json(r.get("history_sync_debug") or {})
        st.markdown("**xi_validation**")
        st.json(r.get("xi_validation") or {})
        st.markdown("**prediction_layer_debug**")
        st.json(r.get("prediction_layer_debug") or {})
