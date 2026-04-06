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
import history_sync
import ipl_teams
import ipl_squad
import predictor
import rules_xi
import win_probability_engine
from history_context import build_history_context
from venues import resolve_venue

_perf_logger = logging.getLogger("ipl_predictor.perf")


def _coach_signature(r: dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(r.get("team_a", {}).get("name") or ""),
            str(r.get("team_b", {}).get("name") or ""),
            ",".join(sorted(str(x.get("name") or "") for x in (r.get("team_a", {}).get("xi") or []))),
            ",".join(sorted(str(x.get("name") or "") for x in (r.get("team_b", {}).get("xi") or []))),
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
    player = predictor.SquadPlayer(
        name=str(row.get("name") or ""),
        role=role,
        is_overseas=bool(row.get("overseas")),
        player_key=str(row.get("player_key") or row.get("canonical_player_key") or ""),
        team_display_name=str(row.get("_team_name") or ""),
        canonical_team_key=str(row.get("_canonical_team_key") or ""),
        canonical_player_key=str(row.get("canonical_player_key") or row.get("player_key") or ""),
        role_bucket=str(row.get("role_bucket") or ipl_squad.BATTER),
        bat_skill=float(row.get("bat_skill") or bat_skill),
        bowl_skill=float(row.get("bowl_skill") or bowl_skill),
        is_wicketkeeper=bool(row.get("is_wk_role") or row.get("is_keeper") or is_wk),
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
        errors.extend(v.message for v in res.violations)
    return xi, errors


def _recalculate_final_win_prediction(
    r: dict[str, Any],
    squad_map_a: dict[str, dict[str, Any]],
    squad_map_b: dict[str, dict[str, Any]],
    final_names_a: list[str],
    final_names_b: list[str],
) -> tuple[Optional[dict[str, float]], list[str]]:
    xi_a, errs_a = _validate_manual_xi(str(r.get("team_a", {}).get("name") or ""), squad_map_a, final_names_a, r.get("conditions") or {})
    xi_b, errs_b = _validate_manual_xi(str(r.get("team_b", {}).get("name") or ""), squad_map_b, final_names_b, r.get("conditions") or {})
    if errs_a or errs_b:
        return None, [*errs_a, *errs_b]
    try:
        venue = resolve_venue(str((r.get("conditions") or {}).get("venue") or ""))
        conditions = dict(r.get("conditions") or {})
        venue_keys = list(((r.get("learning_context") or {}).get("venue_keys_tried") or []))
        toss_key = str(((r.get("toss_scenario") or {}).get("key") or "unknown"))
        a_bats_first = (r.get("toss_scenario") or {}).get("team_a_bats_first")
        is_night = bool((r.get("learning_context") or {}).get("fixture_night"))
        order_a = predictor.build_batting_order(
            xi_a,
            conditions,
            team_name=str(r["team_a"]["name"]),
            venue_keys=venue_keys,
            out_warnings=[],
        )
        order_b = predictor.build_batting_order(
            xi_b,
            conditions,
            team_name=str(r["team_b"]["name"]),
            venue_keys=venue_keys,
            out_warnings=[],
        )
        match_rows = db.fetch_match_results_meta(450)
        hctx = build_history_context()
        win = win_probability_engine.compute_win_probability(
            str(r["team_a"]["name"]),
            str(r["team_b"]["name"]),
            xi_a,
            xi_b,
            order_a,
            order_b,
            venue,
            conditions,
            venue_keys=venue_keys,
            match_rows=match_rows,
            toss_scenario_key=toss_key,
            a_bats_first_selected=a_bats_first,
            chase_share_by_venue=hctx.chase_share_by_venue,
            is_night_fixture=is_night,
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
            raise ValueError("win engine did not return a usable Team A win percentage")
        team_a_final = float(team_a_selected)
        team_b_selected = win_dict.get("team_b_win_pct_selected_toss")
        if team_b_selected is None:
            team_b_selected = getattr(win, "team_b_win_pct_selected_toss", None)
        if team_b_selected is None:
            team_b_selected = 100.0 - team_a_final
        return {
            "team_a_final_win_pct": team_a_final,
            "team_b_final_win_pct": float(team_b_selected),
        }, []
    except Exception as exc:
        return None, [f"Final win recalculation unavailable: {exc}"]


def _fetch_direct_matchup(batter_key: str, bowler_key: str) -> Optional[dict[str, Any]]:
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT balls, runs, dismissals, strike_rate, dot_ball_pct, boundary_pct,
                   innings_count, match_count, sample_size_confidence
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


def _fetch_batter_phase_summary(batter_key: str, phase: str) -> Optional[dict[str, Any]]:
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT balls, runs, dismissals, strike_rate, dot_ball_pct, sample_size_confidence
            FROM batter_vs_phase_summary
            WHERE batter_key = ? AND bowling_phase = ?
            """,
            (batter_key, phase),
        ).fetchone()
    return dict(row) if row else None


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


def render_stored_prediction_results(
    r: dict[str, Any],
    *,
    show_advanced_prediction_debug: bool,
    selection_debug_top15_for_side: Callable[..., Any],
) -> None:
    _t_render = time.perf_counter()
    cond = r["conditions"]
    disp_a = r["team_a"]["name"]
    disp_b = r["team_b"]["name"]

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

    def _direct_matchup_card(
        left_row: dict[str, Any],
        right_row: dict[str, Any],
        *,
        key_prefix: str,
    ) -> None:
        left_name = str(left_row.get("name") or "Player A")
        right_name = str(right_row.get("name") or "Player B")
        left_key = str(left_row.get("player_key") or "").strip().lower()
        right_key = str(right_row.get("player_key") or "").strip().lower()
        left_meta = left_row.get("_meta") if isinstance(left_row.get("_meta"), dict) else {}
        right_meta = right_row.get("_meta") if isinstance(right_row.get("_meta"), dict) else {}
        left_role = str(left_row.get("role_bucket") or "")
        right_role = str(right_row.get("role_bucket") or "")
        left_is_bowler = left_role in (ipl_squad.BOWLER, ipl_squad.ALL_ROUNDER)
        right_is_bowler = right_role in (ipl_squad.BOWLER, ipl_squad.ALL_ROUNDER)

        batter_row = left_row
        bowler_row = right_row
        if left_is_bowler and not right_is_bowler:
            batter_row, bowler_row = right_row, left_row
        batter_key = str(batter_row.get("player_key") or "").strip().lower()
        bowler_key = str(bowler_row.get("player_key") or "").strip().lower()
        direct = _fetch_direct_matchup(batter_key, bowler_key)
        batting_hand = str((batter_row.get("_meta") or {}).get("batting_hand") or "")
        bowling_type_bucket = str((bowler_row.get("_meta") or {}).get("bowling_type_bucket") or "")
        phase = "powerplay" if "open" in str(batter_row.get("batting_band") or "").lower() else "middle"
        fallback_bat = _fetch_batter_type_fallback(batter_key, bowling_type_bucket)
        fallback_bowl = _fetch_bowler_hand_fallback(bowler_key, batting_hand)
        fallback_phase = _fetch_batter_phase_summary(batter_key, phase)

        title_html = (
            f'<div style="font-size:18px;font-weight:700;">{left_name} <span style="color:#6b7280;">vs</span> {right_name}</div>'
        )
        st.markdown(title_html, unsafe_allow_html=True)
        if direct:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Balls", int(direct.get("balls") or 0))
            c2.metric("Runs", int(direct.get("runs") or 0))
            c3.metric("Dismissals", int(direct.get("dismissals") or 0))
            c4.metric("SR / Econ", f"{float(direct.get('strike_rate') or 0.0):.1f}")
            c5.metric("Confidence", f"{100 * float(direct.get('sample_size_confidence') or 0.0):.0f}%")
            st.caption(_recommendation_from_matchup(direct, mode="direct"))
        else:
            st.info("No meaningful direct history yet — using fallback profile.")

        fb1, fb2, fb3 = st.columns(3)
        with fb1:
            if fallback_bat:
                st.markdown("**Batter vs bowling type**")
                st.write(
                    {
                        "type": bowling_type_bucket or "unknown",
                        "balls": int(fallback_bat.get("balls") or 0),
                        "runs": int(fallback_bat.get("runs") or 0),
                        "dismissals": int(fallback_bat.get("dismissals") or 0),
                        "strike_rate": round(float(fallback_bat.get("strike_rate") or 0.0), 1),
                        "confidence": round(100 * float(fallback_bat.get("sample_size_confidence") or 0.0), 0),
                    }
                )
            else:
                st.caption("No bowling-type fallback.")
        with fb2:
            if fallback_bowl:
                st.markdown("**Bowler vs batting hand**")
                st.write(
                    {
                        "hand": batting_hand or "unknown",
                        "balls": int(fallback_bowl.get("balls") or 0),
                        "runs": int(fallback_bowl.get("runs") or 0),
                        "dismissals": int(fallback_bowl.get("dismissals") or 0),
                        "economy": round(float(fallback_bowl.get("economy") or 0.0), 2),
                        "confidence": round(100 * float(fallback_bowl.get("sample_size_confidence") or 0.0), 0),
                    }
                )
            else:
                st.caption("No batting-hand fallback.")
        with fb3:
            if fallback_phase:
                st.markdown("**Phase profile**")
                st.write(
                    {
                        "phase": phase,
                        "balls": int(fallback_phase.get("balls") or 0),
                        "runs": int(fallback_phase.get("runs") or 0),
                        "dismissals": int(fallback_phase.get("dismissals") or 0),
                        "strike_rate": round(float(fallback_phase.get("strike_rate") or 0.0), 1),
                        "confidence": round(100 * float(fallback_phase.get("sample_size_confidence") or 0.0), 0),
                    }
                )
            else:
                st.caption("No phase fallback.")

    st.subheader("Matchup Studio")
    ms1, ms2, ms3, ms4 = st.columns(4)

    def _top_pairs(
        bat_keys: list[str], bowl_keys: list[str], *, limit: int = 8
    ) -> pd.DataFrame:
        if not bat_keys or not bowl_keys:
            return pd.DataFrame()
        qm_a = ",".join("?" * len(bat_keys))
        qm_b = ",".join("?" * len(bowl_keys))
        with db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT batter_key, bowler_key, balls, runs, dismissals, strike_rate, dot_ball_pct,
                       sample_size_confidence
                FROM batter_bowler_matchup_summary
                WHERE batter_key IN ({qm_a}) AND bowler_key IN ({qm_b})
                ORDER BY sample_size_confidence DESC, dismissals DESC, dot_ball_pct DESC, balls DESC
                LIMIT ?
                """,
                [*bat_keys, *bowl_keys, int(limit)],
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()

    def _bat_when_phase(bat_keys: list[str], *, limit: int = 10) -> pd.DataFrame:
        if not bat_keys:
            return pd.DataFrame()
        qm = ",".join("?" * len(bat_keys))
        with db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT batter_key, bowling_phase, balls, strike_rate, dot_ball_pct, sample_size_confidence
                FROM batter_vs_phase_summary
                WHERE batter_key IN ({qm})
                ORDER BY sample_size_confidence DESC, strike_rate DESC, balls DESC
                LIMIT ?
                """,
                [*bat_keys, int(limit)],
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()

    def _bat_vs_spin_pace(bat_keys: list[str], *, limit: int = 10) -> pd.DataFrame:
        if not bat_keys:
            return pd.DataFrame()
        qm = ",".join("?" * len(bat_keys))
        with db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT batter_key, pace_spin_bucket, balls, strike_rate, dot_ball_pct, sample_size_confidence
                FROM batter_vs_spin_pace_summary
                WHERE batter_key IN ({qm})
                ORDER BY sample_size_confidence DESC, balls DESC
                LIMIT ?
                """,
                [*bat_keys, int(limit)],
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()

    final_rows_a = _final_rows("team_a", final_names_a, squad_map_a)
    final_rows_b = _final_rows("team_b", final_names_b, squad_map_b)
    final_keys_a = [str(row.get("player_key") or "").strip().lower() for row in final_rows_a if row.get("player_key")]
    final_keys_b = [str(row.get("player_key") or "").strip().lower() for row in final_rows_b if row.get("player_key")]
    final_disp = {
        **{str(row.get("player_key") or "").strip().lower(): str(row.get("name") or "") for row in final_rows_a},
        **{str(row.get("player_key") or "").strip().lower(): str(row.get("name") or "") for row in final_rows_b},
    }

    def _name_for_key(k: str) -> str:
        return final_disp.get(str(k or "").strip().lower(), str(k or "—"))

    top_order_df = _top_pairs(final_keys_a[:4], final_keys_b[:3], limit=4)
    middle_spin_df = _bat_when_phase(final_keys_a[3:8] + final_keys_b[3:8], limit=8)
    finishers_df = _top_pairs(final_keys_a[6:11], final_keys_b[-4:], limit=4)
    hand_df = _bat_vs_spin_pace(final_keys_a + final_keys_b, limit=8)

    with ms1:
        st.markdown("**Top order vs new ball**")
        if top_order_df.empty:
            st.caption("No strong direct read yet.")
        else:
            row = top_order_df.iloc[0]
            st.metric("Best bowling edge", f"{_name_for_key(row['bowler']) if 'bowler' in row else _name_for_key(row['bowler_key'])}")
            st.caption(f"Against {_name_for_key(row.get('batter_key'))} · {int(row.get('balls') or 0)} balls · conf {100 * float(row.get('sample_size_confidence') or 0):.0f}%")
    with ms2:
        st.markdown("**Middle order vs spin**")
        if middle_spin_df.empty:
            st.caption("No strong spin-phase read yet.")
        else:
            row = middle_spin_df.iloc[0]
            st.metric("Best phase target", _name_for_key(row.get("batter_key")))
            st.caption(f"{str(row.get('bowling_phase') or 'middle').title()} · SR {float(row.get('strike_rate') or 0):.1f}")
    with ms3:
        st.markdown("**Finishers vs death**")
        if finishers_df.empty:
            st.caption("No strong death-overs read yet.")
        else:
            row = finishers_df.iloc[0]
            st.metric("Key matchup", _name_for_key(row.get("batter_key")))
            st.caption(f"vs {_name_for_key(row.get('bowler_key'))} · SR {float(row.get('strike_rate') or 0):.1f}")
    with ms4:
        st.markdown("**Left/right profile**")
        if hand_df.empty:
            st.caption("No hand/type signal yet.")
        else:
            row = hand_df.iloc[0]
            st.metric("Best type fit", _name_for_key(row.get("batter_key")))
            st.caption(f"vs {str(row.get('pace_spin_bucket') or '').title()} · SR {float(row.get('strike_rate') or 0):.1f}")

    st.markdown("**Matchup recommendations**")
    if top_order_df.empty and middle_spin_df.empty and finishers_df.empty:
        st.caption("Historical matchup coverage is still sparse for this fixture.")
    else:
        recs: list[str] = []
        if not top_order_df.empty:
            row = top_order_df.iloc[0]
            recs.append(
                f"Use {_name_for_key(row.get('bowler_key'))} early against {_name_for_key(row.get('batter_key'))} when the new ball is live."
            )
        if not middle_spin_df.empty:
            row = middle_spin_df.iloc[0]
            recs.append(
                f"Middle overs look favorable for {_name_for_key(row.get('batter_key'))} against {str(row.get('bowling_phase') or 'middle')} profiles."
            )
        if not finishers_df.empty:
            row = finishers_df.iloc[0]
            recs.append(
                f"Keep {_name_for_key(row.get('bowler_key'))} available late if {_name_for_key(row.get('batter_key'))} is still in."
            )
        for rec in recs[:3]:
            st.write(f"• {rec}")

    matchup_mode = st.radio(
        "Mode",
        ("Direct matchup", "Selection comparison"),
        horizontal=True,
        key="matchup_studio_mode",
    )
    if matchup_mode == "Direct matchup":
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
            _direct_matchup_card(row_a, row_b, key_prefix="direct_matchup")
    else:
        sc1, sc2, sc3 = st.columns([1.2, 1, 1])
        compare_team = sc1.radio(
            "Compare for",
            ("team_a", "team_b"),
            format_func=lambda s: disp_a if s == "team_a" else disp_b,
            horizontal=True,
            key="matchup_compare_team",
        )
        own_rows = team_a_rows if compare_team == "team_a" else team_b_rows
        opp_rows = final_rows_b if compare_team == "team_a" else final_rows_a
        own_map = squad_map_a if compare_team == "team_a" else squad_map_b
        own_names = [str(row.get("name") or "") for row in own_rows]
        with sc2:
            player_1 = st.selectbox("Candidate 1", own_names, key="matchup_compare_p1")
        with sc3:
            player_2 = st.selectbox(
                "Candidate 2",
                own_names,
                index=1 if len(own_names) > 1 else 0,
                key="matchup_compare_p2",
            )

        def _comparison_score(candidate_row: dict[str, Any], opposition_rows: list[dict[str, Any]]) -> dict[str, float]:
            cand_key = str(candidate_row.get("player_key") or "").strip().lower()
            role_bucket = str(candidate_row.get("role_bucket") or "")
            bowling_type_bucket = str((candidate_row.get("_meta") or {}).get("bowling_type_bucket") or "")
            batting_hand = str((candidate_row.get("_meta") or {}).get("batting_hand") or "")
            batting_edge = 0.0
            bowling_edge = 0.0
            for opp in opposition_rows:
                opp_key = str(opp.get("player_key") or "").strip().lower()
                direct = _fetch_direct_matchup(cand_key, opp_key)
                if direct:
                    batting_edge = max(
                        batting_edge,
                        float(direct.get("strike_rate") or 0.0) * float(direct.get("sample_size_confidence") or 0.0),
                    )
                direct_against = _fetch_direct_matchup(opp_key, cand_key)
                if direct_against:
                    bowling_edge = max(
                        bowling_edge,
                        float(direct_against.get("dismissals") or 0.0) * 20.0
                        + float(direct_against.get("dot_ball_pct") or 0.0) * 100.0,
                    )
            role_fit = 8.0 if role_bucket in (ipl_squad.BOWLER, ipl_squad.ALL_ROUNDER) else 5.0
            venue_fit = 4.0 if ("spin" in bowling_type_bucket and float(cond.get("spin_friendliness") or 0.0) >= 0.55) else 0.0
            venue_fit += 4.0 if ("pace" in bowling_type_bucket and float(cond.get("pace_bias") or 0.0) >= 0.55) else 0.0
            venue_fit += 2.0 if batting_hand else 0.0
            return {
                "batting_edge": batting_edge,
                "bowling_edge": bowling_edge,
                "venue_fit": venue_fit,
                "role_fit": role_fit,
                "total": batting_edge + bowling_edge + venue_fit + role_fit,
            }

        row_1 = own_map.get(player_1)
        row_2 = own_map.get(player_2)
        if row_1 and row_2:
            score_1 = _comparison_score(row_1, opp_rows)
            score_2 = _comparison_score(row_2, opp_rows)
            cmp1, cmp2, cmp3 = st.columns(3)
            with cmp1:
                st.markdown(f"**{player_1}**")
                st.metric("Batting matchup edge", f"{score_1['batting_edge']:.1f}")
                st.metric("Bowling matchup edge", f"{score_1['bowling_edge']:.1f}")
                st.metric("Venue / role fit", f"{score_1['venue_fit'] + score_1['role_fit']:.1f}")
            with cmp2:
                st.markdown(f"**{player_2}**")
                st.metric("Batting matchup edge", f"{score_2['batting_edge']:.1f}")
                st.metric("Bowling matchup edge", f"{score_2['bowling_edge']:.1f}")
                st.metric("Venue / role fit", f"{score_2['venue_fit'] + score_2['role_fit']:.1f}")
            with cmp3:
                st.markdown("**Recommendation**")
                diff = score_1["total"] - score_2["total"]
                if diff > 8.0:
                    st.success(f"Prefer {player_1}")
                elif diff < -8.0:
                    st.success(f"Prefer {player_2}")
                else:
                    st.info("Balanced call")
                st.caption("Advisory only — this does not auto-change the XI.")

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
    if (
        not isinstance(recalc_result, tuple)
        or len(recalc_result) != 2
    ):
        final_payload, recalc_errors = None, ["Final win recalculation returned an invalid result."]
    else:
        final_payload, recalc_errors = recalc_result
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
        st.write(f"{disp_a}: " + " -> ".join(r.get("team_a", {}).get("batting_order") or []))
        st.write(f"{disp_b}: " + " -> ".join(r.get("team_b", {}).get("batting_order") or []))

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
