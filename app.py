"""Streamlit IPL Playing XI Predictor — weather, venue intelligence, multi-perspective XI."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st

import config
import db
import ipl_squad
import ipl_teams
import learner
import predict_ui_render
import predictor
import squad_fetch
import stage1_audit
import streamlit_db_init
import time_utils
import weather
from venues import list_venue_choices, resolve_venue

_perf_logger = logging.getLogger("ipl_predictor.perf")


def _selection_debug_top15_for_side(r: dict, side: str) -> tuple["pd.DataFrame", dict]:
    """
    Top 15 squad players by final_selection_score with Stage-3 selection_model fields.
    ``side`` is ``team_a`` or ``team_b``.
    """
    pld = r.get("prediction_layer_debug") or {}
    team = r.get(side) or {}
    block = pld.get(side) or {}
    raw = list(block.get("scoring_breakdown_per_player") or [])
    raw.sort(key=lambda x: float(x.get("final_selection_score") or 0), reverse=True)
    raw = raw[:15]
    xi_rows = team.get("xi") or []
    xi_names = {str(row.get("name") or "").strip() for row in xi_rows if row.get("name")}
    impact_names = {
        str(row.get("name") or "").strip()
        for row in (team.get("impact_subs") or [])
        if row.get("name")
    }
    rows_out: list[dict] = []
    for row in raw:
        name = str(row.get("squad_display_name") or row.get("player_name") or "").strip()
        smb = row.get("selection_model_base")
        if not isinstance(smb, dict):
            smb = {}
        tact = row.get("tactical_adjustment_total")
        if tact is None and isinstance(row.get("selection_model_tactical"), dict):
            tact = sum(
                float(v)
                for v in row["selection_model_tactical"].values()
                if isinstance(v, (int, float))
            )
        comps = row.get("recent_form_competitions_used")
        if not comps:
            rf_det = row.get("recent_form_detail")
            if isinstance(rf_det, dict):
                cu = rf_det.get("competitions_used")
                if isinstance(cu, list) and cu:
                    comps = ", ".join(str(c) for c in cu[:20])
        reason = (
            row.get("selection_reason_summary")
            or row.get("selection_model_explain_line")
            or ""
        )
        reason_s = str(reason).replace("\n", " ").strip()
        if len(reason_s) > 220:
            reason_s = reason_s[:217] + "…"
        rows_out.append(
            {
                "player": name,
                "in_playing_xi": "yes" if name in xi_names else "no",
                "impact_candidate": "yes" if name in impact_names else "no",
                "recent_form_score": smb.get("recent_form_score"),
                "ipl_history_and_role_score": smb.get("ipl_history_and_role_score"),
                "team_balance_fit_score": smb.get("team_balance_fit_score"),
                "venue_experience_score": smb.get("venue_experience_score"),
                "tactical_adjustment_total": tact,
                "final_selection_score": row.get("final_selection_score"),
                "recent_form_competitions": comps or "",
                "reason_summary": reason_s,
            }
        )
    sel_dbg = (r.get("selection_debug") or {}).get(side) or {}
    xi_val = sel_dbg.get("xi_validation") if isinstance(sel_dbg.get("xi_validation"), dict) else {}
    return pd.DataFrame(rows_out), xi_val


def _ensure_squad_state() -> None:
    if "sq_a" not in st.session_state:
        st.session_state["sq_a"] = ""
    if "sq_b" not in st.session_state:
        st.session_state["sq_b"] = ""
    if "_slug_a_cached" not in st.session_state:
        st.session_state["_slug_a_cached"] = None
    if "_slug_b_cached" not in st.session_state:
        st.session_state["_slug_b_cached"] = None
    if "squad_a_error" not in st.session_state:
        st.session_state["squad_a_error"] = None
    if "squad_b_error" not in st.session_state:
        st.session_state["squad_b_error"] = None
    if "squad_a_parse_debug" not in st.session_state:
        st.session_state["squad_a_parse_debug"] = None
    if "squad_b_parse_debug" not in st.session_state:
        st.session_state["squad_b_parse_debug"] = None
    if "cap_a" not in st.session_state:
        st.session_state["cap_a"] = ""
    if "cap_b" not in st.session_state:
        st.session_state["cap_b"] = ""
    if "wk_a" not in st.session_state:
        st.session_state["wk_a"] = ""
    if "wk_b" not in st.session_state:
        st.session_state["wk_b"] = ""


def _maybe_fetch_squad(slug: str, side: str) -> None:
    """Fetch official squad when `slug` differs from cached slug for that side."""
    cache_key = "_slug_a_cached" if side == "a" else "_slug_b_cached"
    text_key = "sq_a" if side == "a" else "sq_b"
    err_key = "squad_a_error" if side == "a" else "squad_b_error"
    keys_key = "fetched_keys_a" if side == "a" else "fetched_keys_b"
    pre_k = "_pre_change_fetched_a" if side == "a" else "_pre_change_fetched_b"
    cap_k = "cap_a" if side == "a" else "cap_b"
    wk_k = "wk_a" if side == "a" else "wk_b"
    if slug == st.session_state.get(cache_key):
        return
    st.session_state[pre_k] = st.session_state.get(keys_key)
    st.session_state[keys_key] = None
    st.session_state[cap_k] = ""
    st.session_state[wk_k] = ""
    dbg_key = "squad_a_parse_debug" if side == "a" else "squad_b_parse_debug"
    players, err, dbg = squad_fetch.fetch_squad_for_slug(slug)
    st.session_state[cache_key] = slug
    st.session_state[dbg_key] = dbg
    if err:
        st.session_state[err_key] = err
        if not str(st.session_state.get(text_key, "")).strip():
            st.session_state[text_key] = ""
    else:
        st.session_state[err_key] = None
        st.session_state[text_key] = squad_fetch.format_squad_text(players)
        st.session_state[keys_key] = frozenset(
            learner.normalize_player_key(getattr(m, "name", "") or "") for m in players
        )
        st.session_state[pre_k] = None


def main() -> None:
    _t_startup = time.perf_counter()
    st.set_page_config(
        page_title="IPL Playing XI Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _t_db = time.perf_counter()
    streamlit_db_init.ensure_db_schema_initialized(streamlit_db_init.db_init_signature())
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "app_db_ensure_ms=%.2f",
            (time.perf_counter() - _t_db) * 1000.0,
        )
    _ensure_squad_state()
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "app_startup_to_main_ready_ms=%.2f",
            (time.perf_counter() - _t_startup) * 1000.0,
        )

    with st.sidebar:
        st.caption("**Predict** (this page)")
        if st.button("Open Admin & maintenance 🛠️", use_container_width=True, key="nav_open_admin"):
            st.switch_page("pages/1_Admin_and_maintenance.py")
        st.divider()
        st.caption(
            "Ingest, Cricsheet, recent-form cache, SQLite audit, and scorecard tools live on **Admin & maintenance**."
        )

    st.title("IPL Playing XI Predictor")
    st.caption(
        "Weighted coach, player, analyst, opposition, and learned-history views — "
        "with weather (IST), venue conditions, XI constraints, impact subs, toss leverage, and win odds."
    )

    st.subheader("Pre-match inputs")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        team_a_slug = st.selectbox(
            "Team A (Home)",
            options=ipl_teams.TEAM_SLUGS,
            format_func=lambda s: ipl_teams.label_for_slug(s),
            key="select_team_a",
        )
    with r1c2:
        b_options = [s for s in ipl_teams.TEAM_SLUGS if s != team_a_slug]
        team_b_slug = st.selectbox(
            "Team B (Away)",
            options=b_options,
            format_func=lambda s: ipl_teams.label_for_slug(s),
            key="select_team_b",
        )

    team_a_name = ipl_teams.label_for_slug(team_a_slug)
    team_b_name = ipl_teams.label_for_slug(team_b_slug)

    venue_options = list_venue_choices()
    v_labels = [f"{d} ({k})" for k, d in venue_options]
    v_keys = [k for k, _ in venue_options]
    v_pick = st.selectbox("Venue", range(len(v_labels)), format_func=lambda i: v_labels[i])
    venue_key = v_keys[v_pick]
    venue_custom = st.text_input("Venue override (optional free text)", value="")
    venue = resolve_venue(venue_custom.strip() or venue_key)

    md = st.date_input("Match date", value=date.today())
    st.caption("All match times use **India Standard Time (IST)**.")
    mt = st.time_input(
        "Match Time (IST)",
        value=datetime.now(time_utils.IST).time().replace(second=0, microsecond=0),
    )
    match_time_ist = time_utils.combine_date_time_ist(md, mt)

    unavailable = st.text_area(
        "Unavailable players (one name per line — matched loosely)",
        height=100,
        key="unavailable_input",
    )

    _maybe_fetch_squad(team_a_slug, "a")
    _maybe_fetch_squad(team_b_slug, "b")

    st.subheader("Squads (auto-loaded from IPLT20; editable)")
    fb1, fb2 = st.columns(2)
    with fb1:
        if st.button("Refresh squads from IPLT20", help="Re-download both squads from official pages"):
            st.session_state["_slug_a_cached"] = None
            st.session_state["_slug_b_cached"] = None
            st.rerun()
    with fb2:
        st.caption(f"Source: `{squad_fetch.SQUAD_URL_TEMPLATE}`")

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.get("squad_a_error"):
            st.warning(f"Team A squad fetch: {st.session_state['squad_a_error']}")
        squad_a = st.text_area(
            f"{team_a_name} squad",
            height=240,
            key="sq_a",
            help="IPL format: Name | Batter (or WK-Batter, All-Rounder, Bowler). Optional: | overseas. "
            "Legacy comma tags still work.",
        )
    with c2:
        if st.session_state.get("squad_b_error"):
            st.warning(f"Team B squad fetch: {st.session_state['squad_b_error']}")
        squad_b = st.text_area(
            f"{team_b_name} squad",
            height=240,
            key="sq_b",
            help="IPL format: Name | Batter (or WK-Batter, All-Rounder, Bowler). Optional: | overseas. "
            "Auto-fetch also marks overseas from IPL’s foreign-player icon when present.",
        )

    pa = predictor.parse_squad_text(squad_a)
    pb = predictor.parse_squad_text(squad_b)
    cap_names_a = [""] + [p.name for p in pa]
    cap_names_b = [""] + [p.name for p in pb]
    wk_order_a = [p.name for p in pa if p.role_bucket == ipl_squad.WK_BATTER] + [
        p.name for p in pa if p.role_bucket != ipl_squad.WK_BATTER
    ]
    wk_order_b = [p.name for p in pb if p.role_bucket == ipl_squad.WK_BATTER] + [
        p.name for p in pb if p.role_bucket != ipl_squad.WK_BATTER
    ]
    wk_names_a = [""] + wk_order_a
    wk_names_b = [""] + wk_order_b
    for key, opts in (
        ("cap_a", cap_names_a),
        ("cap_b", cap_names_b),
        ("wk_a", wk_names_a),
        ("wk_b", wk_names_b),
    ):
        cur = st.session_state.get(key)
        if cur and cur not in opts:
            st.session_state[key] = ""

    st.subheader("Captain & wicketkeeper (optional XI priors)")
    st.caption(
        "Choices apply **strong selection-score boosts** for the playing XI (not hard locks). "
        "WK list lists **WK-Batter** roles first, then the rest of the squad."
    )
    ac1, ac2 = st.columns(2)
    with ac1:
        st.selectbox(
            f"{team_a_name} captain",
            cap_names_a,
            key="cap_a",
            format_func=lambda x: "(auto — no captain prior)" if x == "" else x,
            help="Boosts selection_score; may still be omitted if constraints force it.",
        )
        st.selectbox(
            f"{team_a_name} wicketkeeper",
            wk_names_a,
            key="wk_a",
            format_func=lambda x: "(auto — no WK prior)" if x == "" else x,
        )
    with ac2:
        st.selectbox(
            f"{team_b_name} captain",
            cap_names_b,
            key="cap_b",
            format_func=lambda x: "(auto — no captain prior)" if x == "" else x,
        )
        st.selectbox(
            f"{team_b_name} wicketkeeper",
            wk_names_b,
            key="wk_b",
            format_func=lambda x: "(auto — no WK prior)" if x == "" else x,
        )

    prediction_tuning_debug = st.checkbox(
        "Prediction tuning debug",
        value=False,
        key="prediction_tuning_debug_enabled",
        help="Enable on-demand SQLite checks for squad linkage and batting-order samples. Default off — no extra queries.",
    )
    if prediction_tuning_debug:
        with st.expander("Prediction tuning tools", expanded=True):
            st.caption(
                "Queries run **only** when you click a button. Full ingest, audit, and DB maintenance: **Admin & maintenance** page."
            )
            if st.button(
                "Verify current squad ↔ raw SQLite linkage (Teams A & B)",
                key="predict_tuning_squad_link",
            ):
                la = stage1_audit.squad_raw_history_linkage_for_team(
                    squad_a, team_a_name, opponent_label=team_b_name
                )
                lb = stage1_audit.squad_raw_history_linkage_for_team(
                    squad_b, team_b_name, opponent_label=team_a_name
                )
                st.json(
                    {
                        "team_a": la.get("summary"),
                        "team_a_per_player": la.get("per_player"),
                        "team_b": lb.get("summary"),
                        "team_b_per_player": lb.get("per_player"),
                    }
                )
            if st.button(
                "Show batting order vs recent matches (player_batting_positions sample)",
                key="predict_tuning_bat_recent",
            ):
                with db.connection() as conn:
                    st.json(stage1_audit.batting_position_ingest_sample(conn, limit=10))

    toss_labels = [lbl for _k, lbl in predictor.TOSS_SCENARIO_OPTIONS]
    toss_keys = [k for k, _lbl in predictor.TOSS_SCENARIO_OPTIONS]
    toss_ix = st.selectbox(
        "Toss scenario (XI history, impact subs, win %)",
        range(len(toss_labels)),
        format_func=lambda i: toss_labels[i],
        help="Unknown = neutral win % and no chase/defend tilt. Known toss feeds SQLite-backed history signals, "
        "impact subs ordering, logistic model, and headline win %.",
    )
    toss_key = toss_keys[int(toss_ix)]

    if st.button("Run prediction", type="primary"):
        _t_run_btn = time.perf_counter()
        with st.spinner("Fetching weather (IST), reading SQLite history (if loaded), building projections…"):
            w = weather.fetch_weather(venue, match_time_ist)
            parsed_a = predictor.parse_squad_text(squad_a)
            parsed_b = predictor.parse_squad_text(squad_b)
            parsed_keys_a = {learner.normalize_player_key(p.name) for p in parsed_a}
            parsed_keys_b = {learner.normalize_player_key(p.name) for p in parsed_b}
            parsed_keys_a.discard("")
            parsed_keys_b.discard("")
            fe_a = st.session_state.get("fetched_keys_a")
            fe_b = st.session_state.get("fetched_keys_b")
            pre_a = st.session_state.get("_pre_change_fetched_a")
            pre_b = st.session_state.get("_pre_change_fetched_b")
            stale_a: set[str] = set()
            stale_b: set[str] = set()
            if isinstance(fe_a, frozenset) and fe_a and isinstance(pre_a, frozenset) and pre_a:
                stale_a = (parsed_keys_a - fe_a) & pre_a
            if isinstance(fe_b, frozenset) and fe_b and isinstance(pre_b, frozenset) and pre_b:
                stale_b = (parsed_keys_b - fe_b) & pre_b
            result = predictor.run_prediction(
                team_a_name,
                team_b_name,
                squad_a,
                squad_b,
                unavailable,
                venue,
                match_time_ist,
                w,
                toss_scenario_key=toss_key,
                team_a_captain_display_name=st.session_state.get("cap_a") or "",
                team_b_captain_display_name=st.session_state.get("cap_b") or "",
                team_a_wicketkeeper_display_name=st.session_state.get("wk_a") or "",
                team_b_wicketkeeper_display_name=st.session_state.get("wk_b") or "",
                team_a_fetched_squad_player_keys=set(fe_a) if isinstance(fe_a, frozenset) and fe_a else None,
                team_b_fetched_squad_player_keys=set(fe_b) if isinstance(fe_b, frozenset) and fe_b else None,
                team_a_stale_cached_player_keys=stale_a or None,
                team_b_stale_cached_player_keys=stale_b or None,
            )
        if getattr(config, "PREDICTION_TIMING_LOG", False):
            _perf_logger.info(
                "app_run_prediction_button_block_ms=%.2f",
                (time.perf_counter() - _t_run_btn) * 1000.0,
            )
        st.session_state["last_prediction"] = result
        st.session_state["last_venue"] = venue
        st.session_state["last_weather"] = w
        st.session_state["last_team_a_slug"] = team_a_slug
        st.session_state["last_team_b_slug"] = team_b_slug

    if "last_prediction" in st.session_state:
        r = st.session_state["last_prediction"]
        show_advanced_prediction_debug = st.checkbox(
            "Show advanced prediction debug (large tables, SQLite JSON, selection top-15)",
            value=False,
            key="advanced_prediction_debug",
            help="Off by default for speed. Does not change prediction results — only what is shown.",
        )
        predict_ui_render.render_stored_prediction_results(
            r,
            show_advanced_prediction_debug=show_advanced_prediction_debug,
            selection_debug_top15_for_side=_selection_debug_top15_for_side,
        )


if __name__ == "__main__":
    main()
