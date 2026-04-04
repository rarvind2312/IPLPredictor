"""Streamlit IPL Predictor."""

from __future__ import annotations

import base64
import logging
import time
from datetime import date, datetime, time as dt_time
from pathlib import Path

_APP_IMPORT_T0 = time.perf_counter()

import pandas as pd
import streamlit as st

import audit_profile
import config
import cricsheet_all_ingest
import db
import ipl_squad
import ipl_teams
import learner
import predict_ui_render
import predictor
import recent_form_cache
import squad_fetch
import streamlit_db_init
import time_utils
import weather
from venues import list_venue_choices, resolve_venue

_APP_IMPORT_DONE = time.perf_counter()

_perf_logger = logging.getLogger("ipl_predictor.perf")
_LOCAL_IPL_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "team_logos" / "IPL.png"


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
    if "_fetch_squads_requested" not in st.session_state:
        st.session_state["_fetch_squads_requested"] = bool(getattr(config, "AUTO_FETCH_SQUADS_ON_START", False))


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
    _t_sf = time.perf_counter()
    players, err, dbg = squad_fetch.fetch_squad_for_slug(slug)
    audit_profile.append_session_audit_event(
        "squad_fetch",
        f"iplt20_squad_{side}",
        (time.perf_counter() - _t_sf) * 1000.0,
        extra={"slug": slug},
    )
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
    _t_main = time.perf_counter()
    _t_pc = time.perf_counter()
    st.set_page_config(
        page_title="IPL Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>"
        "[data-testid='stDeployButton']{display:none !important;}"
        "[data-testid='stStatusWidget']{display:none !important;}"
        ".block-container{padding-top:1.35rem !important;}"
        "@keyframes ipl-logo-pulse{"
        "0%{transform:scale(0.9);opacity:0.45;}"
        "20%{transform:scale(1.0);opacity:1;}"
        "100%{transform:scale(0.9);opacity:0.45;}"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )
    _set_page_ms = (time.perf_counter() - _t_pc) * 1000.0
    _t_db = time.perf_counter()
    streamlit_db_init.ensure_db_schema_initialized(streamlit_db_init.db_init_signature())
    _db_ensure_ms = (time.perf_counter() - _t_db) * 1000.0
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info("app_db_ensure_ms=%.2f", _db_ensure_ms)
    _t_es = time.perf_counter()
    _ensure_squad_state()
    _ensure_squad_ms = (time.perf_counter() - _t_es) * 1000.0
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "app_startup_to_main_ready_ms=%.2f",
            (time.perf_counter() - _t_main) * 1000.0,
        )

    _t_side = time.perf_counter()
    with st.sidebar:
        st.caption("**Predict** (this page)")
        if st.button("Open Admin & maintenance 🛠️", width="stretch", key="nav_open_admin"):
            st.switch_page("pages/1_Admin_and_maintenance.py")
        st.divider()
        st.subheader("Cricsheet all_json + recent form")
        st.caption(
            f"Reads every numeric JSON file under **{config.CRICSHEET_ALL_JSON_DIR}** "
            f"(env **IPL_CRICSHEET_ALL_JSON_DIR**). Button-driven only."
        )
        if st.button(
            "Ingest Cricsheet all_json archive",
            help="Walks all numeric *.json under all_json. Very large folders can take a long time.",
            key="predict_sidebar_all_json_ingest",
        ):
            with st.spinner("Ingesting all_json archive…"):
                s_all = cricsheet_all_ingest.run_cricsheet_all_archive_ingest()
            processed_run = (
                int(s_all.matches_inserted)
                + int(s_all.matches_resynced_duplicate)
                + int(s_all.matches_skipped_duplicate_url)
                + int(s_all.matches_skipped_duplicate)
                + int(s_all.matches_skipped_malformed)
            )
            st.success(
                f"JSON files in folder: **{s_all.json_files_seen}** · "
                f"processed this run: **{processed_run}** · "
                f"matches inserted: **{s_all.matches_inserted}** · "
                f"resynced (same teams+date): **{s_all.matches_resynced_duplicate}** · "
                f"skipped duplicate URL: **{s_all.matches_skipped_duplicate_url}** · "
                f"skipped duplicate match: **{s_all.matches_skipped_duplicate}** · "
                f"malformed: **{s_all.matches_skipped_malformed}**."
            )
            vfs = recent_form_cache.recent_form_validation_summary()
            st.caption(
                f"SQLite after ingest — distinct **T20** matches in DB: **{vfs['distinct_t20_matches_in_db']}** · "
                f"``player_match_stats`` rows: **{vfs['player_match_stats_rows']}** · "
                f"``player_recent_form_cache`` rows: **{vfs['player_recent_form_cache_rows']}** "
                f"(rebuild cache to refresh)."
            )
            if s_all.warnings:
                for w in s_all.warnings[:40]:
                    st.warning(w)
                if len(s_all.warnings) > 40:
                    st.caption(f"… and {len(s_all.warnings) - 40} more warnings.")
            with st.expander("Validation snapshot after ingest (JSON)"):
                st.json(vfs)

        if st.button(
            "Rebuild recent-form cache (T20)",
            help="Rebuilds player_recent_form_cache from player_match_stats + matches (SQLite only).",
            key="predict_sidebar_rf_cache_rebuild",
        ):
            with st.spinner("Rebuilding player_recent_form_cache …"):
                rs = recent_form_cache.rebuild_player_recent_form_cache()
            vfs = recent_form_cache.recent_form_validation_summary()
            st.success(
                f"Cache rows written: **{rs.get('players_cached', 0)}** · "
                f"``player_match_stats`` rows scanned: **{rs.get('rows_scanned', 0)}** · "
                f"distinct **T20** matches counted: **{rs.get('t20_distinct_matches', 0)}** · "
                f"errors: **{rs.get('errors', 0)}**."
            )
            st.caption(
                f"``player_recent_form_cache`` total rows: **{vfs['player_recent_form_cache_rows']}** · "
                f"distinct T20 matches in DB: **{vfs['distinct_t20_matches_in_db']}** · "
                f"reference date: **{rs.get('reference_as_of_date', '')}**."
            )
            with st.expander("Recent-form validation summary (JSON)"):
                st.json(vfs)

        if st.button(
            "Show recent-form validation summary",
            help="Read-only: distinct T20 matches, cache row count, PMS rows, sample players (no rebuild).",
            key="predict_sidebar_rf_validation_only",
        ):
            vfs = recent_form_cache.recent_form_validation_summary()
            st.success(
                f"Distinct **T20** matches in DB: **{vfs['distinct_t20_matches_in_db']}** · "
                f"``player_recent_form_cache`` rows: **{vfs['player_recent_form_cache_rows']}** · "
                f"``player_match_stats`` rows: **{vfs['player_match_stats_rows']}**."
            )
            with st.expander("Full validation summary (JSON)"):
                st.json(vfs)

        if st.button(
            "Rebuild prediction summary tables",
            help="Recomputes prediction_summary_* tables used to avoid runtime GROUP BY scans.",
            key="predict_sidebar_summary_rebuild",
        ):
            with st.spinner("Rebuilding prediction summary tables …"):
                ssum = db.rebuild_prediction_summary_tables()
            st.success(
                f"Prediction summaries rebuilt in **{ssum.get('elapsed_ms', 0)} ms** "
                f"(match_xi rows: **{ssum.get('match_xi_rows', 0)}**, "
                f"match_batting rows: **{ssum.get('match_batting_rows', 0)}**, "
                f"match_results rows: **{ssum.get('match_results_rows', 0)}**)."
            )
            with st.expander("Prediction summary rebuild details (JSON)"):
                st.json(ssum)

        st.divider()
        st.caption(
            "IPL Cricsheet readme backfill/sync, Stage-2 derive, SQLite audit, and scorecard tools: "
            "**Admin & maintenance**."
        )

    _sidebar_ms = (time.perf_counter() - _t_side) * 1000.0
    _t_body = time.perf_counter()
    def _img_html(path: str, *, width: int, square: bool = True) -> str:
        p = Path(path)
        if not p.is_file():
            return ""
        raw = p.read_bytes()
        probe = raw[:512].lstrip().lower()
        mime = "image/svg+xml" if probe.startswith(b"<svg") or b"<svg" in probe else "image/png"
        data = base64.b64encode(raw).decode("ascii")
        if square:
            style = (
                f"width:{int(width)}px;height:{int(width)}px;"
                "object-fit:contain;display:block;margin:0 auto;"
            )
        else:
            style = (
                f"max-width:{int(width)}px;width:100%;height:auto;"
                "display:block;margin:0 auto;"
            )
        return (
            f'<img src="data:{mime};base64,{data}" '
            f'style="{style}" />'
        )

    def _render_team_logo(slug: str, *, width: int = 40, show_label: bool = False, center: bool = True) -> None:
        lab = ipl_teams.label_for_slug(slug)
        logo_path = ipl_teams.team_logo_path_for_slug(slug)
        if logo_path:
            img = _img_html(logo_path, width=width, square=True)
            if img:
                wrapper_style = "text-align:center;" if center else ""
                st.markdown(f'<div style="{wrapper_style}">{img}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**{lab}**")
        else:
            st.markdown(f"**{lab}**")
        if show_label:
            align = "center" if center else "left"
            st.markdown(
                f'<div style="text-align:{align}; font-size:0.8rem; color:#6b7280;">{lab}</div>',
                unsafe_allow_html=True,
            )

    def _render_logo_row(slugs: list[str], *, width: int) -> None:
        cells: list[str] = []
        for slug in slugs:
            logo_path = ipl_teams.team_logo_path_for_slug(slug)
            if logo_path:
                img = _img_html(logo_path, width=width, square=True)
                cells.append(
                    '<div style="display:flex; justify-content:center; align-items:center; min-height:{0}px;">{1}</div>'.format(
                        int(width) + 18, img
                    )
                )
            else:
                lab = ipl_teams.label_for_slug(slug)
                cells.append(
                    '<div style="width:100%; min-height:{0}px; display:flex; align-items:center; '
                    'justify-content:center; font-size:12px; font-weight:600; text-align:center;">{1}</div>'.format(
                        int(width) + 18, lab
                    )
                )
        st.markdown(
            '<div style="display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); '
            'justify-items:center; align-items:center; column-gap:16px; row-gap:12px; width:100%;">'
            + "".join(cells)
            + "</div>",
            unsafe_allow_html=True,
        )

    def _render_loading_logo_shuffle(slugs: list[str], *, width: int) -> None:
        cells: list[str] = []
        for idx, slug in enumerate(slugs):
            logo_path = ipl_teams.team_logo_path_for_slug(slug)
            if logo_path:
                img = _img_html(logo_path, width=width, square=True)
                cells.append(
                    '<div style="display:flex; justify-content:center; align-items:center; '
                    'min-height:{0}px; animation:ipl-logo-pulse 1.8s ease-in-out infinite; animation-delay:{1:.2f}s;">{2}</div>'.format(
                        int(width) + 14, float(idx) * 0.12, img
                    )
                )
            else:
                lab = ipl_teams.label_for_slug(slug)
                cells.append(
                    '<div style="min-height:{0}px; display:flex; justify-content:center; align-items:center; '
                    'font-size:12px; font-weight:600; animation:ipl-logo-pulse 1.8s ease-in-out infinite; animation-delay:{1:.2f}s;">{2}</div>'.format(
                        int(width) + 14, float(idx) * 0.12, lab
                    )
                )
        st.markdown(
            '<div style="display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); '
            'justify-items:center; align-items:center; column-gap:14px; row-gap:10px; width:100%; margin-top:0.25rem;">'
            + "".join(cells)
            + "</div>",
            unsafe_allow_html=True,
        )

    if _LOCAL_IPL_LOGO_PATH.is_file():
        st.markdown(
            '<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:0.35rem; margin-bottom:0.2rem;">'
            + _img_html(str(_LOCAL_IPL_LOGO_PATH), width=220, square=False)
            + '<h1 style="text-align:center; margin:0.15rem 0 0 0;">IPL Predictor</h1>'
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<h1 style="text-align:center; margin:0.1rem 0;">IPL Predictor</h1>', unsafe_allow_html=True)

    # Initial-load IPL-only logo strip (no generic icons).
    slugs = list(ipl_teams.TEAM_SLUGS)
    row1 = slugs[:5]
    row2 = slugs[5:]
    _render_logo_row(row1, width=132)
    _render_logo_row(row2, width=132)

    st.subheader("Pre-match inputs")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        team_a_slug = st.selectbox(
            "Home",
            options=ipl_teams.TEAM_SLUGS,
            format_func=lambda s: ipl_teams.label_for_slug(s),
            key="select_team_a",
        )
    with r1c2:
        b_options = [s for s in ipl_teams.TEAM_SLUGS if s != team_a_slug]
        team_b_slug = st.selectbox(
            "Away",
            options=b_options,
            format_func=lambda s: ipl_teams.label_for_slug(s),
            key="select_team_b",
        )

    team_a_name = ipl_teams.label_for_slug(team_a_slug)
    team_b_name = ipl_teams.label_for_slug(team_b_slug)
    l1, l2 = st.columns(2)
    with l1:
        _render_team_logo(team_a_slug, width=118, show_label=True, center=True)
    with l2:
        _render_team_logo(team_b_slug, width=118, show_label=True, center=True)

    venue_options = list_venue_choices()
    v_labels = [f"{d} ({k})" for k, d in venue_options]
    v_keys = [k for k, _ in venue_options]
    v_pick = st.selectbox("Venue", range(len(v_labels)), format_func=lambda i: v_labels[i])
    venue_key = v_keys[v_pick]
    venue_custom = st.text_input("Venue override (optional free text)", value="")
    venue = resolve_venue(venue_custom.strip() or venue_key)

    md = st.date_input("Match date", value=date.today())
    time_options: list[tuple[str, dt_time]] = [
        ("Afternoon match (15:30 IST)", dt_time(15, 30)),
        ("Night match (19:30 IST)", dt_time(19, 30)),
    ]
    default_idx = 1
    mt_idx = st.selectbox(
        "Match start slot (IST)",
        options=range(len(time_options)),
        index=default_idx,
        format_func=lambda i: time_options[i][0],
        key="match_start_slot_ist",
    )
    mt = time_options[int(mt_idx)][1]
    match_time_ist = time_utils.combine_date_time_ist(md, mt)

    unavailable = st.text_area(
        "Unavailable players (one name per line — matched loosely)",
        height=100,
        key="unavailable_input",
    )

    _maybe_fetch_squad(team_a_slug, "a")
    _maybe_fetch_squad(team_b_slug, "b")

    st.subheader("Squads")
    fb1, fb2 = st.columns(2)
    with fb1:
        if st.button("Refresh squads from IPLT20", help="Re-download both squads from official pages"):
            st.session_state["_fetch_squads_requested"] = True
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

    toss_labels = [lbl for _k, lbl in predictor.TOSS_SCENARIO_OPTIONS]
    toss_keys = [k for k, _lbl in predictor.TOSS_SCENARIO_OPTIONS]
    toss_ix = st.selectbox(
        "Toss scenario",
        range(len(toss_labels)),
        format_func=lambda i: toss_labels[i],
        help="Unknown = neutral toss context. Known toss applies the selected innings scenario.",
    )
    toss_key = toss_keys[int(toss_ix)]

    if st.button("Run prediction", type="primary"):
        _t_run_btn = time.perf_counter()
        loading_slot = st.empty()
        with loading_slot.container():
            st.markdown('<h3 style="text-align:center;">Building prediction...</h3>', unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align:center; color:#6b7280; margin-bottom:0.6rem;">Loading uses IPL team logos only.</div>',
                unsafe_allow_html=True,
            )
            ls1, ls2 = st.columns(2)
            with ls1:
                _render_team_logo(team_a_slug, width=118, show_label=True, center=True)
            with ls2:
                _render_team_logo(team_b_slug, width=118, show_label=True, center=True)
            _render_loading_logo_shuffle(list(ipl_teams.TEAM_SLUGS), width=72)

        _t_wx = time.perf_counter()
        w = weather.fetch_weather(venue, match_time_ist)
        audit_profile.append_session_audit_event(
            "prediction_run",
            "weather.fetch_weather",
            (time.perf_counter() - _t_wx) * 1000.0,
            extra={"venue_key": getattr(venue, "key", "")},
        )
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
        _t_pred = time.perf_counter()
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
        audit_profile.append_session_audit_event(
            "prediction_run",
            "predictor.run_prediction",
            (time.perf_counter() - _t_pred) * 1000.0,
        )
        if audit_profile.audit_enabled():
            st.session_state["_audit_last_prediction_audit"] = result.get("audit_prediction")
        loading_slot.empty()
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
        predict_ui_render.render_stored_prediction_results(
            r,
            show_advanced_prediction_debug=False,
            selection_debug_top15_for_side=_selection_debug_top15_for_side,
        )

    _main_body_ms = (time.perf_counter() - _t_body) * 1000.0
    _full_main_ms = (time.perf_counter() - _t_main) * 1000.0
    if audit_profile.audit_enabled():
        st.session_state["_audit_startup_breakdown"] = {
            "module_imports_ms": round((_APP_IMPORT_DONE - _APP_IMPORT_T0) * 1000.0, 2),
            "main_set_page_config_ms": round(_set_page_ms, 2),
            "main_db_ensure_schema_wall_ms": round(_db_ensure_ms, 2),
            "db_init_schema_cache_miss_ms": st.session_state.get("_audit_db_init_schema_cache_miss_ms"),
            "main_ensure_squad_state_ms": round(_ensure_squad_ms, 2),
            "main_sidebar_construct_ms": round(_sidebar_ms, 2),
            "main_column_body_ms": round(_main_body_ms, 2),
            "main_function_total_this_rerun_ms": round(_full_main_ms, 2),
        }


if __name__ == "__main__":
    main()
