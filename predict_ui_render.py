"""Render stored prediction results in Streamlit (no prediction engine calls)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd
import streamlit as st

import config
import history_sync

_perf_logger = logging.getLogger("ipl_predictor.perf")


def render_stored_prediction_results(
    r: dict[str, Any],
    *,
    show_advanced_prediction_debug: bool,
    selection_debug_top15_for_side: Callable[..., Any],
) -> None:
    cond = r["conditions"]
    disp_a = r["team_a"]["name"]
    disp_b = r["team_b"]["name"]
    hlw = (r.get("history_sync_debug") or {}).get("local_history_warning")
    if hlw:
        st.warning(hlw)
    xv = r.get("xi_validation") or {}
    for wmsg in xv.get("strict_validation_warnings") or []:
        if not wmsg:
            continue
        low = str(wmsg).lower()
        if "major_linkage_failure" in low or "sqlite history snapshot failed" in low:
            st.error(str(wmsg))
        else:
            st.warning(str(wmsg))

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

    conf = r.get("prediction_confidence") or {}
    st.subheader("Prediction confidence")
    st.metric(
        "Model confidence",
        f"{100 * float(conf.get('score', 0)):.1f}%",
        help="Rule-based: DB depth, XI history coverage, perspective agreement, strength separation.",
    )
    with st.expander("Optional: confidence breakdown & timing", expanded=False):
        st.json(conf)
        ptiming = r.get("prediction_timing_ms")
        if ptiming:
            st.caption("From ``IPL_PREDICTION_TIMING=true``")
            st.json(ptiming)

    wp = r["win_probability"]
    eng = r.get("win_probability_engine") or {}
    ts = r.get("toss_scenario") or {}
    st.subheader("Win probability (rule-based engine)")
    st.caption(
        "Weighted factors: head-to-head, venue record, XI strength, batting order strength, "
        "phase bowling, top-order matchups, weather/conditions, toss innings-role (chase/defend history), "
        "and venue chase bias with dew/night tilt. Clamped to 30–70%."
    )
    headline_help = (
        "Uses **selected toss** when known; otherwise matches **neutral toss** (average of the two innings orders)."
    )
    st.metric(
        f"P({disp_a} wins) — headline",
        f"{100 * wp['team_a_win']:.1f}%",
        help=headline_help,
    )
    if eng.get("team_a_win_pct_neutral_toss") is not None:
        st.metric(
            "P(Team A wins) — neutral toss (avg scenarios)",
            f"{float(eng.get('team_a_win_pct_neutral_toss', 50)):.1f}%",
            help="Unconditional on toss: mean of P(A|A bats first) and P(A|B bats first).",
        )
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
                st.dataframe(pd.DataFrame(sqa), use_container_width=True, hide_index=True)
            if sqb:
                st.caption(disp_b)
                st.dataframe(pd.DataFrame(sqb), use_container_width=True, hide_index=True)
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
                            st.dataframe(pd.DataFrame(hu), use_container_width=True, hide_index=True)
                        else:
                            st.caption(
                                f"**{len(hu)}** rows — enable “Render full prediction-layer tables” above."
                            )
                    sc = block.get("scoring_breakdown_per_player") or []
                    if sc:
                        st.markdown("*Scoring breakdown (scored squad — history vs composite vs selection)*")
                        st.dataframe(pd.DataFrame(sc), use_container_width=True, hide_index=True)
                    omit = block.get("omitted_from_playing_xi") or []
                    if omit:
                        st.markdown("*High composite / selection rank — not in playing XI*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(omit), use_container_width=True, hide_index=True)
                        else:
                            st.caption(f"**{len(omit)}** rows — enable full tables above.")
                    bod = block.get("xi_batting_order_diagnostics") or []
                    if bod:
                        st.markdown("*Selected XI — batting-order diagnostics*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(bod), use_container_width=True, hide_index=True)
                        else:
                            st.caption(f"**{len(bod)}** rows — enable full tables above.")
                    imp = block.get("impact_sub_ranking") or []
                    if imp:
                        st.markdown("*Impact sub ranking (model components)*")
                        if heavy_pld:
                            st.dataframe(pd.DataFrame(imp), use_container_width=True, hide_index=True)
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
                st.dataframe(df_dbg, use_container_width=True, hide_index=True)
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

    if eng:
        fav = eng.get("overall_favourite", "")
        st.markdown(f"**Overall predicted favourite:** {fav}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                f"If {disp_a} bats first — P({disp_a} wins)",
                f"{eng.get('team_a_win_pct_if_a_bats_first', 0):.1f}%",
            )
        with c2:
            st.metric(
                f"If {disp_b} bats first — P({disp_a} wins)",
                f"{eng.get('team_a_win_pct_if_b_bats_first', 0):.1f}%",
            )
        st.markdown("**Why (short read)**")
        st.write(eng.get("explanation", ""))
        if show_advanced_prediction_debug:
            with st.expander("Factor scores by scenario (0–100 per side)"):
                st.json(eng.get("scenario_factors", {}))
    st.caption(
        f"If **{disp_a} bats first**: {100 * r['win_if_team_a_bats_first']['team_a_win']:.1f}% · "
        f"If **{disp_a} bowls first** (i.e. {disp_b} bats first): "
        f"{100 * r['win_if_team_a_bowls_first']['team_a_win']:.1f}%"
    )
    if show_advanced_prediction_debug:
        with st.expander("Legacy logistic win model (comparison)"):
            leg = r.get("win_probability_logistic") or {}
            if leg.get("marginal"):
                m = leg["marginal"]
                st.metric(
                    "Logistic P(A) (uses toss scenario if set)",
                    f"{100 * float(m.get('team_a_win', 0)):.1f}%",
                )

    te = r["toss_effects"]
    st.subheader("Toss / innings leverage (Team A perspective)")
    st.write(
        {
            "Bat-first structural edge": round(te["bat_first_edge_team_a"], 4),
            "Bowl-first (chase) edge": round(te["bowl_first_edge_team_a"], 4),
            "Venue bat-first prior": round(te["venue_bat_first_prior"], 4),
            "Dew chase tilt": round(te["dew_chase_factor"], 4),
            "Venue chase win share (learned)": round(te.get("venue_chase_win_share", 0.5), 4),
            "Chase-prior logit (when A chases)": round(te.get("chase_boost_logit_applied", 0.0), 5),
        }
    )
    st.caption(
        "Positive bat-first edge favours defending a total; positive bowl-first edge favours chasing "
        "under the model’s dew and venue read."
    )

    st.subheader("Predicted line-ups & impact subs")
    tc1, tc2 = st.columns(2)
    for col, side_key, tname in (
        (tc1, "team_a", disp_a),
        (tc2, "team_b", disp_b),
    ):
        with col:
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
            xi_df = pd.DataFrame(r[side_key]["xi"])
            if not xi_df.empty:
                st.dataframe(xi_df, use_container_width=True, hide_index=True)
            hist_cols = [
                c
                for c in (
                    "name",
                    "bat_order",
                    "selection_score",
                    "history_xi_score",
                    "history_batting_ema",
                    "batting_order_source",
                    "batting_order_final",
                    "used_current_season_history",
                    "used_prior_season_fallback",
                    "used_venue_history",
                    "fallback_heuristics_only",
                    "recent5_xi_rate",
                    "venue_xi_rate",
                    "prior_season_xi_rate",
                    "xi_selection_tier",
                    "xi_used_prior_season_rows",
                )
                if c in xi_df.columns
            ]
            if show_advanced_prediction_debug and hist_cols and not xi_df.empty:
                with st.expander(f"{r[side_key]['name']} — XI / batting history signals"):
                    st.dataframe(
                        xi_df[hist_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
            st.markdown("**Batting order**")
            st.write(" → ".join(r[side_key]["batting_order"]))
            st.markdown("**Impact subs (5)**")
            st.dataframe(
                pd.DataFrame(r[side_key]["impact_subs"]),
                use_container_width=True,
                hide_index=True,
            )
