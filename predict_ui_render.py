"""Render stored prediction results in Streamlit (no prediction engine calls)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd
import streamlit as st

import audit_profile
import config
import db
import history_sync
import ipl_teams

_perf_logger = logging.getLogger("ipl_predictor.perf")


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
            st.markdown("**Impact subs (5)**")
            st.dataframe(
                pd.DataFrame(r[side_key]["impact_subs"]),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Historical Matchups")

    def _matchup_snapshot() -> dict[str, Any]:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT matchup_summaries_updated_at, direct_delivery_rows_seen, notes "
                "FROM matchup_summary_rebuild_state WHERE id=1"
            ).fetchone()
            bbms = conn.execute("SELECT COUNT(*) AS n FROM batter_bowler_matchup_summary").fetchone()
            bvbt = conn.execute("SELECT COUNT(*) AS n FROM batter_vs_bowling_type_summary").fetchone()
            bvhs = conn.execute("SELECT COUNT(*) AS n FROM bowler_vs_batting_hand_summary").fetchone()
            bvbsp = conn.execute("SELECT COUNT(*) AS n FROM batter_vs_spin_pace_summary").fetchone()
            bvbph = conn.execute("SELECT COUNT(*) AS n FROM batter_vs_phase_summary").fetchone()
            bbb = conn.execute("SELECT COUNT(*) AS n FROM match_ball_by_ball").fetchone()
        updated_at = float((row["matchup_summaries_updated_at"] if row else 0.0) or 0.0)
        return {
            "matchup_summaries_updated_at": updated_at,
            "direct_delivery_rows_seen": int((row["direct_delivery_rows_seen"] if row else 0) or 0),
            "notes": str((row["notes"] if row else "") or ""),
            "row_counts": {
                "match_ball_by_ball": int((bbb["n"] if bbb else 0) or 0),
                "batter_bowler_matchup_summary": int((bbms["n"] if bbms else 0) or 0),
                "batter_vs_bowling_type_summary": int((bvbt["n"] if bvbt else 0) or 0),
                "bowler_vs_batting_hand_summary": int((bvhs["n"] if bvhs else 0) or 0),
                "batter_vs_spin_pace_summary": int((bvbsp["n"] if bvbsp else 0) or 0),
                "batter_vs_phase_summary": int((bvbph["n"] if bvbph else 0) or 0),
            },
        }

    def _xi_key_map(side_key: str) -> tuple[list[str], dict[str, str]]:
            xi_rows = r.get(side_key, {}).get("xi") or []
            keys: list[str] = []
            disp: dict[str, str] = {}
            for row in xi_rows:
                if not isinstance(row, dict):
                    continue
                pk = str(row.get("player_key") or "").strip().lower()
                nm = str(row.get("name") or "").strip()
                if pk and pk not in disp:
                    disp[pk] = nm or pk
                if pk and pk not in keys:
                    keys.append(pk)
            return keys[:11], disp

    keys_a, disp_a_map = _xi_key_map("team_a")
    keys_b, disp_b_map = _xi_key_map("team_b")
    disp_all = {**disp_a_map, **disp_b_map}

    def _name_for_key(k: str) -> str:
        kk = str(k or "").strip().lower()
        return disp_all.get(kk, kk or "—")

    def _top_pairs(
        bat_keys: list[str], bowl_keys: list[str], *, limit: int = 20
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
        dfp = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        if dfp.empty:
            return dfp
        dfp["batter"] = dfp["batter_key"].map(_name_for_key)
        dfp["bowler"] = dfp["bowler_key"].map(_name_for_key)
        try:
            dfp["dismissals_per_100_balls"] = (
                100.0 * dfp["dismissals"].astype(float) / dfp["balls"].clip(lower=1).astype(float)
            ).round(3)
        except Exception:
            dfp["dismissals_per_100_balls"] = None
        return dfp[
            [
                "bowler",
                "batter",
                "balls",
                "dismissals",
                "dismissals_per_100_balls",
                "strike_rate",
                "dot_ball_pct",
                "sample_size_confidence",
            ]
        ]

    def _bat_when_phase(bat_keys: list[str], *, limit: int = 40) -> pd.DataFrame:
        if not bat_keys:
            return pd.DataFrame()
        qm = ",".join("?" * len(bat_keys))
        with db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT batter_key, bowling_phase, balls, strike_rate, dot_ball_pct,
                       sample_size_confidence
                FROM batter_vs_phase_summary
                WHERE batter_key IN ({qm})
                ORDER BY sample_size_confidence DESC, strike_rate DESC, balls DESC
                LIMIT ?
                """,
                [*bat_keys, int(limit)],
            ).fetchall()
        dfp = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        if dfp.empty:
            return dfp
        dfp["batter"] = dfp["batter_key"].map(_name_for_key)
        return dfp[
            [
                "batter",
                "bowling_phase",
                "balls",
                "strike_rate",
                "dot_ball_pct",
                "sample_size_confidence",
            ]
        ]

    def _bat_vs_spin_pace(bat_keys: list[str], *, limit: int = 40) -> pd.DataFrame:
        if not bat_keys:
            return pd.DataFrame()
        qm = ",".join("?" * len(bat_keys))
        with db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT batter_key, pace_spin_bucket, balls, strike_rate, dot_ball_pct,
                       sample_size_confidence
                FROM batter_vs_spin_pace_summary
                WHERE batter_key IN ({qm})
                ORDER BY sample_size_confidence DESC, balls DESC
                LIMIT ?
                """,
                [*bat_keys, int(limit)],
            ).fetchall()
        dfp = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        if dfp.empty:
            return dfp
        dfp["batter"] = dfp["batter_key"].map(_name_for_key)
        return dfp[
            [
                "batter",
                "pace_spin_bucket",
                "balls",
                "strike_rate",
                "dot_ball_pct",
                "sample_size_confidence",
            ]
        ]

    if not keys_a or not keys_b:
        st.info("No historical matchup rows available for this prediction yet.")
    else:
        with st.expander("Matchup recommendations", expanded=False):
            st.markdown("**Who to bowl to whom**")
            ca, cb = st.columns(2)
            with ca:
                df_ab = _top_pairs(keys_b, keys_a, limit=20)
                if df_ab.empty:
                    st.caption("No direct historical rows available.")
                else:
                    st.dataframe(df_ab, use_container_width=True, hide_index=True)
            with cb:
                df_ba = _top_pairs(keys_a, keys_b, limit=20)
                if df_ba.empty:
                    st.caption("No direct historical rows available.")
                else:
                    st.dataframe(df_ba, use_container_width=True, hide_index=True)

            st.markdown("**Who to bat when**")
            df_phase = _bat_when_phase([*keys_a, *keys_b], limit=60)
            if df_phase.empty:
                st.caption("No historical phase rows available.")
            else:
                st.dataframe(df_phase, use_container_width=True, hide_index=True)

            st.markdown("**Who to bat against**")
            df_sp = _bat_vs_spin_pace([*keys_a, *keys_b], limit=60)
            if df_sp.empty:
                st.caption("No historical type rows available.")
            else:
                st.dataframe(df_sp, use_container_width=True, hide_index=True)
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
            st.dataframe(df_dbg, use_container_width=True, hide_index=True)
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
