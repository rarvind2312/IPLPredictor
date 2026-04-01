"""
Admin & maintenance: IPL Cricsheet readme ingest, derive, SQLite audit, scorecard ingest.

**all_json** ingest and **recent-form cache** controls are on the **Predict** page sidebar.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

import streamlit as st

import config
import cricsheet_ingest
import db
import ipl_teams
import learner
import predictor
import stage_derive
import stage1_audit
import streamlit_db_init
from parsers.router import parse_scorecard

_perf_logger = logging.getLogger("ipl_predictor.perf")


def _session_squads_and_labels() -> tuple[str, str, str, str]:
    """Squads and display names from main Predict page session (if available)."""
    squad_a = str(st.session_state.get("sq_a") or "")
    squad_b = str(st.session_state.get("sq_b") or "")
    sa = st.session_state.get("select_team_a")
    sb = st.session_state.get("select_team_b")
    try:
        team_a_name = ipl_teams.label_for_slug(sa) if sa else "Team A"
    except Exception:  # noqa: BLE001
        team_a_name = "Team A"
    try:
        team_b_name = ipl_teams.label_for_slug(sb) if sb else "Team B"
    except Exception:  # noqa: BLE001
        team_b_name = "Team B"
    return squad_a, squad_b, team_a_name, team_b_name


def main() -> None:
    st.set_page_config(page_title="Admin — IPL Predictor", layout="wide")
    _t0 = time.perf_counter()
    streamlit_db_init.ensure_db_schema_initialized(streamlit_db_init.db_init_signature())
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        _perf_logger.info(
            "admin_page_db_ensure_ms=%.2f",
            (time.perf_counter() - _t0) * 1000.0,
        )

    with st.sidebar:
        if st.button("Back to Predict 🏏", use_container_width=True, key="nav_back_predict"):
            st.switch_page("app.py")
        st.caption("**Admin & maintenance** (this page)")
        st.divider()

    st.title("Admin & maintenance")
    st.caption(
        "IPL Cricsheet readme ingest, profile derive, SQLite audit/repair, and optional scorecard ingest. "
        "**all_json** ingest and **recent-form cache** live on the **Predict** page sidebar."
    )

    squad_a, squad_b, team_a_name, team_b_name = _session_squads_and_labels()
    if squad_a.strip() or squad_b.strip():
        st.info(
            f"Using squads from **Predict** session state when applicable ({team_a_name} vs {team_b_name}). "
            "Adjust teams and squads on **Predict**, then return here."
        )

    st.subheader("Cricsheet ingest (Stage 1)")
    st.caption(
        f"Reads **{config.DATA_DIR / 'readme.txt'}** (or other readme candidates) and "
        f"**{config.CRICSHEET_JSON_DIR}/*.json**. Writes normalized rows into SQLite only — "
        "no learning aggregates in this step. "
        f"Default rebuild window: last **{config.CRICSHEET_HISTORY_SEASON_COUNT}** seasons; "
        f"full archive also via checkbox or env **IPL_CRICSHEET_FULL_ARCHIVE_INGEST=true**."
    )
    if st.button("Initial Backfill Cricsheet", help="Scan the JSON folder; ingest every match not yet in the DB."):
        with st.spinner("Ingesting Cricsheet archive…"):
            s = cricsheet_ingest.run_initial_cricsheet_backfill()
        st.success(
            f"Inserted **{s.matches_inserted}** matches · skipped (already stored) **{s.matches_skipped_duplicate}** · "
            f"skipped (malformed) **{s.matches_skipped_malformed}** · "
            f"player stat rows **{s.player_stats_rows_inserted}** · "
            f"batting-position rows **{s.batting_position_rows_inserted}** · "
            f"phase rows **{s.phase_rows_inserted}**."
        )
        st.caption(
            f"Readme rows: **{s.readme_rows_total}** · JSON files on disk: **{s.json_files_on_disk}** · "
            f"readme index without JSON file: **{s.readme_rows_missing_json}**."
        )
        if s.warnings:
            for w in s.warnings[:50]:
                st.warning(w)
            if len(s.warnings) > 50:
                st.caption(f"… and {len(s.warnings) - 50} more warnings.")

    if st.button("Sync New Cricsheet Matches", help="Ingest only JSON files whose numeric id is not yet in SQLite."):
        with st.spinner("Syncing new Cricsheet JSON files…"):
            s = cricsheet_ingest.run_sync_new_cricsheet_matches()
        st.success(
            f"Inserted **{s.matches_inserted}** matches · skipped **{s.matches_skipped_duplicate}** · "
            f"malformed **{s.matches_skipped_malformed}** · player stat rows **{s.player_stats_rows_inserted}** · "
            f"batting-position rows **{s.batting_position_rows_inserted}** · phase rows **{s.phase_rows_inserted}**."
        )
        if s.warnings:
            for w in s.warnings[:40]:
                st.warning(w)

    st.caption(
        f"**all_json + recent-form cache** — use the **Predict** page sidebar (same controls as the checkpoint)."
    )

    with st.expander("Reset database (destructive)", expanded=False):
        st.caption(
            f"Deletes **{config.DB_PATH}** and WAL/SHM. Close other tabs using this DB. "
            "After wipe, open **Predict** once to re-init schema if needed."
        )
        sw = st.checkbox(
            "I understand this wipes all data (cannot undo).",
            value=False,
            key="admin_sidebar_db_wipe_confirm",
        )
        if st.button(
            "Delete SQLite DB and recreate empty schema",
            disabled=not sw,
            key="admin_sidebar_db_wipe_go",
        ):
            out = db.remove_sqlite_database_files()
            db.init_schema()
            st.success(f"Removed **{out.get('removed_paths') or []}** · empty schema OK.")
            st.warning(
                "Re-ingest IPL readme JSON from this page as needed; use **Predict** sidebar for **all_json** "
                "and **Rebuild recent-form cache (T20)**, then derive."
            )
            streamlit_db_init.ensure_db_schema_initialized.clear()

    st.divider()
    st.subheader("Derive profiles (Stage 2)")
    st.caption(
        f"Uses **SQLite only** (match years ≥ **{stage_derive.derive_min_season_year()}**, "
        f"last **{config.DERIVE_HISTORY_SEASONS}** seasons). Does **not** read Cricsheet JSON."
    )
    if st.button("Rebuild Profiles", help="Refresh player_franchise_features + player_profiles + team/venue patterns."):
        with st.spinner("Deriving from SQLite…"):
            dr = stage_derive.run_rebuild_profiles()
        st.success(
            f"Player profile rows **{dr.player_profiles_built}** · franchise feature upserts **{dr.franchise_feature_players_touched}** · "
            f"team summaries **{dr.team_derived_summary_rows}** · venue/team pattern pairs **{dr.venue_team_pattern_rows}** / "
            f"selection rows **{dr.team_selection_rows}**."
        )
        st.caption(
            f"Sparse-history players (< {config.DERIVE_SPARSE_PLAYER_SAMPLES} XI matches): **{dr.sparse_history_players}** · "
            f"Low-confidence profiles (≤ {config.DERIVE_FALLBACK_CONFIDENCE_MAX}): **{dr.fallback_profile_players}**."
        )
        with st.expander("Derive debug (row counts + samples)"):
            snap = stage_derive.derive_debug_snapshot()
            st.json(
                {
                    "min_season_year": dr.min_season_year,
                    "table_row_counts": snap,
                    "sparse_player_keys_sample": dr.sparse_player_keys_sample,
                    "fallback_profile_keys_sample": dr.fallback_player_keys_sample,
                    "warnings": dr.warnings,
                }
            )

    if st.button("Rebuild H2H Patterns", help="Recompute head_to_head_patterns from fixtures in the derive window."):
        with st.spinner("Rebuilding head-to-head patterns…"):
            h2 = stage_derive.run_rebuild_h2h_patterns()
        st.success(f"Head-to-head pattern pairs **{h2.head_to_head_pattern_rows}** (season floor **{h2.min_season_year}**).")
        with st.expander("H2H derive debug"):
            st.json(stage_derive.derive_debug_snapshot())

    st.divider()
    st.caption("One-time repair for DBs created before `matches` / `team_match_*`.")
    if st.button("Backfill XI history from stored payloads"):
        n = db.backfill_history_tables_from_results(limit=500)
        st.success(f"Processed **{n}** past matches into history tables.")

    st.divider()
    st.subheader("Stage 1 — SQLite audit & Cricsheet repair")
    st.caption(
        "Diagnose raw history tables vs local Cricsheet readme/JSON. Uses squads from **Predict** session when set."
    )
    if st.button("Show raw counts + readme/SQLite coverage (all franchises)", key="admin_s1_audit_bundle"):
        st.json(stage1_audit.full_stage1_audit_bundle())
    if st.button("Show batting-order vs player_batting_positions (recent matches)", key="admin_s1_audit_bat"):
        with db.connection() as conn:
            st.json(stage1_audit.batting_position_ingest_sample(conn, limit=10))
    if st.button("Show canonical key checks + sample rows", key="admin_s1_audit_canon"):
        with db.connection() as conn:
            st.json(
                {
                    "consistency": stage1_audit.canonical_key_consistency(conn),
                    "samples": stage1_audit.canonical_key_sample_rows(conn, per_table=5),
                }
            )
    clear_cricsheet_first = st.checkbox(
        "Delete existing Cricsheet `match_results` (and dependent raw rows) before rebuild",
        value=True,
        key="admin_s1_rebuild_clear",
    )
    full_archive_ingest = st.checkbox(
        "Full IPL archive rebuild (all readme IPL rows with JSON on disk, not last-N seasons only)",
        value=bool(getattr(config, "CRICSHEET_FULL_ARCHIVE_INGEST", False)),
        key="admin_s1_full_archive",
    )
    if st.button("Show SQLite calendar audit (Cricsheet-derived matches)", key="admin_s1_cal_audit"):
        st.json(db.sqlite_matches_temporal_audit(cricsheet_derived_only=True))
    if st.button("Rebuild Raw Cricsheet Ingest", key="admin_s1_rebuild_raw"):
        with st.spinner("Rebuilding raw Cricsheet tables…"):
            rb = cricsheet_ingest.run_rebuild_raw_cricsheet_ingest(
                clear_first=clear_cricsheet_first,
                full_archive_ingest=full_archive_ingest,
            )
        st.success(
            f"Mode: **{rb.ingest_mode}** · readme rows considered: **{rb.readme_rows_in_window}** "
            f"(total IPL readme before filter: **{rb.readme_rows_total_before_filter}**) · "
            f"Cricsheet `match_results` cleared (if enabled): **{rb.cleared_match_results}** · "
            f"matches inserted this run: **{rb.matches_inserted}** · malformed: **{rb.matches_skipped_malformed}** · "
            f"readme rows with no JSON file: **{rb.matches_missing_json_on_disk}** · "
            f"payload player_stats rows: **{rb.player_stats_rows_from_payloads}** · "
            f"payload batting-position rows: **{rb.batting_position_rows_from_payloads}** · "
            f"payload phase rows: **{rb.phase_rows_from_payloads}**."
        )
        st.json(asdict(rb))
    if st.button("Verify current squad ↔ raw SQLite linkage (Teams A & B)", key="admin_s1_squad_link"):
        la = stage1_audit.squad_raw_history_linkage_for_team(squad_a, team_a_name, opponent_label=team_b_name)
        lb = stage1_audit.squad_raw_history_linkage_for_team(squad_b, team_b_name, opponent_label=team_a_name)
        st.json(
            {
                "team_a": la.get("summary"),
                "team_a_per_player": la.get("per_player"),
                "team_b": lb.get("summary"),
                "team_b_per_player": lb.get("per_player"),
            }
        )

    st.markdown("---")
    st.caption(
        f"**Reset database (destructive)** — removes **{config.DB_PATH}** (+ WAL/SHM)."
    )
    wipe_db = st.checkbox(
        "I understand this deletes all matches, derived profiles, and caches (cannot undo).",
        value=False,
        key="admin_s1_db_wipe_confirm",
    )
    if st.button(
        "Delete SQLite DB and recreate empty schema",
        disabled=not wipe_db,
        key="admin_s1_db_wipe_go",
    ):
        out = db.remove_sqlite_database_files()
        db.init_schema()
        st.success(
            "Database file removed and empty schema created. "
            f"Paths removed: **{out.get('removed_paths') or '(none — file was missing)'}**."
        )
        st.warning(
            "Next steps: **Initial Backfill** or **Sync** (IPL), **Ingest Cricsheet all_json** and "
            "**Rebuild recent-form cache (T20)** from the **Predict** sidebar, then **Rebuild Profiles** / **Rebuild H2H**."
        )
        streamlit_db_init.ensure_db_schema_initialized.clear()

    st.divider()
    st.subheader("Squad fetch parse debug (from Predict session)")
    for label, dbg_key in ((team_a_name, "squad_a_parse_debug"), (team_b_name, "squad_b_parse_debug")):
        dbg = st.session_state.get(dbg_key)
        st.markdown(f"**{label}**")
        if dbg is None:
            st.caption("No fetch yet — load squads on **Predict**.")
            continue
        st.write(
            {
                "raw_candidates": dbg.raw_candidate_count,
                "cleaned_players": dbg.cleaned_count,
                "methods": dbg.methods_used,
                "foreign_player_icon_hits": getattr(dbg, "foreign_player_icon_hits", 0),
            }
        )
        if getattr(dbg, "raw_extracted_preview", None):
            st.caption("Raw extracted preview (name | bucket hint)")
            st.code("\n".join(dbg.raw_extracted_preview[:25]), language="text")
        if dbg.rejected_sample:
            st.caption("Rejected invalid structured rows")
            st.code("\n".join(dbg.rejected_sample[:40]), language="text")

    st.divider()
    st.subheader("Batting-slot SQLite pipeline (sanity check)")
    st.caption("Inspect ``player_batting_positions`` vs current **Predict** session squad keys.")
    st.json(db.batting_positions_sqlite_pipeline_summary())

    ca = ipl_teams.franchise_label_for_storage(team_a_name)
    cb = ipl_teams.franchise_label_for_storage(team_b_name)
    sa_p = predictor.parse_squad_text(squad_a)
    sb_p = predictor.parse_squad_text(squad_b)
    ska = {learner.normalize_player_key(p.name) for p in sa_p}
    skb = {learner.normalize_player_key(p.name) for p in sb_p}
    ska.discard("")
    skb.discard("")
    st.json(
        {
            "franchise_a": ca,
            "franchise_b": cb,
            "pbp_row_counts_by_franchise": {
                ca: db.count_player_batting_positions_for_franchise(ca),
                cb: db.count_player_batting_positions_for_franchise(cb),
            },
            "squad_player_key_coverage_team_a": db.squad_pbp_coverage_for_franchise(ca, ska),
            "squad_player_key_coverage_team_b": db.squad_pbp_coverage_for_franchise(cb, skb),
        }
    )

    st.divider()
    st.subheader("After the match — manual scorecard ingest (optional)")
    st.caption(
        "Paste a scorecard URL (Cricbuzz, ESPNcricinfo, IPLT20). Separate from **Run prediction** on the Predict page."
    )
    score_url = st.text_input("Scorecard URL", key="admin_score_url")
    with st.expander("Optional learning hints (IST / overseas)"):
        st.caption("Use **IST** start hour and **actual overseas counts** from the scorecard when known.")
        ingest_hour = st.number_input(
            "Match start hour (IST, 0–23, leave -1 if unknown)",
            min_value=-1,
            max_value=23,
            value=-1,
            key="admin_ingest_hour",
        )
        ingest_os_a = st.number_input(
            "Overseas players in first-listed team XI (0–11, -1 unknown)",
            min_value=-1,
            max_value=11,
            value=-1,
            key="admin_ingest_os_a",
        )
        ingest_os_b = st.number_input(
            "Overseas players in second-listed team XI (0–11, -1 unknown)",
            min_value=-1,
            max_value=11,
            value=-1,
            key="admin_ingest_os_b",
        )

    if st.button("Parse & store", key="admin_ingest"):
        if not score_url.strip():
            st.error("Provide a URL.")
        elif db.match_exists_by_url(score_url.strip()):
            st.warning("This URL is already in the database.")
        else:
            payload = parse_scorecard(score_url.strip())
            ing = payload.get("ingestion") or {}
            if ing.get("warnings"):
                for w in ing["warnings"]:
                    st.warning(w)
            if ing.get("errors"):
                for e in ing["errors"]:
                    st.error(e)
            comp = ing.get("completeness") or {}
            missing = [k for k, ok in comp.items() if not ok]
            if missing:
                st.caption(f"Missing / incomplete: {', '.join(missing)}")

            if not ing.get("has_storable_content"):
                st.error("Nothing storable was extracted (fix URL or try another scorecard).")
            else:
                meta = payload.setdefault("meta", {})
                if ingest_hour >= 0:
                    meta["start_hour_local"] = int(ingest_hour)
                    meta["start_hour_timezone"] = "Asia/Kolkata"
                if ingest_os_a >= 0:
                    meta["overseas_in_xi_team_a"] = int(ingest_os_a)
                if ingest_os_b >= 0:
                    meta["overseas_in_xi_team_b"] = int(ingest_os_b)
                try:
                    mid, ins_status = db.insert_parsed_match(payload)
                    if ins_status == "duplicate_url":
                        st.warning("This URL is already in the database.")
                    elif ins_status == "duplicate_match":
                        st.warning(
                            f"This fixture (teams + date) is already stored as match id **{mid}** "
                            "(canonical dedupe — no duplicate rows)."
                        )
                    else:
                        stats = learner.ingest_payload(payload)
                        st.success(
                            f"Stored match id **{mid}** (`match_results` + `matches` / `team_match_xi` / "
                            f"`team_match_summary` when parsed). Updated **{stats['players_updated']}** "
                            "player learnings."
                        )
                    with st.expander("Parsed summary"):
                        st.json(
                            {
                                "teams": payload.get("teams"),
                                "venue": (payload.get("meta") or {}).get("venue"),
                                "date": (payload.get("meta") or {}).get("date"),
                                "winner": (payload.get("meta") or {}).get("winner"),
                                "playing_xi_teams": [x.get("team") for x in payload.get("playing_xi") or []],
                                "batting_order": payload.get("batting_order"),
                                "bowlers_used": payload.get("bowlers_used"),
                                "batting_innings": len(payload.get("batting") or []),
                                "bowling_innings": len(payload.get("bowling") or []),
                                "ingestion": ing,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Database write failed: {exc}")


main()
