"""
Read-only end-to-end pipeline audit for the IPL predictor (Streamlit / diagnostics).

**Not imported by the Streamlit app** after the history-model cleanup (re-enable from ``app.py`` if needed).

Does not alter scoring rules; orchestrates modules and reports PASS / FAIL / WARN per stage.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import config
import cricsheet_convert
import cricsheet_readme
import db
import history_linkage
import ipl_teams
import learner
import predictor
import squad_fetch
import time_utils
import weather
from venues import VenueProfile

STAGE_CODES = (
    "A_squad",
    "B_readme",
    "C_json_lookup",
    "D_json_parse",
    "E_sqlite_write",
    "F_history_linkage",
    "G_predictor_layer",
    "H_batting_order",
    "I_impact",
)

STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("A_squad", "A. Current squad fetch"),
    ("B_readme", "B. Cricsheet index / readme"),
    ("C_json_lookup", "C. JSON file lookup"),
    ("D_json_parse", "D. JSON parse & batting order derivation"),
    ("E_sqlite_write", "E. SQLite materialization (observed)"),
    ("F_history_linkage", "F. SQLite read / history linkage"),
    ("G_predictor_layer", "G. Predictor layer (selection scores)"),
    ("H_batting_order", "H. Batting order layer"),
    ("I_impact", "I. Impact subs layer"),
)


def _stage(
    *,
    ok: bool,
    warn: bool = False,
    code: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    errors: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    errs = list(errors or [])
    warns = list(warnings or [])
    if ok and not warn:
        status = "PASS"
    elif ok and warn:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "status": status,
        "failure_code": code,
        "details": details or {},
        "errors": errs,
        "warnings": warns,
    }


def _readme_h2h_rows(
    filtered: list[cricsheet_readme.CricsheetReadmeRow],
    canon_a: str,
    canon_b: str,
) -> list[cricsheet_readme.CricsheetReadmeRow]:
    out: list[cricsheet_readme.CricsheetReadmeRow] = []
    for r in filtered:
        if cricsheet_readme.row_involves_team_name(r, canon_a, canonical=True) and cricsheet_readme.row_involves_team_name(
            r, canon_b, canonical=True
        ):
            out.append(r)
    out.sort(key=lambda x: x.match_date, reverse=True)
    return out


def _pick_probe_readme_row(
    h2h_rows: list[cricsheet_readme.CricsheetReadmeRow],
    target: Optional[date],
) -> Optional[cricsheet_readme.CricsheetReadmeRow]:
    if not h2h_rows:
        return None
    if target is None:
        return h2h_rows[0]

    def _d(r: cricsheet_readme.CricsheetReadmeRow) -> Optional[date]:
        try:
            return datetime.strptime(r.match_date[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    scored: list[tuple[int, cricsheet_readme.CricsheetReadmeRow]] = []
    for r in h2h_rows:
        rd = _d(r)
        if rd is None:
            continue
        scored.append((abs((rd - target).days), r))
    if not scored:
        return h2h_rows[0]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _sqlite_table_counts_for_match(conn: Any, match_id: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for table, clause in (
        ("matches", "id = ?"),
        ("team_match_summary", "match_id = ?"),
        ("team_match_xi", "match_id = ?"),
        ("player_match_stats", "match_id = ?"),
        ("player_batting_positions", "match_id = ?"),
    ):
        try:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {clause}", (match_id,)).fetchone()
            out[table] = int(row["c"] if hasattr(row, "keys") else row[0])
        except Exception:
            out[table] = -1
    return out


def _franchise_totals(conn: Any, team_key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in ("team_match_xi", "player_match_stats", "player_batting_positions"):
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM {table} WHERE team_key = ?",
            (team_key,),
        ).fetchone()
        out[table] = int(row["c"] if hasattr(row, "keys") else row[0])
    return out


def _infer_top_causes(stages: dict[str, Any], pred: Optional[dict[str, Any]]) -> list[str]:
    causes: list[str] = []
    for key, label in (
        ("A_squad", "Squad fetch or parsing failed — XI inputs are unreliable."),
        ("B_readme", "Cricsheet readme missing or unreadable — no local index for JSON paths."),
        ("C_json_lookup", "Expected H2H (or union) JSON file missing under data/ipl_json."),
        ("D_json_parse", "Ball-by-ball JSON parse or batting-order derivation failed."),
        ("E_sqlite_write", "SQLite has little or no materialized history for this franchise/match."),
        ("F_history_linkage", "Squad player_keys do not join to team_match_xi / PBP rows (names or keys mismatch)."),
        ("G_predictor_layer", "Predictor run failed or selection scores dominated by composite fallback."),
        ("H_batting_order", "Many XI players use role_fallback for batting order (weak slot history)."),
        ("I_impact", "Impact layer returned sparse diagnostics (check toss / chase context)."),
    ):
        st = (stages.get(key) or {}).get("status")
        if st == "FAIL":
            causes.append(label)
        if len(causes) >= 3:
            return causes

    if pred:
        xv = pred.get("xi_validation") or {}
        for w in xv.get("strict_validation_warnings") or []:
            if w and "batting" in str(w).lower():
                causes.append(str(w)[:200])
                break
        pld = pred.get("prediction_layer_debug") or {}
        for side in ("team_a", "team_b"):
            bl = pld.get(side) or {}
            for row in bl.get("scoring_breakdown_per_player") or []:
                if not row.get("has_usable_history") and float(row.get("role_fallback_score") or 0) > 0.35:
                    causes.append(
                        "Several squad players lack usable SQLite/Cricsheet history — "
                        "selection_score leans on composite / role fallback."
                    )
                    break
            if len(causes) >= 3:
                break

    while len(causes) < 3:
        causes.append("No additional ranked causes inferred — review stage details below.")
        if len(causes) >= 3:
            break
    return causes[:3]


def run_full_pipeline_audit(
    *,
    team_a_name: str,
    team_b_name: str,
    venue: VenueProfile,
    squad_a_text: str,
    squad_b_text: str,
    unavailable_text: str = "",
    match_time_ist: datetime,
    audit_season_year: Optional[int] = None,
    audit_target_match_date: Optional[date] = None,
    toss_scenario_key: str = "unknown",
) -> dict[str, Any]:
    """
    Run stages A–I and return structured audit payload + summary.

    Stage G–I call ``predictor.run_prediction`` once (same as production) and only read outputs.
    """
    stages: dict[str, Any] = {}
    failure_chain: list[str] = []
    pred: Optional[dict[str, Any]] = None

    canon_a = ipl_teams.franchise_label_for_storage(team_a_name) or team_a_name.strip()
    canon_b = ipl_teams.franchise_label_for_storage(team_b_name) or team_b_name.strip()
    ck_a = ipl_teams.canonical_team_key_for_franchise(canon_a)
    ck_b = ipl_teams.canonical_team_key_for_franchise(canon_b)

    cur_season = int(audit_season_year or getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    n_seasons = int(getattr(config, "CRICSHEET_HISTORY_SEASON_COUNT", 5))

    # --- A. Squad fetch ---
    squad_a_url = squad_b_url = None
    sa_members = sb_members = []
    err_a = err_b = None
    slug_a = ipl_teams.slug_for_canonical_label(canon_a)
    slug_b = ipl_teams.slug_for_canonical_label(canon_b)
    if slug_a:
        squad_a_url = squad_fetch.SQUAD_URL_TEMPLATE.format(slug=slug_a)
        sa_members, err_a, _dbg_a = squad_fetch.fetch_squad_for_slug(slug_a)
    else:
        err_a = f"No IPL slug for franchise {canon_a!r}"
    if slug_b:
        squad_b_url = squad_fetch.SQUAD_URL_TEMPLATE.format(slug=slug_b)
        sb_members, err_b, _dbg_b = squad_fetch.fetch_squad_for_slug(slug_b)
    else:
        err_b = f"No IPL slug for franchise {canon_b!r}"

    parsed_a = predictor.parse_squad_text(squad_a_text)
    parsed_b = predictor.parse_squad_text(squad_b_text)
    filt_a = predictor.filter_unavailable(parsed_a, unavailable_text)
    filt_b = predictor.filter_unavailable(parsed_b, unavailable_text)

    a_ok = err_a is None and len(sa_members) > 0
    b_ok = err_b is None and len(sb_members) > 0
    a_errors = [f"team_a: {err_a}"] if err_a else []
    b_errors = [f"team_b: {err_b}"] if err_b else []
    if len(parsed_a) == 0:
        a_errors.append("team_a: parsed squad text is empty")
    if len(parsed_b) == 0:
        b_errors.append("team_b: parsed squad text is empty")
    text_ok = len(parsed_a) > 0 and len(parsed_b) > 0
    fetch_failed = err_a is not None or err_b is not None

    stages["A_squad"] = _stage(
        ok=text_ok,
        warn=fetch_failed or not a_ok or not b_ok,
        code="squad_fetch_failed" if fetch_failed else ("squad_text_empty" if not text_ok else None),
        details={
            "team_a_name": canon_a,
            "team_b_name": canon_b,
            "slug_a": slug_a,
            "slug_b": slug_b,
            "squad_url_team_a": squad_a_url,
            "squad_url_team_b": squad_b_url,
            "fetched_count_team_a": len(sa_members),
            "fetched_count_team_b": len(sb_members),
            "parsed_squad_text_size_team_a": len(parsed_a),
            "parsed_squad_text_size_team_b": len(parsed_b),
            "unavailable_filter_text_len": len((unavailable_text or "").strip()),
            "eligible_after_unavailable_team_a": len(filt_a),
            "eligible_after_unavailable_team_b": len(filt_b),
        },
        errors=a_errors + b_errors,
        warnings=[]
        if not fetch_failed
        else ["Live squad fetch failed — audit uses squad text areas only (if non-empty)."],
    )
    if stages["A_squad"]["status"] == "FAIL":
        failure_chain.append("squad_fetch_failed" if fetch_failed else "squad_text_empty")

    # --- B. Readme ---
    readme_path = cricsheet_readme.resolve_readme_path()
    readme_rows: list[cricsheet_readme.CricsheetReadmeRow] = []
    parse_err: Optional[str] = None
    if readme_path is None:
        parse_err = "readme_not_found"
    else:
        try:
            readme_rows = cricsheet_readme.parse_cricsheet_readme(readme_path)
        except Exception as exc:  # noqa: BLE001
            parse_err = f"{type(exc).__name__}: {exc}"

    years = cricsheet_readme.season_years_window(cur_season, n_seasons)
    filtered = (
        cricsheet_readme.filter_rows_by_seasons(readme_rows, years) if readme_rows else []
    )
    rows_a = cricsheet_readme.filter_rows_by_team_name(filtered, canon_a, canonical=True)
    rows_b = cricsheet_readme.filter_rows_by_team_name(filtered, canon_b, canonical=True)
    h2h_readme = _readme_h2h_rows(filtered, canon_a, canon_b)
    latest_h2h = h2h_readme[0].as_dict() if h2h_readme else None

    sqlite_h2h = db.h2h_fixtures_between_franchises(canon_a, canon_b, limit=40)
    latest_sqlite_h2h = sqlite_h2h[0] if sqlite_h2h else None

    stages["B_readme"] = _stage(
        ok=parse_err is None and len(readme_rows) > 0,
        warn=len(rows_a) == 0 or len(rows_b) == 0 or len(h2h_readme) == 0,
        code="readme_parse_failed" if parse_err else ("thin_index" if len(filtered) < 10 else None),
        details={
            "readme_path": str(readme_path) if readme_path else None,
            "readme_rows_total_ipl_male": len(readme_rows),
            "season_years_window": sorted(years, reverse=True),
            "audit_season_anchor_year": cur_season,
            "matches_team_a_last_n_seasons": len(rows_a),
            "matches_team_b_last_n_seasons": len(rows_b),
            "h2h_matches_readme_index": len(h2h_readme),
            "latest_h2h_readme_row": latest_h2h,
            "h2h_fixtures_in_sqlite": len(sqlite_h2h),
            "latest_prior_h2h_sqlite": latest_sqlite_h2h,
        },
        errors=[parse_err] if parse_err else [],
        warnings=[]
        if len(h2h_readme) > 0
        else ["no_h2h_match_found_in_readme_index_last_n_seasons"],
    )
    if stages["B_readme"]["status"] == "FAIL":
        failure_chain.append("readme_parse_failed")
    elif len(h2h_readme) == 0:
        failure_chain.append("no_h2h_match_found")

    # --- C & D. JSON probe (prefer H2H readme row; else latest match involving Team A) ---
    jdir = Path(getattr(config, "CRICSHEET_JSON_DIR", config.DATA_DIR / "ipl_json"))
    probe_row = _pick_probe_readme_row(h2h_readme, audit_target_match_date)
    if probe_row is None and rows_a:
        probe_row = sorted(rows_a, key=lambda x: x.match_date, reverse=True)[0]
    probe_mid = probe_row.match_id if probe_row else None
    h2h_id_set = {r.match_id for r in h2h_readme}
    probe_is_h2h_readme = bool(probe_mid and probe_mid in h2h_id_set)
    json_path = jdir / f"{probe_mid}.json" if probe_mid else None
    file_exists = bool(json_path and json_path.is_file())
    parse_payload: Optional[dict[str, Any]] = None
    json_err: Optional[str] = None
    d_details: dict[str, Any] = {}

    if not probe_mid:
        stages["C_json_lookup"] = _stage(
            ok=False,
            code="no_probe_match_id",
            details={
                "json_dir": str(jdir),
                "reason": "No readme row available for probe (no H2H and no Team A rows in window)",
            },
            errors=["Cannot build JSON path — expand Cricsheet readme coverage or season window."],
        )
        stages["D_json_parse"] = _stage(
            ok=False,
            code="json_parse_skipped",
            details={},
            errors=["Skipped — no JSON path."],
        )
        failure_chain.append("no_h2h_match_found")
    else:
        stages["C_json_lookup"] = _stage(
            ok=file_exists,
            code="json_file_missing" if not file_exists else None,
            details={
                "match_id": probe_mid,
                "json_path": str(json_path),
                "file_exists": file_exists,
                "probe_is_h2h_readme_row": probe_is_h2h_readme,
            },
            errors=[] if file_exists else [f"Missing file: {json_path}"],
        )
        if not file_exists:
            failure_chain.append("json_file_missing")

        if file_exists:
            try:
                parse_payload = cricsheet_convert.load_cricsheet_payload(
                    json_path, cricsheet_match_id=str(probe_mid)
                )
            except Exception as exc:  # noqa: BLE001
                json_err = f"{type(exc).__name__}: {exc}"

            if json_err:
                stages["D_json_parse"] = _stage(
                    ok=False,
                    code="json_parse_failed",
                    details={"path": str(json_path)},
                    errors=[json_err],
                )
                failure_chain.append("json_parse_failed")
            else:
                assert parse_payload is not None
                meta = parse_payload.get("meta") or {}
                teams = list(parse_payload.get("teams") or [])
                xi = parse_payload.get("playing_xi") or []
                bo = parse_payload.get("batting_order") or []
                bu = parse_payload.get("bowlers_used") or []
                ibo = parse_payload.get("innings_batting_orders") or []
                xi_by_team = {str(s.get("team") or ""): list(s.get("players") or [])[:11] for s in xi}
                bo_ok = bool(ibo) or any(len((b.get("order") or [])) > 0 for b in bo)
                d_details = {
                    "parsed_teams": teams,
                    "parsed_venue": meta.get("venue"),
                    "parsed_date": meta.get("date"),
                    "playing_xi_by_team": xi_by_team,
                    "batting_order_blocks": bo,
                    "innings_batting_orders_innings": len(ibo),
                    "bowlers_used_blocks": bu,
                }
                stages["D_json_parse"] = _stage(
                    ok=bo_ok,
                    warn=not bo_ok,
                    code="batting_order_extraction_empty" if not bo_ok else None,
                    details=d_details,
                    errors=[] if bo_ok else ["batting_order_extraction_failed (no innings_batting_orders or empty orders)"],
                    warnings=[] if bo_ok else ["Derived batting order empty — check deliveries in JSON."],
                )
                if not bo_ok:
                    failure_chain.append("batting_order_extraction_failed")
        else:
            stages["D_json_parse"] = _stage(
                ok=False,
                code="json_parse_skipped",
                details={},
                errors=["JSON file missing — parse not attempted."],
            )

    # --- E. SQLite write (observed) ---
    e_details: dict[str, Any] = {
        "probe_match_id": int(probe_mid) if (probe_mid and str(probe_mid).isdigit()) else None,
        "note": "Observational counts only — audit does not insert rows.",
    }
    try:
        with db.connection() as conn:
            e_details["franchise_totals_team_a"] = _franchise_totals(conn, ck_a)
            e_details["franchise_totals_team_b"] = _franchise_totals(conn, ck_b)
            if probe_mid and str(probe_mid).isdigit():
                mid_int = int(probe_mid)
                row = conn.execute(
                    "SELECT id FROM matches WHERE id = ? LIMIT 1",
                    (mid_int,),
                ).fetchone()
                if row:
                    e_details["counts_for_probe_match_id"] = _sqlite_table_counts_for_match(conn, mid_int)
                else:
                    e_details["counts_for_probe_match_id"] = None
                    e_details["probe_match_not_in_sqlite"] = True
    except Exception as exc:  # noqa: BLE001
        stages["E_sqlite_write"] = _stage(
            ok=False,
            code="sqlite_inspect_failed",
            details=e_details,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        failure_chain.append("sqlite_inspect_failed")
    else:
        tmx_a = (e_details.get("franchise_totals_team_a") or {}).get("team_match_xi", 0)
        tmx_b = (e_details.get("franchise_totals_team_b") or {}).get("team_match_xi", 0)
        thin = tmx_a < 3 and tmx_b < 3
        no_probe = bool(e_details.get("probe_match_not_in_sqlite"))
        stages["E_sqlite_write"] = _stage(
            ok=True,
            warn=thin or no_probe,
            code="sqlite_thin_or_probe_missing" if (thin or no_probe) else None,
            details=e_details,
            warnings=[]
            if not no_probe
            else [f"Probe match id {probe_mid} not in SQLite matches — run prediction / ingest to materialize."]
            + ([] if not thin else ["Both franchises have very few team_match_xi rows in SQLite (<3 each)."]),
        )
        if thin:
            failure_chain.append("sqlite_history_thin")

    # --- F. History linkage (canonical keys only; same helper as predictor / history_xi) ---
    f_rows: list[dict[str, Any]] = []
    f_summaries: list[dict[str, Any]] = []
    try:
        link_a = history_linkage.link_current_squad_to_history(
            filt_a, canon_a, opponent_canonical_label=canon_b
        )
        link_b = history_linkage.link_current_squad_to_history(
            filt_b, canon_b, opponent_canonical_label=canon_a
        )
        f_summaries = [link_a["summary"], link_b["summary"]]
        for r in link_a["per_player"]:
            row = dict(r)
            row["side"] = "A"
            f_rows.append(row)
        for r in link_b["per_player"]:
            row = dict(r)
            row["side"] = "B"
            f_rows.append(row)
    except Exception as exc:  # noqa: BLE001
        stages["F_history_linkage"] = _stage(
            ok=False,
            code="history_linkage_query_failed",
            details={"rows": f_rows, "summaries": f_summaries},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        failure_chain.append("history_linkage_failed")
    else:
        n_players = len(f_rows)
        no_link = sum(
            1
            for r in f_rows
            if int(r.get("team_match_xi_rows") or 0) < 1
            and int(r.get("player_match_stats_rows") or 0) < 1
        )
        frac_bad = no_link / max(1, n_players)
        health_a = (f_summaries[0] or {}).get("stage_f_team_health") if f_summaries else ""
        health_b = (f_summaries[1] or {}).get("stage_f_team_health") if len(f_summaries) > 1 else ""
        fail_hard = any((s or {}).get("stage_f_team_health") == "major_linkage_failure" for s in f_summaries)
        f_ok = not fail_hard
        fail_msg = (
            "Stage F major_linkage_failure: most core players lack a SQLite history key for one or both teams."
        )
        warn_soft = frac_bad > 0.02 or any(
            (s or {}).get("stage_f_team_health") not in ("healthy",) for s in f_summaries
        )
        stages["F_history_linkage"] = _stage(
            ok=f_ok,
            warn=warn_soft,
            code="history_linkage_weak" if not f_ok else None,
            details={
                "per_player": f_rows,
                "summaries_by_team": f_summaries,
                "stage_f_team_health": [health_a, health_b],
                "players_without_tmx_or_pms": no_link,
                "fraction": round(frac_bad, 4),
                "matched_current_squad_players": sum(int(s.get("matched_with_usable_history_rows") or 0) for s in f_summaries),
                "unmatched_current_squad_players": sum(int(s.get("unmatched_or_zero_history_rows") or 0) for s in f_summaries),
                "pct_current_squad_with_usable_history": [
                    s.get("pct_current_squad_with_usable_history") for s in f_summaries
                ],
            },
            errors=[fail_msg] if not f_ok else [],
            warnings=[]
            if not warn_soft
            else [
                f"{no_link}/{n_players} squad players lack team_match_xi + player_match_stats rows; "
                f"Stage F health: team_a={health_a!r}, team_b={health_b!r}."
            ],
        )
        if not f_ok:
            failure_chain.append("history_linkage_failed")

    # --- G, H, I via run_prediction ---
    try:
        w = weather.fetch_weather(venue, match_time_ist)
        pred = predictor.run_prediction(
            team_a_name,
            team_b_name,
            squad_a_text,
            squad_b_text,
            unavailable_text,
            venue,
            match_time_ist,
            w,
            toss_scenario_key=toss_scenario_key,
        )
    except Exception as exc:  # noqa: BLE001
        stages["G_predictor_layer"] = _stage(
            ok=False,
            code="run_prediction_failed",
            details={},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        stages["H_batting_order"] = _stage(
            ok=False,
            code="skipped_after_predictor_failure",
            details={},
            errors=["Skipped."],
        )
        stages["I_impact"] = _stage(
            ok=False,
            code="skipped_after_predictor_failure",
            details={},
            errors=["Skipped."],
        )
        failure_chain.append("prediction_run_failed")
    else:
        assert pred is not None
        pld = pred.get("prediction_layer_debug") or {}

        g_team_a = pld.get("team_a") or {}
        g_team_b = pld.get("team_b") or {}
        g_rows = (g_team_a.get("scoring_breakdown_per_player") or []) + (
            g_team_b.get("scoring_breakdown_per_player") or []
        )
        g_eligible = []
        for row in g_rows:
            g_eligible.append(
                {
                    "player_name": row.get("player_name"),
                    "canonical_player_key": row.get("canonical_player_key"),
                    "history_xi_score": row.get("history_xi_score"),
                    "batting_position_score": row.get("batting_position_score"),
                    "direct_h2h_score": row.get("direct_h2h_score"),
                    "venue_score": row.get("venue_score"),
                    "role_fallback_score": row.get("role_fallback_score"),
                    "weather_score": row.get("weather_score"),
                    "final_selection_score": row.get("final_selection_score"),
                    "has_usable_history": row.get("has_usable_history"),
                }
            )

        n_elig = len(g_eligible)
        low_hist = sum(1 for r in g_eligible if not r.get("has_usable_history"))
        dom_fb = sum(
            1
            for r in g_eligible
            if not r.get("has_usable_history")
            and float(r.get("role_fallback_score") or 0) > float(r.get("history_xi_score") or 0)
        )

        stages["G_predictor_layer"] = _stage(
            ok=n_elig > 0,
            warn=n_elig > 0
            and (low_hist > n_elig * 0.45 or dom_fb > n_elig * 0.35),
            code="no_scoring_breakdown_rows"
            if n_elig == 0
            else ("prediction_fallback_dominating" if dom_fb > n_elig * 0.4 else None),
            details={"per_eligible_player": g_eligible[:60]},
            errors=["No scoring_breakdown_per_player rows returned — check squad text."]
            if n_elig == 0
            else [],
            warnings=[]
            if n_elig == 0 or low_hist <= n_elig * 0.45
            else ["Many squad players lack usable history — composite / fallback dominates selection_score."],
        )
        if n_elig == 0:
            failure_chain.append("prediction_layer_empty")
        elif dom_fb > n_elig * 0.4:
            failure_chain.append("prediction_fallback_dominating")

        h_details = {
            "team_a": g_team_a.get("xi_batting_order_diagnostics") or [],
            "team_b": g_team_b.get("xi_batting_order_diagnostics") or [],
        }
        h_rows = (h_details["team_a"] or []) + (h_details["team_b"] or [])
        bad_bo = sum(
            1
            for r in h_rows
            if str(r.get("batting_order_source") or "").startswith("role_fallback")
        )
        stages["H_batting_order"] = _stage(
            ok=bad_bo < max(1, len(h_rows) // 2),
            warn=bad_bo > 0,
            code="batting_order_weak" if bad_bo >= max(1, len(h_rows) // 2) else None,
            details=h_details,
            warnings=[f"{bad_bo} XI batting-order rows use role_fallback proxy."]
            if bad_bo
            else [],
        )
        if bad_bo >= max(1, len(h_rows) // 2):
            failure_chain.append("batting_order_weak")

        i_details = {
            "team_a_candidates": pred.get("team_a", {}).get("impact_subs") or [],
            "team_b_candidates": pred.get("team_b", {}).get("impact_subs") or [],
            "team_a_ranking": g_team_a.get("impact_sub_ranking") or [],
            "team_b_ranking": g_team_b.get("impact_sub_ranking") or [],
            "toss_scenario_used": (pred.get("toss_scenario") or {}).get("key"),
            "team_a_bats_first_resolved": (pred.get("toss_scenario") or {}).get("team_a_bats_first"),
            "chase_context_team_a": (pred.get("toss_scenario") or {}).get("chase_context_team_a"),
            "chase_context_team_b": (pred.get("toss_scenario") or {}).get("chase_context_team_b"),
        }
        stages["I_impact"] = _stage(
            ok=len(i_details["team_a_ranking"]) + len(i_details["team_b_ranking"]) > 0,
            warn=len(i_details["team_a_ranking"]) + len(i_details["team_b_ranking"]) == 0,
            code="impact_sparse" if not i_details["team_a_ranking"] and not i_details["team_b_ranking"] else None,
            details=i_details,
        )

    # --- Summary ---
    any_fail = any((stages.get(k) or {}).get("status") == "FAIL" for k in STAGE_CODES if k in stages)
    any_warn = any((stages.get(k) or {}).get("status") == "WARN" for k in STAGE_CODES if k in stages)
    if any_fail:
        overall = "FAIL"
    elif any_warn:
        overall = "WARN"
    else:
        overall = "PASS"

    stage_pass_fail = {k: (stages.get(k) or {}).get("status", "SKIP") for k in STAGE_CODES if k in stages}

    pred_trim: Optional[dict[str, Any]] = None
    if pred:
        pred_trim = {
            "xi_validation": pred.get("xi_validation"),
            "history_sync_debug": pred.get("history_sync_debug"),
            "prediction_layer_debug": pred.get("prediction_layer_debug"),
            "batting_order_summary": pred.get("batting_order_summary"),
            "toss_scenario": pred.get("toss_scenario"),
            "strict_validation_warnings": (pred.get("xi_validation") or {}).get("strict_validation_warnings"),
            "team_a": {
                "name": (pred.get("team_a") or {}).get("name"),
                "xi": (pred.get("team_a") or {}).get("xi"),
                "batting_order": (pred.get("team_a") or {}).get("batting_order"),
                "impact_subs": (pred.get("team_a") or {}).get("impact_subs"),
            },
            "team_b": {
                "name": (pred.get("team_b") or {}).get("name"),
                "xi": (pred.get("team_b") or {}).get("xi"),
                "batting_order": (pred.get("team_b") or {}).get("batting_order"),
                "impact_subs": (pred.get("team_b") or {}).get("impact_subs"),
            },
        }

    return {
        "meta": {
            "team_a": canon_a,
            "team_b": canon_b,
            "venue_key": venue.key,
            "audit_season_year": cur_season,
            "audit_target_match_date": str(audit_target_match_date) if audit_target_match_date else None,
            "toss_scenario_key": toss_scenario_key,
        },
        "stages": stages,
        "summary": {
            "overall_pipeline_health": overall,
            "stage_pass_fail": stage_pass_fail,
            "failure_chain": failure_chain,
            "top_3_likely_causes": _infer_top_causes(stages, pred),
        },
        "prediction_snapshot": pred_trim,
    }
