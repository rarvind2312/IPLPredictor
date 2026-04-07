"""
Shared UI helpers for selection / scoring debug tables (Streamlit).

Prediction engine and scoring logic live elsewhere; this module only shapes rows for display.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def selection_debug_top15_dataframe_for_side(
    r: dict[str, Any],
    side: str,
    *,
    include_reason_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Top 15 squad players by ``final_selection_score`` with Stage-3 selection_model fields.

    ``side`` is ``team_a`` or ``team_b``.

    When ``include_reason_columns`` is False, omit ``recent_form_competitions`` and
    ``reason_summary`` columns (legacy Admin page table shape).
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
    rows_out: list[dict[str, Any]] = []
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
        out_row: dict[str, Any] = {
            "player": name,
            "in_playing_xi": "yes" if name in xi_names else "no",
            "impact_candidate": "yes" if name in impact_names else "no",
            "recent_form_score": smb.get("recent_form_score"),
            "ipl_history_and_role_score": smb.get("ipl_history_and_role_score"),
            "team_balance_fit_score": smb.get("team_balance_fit_score"),
            "venue_experience_score": smb.get("venue_experience_score"),
            "tactical_adjustment_total": tact,
            "final_selection_score": row.get("final_selection_score"),
        }
        if include_reason_columns:
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
            out_row["recent_form_competitions"] = comps or ""
            out_row["reason_summary"] = reason_s
        rows_out.append(out_row)
    sel_dbg = (r.get("selection_debug") or {}).get(side) or {}
    xi_val = sel_dbg.get("xi_validation") if isinstance(sel_dbg.get("xi_validation"), dict) else {}
    return pd.DataFrame(rows_out), xi_val
