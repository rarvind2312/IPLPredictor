"""
Temporary full-pipeline audit logging (XI, impact subs, batting order).

Enable with environment variable::

    IPL_FULL_PIPELINE_AUDIT=1

Logs JSON lines to logger ``ipl_predictor.full_pipeline_audit`` at WARNING level.
No scoring or selection logic — emit helpers only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

_audit_log = logging.getLogger("ipl_predictor.full_pipeline_audit")


def enabled() -> bool:
    v = (os.environ.get("IPL_FULL_PIPELINE_AUDIT") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def emit(team: str, audit_event: str, payload: Optional[dict[str, Any]] = None) -> None:
    if not enabled():
        return
    body: dict[str, Any] = {"team": team, "audit_event": audit_event}
    if payload:
        body.update(payload)
    _audit_log.warning("%s", json.dumps(body, default=str))


def marquee_tier_str(p: Any) -> str:
    hd = getattr(p, "history_debug", None) or {}
    return str(hd.get("marquee_tier") or "").strip().lower()


def tier_val_from_str(tier: str) -> int:
    t = (tier or "").lower()
    if t == "tier_1":
        return 3
    if t == "tier_2":
        return 2
    if t == "tier_3":
        return 1
    return 0


def player_row_pre_scoring(p: Any) -> dict[str, Any]:
    hd = getattr(p, "history_debug", None) or {}
    meta = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    r5_raw = hd.get("recent5_xi_rate")
    recent: Any = None
    try:
        if r5_raw is not None:
            recent = round(float(r5_raw), 4)
    except (TypeError, ValueError):
        recent = None
    return {
        "name": getattr(p, "name", ""),
        "primary_role": str(meta.get("primary_role") or getattr(p, "role", "") or ""),
        "role_description": str(meta.get("role_description") or ""),
        "marquee_tier": marquee_tier_str(p) or None,
        "recent_xi_rate": recent,
    }


def player_row_post_scoring(p: Any, rank: int) -> dict[str, Any]:
    hd = getattr(p, "history_debug", None) or {}
    smd = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
    base = smd.get("base_score_breakdown") if isinstance(smd.get("base_score_breakdown"), dict) else {}
    row = player_row_pre_scoring(p)
    row["selection_score"] = round(float(getattr(p, "selection_score", 0.0) or 0.0), 5)
    row["score_breakdown"] = dict(base) if base else None
    row["rank"] = rank
    return row


def tier_counts(players: list[Any]) -> dict[str, int]:
    c = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "other": 0}
    for p in players:
        t = marquee_tier_str(p)
        if t == "tier_1":
            c["tier_1"] += 1
        elif t == "tier_2":
            c["tier_2"] += 1
        elif t == "tier_3":
            c["tier_3"] += 1
        else:
            c["other"] += 1
    return c


def role_counts(players: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in players:
        rb = str(getattr(p, "role_bucket", "") or "unknown")
        out[rb] = out.get(rb, 0) + 1
    return out


def xi_name_list(players: list[Any]) -> list[str]:
    return [str(getattr(p, "name", "") or "") for p in players]


def tier_preference_violation(scored: list[Any], xi_names_set: set[str]) -> dict[str, Any]:
    in_xi_t3: list[str] = []
    bench_t1: list[str] = []
    for p in scored:
        nm = getattr(p, "name", "")
        if not nm:
            continue
        tv = tier_val_from_str(marquee_tier_str(p))
        if nm in xi_names_set and tv == 1:
            in_xi_t3.append(nm)
        if nm not in xi_names_set and marquee_tier_str(p) == "tier_1":
            bench_t1.append(nm)
    viol = bool(in_xi_t3 and bench_t1)
    return {
        "tier_preference_violation": viol,
        "tier_1_not_in_xi": bench_t1[:16],
        "tier_3_in_xi": in_xi_t3[:16],
    }


def unused_higher_tier_players(scored: list[Any], xi_names_set: set[str]) -> list[str]:
    """tier_1 / tier_2 squad players not in final XI (audit summary)."""
    out: list[str] = []
    for p in scored:
        nm = getattr(p, "name", "")
        if not nm or nm in xi_names_set:
            continue
        if marquee_tier_str(p) in ("tier_1", "tier_2"):
            out.append(nm)
    return out[:24]


def emit_xi_stages(
    team: str,
    scored: list[Any],
    xi_base: list[Any],
    xi_after_conditions: list[Any],
    condition_changes: list[dict[str, Any]],
    xi_after_overseas: list[Any],
    overseas_debug: dict[str, Any],
    xi_final: list[Any],
    repair_swaps: list[dict[str, Any]],
) -> None:
    if not enabled():
        return
    byn = {getattr(p, "name", ""): p for p in scored if getattr(p, "name", "")}
    n0 = xi_name_list(xi_base)
    emit(
        team,
        "xi_initial_selection",
        {
            "xi_names": n0,
            "tier_counts": tier_counts(xi_base),
            "role_counts": role_counts(xi_base),
        },
    )
    n1 = xi_name_list(xi_after_conditions)
    n2 = xi_name_list(xi_after_overseas)
    nf = xi_name_list(xi_final)
    phases: list[dict[str, Any]] = []

    cond_swaps = [enrich_swap_tiers(dict(c), byn) for c in (condition_changes or [])]
    if n1 != n0:
        emit_override(
            team,
            "xi",
            "condition_adjustments",
            n0,
            n1,
            "scenario/venue condition swaps from base XI (see condition_changes)",
        )
    phases.append({"phase": "condition_adjustments", "xi_names_after": n1, "swaps": cond_swaps})

    os_swaps = list((overseas_debug or {}).get("overseas_swaps") or [])
    enriched_os: list[dict[str, Any]] = []
    for s in os_swaps:
        if isinstance(s, dict):
            enriched_os.append(enrich_swap_tiers(s, byn))
    if n2 != n1:
        emit_override(
            team,
            "xi",
            "overseas_preference",
            n1,
            n2,
            str((overseas_debug or {}).get("why_4th_overseas_selected_or_not") or "overseas_target_or_minimum"),
        )
    phases.append({"phase": "overseas_preference", "xi_names_after": n2, "swaps": enriched_os})

    enriched_rep: list[dict[str, Any]] = []
    for s in repair_swaps or []:
        if isinstance(s, dict):
            o = dict(s)
            o_out = str(o.get("out") or "")
            o_in = str(o.get("in") or "")
            op = byn.get(o_out)
            ip = byn.get(o_in)
            o["old_tier"] = marquee_tier_str(op) if op else ""
            o["new_tier"] = marquee_tier_str(ip) if ip else ""
            enriched_rep.append(o)
        else:
            enriched_rep.append({"raw": s})
    if nf != n2:
        emit_override(
            team,
            "xi",
            "rules_xi_repair",
            n2,
            nf,
            "rules_xi.validate_xi repair / select_playing_xi fallback (see repair_swaps)",
        )
    phases.append({"phase": "rules_repair", "xi_names_after": nf, "swaps": enriched_rep})

    emit(team, "xi_post_repair", {"phases": phases, "final_xi_names": nf})
    xset = set(nf)
    summ = tier_preference_violation(scored, xset)
    emit(
        team,
        "xi_final_summary",
        {
            "final_xi_names": nf,
            "unused_higher_tier_players": unused_higher_tier_players(scored, xset),
            **summ,
        },
    )


def emit_impact_stages(
    team: str,
    squad: list[Any],
    xi: list[Any],
    picked: list[Any],
    dbg_all: list[dict[str, Any]],
    *,
    fallback_used: bool,
    engine_top5_names: list[str],
) -> None:
    if not enabled():
        return
    xi_names = {getattr(p, "name", "") for p in xi}
    byn = {getattr(p, "name", ""): p for p in squad if getattr(p, "name", "")}
    bench = [p for p in squad if getattr(p, "name", "") not in xi_names]
    pre: list[dict[str, Any]] = []
    for p in bench:
        hd = getattr(p, "history_debug", None) or {}
        r5 = hd.get("recent5_xi_rate")
        try:
            rx = round(float(r5), 4) if r5 is not None else None
        except (TypeError, ValueError):
            rx = None
        pre.append(
            {
                "name": getattr(p, "name", ""),
                "marquee_tier": marquee_tier_str(p) or None,
                "role": str(getattr(p, "role_bucket", "")),
                "recent_xi_rate": rx,
            }
        )
    emit(team, "impact_candidates_pre_rank", {"candidates": pre})

    post: list[dict[str, Any]] = []
    for row in dbg_all:
        if not isinstance(row, dict):
            continue
        nm = str(row.get("name") or "")
        pl = byn.get(nm)
        post.append(
            {
                "name": nm,
                "impact_total_score": row.get("impact_total_score"),
                "impact_sub_rank": row.get("impact_sub_rank"),
                "marquee_tier": marquee_tier_str(pl) if pl else None,
            }
        )
    emit(team, "impact_candidates_post_rank", {"ranked": post})

    picked_names = [getattr(p, "name", "") for p in picked]
    reasons: list[dict[str, Any]] = []
    engine_order = [str(r.get("name") or "") for r in dbg_all if isinstance(r, dict)]
    engine_idx = {nm: i for i, nm in enumerate(engine_order)}
    for pn in picked_names:
        pi = engine_idx.get(pn, 9999)
        p_tier = tier_val_from_str(marquee_tier_str(byn.get(pn)))
        for j, nm2 in enumerate(engine_order):
            if nm2 in xi_names or nm2 == pn:
                continue
            t2 = tier_val_from_str(marquee_tier_str(byn.get(nm2)))
            if t2 > p_tier and j < pi:
                reasons.append(
                    {
                        "picked": pn,
                        "higher_tier_not_selected": nm2,
                        "skipped_engine_rank_1based": j + 1,
                        "picked_engine_rank_1based": pi + 1 if pi < 9000 else None,
                    }
                )
                break

    emit(
        team,
        "impact_final_selection",
        {
            "final_impact_subs": picked_names,
            "engine_pure_top5": list(engine_top5_names),
            "higher_tier_bench_not_in_top5": [
                getattr(p, "name", "")
                for p in bench
                if marquee_tier_str(p) in ("tier_1", "tier_2")
                and getattr(p, "name", "") not in set(picked_names)
            ][:16],
            "selected_over_higher_tier_reason": reasons[:16],
            "diversity_or_fallback_used": bool(fallback_used),
        },
    )
    if picked_names != engine_top5_names:
        emit_override(
            team,
            "impact",
            "predictor_impact_subs_diversity_protected_or_fallback",
            engine_top5_names,
            picked_names,
            "impact_subs(): diversity buckets, protected bench core, or sub-5 fallback fill vs pure engine top-5 order",
        )


def enrich_swap_tiers(
    change: dict[str, Any],
    by_name: dict[str, Any],
) -> dict[str, Any]:
    o = {**change}
    op = by_name.get(str(change.get("out") or ""))
    ip = by_name.get(str(change.get("in") or ""))
    o["old_tier"] = marquee_tier_str(op) if op else ""
    o["new_tier"] = marquee_tier_str(ip) if ip else ""
    return o


def emit_override(team: str, kind: str, stage: str, before: Any, after: Any, reason: str) -> None:
    emit(
        team,
        f"{kind}_override_applied",
        {"override_kind": kind, "stage": stage, "before": before, "after": after, "reason": reason},
    )


def batting_discrete_violations(
    order: list[str],
    eligibility_by_name: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    bad: list[dict[str, Any]] = []
    for i, nm in enumerate(order):
        prof = eligibility_by_name.get(nm) or {}
        raw = prof.get("allowed_slots") or []
        al = []
        for x in raw:
            try:
                v = int(x)
                if 1 <= v <= 11:
                    al.append(v)
            except (TypeError, ValueError):
                continue
        pos = i + 1
        if al and pos not in al:
            bad.append({"name": nm, "slot": pos, "allowed_batting_slots": al})
    return len(bad), bad


def batting_order_inputs_rows(
    xi: list[Any],
    eligibility_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in xi:
        nm = getattr(p, "name", "")
        hd = getattr(p, "history_debug", None) or {}
        meta = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
        prof = eligibility_by_name.get(nm) or {}
        rows.append(
            {
                "name": nm,
                "primary_role": str(meta.get("primary_role") or getattr(p, "role", "") or ""),
                "batting_band": str(prof.get("band") or ""),
                "allowed_batting_slots": list(prof.get("allowed_slots") or []),
                "preferred_batting_slots": list(prof.get("preferred_slots") or []),
            }
        )
    return rows
