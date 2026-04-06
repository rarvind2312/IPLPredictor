"""
Batting order what-if: phase heuristics, opposition bowling projection, and recommendation text.

Keeps XI fixed; only batting order changes are evaluated elsewhere (UI + win engine).
"""

from __future__ import annotations

from typing import Any

import db
import ipl_teams
import learner


def slot_likely_phases(slot_1based: int) -> tuple[str, str]:
    """
    Map batting position (1–11) to a short phase label and longer description (MVP heuristics).
    """
    s = int(slot_1based)
    if s <= 0:
        s = 1
    if s > 11:
        s = 11
    if s <= 2:
        return "powerplay", "early overs — new ball, field up; mainly pace and one spinner possible."
    if s == 3:
        return "transition", "powerplay into middle — still some new-ball threat, spin often introduced."
    if s <= 5:
        return "middle", "middle overs — spin common, match-ups matter."
    if s <= 7:
        return "middle_death", "late middle into death — slower balls, yorkers, hard lengths."
    return "lower_death", "lower order / death — high risk, boundary hunting."


def _norm_pk(name: str) -> str:
    return learner.normalize_player_key(name or "")[:80]


def _bowling_type_from_row(row: dict[str, Any]) -> str:
    bt = str(row.get("bowling_type") or "").strip().lower()
    if bt and bt not in ("unknown", "none", ""):
        return bt
    meta = row.get("_meta") if isinstance(row.get("_meta"), dict) else {}
    btb = str(meta.get("bowling_type_bucket") or "").strip().lower()
    return btb or "unknown"


def _primary_phase_for_bowler(phase_summary: dict[str, Any]) -> str:
    """Pick powerplay / middle / death by largest share."""
    if not phase_summary:
        return "middle"
    pp = float(phase_summary.get("powerplay_share") or 0.0)
    md = float(phase_summary.get("middle_share") or 0.0)
    dt = float(phase_summary.get("death_share") or 0.0)
    m = max(pp, md, dt)
    if m <= 0:
        return "middle"
    if m == pp:
        return "powerplay"
    if m == dt:
        return "death"
    return "middle"


def project_opposition_bowling_plan(
    opposition_xi_names: list[str],
    opposition_squad_map: dict[str, dict[str, Any]],
    opposition_franchise_label: str,
) -> dict[str, list[tuple[str, str, str]]]:
    """
    Phase → list of (display_name, bowling_type_bucket-ish, primary_phase_tag).

    Uses ``db.fetch_bowler_phase_summary_batch`` when data exists; otherwise role_bucket heuristics.
    """
    lab = ipl_teams.canonical_franchise_label(opposition_franchise_label) or (opposition_franchise_label or "").strip()
    tk = ipl_teams.canonical_team_key_for_franchise(lab)[:80]
    keys = [_norm_pk(n) for n in opposition_xi_names if str(n).strip()]
    keys = list(dict.fromkeys([k for k in keys if k]))
    phase_blob = db.fetch_bowler_phase_summary_batch(tk, keys) if tk and keys else {}

    bowlers: list[tuple[str, str, str]] = []
    for nm in opposition_xi_names:
        row = opposition_squad_map.get(nm) or {}
        rb = str(row.get("role_bucket") or "")
        if rb not in ("Bowler", "All-Rounder"):
            continue
        pk = _norm_pk(nm)
        ps = phase_blob.get(pk) or {}
        phase_tag = _primary_phase_for_bowler(ps)
        btype = _bowling_type_from_row(row)
        bowlers.append((nm, btype, phase_tag))

    buckets: dict[str, list[tuple[str, str, str]]] = {"powerplay": [], "middle": [], "death": []}
    for nm, btype, phase_tag in bowlers:
        if phase_tag == "powerplay":
            buckets["powerplay"].append((nm, btype, phase_tag))
        elif phase_tag == "death":
            buckets["death"].append((nm, btype, phase_tag))
        else:
            buckets["middle"].append((nm, btype, phase_tag))

    if not bowlers:
        return buckets
    if not buckets["middle"]:
        buckets["middle"] = [(nm, bt, "middle") for nm, bt, _ in bowlers]
    if not buckets["powerplay"] and bowlers:
        nm, bt, _ = bowlers[0]
        buckets["powerplay"].append((nm, bt, "powerplay"))
    if not buckets["death"] and bowlers:
        nm, bt, _ = bowlers[-1]
        buckets["death"].append((nm, bt, "death"))
    return buckets


def likely_opposition_for_slot(
    slot_phase: str,
    plan: dict[str, list[tuple[str, str, str]]],
) -> list[tuple[str, str]]:
    """Names and types most relevant to this batting slot phase."""
    if slot_phase in ("powerplay", "transition"):
        pool = list(plan.get("powerplay") or []) + list(plan.get("middle") or [])[:3]
    elif slot_phase in ("middle",):
        pool = list(plan.get("middle") or []) + list(plan.get("powerplay") or [])[:2]
    elif slot_phase in ("middle_death",):
        pool = list(plan.get("middle") or []) + list(plan.get("death") or [])
    else:
        pool = list(plan.get("death") or []) + list(plan.get("middle") or [])[:2]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for nm, bt, _ in pool:
        if nm in seen:
            continue
        seen.add(nm)
        out.append((nm, bt))
        if len(out) >= 6:
            break
    return out


def _batter_hand_and_band(row: dict[str, Any]) -> tuple[str, str]:
    meta = row.get("_meta") if isinstance(row.get("_meta"), dict) else {}
    hand = str(meta.get("batting_hand") or "").strip().lower()
    if hand not in ("right", "left"):
        hand = ""
    band = str(meta.get("likely_batting_band") or row.get("batting_band") or "").strip().lower()
    return hand, band


def assess_move_favorability(
    *,
    batter_row: dict[str, Any],
    slot_phase: str,
    opp_types: list[str],
    pace_bias: float,
    spin_friendliness: float,
) -> tuple[str, str]:
    """
    Return (verdict_label, explanation) — verdict: favorable | neutral | risky.
    MVP: use venue pace/spin tilt + rough type mix (pace vs spin strings).
    """
    spin_n = sum(1 for t in opp_types if any(x in t for x in ("spin", "wrist", "orthodox", "offbreak", "legbreak", "slow")))
    pace_n = sum(1 for t in opp_types if t in ("pace", "right_arm_fast", "right_arm_fast_medium", "left_arm_fast", "left_arm_fast_medium") or "fast" in t)

    _, band = _batter_hand_and_band(batter_row)
    top_aggr = band in ("opener", "top_order", "finisher") or bool(batter_row.get("is_opener_candidate"))

    risky = False
    favorable = False
    bits: list[str] = []

    if slot_phase == "powerplay" and spin_friendliness > 0.62 and spin_n >= pace_n:
        risky = True
        bits.append("Early slot but spin-heavy opposition lean at this venue profile — can be tricky to force pace.")
    if slot_phase in ("lower_death", "middle_death") and pace_n >= 3:
        if not top_aggr and band not in ("finisher", "lower_middle", "middle_order"):
            risky = True
            bits.append("Death-adjacent overs with several pace options — lower-order hitters can be exposed.")
    if slot_phase in ("middle", "transition") and spin_n >= 2 and spin_friendliness >= 0.55:
        if band in ("middle_order", "anchor", ""):
            favorable = True
            bits.append("Middle-overs spin aligns with a stabilising middle-order role.")

    if pace_bias >= 0.58 and slot_phase in ("powerplay", "transition") and pace_n >= 2:
        if top_aggr:
            favorable = True
            bits.append("Pace-friendly venue and new-ball overs suit an aggressive top-order profile.")

    if favorable and not risky:
        verdict = "favorable"
    elif risky and not favorable:
        verdict = "risky"
    else:
        verdict = "neutral"
        if not bits:
            bits.append("Trade-offs are balanced on available signals — monitor match-ups in-game.")

    return verdict, " ".join(bits)


def order_position_map(order: list[str]) -> dict[str, int]:
    return {str(nm): i + 1 for i, nm in enumerate(order)}


def moved_players(baseline: list[str], edited: list[str]) -> list[tuple[str, int, int]]:
    """Players whose 1-based slot changed."""
    if len(baseline) != len(edited) or set(baseline) != set(edited):
        return []
    bmap = order_position_map(baseline)
    out: list[tuple[str, int, int]] = []
    for nm in baseline:
        o, n = bmap[str(nm)], order_position_map(edited)[str(nm)]
        if o != n:
            out.append((str(nm), o, n))
    return out


def build_recommendation_cards(
    *,
    baseline_order: list[str],
    edited_order: list[str],
    batter_squad_map: dict[str, dict[str, Any]],
    opposition_xi: list[str],
    opposition_squad_map: dict[str, dict[str, Any]],
    opposition_franchise_label: str,
    pace_bias: float,
    spin_friendliness: float,
) -> list[dict[str, Any]]:
    plan = project_opposition_bowling_plan(opposition_xi, opposition_squad_map, opposition_franchise_label)
    cards: list[dict[str, Any]] = []
    for name, old_p, new_p in moved_players(baseline_order, edited_order):
        phase_key, phase_desc = slot_likely_phases(new_p)
        opps = likely_opposition_for_slot(phase_key, plan)
        opp_names = [x[0] for x in opps]
        opp_types = [x[1] for x in opps]
        row = batter_squad_map.get(name) or {}
        verdict, expl = assess_move_favorability(
            batter_row=row,
            slot_phase=phase_key,
            opp_types=opp_types,
            pace_bias=pace_bias,
            spin_friendliness=spin_friendliness,
        )
        type_summary = ", ".join(sorted({t for t in opp_types if t and t != "unknown"})) or "mixed / data thin"
        cards.append(
            {
                "name": name,
                "old_position": old_p,
                "new_position": new_p,
                "likely_phase": phase_key.replace("_", " "),
                "phase_detail": phase_desc,
                "likely_bowlers": ", ".join(opp_names[:5]) if opp_names else "insufficient phase data — assume usual attack mix",
                "likely_bowling_types": type_summary,
                "verdict": verdict,
                "recommendation": (
                    f"**{name}** moves from **#{old_p}** to **#{new_p}** ({phase_key.replace('_', ' ')}): "
                    f"{phase_desc} "
                    f"Likely to see: {', '.join(opp_names[:5]) if opp_names else 'opposition attack (thin DB projection)'}. "
                    f"Bowling types skew: **{type_summary}**. "
                    f"Tactical read: **{verdict}** — {expl}"
                ),
            }
        )
    return cards
