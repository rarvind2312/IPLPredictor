"""
Intra-squad effective history key collisions (Stage F).

Two different squad display names must not share the same ``history_lookup_key`` in one run;
otherwise one player inherits another's SQLite history (wrong XI risk).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import learner
import player_alias_resolve


def _layer_rank(resolution_layer_used: str) -> int:
    s = (resolution_layer_used or "").strip()
    return {
        "exact": 100,
        "layer_b": 86,
        "layer_c": 76,
        "layer_d": 66,
        "layer_d_relaxed_unique_surname": 61,
        "debutant_suppression": 5,
        "ambiguous": 2,
        "unresolved": 1,
    }.get(s, 40)


def _collision_sort_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    """Higher tuple wins. Deterministic final tie-break: player name."""
    rt = str(row.get("resolution_type") or "")
    is_exact = 1 if rt == "exact_match" else 0
    conf = float(row.get("alias_confidence") or 0.0)
    ld = row.get("resolution_layer_debug") if isinstance(row.get("resolution_layer_debug"), dict) else {}
    lr = _layer_rank(str(ld.get("resolution_layer_used") or ""))
    squad_name = str(row.get("player_name") or row.get("squad_display_name") or "")
    hk = str(row.get("history_lookup_key") or row.get("resolved_history_key") or "")
    matched = str(row.get("matched_history_player_name") or "").strip() or None
    g_align, s_align = player_alias_resolve.squad_given_surname_alignment_scores(squad_name, hk, matched)
    support = int(row.get("team_match_xi_rows") or 0) + int(row.get("player_match_stats_rows") or 0)
    name = str(row.get("player_name") or "")
    return (is_exact, conf, lr, g_align, s_align, support, name)


def _clear_row_history_counts(row: dict[str, Any]) -> None:
    row["team_match_xi_rows"] = 0
    row["player_match_stats_rows"] = 0
    row["player_batting_positions_rows"] = 0
    row["latest_match_date"] = None
    row["h2h_rows_vs_opponent"] = 0
    row["matched_history_player_name"] = None


def apply_intrasquad_effective_key_collisions(per_player: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Mutates ``per_player`` rows in place when two+ players share the same non-empty
    ``history_lookup_key``. Exactly one winner keeps the key; losers are downgraded.

    Returns summary stats for linkage ``summary`` augmentation.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_player:
        k = row.get("history_lookup_key")
        sk = str(k).strip() if k else ""
        if sk:
            groups[sk].append(row)

    n_groups = 0
    n_losers = 0
    for row in per_player:
        row.setdefault("collision_resolution_outcome", "no_collision")

    for collided_key, members in groups.items():
        if len(members) < 2:
            continue
        n_groups += 1
        ordered = sorted(members, key=_collision_sort_tuple, reverse=True)
        winner = ordered[0]
        names = [str(m.get("player_name") or "") for m in members]
        for m in members:
            m["collided_history_key"] = collided_key
            m["collision_group_members"] = list(names)

        winner["collision_resolution_outcome"] = "winner"
        winner["collision_winner_player_name"] = str(winner.get("player_name") or "")
        winner["collision_winner_resolution_type"] = str(winner.get("resolution_type") or "")
        winner["collision_winner_confidence"] = float(winner.get("alias_confidence") or 0.0)

        for loser in ordered[1:]:
            n_losers += 1
            loser["pre_collision_history_lookup_key"] = collided_key
            loser["pre_collision_resolution_type"] = str(loser.get("resolution_type") or "")
            loser["history_lookup_key"] = None
            loser["resolved_history_key"] = None
            loser["global_resolved_history_key"] = None
            loser["used_global_resolved_key_for_prior"] = False
            loser["global_alias_resolution_type"] = None
            loser["global_alias_confidence"] = None
            loser["global_alias_layer_used"] = None
            loser["resolution_type"] = "ambiguous_alias_collision"
            loser["history_status"] = "history_key_collision_loser"
            loser["rolled_up_interpretation"] = "history_key_collision_loser"
            loser["collision_resolution_outcome"] = "lost_collision"
            loser["collision_winner_player_name"] = str(winner.get("player_name") or "")
            loser["collision_winner_resolution_type"] = str(winner.get("resolution_type") or "")
            loser["collision_winner_confidence"] = float(winner.get("alias_confidence") or 0.0)
            _clear_row_history_counts(loser)

    return {
        "history_key_collision_groups": int(n_groups),
        "history_key_collision_losers": int(n_losers),
    }


__all__ = ["apply_intrasquad_effective_key_collisions"]
