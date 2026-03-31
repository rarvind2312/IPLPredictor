"""Intra-squad history_lookup_key collision resolution (Stage F)."""

from __future__ import annotations

import history_key_collision


def _row(
    name: str,
    *,
    key: str,
    rt: str,
    conf: float,
    layer: str,
    tmx: int = 0,
    pms: int = 0,
    matched: str | None = None,
) -> dict:
    return {
        "player_name": name,
        "resolution_type": rt,
        "alias_confidence": conf,
        "history_lookup_key": key,
        "resolved_history_key": key,
        "team_match_xi_rows": tmx,
        "player_match_stats_rows": pms,
        "matched_history_player_name": matched,
        "resolution_layer_debug": {"resolution_layer_used": layer},
        "ambiguous_candidates": (),
        "layer_b_variant_hits": (),
        "surname_bucket_size": 0,
    }


def test_rohit_beats_raghu_on_same_rg_sharma_key() -> None:
    rows = [
        _row("Rohit Sharma", key="rg sharma", rt="alias_match", conf=0.88, layer="layer_d"),
        _row("Raghu Sharma", key="rg sharma", rt="alias_match", conf=0.81, layer="layer_d"),
    ]
    history_key_collision.apply_intrasquad_effective_key_collisions(rows)
    rohit = next(r for r in rows if r["player_name"] == "Rohit Sharma")
    raghu = next(r for r in rows if r["player_name"] == "Raghu Sharma")
    assert rohit["history_lookup_key"] == "rg sharma"
    assert rohit["collision_resolution_outcome"] == "winner"
    assert raghu["resolution_type"] == "ambiguous_alias_collision"
    assert raghu["history_lookup_key"] is None
    assert raghu["resolved_history_key"] is None
    assert raghu["collision_resolution_outcome"] == "lost_collision"
    assert raghu["team_match_xi_rows"] == 0


def test_exact_match_wins_over_alias_same_key() -> None:
    rows = [
        _row("Virat Kohli", key="v kohli", rt="exact_match", conf=1.0, layer="exact"),
        _row("V Kohli", key="v kohli", rt="alias_match", conf=0.9, layer="layer_b"),
    ]
    history_key_collision.apply_intrasquad_effective_key_collisions(rows)
    exact = rows[0]
    alias = rows[1]
    assert exact["collision_resolution_outcome"] == "winner"
    assert alias["history_lookup_key"] is None
    assert alias["resolution_type"] == "ambiguous_alias_collision"


def test_alias_higher_confidence_wins() -> None:
    rows = [
        _row("A X", key="a x", rt="alias_match", conf=0.72, layer="layer_c"),
        _row("A Xy", key="a x", rt="alias_match", conf=0.91, layer="layer_c"),
    ]
    history_key_collision.apply_intrasquad_effective_key_collisions(rows)
    hi = next(r for r in rows if r["player_name"] == "A Xy")
    lo = next(r for r in rows if r["player_name"] == "A X")
    assert hi["collision_resolution_outcome"] == "winner"
    assert lo["history_lookup_key"] is None


def test_no_collision_single_holder() -> None:
    rows = [_row("Solo", key="solo y", rt="exact_match", conf=1.0, layer="exact")]
    history_key_collision.apply_intrasquad_effective_key_collisions(rows)
    assert rows[0]["collision_resolution_outcome"] == "no_collision"
    assert rows[0]["history_lookup_key"] == "solo y"
