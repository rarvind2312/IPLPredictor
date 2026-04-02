"""
Canonical rule specification (single source of truth) for:
- XI membership constraints (hard + semi-hard)
- Batting-order constraints (hard guardrails)
- Selection priority order (tie-break intent)

This module is intentionally declarative: it defines *what* the rules are.
Implementation lives in ``rules_xi.py`` and the batting-order logic in ``predictor.py``.
"""

from __future__ import annotations

from typing import Any

import config


CANONICAL_RULE_SPEC: dict[str, Any] = {
    "xi": {
        "hard_constraints": {
            "xi_size": 11,
            "overseas": {
                "min": int(getattr(config, "MIN_OVERSEAS_IN_XI", 3)),
                "max": int(getattr(config, "MAX_OVERSEAS", 4)),
            },
            "designated_keeper_required": True,
            "bowling_options_min": int(getattr(config, "MIN_BOWLING_OPTIONS", 5)),
            "pacers_min": int(getattr(config, "MIN_PACE_OPTIONS_IN_XI", 3)),
            "spinners_min": int(getattr(config, "MIN_SPINNER_OPTIONS_IN_XI", 1)),
            "spinners_allow_pace_only_override": True,
        },
        "semi_hard_constraints": {
            "wk_role_players_max": 2,
            "wk_role_players_allow_marquee_override": True,
            "structural_all_rounders_max": 3,
            "structural_all_rounders_allow_unavoidable_override": True,
        },
        "selection_priority_order": [
            "squad_truth",
            "marquee_core_lock",
            "last_xi_continuity",
            "recent_form",
            "strong_ipl_history",
            "stable_role_identity",
            "team_structure_fit",
            "overseas_target",
            "conditions_small_tweak",
            "matchup_tie_break_only",
        ],
    },
    "batting_order": {
        "hard_constraints": {
            "specialist_bowler_positions": {"min": 8, "max": 11},
            "top_order_batter_allowed_max_position": 5,
            "opener_allowed_max_position": 3,
            "bowling_all_rounder_below_top_order_default": True,
        },
        "batting_bands": [
            "opener",
            "top_order",
            "middle_order",
            "finisher",
            "bowling_all_rounder",
            "specialist_bowler",
        ],
    },
}

