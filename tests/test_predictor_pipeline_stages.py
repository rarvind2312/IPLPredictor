from __future__ import annotations

import unittest

import predictor
from ipl_squad import ALL_ROUNDER, BATTER, BOWLER, WK_BATTER


def _player(
    name: str,
    *,
    role_bucket: str,
    role: str,
    bat_skill: float,
    bowl_skill: float,
    batting_band: str,
    role_band: str,
    recent_ema: float = 0.0,
    recent_rows: int = 0,
    dominant_position: float = 0.0,
    is_wicketkeeper: bool = False,
    bowling_type: str | None = None,
    player_metadata: dict | None = None,
) -> predictor.SquadPlayer:
    p = predictor.SquadPlayer(
        name=name,
        role=role,
        role_bucket=role_bucket,
        is_overseas=False,
        bat_skill=bat_skill,
        bowl_skill=bowl_skill,
        is_wicketkeeper=is_wicketkeeper,
        bowling_type=bowling_type,
    )
    p.history_debug = {
        "batting_band": batting_band,
        "role_band": role_band,
        "current_team_recent_batting_position_ema": recent_ema,
        "current_team_recent_batting_rows": recent_rows,
        "dominant_position": dominant_position,
        "player_metadata_source_runtime": "squad_json",
        "selection_model_debug": {
            "base_score_breakdown": {
                "team_balance_fit_score": 0.64,
                "last_match_continuity_score": 0.62,
                "recent_form_score": 0.58,
            },
            "last_match_detail": {"was_in_last_match_xi": True},
        },
        "selection_reason_summary": f"{name} selection summary",
    }
    if player_metadata is not None:
        p.history_debug["player_metadata"] = dict(player_metadata)
    return p


class TestPredictorPipelineStages(unittest.TestCase):
    def test_assign_batting_order_stage_uses_final_xi_only_and_writes_reasons(self) -> None:
        xi = [
            _player("Opener One", role_bucket=BATTER, role="bat", bat_skill=0.8, bowl_skill=0.2, batting_band="opener", role_band="opener", recent_ema=1.4, recent_rows=4, dominant_position=1.0),
            _player("Opener Two", role_bucket=BATTER, role="bat", bat_skill=0.76, bowl_skill=0.2, batting_band="top_order", role_band="top_order", recent_ema=2.8, recent_rows=4, dominant_position=2.0),
            _player("Keeper", role_bucket=WK_BATTER, role="wk", bat_skill=0.72, bowl_skill=0.2, batting_band="middle_order", role_band="wicketkeeper_batter", recent_ema=4.7, recent_rows=3, dominant_position=4.0, is_wicketkeeper=True),
            _player("Middle One", role_bucket=BATTER, role="bat", bat_skill=0.71, bowl_skill=0.2, batting_band="middle_order", role_band="middle_order", recent_ema=5.2, recent_rows=3, dominant_position=5.0),
            _player("Middle Two", role_bucket=ALL_ROUNDER, role="all", bat_skill=0.69, bowl_skill=0.56, batting_band="middle_order", role_band="batting_allrounder", recent_ema=6.2, recent_rows=3, dominant_position=6.0),
            _player("Lower Middle", role_bucket=ALL_ROUNDER, role="all", bat_skill=0.58, bowl_skill=0.62, batting_band="lower_middle", role_band="balanced_allrounder", recent_ema=8.0, recent_rows=2, dominant_position=8.0),
            _player("Bowling AR", role_bucket=ALL_ROUNDER, role="all", bat_skill=0.45, bowl_skill=0.74, batting_band="lower_order", role_band="bowling_allrounder", dominant_position=9.0),
            _player("Bowler One", role_bucket=BOWLER, role="bowl", bat_skill=0.2, bowl_skill=0.8, batting_band="", role_band="death_bowler", bowling_type="right_arm_fast"),
            _player("Bowler Two", role_bucket=BOWLER, role="bowl", bat_skill=0.2, bowl_skill=0.77, batting_band="", role_band="powerplay_bowler", bowling_type="right_arm_fast_medium"),
            _player("Bowler Three", role_bucket=BOWLER, role="bowl", bat_skill=0.18, bowl_skill=0.75, batting_band="", role_band="middle_overs_spinner", bowling_type="finger_spin"),
            _player("Bowler Four", role_bucket=BOWLER, role="bowl", bat_skill=0.17, bowl_skill=0.74, batting_band="", role_band="utility_bowler", bowling_type="left_arm_fast_medium"),
        ]
        conditions = {
            "batting_friendliness": 0.5,
            "dew_risk": 0.5,
            "spin_friendliness": 0.5,
            "pace_bias": 0.5,
            "rain_disruption_risk": 0.0,
        }

        order = predictor._assign_batting_order_stage(
            xi,
            conditions,
            team_name="Test XI",
            venue_keys=[],
            out_warnings=[],
        )

        self.assertEqual(set(order), {p.name for p in xi})
        for p in xi:
            self.assertTrue(p.history_debug.get("batting_slot_eligibility_source"))
            self.assertTrue(p.history_debug.get("batting_slot_assignment_reason"))
            self.assertIn("Stage 2 batting slot", p.history_debug.get("batting_slot_assignment_reason"))

    def test_annotate_xi_selection_stage_writes_stage_reasons(self) -> None:
        p1 = _player("Core Batter", role_bucket=BATTER, role="bat", bat_skill=0.8, bowl_skill=0.2, batting_band="top_order", role_band="top_order")
        p2 = _player("Core Keeper", role_bucket=WK_BATTER, role="wk", bat_skill=0.7, bowl_skill=0.2, batting_band="middle_order", role_band="wicketkeeper_batter", is_wicketkeeper=True)
        p3 = _player("Bench Bowler", role_bucket=BOWLER, role="bowl", bat_skill=0.2, bowl_skill=0.78, batting_band="", role_band="death_bowler", bowling_type="right_arm_fast")
        p1.history_debug["marquee_tier"] = "tier_1"
        p2.history_debug["wicketkeeper_selected_for_team"] = True

        predictor._annotate_xi_selection_stage(
            [p1, p2, p3],
            [p1, p2],
            [p1, p2],
            team_name="Test XI",
            scenario_branch="if_team_bats_first",
            condition_changes=[],
            repair_swaps=[],
        )

        self.assertEqual(p1.history_debug.get("xi_stage"), "selected_xi")
        self.assertEqual(p3.history_debug.get("xi_stage"), "bench")
        self.assertTrue(p1.history_debug.get("xi_selection_factors"))
        self.assertIn("Stage 1 XI selection", p1.history_debug.get("xi_selection_stage_reason"))
        self.assertIn("Bench after Stage 1 XI selection", p3.history_debug.get("xi_selection_stage_reason"))

    def test_batting_slot_eligibility_profile_uses_registry_slot_defaults_when_history_is_weak(self) -> None:
        keeper = _player(
            "Registry Keeper",
            role_bucket=WK_BATTER,
            role="wk",
            bat_skill=0.72,
            bowl_skill=0.1,
            batting_band="",
            role_band="wicketkeeper_batter",
            is_wicketkeeper=True,
            player_metadata={
                "allowed_batting_slots": [5, 6, 7],
                "preferred_batting_slots": [5, 6],
                "opener_eligible": False,
                "finisher_eligible": False,
                "floater_eligible": True,
                "role_description": "wicketkeeper_batter",
            },
        )

        profile = predictor._batting_slot_eligibility_profile(keeper)
        self.assertEqual(profile.get("allowed_slots"), [5, 6, 7])
        self.assertEqual(profile.get("preferred_slots"), [5, 6])
        self.assertEqual(profile.get("primary_source"), "preferred_slots_default")
        self.assertFalse(profile.get("opener_eligible"))

    def test_assign_batting_order_stage_respects_registry_slot_constraints(self) -> None:
        xi = [
            _player("Opener One", role_bucket=BATTER, role="bat", bat_skill=0.8, bowl_skill=0.2, batting_band="opener", role_band="opener", recent_ema=1.4, recent_rows=4, dominant_position=1.0, player_metadata={"allowed_batting_slots": [1, 2, 3], "preferred_batting_slots": [1, 2], "opener_eligible": True}),
            _player("Opener Two", role_bucket=BATTER, role="bat", bat_skill=0.76, bowl_skill=0.2, batting_band="top_order", role_band="top_order", recent_ema=2.8, recent_rows=4, dominant_position=2.0, player_metadata={"allowed_batting_slots": [1, 2, 3, 4], "preferred_batting_slots": [2, 3], "opener_eligible": True}),
            _player("Top Batter", role_bucket=BATTER, role="bat", bat_skill=0.74, bowl_skill=0.2, batting_band="top_order", role_band="top_order", recent_ema=3.8, recent_rows=3, dominant_position=4.0, player_metadata={"allowed_batting_slots": [2, 3, 4, 5], "preferred_batting_slots": [3, 4]}),
            _player("Middle One", role_bucket=BATTER, role="bat", bat_skill=0.71, bowl_skill=0.2, batting_band="middle_order", role_band="middle_order", recent_ema=5.2, recent_rows=3, dominant_position=5.0, player_metadata={"allowed_batting_slots": [4, 5, 6, 7], "preferred_batting_slots": [5, 6]}),
            _player("Keeper", role_bucket=WK_BATTER, role="wk", bat_skill=0.72, bowl_skill=0.2, batting_band="", role_band="wicketkeeper_batter", is_wicketkeeper=True, player_metadata={"allowed_batting_slots": [5, 6, 7], "preferred_batting_slots": [5, 6], "role_description": "wicketkeeper_batter", "floater_eligible": True}),
            _player("Finisher", role_bucket=BATTER, role="bat", bat_skill=0.69, bowl_skill=0.2, batting_band="", role_band="middle_order", player_metadata={"allowed_batting_slots": [5, 6, 7, 8], "preferred_batting_slots": [6, 7], "finisher_eligible": True, "role_description": "middle_order"}),
            _player("Bowling AR", role_bucket=ALL_ROUNDER, role="all", bat_skill=0.45, bowl_skill=0.74, batting_band="", role_band="bowling_allrounder", player_metadata={"allowed_batting_slots": [7, 8, 9, 10, 11], "preferred_batting_slots": [8, 9], "floater_eligible": True, "role_description": "bowling_allrounder"}),
            _player("Bowler One", role_bucket=BOWLER, role="bowl", bat_skill=0.2, bowl_skill=0.8, batting_band="", role_band="death_bowler", bowling_type="right_arm_fast", player_metadata={"allowed_batting_slots": [8, 9, 10, 11], "preferred_batting_slots": [9, 10], "role_description": "bowler"}),
            _player("Bowler Two", role_bucket=BOWLER, role="bowl", bat_skill=0.2, bowl_skill=0.77, batting_band="", role_band="powerplay_bowler", bowling_type="right_arm_fast_medium", player_metadata={"allowed_batting_slots": [8, 9, 10, 11], "preferred_batting_slots": [9, 10], "role_description": "bowler"}),
            _player("Bowler Three", role_bucket=BOWLER, role="bowl", bat_skill=0.18, bowl_skill=0.75, batting_band="", role_band="middle_overs_spinner", bowling_type="finger_spin", player_metadata={"allowed_batting_slots": [8, 9, 10, 11], "preferred_batting_slots": [9, 10], "role_description": "bowler"}),
            _player("Bowler Four", role_bucket=BOWLER, role="bowl", bat_skill=0.17, bowl_skill=0.74, batting_band="", role_band="utility_bowler", bowling_type="left_arm_fast_medium", player_metadata={"allowed_batting_slots": [8, 9, 10, 11], "preferred_batting_slots": [9, 10], "role_description": "bowler"}),
        ]
        conditions = {
            "batting_friendliness": 0.5,
            "dew_risk": 0.5,
            "spin_friendliness": 0.5,
            "pace_bias": 0.5,
            "rain_disruption_risk": 0.0,
        }

        order = predictor._assign_batting_order_stage(
            xi,
            conditions,
            team_name="Constraint XI",
            venue_keys=[],
            out_warnings=[],
        )

        slot_by_name = {name: idx + 1 for idx, name in enumerate(order)}
        self.assertGreaterEqual(slot_by_name["Keeper"], 5)
        self.assertIn(slot_by_name["Finisher"], {5, 6, 7, 8})
        self.assertGreaterEqual(slot_by_name["Bowling AR"], 7)


if __name__ == "__main__":
    unittest.main()
