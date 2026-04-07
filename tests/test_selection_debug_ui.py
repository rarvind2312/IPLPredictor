"""selection_debug_ui: table shape parity (Predict vs Admin debug)."""

from __future__ import annotations

import unittest

import selection_debug_ui


class TestSelectionDebugUi(unittest.TestCase):
    def test_include_reason_columns_preserves_column_sets(self) -> None:
        r = {
            "prediction_layer_debug": {
                "team_a": {
                    "scoring_breakdown_per_player": [
                        {
                            "squad_display_name": "Player One",
                            "final_selection_score": 10.0,
                            "selection_model_base": {
                                "recent_form_score": 0.5,
                                "ipl_history_and_role_score": 0.4,
                                "team_balance_fit_score": 0.3,
                                "venue_experience_score": 0.2,
                            },
                            "selection_reason_summary": "line1\nline2",
                            "recent_form_competitions_used": "IPL",
                        }
                    ]
                }
            },
            "team_a": {"xi": [{"name": "Player One"}], "impact_subs": []},
            "selection_debug": {"team_a": {}},
        }
        df_full, xi1 = selection_debug_ui.selection_debug_top15_dataframe_for_side(
            r, "team_a", include_reason_columns=True
        )
        df_slim, xi2 = selection_debug_ui.selection_debug_top15_dataframe_for_side(
            r, "team_a", include_reason_columns=False
        )
        self.assertEqual(xi1, xi2)
        slim_cols = [
            "player",
            "in_playing_xi",
            "impact_candidate",
            "recent_form_score",
            "ipl_history_and_role_score",
            "team_balance_fit_score",
            "venue_experience_score",
            "tactical_adjustment_total",
            "final_selection_score",
        ]
        self.assertEqual(list(df_slim.columns), slim_cols)
        self.assertEqual(list(df_full.columns), slim_cols + ["recent_form_competitions", "reason_summary"])
        for c in slim_cols:
            self.assertEqual(df_full.iloc[0][c], df_slim.iloc[0][c], msg=c)
        self.assertEqual(df_full.iloc[0]["recent_form_competitions"], "IPL")
        self.assertIn("line1", df_full.iloc[0]["reason_summary"])
        self.assertNotIn("\n", df_full.iloc[0]["reason_summary"])


if __name__ == "__main__":
    unittest.main()
