"""Tests for modular Stage-3 selection_model (T20 recent form, team balance, venue/tactical)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import config
import db
import selection_model


class TestT20FamilyFilter(unittest.TestCase):
    def test_ipl_and_t20_strings_accepted(self) -> None:
        self.assertTrue(db.match_row_is_t20_family("IPL", None))
        self.assertTrue(db.match_row_is_t20_family("Vitality Blast", None))
        self.assertTrue(db.match_row_is_t20_family("Foo", "T20"))

    def test_explicit_non_t20_rejected(self) -> None:
        self.assertFalse(db.match_row_is_t20_family("International", "Test"))
        self.assertFalse(db.match_row_is_t20_family("International", "ODI"))

    def test_test_match_name_excluded_when_not_also_t20(self) -> None:
        self.assertFalse(db.match_row_is_t20_family("Sheffield Shield", None))


class TestRecentFormWeighting(unittest.TestCase):
    def test_bowler_weights_bowling_more(self) -> None:
        p = SimpleNamespace(
            role_bucket=selection_model.BOWLER,
            bat_skill=0.5,
            bowl_skill=0.8,
            bowling_type="pace",
            composite=0.5,
            name="X",
        )
        cache_row = {
            "t20_matches_in_window": 4,
            "batting_recent_form": 0.35,
            "bowling_recent_form": 0.82,
            "last_t20_match_date": "2024-01-01",
            "competitions_json": '["IPL"]',
            "matches_last_30d": 2,
            "matches_last_60d": 4,
            "matches_last_150d": 4,
            "sample_confidence": 0.55,
            "debug_json": "{}",
        }
        s, dbg = selection_model._recent_form_score_player(
            p, cache_row, {"recent_usage_score": 0.5}, reference_date=date(2024, 6, 1)
        )
        self.assertGreater(dbg["recent_form_role_weights"]["bowling"], dbg["recent_form_role_weights"]["batting"])
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


class TestTeamBalanceEffect(unittest.TestCase):
    def test_wicketkeeper_boost_when_gap(self) -> None:
        def _p(
            name: str,
            *,
            rb: str,
            wk: bool = False,
            ol: float = 0.1,
            fl: float = 0.1,
            ppl: float = 0.1,
            dth: float = 0.1,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                name=name,
                role_bucket=rb,
                is_wicketkeeper=wk,
                is_opener_candidate=False,
                is_finisher_candidate=False,
                bowling_type="pace" if rb == selection_model.BOWLER else None,
                bat_skill=0.5,
                bowl_skill=0.5,
                history_debug={
                    "derive_player_profile": {
                        "opener_likelihood": ol,
                        "finisher_likelihood": fl,
                        "powerplay_bowler_likelihood": ppl,
                        "death_bowler_likelihood": dth,
                    }
                },
            )

        squad: list[SimpleNamespace] = [_p(f"B{i}", rb=selection_model.BATTER) for i in range(10)]
        squad.append(_p("Bowler1", rb=selection_model.BOWLER, ppl=0.55, dth=0.55))
        squad.append(_p("Keeper", rb=selection_model.WK_BATTER, wk=True))
        prelim = {f"B{i}": 0.96 - i * 0.001 for i in range(10)}
        prelim["Bowler1"] = 0.95
        prelim["Keeper"] = 0.2
        tb = selection_model._team_balance_fit_scores(squad, prelim)
        self.assertGreater(tb["Keeper"], tb["B0"])


class TestVenueModifier(unittest.TestCase):
    def test_spin_friendly_nudges_spinner(self) -> None:
        p = SimpleNamespace(
            role_bucket=selection_model.BOWLER,
            bowling_type="wrist spin",
            bat_skill=0.4,
            bowl_skill=0.7,
            name="S",
        )
        hd = {"venue_xi_rate": 0.4}
        ds = {"venue_fit_score": 0.5}
        cond = {
            "batting_friendliness": 0.5,
            "spin_friendliness": 0.62,
            "pace_bias": 0.45,
        }
        s_spin, _ = selection_model._venue_experience_score(
            p, hd, ds, {}, "k", cond
        )
        p2 = SimpleNamespace(
            role_bucket=selection_model.BOWLER,
            bowling_type="fast",
            bat_skill=0.4,
            bowl_skill=0.7,
            name="F",
        )
        s_seam, _ = selection_model._venue_experience_score(
            p2, hd, ds, {}, "k", cond
        )
        self.assertGreater(s_spin, s_seam)


class TestImpactDecisionLogic(unittest.TestCase):
    def test_explanation_mentions_margin_when_below_xi(self) -> None:
        import impact_subs_engine

        squad = [
            SimpleNamespace(
                name="A",
                role_bucket=selection_model.BATTER,
                player_key="a",
                history_debug={"bench_near_xi_margin": -0.08},
                selection_score=0.4,
                history_xi_score=0.3,
                bat_skill=0.6,
                bowl_skill=0.3,
                composite=0.5,
                is_overseas=False,
                bowling_type=None,
                is_finisher_candidate=False,
            )
        ]
        xi = [
            SimpleNamespace(
                name=f"X{i}",
                role_bucket=selection_model.BATTER,
                selection_score=0.85,
            )
            for i in range(11)
        ]
        top, dbg_top, _ = impact_subs_engine.rank_impact_sub_candidates(
            squad,
            xi,
            team_display_name="T",
            canonical_team_key="t1",
            venue_key="v1",
            venue_key_candidates=None,
            is_chasing=None,
            conditions={
                "batting_friendliness": 0.58,
                "spin_friendliness": 0.5,
                "pace_bias": 0.5,
                "rain_disruption_risk": 0.0,
                "dew_risk": 0.5,
            },
            team_bats_first=True,
        )
        self.assertEqual(len(top), 1)
        self.assertIn("margin", dbg_top[0]["impact_xi_projection_explanation"].lower())


class TestDbRecentFetch(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = str(Path(self.tmp.name) / "sel_model.sqlite")

    def tearDown(self) -> None:
        config.DB_PATH = self._orig
        self.tmp.cleanup()

    def test_fetch_groups_by_player(self) -> None:
        db.init_schema()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO match_results (id, url, source, created_at) VALUES (1, 'u1', 'test', 1.0)
                """
            )
            conn.execute(
                """
                INSERT INTO matches (
                    id, competition, match_date, venue, team_a, team_b, result,
                    scorecard_url, source, batting_first, created_at
                ) VALUES (1, 'IPL', '2024-03-01', 'V', 'A', 'B', '', '', 't', NULL, 1.0)
                """
            )
            if "match_format" in db._matches_table_columns(conn):
                conn.execute("UPDATE matches SET match_format = 'T20' WHERE id = 1")
            conn.execute(
                """
                INSERT INTO player_match_stats (
                    match_id, team_name, team_key, player_name, player_key,
                    canonical_team_key, canonical_player_key,
                    runs, balls, overs_bowled, wickets, runs_conceded, economy
                ) VALUES (1, 'A', 'fr', 'P1', 'p1', 'fr', 'p1', 30, 20, NULL, NULL, NULL, NULL)
                """
            )
        rows = db.fetch_recent_pms_rows_for_squad_players("fr", ["p1"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_key"], "p1")


if __name__ == "__main__":
    unittest.main()
