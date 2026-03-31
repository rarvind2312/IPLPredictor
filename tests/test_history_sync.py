"""SQLite match history, local history debug (no pre-match scraping), and prediction resilience."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
import db
import history_sync
import utils


def _xi_players(prefix: str) -> list[str]:
    return [f"{prefix}P{i}" for i in range(1, 12)]


def _sample_payload(url: str, date: str = "2025-04-10") -> dict:
    ta, tb = "Mumbai Indians", "Chennai Super Kings"
    pl_a = _xi_players("MI")
    pl_b = _xi_players("CSK")
    return {
        "meta": {
            "url": url,
            "source": "ipl",
            "venue": "Wankhede",
            "date": date,
            "winner": ta,
            "batting_first": tb,
            "margin": "MI won by 5 wickets",
        },
        "teams": [ta, tb],
        "playing_xi": [
            {"team": ta, "players": pl_a},
            {"team": tb, "players": pl_b},
        ],
        "batting": [
            {"team": ta, "rows": [{"player": pl_a[0], "position": 1, "runs": 40, "balls": 30}]},
            {"team": tb, "rows": [{"player": pl_b[0], "position": 1, "runs": 20, "balls": 15}]},
        ],
        "bowling": [
            {"team": tb, "rows": [{"player": pl_b[10], "overs": 4.0, "maidens": 0, "runs": 30, "wickets": 2}]},
        ],
        "ingestion": {
            "fetch_ok": True,
            "errors": [],
            "has_storable_content": True,
        },
    }


def _squad_block(tag: str) -> str:
    lines = [f"{tag} Wicket Keeper Alpha | WK-Batter"]
    for i in range(6):
        lines.append(f"{tag} Batter Squad Player {i} | Batter")
    lines.append(f"{tag} All Rounder One | All-Rounder")
    for i in range(5):
        lines.append(f"{tag} Pace Bowler Number {i} | Bowler")
    lines.append(f"{tag} All Rounder Two | All-Rounder")
    return "\n".join(lines)


class TestHistorySyncDb(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp()) / "hist.sqlite"
        self._patch = unittest.mock.patch.object(config, "DB_PATH", self._tmp)
        self._patch.start()
        db.init_schema()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_url_duplicate_returns_duplicate_url(self) -> None:
        p = _sample_payload("https://www.iplt20.com/match/2025/1")
        mid1, s1 = db.insert_parsed_match(p)
        self.assertEqual(s1, "inserted")
        mid2, s2 = db.insert_parsed_match(p)
        self.assertEqual(s2, "duplicate_url")
        self.assertEqual(mid1, mid2)

    def test_canonical_duplicate_different_url(self) -> None:
        p1 = _sample_payload("https://www.iplt20.com/match/2025/1", "2025-04-10")
        mid1, s1 = db.insert_parsed_match(p1)
        self.assertEqual(s1, "inserted")
        p2 = _sample_payload("https://www.cricbuzz.com/live-cricket-scores/99999/foo", "2025-04-10")
        mid2, s2 = db.insert_parsed_match(p2)
        self.assertEqual(s2, "duplicate_match")
        self.assertEqual(mid1, mid2)

    def test_get_cached_match_count_and_samples(self) -> None:
        p = _sample_payload("https://www.iplt20.com/match/2025/2")
        db.insert_parsed_match(p)
        n = db.get_cached_match_count_for_franchise("Mumbai Indians")
        self.assertGreaterEqual(n, 1)
        rows = db.franchise_recent_match_summaries("Mumbai Indians", limit=3)
        self.assertTrue(any("Mumbai" in str(r.get("team_a") or r.get("team_b")) for r in rows))

    def test_history_rows_include_batting_position(self) -> None:
        p = _sample_payload("https://www.iplt20.com/match/2025/3")
        mid, _st = db.insert_parsed_match(p)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT batting_position FROM team_match_xi WHERE match_id = ? AND batting_position IS NOT NULL LIMIT 1",
                (mid,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNotNone(row[0])

    def test_utils_canonical_key_order_insensitive(self) -> None:
        k1 = utils.canonical_match_identity_key("Mumbai Indians", "Chennai Super Kings", "2025-04-10")
        k2 = utils.canonical_match_identity_key("Chennai Super Kings", "Mumbai Indians", "2025-04-10")
        self.assertEqual(k1, k2)


class TestLocalHistoryDebug(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp()) / "hist3.sqlite"
        self._patch = unittest.mock.patch.object(config, "DB_PATH", self._tmp)
        self._patch.start()
        db.init_schema()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_local_history_debug_is_sqlite_only(self) -> None:
        dbg = history_sync.local_history_debug_for_prediction(
            "Mumbai Indians",
            squad_player_names=["Rohit Sharma"],
        )
        self.assertTrue(dbg.get("deprecated_prematch_internet_sync_removed"))
        self.assertEqual(dbg.get("history_source"), "local_sqlite_only")
        self.assertIsNotNone(dbg.get("squad_vs_history_match_report"))


class TestPredictionFailsafe(unittest.TestCase):
    @patch("predictor.history_sync.local_history_debug_for_prediction", side_effect=RuntimeError("db error"))
    def test_run_prediction_continues_when_local_history_raises(self, _mock: MagicMock) -> None:
        from venues import VENUES

        import predictor

        venue = next(iter(VENUES.values()))
        weather = {
            "ok": False,
            "error": "x",
            "temperature_c": 28.0,
            "relative_humidity_pct": 50.0,
            "precipitation_mm": 0.0,
            "precipitation_probability_pct": 0.0,
            "cloud_cover_pct": 40.0,
            "wind_kmh": 10.0,
            "hour_iso": "2026-04-01T19:30:00",
        }
        sa = _squad_block("MA")
        sb = _squad_block("CB")
        r = predictor.run_prediction(
            "Mumbai Indians",
            "Chennai Super Kings",
            sa,
            sb,
            "",
            venue,
            datetime(2026, 4, 1, 19, 30),
            weather,
            toss_scenario_key="unknown",
        )
        self.assertIn("team_a", r)
        self.assertIn("history_sync_debug", r)
        self.assertIn("prediction_layer_debug", r)
        self.assertIsNotNone(r["history_sync_debug"].get("local_history_warning"))


if __name__ == "__main__":
    unittest.main()
