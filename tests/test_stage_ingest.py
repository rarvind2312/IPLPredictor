"""Tests for Stage 1 Cricsheet folder ingest (isolated SQLite + temp JSON dir)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import config
import cricsheet_ingest
import db

_MINIMAL_CRICSHEET = {
    "info": {
        "balls_per_over": 6,
        "teams": ["Chennai Super Kings", "Mumbai Indians"],
        "dates": ["2020-04-01"],
        "season": "2020",
        "venue": "Wankhede",
        "city": "Mumbai",
        "toss": {"winner": "Mumbai Indians", "decision": "field"},
        "outcome": {"winner": "Chennai Super Kings", "by": {"runs": 5}},
        "players": {
            "Chennai Super Kings": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"],
            "Mumbai Indians": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11"],
        },
    },
    "innings": [
        {
            "team": "Chennai Super Kings",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [{"batter": "P1", "bowler": "Q1", "runs": {"batter": 1, "total": 1}}],
                }
            ],
        },
        {
            "team": "Mumbai Indians",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [{"batter": "Q1", "bowler": "P1", "runs": {"batter": 0, "total": 0}}],
                }
            ],
        },
    ],
}


class TestStageIngest(unittest.TestCase):
    def setUp(self) -> None:
        self._db_orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        config.DB_PATH = str(root / "stage_ingest_test.sqlite")
        self.json_dir = root / "ipl_json"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.empty_readme = root / "readme_empty.txt"
        self.empty_readme.write_text("# no rows\n", encoding="utf-8")

    def tearDown(self) -> None:
        config.DB_PATH = self._db_orig
        self.tmp.cleanup()

    def test_ingest_then_skip_duplicate_and_malformed(self) -> None:
        db.init_schema()
        mid = "999999001"
        (self.json_dir / f"{mid}.json").write_text(
            json.dumps(_MINIMAL_CRICSHEET),
            encoding="utf-8",
        )
        s1 = cricsheet_ingest.run_cricsheet_folder_ingest(
            json_dir=self.json_dir,
            readme_path=self.empty_readme,
            report_readme_gaps=False,
        )
        self.assertEqual(s1.matches_inserted, 1)
        self.assertGreaterEqual(s1.player_stats_rows_inserted, 1)
        self.assertGreaterEqual(s1.batting_position_rows_inserted, 1)
        self.assertEqual(s1.readme_rows_total, 0)

        s2 = cricsheet_ingest.run_cricsheet_folder_ingest(
            json_dir=self.json_dir,
            readme_path=self.empty_readme,
            report_readme_gaps=False,
        )
        self.assertEqual(s2.matches_inserted, 0)
        self.assertGreaterEqual(s2.matches_skipped_duplicate, 1)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT cricsheet_match_id, city, season FROM matches WHERE cricsheet_match_id = ?",
                (mid,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(str(row["cricsheet_match_id"]), mid)
        self.assertEqual(row["city"], "Mumbai")
        self.assertEqual(row["season"], "2020")


if __name__ == "__main__":
    unittest.main()
