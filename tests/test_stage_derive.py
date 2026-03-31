"""Stage 2 derive: SQLite-only rebuilds (isolated DB)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
import db
import stage_derive


class TestStageDerive(unittest.TestCase):
    def setUp(self) -> None:
        self._db_orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = str(Path(self.tmp.name) / "derive_test.sqlite")

    def tearDown(self) -> None:
        config.DB_PATH = self._db_orig
        self.tmp.cleanup()

    def test_h2h_rebuild_empty_db(self) -> None:
        db.init_schema()
        s = stage_derive.run_rebuild_h2h_patterns()
        self.assertEqual(s.head_to_head_pattern_rows, 0)
        self.assertEqual(s.min_season_year, stage_derive.derive_min_season_year())

    def test_profiles_rebuild_empty(self) -> None:
        db.init_schema()
        s = stage_derive.run_rebuild_profiles()
        self.assertEqual(s.player_profiles_built, 0)
        snap = stage_derive.derive_debug_snapshot()
        self.assertIn("player_profiles_rows", snap)


if __name__ == "__main__":
    unittest.main()
