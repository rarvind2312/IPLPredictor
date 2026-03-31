"""Tests for Cricsheet-driven recent-form cache (SQLite-only at prediction time)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import config
import db
import recent_form_cache
import selection_model
from types import SimpleNamespace


class TestGroupUnionMatches(unittest.TestCase):
    def test_union_includes_beyond_last_n_when_in_month_window(self) -> None:
        ref = date(2024, 6, 15)
        rows = [
            {"match_id": i, "match_date": f"2024-06-{10 - i:02d}"} for i in range(1, 8)
        ]
        u = recent_form_cache._group_union_matches(rows, last_n=5, months=5, ref=ref)
        ids = {r["match_id"] for r in u}
        self.assertEqual(ids, set(range(1, 8)))

    def test_old_match_outside_months_dropped_unless_in_last_n(self) -> None:
        ref = date(2024, 6, 15)
        # Newest-first (same as SQL ORDER BY match_date DESC).
        rows = [
            {"match_id": 2, "match_date": "2024-06-10"},
            {"match_id": 3, "match_date": "2023-06-01"},
            {"match_id": 1, "match_date": "2023-01-01"},
        ]
        u1 = recent_form_cache._group_union_matches(rows, last_n=1, months=5, ref=ref)
        self.assertEqual({r["match_id"] for r in u1}, {2})
        u2 = recent_form_cache._group_union_matches(rows, last_n=2, months=5, ref=ref)
        # Match 3 is outside ~5-month window; still included via last_n=2 with match 2.
        self.assertEqual({r["match_id"] for r in u2}, {2, 3})


class TestRebuildRecentFormCache(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = str(Path(self.tmp.name) / "rfc.sqlite")

    def tearDown(self) -> None:
        config.DB_PATH = self._orig
        self.tmp.cleanup()

    def test_rebuild_populates_cache_for_t20_player(self) -> None:
        db.init_schema()
        with db.connection() as conn:
            for mid, day in [(1, "2024-05-01"), (2, "2024-05-15"), (3, "2024-05-20")]:
                conn.execute(
                    """
                    INSERT INTO match_results (id, url, source, created_at)
                    VALUES (?, ?, 'test', 1.0)
                    """,
                    (mid, f"u{mid}"),
                )
                conn.execute(
                    """
                    INSERT INTO matches (
                        id, competition, match_date, venue, team_a, team_b, result,
                        scorecard_url, source, batting_first, created_at
                    ) VALUES (?, 'IPL', ?, 'V', 'A', 'B', '', '', 't', NULL, 1.0)
                    """,
                    (mid, day),
                )
                if "match_format" in db._matches_table_columns(conn):
                    conn.execute("UPDATE matches SET match_format = 'T20' WHERE id = ?", (mid,))
                conn.execute(
                    """
                    INSERT INTO player_match_stats (
                        match_id, team_name, team_key, player_name, player_key,
                        canonical_team_key, canonical_player_key,
                        runs, balls, fours, sixes, strike_rate,
                        overs_bowled, wickets, runs_conceded, economy, batting_position
                    ) VALUES (?, 'A', 'fr', 'Zed', 'zedplayer', 'fr', 'zedplayer',
                        20, 12, 1, 1, 160.0, 2.0, 1, 20, 10.0, 3)
                    """,
                    (mid,),
                )
            conn.commit()
            stats = recent_form_cache.rebuild_player_recent_form_cache(
                reference_iso_date="2024-06-01",
                conn=conn,
            )
        self.assertGreaterEqual(stats.get("players_cached", 0), 1)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM player_recent_form_cache WHERE player_key = ?",
                ("zedplayer",),
            ).fetchone()
        self.assertIsNotNone(row)
        d = dict(row)
        self.assertGreaterEqual(int(d["t20_matches_in_window"]), 3)
        dbg = json.loads(d["debug_json"] or "{}")
        self.assertGreaterEqual(int(dbg.get("t20_matches_in_union_window", 0)), 3)


class TestSelectionUsesCache(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = str(Path(self.tmp.name) / "sel_cache.sqlite")

    def tearDown(self) -> None:
        config.DB_PATH = self._orig
        self.tmp.cleanup()

    def test_apply_selection_reads_cache_not_pms(self) -> None:
        db.init_schema()
        with db.connection() as conn:
            now = 1.0
            conn.execute(
                "INSERT INTO match_results (id, url, source, created_at) VALUES (1, 'x', 't', ?)",
                (now,),
            )
            conn.execute(
                """
                INSERT INTO matches (
                    id, competition, match_date, venue, team_a, team_b, result,
                    scorecard_url, source, batting_first, created_at
                ) VALUES (1, 'IPL', '2024-01-01', 'V', 'A', 'B', '', '', 't', NULL, ?)
                """,
                (now,),
            )
            if "match_format" in db._matches_table_columns(conn):
                conn.execute("UPDATE matches SET match_format = 'T20' WHERE id = 1")
            conn.execute(
                """
                INSERT INTO player_recent_form_cache (
                    player_key, last_updated, reference_as_of_date,
                    t20_matches_in_window, batting_recent_form, bowling_recent_form,
                    combined_recent_form, last_t20_match_date, competitions_json,
                    matches_last_30d, matches_last_60d, matches_last_150d,
                    sample_confidence, debug_json
                ) VALUES ('pk1', ?, '2024-06-01', 5, 0.9, 0.2, 0.55, '2024-01-01',
                    '["IPL"]', 1, 2, 5, 0.7, '{}')
                """,
                (now,),
            )
            conn.commit()

        p = SimpleNamespace(
            name="N1",
            player_key="pk1",
            role_bucket=selection_model.BATTER,
            bat_skill=0.6,
            bowl_skill=0.3,
            bowling_type="pace",
            composite=0.5,
            history_debug={},
        )
        selection_model.apply_selection_model(
            [p],
            conditions={
                "batting_friendliness": 0.5,
                "spin_friendliness": 0.5,
                "pace_bias": 0.5,
            },
            franchise_team_key="other_franchise",
            profiles={},
            venue_weights={},
            pattern_row=None,
            fixture_context={"reference_iso_date": "2024-06-01"},
            hn_by_player={"pk1": 0.5},
            history_weights_by_pk={"pk1": 0.9},
            composite_by_player={"N1": 0.5},
        )
        dbg = p.history_debug.get("selection_model_debug", {})
        rf_detail = dbg.get("recent_form_detail") or {}
        self.assertEqual(rf_detail.get("recent_form_source"), "player_recent_form_cache")
        self.assertGreater(float(rf_detail.get("batting_recent_form", 0)), 0.85)
