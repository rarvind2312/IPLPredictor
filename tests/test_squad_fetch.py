"""Tests for IPLT20 squad HTML extraction (structured IplSquadMember)."""

from __future__ import annotations

import json
import unittest

import ipl_squad
import squad_fetch


class TestCleanCandidatePlayerNames(unittest.TestCase):
    def test_rejects_nav_footer_labels(self) -> None:
        raw = [
            ("ABOUT", "all"),
            ("GUIDELINES", "all"),
            ("CONTACT", "all"),
            ("Virat Kohli", "bat"),
            ("MS Dhoni", "wk"),
        ]
        accepted, rejected = squad_fetch.clean_candidate_player_names(raw)
        names = [a[0] for a in accepted]
        self.assertIn("Virat Kohli", names)
        self.assertIn("MS Dhoni", names)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(any("ABOUT" in r for r in rejected))

    def test_rejects_all_caps_short(self) -> None:
        acc, rej = squad_fetch.clean_candidate_player_names([("TERMS", "all")])
        self.assertEqual(acc, [])
        self.assertTrue(rej)


class TestSplitEmbeddedRole(unittest.TestCase):
    def test_strips_batter_suffix(self) -> None:
        base, b = ipl_squad.split_embedded_role_from_name("Shubham Dubey Batter")
        self.assertEqual(base, "Shubham Dubey")
        self.assertEqual(b, ipl_squad.BATTER)


class TestExtractSquadFromHtml(unittest.TestCase):
    def test_footer_headings_ignored(self) -> None:
        html = """
        <html><body>
        <main>
          <h2>Batters</h2>
          <h3>Rohit Sharma</h3>
          <a href="/players/123/jasprit-bumrah">Jasprit Bumrah</a>
        </main>
        <footer>
          <h3>ABOUT</h3>
          <h3>CONTACT</h3>
          <a href="/about">About Us</a>
        </footer>
        </body></html>
        """
        players, dbg = squad_fetch.extract_squad_players_from_html(html, source="test")
        names = [p.name for p in players]
        self.assertIn("Rohit Sharma", names)
        self.assertIn("Jasprit Bumrah", names)
        self.assertNotIn("ABOUT", names)
        self.assertNotIn("CONTACT", names)
        self.assertGreaterEqual(dbg.cleaned_count, 2)
        self.assertGreaterEqual(dbg.raw_candidate_count, 2)
        rohit = next(p for p in players if p.name == "Rohit Sharma")
        self.assertEqual(rohit.role_bucket, ipl_squad.BATTER)

    def test_next_data_squad_list(self) -> None:
        payload = {
            "props": {
                "pageProps": {
                    "batters": [
                        {
                            "playerName": "Foo Bar",
                            "jerseyNo": 10,
                            "playingRole": "Batter",
                            "playerSkill": "BAT",
                        },
                        {
                            "playerName": "Terms Policy",
                            "jerseyNo": 1,
                            "playingRole": "Batter",
                        },
                    ]
                    * 4
                }
            }
        }
        inner = payload["props"]["pageProps"]["batters"]
        assert len(inner) == 8
        script = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload)
            + "</script>"
        )
        html = f"<html><body>{script}</body></html>"
        players, dbg = squad_fetch.extract_squad_players_from_html(html, source="test")
        self.assertTrue(any(p.name == "Foo Bar" for p in players))
        self.assertFalse(any("Terms Policy" == p.name for p in players))
        self.assertIsInstance(dbg.rejected_sample, list)
        foo = next(p for p in players if p.name == "Foo Bar")
        self.assertEqual(foo.role_bucket, ipl_squad.BATTER)

    def test_foreign_player_icon_marks_overseas(self) -> None:
        html = """
        <html><body><main>
          <div class="squad-player-card">
            <img src="https://www.iplt20.com/assets/images/teams-foreign-player-icon.svg" alt="" />
            <a href="/players/1/test-player">Trent Boult</a>
          </div>
          <div class="squad-player-card">
            <a href="/players/2/local">Rohit Sharma</a>
          </div>
        </main></body></html>
        """
        players, dbg = squad_fetch.extract_squad_players_from_html(html, source="test")
        boult = next((p for p in players if "Boult" in p.name), None)
        rohit = next((p for p in players if "Rohit" in p.name), None)
        self.assertIsNotNone(boult)
        self.assertIsNotNone(rohit)
        self.assertTrue(boult.overseas)
        self.assertFalse(rohit.overseas)
        self.assertGreaterEqual(getattr(dbg, "foreign_player_icon_hits", 0), 1)


if __name__ == "__main__":
    unittest.main()
