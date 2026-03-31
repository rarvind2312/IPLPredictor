"""Unit-style checks for scorecard ingestion (no live network by default)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from parsers import cricbuzz_parser, cricinfo_parser, ipl_parser
from parsers.router import parse_scorecard
from parsers.schema import (
    attach_ingestion_meta,
    compute_completeness,
    enrich_payload,
    has_storable_content,
    scorecard_core_empty,
)


SAMPLE_CRICINFO_HTML = """
<!DOCTYPE html>
<html><head>
<title>Mumbai Indians vs Chennai Super Kings - IPL Match</title>
<meta property="og:title" content="MI vs CSK" />
</head><body>
<span>Venue: Wankhede Stadium, Mumbai</span>
<span>Match 12 Apr 2024</span>
<p>Mumbai Indians won by 5 wickets</p>
<h3>Chennai Super Kings Innings</h3>
<table>
<tr><th>Batter</th><th>R</th><th>B</th></tr>
<tr><td><a href="/player/ruturaj-gaikwad-123">Ruturaj Gaikwad</a></td><td>45</td><td>32</td></tr>
<tr><td><a href="/player/devon-conway-456">Devon Conway</a></td><td>30</td><td>24</td></tr>
<tr><td><a href="/player/ajinkya-789">Ajinkya Rahane</a></td><td>12</td><td>10</td></tr>
</table>
<h3>Mumbai Indians Innings</h3>
<table>
<tr><td><a href="/player/rohit-1">Rohit Sharma</a></td><td>50</td><td>40</td></tr>
<tr><td><a href="/player/ishan-2">Ishan Kishan</a></td><td>20</td><td>15</td></tr>
<tr><td><a href="/player/surya-3">Suryakumar Yadav</a></td><td>25</td><td>18</td></tr>
</table>
<h3>Playing XI - Mumbai Indians</h3>
<div><a href="/player/rohit-1">Rohit Sharma</a>
<a href="/player/ishan-2">Ishan Kishan</a>
<a href="/player/surya-3">Suryakumar Yadav</a>
<a href="/player/tim-4">Tim David</a>
<a href="/player/p5">P5</a><a href="/player/p6">P6</a><a href="/player/p7">P7</a>
<a href="/player/p8">P8</a><a href="/player/p9">P9</a><a href="/player/p10">P10</a>
<a href="/player/p11">P11</a>
</div>
</body></html>
"""

SAMPLE_CRICBUZZ_HTML = """
<!DOCTYPE html><html><head>
<title>Royal Challengers Bangalore vs Kolkata Knight Riders | Cricbuzz</title>
</head><body>
<div>Venue: M Chinnaswamy Stadium, Bengaluru</div>
<div>Toss: Kolkata Knight Riders, elected to bowl first</div>
<div>RCB won by 10 runs</div>
<div class="cb-play11"><a href="/profiles/virat-kohli">Virat Kohli</a>
<a href="/profiles/faf-du-plessis">Faf du Plessis</a>
<a href="/profiles/glenn-maxwell">Glenn Maxwell</a>
<a href="/profiles/dk">Dinesh Karthik</a>
<a href="/profiles/w1">W1</a><a href="/profiles/w2">W2</a><a href="/profiles/w3">W3</a>
<a href="/profiles/w4">W4</a><a href="/profiles/w5">W5</a><a href="/profiles/w6">W6</a>
<a href="/profiles/w7">W7</a>
</div>
<table><tr><td><a href="/profiles/virat-kohli">Virat Kohli</a></td><td>80</td><td>50</td></tr>
<tr><td><a href="/profiles/faf-du-plessis">Faf du Plessis</a></td><td>40</td><td>30</td></tr>
<tr><td><a href="/profiles/glenn-maxwell">Glenn Maxwell</a></td><td>25</td><td>12</td></tr>
</table>
<table><tr><th>O</th><th>M</th><th>R</th><th>W</th></tr>
<tr><td><a href="/profiles/sunil-narine">Sunil Narine</a></td><td>4.0</td><td>0</td><td>30</td><td>2</td></tr>
<tr><td><a href="/profiles/andre-russell">Andre Russell</a></td><td>3.0</td><td>0</td><td>28</td><td>1</td></tr>
</table>
</body></html>
"""

def _ipl_next_data_match_html() -> str:
    """Synthetic IPL Next.js payload (shape mirrors official __NEXT_DATA__ match pages)."""
    import json

    payload = {
        "props": {
            "pageProps": {
                "matchDetail": {
                    "team1": {"fullName": "Delhi Capitals", "shortName": "DC"},
                    "team2": {"fullName": "Sunrisers Hyderabad", "shortName": "SRH"},
                    "venue": {"name": "Arun Jaitley Stadium, Delhi"},
                    "matchDate": "2024-04-01",
                    "result": "Delhi Capitals won the match by 7 runs",
                    "innings": [
                        {
                            "team": {"fullName": "Delhi Capitals"},
                            "batsmen": [
                                {"playerName": "David Warner", "runs": 60, "balls": 40},
                                {"playerName": "Mitchell Marsh", "runs": 35, "balls": 22},
                                {"playerName": "Rilee Rossouw", "runs": 20, "balls": 15},
                            ],
                            "bowlers": [
                                {"playerName": "Bhuvneshwar Kumar", "overs": 4.0, "maidens": 0, "runs": 32, "wickets": 1},
                                {"playerName": "Umran Malik", "overs": 4.0, "maidens": 0, "runs": 40, "wickets": 0},
                            ],
                        }
                    ],
                }
            }
        }
    }
    script = (
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload)
        + "</script>"
    )
    return f"""<!DOCTYPE html><html><head>
<title>IPL 2024 | Match 99: DC vs SRH</title>
</head><body><main>{script}
<p style="color:#fff;">#fff;</p>
</main></body></html>"""


SAMPLE_IPL_HTML = _ipl_next_data_match_html()


class TestParsersRobustness(unittest.TestCase):
    def test_cricinfo_never_crashes_on_garbage(self) -> None:
        r = cricinfo_parser.parse("<html>not a scorecard", "https://www.espncricinfo.com/x")
        self.assertIsInstance(r, dict)
        self.assertIn("meta", r)

    def test_cricbuzz_never_crashes_on_garbage(self) -> None:
        r = cricbuzz_parser.parse("", "https://www.cricbuzz.com/x")
        self.assertIsInstance(r, dict)

    def test_ipl_never_crashes_on_garbage(self) -> None:
        r = ipl_parser.parse("<html/>", "https://www.iplt20.com/match/1")
        self.assertIsInstance(r, dict)


class TestCricinfoSample(unittest.TestCase):
    def test_extracts_core_fields(self) -> None:
        r = cricinfo_parser.parse(
            SAMPLE_CRICINFO_HTML,
            "https://www.espncricinfo.com/series/ipl-2024/match/123456",
        )
        self.assertGreaterEqual(len(r.get("teams") or []), 1)
        self.assertTrue(r.get("batting"))
        meta = r.get("meta") or {}
        self.assertTrue(meta.get("venue") or meta.get("margin"))


class TestIPLSample(unittest.TestCase):
    def test_structured_next_data_teams_venue_date(self) -> None:
        r = ipl_parser.parse(SAMPLE_IPL_HTML, "https://www.iplt20.com/match/2024/99")
        self.assertEqual(
            r.get("teams"),
            ["Delhi Capitals", "Sunrisers Hyderabad"],
        )
        meta = r.get("meta") or {}
        self.assertIn("Arun Jaitley", meta.get("venue") or "")
        self.assertEqual(meta.get("date"), "2024-04-01")
        self.assertNotEqual(meta.get("venue", "").strip(), "#fff;")
        dbg = meta.get("ipl_parse_debug") or {}
        self.assertEqual(dbg.get("parser_path"), "next_data")
        self.assertTrue(dbg.get("scorecard_container_found"))

    def test_tables_and_enrichment(self) -> None:
        r = ipl_parser.parse(SAMPLE_IPL_HTML, "https://www.iplt20.com/match/2024/99")
        w: list[str] = []
        enrich_payload(r, SAMPLE_IPL_HTML, "ipl", w)
        self.assertTrue(r.get("batting"))
        self.assertTrue(r.get("bowling"))
        self.assertTrue(r.get("bowlers_used"))
        meta = r.get("meta") or {}
        self.assertIn("won", (meta.get("margin") or "").lower())
        self.assertFalse(scorecard_core_empty(r))
        self.assertTrue(has_storable_content(r))


class TestIPLNoScorecard(unittest.TestCase):
    def test_warns_when_no_structured_scorecard(self) -> None:
        """Title-only shell page: no __NEXT_DATA__ scorecard → not storable, parse_ok false."""
        html = """<!DOCTYPE html><html><head>
<title>IPL 2026 | Match 3: RR vs CSK</title></head><body>
<div>Venue: #fff;</div><p>Some filler text 15 April 2026</p></body></html>"""
        r = ipl_parser.parse(html, "https://www.iplt20.com/match/2026/2419")
        w: list[str] = []
        enrich_payload(r, html, "ipl", w)
        out = attach_ingestion_meta(
            r,
            source="ipl",
            fetch_ok=True,
            fetch_error=None,
            parse_errors=[],
            warnings=w,
        )
        self.assertFalse(out["ingestion"]["has_storable_content"])
        self.assertFalse(out["ingestion"]["parse_ok"])
        self.assertTrue(out["ingestion"]["ipl_scorecard_missing"])
        self.assertTrue(
            any("scorecard data was not found" in x for x in out["ingestion"]["warnings"])
        )


class TestCricbuzzSample(unittest.TestCase):
    def test_playing_xi_and_margin(self) -> None:
        r = cricbuzz_parser.parse(SAMPLE_CRICBUZZ_HTML, "https://www.cricbuzz.com/live-cricket-scores/123")
        self.assertTrue(r.get("playing_xi"))
        meta = r.get("meta") or {}
        self.assertIn("won", (meta.get("margin") or "").lower())


class TestSchemaEnrichment(unittest.TestCase):
    def test_derives_batting_order_and_bowlers(self) -> None:
        payload = {
            "meta": {"url": "http://x", "source": "test"},
            "teams": ["A", "B"],
            "playing_xi": [],
            "batting": [
                {
                    "team": "A",
                    "rows": [
                        {"player": "P2", "position": 2, "runs": 1, "balls": 1},
                        {"player": "P1", "position": 1, "runs": 0, "balls": 1},
                    ],
                }
            ],
            "bowling": [
                {
                    "team": "B",
                    "rows": [
                        {"player": "Bowler A", "overs": 4.0, "maidens": 0, "runs": 30, "wickets": 2},
                        {"player": "Bowler B", "overs": 4.0, "maidens": 0, "runs": 28, "wickets": 1},
                    ],
                }
            ],
        }
        w: list[str] = []
        enrich_payload(payload, "<html/>", "test", w)
        self.assertEqual(payload["batting_order"][0]["order"], ["P1", "P2"])
        self.assertEqual(payload["bowlers_used"][0]["bowlers"], ["Bowler A", "Bowler B"])
        c = compute_completeness(payload)
        self.assertTrue(c["batting_order"])
        self.assertTrue(c["bowlers_used"])


class TestRouter(unittest.TestCase):
    def test_empty_url(self) -> None:
        p = parse_scorecard("  ")
        self.assertIn("ingestion", p)
        self.assertFalse(p["ingestion"]["has_storable_content"])

    def test_unsupported_domain_no_fetch(self) -> None:
        p = parse_scorecard("https://example.com/scorecard/1")
        self.assertFalse(p["ingestion"]["fetch_ok"])
        self.assertTrue(any("unsupported" in e for e in p["ingestion"]["errors"]))

    @patch("parsers.router.fetch_html_safe")
    def test_cricinfo_end_to_end_mocked_fetch(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = (SAMPLE_CRICINFO_HTML, None)
        p = parse_scorecard("https://www.espncricinfo.com/series/ipl/match/1")
        self.assertTrue(p["ingestion"]["fetch_ok"])
        self.assertEqual(p["ingestion"]["source"], "cricinfo")
        self.assertTrue(has_storable_content(p))
        self.assertIn("batting_order", p)
        self.assertIn("bowlers_used", p)

    @patch("parsers.router.fetch_html_safe")
    def test_parse_exception_still_returns_payload(self, mock_fetch: MagicMock) -> None:
        def boom(*_a, **_k):
            raise RuntimeError("boom")

        mock_fetch.return_value = ("<html/>", None)
        with patch("parsers.cricinfo_parser.parse", side_effect=boom):
            p = parse_scorecard("https://www.espncricinfo.com/match/1")
            self.assertFalse(p["ingestion"]["parse_ok"])
            self.assertTrue(any("parse_exception" in e for e in p["ingestion"]["errors"]))


if __name__ == "__main__":
    unittest.main()
