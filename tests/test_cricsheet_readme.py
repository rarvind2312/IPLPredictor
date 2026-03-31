"""Tests for local Cricsheet readme parsing (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cricsheet_readme


class TestCricsheetReadme(unittest.TestCase):
    def test_parse_sample_line(self) -> None:
        line = "2025-05-20 - club - IPL - male - 1473500 - Chennai Super Kings vs Rajasthan Royals"
        row = cricsheet_readme.parse_cricsheet_readme_line(line)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.match_date, "2025-05-20")
        self.assertEqual(row.team_type, "club")
        self.assertEqual(row.competition, "IPL")
        self.assertEqual(row.gender, "male")
        self.assertEqual(row.match_id, "1473500")
        self.assertEqual(row.team1, "Chennai Super Kings")
        self.assertEqual(row.team2, "Rajasthan Royals")

    def test_ignores_header_and_non_ipl(self) -> None:
        self.assertIsNone(cricsheet_readme.parse_cricsheet_readme_line("The matches contained"))
        self.assertIsNone(
            cricsheet_readme.parse_cricsheet_readme_line(
                "2020-01-01 - international - ODI - male - 999 - A vs B"
            )
        )

    def test_bundle_readme_if_present(self) -> None:
        root = Path(__file__).resolve().parent.parent
        readme = cricsheet_readme.resolve_readme_path(
            (root / "data" / "ipl_json" / "README.txt",),
        )
        if readme is None:
            self.skipTest("Local Cricsheet README not in workspace")
        rows = cricsheet_readme.parse_cricsheet_readme(readme)
        self.assertGreater(len(rows), 100)
        self.assertTrue(all(r.competition == "IPL" for r in rows))

    def test_season_window(self) -> None:
        y = cricsheet_readme.season_years_window(2026, 5)
        self.assertEqual(y, {2026, 2025, 2024, 2023, 2022})

    def test_flexible_dash_spacing(self) -> None:
        line = "2025-05-20  -  club  -  IPL  -  male  -  1473500  -  Chennai Super Kings vs Rajasthan Royals"
        row = cricsheet_readme.parse_cricsheet_readme_line(line)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.match_id, "1473500")

    def test_vs_case_insensitive(self) -> None:
        line = "2025-05-20 - club - IPL - male - 1 - Mumbai Indians VS Chennai Super Kings"
        row = cricsheet_readme.parse_cricsheet_readme_line(line)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.team1, "Mumbai Indians")
        self.assertEqual(row.team2, "Chennai Super Kings")

    def test_filter_last_n_seasons(self) -> None:
        rows = [
            cricsheet_readme.parse_cricsheet_readme_line(
                f"{y}-05-01 - club - IPL - male - {y}00 - A vs B"
            )
            for y in range(2018, 2028)
        ]
        rows = [r for r in rows if r is not None]
        f = cricsheet_readme.filter_last_n_seasons(rows, current_season_year=2026, n_seasons=5)
        years = {r.season_year for r in f}
        self.assertEqual(years, {2026, 2025, 2024, 2023, 2022})

    def test_extract_match_index_rows_dict(self) -> None:
        content = "Header line\n\n2025-05-20 - club - IPL - male - 1473500 - X vs Y\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=True,
            encoding="utf-8",
        ) as f:
            f.write(content)
            f.flush()
            d = cricsheet_readme.extract_match_index_rows(f.name)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["match_id"], "1473500")


if __name__ == "__main__":
    unittest.main()
