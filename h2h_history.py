"""
Head-to-head (Team A vs Team B) history helpers: recency weights and row filtering.

Uses only local SQLite / Cricsheet-backed tables. IPL franchise names are resolved via
``ipl_teams`` where needed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import ipl_teams
import learner


def _norm_franchise(s: str) -> str:
    return learner.normalize_player_key(s or "")[:120]


def rows_are_h2h(team_a_label: str, team_b_label: str, row_team_a: str, row_team_b: str) -> bool:
    """True if the row is a fixture between the two franchises (order-independent)."""
    ca = ipl_teams.canonical_franchise_label(team_a_label) or (team_a_label or "").strip()
    cb = ipl_teams.canonical_franchise_label(team_b_label) or (team_b_label or "").strip()
    ra = ipl_teams.canonical_franchise_label_from_history_name(row_team_a) or (row_team_a or "").strip()
    rb = ipl_teams.canonical_franchise_label_from_history_name(row_team_b) or (row_team_b or "").strip()
    na, nb = _norm_franchise(ca), _norm_franchise(cb)
    rna, rnb = _norm_franchise(ra), _norm_franchise(rb)
    if len(na) < 2 or len(nb) < 2 or len(rna) < 2 or len(rnb) < 2:
        return False
    return {na, nb} == {rna, rnb}


def team_equals_label(team_label: str, row_team: str) -> bool:
    c = ipl_teams.canonical_franchise_label(team_label) or (team_label or "").strip()
    r = ipl_teams.canonical_franchise_label_from_history_name(row_team) or (row_team or "").strip()
    return _norm_franchise(c) == _norm_franchise(r)


def year_from_match_row(row: dict[str, Any], *, fallback_created: bool = True) -> Optional[int]:
    md = row.get("match_date")
    if md:
        m = re.search(r"(20\d{2})", str(md))
        if m:
            return int(m.group(1))
    if fallback_created and row.get("created_at") is not None:
        try:
            import time

            y = time.gmtime(float(row["created_at"])).tm_year
            if 2000 <= y <= 2100:
                return int(y)
        except (TypeError, ValueError, OSError):
            pass
    return None


def recency_weight(match_year: Optional[int], current_season_year: int) -> float:
    """
    Down-weight older H2H seasons so current-season patterns dominate.

    Current season → 1.0; prior season → ~0.82; further back decays by ~0.72 per year, floor 0.28.
    """
    cur = int(current_season_year)
    if match_year is None:
        return 0.55
    y = int(match_year)
    if y >= cur:
        return 1.0
    if y == cur - 1:
        return 0.82
    d = cur - y
    return max(0.28, 0.82 * (0.72 ** max(0, d - 1)))


def filter_match_rows_to_h2h(
    rows: list[dict[str, Any]],
    team_a_label: str,
    team_b_label: str,
) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if rows_are_h2h(
            team_a_label,
            team_b_label,
            str(r.get("team_a") or ""),
            str(r.get("team_b") or ""),
        )
    ]


def sort_h2h_rows_recent_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[str, float]:
        d = str(r.get("match_date") or "")
        cid = float(r.get("created_at") or 0)
        return (d, cid)

    return sorted(rows, key=key, reverse=True)


def venue_matches_keys(venue_raw: str, venue_keys: list[str]) -> bool:
    if not venue_raw or not venue_keys:
        return False
    vn = learner.normalize_player_key(str(venue_raw))[:80]
    rv = str(venue_raw).lower()
    for vk in venue_keys:
        if not vk:
            continue
        k = learner.normalize_player_key(str(vk))[:80]
        if k in vn or vn in k or vk.lower() in rv or rv in vk.lower():
            return True
    return False
