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


# Ordered (side1_key, side2_key): first match with both non-empty wins.
_TEAM_PAIR_COLUMN_KEYS: tuple[tuple[str, str], ...] = (
    ("team_a", "team_b"),
    ("team1", "team2"),
    ("home_team", "away_team"),
    ("team_a_name", "team_b_name"),
    ("home_team_name", "away_team_name"),
    ("team_home", "team_away"),
    ("side_1_team", "side_2_team"),
    ("homeTeam", "awayTeam"),
)


def row_team_names_pair(row: dict[str, Any]) -> tuple[str, str]:
    """
    Extract the two opponent team name fields from a match / meta row.

    Does not assume only ``team_a`` / ``team_b`` — supports alternate schemas and
    passes through any extra keys present on ``row`` (e.g. after ``SELECT *``).
    """
    for k1, k2 in _TEAM_PAIR_COLUMN_KEYS:
        v1 = str(row.get(k1) or "").strip()
        v2 = str(row.get(k2) or "").strip()
        if v1 and v2:
            return v1, v2
    for k1, k2 in _TEAM_PAIR_COLUMN_KEYS:
        v1 = str(row.get(k1) or "").strip()
        v2 = str(row.get(k2) or "").strip()
        if v1 or v2:
            return v1, v2
    # Last resort: any two distinct non-empty string columns whose names suggest a side team.
    _skip_keys = frozenset(
        {
            "winner",
            "batting_first",
            "venue",
            "match_date",
            "created_at",
            "id",
            "match_id",
        }
    )
    found: list[tuple[str, str]] = []
    for k in sorted(row.keys(), key=lambda x: str(x).lower()):
        if k in _skip_keys or str(k).lower() in _skip_keys:
            continue
        lk = str(k).lower()
        if "team" not in lk:
            continue
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if len(s) < 2:
            continue
        found.append((str(k), s))
    if len(found) >= 2:
        return found[0][1], found[1][1]
    return "", ""


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


def h2h_score_summary_display(row: dict[str, Any]) -> str:
    """Single cell: batting order + result / margin text for Team vs Team tables."""
    bf = str(row.get("batting_first") or "").strip()
    rt = str(row.get("result_text") or "").strip()
    mg = str(row.get("margin") or "").strip()
    parts: list[str] = []
    if bf:
        parts.append(f"Batting first: {bf}")
    if rt:
        parts.append(rt)
    elif mg:
        parts.append(mg)
    return "; ".join(parts) if parts else "—"


def enrich_h2h_summary_rows_from_match_results(
    rows: list[dict[str, Any]],
    enrich_by_id: dict[int, dict[str, str]],
) -> None:
    """
    Fill missing winner / margin / result_text on prediction-summary-shaped rows using
    ``fetch_match_result_display_enrichment_by_ids`` output (mutates rows in place).
    """
    for r in rows:
        mid = int(r.get("id") or r.get("match_id") or 0)
        if not mid:
            continue
        ex = enrich_by_id.get(mid)
        if not ex:
            continue
        w_sum = str(r.get("winner") or "").strip()
        if not w_sum:
            r["winner"] = str(ex.get("winner_mr") or ex.get("winner_m") or "").strip()

        m_sum = str(r.get("margin") or "").strip()
        if not m_sum:
            m_mr = str(ex.get("margin_mr") or "").strip()
            rt_m = str(ex.get("result_text_m") or "").strip()
            res_m = str(ex.get("result_m") or "").strip()
            r["margin"] = m_mr or rt_m or res_m

        rt_existing = str(r.get("result_text") or "").strip()
        if not rt_existing:
            rt_m = str(ex.get("result_text_m") or "").strip()
            res_m = str(ex.get("result_m") or "").strip()
            r["result_text"] = rt_m or res_m


def filter_match_rows_to_h2h(
    rows: list[dict[str, Any]],
    team_a_label: str,
    team_b_label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        ra, rb = row_team_names_pair(r)
        if rows_are_h2h(team_a_label, team_b_label, ra, rb):
            out.append(r)
    return out


def sort_h2h_rows_recent_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[str, float]:
        d = str(r.get("match_date") or "")
        cid = float(r.get("created_at") or 0)
        return (d, cid)

    return sorted(rows, key=key, reverse=True)


def winner_side_for_h2h_row(row: dict[str, Any], label_a: str, label_b: str) -> str:
    """
    Map a summary/meta row to ``a`` (``label_a`` franchise won), ``b``, ``none``, or ``unknown``.
    """
    la = ipl_teams.canonical_franchise_label(label_a) or (label_a or "").strip()
    lb = ipl_teams.canonical_franchise_label(label_b) or (label_b or "").strip()
    w = str(row.get("winner") or "").strip()
    low = w.lower()
    if not w or "tie" in low or "no result" in low or "abandon" in low:
        return "none"
    ta, tb = row_team_names_pair(row)
    if team_equals_label(w, ta):
        if team_equals_label(la, ta):
            return "a"
        if team_equals_label(lb, ta):
            return "b"
    if team_equals_label(w, tb):
        if team_equals_label(la, tb):
            return "a"
        if team_equals_label(lb, tb):
            return "b"
    if team_equals_label(w, la):
        return "a"
    if team_equals_label(w, lb):
        return "b"
    return "unknown"


def winner_display_for_row(row: dict[str, Any], label_a: str, label_b: str) -> str:
    side = winner_side_for_h2h_row(row, label_a, label_b)
    ta, tb = row_team_names_pair(row)
    if side == "a":
        return str(label_a or ta or "—")
    if side == "b":
        return str(label_b or tb or "—")
    return str(row.get("winner") or "—")


def summarize_h2h_fixture_rows(
    rows: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """
    Win split and last-3 trend from ``prediction_summary_match_meta``-shaped rows.

    ``avg_first_innings_runs`` is always ``None`` here (not present in that feed).
    """
    la = ipl_teams.canonical_franchise_label(label_a) or (label_a or "").strip()
    lb = ipl_teams.canonical_franchise_label(label_b) or (label_b or "").strip()
    wins_a = wins_b = ties = 0
    for row in rows:
        side = winner_side_for_h2h_row(row, la, lb)
        if side == "a":
            wins_a += 1
        elif side == "b":
            wins_b += 1
        else:
            ties += 1
    la3 = lb3 = 0
    for row in rows[:3]:
        s = winner_side_for_h2h_row(row, la, lb)
        if s == "a":
            la3 += 1
        elif s == "b":
            lb3 += 1
    return {
        "label_a": la,
        "label_b": lb,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties_nr": ties,
        "last3_wins_a": la3,
        "last3_wins_b": lb3,
        "avg_first_innings_runs": None,
    }


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


_IPL_VENUE_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("chidambaram", "chepauk", "chennai", "mambalam"),
    ("wankhede", "brabourne", "mumbai", "dy", "patil"),
    ("eden", "kolkata", "calcutta"),
    ("jaitley", "feroz", "kotla", "delhi", "arun"),
    ("chinnaswamy", "bengaluru", "bangalore"),
    ("sawai", "jaipur", "mansingh"),
    ("punjab", "mohali", "bindra"),
    ("rajiv", "hyderabad", "uppal"),
    ("bharat", "lucknow", "ekana"),
    ("narendra", "ahmedabad", "motera"),
)


def expand_matchup_venue_tokens(venue_display: str, venue_keys: list[str]) -> list[str]:
    """
    Build a loose search token list for venue matching (display string + model keys + IPL aliases).

    Used so rows like "MA Chidambaram Stadium, Chennai" match fixture keys such as "Chepauk" or "Chennai".
    """
    import re

    parts: list[str] = []
    for s in [venue_display, *venue_keys]:
        s = (s or "").strip()
        if not s:
            continue
        parts.append(s)
        for chunk in re.split(r"[,;/]", s):
            c = chunk.strip()
            if c:
                parts.append(c)
    tokens: set[str] = set()
    for p in parts:
        n = re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()
        if not n:
            continue
        tokens.add(n)
        for w in n.split():
            if len(w) >= 3:
                tokens.add(w)
    blob = " ".join(sorted(tokens))
    for group in _IPL_VENUE_ALIAS_GROUPS:
        if any(g in blob for g in group):
            tokens.update(group)
    return sorted(tokens)


def venue_row_matches_relaxed(venue_raw: str, expanded_tokens: list[str]) -> bool:
    """True if a history row's venue string matches any expanded fixture token (substring / word overlap)."""
    import re

    if not venue_raw or not expanded_tokens:
        return False
    v = re.sub(r"[^a-z0-9]+", " ", str(venue_raw).lower()).strip()
    if not v:
        return False
    vw = {w for w in v.split() if len(w) >= 3}
    for t in expanded_tokens:
        tn = re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()
        if not tn:
            continue
        if tn in v or v in tn:
            return True
        tw = {w for w in tn.split() if len(w) >= 3}
        if not tw:
            continue
        common = tw & vw
        if not common:
            continue
        if any(len(w) >= 5 for w in common):
            return True
        if len(common) >= 2:
            return True
    return False
