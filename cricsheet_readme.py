"""
Parse Cricsheet IPL bundle readme index (``data/readme.txt`` or ``data/ipl_json/README.txt``).

The file begins with explanatory prose, then one match per line::

    2025-05-20 - club - IPL - male - 1473500 - Chennai Super Kings vs Rajasthan Royals

Fields: match_date, team_type, competition, gender, numeric match_id, then
``<team1> vs <team2>`` (teams may contain spaces; split on the first case-insensitive `` vs ``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Leading record: date — team_type — competition — gender — id — remainder (teams)
_LINE_RE = re.compile(
    r"""
    ^(?P<match_date>\d{4}-\d{2}-\d{2})      # YYYY-MM-DD
    \s*-\s*
    (?P<team_type>club|international)      # per Cricsheet readme
    \s*-\s*
    (?P<competition>\S+)                    # IPL, Test, IT20, …
    \s*-\s*
    (?P<gender>male|female)
    \s*-\s*
    (?P<match_id>\d+)
    \s*-\s*
    (?P<teams>.+)$                          # "Team A vs Team B" (trimmed)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VS_SPLIT_RE = re.compile(r"\s+vs\s+", re.IGNORECASE)

# Lines that are clearly not match rows (extra safety on top of regex)
_URL_START_RE = re.compile(r"^https?://", re.I)
_BLANK_OR_COMMENT_RE = re.compile(r"^\s*(#|$)")


@dataclass(frozen=True)
class CricsheetReadmeRow:
    """One parsed match index line from the Cricsheet readme."""

    match_date: str
    team_type: str
    competition: str
    gender: str
    match_id: str
    team1: str
    team2: str

    def as_dict(self) -> dict[str, str]:
        return {
            "match_date": self.match_date,
            "team_type": self.team_type,
            "competition": self.competition,
            "gender": self.gender,
            "match_id": self.match_id,
            "team1": self.team1,
            "team2": self.team2,
        }

    @property
    def season_year(self) -> int:
        """Calendar year from ``match_date`` (first four digits)."""
        return int(self.match_date[:4])


def _normalize_unicode_line(line: str) -> str:
    """Normalize newlines / odd spaces; do not alter inner team spelling."""
    s = line.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFC", s)
    return s.strip()


def _is_probably_header_line(stripped_line: str) -> bool:
    if not stripped_line:
        return True
    if _BLANK_OR_COMMENT_RE.match(stripped_line):
        return True
    if _URL_START_RE.match(stripped_line):
        return True
    # Match rows always start with a full ISO date
    if len(stripped_line) < 10 or not re.match(r"^\d{4}-\d{2}-\d{2}", stripped_line):
        return True
    low = stripped_line.lower()
    if low.startswith("the ") or low.startswith("this ") or low.startswith("you can"):
        return True
    return False


def _split_teams_blob(teams_blob: str) -> Optional[tuple[str, str]]:
    raw = teams_blob.strip()
    if not raw:
        return None
    parts = _VS_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) != 2:
        return None
    t1, t2 = parts[0].strip(), parts[1].strip()
    if not t1 or not t2:
        return None
    return t1, t2


def parse_cricsheet_readme_line(
    line: str,
    *,
    competition: Optional[str] = "IPL",
    genders: Optional[set[str]] = None,
) -> Optional[CricsheetReadmeRow]:
    """
    Parse a single line into a row, or ``None`` if it is header/invalid.

    Parameters
    ----------
    competition:
        If set (default ``"IPL"``), only lines with this competition (case-insensitive) match.
        Pass ``None`` to accept any competition code.
    genders:
        Allowed gender values after lowercasing, e.g. ``{"male"}`` or ``{"male", "female"}``.
        Default is men’s IPL only: ``{"male"}``.
    """
    if genders is None:
        genders = {"male"}

    s = _normalize_unicode_line(line)
    if _is_probably_header_line(s):
        return None

    m = _LINE_RE.match(s)
    if not m:
        return None

    comp_raw = m.group("competition").strip()
    comp_norm = comp_raw.upper()
    if competition is not None and comp_norm != competition.strip().upper():
        return None

    gender_norm = m.group("gender").strip().lower()
    if gender_norm not in genders:
        return None

    teams = _split_teams_blob(m.group("teams"))
    if teams is None:
        return None
    t1, t2 = teams

    return CricsheetReadmeRow(
        match_date=m.group("match_date"),
        team_type=m.group("team_type").strip().lower(),
        competition=comp_norm,
        gender=gender_norm,
        match_id=m.group("match_id").strip(),
        team1=t1,
        team2=t2,
    )


def strip_utf8_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def parse_cricsheet_readme(
    readme_path: Path | str,
    *,
    competition: Optional[str] = "IPL",
    genders: Optional[set[str]] = None,
) -> list[CricsheetReadmeRow]:
    """
    Read ``readme_path`` and return all valid match rows (skips header prose automatically).

    Encoding: UTF-8 with replacement on errors; strips a leading BOM if present.
    """
    p = Path(readme_path)
    text = strip_utf8_bom(p.read_text(encoding="utf-8", errors="replace"))
    rows: list[CricsheetReadmeRow] = []
    for line in text.splitlines():
        row = parse_cricsheet_readme_line(line, competition=competition, genders=genders)
        if row is not None:
            rows.append(row)
    return rows


def extract_match_index_rows(
    readme_path: Path | str,
    *,
    competition: Optional[str] = "IPL",
    genders: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Same as ``parse_cricsheet_readme``, but each row is a JSON-friendly dict."""
    return [r.as_dict() for r in parse_cricsheet_readme(readme_path, competition=competition, genders=genders)]


def resolve_readme_path(candidates: tuple[Path, ...] | list[Path] | None = None) -> Optional[Path]:
    """Return the first existing readme path (defaults to ``config.CRICSHEET_README_CANDIDATES``)."""
    from config import CRICSHEET_README_CANDIDATES

    seq = list(candidates) if candidates is not None else list(CRICSHEET_README_CANDIDATES)
    for c in seq:
        if Path(c).is_file():
            return Path(c)
    return None


def load_readme_rows(
    readme_path: Optional[Path | str] = None,
    *,
    competition: Optional[str] = "IPL",
    genders: Optional[set[str]] = None,
) -> list[CricsheetReadmeRow]:
    """
    Parse the default readme (``data/readme.txt`` or next candidate) if ``readme_path`` is omitted.
    Returns an empty list when no file exists.
    """
    p = Path(readme_path) if readme_path is not None else resolve_readme_path()
    if p is None:
        return []
    return parse_cricsheet_readme(p, competition=competition, genders=genders)


# --- Season filtering -----------------------------------------------------------

def season_years_window(current_season_year: int, n_seasons: int) -> set[int]:
    """Calendar years from ``current_season_year`` back through ``n_seasons`` seasons inclusive."""
    n = max(1, int(n_seasons))
    y = int(current_season_year)
    return {y - i for i in range(n)}


def row_season_year(row: CricsheetReadmeRow | dict[str, Any]) -> int:
    """Calendar year from ``match_date`` (``YYYY-MM-DD``)."""
    d = row["match_date"] if isinstance(row, dict) else row.match_date
    s = str(d).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    raise ValueError(f"Bad match_date for season: {d!r}")


def filter_rows_by_seasons(
    rows: Iterable[CricsheetReadmeRow],
    allowed_years: set[int],
) -> list[CricsheetReadmeRow]:
    """Keep rows whose ``match_date`` year is in ``allowed_years``."""
    ys = allowed_years
    return [r for r in rows if row_season_year(r) in ys]


def filter_last_n_seasons(
    rows: Iterable[CricsheetReadmeRow],
    *,
    current_season_year: int,
    n_seasons: int = 5,
) -> list[CricsheetReadmeRow]:
    """
    Keep rows in the last ``n_seasons`` calendar years ending at ``current_season_year``.

    Example: ``current_season_year=2026``, ``n_seasons=5`` → 2026, 2025, 2024, 2023, 2022.
    """
    window = season_years_window(current_season_year, n_seasons)
    return filter_rows_by_seasons(rows, window)


# --- Team name filtering --------------------------------------------------------

def _norm_team_token(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def row_involves_team_name(
    row: CricsheetReadmeRow,
    team_query: str,
    *,
    canonical: bool = True,
) -> bool:
    """
    True if either side equals ``team_query`` after optional IPL canonicalization.

    When ``canonical`` is True, uses ``ipl_teams.canonical_franchise_label_from_history_name``
    for readme strings and the query (so "Royal Challengers Bangalore" matches Bengaluru).
    """
    q = (team_query or "").strip()
    if not q:
        return False

    if canonical:
        import ipl_teams

        q_label = ipl_teams.canonical_franchise_label_from_history_name(q)
        if q_label is None:
            q_label = ipl_teams.canonical_franchise_label(q) or q
        q_key = _norm_team_token(q_label)

        for side in (row.team1, row.team2):
            lab = ipl_teams.canonical_franchise_label_from_history_name(side)
            if lab is None:
                lab = ipl_teams.canonical_franchise_label(side) or side
            if _norm_team_token(lab) == q_key:
                return True
        return False

    q_key = _norm_team_token(q)
    return _norm_team_token(row.team1) == q_key or _norm_team_token(row.team2) == q_key


def filter_rows_by_team_name(
    rows: Iterable[CricsheetReadmeRow],
    team_name: str,
    *,
    canonical: bool = True,
) -> list[CricsheetReadmeRow]:
    """Keep rows where ``team_name`` appears as team1 or team2."""
    return [r for r in rows if row_involves_team_name(r, team_name, canonical=canonical)]


def filter_rows_by_any_team_name(
    rows: Iterable[CricsheetReadmeRow],
    team_names: Iterable[str],
    *,
    canonical: bool = True,
) -> list[CricsheetReadmeRow]:
    """Keep rows involving **any** of the given franchise / team strings (union, not H2H-only)."""
    names = [n for n in team_names if str(n).strip()]
    if not names:
        return []
    out: list[CricsheetReadmeRow] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for r in rows:
        if any(row_involves_team_name(r, n, canonical=canonical) for n in names):
            key = (r.match_date, r.match_id, r.team1, r.team2, r.competition, r.gender)
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def rows_involving_franchises(
    rows: list[CricsheetReadmeRow],
    franchise_labels: list[str],
) -> list[CricsheetReadmeRow]:
    """
    Keep rows where either side maps to one of the canonical franchise labels.

    Delegates to ``filter_rows_by_any_team_name`` with ``canonical=True``.
    """
    return filter_rows_by_any_team_name(rows, franchise_labels, canonical=True)


__all__ = [
    "CricsheetReadmeRow",
    "extract_match_index_rows",
    "filter_last_n_seasons",
    "filter_rows_by_any_team_name",
    "filter_rows_by_seasons",
    "filter_rows_by_team_name",
    "load_readme_rows",
    "parse_cricsheet_readme",
    "parse_cricsheet_readme_line",
    "resolve_readme_path",
    "row_involves_team_name",
    "row_season_year",
    "rows_involving_franchises",
    "season_years_window",
    "strip_utf8_bom",
]
