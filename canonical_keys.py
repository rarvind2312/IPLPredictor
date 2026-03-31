"""
Single source of truth for canonical player and team identity strings used in SQLite and lookups.

Rules (player / team names):
- Unicode normalize (NFKC), strip combining marks
- Unify common apostrophe / quote / dash characters
- ASCII hyphen and unicode dashes → space
- Lowercase
- Remove punctuation (non-word, non-space) → space
- Collapse repeated spaces and trim

Examples:
  "MS Dhoni" → "ms dhoni"
  "Ruturaj Gaikwad" → "ruturaj gaikwad"
  "Lhuan-dre Pretorius" → "lhuan dre pretorius"
  "Chennai Super Kings" → "chennai super kings"
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Unicode dashes / minus → treat like hyphen (then become space)
_DASH_CHARS: Final = frozenset(
    "\u2010\u2011\u2012\u2013\u2014\u2212\u2015\uFE58\uFE63\uFF0D"
)

# Map fancy quotes to ASCII (then stripped or spaced via punct rule)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00b4": "'",
        "\u0060": "'",
    }
)


def canonical_player_key(name: str) -> str:
    """
    Normalize a person name to a single canonical key (before optional DB truncation).

    Initials and dotted forms become space-separated tokens (e.g. "M. S. Dhoni" → "m s dhoni").
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name).strip())
    s = s.translate(_QUOTE_TRANSLATION)
    s = s.replace("-", " ")
    for ch in _DASH_CHARS:
        s = s.replace(ch, " ")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Word chars + spaces only; everything else → space (removes punctuation)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_team_key(team_name: str) -> str:
    """
    Normalize a franchise / team display string the same way as player names.

    "Chennai Super Kings" → "chennai super kings"
    """
    return canonical_player_key(team_name)


__all__ = ["canonical_player_key", "canonical_team_key"]
