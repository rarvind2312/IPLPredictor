"""Official IPL team slugs (internal) and display labels for the UI."""

from __future__ import annotations

from typing import Optional

import canonical_keys
import learner

# (slug, display_label) — order preserved for dropdowns
IPL_TEAMS: list[tuple[str, str]] = [
    ("chennai-super-kings", "Chennai Super Kings"),
    ("mumbai-indians", "Mumbai Indians"),
    ("kolkata-knight-riders", "Kolkata Knight Riders"),
    ("royal-challengers-bengaluru", "Royal Challengers Bengaluru"),
    ("sunrisers-hyderabad", "Sunrisers Hyderabad"),
    ("rajasthan-royals", "Rajasthan Royals"),
    ("gujarat-titans", "Gujarat Titans"),
    ("delhi-capitals", "Delhi Capitals"),
    ("lucknow-super-giants", "Lucknow Super Giants"),
    ("punjab-kings", "Punjab Kings"),
]

SLUG_TO_LABEL: dict[str, str] = {s: l for s, l in IPL_TEAMS}
TEAM_SLUGS: list[str] = [s for s, _ in IPL_TEAMS]

# Cricsheet / archive names that differ from current IPL branding (normalized keys).
_HISTORY_FRANCHISE_ALIASES: dict[str, str] = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
}


def label_for_slug(slug: str) -> str:
    return SLUG_TO_LABEL.get(slug, slug.replace("-", " ").title())


def canonical_franchise_label_from_history_name(team_input: str) -> Optional[str]:
    """
    Map an archive/readme team string to the current IPL display label.

    Handles renames (e.g. Royal Challengers Bangalore → Bengaluru) so Cricsheet rows
    join to the same franchise as the official squad list.
    """
    base = canonical_franchise_label(team_input)
    if base:
        return base
    k = learner.normalize_player_key(team_input or "")
    if not k:
        return None
    return _HISTORY_FRANCHISE_ALIASES.get(k)


def franchise_label_for_storage(team_input: str) -> str:
    """
    Single display label for SQLite ``team_name`` / history joins.

    Resolves archive names (readme / Cricsheet ``info.teams``) to the same official
    label as the UI squad dropdown so ``team_key`` and ``canonical_franchise_label()``
    lookups stay aligned.
    """
    s = (team_input or "").strip()
    if not s:
        return ""
    resolved = canonical_franchise_label_from_history_name(s)
    if resolved:
        return resolved
    return canonical_franchise_label(s) or s


def canonical_franchise_label(team_input: str) -> Optional[str]:
    """
    Resolve arbitrary text to the official IPL display label for this franchise.

    Uses **exact normalized equality** only (no substring fuzzy match between franchises).
    Returns None if no known IPL team matches.
    """
    ti = learner.normalize_player_key(team_input or "")
    if not ti:
        return None
    for slug, label in IPL_TEAMS:
        nl = learner.normalize_player_key(label)
        if nl and ti == nl:
            return label
        ns = learner.normalize_player_key(slug.replace("-", " "))
        if ns and ti == ns:
            return label
    return None


def canonical_franchise_label_or_raise(team_input: str) -> str:
    lab = canonical_franchise_label(team_input)
    if not lab:
        raise ValueError(
            f"Unknown IPL franchise {team_input!r}; use an official team name from the IPL list."
        )
    return lab


def canonical_team_key_for_franchise(canonical_label: str) -> str:
    """Single normalized key used for history storage lookups (first 80 chars)."""
    lab = franchise_label_for_storage(canonical_label) or (canonical_label or "").strip()
    return canonical_keys.canonical_team_key(lab)[:80]


def franchise_row_matches_canonical(
    *,
    stored_team_name: str,
    stored_team_key: str,
    canonical_label: str,
) -> bool:
    """True if a ``team_match_xi`` / ``player_batting_positions`` row belongs to this franchise."""
    ck = canonical_team_key_for_franchise(canonical_label)
    label_l = (canonical_label or "").strip().lower()
    rtk = (stored_team_key or "").strip()
    rtn = (stored_team_name or "").strip()
    if rtk == ck:
        return True
    if rtn.lower() == label_l:
        return True
    resolved_name = franchise_label_for_storage(rtn)
    if resolved_name.lower() == label_l:
        return True
    if canonical_team_key_for_franchise(resolved_name) == ck:
        return True
    return False


def slug_for_canonical_label(canonical_label: str) -> Optional[str]:
    """Official IPLT20 path slug for a resolved franchise display name."""
    lab = (canonical_label or "").strip()
    if not lab:
        return None
    for slug, label in IPL_TEAMS:
        if label == lab:
            return slug
    return None
