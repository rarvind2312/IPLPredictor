"""Shared normalization helpers (URLs, match identity) for sync, DB, and parsers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import canonical_keys

# Must match ``learner._PLAYER_KEY_MAX_LEN`` (cannot import learner: db → utils → learner → db).
_PLAYER_KEY_MAX_LEN = 80


def read_json_utf8(path: Path) -> Optional[Any]:
    """
    Read ``path`` as UTF-8 JSON. Returns ``None`` if the path is not a file, is unreadable,
    or does not contain valid JSON.

    Callers that need logging or type checks should handle ``None`` the same way they
    previously handled ``json.loads`` failures.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — mirror prior broad json.loads guards
        return None


def _norm_player_key(name: str) -> str:
    """Same identity as ``learner.normalize_player_key`` (canonical + truncation)."""
    s = canonical_keys.canonical_player_key(name or "")
    return s[:_PLAYER_KEY_MAX_LEN] if s else ""


def normalize_match_date_prefix(match_date: Optional[str]) -> str:
    """Stable date prefix for cross-source dedupe (prefer YYYY-MM-DD)."""
    if match_date is None:
        return ""
    s = str(match_date).strip()
    if not s:
        return ""
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    m = re.match(r"^(\d{4})", s)
    return m.group(1) if m else ""


def canonical_match_identity_key(
    team_a: Optional[str],
    team_b: Optional[str],
    match_date: Optional[str],
) -> str:
    """
    Order-insensitive identity for a completed fixture (IPL-only usage).

    Same teams + same calendar date (or year-only fallback) → same key.
    """
    ta = _norm_player_key(team_a or "")
    tb = _norm_player_key(team_b or "")
    if not ta or not tb:
        return ""
    t1, t2 = sorted([ta, tb])
    d = normalize_match_date_prefix(match_date)
    return f"{t1}|{t2}|{d}"


def normalize_scorecard_url(url: str) -> str:
    """Canonical absolute HTTPS URL for scorecard fetch + dedupe."""
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#")[0].split("?")[0].strip()
    if u.startswith("//"):
        u = "https:" + u
    if not u.startswith("http"):
        path = u if u.startswith("/") else "/" + u.lstrip("/")
        u = "https://www.iplt20.com" + path
    u = u.replace("http://", "https://")
    u = re.sub(r"^https://iplt20\.com", "https://www.iplt20.com", u, flags=re.I)
    u = re.sub(
        r"^https://www\.espncricinfo\.com",
        "https://www.espncricinfo.com",
        u,
        flags=re.I,
    )
    return u.rstrip("/")
