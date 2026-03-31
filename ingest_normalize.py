"""
Canonical name normalization for **ingest** only (Cricsheet → SQLite).

Prediction and other stages should read SQLite, not re-parse JSON. All Cricsheet JSON
conversion should route team/player strings through this module so keys stay consistent.
"""

from __future__ import annotations

from typing import Literal

import ipl_teams
import learner


def normalize_team_display_for_ingest(raw: str) -> str:
    """IPL franchise storage label (aligned with ``ipl_teams.franchise_label_for_storage``)."""
    s = (raw or "").strip()
    if not s:
        return ""
    return ipl_teams.franchise_label_for_storage(s) or s


def normalize_team_key_for_ingest(display_label: str) -> str:
    """Canonical team key for SQLite ``team_key`` columns (max 80 chars)."""
    return ipl_teams.canonical_team_key_for_franchise(display_label)[:80]


def normalize_player_display_for_ingest(raw: str) -> str:
    """Trimmed display name as stored in ``player_name`` columns."""
    return str(raw or "").strip()


def normalize_player_key_for_ingest(raw: str) -> str:
    """Canonical player key (aligned with ``learner.normalize_player_key``)."""
    return learner.normalize_player_key(raw)


def normalize_for_ingest_identity(
    name: str,
    *,
    entity: Literal["player", "team"],
) -> tuple[str, str]:
    """
    Shared ingest normalization: returns ``(display_form, canonical_key)``.

    - **team**: display is the franchise storage label; key is ``team_key``.
    - **player**: display is trimmed; key is normalized ``player_key``.
    """
    if entity == "player":
        d = normalize_player_display_for_ingest(name)
        return d, normalize_player_key_for_ingest(d)
    d = normalize_team_display_for_ingest(name)
    return d, normalize_team_key_for_ingest(d)


__all__ = [
    "normalize_for_ingest_identity",
    "normalize_player_display_for_ingest",
    "normalize_player_key_for_ingest",
    "normalize_team_display_for_ingest",
    "normalize_team_key_for_ingest",
]
