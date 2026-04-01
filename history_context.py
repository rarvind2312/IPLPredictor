"""Build normalized lookup tables for rule-based history signals (uses learner.normalize)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import config
import db
import learner


@dataclass
class HistoryContext:
    db_match_count: int = 0
    max_xi_picks: int = 0
    xi_by_player: dict[str, int] = field(default_factory=dict)
    avg_slot_by_player: dict[str, tuple[float, int]] = field(default_factory=dict)
    bowl_balls_avg_by_player: dict[str, tuple[float, int]] = field(default_factory=dict)
    # (venue_key, team_key, player_key) -> times in XI at that ground for that side label
    venue_team_player_xi: dict[tuple[str, str, str], int] = field(default_factory=dict)
    venue_team_matches: dict[tuple[str, str], int] = field(default_factory=dict)
    night_xi_by_player: dict[str, int] = field(default_factory=dict)
    day_xi_by_player: dict[str, int] = field(default_factory=dict)
    overseas_mix: dict[tuple[str, str], dict[int, int]] = field(default_factory=dict)
    chase_share_by_venue: dict[str, tuple[float, int]] = field(default_factory=dict)

    def pick_overseas_mix(self, venue_keys: list[str], team_key: str) -> dict[int, int]:
        for vk in venue_keys:
            mix = self.overseas_mix.get((vk, team_key))
            if mix and sum(mix.values()) >= config.LEARN_MIN_SAMPLES_OVERSEAS_MIX // 2:
                return mix
        return {}


def _norm_venue_team_counts() -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str], int]]:
    """Per-player XI counts at (venue, team); distinct matches per (venue, team)."""
    raw_xt = db.venue_team_xi_raw()
    vtp: dict[tuple[str, str, str], int] = {}
    match_sets: dict[tuple[str, str], set[int]] = {}

    for venue_raw, team_raw, player_name, c in raw_xt:
        vk = learner.normalize_player_key(str(venue_raw))[:80]
        tk = learner.normalize_player_key(str(team_raw))[:80]
        pk = learner.normalize_player_key(str(player_name))
        if not vk or not tk or not pk:
            continue
        key3 = (vk, tk, pk)
        vtp[key3] = vtp.get(key3, 0) + int(c)

    for row in db.match_xi_team_venue_rows():
        mid, venue_raw, team_raw = row
        vk = learner.normalize_player_key(str(venue_raw))[:80]
        tk = learner.normalize_player_key(str(team_raw))[:80]
        if not vk or not tk:
            continue
        match_sets.setdefault((vk, tk), set()).add(int(mid))

    vtm = {k: len(s) for k, s in match_sets.items()}
    return vtp, vtm


def _overseas_mix_all() -> dict[tuple[str, str], dict[int, int]]:
    out: dict[tuple[str, str], dict[int, int]] = {}
    for row in db.learned_overseas_mix_raw():
        vk, tk, n, t = row
        out.setdefault((vk, tk), {})[int(n)] = int(t)
    return out


def _chase_by_venue() -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for row in db.learned_venue_team_chase_rollup():
        vk, bat_w, bowl_w = row
        bt = int(bat_w or 0)
        bw = int(bowl_w or 0)
        tot = bt + bw
        out[vk] = (bw / max(1, tot), tot)
    return out


def build_history_context() -> HistoryContext:
    return _build_history_context_cached(db.db_runtime_signature())


@lru_cache(maxsize=4)
def _build_history_context_cached(_sig: tuple[str, int, int, int]) -> HistoryContext:
    ctx = HistoryContext()
    ctx.db_match_count = db.count_stored_matches()
    for name, c in db.xi_pick_counts_raw():
        k = learner.normalize_player_key(name)
        if k:
            ctx.xi_by_player[k] = int(c)
    vals = list(ctx.xi_by_player.values())
    ctx.max_xi_picks = max(vals) if vals else 0
    if ctx.max_xi_picks == 0:
        ctx.max_xi_picks = db.max_xi_pick_count()

    for name, av, n in db.avg_batting_position_raw():
        k = learner.normalize_player_key(name)
        if k:
            ctx.avg_slot_by_player[k] = (float(av), int(n))

    for name, av, n in db.bowling_usage_raw():
        k = learner.normalize_player_key(name)
        if k:
            ctx.bowl_balls_avg_by_player[k] = (float(av), int(n))

    vtp, vtm = _norm_venue_team_counts()
    ctx.venue_team_player_xi = vtp
    ctx.venue_team_matches = vtm

    for name, is_night, c in db.night_day_xi_raw():
        k = learner.normalize_player_key(name)
        if not k:
            continue
        if int(is_night) == 1:
            ctx.night_xi_by_player[k] = ctx.night_xi_by_player.get(k, 0) + int(c)
        else:
            ctx.day_xi_by_player[k] = ctx.day_xi_by_player.get(k, 0) + int(c)

    ctx.overseas_mix = _overseas_mix_all()
    ctx.chase_share_by_venue = _chase_by_venue()
    return ctx


def venue_lookup_keys(venue_profile: Any) -> list[str]:
    """Try multiple normalized venue keys (DB ingest strings may differ)."""
    keys: list[str] = []
    for attr in ("key", "display_name", "city"):
        v = getattr(venue_profile, attr, None)
        if v and str(v).strip():
            nk = learner.normalize_player_key(str(v))[:80]
            if nk and nk not in keys:
                keys.append(nk)
    return keys
