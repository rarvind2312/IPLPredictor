"""
Update and read learned signals from stored matches (SQLite).

Actual playing XIs, batting order, and per-match context are also materialized into
`matches`, `team_match_xi`, and `team_match_summary` inside `db.insert_parsed_match`
(see `db._sync_history_match_tables`). XI prediction consumes those tables via
`history_xi` — this module focuses on player/venue aggregates (`learned_player`, etc.).

`ingest_payload` should run **once per new** stored match. Duplicate URL or canonical fixture
skips learning when `insert_parsed_match` reports ``duplicate_url`` or ``duplicate_match``.
Cricsheet-derived rows reach SQLite via the **ingest** stage (e.g. ``cricsheet_ingest``), not during prediction.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Optional

import canonical_keys
import config
import db

# SQLite / join keys are capped for column width and index size.
_PLAYER_KEY_MAX_LEN = 80


def normalize_player_key(name: str) -> str:
    """Canonical identity for a player; aligned with ``canonical_keys.canonical_player_key`` (truncated)."""
    s = canonical_keys.canonical_player_key(name or "")
    return s[:_PLAYER_KEY_MAX_LEN] if s else ""


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _batting_impact(runs: int, balls: int) -> float:
    if balls <= 0 and runs <= 0:
        return 0.45
    if balls <= 0:
        balls = max(1, runs // 2 + 1)
    sr = runs / max(1, balls) * 100.0
    # Normalize strike rate to 0–1-ish for T20
    base = 0.35 + min(0.55, max(0.0, (sr - 95) / 95.0))
    boundary_boost = min(0.15, runs / 120.0)
    return max(0.05, min(0.98, base + boundary_boost))


def _bowling_impact(wickets: int, runs_conceded: int, balls_bowled_est: int) -> float:
    if balls_bowled_est <= 0:
        balls_bowled_est = max(6, wickets * 12)
    econ = runs_conceded / max(1, balls_bowled_est) * 6.0
    wk = min(5, wickets)
    base = 0.35 + wk * 0.09
    econ_adj = max(0.0, min(0.25, (9.5 - econ) / 18.0))
    return max(0.05, min(0.98, base + econ_adj))


def _merge_impact(bat: float, bowl: float, had_bat: bool, had_bowl: bool) -> float:
    if had_bat and had_bowl:
        return 0.5 * bat + 0.5 * bowl
    if had_bat:
        return bat
    if had_bowl:
        return bowl
    return 0.5


def ingest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Apply learning rules to a parsed scorecard payload (already structured).
    Updates learned_player and learned_venue_team aggregates.
    """
    meta = payload.get("meta") or {}
    teams = list(payload.get("teams") or [])
    winner = (meta.get("winner") or "").strip()
    batting_first = (meta.get("batting_first") or "").strip()
    venue_raw = (meta.get("venue") or "").strip()

    # Aggregate per player from batting + bowling tables
    per_key: dict[str, dict[str, Any]] = {}

    def bump(key: str) -> dict[str, Any]:
        if key not in per_key:
            per_key[key] = {
                "runs": 0,
                "balls": 0,
                "wickets": 0,
                "runs_conc": 0,
                "balls_bowled_est": 0,
                "in_xi": False,
            }
        return per_key[key]

    for side in payload.get("playing_xi") or []:
        t = side.get("team") or ""
        for p in side.get("players") or []:
            k = normalize_player_key(str(p))
            bump(k)["in_xi"] = True

    for inn in payload.get("batting") or []:
        for row in inn.get("rows") or []:
            k = normalize_player_key(str(row.get("player") or ""))
            o = bump(k)
            o["runs"] += _safe_int(row.get("runs"))
            o["balls"] += _safe_int(row.get("balls"))

    for inn in payload.get("bowling") or []:
        for row in inn.get("rows") or []:
            k = normalize_player_key(str(row.get("player") or ""))
            o = bump(k)
            w = _safe_int(row.get("wickets"))
            rc = _safe_int(row.get("runs"))
            overs = _safe_float(row.get("overs"))
            balls_est = int(round(overs * 6)) if overs > 0 else w * 18
            o["wickets"] += w
            o["runs_conc"] += rc
            o["balls_bowled_est"] += max(balls_est, w * 6)

    existing = db.get_learned_players()
    alpha = config.LEARN_EMA_ALPHA
    updates: list[tuple[str, dict[str, Any]]] = []

    for key, st in per_key.items():
        bat_i = _batting_impact(st["runs"], st["balls"])
        bowl_i = _bowling_impact(st["wickets"], st["runs_conc"], st["balls_bowled_est"])
        had_bat = st["balls"] > 0 or st["runs"] > 0
        had_bowl = st["balls_bowled_est"] > 0 or st["wickets"] > 0
        match_impact = _merge_impact(bat_i, bowl_i, had_bat, had_bowl)

        prev = existing.get(key)
        if prev:
            ema = (1 - alpha) * float(prev["impact_ema"]) + alpha * match_impact
            mdb = int(prev["matches_in_db"]) + 1
            xi_app = int(prev["xi_appearances"]) + (1 if st["in_xi"] else 0)
            br = int(prev["batting_runs"]) + st["runs"]
            bb = int(prev["batting_balls"]) + st["balls"]
            wk = int(prev["wickets"]) + st["wickets"]
            bbowl = int(prev["balls_bowled"]) + st["balls_bowled_est"]
        else:
            ema = match_impact
            mdb = 1
            xi_app = 1 if st["in_xi"] else 0
            br = st["runs"]
            bb = st["balls"]
            wk = st["wickets"]
            bbowl = st["balls_bowled_est"]

        updates.append(
            (
                key,
                {
                    "matches_in_db": mdb,
                    "xi_appearances": xi_app,
                    "batting_runs": br,
                    "batting_balls": bb,
                    "wickets": wk,
                    "balls_bowled": bbowl,
                    "impact_ema": ema,
                },
            )
        )

    if updates:
        db.upsert_learned_players(updates)

    # Venue + toss outcome learning (coarse)
    vkey = normalize_player_key(venue_raw)[:80] or "unknown_venue"
    for t in teams:
        if not t:
            continue
        tk = normalize_player_key(t)[:80]
        bat_first_win = 0
        bowl_first_win = 0
        if winner and winner.lower() == t.lower():
            if batting_first and batting_first.lower() == t.lower():
                bat_first_win = 1
            elif batting_first:
                bowl_first_win = 1
        if bat_first_win or bowl_first_win:
            db.upsert_venue_team(
                vkey,
                tk,
                bat_first_win_delta=bat_first_win,
                bowl_first_win_delta=bowl_first_win,
            )

    # Optional: overseas counts in XI (user-supplied on ingest) → venue–team mix table
    oa = meta.get("overseas_in_xi_team_a")
    ob = meta.get("overseas_in_xi_team_b")
    if len(teams) > 0 and oa is not None and str(oa).strip() != "":
        try:
            na = max(0, min(11, int(float(oa))))
            tk0 = normalize_player_key(str(teams[0]))[:80]
            if tk0:
                db.bump_overseas_mix(vkey, tk0, na, 1)
        except (TypeError, ValueError):
            pass
    if len(teams) > 1 and ob is not None and str(ob).strip() != "":
        try:
            nb = max(0, min(11, int(float(ob))))
            tk1 = normalize_player_key(str(teams[1]))[:80]
            if tk1:
                db.bump_overseas_mix(vkey, tk1, nb, 1)
        except (TypeError, ValueError):
            pass

    return {"players_updated": len(updates), "venue_key": vkey}


def load_learned_map() -> dict[str, dict[str, Any]]:
    return _load_learned_map_cached(db.db_runtime_signature())


@lru_cache(maxsize=4)
def _load_learned_map_cached(_sig: tuple[str, int, int, int]) -> dict[str, dict[str, Any]]:
    rows = db.get_learned_players()
    return {
        k: {
            "impact_ema": float(r["impact_ema"]),
            "matches_in_db": int(r["matches_in_db"]),
            "xi_appearances": int(r["xi_appearances"]),
        }
        for k, r in rows.items()
    }


def learned_boost_for_player(name: str, learned: Optional[dict[str, dict[str, Any]]] = None) -> float:
    """Returns 0–1 adjustment anchor; 0.5 = neutral if unknown."""
    lm = learned if learned is not None else load_learned_map()
    key = normalize_player_key(name)
    row = lm.get(key)
    if not row:
        return 0.5
    return float(row["impact_ema"])


def venue_toss_edge(
    venue_key: str,
    team_name: str,
    *,
    learned_rows: Optional[list] = None,
) -> dict[str, float]:
    """
    Empirical edge for team at venue when batting first vs bowling first.
    Returns prior logits contribution (small).
    """
    vk = normalize_player_key(venue_key)[:80]
    tk = normalize_player_key(team_name)[:80]
    rows = learned_rows if learned_rows is not None else db.get_venue_team_stats()
    bat_w = 1
    bowl_w = 1
    for r in rows:
        if r["venue_key"] == vk and r["team_key"] == tk:
            bat_w += int(r["wins_bat_first"])
            bowl_w += int(r["wins_bowl_first"])
    total = bat_w + bowl_w
    p_bat = bat_w / total
    # Map to small logit bias in [-0.25, 0.25]
    edge = (p_bat - 0.5) * 0.5
    return {"bat_first_logit": edge, "bowl_first_logit": -edge, "sample_bat": bat_w, "sample_bowl": bowl_w}


def rehydrate_payload_from_db(match_id: int) -> Optional[dict[str, Any]]:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT raw_payload FROM match_results WHERE id = ?", (match_id,)
        ).fetchone()
    if not row:
        return None
    return json.loads(row["raw_payload"])
