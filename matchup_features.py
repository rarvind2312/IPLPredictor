"""
Recompute per-player, per-franchise matchup features from Cricsheet-backed SQLite tables
(``player_match_stats``, ``player_phase_usage``).

Phase buckets (Cricsheet ``over`` is 0-based; over 0 == 1st over):
  - powerplay: overs 1–6  → over indices 0–5
  - middle:    overs 7–15 → over indices 6–14
  - death:     overs 16–20 → over indices 15–19
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _innings_aggressor_score(
    runs: Optional[int],
    balls: Optional[int],
    fours: Optional[int],
    sixes: Optional[int],
    strike_rate: Optional[float],
) -> Optional[float]:
    """Single-innings batting aggressor in ~[0, 1] from strike rate + boundary rate + runs."""
    r = int(runs or 0)
    b = int(balls or 0)
    if b <= 0 and r <= 0:
        return None
    if b <= 0:
        b = max(1, r // 2 + 1)
    f = int(fours or 0)
    s = int(sixes or 0)
    sr = float(strike_rate) if strike_rate is not None else 100.0 * r / max(1, b)
    sr_n = _clamp01((sr - 98.0) / 95.0)
    b_pct = _clamp01((4 * f + 6 * s) / float(max(1, b)))
    run_p = _clamp01(r / 72.0)
    return _clamp01(0.40 * sr_n + 0.38 * b_pct + 0.22 * run_p)


def _innings_bowling_control_score(
    overs: Optional[float],
    wickets: Optional[int],
    runs_conceded: Optional[int],
    economy: Optional[float],
) -> Optional[float]:
    """Single-match bowling control ~[0, 1] from economy and wickets (T20 scale)."""
    ov = float(overs or 0)
    if ov <= 0:
        return None
    wk = int(wickets or 0)
    rc = int(runs_conceded or 0)
    econ = float(economy) if economy is not None else (6.0 * rc / max(1.0, ov * 6.0))
    econ_n = _clamp01((9.2 - econ) / 5.5)
    wk_n = _clamp01(wk / 4.0)
    return _clamp01(0.58 * econ_n + 0.42 * wk_n)


def _match_year_filter_sql(min_season_year: Optional[int], alias: str = "m") -> tuple[str, list[Any]]:
    """Restrict to calendar years ``>= min_season_year`` using ``YYYY`` prefix of ``match_date``."""
    if min_season_year is None:
        return "", []
    ys = f"{int(min_season_year):04d}"
    return (
        f" AND ({alias}.match_date IS NOT NULL AND length(trim({alias}.match_date)) >= 4 "
        f"AND substr(trim({alias}.match_date), 1, 4) >= ?)",
        [ys],
    )


def _ema_oldest_first(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    e = float(values[0])
    a = float(alpha)
    for x in values[1:]:
        e = a * float(x) + (1.0 - a) * e
    return float(e)


def refresh_franchise_features(
    conn: sqlite3.Connection,
    franchise_team_key: str,
    *,
    min_season_year: Optional[int] = None,
) -> int:
    """
    Rebuild ``player_franchise_features`` for every player seen in ``player_match_stats``
    for ``franchise_team_key``.

    When ``min_season_year`` is set, only rows whose ``matches.match_date`` year is at
    least that value are used (Stage 2 derive window).

    Returns number of player rows upserted.
    """
    fk = (franchise_team_key or "").strip()[:80]
    if not fk:
        return 0

    ysql, ypar = _match_year_filter_sql(min_season_year, "m")
    pkeys = conn.execute(
        f"""
        SELECT DISTINCT s.player_key
        FROM player_match_stats s
        JOIN matches m ON m.id = s.match_id
        WHERE s.team_key = ?{ysql}
        """,
        (fk, *ypar),
    ).fetchall()
    now = time.time()
    alpha = float(getattr(config, "HISTORY_BAT_SLOT_EMA_ALPHA", 0.42))
    lim = int(getattr(config, "MATCHUP_FEATURE_MAX_INNINGS_ROWS", 90))
    lim_m = int(getattr(config, "MATCHUP_FEATURE_MAX_MATCHES_PHASE", 45))

    n_upsert = 0
    for (pk,) in pkeys:
        pk_s = str(pk or "").strip()[:80]
        if not pk_s:
            continue

        rows = conn.execute(
            f"""
            SELECT s.batting_position, s.runs, s.balls, s.fours, s.sixes, s.strike_rate,
                   s.overs_bowled, s.wickets, s.runs_conceded, s.economy,
                   s.vs_spin_balls_faced, s.vs_pace_balls_faced,
                   m.match_date, m.id AS mid
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.player_key = ? AND s.team_key = ?{ysql}
            ORDER BY m.match_date ASC NULLS FIRST, m.id ASC
            LIMIT ?
            """,
            (pk_s, fk, *ypar, lim),
        ).fetchall()

        # Batting-slot EMA: use the most recent innings that have a stored position (ball-by-ball
        # extract), chronological within that window so forward-EMA weights recent slots more.
        pos_rows = conn.execute(
            f"""
            SELECT s.batting_position
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.player_key = ? AND s.team_key = ?
              AND s.batting_position IS NOT NULL{ysql}
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
            """,
            (pk_s, fk, *ypar, lim),
        ).fetchall()

        positions: list[float] = []
        for r in reversed(pos_rows):
            bp = r["batting_position"]
            try:
                positions.append(float(bp))
            except (TypeError, ValueError):
                continue

        batting_pos_ema = _ema_oldest_first(positions, alpha) if positions else None
        tail = [round(x, 3) for x in positions[-30:]]

        agg_scores: list[float] = []
        ctrl_scores: list[float] = []
        spin_b = pace_b = 0
        for r in rows:
            ag = _innings_aggressor_score(
                r["runs"], r["balls"], r["fours"], r["sixes"], r["strike_rate"]
            )
            if ag is not None:
                agg_scores.append(ag)
            cg = _innings_bowling_control_score(
                r["overs_bowled"], r["wickets"], r["runs_conceded"], r["economy"]
            )
            if cg is not None:
                ctrl_scores.append(cg)
            spin_b += int(r["vs_spin_balls_faced"] or 0)
            pace_b += int(r["vs_pace_balls_faced"] or 0)

        batting_aggr = sum(agg_scores) / len(agg_scores) if agg_scores else 0.0
        bowl_ctrl = sum(ctrl_scores) / len(ctrl_scores) if ctrl_scores else 0.0
        tagged = spin_b + pace_b
        vs_spin_t = (spin_b / tagged) if tagged > 0 else 0.0
        vs_pace_t = (pace_b / tagged) if tagged > 0 else 0.0

        mids_rows = conn.execute(
            f"""
            SELECT DISTINCT s.match_id
            FROM player_match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE s.player_key = ? AND s.team_key = ?{ysql}
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
            """,
            (pk_s, fk, *ypar, lim_m),
        ).fetchall()
        mids = [int(r["match_id"]) for r in mids_rows]
        pp_b = mid_b = death_b = 0
        pp_m = mid_m = death_m = 0
        if mids:
            qm = ",".join("?" * len(mids))
            ph = conn.execute(
                f"""
                SELECT phase, match_id, balls
                FROM player_phase_usage
                WHERE player_key = ? AND team_key = ? AND role = 'bowl'
                  AND match_id IN ({qm})
                """,
                [pk_s, fk] + mids,
            ).fetchall()
            balls_by_mid_phase: dict[tuple[int, str], int] = {}
            for r in ph:
                key = (int(r["match_id"]), str(r["phase"] or ""))
                balls_by_mid_phase[key] = balls_by_mid_phase.get(key, 0) + int(r["balls"] or 0)
            for mid in mids:
                b_pp = balls_by_mid_phase.get((mid, "powerplay"), 0)
                b_md = balls_by_mid_phase.get((mid, "middle"), 0)
                b_dt = balls_by_mid_phase.get((mid, "death"), 0)
                pp_b += b_pp
                mid_b += b_md
                death_b += b_dt
                if b_pp > 0:
                    pp_m += 1
                if b_md > 0:
                    mid_m += 1
                if b_dt > 0:
                    death_m += 1

        total_bowl_balls = pp_b + mid_b + death_b
        pp_share = (pp_b / total_bowl_balls) if total_bowl_balls > 0 else 0.0
        mid_share = (mid_b / total_bowl_balls) if total_bowl_balls > 0 else 0.0
        death_share = (death_b / total_bowl_balls) if total_bowl_balls > 0 else 0.0

        nm = max(1, len(mids))
        rate_pp = pp_m / float(nm)
        rate_mid = mid_m / float(nm)
        rate_death = death_m / float(nm)

        conn.execute(
            """
            INSERT INTO player_franchise_features (
                player_key, franchise_team_key,
                batting_position_ema, batting_slot_samples, batting_positions_tail_json,
                pp_overs_bowled, middle_overs_bowled, death_overs_bowled,
                pp_bowl_ball_share, middle_bowl_ball_share, death_bowl_ball_share,
                phase_bowl_rate_pp, phase_bowl_rate_middle, phase_bowl_rate_death,
                vs_spin_balls, vs_pace_balls,
                vs_spin_tendency, vs_pace_tendency,
                batting_aggressor_score, bowling_control_score,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_key, franchise_team_key) DO UPDATE SET
                batting_position_ema = excluded.batting_position_ema,
                batting_slot_samples = excluded.batting_slot_samples,
                batting_positions_tail_json = excluded.batting_positions_tail_json,
                pp_overs_bowled = excluded.pp_overs_bowled,
                middle_overs_bowled = excluded.middle_overs_bowled,
                death_overs_bowled = excluded.death_overs_bowled,
                pp_bowl_ball_share = excluded.pp_bowl_ball_share,
                middle_bowl_ball_share = excluded.middle_bowl_ball_share,
                death_bowl_ball_share = excluded.death_bowl_ball_share,
                phase_bowl_rate_pp = excluded.phase_bowl_rate_pp,
                phase_bowl_rate_middle = excluded.phase_bowl_rate_middle,
                phase_bowl_rate_death = excluded.phase_bowl_rate_death,
                vs_spin_balls = excluded.vs_spin_balls,
                vs_pace_balls = excluded.vs_pace_balls,
                vs_spin_tendency = excluded.vs_spin_tendency,
                vs_pace_tendency = excluded.vs_pace_tendency,
                batting_aggressor_score = excluded.batting_aggressor_score,
                bowling_control_score = excluded.bowling_control_score,
                last_updated = excluded.last_updated
            """,
            (
                pk_s,
                fk,
                batting_pos_ema,
                len(positions),
                json.dumps(tail, ensure_ascii=False),
                round(pp_b / 6.0, 4),
                round(mid_b / 6.0, 4),
                round(death_b / 6.0, 4),
                round(pp_share, 5),
                round(mid_share, 5),
                round(death_share, 5),
                round(rate_pp, 5),
                round(rate_mid, 5),
                round(rate_death, 5),
                spin_b,
                pace_b,
                round(vs_spin_t, 5),
                round(vs_pace_t, 5),
                round(batting_aggr, 5),
                round(bowl_ctrl, 5),
                now,
            ),
        )
        n_upsert += 1

    logger.info("matchup_features: refreshed franchise_key=%s players=%d", fk, n_upsert)
    return n_upsert


def refresh_franchise_features_for_team_keys(
    conn: sqlite3.Connection,
    team_keys: set[str],
    *,
    min_season_year: Optional[int] = None,
) -> int:
    total = 0
    for tk in sorted(team_keys):
        t = (tk or "").strip()[:80]
        if t:
            total += refresh_franchise_features(conn, t, min_season_year=min_season_year)
    return total
