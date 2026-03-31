"""
Rebuild ``player_recent_form_cache`` from SQLite ``player_match_stats`` + ``matches``.

**Ingest-only inputs**: rows must be populated by Cricsheet JSON → ``insert_parsed_match``
(with ``match_format`` set for non-IPL archives). Prediction reads this cache only — never scans JSON.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import groupby
from typing import Any, Optional

import config
import db
import matchup_features

logger = logging.getLogger(__name__)


def _parse_iso(s: Optional[str]) -> Optional[date]:
    if not s or not str(s).strip():
        return None
    try:
        return datetime.fromisoformat(str(s).strip()[:10]).date()
    except ValueError:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _t20_sql_filter() -> tuple[str, list[Any]]:
    """SQL predicate on ``matches m`` aligned with :func:`db.match_row_is_t20_family`."""
    params: list[Any] = []
    or_parts: list[str] = []
    for s in config.T20_FAMILY_COMPETITION_SUBSTRINGS:
        or_parts.append("instr(lower(coalesce(m.competition,'')), ?) > 0")
        params.append(s)
    legacy = "(" + " OR ".join(or_parts) + ")"
    sql = (
        "(lower(trim(coalesce(m.match_format,''))) IN ('t20','t20i') "
        "OR ((trim(coalesce(m.match_format,'')) = '' OR m.match_format IS NULL) AND "
        f"{legacy}))"
    )
    return sql, params


def _group_union_matches(
    rows: list[dict[str, Any]],
    *,
    last_n: int,
    months: int,
    ref: date,
) -> list[dict[str, Any]]:
    cutoff = ref - timedelta(days=int(round(months * 30.5)))
    by_mid: dict[int, dict[str, Any]] = {}
    order_ids: list[int] = []
    for r in rows:
        mid = int(r["match_id"])
        if mid not in by_mid:
            by_mid[mid] = r
            order_ids.append(mid)
    want: set[int] = set()
    for mid in order_ids[:last_n]:
        want.add(mid)
    for r in rows:
        mid = int(r["match_id"])
        d = _parse_iso(r.get("match_date"))
        if d is not None and d >= cutoff:
            want.add(mid)
    out: list[dict[str, Any]] = []
    for mid in order_ids:
        if mid in want:
            out.append(by_mid[mid])
    return out


def _batting_form(rows: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for r in rows:
        a = matchup_features._innings_aggressor_score(
            r.get("runs"),
            r.get("balls"),
            r.get("fours"),
            r.get("sixes"),
            r.get("strike_rate"),
        )
        if a is not None:
            scores.append(float(a))
    if not scores:
        return 0.48
    return _clamp01(sum(scores) / len(scores))


def _bowling_form(rows: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for r in rows:
        c = matchup_features._innings_bowling_control_score(
            r.get("overs_bowled"),
            r.get("wickets"),
            r.get("runs_conceded"),
            r.get("economy"),
        )
        if c is not None:
            scores.append(float(c))
    if not scores:
        return 0.48
    return _clamp01(sum(scores) / len(scores))


def _phase_shares(
    conn: sqlite3.Connection, player_key: str, mids: set[int]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not mids:
        return None, None, None
    qm = ",".join("?" * len(mids))
    rows = conn.execute(
        f"""
        SELECT phase, SUM(balls) AS b FROM player_match_role_usage
        WHERE player_key = ? AND lower(role) = 'bowl' AND match_id IN ({qm})
        GROUP BY phase
        """,
        (player_key, *sorted(mids)),
    ).fetchall()
    tot = 0.0
    by_ph: dict[str, float] = defaultdict(float)
    for r in rows:
        ph = str(r["phase"] or "").lower()
        b = float(r["b"] or 0)
        by_ph[ph] += b
        tot += b
    if tot <= 0:
        return None, None, None
    return (
        _clamp01(by_ph.get("powerplay", 0.0) / tot),
        _clamp01(by_ph.get("middle", 0.0) / tot),
        _clamp01(by_ph.get("death", 0.0) / tot),
    )


def _confidence(n_matches: int) -> float:
    return _clamp01(1.0 - math.exp(-float(n_matches) / 5.5))


def _count_days_matches(
    rows_all: list[dict[str, Any]], ref: date, days: int
) -> int:
    cutoff = ref - timedelta(days=days)
    seen: set[int] = set()
    for r in rows_all:
        d = _parse_iso(r.get("match_date"))
        if d is None or d < cutoff:
            continue
        if not db.match_row_is_t20_family(r.get("competition"), r.get("match_format")):
            continue
        seen.add(int(r["match_id"]))
    return len(seen)


def rebuild_player_recent_form_cache(
    *,
    reference_iso_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """
    Full refresh of ``player_recent_form_cache`` for all players with T20-family rows.

    Uses global ``player_key`` (all teams / competitions).
    """
    import db as db_mod

    ref = _parse_iso(reference_iso_date) or date.today()
    last_n = int(getattr(config, "SELECTION_RECENT_FORM_LAST_N_MATCHES", 5))
    months = int(getattr(config, "SELECTION_RECENT_FORM_MONTHS", 5))
    t20_sql, t20_params = _t20_sql_filter()
    now = time.time()
    stats: dict[str, Any] = {
        "players_cached": 0,
        "rows_scanned": 0,
        "t20_distinct_matches": 0,
        "errors": 0,
    }

    def _process_connection(c: sqlite3.Connection) -> None:
        chk = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_recent_form_cache'"
        ).fetchone()
        if not chk:
            logger.warning("player_recent_form_cache missing — run DB migrations")
            return
        c.execute("DELETE FROM player_recent_form_cache")
        mid_rows = c.execute(
            f"""
            SELECT COUNT(DISTINCT p.match_id) FROM player_match_stats p
            JOIN matches m ON m.id = p.match_id
            WHERE m.match_date IS NOT NULL AND trim(m.match_date) != ''
              AND {t20_sql}
            """,
            t20_params,
        ).fetchone()
        stats["t20_distinct_matches"] = int(mid_rows[0] if mid_rows else 0)

        cur = c.execute(
            f"""
            SELECT p.player_key AS player_key, p.match_id AS match_id,
                   m.match_date AS match_date, m.competition AS competition,
                   m.match_format AS match_format,
                   p.runs AS runs, p.balls AS balls, p.fours AS fours, p.sixes AS sixes,
                   p.strike_rate AS strike_rate,
                   p.overs_bowled AS overs_bowled, p.wickets AS wickets,
                   p.runs_conceded AS runs_conceded, p.economy AS economy,
                   p.batting_position AS batting_position
            FROM player_match_stats p
            JOIN matches m ON m.id = p.match_id
            WHERE m.match_date IS NOT NULL AND trim(m.match_date) != ''
              AND {t20_sql}
            ORDER BY p.player_key ASC, m.match_date DESC, p.match_id DESC
            """,
            t20_params,
        )
        for pk, grp in groupby(cur, key=lambda x: str(x["player_key"] or "").strip()):
            if not pk:
                continue
            plist = [dict(r) for r in grp]
            stats["rows_scanned"] += len(plist)
            try:
                scoped = _group_union_matches(plist, last_n=last_n, months=months, ref=ref)
                if not scoped:
                    continue
                mids = {int(r["match_id"]) for r in scoped}
                bat_f = _batting_form(scoped)
                bowl_f = _bowling_form(scoped)
                combined = _clamp01(0.5 * bat_f + 0.5 * bowl_f)
                comps = sorted(
                    {str(r.get("competition") or "").strip() for r in scoped if r.get("competition")}
                )
                last_dt = max(
                    (_parse_iso(r.get("match_date")) for r in scoped if r.get("match_date")),
                    default=None,
                )
                last_s = last_dt.isoformat() if last_dt else ""
                pos_vals: list[float] = []
                for r in scoped:
                    bp = r.get("batting_position")
                    if bp is None:
                        continue
                    try:
                        pos_vals.append(float(bp))
                    except (TypeError, ValueError):
                        continue
                pos_ema = sum(pos_vals) / len(pos_vals) if pos_vals else None
                pp_s, mid_s, de_s = _phase_shares(c, pk, mids)
                n30 = _count_days_matches(plist, ref, 30)
                n60 = _count_days_matches(plist, ref, 60)
                n150 = _count_days_matches(plist, ref, 150)
                conf = _confidence(len(scoped))
                dbg = {
                    "t20_matches_in_union_window": len(scoped),
                    "competitions_used": comps,
                    "last_n_matches_param": last_n,
                    "months_window_param": months,
                    "reference_as_of": ref.isoformat(),
                }
                c.execute(
                    """
                    INSERT OR REPLACE INTO player_recent_form_cache (
                        player_key, last_updated, reference_as_of_date,
                        t20_matches_in_window, batting_recent_form, bowling_recent_form,
                        combined_recent_form, last_t20_match_date, competitions_json,
                        matches_last_30d, matches_last_60d, matches_last_150d,
                        recent_batting_position_ema, bowling_pp_ball_share,
                        bowling_middle_ball_share, bowling_death_ball_share,
                        sample_confidence, debug_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pk[:80],
                        now,
                        ref.isoformat(),
                        len(scoped),
                        round(bat_f, 6),
                        round(bowl_f, 6),
                        round(combined, 6),
                        last_s,
                        json.dumps(comps, ensure_ascii=False),
                        n30,
                        n60,
                        n150,
                        round(pos_ema, 4) if pos_ema is not None else None,
                        pp_s,
                        mid_s,
                        de_s,
                        round(conf, 6),
                        json.dumps(dbg, ensure_ascii=False),
                    ),
                )
                stats["players_cached"] += 1
            except Exception:
                stats["errors"] += 1
                logger.exception("recent_form_cache player_key=%s", pk)

    if conn is not None:
        _process_connection(conn)
    else:
        with db_mod.connection() as c:
            _process_connection(c)
    stats["reference_as_of_date"] = ref.isoformat()
    return stats


def recent_form_validation_summary() -> dict[str, Any]:
    """Aggregate counts for UI / audits."""
    import db as db_mod

    with db_mod.connection() as conn:
        t20_sql, t20_params = _t20_sql_filter()
        n_t20_matches = conn.execute(
            f"""
            SELECT COUNT(DISTINCT m.id) FROM matches m
            WHERE m.match_date IS NOT NULL AND trim(m.match_date) != ''
              AND {t20_sql}
            """,
            t20_params,
        ).fetchone()
        n_cache = conn.execute(
            "SELECT COUNT(*) FROM player_recent_form_cache"
        ).fetchone()
        n_pms = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()
        sample = conn.execute(
            """
            SELECT player_key, t20_matches_in_window, last_t20_match_date, competitions_json,
                   batting_recent_form, bowling_recent_form, sample_confidence
            FROM player_recent_form_cache
            ORDER BY t20_matches_in_window DESC
            LIMIT 3
            """
        ).fetchall()
    return {
        "distinct_t20_matches_in_db": int(n_t20_matches[0] if n_t20_matches else 0),
        "player_recent_form_cache_rows": int(n_cache[0] if n_cache else 0),
        "player_match_stats_rows": int(n_pms[0] if n_pms else 0),
        "sample_top_players": [dict(r) for r in (sample or [])],
    }
