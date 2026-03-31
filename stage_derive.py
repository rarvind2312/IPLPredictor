"""
Stage 2 — **DERIVE**: SQLite normalized history → derived player / team / H2H profiles.

Reads only ingested tables (``team_match_xi``, ``player_batting_positions``, ``matches``,
``player_match_stats``, ``player_phase_usage``, ``team_match_summary``, ``match_results``).
Does **not** read raw Cricsheet JSON.

Run from the Streamlit admin buttons or call ``run_rebuild_profiles`` / ``run_rebuild_h2h_patterns``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import canonical_keys
import config
import db
import ipl_teams
import matchup_features

logger = logging.getLogger(__name__)


def derive_min_season_year(
    *,
    current_season_year: Optional[int] = None,
    n_seasons: Optional[int] = None,
) -> int:
    """First calendar year in the derive window (inclusive ``n_seasons`` ending at ``current``)."""
    cur = int(current_season_year or getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    n = int(n_seasons or getattr(config, "DERIVE_HISTORY_SEASONS", 5))
    n = max(1, n)
    return cur - (n - 1)


def _venue_key(venue: str) -> str:
    return canonical_keys.canonical_player_key(str(venue or "").strip())[:80] or "unknown_venue"


def _team_key_from_match_side(name: str) -> str:
    lab = ipl_teams.franchise_label_for_storage(name) or str(name or "").strip()
    return ipl_teams.canonical_team_key_for_franchise(lab)[:80]


def _year_clause(alias: str = "m") -> tuple[str, str]:
    """Returns (sql_fragment, yyyy string) for bind param."""
    y = f"{derive_min_season_year():04d}"
    frag = (
        f" AND ({alias}.match_date IS NOT NULL AND length(trim({alias}.match_date)) >= 4 "
        f"AND substr(trim({alias}.match_date), 1, 4) >= ?)"
    )
    return frag, y


def _recency_weights(n: int) -> list[float]:
    if n <= 0:
        return []
    h = float(getattr(config, "DERIVE_RECENCY_HALFLIFE_MATCHES", 20.0))
    h = max(1.0, h)
    out: list[float] = []
    for i in range(n):
        age = n - 1 - i
        out.append(math.exp(-age * math.log(2) / h))
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((z - m) ** 2 for z in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, v))


@dataclass
class DeriveRunSummary:
    player_profiles_built: int = 0
    team_derived_summary_rows: int = 0
    venue_team_pattern_rows: int = 0
    team_selection_rows: int = 0
    head_to_head_pattern_rows: int = 0
    franchise_feature_players_touched: int = 0
    sparse_history_players: int = 0
    fallback_profile_players: int = 0
    sparse_player_keys_sample: list[str] = field(default_factory=list)
    fallback_player_keys_sample: list[str] = field(default_factory=list)
    min_season_year: int = 0
    warnings: list[str] = field(default_factory=list)


def _load_franchise_row(conn: sqlite3.Connection, pk: str, fk: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """
        SELECT * FROM player_franchise_features
        WHERE player_key = ? AND franchise_team_key = ?
        """,
        (pk, fk),
    ).fetchone()
    return dict(row) if row else None


def _build_player_profile_for_pair(
    conn: sqlite3.Connection,
    pk: str,
    fk: str,
    yfrag: str,
    ypar: str,
    now: float,
) -> Optional[dict[str, Any]]:
    team_matches = conn.execute(
        f"""
        SELECT COUNT(DISTINCT t.match_id) AS c
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE t.team_key = ?{yfrag}
        """,
        (fk, ypar),
    ).fetchone()
    n_team = int(team_matches["c"] or 0) if team_matches else 0

    xi_row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT t.match_id) AS c
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE t.player_key = ? AND t.team_key = ?{yfrag}
        """,
        (pk, fk, ypar),
    ).fetchone()
    xi_n = int(xi_row["c"] or 0) if xi_row else 0

    pbp_rows = conn.execute(
        f"""
        SELECT p.batting_position, m.match_date, m.id
        FROM player_batting_positions p
        JOIN matches m ON m.id = p.match_id
        WHERE p.player_key = ? AND p.team_key = ?{yfrag}
        ORDER BY m.match_date ASC NULLS FIRST, m.id ASC, p.innings_number ASC
        """,
        (pk, fk, ypar),
    ).fetchall()

    slots: list[float] = []
    w_op = w_mid = w_fin = 0.0
    w_sum = 0.0
    wts = _recency_weights(len(pbp_rows))
    for i, r in enumerate(pbp_rows):
        try:
            pos = float(r["batting_position"])
        except (TypeError, ValueError):
            continue
        slots.append(pos)
        w = wts[i] if i < len(wts) else 1.0
        w_sum += w
        if pos <= 2.5:
            w_op += w
        elif pos <= 6.5:
            w_mid += w
        else:
            w_fin += w

    opener_l = (w_op / w_sum) if w_sum > 0 else 0.0
    middle_l = (w_mid / w_sum) if w_sum > 0 else 0.0
    finisher_l = (w_fin / w_sum) if w_sum > 0 else 0.0

    xi_freq = (xi_n / max(1, n_team)) if n_team > 0 else 0.0

    # Venue fit: XI rate at each venue vs global XI rate for this player
    vrows = conn.execute(
        f"""
        SELECT COALESCE(NULLIF(trim(m.venue), ''), '') AS v, COUNT(DISTINCT t.match_id) AS c
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE t.player_key = ? AND t.team_key = ?{yfrag} AND m.venue IS NOT NULL AND trim(m.venue) != ''
        GROUP BY v
        """,
        (pk, fk, ypar),
    ).fetchall()
    venue_scores: list[float] = []
    global_rate = xi_n / max(1, n_team) if n_team else 0.0
    for vr in vrows:
        v = str(vr["v"] or "")
        if not v:
            continue
        vm = conn.execute(
            f"""
            SELECT COUNT(DISTINCT t.match_id) AS c
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key = ? AND COALESCE(trim(m.venue),'') = ?{yfrag}
            """,
            (fk, v, ypar),
        ).fetchone()
        denom = int(vm["c"] or 0) if vm else 0
        num = int(vr["c"] or 0)
        local = num / max(1, denom)
        if global_rate > 1e-6:
            venue_scores.append(_clamp01(local / global_rate))
        elif local > 0:
            venue_scores.append(1.0)
    venue_fit = sum(venue_scores) / len(venue_scores) if venue_scores else 0.5

    stdev = _std(slots)
    role_stab = _clamp01(1.0 / (1.0 + stdev))

    third = max(1, n_team // 3)
    recent_n = conn.execute(
        f"""
        WITH recent AS (
            SELECT t.match_id
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.player_key = ? AND t.team_key = ?{yfrag}
            ORDER BY m.match_date DESC NULLS LAST, m.id DESC
            LIMIT ?
        )
        SELECT COUNT(DISTINCT match_id) AS c FROM recent
        """,
        (pk, fk, ypar, third),
    ).fetchone()
    rct = int(recent_n["c"] or 0) if recent_n else 0
    recent_usage = rct / max(1, xi_n) if xi_n > 0 else 0.0

    h2h_counts: dict[str, int] = defaultdict(int)
    hrows = conn.execute(
        f"""
        SELECT DISTINCT m.id AS mid,
          CASE
            WHEN lower(trim(t.team_name)) = lower(trim(m.team_a)) THEN m.team_b
            ELSE m.team_a
          END AS opp
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE t.player_key = ? AND t.team_key = ?{yfrag}
        """,
        (pk, fk, ypar),
    ).fetchall()
    for hr in hrows:
        on = str(hr["opp"] or "").strip()
        if not on:
            continue
        ok = _team_key_from_match_side(on)
        if ok:
            h2h_counts[ok] += 1

    fr = _load_franchise_row(conn, pk, fk)
    b_ema = float(fr["batting_position_ema"]) if fr and fr.get("batting_position_ema") is not None else None
    pp_l = float(fr["phase_bowl_rate_pp"]) if fr else 0.0
    mid_l = float(fr["phase_bowl_rate_middle"]) if fr else 0.0
    dt_l = float(fr["phase_bowl_rate_death"]) if fr else 0.0
    bag = float(fr["batting_aggressor_score"]) if fr else 0.0
    bctl = float(fr["bowling_control_score"]) if fr else 0.0
    vs_s = float(fr["vs_spin_tendency"]) if fr else 0.0
    vs_p = float(fr["vs_pace_tendency"]) if fr else 0.0

    sample_m = xi_n
    parts = [
        _clamp01(math.log1p(sample_m) / math.log1p(24)),
        _clamp01(len(slots) / 12.0),
        _clamp01(xi_freq * 2.0),
    ]
    conf = sum(parts) / len(parts)
    sparse_thr = int(getattr(config, "DERIVE_SPARSE_PLAYER_SAMPLES", 3))
    fb_thr = float(getattr(config, "DERIVE_FALLBACK_CONFIDENCE_MAX", 0.35))
    if sample_m < sparse_thr or len(slots) < 2:
        conf = min(conf, fb_thr)

    hint_row = conn.execute(
        "SELECT player_name FROM team_match_xi WHERE player_key = ? AND team_key = ? LIMIT 1",
        (pk, fk),
    ).fetchone()
    hint = str(hint_row["player_name"] or "")[:120] if hint_row else None

    return {
        "player_key": pk,
        "franchise_team_key": fk,
        "display_name_hint": hint,
        "xi_selection_frequency": round(xi_freq, 6),
        "batting_position_ema": round(b_ema, 4) if b_ema is not None else None,
        "opener_likelihood": round(opener_l, 5),
        "middle_order_likelihood": round(middle_l, 5),
        "finisher_likelihood": round(finisher_l, 5),
        "powerplay_bowler_likelihood": round(pp_l, 5),
        "middle_overs_bowler_likelihood": round(mid_l, 5),
        "death_bowler_likelihood": round(dt_l, 5),
        "batting_aggressor_score": round(bag, 5),
        "bowling_control_score": round(bctl, 5),
        "batting_vs_spin_tendency": round(vs_s, 5),
        "batting_vs_pace_tendency": round(vs_p, 5),
        "venue_fit_score": round(venue_fit, 5),
        "role_stability_score": round(role_stab, 5),
        "recent_usage_score": round(recent_usage, 5),
        "h2h_basis_json": json.dumps(dict(sorted(h2h_counts.items(), key=lambda kv: -kv[1])[:40]), ensure_ascii=False),
        "profile_confidence": round(conf, 5),
        "sample_matches": sample_m,
        "last_updated": now,
    }


def rebuild_player_profiles(conn: sqlite3.Connection, summary: DeriveRunSummary) -> int:
    yfrag, ypar = _year_clause("m")
    now = time.time()
    pairs = conn.execute(
        f"""
        SELECT DISTINCT t.player_key, t.team_key
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE t.player_key IS NOT NULL AND trim(t.player_key) != ''
          AND t.team_key IS NOT NULL AND trim(t.team_key) != ''{yfrag}
        """,
        (ypar,),
    ).fetchall()

    sparse_thr = int(getattr(config, "DERIVE_SPARSE_PLAYER_SAMPLES", 3))
    fb_thr = float(getattr(config, "DERIVE_FALLBACK_CONFIDENCE_MAX", 0.35))
    sparse_sample: list[str] = []
    fb_sample: list[str] = []

    n = 0
    summary.sparse_history_players = 0
    summary.fallback_profile_players = 0
    for r in pairs:
        pk = str(r["player_key"] or "").strip()[:80]
        fk = str(r["team_key"] or "").strip()[:80]
        if not pk or not fk:
            continue
        row = _build_player_profile_for_pair(conn, pk, fk, yfrag, ypar, now)
        if not row:
            continue
        conn.execute(
            """
            INSERT INTO player_profiles (
                player_key, franchise_team_key, display_name_hint,
                xi_selection_frequency, batting_position_ema,
                opener_likelihood, middle_order_likelihood, finisher_likelihood,
                powerplay_bowler_likelihood, middle_overs_bowler_likelihood, death_bowler_likelihood,
                batting_aggressor_score, bowling_control_score,
                batting_vs_spin_tendency, batting_vs_pace_tendency,
                venue_fit_score, role_stability_score, recent_usage_score,
                h2h_basis_json, profile_confidence, sample_matches, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_key, franchise_team_key) DO UPDATE SET
                display_name_hint = excluded.display_name_hint,
                xi_selection_frequency = excluded.xi_selection_frequency,
                batting_position_ema = excluded.batting_position_ema,
                opener_likelihood = excluded.opener_likelihood,
                middle_order_likelihood = excluded.middle_order_likelihood,
                finisher_likelihood = excluded.finisher_likelihood,
                powerplay_bowler_likelihood = excluded.powerplay_bowler_likelihood,
                middle_overs_bowler_likelihood = excluded.middle_overs_bowler_likelihood,
                death_bowler_likelihood = excluded.death_bowler_likelihood,
                batting_aggressor_score = excluded.batting_aggressor_score,
                bowling_control_score = excluded.bowling_control_score,
                batting_vs_spin_tendency = excluded.batting_vs_spin_tendency,
                batting_vs_pace_tendency = excluded.batting_vs_pace_tendency,
                venue_fit_score = excluded.venue_fit_score,
                role_stability_score = excluded.role_stability_score,
                recent_usage_score = excluded.recent_usage_score,
                h2h_basis_json = excluded.h2h_basis_json,
                profile_confidence = excluded.profile_confidence,
                sample_matches = excluded.sample_matches,
                last_updated = excluded.last_updated
            """,
            (
                row["player_key"],
                row["franchise_team_key"],
                row["display_name_hint"],
                row["xi_selection_frequency"],
                row["batting_position_ema"],
                row["opener_likelihood"],
                row["middle_order_likelihood"],
                row["finisher_likelihood"],
                row["powerplay_bowler_likelihood"],
                row["middle_overs_bowler_likelihood"],
                row["death_bowler_likelihood"],
                row["batting_aggressor_score"],
                row["bowling_control_score"],
                row["batting_vs_spin_tendency"],
                row["batting_vs_pace_tendency"],
                row["venue_fit_score"],
                row["role_stability_score"],
                row["recent_usage_score"],
                row["h2h_basis_json"],
                row["profile_confidence"],
                row["sample_matches"],
                row["last_updated"],
            ),
        )
        n += 1
        sm = int(row["sample_matches"] or 0)
        cf = float(row["profile_confidence"] or 0)
        if sm < sparse_thr:
            summary.sparse_history_players += 1
            if len(sparse_sample) < 40:
                sparse_sample.append(pk)
        if cf <= fb_thr:
            summary.fallback_profile_players += 1
            if len(fb_sample) < 40:
                fb_sample.append(pk)

    summary.sparse_player_keys_sample = sparse_sample
    summary.fallback_player_keys_sample = fb_sample
    return n


def _bowling_phase_json_for_team(conn: sqlite3.Connection, fk: str, yfrag: str, ypar: str) -> str:
    rows = conn.execute(
        f"""
        SELECT p.phase, SUM(p.balls) AS b
        FROM player_phase_usage p
        JOIN matches m ON m.id = p.match_id
        WHERE p.team_key = ? AND p.role = 'bowl'{yfrag}
        GROUP BY p.phase
        """,
        (fk, ypar),
    ).fetchall()
    d = {str(r["phase"] or ""): int(r["b"] or 0) for r in rows}
    return json.dumps(d, ensure_ascii=False)


def _team_stability_from_summaries(conn: sqlite3.Connection, fk: str, yfrag: str, ypar: str) -> tuple[float, float, float]:
    rows = conn.execute(
        f"""
        SELECT s.batting_order_json, s.playing_xi_json, s.overseas_combo_json
        FROM team_match_summary s
        JOIN matches m ON m.id = s.match_id
        WHERE s.team_key = ?{yfrag}
        ORDER BY m.match_date ASC NULLS FIRST, m.id ASC
        """,
        (fk, ypar),
    ).fetchall()
    if not rows:
        return 0.5, 0.5, 0.5
    open_sigs: list[str] = []
    finish_sigs: list[str] = []
    keepers: list[str] = []
    for r in rows:
        try:
            bo = json.loads(r["batting_order_json"] or "[]")
        except json.JSONDecodeError:
            bo = []
        if isinstance(bo, list) and len(bo) >= 2:
            open_sigs.append(f"{bo[0]}|{bo[1]}")
        elif isinstance(bo, list) and len(bo) == 1:
            open_sigs.append(str(bo[0]))
        if isinstance(bo, list) and len(bo) >= 3:
            finish_sigs.append("|".join(str(x) for x in bo[-3:]))
        try:
            xi = json.loads(r["playing_xi_json"] or "[]")
        except json.JSONDecodeError:
            xi = []
        if isinstance(xi, list):
            for name in xi:
                n = str(name).lower()
                if "wk" in n or "keeper" in n:
                    keepers.append(canonical_keys.canonical_player_key(str(name))[:40])
                    break
    def _stab(sig_list: list[str]) -> float:
        if len(sig_list) < 2:
            return 0.5
        mc = Counter(sig_list).most_common(1)[0][1]
        return _clamp01(mc / len(sig_list))

    op_st = _stab(open_sigs)
    fn_st = _stab(finish_sigs)
    kp_st = _stab(keepers) if keepers else 0.5
    return kp_st, op_st, fn_st


def rebuild_team_derived_and_venue(conn: sqlite3.Connection, summary: DeriveRunSummary) -> None:
    yfrag, ypar = _year_clause("m")
    now = time.time()
    team_keys = [
        str(r[0]).strip()[:80]
        for r in conn.execute(
            f"""
            SELECT DISTINCT t.team_key
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key IS NOT NULL AND trim(t.team_key) != ''{yfrag}
            """,
            (ypar,),
        ).fetchall()
    ]
    team_keys = sorted(set(team_keys))

    key_to_label: dict[str, str] = {}
    for slug in ipl_teams.TEAM_SLUGS:
        lab = ipl_teams.label_for_slug(slug)
        key_to_label[ipl_teams.canonical_team_key_for_franchise(lab)[:80]] = lab

    mrows_all = conn.execute(
        f"""
        SELECT mr.batting_first, mr.winner, m.team_a, m.team_b
        FROM matches m
        JOIN match_results mr ON mr.id = m.id
        WHERE m.team_a IS NOT NULL AND m.team_b IS NOT NULL{yfrag}
        """,
        (ypar,),
    ).fetchall()
    chase_def_map: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for mr in mrows_all:
        ta = str(mr["team_a"] or "")
        tb = str(mr["team_b"] or "")
        kta = _team_key_from_match_side(ta)
        ktb = _team_key_from_match_side(tb)
        wn = str(mr["winner"] or "").strip()
        bf = str(mr["batting_first"] or "").strip()
        if not wn or not bf:
            continue
        kw = _team_key_from_match_side(wn)
        if kw not in (kta, ktb):
            continue
        lab = key_to_label.get(kw)
        if not lab:
            continue
        bucket = chase_def_map[kw]
        if bf.lower() == lab.strip().lower():
            bucket[1] += 1
        else:
            bucket[0] += 1

    for fk in team_keys:
        if not fk:
            continue
        n_matches = conn.execute(
            f"""
            SELECT COUNT(DISTINCT t.match_id) AS c
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key = ?{yfrag}
            """,
            (fk, ypar),
        ).fetchone()
        nm = int(n_matches["c"] or 0) if n_matches else 0

        xi_counts = conn.execute(
            f"""
            SELECT t.player_key, COUNT(DISTINCT t.match_id) AS c
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key = ?{yfrag}
            GROUP BY t.player_key
            ORDER BY c DESC
            LIMIT 16
            """,
            (fk, ypar),
        ).fetchall()
        core = [{"player_key": str(r["player_key"]), "xi_matches": int(r["c"] or 0)} for r in xi_counts]

        os_counts: Counter[int] = Counter()
        os_rows = conn.execute(
            f"""
            SELECT s.overseas_combo_json
            FROM team_match_summary s
            JOIN matches m ON m.id = s.match_id
            WHERE s.team_key = ?{yfrag}
            """,
            (fk, ypar),
        ).fetchall()
        for orow in os_rows:
            try:
                blob = json.loads(orow["overseas_combo_json"] or "{}")
            except json.JSONDecodeError:
                blob = {}
            if isinstance(blob, dict):
                for side in ("team_a_overseas", "team_b_overseas"):
                    if side in blob:
                        try:
                            os_counts[int(blob[side])] += 1
                        except (TypeError, ValueError):
                            pass
        os_json = json.dumps({str(k): v for k, v in os_counts.most_common(8)}, ensure_ascii=False)

        kp, op_st, fn_st = _team_stability_from_summaries(conn, fk, yfrag, ypar)
        bowl_json = _bowling_phase_json_for_team(conn, fk, yfrag, ypar)

        cd = chase_def_map.get(fk, [0, 0])
        chase, defend = int(cd[0]), int(cd[1])
        chase_def = json.dumps({"chase_wins": chase, "defend_wins": defend}, ensure_ascii=False)

        conn.execute(
            """
            INSERT INTO team_derived_summary (
                team_key, preferred_xi_core_json, preferred_overseas_combinations_json,
                keeper_consistency, opener_stability, finisher_stability,
                bowling_composition_patterns_json, chase_vs_defend_json,
                sample_matches, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_key) DO UPDATE SET
                preferred_xi_core_json = excluded.preferred_xi_core_json,
                preferred_overseas_combinations_json = excluded.preferred_overseas_combinations_json,
                keeper_consistency = excluded.keeper_consistency,
                opener_stability = excluded.opener_stability,
                finisher_stability = excluded.finisher_stability,
                bowling_composition_patterns_json = excluded.bowling_composition_patterns_json,
                chase_vs_defend_json = excluded.chase_vs_defend_json,
                sample_matches = excluded.sample_matches,
                last_updated = excluded.last_updated
            """,
            (
                fk,
                json.dumps(core, ensure_ascii=False),
                os_json,
                kp,
                op_st,
                fn_st,
                bowl_json,
                chase_def,
                nm,
                now,
            ),
        )
        summary.team_derived_summary_rows += 1

    # Venue + team patterns and team_selection_patterns
    vrows = conn.execute(
        f"""
        SELECT DISTINCT COALESCE(NULLIF(trim(m.venue), ''), '') AS venue, t.team_key
        FROM team_match_xi t
        JOIN matches m ON m.id = t.match_id
        WHERE m.venue IS NOT NULL AND trim(m.venue) != ''{yfrag}
        """,
        (ypar,),
    ).fetchall()
    for vr in vrows:
        v_raw = str(vr["venue"] or "").strip()
        tkey = str(vr["team_key"] or "").strip()[:80]
        if not v_raw or not tkey:
            continue
        vk = _venue_key(v_raw)
        nm = conn.execute(
            f"""
            SELECT COUNT(DISTINCT t.match_id) AS c
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key = ? AND COALESCE(trim(m.venue),'') = ?{yfrag}
            """,
            (tkey, v_raw, ypar),
        ).fetchone()
        nmv = int(nm["c"] or 0) if nm else 0
        xic = conn.execute(
            f"""
            SELECT t.player_key, COUNT(DISTINCT t.match_id) AS c
            FROM team_match_xi t
            JOIN matches m ON m.id = t.match_id
            WHERE t.team_key = ? AND COALESCE(trim(m.venue),'') = ?{yfrag}
            GROUP BY t.player_key
            ORDER BY c DESC
            LIMIT 18
            """,
            (tkey, v_raw, ypar),
        ).fetchall()
        xi_json = json.dumps(
            [{"player_key": str(r["player_key"]), "count": int(r["c"] or 0)} for r in xic],
            ensure_ascii=False,
        )
        kp, op_st, fn_st = _team_stability_from_summaries(conn, tkey, yfrag, ypar)
        bowl_json = _bowling_phase_json_for_team(conn, tkey, yfrag, ypar)
        os_json = "{}"
        chase_def_local = "{}"
        row_vals = (
            nmv,
            xi_json,
            os_json,
            xi_json,
            kp,
            op_st,
            fn_st,
            bowl_json,
            chase_def_local,
            now,
        )
        conn.execute(
            """
            INSERT INTO venue_team_patterns (
                venue_key, team_key, sample_matches, xi_frequency_json, overseas_combo_json,
                preferred_xi_core_json, keeper_consistency, opener_stability, finisher_stability,
                bowling_composition_json, chase_defend_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(venue_key, team_key) DO UPDATE SET
                sample_matches = excluded.sample_matches,
                xi_frequency_json = excluded.xi_frequency_json,
                overseas_combo_json = excluded.overseas_combo_json,
                preferred_xi_core_json = excluded.preferred_xi_core_json,
                keeper_consistency = excluded.keeper_consistency,
                opener_stability = excluded.opener_stability,
                finisher_stability = excluded.finisher_stability,
                bowling_composition_json = excluded.bowling_composition_json,
                chase_defend_json = excluded.chase_defend_json,
                last_updated = excluded.last_updated
            """,
            (vk, tkey) + row_vals,
        )
        conn.execute(
            """
            INSERT INTO team_selection_patterns (
                team_key, venue_key, sample_matches, xi_frequency_json, overseas_combo_json,
                preferred_xi_core_json, keeper_consistency, opener_stability, finisher_stability,
                bowling_composition_json, chase_defend_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_key, venue_key) DO UPDATE SET
                sample_matches = excluded.sample_matches,
                xi_frequency_json = excluded.xi_frequency_json,
                overseas_combo_json = excluded.overseas_combo_json,
                preferred_xi_core_json = excluded.preferred_xi_core_json,
                keeper_consistency = excluded.keeper_consistency,
                opener_stability = excluded.opener_stability,
                finisher_stability = excluded.finisher_stability,
                bowling_composition_json = excluded.bowling_composition_json,
                chase_defend_json = excluded.chase_defend_json,
                last_updated = excluded.last_updated
            """,
            (tkey, vk) + row_vals,
        )
        summary.venue_team_pattern_rows += 1
        summary.team_selection_rows += 1


def rebuild_head_to_head_patterns(conn: sqlite3.Connection, summary: DeriveRunSummary) -> int:
    yfrag, ypar = _year_clause("m")
    now = time.time()
    rows = conn.execute(
        f"""
        SELECT m.id, m.match_date, m.team_a, m.team_b, mr.winner
        FROM matches m
        JOIN match_results mr ON mr.id = m.id
        WHERE m.team_a IS NOT NULL AND trim(m.team_a) != ''
          AND m.team_b IS NOT NULL AND trim(m.team_b) != ''{yfrag}
        ORDER BY m.match_date ASC NULLS FIRST, m.id ASC
        """,
        (ypar,),
    ).fetchall()
    n = len(rows)
    wts = _recency_weights(n)
    agg2: dict[tuple[str, str], dict[str, Any]] = {}
    for i, r in enumerate(rows):
        ka = _team_key_from_match_side(str(r["team_a"] or ""))
        kb = _team_key_from_match_side(str(r["team_b"] or ""))
        if not ka or not kb or ka == kb:
            continue
        k1, k2 = (ka, kb) if ka < kb else (kb, ka)
        w = wts[i] if i < len(wts) else 1.0
        b = agg2.setdefault((k1, k2), {"w": 0.0, "wa": 0.0, "wb": 0.0, "n": 0})
        b["w"] += w
        b["n"] += 1
        wn = str(r["winner"] or "").strip()
        if not wn:
            continue
        kw = _team_key_from_match_side(wn)
        if kw == k1:
            b["wa"] += w
        elif kw == k2:
            b["wb"] += w

    count = 0
    for (k1, k2), b in agg2.items():
        basis = json.dumps(
            {
                "team_a_key": k1,
                "team_b_key": k2,
                "matches": b["n"],
                "weight_sum": round(b["w"], 4),
            },
            ensure_ascii=False,
        )
        conn.execute(
            """
            INSERT INTO head_to_head_patterns (
                team_a_key, team_b_key, sample_matches, weight_sum,
                team_a_wins_weighted, team_b_wins_weighted, head_to_head_basis_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_a_key, team_b_key) DO UPDATE SET
                sample_matches = excluded.sample_matches,
                weight_sum = excluded.weight_sum,
                team_a_wins_weighted = excluded.team_a_wins_weighted,
                team_b_wins_weighted = excluded.team_b_wins_weighted,
                head_to_head_basis_json = excluded.head_to_head_basis_json,
                last_updated = excluded.last_updated
            """,
            (
                k1,
                k2,
                int(b["n"]),
                float(b["w"]),
                float(b["wa"]),
                float(b["wb"]),
                basis,
                now,
            ),
        )
        count += 1
    summary.head_to_head_pattern_rows = count
    return count


def run_rebuild_profiles(
    *,
    min_season_year: Optional[int] = None,
) -> DeriveRunSummary:
    """
    Refresh ``player_franchise_features``, ``player_profiles``, team/venue pattern tables.
    Does **not** rebuild ``head_to_head_patterns`` (use ``run_rebuild_h2h_patterns``).
    """
    summary = DeriveRunSummary()
    my = int(min_season_year if min_season_year is not None else derive_min_season_year())
    summary.min_season_year = my

    with db.connection() as conn:
        summary.franchise_feature_players_touched = 0
        for tk in sorted(
            {
                str(r[0]).strip()[:80]
                for r in conn.execute(
                    "SELECT DISTINCT team_key FROM player_match_stats WHERE team_key IS NOT NULL AND trim(team_key) != ''"
                ).fetchall()
            }
        ):
            if tk:
                summary.franchise_feature_players_touched += matchup_features.refresh_franchise_features(
                    conn, tk, min_season_year=my
                )

        summary.player_profiles_built = rebuild_player_profiles(conn, summary)
        rebuild_team_derived_and_venue(conn, summary)

    return summary


def run_rebuild_h2h_patterns(
    *,
    min_season_year: Optional[int] = None,
) -> DeriveRunSummary:
    """Rebuild only ``head_to_head_patterns`` from SQLite fixtures."""
    summary = DeriveRunSummary()
    summary.min_season_year = int(min_season_year if min_season_year is not None else derive_min_season_year())
    with db.connection() as conn:
        rebuild_head_to_head_patterns(conn, summary)
    return summary


def derive_debug_snapshot() -> dict[str, Any]:
    """Lightweight counts for UI debug."""
    with db.connection() as conn:
        def c(q: str) -> int:
            return int(conn.execute(q).fetchone()[0])

        return {
            "player_profiles_rows": c("SELECT COUNT(*) FROM player_profiles"),
            "team_derived_summary_rows": c("SELECT COUNT(*) FROM team_derived_summary"),
            "venue_team_patterns_rows": c("SELECT COUNT(*) FROM venue_team_patterns"),
            "team_selection_patterns_rows": c("SELECT COUNT(*) FROM team_selection_patterns"),
            "head_to_head_patterns_rows": c("SELECT COUNT(*) FROM head_to_head_patterns"),
            "player_franchise_features_rows": c("SELECT COUNT(*) FROM player_franchise_features"),
        }


__all__ = [
    "DeriveRunSummary",
    "derive_debug_snapshot",
    "derive_min_season_year",
    "rebuild_head_to_head_patterns",
    "run_rebuild_h2h_patterns",
    "run_rebuild_profiles",
]
