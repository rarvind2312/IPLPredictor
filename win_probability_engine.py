"""
Deterministic, rule-based win probability from weighted squad + venue + history factors.

Outputs P(Team A wins) for each toss scenario (A bats first vs B bats first).
Designed so factors can be swapped for learned models later without changing the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

import config
import db
import h2h_history
import ipl_teams
import learner

from venues import VenueProfile


class _PlayerLike(Protocol):
    name: str
    role: str
    bat_skill: float
    bowl_skill: float
    composite: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_team(s: str) -> str:
    return learner.normalize_player_key(s or "")[:120]


def _teams_same_pair(ta: str, tb: str, ra: str, rb: str) -> bool:
    """Whether row teams are the same two sides as (ta, tb), order-independent."""
    na, nb = _norm_team(ta), _norm_team(tb)
    rna, rnb = _norm_team(ra), _norm_team(rb)
    if len(na) < 2 or len(nb) < 2 or len(rna) < 2 or len(rnb) < 2:
        return False
    return {na, nb} == {rna, rnb}


def _team_equals(a: str, b: str) -> bool:
    na, nb = _norm_team(a), _norm_team(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _team_played_here(team: str, ra: str, rb: str) -> bool:
    return _team_equals(team, ra) or _team_equals(team, rb)


def _venue_row_matches(row_venue: str, venue_keys: Iterable[str]) -> bool:
    rv = _norm_team(row_venue)
    if not rv:
        return False
    for vk in venue_keys:
        k = _norm_team(vk)
        if not k:
            continue
        if k in rv or rv in k:
            return True
    return False


def _bowler_pool(xi: list[_PlayerLike]) -> list[_PlayerLike]:
    return [p for p in xi if p.bowl_skill >= 0.36 or p.role in ("bowl", "all", "wk")]


def _spin_tendency(p: _PlayerLike) -> float:
    """Proxy 0–1: how spin-oriented the bowler profile is (rule-based, not a label)."""
    if p.role == "bowl" and p.bat_skill < 0.40:
        return 0.22
    if p.role == "all":
        return 0.55 + 0.15 * (p.bat_skill - 0.55)
    if p.role == "bat":
        return 0.12
    if p.role == "wk":
        return 0.25
    return 0.38


def _pace_tendency(p: _PlayerLike) -> float:
    return _clamp(1.0 - _spin_tendency(p) * 0.92, 0.08, 1.0)


def _batter_spin_preference(p: _PlayerLike) -> float:
    """Proxy for who is likely to target spin vs pace (no handedness in squad schema)."""
    if p.role == "all":
        return 0.58
    if p.role == "bat":
        return 0.44
    if p.role == "bowl":
        return 0.52
    return 0.50


def _xi_by_name(xi: list[_PlayerLike]) -> dict[str, _PlayerLike]:
    return {p.name: p for p in xi}


def team_chase_defend_rates(
    team: str, rows: list[dict[str, Any]]
) -> tuple[float, float, int, int]:
    """
    From stored results: win rate when this team batted first (defend/set total)
    vs when they chased (batted second). Rows need ``batting_first`` and ``winner``.
    """
    cw = cl = dw = dl = 0
    for row in rows:
        bf = (row.get("batting_first") or "").strip()
        w = (row.get("winner") or "").strip()
        ta = str(row.get("team_a") or "")
        tb = str(row.get("team_b") or "")
        if not bf or not w:
            continue
        if not (_team_equals(team, ta) or _team_equals(team, tb)):
            continue
        if not (_team_equals(w, ta) or _team_equals(w, tb)):
            continue
        t_bat_first = _team_equals(team, bf)
        t_won = _team_equals(team, w)
        if t_bat_first:
            if t_won:
                dw += 1
            else:
                dl += 1
        else:
            if t_won:
                cw += 1
            else:
                cl += 1
    n_chase = cw + cl
    n_def = dw + dl
    chase_wr = (cw / n_chase) if n_chase else 0.5
    def_wr = (dw / n_def) if n_def else 0.5
    return chase_wr, def_wr, n_chase, n_def


def toss_role_scores(
    team_a: str,
    team_b: str,
    rows: list[dict[str, Any]],
    *,
    a_bats_first: bool,
) -> tuple[float, float, list[str]]:
    """
    Historical fit for the innings role implied by the toss scenario:
    if A bats first, A's defend record vs B's chase record; else the reverse.
    """
    ca, da, nc, nda = team_chase_defend_rates(team_a, rows)
    cb, db, nc_b, ndb = team_chase_defend_rates(team_b, rows)
    notes: list[str] = []

    def shrink(wr: float, n: int) -> float:
        if n <= 0:
            return 0.5
        return 0.5 + (wr - 0.5) * min(1.0, n / max(6.0, float(n)))

    if a_bats_first:
        sa = 50.0 + 26.0 * (shrink(da, nda) - 0.5)
        sb = 50.0 + 26.0 * (shrink(cb, nc_b) - 0.5)
        notes.append(
            f"Innings roles: {team_a} defend ~{da:.2f} (n={nda}), {team_b} chase ~{cb:.2f} (n={nc_b})"
        )
    else:
        sa = 50.0 + 26.0 * (shrink(ca, nc) - 0.5)
        sb = 50.0 + 26.0 * (shrink(db, ndb) - 0.5)
        notes.append(
            f"Innings roles: {team_a} chase ~{ca:.2f} (n={nc}), {team_b} defend ~{db:.2f} (n={ndb})"
        )
    return _clamp(sa, 12.0, 88.0), _clamp(sb, 12.0, 88.0), notes


def chase_environment_scores(
    *,
    a_bats_first: bool,
    venue_chase_share: float,
    venue_chase_n: int,
    conditions: dict[str, Any],
    is_night_fixture: bool,
) -> tuple[float, float, list[str]]:
    """
    Tilt scores toward the side batting second when venue, dew, and night favour chasing.
    """
    dew = float(conditions["dew_risk"])
    rain = float(conditions["rain_disruption_risk"])
    hum = float((conditions.get("weather_snapshot") or {}).get("relative_humidity_pct") or 55.0)
    hum = _clamp(hum / 100.0, 0.0, 1.0)

    venue_tilt = (venue_chase_share - 0.5) * 16.0 * min(1.0, venue_chase_n / max(8.0, float(venue_chase_n)))
    dew_tilt = (dew - 0.5) * 11.0
    night_tilt = (0.55 if is_night_fixture else 0.0) * dew * 7.0
    hum_tilt = (hum - 0.55) * 3.0 * dew
    rain_penalty = rain * 5.0
    raw_tilt = venue_tilt + dew_tilt + night_tilt + hum_tilt - rain_penalty

    if a_bats_first:
        score_a = 50.0 - 0.42 * raw_tilt
        score_b = 50.0 + 0.42 * raw_tilt
    else:
        score_a = 50.0 + 0.42 * raw_tilt
        score_b = 50.0 - 0.42 * raw_tilt

    notes = [
        f"Chase env: venue chase share {venue_chase_share:.2f} (n≈{venue_chase_n}), "
        f"dew {dew:.2f}, night {int(is_night_fixture)}, tilt {raw_tilt:.1f} pts (2nd innings side)"
    ]
    return (
        _clamp(score_a, 10.0, 90.0),
        _clamp(score_b, 10.0, 90.0),
        notes,
    )


def build_chase_defend_context(
    team_a: str,
    team_b: str,
    rows: list[dict[str, Any]],
    venue_keys: list[str],
    chase_share_by_venue: dict[str, tuple[float, int]],
    conditions: dict[str, Any],
    *,
    is_night_fixture: bool,
) -> dict[str, Any]:
    ca, da, nca, nda = team_chase_defend_rates(team_a, rows)
    cb, db, ncb, ndb = team_chase_defend_rates(team_b, rows)
    v_share, v_n = 0.5, 0
    for vk in venue_keys:
        row = chase_share_by_venue.get(vk)
        if row:
            v_share, v_n = float(row[0]), int(row[1])
            break
    dew = float(conditions["dew_risk"])
    night = bool(is_night_fixture)
    env_tilt = (v_share - 0.5) * 12.0 + (dew - 0.5) * 9.0 + (0.5 if night else 0.0) * dew * 6.0
    return {
        "team_a_chase_win_rate": round(ca, 4),
        "team_a_defend_win_rate": round(da, 4),
        "team_a_chase_sample_matches": nca,
        "team_a_defend_sample_matches": nda,
        "team_b_chase_win_rate": round(cb, 4),
        "team_b_defend_win_rate": round(db, 4),
        "team_b_chase_sample_matches": ncb,
        "team_b_defend_sample_matches": ndb,
        "venue_chase_win_share": round(v_share, 4),
        "venue_chase_total_decisions": v_n,
        "dew_risk": round(dew, 4),
        "fixture_night": night,
        "environmental_chase_tilt": round(env_tilt, 4),
    }


def _blend_h2h_with_derived_patterns(
    franchise_key_a: str,
    franchise_key_b: str,
    base: tuple[float, float, list[str]],
    dr: Optional[dict[str, Any]],
) -> tuple[float, float, list[str]]:
    """Blend match-row H2H with Stage-2 ``head_to_head_patterns`` when sample is healthy."""
    min_n = int(getattr(config, "WIN_ENG_DERIVED_H2H_MIN_SAMPLES", 6))
    alpha = float(getattr(config, "WIN_ENG_DERIVED_H2H_BLEND", 0.2))
    alpha = max(0.0, min(0.55, alpha))
    if not dr or int(dr.get("sample_matches") or 0) < min_n:
        return base
    ws = float(dr.get("weight_sum") or 0.0)
    if ws <= 1e-9:
        return base
    wa = float(dr.get("team_a_wins_weighted") or 0.0)
    wb = float(dr.get("team_b_wins_weighted") or 0.0)
    k1 = str(dr.get("team_a_key") or "")
    k2 = str(dr.get("team_b_key") or "")
    a = (franchise_key_a or "").strip()[:80]
    b = (franchise_key_b or "").strip()[:80]
    if a == k1:
        edge = (wa - wb) / ws
    elif a == k2:
        edge = (wb - wa) / ws
    else:
        return base
    derived_a = _clamp(50.0 + 44.0 * edge, 8.0, 92.0)
    derived_b = _clamp(100.0 - derived_a, 8.0, 92.0)
    na = (1.0 - alpha) * base[0] + alpha * derived_a
    nb = (1.0 - alpha) * base[1] + alpha * derived_b
    notes = list(base[2]) + [
        f"Blended SQLite head_to_head_patterns (α={alpha:.2f}, n={int(dr.get('sample_matches') or 0)})"
    ]
    return na, nb, notes


def head_to_head_scores(team_a: str, team_b: str, rows: list[dict[str, Any]]) -> tuple[float, float, list[str]]:
    """
    Recency-weighted direct H2H between ``team_a`` and ``team_b`` (IPL SQLite rows).

    Current-season H2H matches count more than older seasons so stale fixtures do not dominate.
    """
    n_max = int(config.WIN_ENG_H2H_MAX_MATCHES)
    cur_season = int(getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    h2h = h2h_history.filter_match_rows_to_h2h(rows, team_a, team_b)
    h2h = h2h_history.sort_h2h_rows_recent_first(h2h)[: max(n_max * 2, 60)]

    w_sum = w_a = w_b = 0.0
    used = 0
    for row in h2h:
        if used >= n_max:
            break
        win = (row.get("winner") or "").strip()
        if not win:
            continue
        y = h2h_history.year_from_match_row(row)
        wt = h2h_history.recency_weight(y, cur_season)
        w_sum += wt
        used += 1
        if h2h_history.team_equals_label(team_a, win):
            w_a += wt
        elif h2h_history.team_equals_label(team_b, win):
            w_b += wt

    notes: list[str] = []
    if w_sum <= 0 or used == 0:
        return 50.0, 50.0, ["No head-to-head results in database for this pair"]

    edge = (w_a - w_b) / w_sum
    score_a = _clamp(50.0 + 44.0 * edge, 8.0, 92.0)
    score_b = _clamp(100.0 - score_a, 8.0, 92.0)
    notes.append(
        f"H2H (recency-weighted, n={used}): {team_a} weighted edge {edge:+.3f} vs {team_b}"
    )
    return score_a, score_b, notes


def venue_h2h_scores(
    team_a: str,
    team_b: str,
    venue_keys: list[str],
    rows: list[dict[str, Any]],
    *,
    min_matches: int = 3,
) -> tuple[float, float, int, list[str]]:
    """
    Head-to-head between the two teams **at a venue matching** ``venue_keys``,
    with the same recency weights as ``head_to_head_scores``.
    """
    h2h = h2h_history.filter_match_rows_to_h2h(rows, team_a, team_b)
    rel = [
        r
        for r in h2h
        if h2h_history.venue_matches_keys(str(r.get("venue") or ""), list(venue_keys or []))
    ]
    rel = h2h_history.sort_h2h_rows_recent_first(rel)[:45]
    cur_season = int(getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    w_sum = w_a = w_b = 0.0
    n_used = 0
    for row in rel:
        win = (row.get("winner") or "").strip()
        if not win:
            continue
        y = h2h_history.year_from_match_row(row)
        wt = h2h_history.recency_weight(y, cur_season)
        w_sum += wt
        n_used += 1
        if h2h_history.team_equals_label(team_a, win):
            w_a += wt
        elif h2h_history.team_equals_label(team_b, win):
            w_b += wt

    notes: list[str] = []
    if w_sum <= 0 or n_used < min_matches:
        return 50.0, 50.0, n_used, notes

    edge = (w_a - w_b) / w_sum
    sa = _clamp(50.0 + 40.0 * edge, 10.0, 90.0)
    sb = _clamp(100.0 - sa, 10.0, 90.0)
    notes.append(f"H2H at this venue (weighted n={n_used}): edge {edge:+.3f} for {team_a}")
    return sa, sb, n_used, notes


def venue_form_scores(
    team_a: str,
    team_b: str,
    venue_keys: list[str],
    rows: list[dict[str, Any]],
) -> tuple[float, float, list[str]]:
    def record_for(t: str) -> tuple[int, int]:
        w = l = 0
        for row in rows:
            if not _venue_row_matches(row.get("venue") or "", venue_keys):
                continue
            if not _team_played_here(t, row["team_a"], row["team_b"]):
                continue
            win = (row.get("winner") or "").strip()
            if not win:
                continue
            if _team_equals(t, win):
                w += 1
            elif _team_equals(t, row["team_a"]) or _team_equals(t, row["team_b"]):
                l += 1
        return w, l

    wa, la = record_for(team_a)
    wb, lb = record_for(team_b)
    notes: list[str] = []

    def pct(w: int, tot: int) -> float:
        if tot <= 0:
            return 0.5
        return w / float(tot)

    ta = wa + la
    tb = wb + lb
    if ta == 0 and tb == 0:
        return 50.0, 50.0, ["No venue-specific results in database for this ground"]
    pa = pct(wa, ta) if ta else 0.5
    pb = pct(wb, tb) if tb else 0.5
    score_a = _clamp(50.0 + 38.0 * (pa - 0.5), 10.0, 90.0)
    score_b = _clamp(50.0 + 38.0 * (pb - 0.5), 10.0, 90.0)
    if ta:
        notes.append(f"{team_a} at this venue (DB): {wa}W/{la}L")
    if tb:
        notes.append(f"{team_b} at this venue (DB): {wb}W/{lb}L")
    return score_a, score_b, notes


def xi_strength_scores(xi_a: list[_PlayerLike], xi_b: list[_PlayerLike]) -> tuple[float, float, list[str]]:
    cap = 11.0 * 1.12
    w = float(getattr(config, "WIN_ENG_XI_STRENGTH_SELECTION_SCORE_WEIGHT", 0.36))
    w = max(0.0, min(0.85, w))

    def norm_sum(xi: list[_PlayerLike]) -> float:
        if not xi:
            return 50.0
        s = 0.0
        for p in xi:
            comp = float(getattr(p, "composite", 0.5))
            sel = float(getattr(p, "selection_score", comp))
            s += (1.0 - w) * comp + w * sel
        n = max(1, len(xi))
        adj = s * (11.0 / n) if n < 11 else s
        return _clamp(100.0 * adj / cap, 5.0, 98.0)

    sa = norm_sum(xi_a)
    sb = norm_sum(xi_b)
    return sa, sb, [f"XI strength blends composite+selection_score (w={w:.2f}, {len(xi_a)}v{len(xi_b)})"]


def batting_order_scores(
    xi: list[_PlayerLike],
    order: list[str],
    *,
    is_chasing: bool,
    conditions: dict[str, Any],
) -> float:
    by_name = _xi_by_name(xi)
    weights = [1.14, 1.11, 1.08, 1.04, 1.0, 0.96, 0.93, 0.90, 0.88, 0.86, 0.84]
    num = den = 0.0
    for i, name in enumerate(order[:11]):
        p = by_name.get(name)
        if not p:
            continue
        w = weights[i] if i < len(weights) else 0.82
        num += w * p.bat_skill
        den += w
    if den <= 0:
        base = 50.0
    else:
        base = 100.0 * (num / den)

    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])
    rain = float(conditions["rain_disruption_risk"])
    hum = float((conditions.get("weather_snapshot") or {}).get("relative_humidity_pct") or 55.0)
    hum = _clamp(hum / 100.0, 0.0, 1.0)

    if is_chasing:
        base += 11.0 * dew * (1.0 - 0.35 * rain) + 4.0 * hum * dew
    else:
        base += 12.0 * (bf - 0.5) * (1.0 - 0.25 * rain) - 5.0 * rain
    return _clamp(base, 4.0, 96.0)


def bowling_phase_score(
    xi: list[_PlayerLike],
    conditions: dict[str, Any],
    *,
    bowls_second: bool,
) -> float:
    pool = _bowler_pool(xi)
    pool.sort(key=lambda p: -p.bowl_skill)
    top = pool[:7]
    if not top:
        return 45.0

    pace_bias = float(conditions["pace_bias"])
    spin_f = float(conditions["spin_friendliness"])
    swing = float(conditions["swing_seam_proxy"])
    dew = float(conditions["dew_risk"])
    bf = float(conditions["batting_friendliness"])

    pace_sorted = sorted(top, key=lambda p: -(_pace_tendency(p) * p.bowl_skill))
    pp = sum(p.bowl_skill * _pace_tendency(p) for p in pace_sorted[:3]) / 3.0
    pp *= 0.92 + 0.14 * pace_bias + 0.10 * swing

    spin_sorted = sorted(top, key=lambda p: -(_spin_tendency(p) * p.bowl_skill))
    mid = sum(p.bowl_skill * _spin_tendency(p) for p in spin_sorted[:3]) / 3.0
    mid *= 0.88 + 0.20 * spin_f * (1.1 - pace_bias)

    death_sorted = sorted(
        top,
        key=lambda p: -(p.bowl_skill * (0.48 + 0.42 * p.bat_skill + 0.12 * (1.0 - bf))),
    )
    death = sum(p.bowl_skill * (0.52 + 0.38 * p.bat_skill) for p in death_sorted[:3]) / 3.0
    if bowls_second:
        death *= 1.0 - 0.48 * dew

    raw = 0.34 * pp + 0.32 * mid + 0.34 * death
    return _clamp(raw * 100.0, 6.0, 96.0)


def matchup_scores(
    order_a: list[str],
    order_b: list[str],
    xi_a: list[_PlayerLike],
    xi_b: list[_PlayerLike],
    conditions: dict[str, Any],
) -> tuple[float, float, list[str]]:
    """Pace/spin/role proxies: top-order batters vs opponent bowling mix."""
    ba = _xi_by_name(xi_a)
    bb = _xi_by_name(xi_b)
    bowlers_a = _bowler_pool(xi_a)
    bowlers_b = _bowler_pool(xi_b)

    def side_matchup(
        order: list[str], opp_bowlers: list[_PlayerLike], self_xi: dict[str, _PlayerLike]
    ) -> float:
        if not opp_bowlers:
            return 52.0
        opp_pace = sum(p.bowl_skill * _pace_tendency(p) for p in opp_bowlers) / len(opp_bowlers)
        opp_spin = sum(p.bowl_skill * _spin_tendency(p) for p in opp_bowlers) / len(opp_bowlers)
        pace_bias = float(conditions["pace_bias"])
        spin_f = float(conditions["spin_friendliness"])
        acc = 0.0
        n = 0
        for name in order[:5]:
            p = self_xi.get(name)
            if not p:
                continue
            bspin = _batter_spin_preference(p)
            hit = p.bat_skill * (
                1.0
                + 0.28 * pace_bias * opp_pace * (1.15 - bspin)
                + 0.24 * spin_f * opp_spin * bspin
                + 0.08 * (opp_spin - opp_pace) * (bspin - 0.5)
            )
            acc += hit
            n += 1
        if n <= 0:
            return 50.0
        return _clamp(100.0 * (acc / n) * 0.95, 8.0, 94.0)

    sa = side_matchup(order_a, bowlers_b, ba)
    sb = side_matchup(order_b, bowlers_a, bb)
    notes = [
        "Matchups use role-based pace/spin proxies (upgrade slot for handedness when data exists).",
    ]
    return sa, sb, notes


def conditions_scores_for_scenario(
    *,
    a_bats_first: bool,
    conditions: dict[str, Any],
) -> tuple[float, float, str]:
    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])
    rain = float(conditions["rain_disruption_risk"])
    spin_f = float(conditions["spin_friendliness"])
    hum = float((conditions.get("weather_snapshot") or {}).get("relative_humidity_pct") or 55.0)
    hum = _clamp(hum / 100.0, 0.0, 1.0)

    if bf >= 0.70:
        pitch = "batting-friendly pitch profile"
    elif bf <= 0.52:
        pitch = "bowling-friendly pitch profile"
    else:
        pitch = "balanced pitch profile"

    pace_bias = float(conditions["pace_bias"])
    if a_bats_first:
        score_a = (
            50.0
            + 15.0 * (bf - 0.5)
            - 11.0 * rain
            - 5.0 * hum
            + 5.0 * spin_f * pace_bias * 0.15
        )
        score_b = 50.0 + 13.0 * dew * (1.0 - 0.38 * rain) + 7.0 * (1.0 - bf) * 0.2
    else:
        score_b = (
            50.0
            + 15.0 * (bf - 0.5)
            - 11.0 * rain
            - 5.0 * hum
            + 5.0 * spin_f * pace_bias * 0.15
        )
        score_a = 50.0 + 13.0 * dew * (1.0 - 0.38 * rain) + 7.0 * (1.0 - bf) * 0.2
    return _clamp(score_a, 10.0, 90.0), _clamp(score_b, 10.0, 90.0), pitch


def _prob_from_totals(total_a: float, total_b: float) -> float:
    p = (
        config.WIN_ENG_PROB_BASE
        + config.WIN_ENG_SCORE_TO_PROB_SCALE * (total_a - total_b)
    )
    return _clamp(p, config.WIN_ENG_PROB_MIN, config.WIN_ENG_PROB_MAX)


@dataclass
class ScenarioFactors:
    head_to_head: tuple[float, float] = (50.0, 50.0)
    venue: tuple[float, float] = (50.0, 50.0)
    xi: tuple[float, float] = (50.0, 50.0)
    batting: tuple[float, float] = (50.0, 50.0)
    bowling: tuple[float, float] = (50.0, 50.0)
    matchup: tuple[float, float] = (50.0, 50.0)
    conditions: tuple[float, float] = (50.0, 50.0)
    toss_role: tuple[float, float] = (50.0, 50.0)
    chase_environment: tuple[float, float] = (50.0, 50.0)


def _weighted_totals(f: ScenarioFactors) -> tuple[float, float]:
    wa = (
        config.WIN_ENG_WEIGHT_HEAD_TO_HEAD * f.head_to_head[0]
        + config.WIN_ENG_WEIGHT_VENUE * f.venue[0]
        + config.WIN_ENG_WEIGHT_XI_STRENGTH * f.xi[0]
        + config.WIN_ENG_WEIGHT_BATTING_ORDER * f.batting[0]
        + config.WIN_ENG_WEIGHT_BOWLING_PHASES * f.bowling[0]
        + config.WIN_ENG_WEIGHT_MATCHUP * f.matchup[0]
        + config.WIN_ENG_WEIGHT_CONDITIONS * f.conditions[0]
        + config.WIN_ENG_WEIGHT_TOSS_ROLE * f.toss_role[0]
        + config.WIN_ENG_WEIGHT_CHASE_ENVIRONMENT * f.chase_environment[0]
    )
    wb = (
        config.WIN_ENG_WEIGHT_HEAD_TO_HEAD * f.head_to_head[1]
        + config.WIN_ENG_WEIGHT_VENUE * f.venue[1]
        + config.WIN_ENG_WEIGHT_XI_STRENGTH * f.xi[1]
        + config.WIN_ENG_WEIGHT_BATTING_ORDER * f.batting[1]
        + config.WIN_ENG_WEIGHT_BOWLING_PHASES * f.bowling[1]
        + config.WIN_ENG_WEIGHT_MATCHUP * f.matchup[1]
        + config.WIN_ENG_WEIGHT_CONDITIONS * f.conditions[1]
        + config.WIN_ENG_WEIGHT_TOSS_ROLE * f.toss_role[1]
        + config.WIN_ENG_WEIGHT_CHASE_ENVIRONMENT * f.chase_environment[1]
    )
    return wa, wb


def _build_scenario(
    *,
    team_a: str,
    team_b: str,
    xi_a: list[_PlayerLike],
    xi_b: list[_PlayerLike],
    order_a: list[str],
    order_b: list[str],
    a_bats_first: bool,
    conditions: dict[str, Any],
    venue_keys: list[str],
    rows: list[dict[str, Any]],
    h2h: tuple[float, float, list[str]],
    venue_s: tuple[float, float, list[str]],
    xi_s: tuple[float, float, list[str]],
    matchup: tuple[float, float, list[str]],
    pitch_note: str,
    venue_chase_share: float,
    venue_chase_n: int,
    is_night_fixture: bool,
) -> tuple[float, ScenarioFactors, list[str]]:
    ca, cb, _pitch = conditions_scores_for_scenario(a_bats_first=a_bats_first, conditions=conditions)

    bat_a = batting_order_scores(xi_a, order_a, is_chasing=not a_bats_first, conditions=conditions)
    bat_b = batting_order_scores(xi_b, order_b, is_chasing=a_bats_first, conditions=conditions)

    bowl_a = bowling_phase_score(
        xi_a, conditions, bowls_second=not a_bats_first
    )
    bowl_b = bowling_phase_score(
        xi_b, conditions, bowls_second=a_bats_first
    )

    vh_a, vh_b, vh_n, vh_notes = venue_h2h_scores(team_a, team_b, venue_keys, rows)
    va0, vb0, vnotes0 = venue_s
    if vh_n >= 3 and vh_notes:
        v_blend = 0.26
        va_m = (1.0 - v_blend) * va0 + v_blend * vh_a
        vb_m = (1.0 - v_blend) * vb0 + v_blend * vh_b
        venue_s = (
            va_m,
            vb_m,
            list(vnotes0) + vh_notes + [f"Venue+H2H blend weight={v_blend:.2f} (n={vh_n})"],
        )

    tr_a, tr_b, tr_notes = toss_role_scores(
        team_a, team_b, rows, a_bats_first=a_bats_first
    )
    ce_a, ce_b, ce_notes = chase_environment_scores(
        a_bats_first=a_bats_first,
        venue_chase_share=venue_chase_share,
        venue_chase_n=venue_chase_n,
        conditions=conditions,
        is_night_fixture=is_night_fixture,
    )

    f = ScenarioFactors(
        head_to_head=(h2h[0], h2h[1]),
        venue=(venue_s[0], venue_s[1]),
        xi=(xi_s[0], xi_s[1]),
        batting=(bat_a, bat_b),
        bowling=(bowl_a, bowl_b),
        matchup=(matchup[0], matchup[1]),
        conditions=(ca, cb),
        toss_role=(tr_a, tr_b),
        chase_environment=(ce_a, ce_b),
    )
    ta, tb = _weighted_totals(f)
    p_a = _prob_from_totals(ta, tb)
    meta_notes = (
        list(h2h[2])
        + list(venue_s[2])
        + list(xi_s[2])
        + list(matchup[2])
        + list(tr_notes)
        + list(ce_notes)
    )
    meta_notes.append(pitch_note)
    return p_a, f, meta_notes


def _factor_display_name(key: str) -> str:
    return {
        "head_to_head": "head-to-head",
        "venue": "venue record",
        "xi": "XI strength",
        "batting": "batting order",
        "bowling": "phase bowling",
        "matchup": "matchups",
        "conditions": "conditions & weather",
        "toss_role": "toss scenario (chase/defend history)",
        "chase_environment": "venue chase bias & dew/night",
    }.get(key, key)


def _explanation_for_scenario(
    team_a: str,
    team_b: str,
    f: ScenarioFactors,
    *,
    a_bats_first: bool,
    pitch_note: str,
    dew: float,
    include_pitch: bool = True,
) -> str:
    """Short natural-language summary from largest weighted edges."""
    weights = {
        "head_to_head": config.WIN_ENG_WEIGHT_HEAD_TO_HEAD,
        "venue": config.WIN_ENG_WEIGHT_VENUE,
        "xi": config.WIN_ENG_WEIGHT_XI_STRENGTH,
        "batting": config.WIN_ENG_WEIGHT_BATTING_ORDER,
        "bowling": config.WIN_ENG_WEIGHT_BOWLING_PHASES,
        "matchup": config.WIN_ENG_WEIGHT_MATCHUP,
        "conditions": config.WIN_ENG_WEIGHT_CONDITIONS,
        "toss_role": config.WIN_ENG_WEIGHT_TOSS_ROLE,
        "chase_environment": config.WIN_ENG_WEIGHT_CHASE_ENVIRONMENT,
    }
    edges: list[tuple[float, str, str]] = []
    for key, w in weights.items():
        sa, sb = getattr(f, key)
        delta = w * (sa - sb)
        if abs(delta) < 0.35:
            continue
        if delta > 0:
            edges.append((delta, key, team_a))
        else:
            edges.append((abs(delta), key, team_b))
    edges.sort(reverse=True, key=lambda x: x[0])
    parts: list[str] = []
    for _mag, key, fav in edges[:3]:
        parts.append(f"{fav} edges {_factor_display_name(key)}")
    chase = team_b if a_bats_first else team_a
    if dew >= 0.55:
        parts.append(f"{chase} chasing benefits slightly from elevated dew risk ({dew:.2f})")
    if include_pitch and pitch_note:
        parts.append(pitch_note[0].upper() + pitch_note[1:])
    if not parts:
        return "Sides are closely matched across weighted factors; probability stays near coin-flip (within model clamp)."
    out = "; ".join(parts)
    if len(out) > 320:
        out = out[:317] + "..."
    return out


@dataclass
class WinEngineResult:
    team_a_name: str
    team_b_name: str
    prob_team_a_if_a_bats_first_pct: float
    prob_team_a_if_b_bats_first_pct: float
    overall_favourite: str
    explanation: str
    scenario_a_factors: dict[str, Any] = field(default_factory=dict)
    scenario_b_factors: dict[str, Any] = field(default_factory=dict)
    toss_scenario_used: str = "unknown"
    a_bats_first_selected: Optional[bool] = None
    team_a_win_pct_neutral_toss: float = 50.0
    team_a_win_pct_selected_toss: float = 50.0
    chase_defend_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        avg_a = (
            self.prob_team_a_if_a_bats_first_pct + self.prob_team_a_if_b_bats_first_pct
        ) / 2.0
        return {
            "team_a_win_pct_if_a_bats_first": round(self.prob_team_a_if_a_bats_first_pct, 2),
            "team_a_win_pct_if_b_bats_first": round(self.prob_team_a_if_b_bats_first_pct, 2),
            "team_b_win_pct_if_a_bats_first": round(
                100.0 - self.prob_team_a_if_a_bats_first_pct, 2
            ),
            "team_b_win_pct_if_b_bats_first": round(
                100.0 - self.prob_team_a_if_b_bats_first_pct, 2
            ),
            "overall_favourite": self.overall_favourite,
            "marginal_team_a_win_pct": round(self.team_a_win_pct_neutral_toss, 2),
            "team_a_win_pct_neutral_toss": round(self.team_a_win_pct_neutral_toss, 2),
            "team_a_win_pct_selected_toss": round(self.team_a_win_pct_selected_toss, 2),
            "toss_scenario_used": self.toss_scenario_used,
            "a_bats_first_selected": self.a_bats_first_selected,
            "chase_defend_context": dict(self.chase_defend_context),
            "explanation": self.explanation,
            "scenario_factors": {
                "a_bats_first": self.scenario_a_factors,
                "b_bats_first": self.scenario_b_factors,
            },
        }


def _factors_to_dict(f: ScenarioFactors) -> dict[str, Any]:
    return {
        "head_to_head": {"team_a": f.head_to_head[0], "team_b": f.head_to_head[1]},
        "venue": {"team_a": f.venue[0], "team_b": f.venue[1]},
        "xi_strength": {"team_a": f.xi[0], "team_b": f.xi[1]},
        "batting_order": {"team_a": f.batting[0], "team_b": f.batting[1]},
        "bowling_phases": {"team_a": f.bowling[0], "team_b": f.bowling[1]},
        "matchup": {"team_a": f.matchup[0], "team_b": f.matchup[1]},
        "conditions_weather": {"team_a": f.conditions[0], "team_b": f.conditions[1]},
        "toss_scenario_chase_defend": {"team_a": f.toss_role[0], "team_b": f.toss_role[1]},
        "chase_environment": {"team_a": f.chase_environment[0], "team_b": f.chase_environment[1]},
    }


def compute_win_probability(
    team_a_name: str,
    team_b_name: str,
    xi_a: list[_PlayerLike],
    xi_b: list[_PlayerLike],
    order_a: list[str],
    order_b: list[str],
    venue: VenueProfile,
    conditions: dict[str, Any],
    venue_keys: Optional[list[str]] = None,
    match_rows: Optional[list[dict[str, Any]]] = None,
    *,
    toss_scenario_key: str = "unknown",
    a_bats_first_selected: Optional[bool] = None,
    chase_share_by_venue: Optional[dict[str, tuple[float, int]]] = None,
    is_night_fixture: bool = False,
) -> WinEngineResult:
    """
    Deterministic win engine. All randomness removed — same inputs => same outputs.

    venue_keys: normalized lookup keys for venue string matching in DB rows.
    match_rows: optional pre-fetched rows to avoid double DB hits in tests.
    toss_scenario_key: label for UI/debug; ``a_bats_first_selected`` picks headline win %.
    """
    rows = match_rows if match_rows is not None else db.fetch_match_results_meta(450)
    vkeys = venue_keys if venue_keys is not None else [venue.key, venue.display_name, venue.city]
    chase_map = chase_share_by_venue if chase_share_by_venue is not None else {}
    v_share, v_n = 0.5, 0
    for vk in vkeys:
        row = chase_map.get(vk)
        if row:
            v_share, v_n = float(row[0]), int(row[1])
            break

    lab_a = ipl_teams.franchise_label_for_storage(team_a_name) or team_a_name
    lab_b = ipl_teams.franchise_label_for_storage(team_b_name) or team_b_name
    tka = ipl_teams.canonical_team_key_for_franchise(lab_a)
    tkb = ipl_teams.canonical_team_key_for_franchise(lab_b)
    h2h_raw = head_to_head_scores(team_a_name, team_b_name, rows)
    dr_h2h = db.fetch_head_to_head_derived(tka, tkb)
    h2h = _blend_h2h_with_derived_patterns(tka, tkb, h2h_raw, dr_h2h)
    venue_s = venue_form_scores(team_a_name, team_b_name, vkeys, rows)
    xi_s = xi_strength_scores(xi_a, xi_b)
    matchup = matchup_scores(order_a, order_b, xi_a, xi_b, conditions)
    pitch_note = conditions_scores_for_scenario(a_bats_first=True, conditions=conditions)[2]

    p_if_a, f_a, _n1 = _build_scenario(
        team_a=team_a_name,
        team_b=team_b_name,
        xi_a=xi_a,
        xi_b=xi_b,
        order_a=order_a,
        order_b=order_b,
        a_bats_first=True,
        conditions=conditions,
        venue_keys=vkeys,
        rows=rows,
        h2h=h2h,
        venue_s=venue_s,
        xi_s=xi_s,
        matchup=matchup,
        pitch_note=pitch_note,
        venue_chase_share=v_share,
        venue_chase_n=v_n,
        is_night_fixture=is_night_fixture,
    )
    p_if_b, f_b, _n2 = _build_scenario(
        team_a=team_a_name,
        team_b=team_b_name,
        xi_a=xi_a,
        xi_b=xi_b,
        order_a=order_a,
        order_b=order_b,
        a_bats_first=False,
        conditions=conditions,
        venue_keys=vkeys,
        rows=rows,
        h2h=h2h,
        venue_s=venue_s,
        xi_s=xi_s,
        matchup=matchup,
        pitch_note=pitch_note,
        venue_chase_share=v_share,
        venue_chase_n=v_n,
        is_night_fixture=is_night_fixture,
    )

    neutral = (p_if_a + p_if_b) / 2.0
    if a_bats_first_selected is True:
        selected = p_if_a
    elif a_bats_first_selected is False:
        selected = p_if_b
    else:
        selected = neutral

    if neutral > 52.5:
        fav = team_a_name
    elif neutral < 47.5:
        fav = team_b_name
    else:
        fav = "Evenly matched"

    dew = float(conditions["dew_risk"])
    ex1 = _explanation_for_scenario(
        team_a_name, team_b_name, f_a, a_bats_first=True, pitch_note=pitch_note, dew=dew
    )
    ex2 = _explanation_for_scenario(
        team_a_name,
        team_b_name,
        f_b,
        a_bats_first=False,
        pitch_note=pitch_note,
        dew=dew,
        include_pitch=False,
    )
    explanation = f"If {team_a_name} bats first: {ex1} If {team_b_name} bats first: {ex2}"

    cd_ctx = build_chase_defend_context(
        team_a_name,
        team_b_name,
        rows,
        vkeys,
        chase_map,
        conditions,
        is_night_fixture=is_night_fixture,
    )
    cd_ctx["chase_bias_applied_summary"] = (
        f"Venue chase share {v_share:.2f} (n={v_n}), dew {dew:.2f}, "
        f"night={int(is_night_fixture)} — blended in chase_environment factor"
    )

    return WinEngineResult(
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        prob_team_a_if_a_bats_first_pct=p_if_a,
        prob_team_a_if_b_bats_first_pct=p_if_b,
        overall_favourite=fav,
        explanation=explanation,
        scenario_a_factors=_factors_to_dict(f_a),
        scenario_b_factors=_factors_to_dict(f_b),
        toss_scenario_used=toss_scenario_key,
        a_bats_first_selected=a_bats_first_selected,
        team_a_win_pct_neutral_toss=neutral,
        team_a_win_pct_selected_toss=selected,
        chase_defend_context=cd_ctx,
    )
