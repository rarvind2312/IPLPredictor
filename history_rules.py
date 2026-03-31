"""
Explainable, rule-based adjustments from historical match data (no ML).

Each function returns (delta, reason_string) where delta is a small contribution
in [-1, 1] before global cap/weighting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import config


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def xi_frequency_term(
    player_xi_count: int,
    ref_max_xi_picks: int,
    db_match_count: int,
) -> Tuple[float, str]:
    """More actual XIs relative to busiest player in DB → slight positive."""
    if db_match_count <= 0 or ref_max_xi_picks <= 0:
        return 0.0, "XI frequency: no reference matches"
    ratio = player_xi_count / max(1, ref_max_xi_picks)
    d = _clamp((ratio ** 0.65) - 0.42)
    return d, f"XI frequency: {player_xi_count} XIs vs ref max {ref_max_xi_picks} (DB matches {db_match_count}) → {d:+.2f}"


def batting_slot_term(
    avg_slot: Optional[float],
    proposed_slot: Optional[int],
    samples: int,
) -> Tuple[float, str]:
    """If we know typical slot and proposed order slot, reward alignment."""
    if avg_slot is None or proposed_slot is None or samples < config.LEARN_MIN_SAMPLES_SLOT:
        return 0.0, "Batting slot: insufficient history"
    diff = abs(float(avg_slot) - float(proposed_slot))
    # Within 1.5 positions = good
    d = _clamp((1.5 - diff) / 1.5 * 0.9)
    return d, f"Batting slot: hist avg {avg_slot:.1f} vs proposed {proposed_slot} (n={samples}) → {d:+.2f}"


def bowling_usage_term(
    avg_balls_when_bowls: float,
    role: str,
    samples: int,
) -> Tuple[float, str]:
    """Frontline bowlers (high balls/match when used) get a small bump in bowling-heavy reads."""
    if samples < config.LEARN_MIN_SAMPLES_BOWL:
        return 0.0, "Bowling usage: insufficient history"
    # T20: 24+ balls often = frontliner
    if role in ("bowl", "all"):
        d = _clamp((avg_balls_when_bowls - 18.0) / 30.0)
        return d, f"Bowling usage: ~{avg_balls_when_bowls:.0f} balls/match when used (n={samples}) → {d:+.2f}"
    d = _clamp((avg_balls_when_bowls - 12.0) / 40.0) * 0.45
    return d, f"Bowling usage (part-time): ~{avg_balls_when_bowls:.0f} balls/match → {d:+.2f}"


def venue_team_xi_term(
    picks_at_venue: int,
    venue_team_matches: int,
) -> Tuple[float, str]:
    """Picked often at this ground for this franchise label."""
    if venue_team_matches < config.LEARN_MIN_SAMPLES_VENUE_TEAM:
        return 0.0, "Venue+team XI: insufficient local samples"
    rate = picks_at_venue / max(1, venue_team_matches)
    d = _clamp((rate - 0.35) * 2.2)
    return d, f"Venue habit: {picks_at_venue}/{venue_team_matches} XIs here → {d:+.2f}"


def overseas_mix_term(
    proposed_os: int,
    mix_counts: dict[int, int],
) -> Tuple[float, str]:
    """Align with historically observed overseas counts at this venue for this team."""
    total = sum(mix_counts.values())
    if total < config.LEARN_MIN_SAMPLES_OVERSEAS_MIX:
        return 0.0, "Overseas mix: no/little history"
    probs = {k: v / total for k, v in mix_counts.items()}
    best_k = max(probs, key=lambda x: probs[x])
    p_this = probs.get(proposed_os, 1e-6)
    p_best = probs[best_k]
    # Prefer matching high-probability buckets; small nudge if exact match
    d = _clamp((p_this / p_best) - 0.55) if p_best > 0 else 0.0
    if proposed_os == best_k:
        d = max(d, 0.15)
    return d, f"Overseas mix: proposing {proposed_os} os vs hist mode {best_k} (p={p_best:.2f}) → {d:+.2f}"


def night_day_term(
    is_night: bool,
    night_xi: int,
    day_xi: int,
) -> Tuple[float, str]:
    """Relative XI rate night vs day for this player."""
    labelled = night_xi + day_xi
    if labelled < config.LEARN_MIN_SAMPLES_DAY_NIGHT:
        return 0.0, "Day/night: insufficient labelled matches"
    p_n = night_xi / max(1, labelled)
    if is_night:
        d = _clamp((p_n - 0.5) * 2.0)
        return d, f"Night match: historical night-XI share {p_n:.2f} → {d:+.2f}"
    d = _clamp(((1 - p_n) - 0.5) * 2.0)
    return d, f"Day match: historical day-XI share {1-p_n:.2f} → {d:+.2f}"


def dew_context_term(
    dew_risk: float,
    player_night_xi: int,
    player_day_xi: int,
    role: str,
) -> Tuple[float, str]:
    """
    Under dew, slight preference for players who have been selected more in night games
    (proxy for trust in chase/bowl-at-death — explainable heuristic).
    """
    n = player_night_xi + player_day_xi
    if n < 2 or dew_risk < config.LEARN_DEW_RISK_THRESHOLD:
        return 0.0, "Dew context: low dew or thin history"
    night_share = player_night_xi / max(1, n)
    base = (night_share - 0.5) * 1.6 * dew_risk
    if role in ("bowl", "all"):
        base *= 1.15
    d = _clamp(base)
    return d, f"Dew {dew_risk:.2f}: night-XI share {night_share:.2f}, role {role} → {d:+.2f}"


def chase_bias_team_term(
    venue_chase_win_share: float,
    sample_total: int,
) -> Tuple[float, str]:
    """
    Venue-level chasing success (bowl-first wins / all wins with known toss path).
    Used at team strength / win-prob layer, not per player.
    """
    if sample_total < config.LEARN_MIN_SAMPLES_CHASE:
        return 0.0, "Chase bias: thin venue sample"
    # chase_win_share = bowl_first wins / (bat_first + bowl_first wins)
    d = _clamp((venue_chase_win_share - 0.5) * 1.2)
    return d, f"Venue chase prior: bowl-first win share {venue_chase_win_share:.2f} (n≈{sample_total}) → {d:+.2f}"


def blend_history_deltas(
    terms: list[Tuple[float, str]],
    weights: list[float],
) -> Tuple[float, list[str]]:
    """Weighted average of term deltas, then scaled by global cap."""
    if not terms:
        return 0.0, []
    w = weights[: len(terms)] + [0.0] * max(0, len(terms) - len(weights))
    wsum = sum(w) or 1.0
    blended = sum(terms[i][0] * w[i] for i in range(len(terms))) / wsum
    lines = [terms[i][1] for i in range(len(terms)) if abs(terms[i][0]) > 0.02 or "insufficient" not in terms[i][1]]
    out = _clamp(
        blended * config.HISTORY_BLEND_SCALE,
        -config.HISTORY_ADJ_PER_PLAYER_CAP,
        config.HISTORY_ADJ_PER_PLAYER_CAP,
    )
    return out, lines
