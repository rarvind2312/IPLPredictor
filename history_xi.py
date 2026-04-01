"""
IPL-only: primary XI and batting-order signals from stored scorecards (`matches` / `team_match_xi`).

Squad lists define eligibility only; this module produces history_xi_score and batting slot EMAs.

This module reads **only local SQLite** (`matches`, ``team_match_xi``, ``team_match_summary``).
Populate those tables in the **ingest** stage (local Cricsheet archive and/or manual **Parse & store**).
Prediction does not pull history from the internet or parse Cricsheet JSON on the fly.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, FrozenSet, Optional

import config
import db
import first_choice_prior
import h2h_history
import history_linkage
import ipl_teams
import learner
import selection_model

logger = logging.getLogger(__name__)


def _year_from_match_date(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(20\d{2})", str(s))
    return int(m.group(1)) if m else None


def _team_chased_row(row: dict[str, Any], team_display: str) -> Optional[bool]:
    bf = (row.get("batting_first") or "").strip().lower()
    if not bf:
        return None
    t = team_display.strip().lower()
    if bf == t:
        return False
    ta = (row.get("team_a") or "").strip().lower()
    tb = (row.get("team_b") or "").strip().lower()
    if t and t in (ta, tb) and bf != t:
        return True
    return None


def _overseas_count_for_team_row(row: dict[str, Any], team_display: str) -> Optional[int]:
    ta = (row.get("team_a") or "").strip()
    tb = (row.get("team_b") or "").strip()
    t = team_display.strip()
    if ta and t.lower() == ta.lower():
        v = row.get("overseas_team_a")
    elif tb and t.lower() == tb.lower():
        v = row.get("overseas_team_b")
    else:
        return None
    if v is None:
        return None
    try:
        return max(0, min(11, int(float(v))))
    except (TypeError, ValueError):
        return None


def _venue_matches_key(venue_raw: str, venue_keys: list[str]) -> bool:
    if not venue_raw or not venue_keys:
        return False
    vn = learner.normalize_player_key(str(venue_raw))[:80]
    rv = str(venue_raw).lower()
    for vk in venue_keys:
        if not vk:
            continue
        if vk in vn or vn in vk:
            return True
        if vk.lower() in rv or rv in vk.lower():
            return True
    return False


def _rows_for_player(all_rows: list[dict[str, Any]], player_key: str) -> list[dict[str, Any]]:
    """
    Rows for a resolved SQLite ``player_key`` only.

    We intentionally **do not** fall back to matching ``player_name`` text: that can bind the wrong
    person when abbreviated Cricsheet names collide. Squad display names stay on ``SquadPlayer.name``.
    """
    pk = (player_key or "").strip()
    if not pk:
        return []
    return [r for r in all_rows if r.get("player_key") == pk]


def _distinct_match_ids(all_rows: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for r in all_rows:
        mid = int(r["match_id"])
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def _recent_xi_rate(player_key: str, all_rows: list[dict[str, Any]], n: int = 5) -> float:
    if not (player_key or "").strip():
        return 0.0
    mids = _distinct_match_ids(all_rows)[:n]
    if not mids:
        return 0.0
    hit = 0
    for mid in mids:
        if any(r["match_id"] == mid and r["player_key"] == player_key for r in all_rows):
            hit += 1
    return hit / float(len(mids))


def _venue_xi_rate(player_key: str, all_rows: list[dict[str, Any]], venue_keys: list[str]) -> float:
    if not (player_key or "").strip():
        return 0.0
    rel = [r for r in all_rows if _venue_matches_key(str(r.get("venue") or ""), venue_keys)]
    mids: list[int] = []
    seen: set[int] = set()
    for r in rel:
        mid = int(r["match_id"])
        if mid not in seen:
            seen.add(mid)
            mids.append(mid)
    if not mids:
        return 0.0
    hit = sum(
        1
        for mid in mids
        if any(r["match_id"] == mid and r["player_key"] == player_key for r in rel)
    )
    return hit / float(len(mids))


def _prior_season_xi_rate(
    player_key: str,
    all_rows: list[dict[str, Any]],
    current_season: int,
) -> float:
    if not (player_key or "").strip():
        return 0.0
    rel = [
        r
        for r in all_rows
        if (_year_from_match_date(r.get("match_date")) or 0) < current_season
    ]
    mids: list[int] = []
    seen: set[int] = set()
    for r in rel:
        mid = int(r["match_id"])
        if mid not in seen:
            seen.add(mid)
            mids.append(mid)
    if not mids:
        return 0.0
    hit = sum(
        1
        for mid in mids
        if any(r["match_id"] == mid and r["player_key"] == player_key for r in rel)
    )
    return hit / float(len(mids))


def _overseas_pattern_xi_rate(
    player_key: str,
    all_rows: list[dict[str, Any]],
    team_display: str,
    guess: int,
) -> float:
    if not (player_key or "").strip():
        return 0.0
    rel = []
    for r in all_rows:
        oc = _overseas_count_for_team_row(r, team_display)
        if oc is None:
            continue
        if abs(int(oc) - int(guess)) <= 1:
            rel.append(r)
    mids: list[int] = []
    seen: set[int] = set()
    for r in rel:
        mid = int(r["match_id"])
        if mid not in seen:
            seen.add(mid)
            mids.append(mid)
    if not mids:
        return 0.0
    hit = sum(
        1
        for mid in mids
        if any(r["match_id"] == mid and r["player_key"] == player_key for r in rel)
    )
    return hit / float(len(mids))


def _chase_defend_xi_rate(
    player_key: str,
    all_rows: list[dict[str, Any]],
    team_display: str,
    chase_fixture: Optional[bool],
) -> float:
    if not (player_key or "").strip():
        return 0.0
    if chase_fixture is None:
        return 0.0
    rel = []
    for r in all_rows:
        ch = _team_chased_row(r, team_display)
        if ch is None:
            continue
        if ch == chase_fixture:
            rel.append(r)
    mids: list[int] = []
    seen: set[int] = set()
    for r in rel:
        mid = int(r["match_id"])
        if mid not in seen:
            seen.add(mid)
            mids.append(mid)
    if not mids:
        return 0.0
    hit = sum(
        1
        for mid in mids
        if any(r["match_id"] == mid and r["player_key"] == player_key for r in rel)
    )
    return hit / float(len(mids))


def _bowl_usage_rate(player_key: str, all_rows: list[dict[str, Any]], last_n_matches: int = 12) -> float:
    if not (player_key or "").strip():
        return 0.0
    mids = _distinct_match_ids(all_rows)[:last_n_matches]
    if not mids:
        return 0.0
    hit = 0
    for mid in mids:
        for r in all_rows:
            if r["match_id"] != mid or r["player_key"] != player_key:
                continue
            try:
                ov = float(r.get("overs_bowled") or 0)
            except (TypeError, ValueError):
                ov = 0.0
            if ov > 0:
                hit += 1
                break
    return hit / float(len(mids))


def _batting_slot_series(
    player_rows_chrono: list[dict[str, Any]],
) -> list[float]:
    """Only real scorecard / ball-by-ball slots — not XI list index."""
    vals: list[float] = []
    for r in player_rows_chrono:
        pos = r.get("batting_position")
        if pos is None:
            continue
        try:
            vals.append(float(pos))
        except (TypeError, ValueError):
            continue
    return vals


def _ema_slots(vals_chrono_oldest_first: list[float]) -> float:
    if not vals_chrono_oldest_first:
        return config.HISTORY_BAT_SLOT_UNKNOWN
    a = float(config.HISTORY_BAT_SLOT_EMA_ALPHA)
    ema = vals_chrono_oldest_first[0]
    for v in vals_chrono_oldest_first[1:]:
        ema = a * float(v) + (1.0 - a) * ema
    return float(ema)


def _weighted_h2h_batting_ema(
    rows_chrono_oldest_first: list[dict[str, Any]],
    cur_season: int,
) -> Optional[float]:
    """EMA over batting positions in H2H matches, with recency-scaled alpha per step."""
    pairs: list[tuple[float, float]] = []
    for r in rows_chrono_oldest_first:
        pos = r.get("batting_position")
        if pos is None:
            continue
        try:
            p = float(pos)
        except (TypeError, ValueError):
            continue
        y = _year_from_match_date(r.get("match_date"))
        w = h2h_history.recency_weight(y, cur_season)
        pairs.append((p, w))
    if len(pairs) < int(config.HISTORY_MIN_SAMPLES_BAT_ORDER):
        return None
    alpha_base = float(config.HISTORY_BAT_SLOT_EMA_ALPHA)
    ema = pairs[0][0]
    for p, wt in pairs[1:]:
        a = min(0.55, alpha_base * (0.62 + 0.38 * wt))
        ema = a * p + (1.0 - a) * ema
    return float(ema)


def _h2h_weighted_xi_rate_for_player(
    pr_chrono_oldest_first: list[dict[str, Any]],
    h2h_fixtures_newest_first: list[dict[str, Any]],
    cur_season: int,
) -> tuple[float, int]:
    """Recency-weighted fraction of H2H fixtures where this player has a stored XI row."""
    pr_mids = {int(r["match_id"]) for r in pr_chrono_oldest_first}
    num = den = 0.0
    n_considered = 0
    for fx in h2h_fixtures_newest_first:
        mid = int(fx["match_id"])
        y = h2h_history.year_from_match_row(fx)
        w = h2h_history.recency_weight(y, cur_season)
        den += w
        n_considered += 1
        if mid in pr_mids:
            num += w
    if den <= 0:
        return 0.0, n_considered
    return num / den, n_considered


def attach_primary_history_to_squad(
    players: list[Any],
    team_name: str,
    venue_keys: list[str],
    *,
    shape: dict[str, float],
    chase_context: Optional[bool] = None,
    opponent_canonical_label: Optional[str] = None,
    h2h_explain: Optional[dict[str, Any]] = None,
    h2h_explain_scope: Optional[str] = None,
    fetched_team_slug: str = "",
    selected_fetched_player_keys: Optional[FrozenSet[str]] = None,
    opposite_fetched_player_keys: Optional[FrozenSet[str]] = None,
    stale_cached_player_keys: Optional[FrozenSet[str]] = None,
    captain_display_name: str = "",
    wicketkeeper_display_name: str = "",
) -> None:
    """
    Mutates **current-squad** players only: history_xi_score, history_batting_ema, history_debug.

    History rows are loaded for **one franchise** (canonical label + strict SQL + Python filter).
    **No players are added** and no history signals are applied to names outside ``players``:
    the loop runs only over the fetched squad list (strict squad scope).
    """
    squad_player_keys = {
        learner.normalize_player_key(getattr(p, "name", "") or "")
        for p in players
    }
    squad_player_keys.discard("")

    canon_label = ipl_teams.franchise_label_for_storage(team_name) or (team_name or "").strip()
    canonical_key = ipl_teams.canonical_team_key_for_franchise(canon_label)
    all_rows = (
        db.history_team_xi_rows_for_franchise(canon_label, limit=650) if canon_label else []
    )
    cur_season = int(getattr(config, "IPL_CURRENT_SEASON_YEAR", 2026))
    os_guess = max(
        0,
        min(
            config.MAX_OVERSEAS,
            int(round(float(shape.get("overseas_density", 0)) * 11 + 0.001)),
        ),
    )

    opp = (
        ipl_teams.franchise_label_for_storage(opponent_canonical_label)
        or (opponent_canonical_label or "").strip()
        if opponent_canonical_label
        else ""
    )
    h2h_fixtures: list[dict[str, Any]] = []
    if opp and canon_label:
        h2h_fixtures = db.h2h_fixtures_between_franchises(canon_label, opp, limit=120)
    h2h_ids: set[int] = {int(x["match_id"]) for x in h2h_fixtures}
    h2h_venue_fx = h2h_history.sort_h2h_rows_recent_first(
        [
            x
            for x in h2h_fixtures
            if h2h_history.venue_matches_keys(str(x.get("venue") or ""), venue_keys)
        ]
    )
    all_rows_h2h = (
        [r for r in all_rows if int(r["match_id"]) in h2h_ids] if h2h_ids else []
    )

    _link_pkg = history_linkage.link_current_squad_to_history(
        players,
        team_name,
        opponent_canonical_label=opponent_canonical_label,
    )
    _link_by_pk = {r["canonical_player_key"]: r for r in _link_pkg["per_player"]}
    _link_summary = _link_pkg["summary"]

    _global_qkeys: list[str] = []
    for _p in players:
        _pk0 = learner.normalize_player_key(getattr(_p, "name", "") or "")
        if not _pk0:
            continue
        _row0 = _link_by_pk.get(_pk0, {})
        _hk0 = str(_row0.get("history_lookup_key") or "").strip()
        _grk0 = str(_row0.get("global_resolved_history_key") or "").strip()
        _global_qkeys.append(_hk0 or _grk0 or _pk0)
    _uq_global = list(dict.fromkeys([k for k in _global_qkeys if k]))
    _glob_xi = db.batch_global_team_match_xi_stats(_uq_global)
    _glob_pbp = db.batch_global_player_batting_slot_ema(_uq_global)
    _glob_prof = db.batch_global_player_profile_aggregates(_uq_global)
    _glob_other_fr = db.batch_player_other_franchise_tmx_counts(_uq_global, canonical_key)

    bat_h2h_players = 0
    xi_h2h_signal_players = 0
    phase_h2h_players = 0

    logger.info(
        "history_xi: franchise=%s canonical_key=%s strict_rows=%d squad_size=%d squad_keys=%d "
        "venue_keys=%s chase_ctx=%s",
        canon_label,
        canonical_key,
        len(all_rows),
        len(players),
        len(squad_player_keys),
        venue_keys[:3],
        chase_context,
    )

    _sel_fetched = selected_fetched_player_keys
    _opp_fetched = opposite_fetched_player_keys or frozenset()
    _stale_keys = stale_cached_player_keys or frozenset()
    _fetch_slug = (fetched_team_slug or "").strip()

    _player_rows_prep: list[tuple[Any, str, str, list[dict[str, Any]], dict[str, Any]]] = []
    for p in players:
        pk = learner.normalize_player_key(p.name)
        if not pk:
            continue
        _lk = _link_by_pk.get(pk, {})
        hk = _lk.get("history_lookup_key")
        sql_key = str(hk).strip() if hk else ""
        pr = _rows_for_player(all_rows, sql_key) if sql_key else []
        _player_rows_prep.append((p, pk, sql_key, pr, _lk))

    _sk_mids: dict[str, set[int]] = defaultdict(set)
    for _p, _pk, sql_key, pr, _lk in _player_rows_prep:
        if sql_key and pr:
            _sk_mids[sql_key].update(int(r["match_id"]) for r in pr)
    _batch_pbp = db.batch_fetch_primary_pbp_slots_for_franchise(
        canon_label, {k: sorted(v) for k, v in _sk_mids.items()}
    )
    _fr_keys = list(dict.fromkeys([t[2] for t in _player_rows_prep if t[2]]))
    _fr_batch = db.batch_get_player_franchise_features(_fr_keys, canonical_key)
    _missing_phase_keys: list[str] = []
    _missing_spin_keys: list[str] = []
    for _sql_key in _fr_keys:
        _fr = _fr_batch.get(_sql_key)
        has_phase = bool(_fr) and (
            float(_fr.get("pp_overs_bowled") or 0)
            + float(_fr.get("middle_overs_bowled") or 0)
            + float(_fr.get("death_overs_bowled") or 0)
        ) > 0.01
        has_spin = bool(_fr) and (int(_fr.get("vs_spin_balls") or 0) + int(_fr.get("vs_pace_balls") or 0)) > 0
        if not has_phase:
            _missing_phase_keys.append(_sql_key)
        if not has_spin:
            _missing_spin_keys.append(_sql_key)
    _phase_rates_batch = db.batch_player_phase_bowl_rates(_missing_phase_keys, canonical_key, limit_matches=40)
    _spin_pace_batch = db.batch_player_spin_pace_faced_share(_missing_spin_keys, canonical_key, limit_rows=80)
    _all_mid_order = _distinct_match_ids(all_rows)
    _recent5_mid_set = set(_all_mid_order[:5])
    _prior_mid_set: set[int] = set()
    _venue_mid_set: set[int] = set()
    _overseas_mid_set: set[int] = set()
    _chase_mid_set: set[int] = set()
    _mid_team_chased: dict[int, Optional[bool]] = {}
    _mid_overseas_n: dict[int, Optional[int]] = {}
    for r in all_rows:
        mid = int(r["match_id"])
        if _venue_matches_key(str(r.get("venue") or ""), venue_keys):
            _venue_mid_set.add(mid)
        y = _year_from_match_date(r.get("match_date"))
        if y is not None and y < cur_season:
            _prior_mid_set.add(mid)
        if mid not in _mid_overseas_n:
            _mid_overseas_n[mid] = _overseas_count_for_team_row(r, canon_label)
            oc = _mid_overseas_n[mid]
            if oc is not None and abs(int(oc) - os_guess) <= 1:
                _overseas_mid_set.add(mid)
        if mid not in _mid_team_chased:
            _mid_team_chased[mid] = _team_chased_row(r, canon_label)
            ch = _mid_team_chased[mid]
            if chase_context is not None and ch is not None and ch == chase_context:
                _chase_mid_set.add(mid)
    _h2h_mid_order = [m for m in _all_mid_order if m in h2h_ids]
    _h2h_mid_set = set(_h2h_mid_order)
    _h2h_overseas_mid_set = _overseas_mid_set & _h2h_mid_set
    _h2h_chase_mid_set = _chase_mid_set & _h2h_mid_set
    _h2h_bw_cap = max(3, min(12, max(1, len(h2h_ids))))
    _h2h_bw_mids = set(_h2h_mid_order[:_h2h_bw_cap])

    def _rate(player_mids: set[int], base_mids: set[int]) -> float:
        if not player_mids or not base_mids:
            return 0.0
        return len(player_mids & base_mids) / float(len(base_mids))

    def _usage_rate(player_mids: set[int], ordered_mids: list[int], n: int) -> float:
        mids = ordered_mids[:n]
        if not player_mids or not mids:
            return 0.0
        return sum(1 for m in mids if m in player_mids) / float(len(mids))

    for p, pk, sql_key, pr, _lk in _player_rows_prep:
        if not _sel_fetched:
            in_selected_team_fetched_squad = True
        else:
            in_selected_team_fetched_squad = pk in _sel_fetched
        in_opposite_team_fetched_squad = pk in _opp_fetched
        stale_cached_entry_detected = pk in _stale_keys
        wrong_side_squad_assignment = (not in_selected_team_fetched_squad) and (
            in_opposite_team_fetched_squad or stale_cached_entry_detected
        )
        grk_link = str(_lk.get("global_resolved_history_key") or "").strip()
        qk_global = sql_key or grk_link or pk
        mids = sorted({int(r["match_id"]) for r in pr})
        player_mid_set = _sk_mids.get(sql_key, set()) if sql_key else set()
        pbp_full = _batch_pbp.get(sql_key, {}) if sql_key else {}
        pbp_by_mid = {m: pbp_full[m] for m in mids if m in pbp_full}
        pr_enriched: list[dict[str, Any]] = []
        for r in pr:
            rd = dict(r)
            if rd.get("batting_position") is None:
                alt = pbp_by_mid.get(int(rd["match_id"]))
                if alt is not None:
                    rd["batting_position"] = float(alt)
            pr_enriched.append(rd)
        pr = pr_enriched
        # chronological oldest first for EMA
        pr_chrono = list(reversed(pr))

        cur_season_rows = [
            r
            for r in pr_chrono
            if (_year_from_match_date(r.get("match_date")) or 0) == cur_season
        ]
        prior_rows = [
            r
            for r in pr_chrono
            if (_year_from_match_date(r.get("match_date")) or 0) < cur_season
        ]

        slots_cur = _batting_slot_series(cur_season_rows)
        slots_prior = _batting_slot_series(prior_rows)
        slot_source = "role_fallback"
        if len(slots_cur) >= int(config.HISTORY_MIN_SAMPLES_BAT_ORDER):
            ema = _ema_slots(slots_cur)
            slot_source = "historical_ema_current_season"
        elif len(slots_prior) >= int(config.HISTORY_MIN_SAMPLES_BAT_ORDER):
            ema = _ema_slots(slots_prior)
            slot_source = "historical_ema_prior_season"
        elif len(_batting_slot_series(pr_chrono)) >= 1:
            ema = _ema_slots(_batting_slot_series(pr_chrono))
            slot_source = "historical_ema_all_seasons_sparse"
        else:
            ema = float(config.HISTORY_BAT_SLOT_UNKNOWN)
            slot_source = "role_fallback"

        fr = _fr_batch.get(sql_key) if sql_key else None
        if fr and int(fr.get("batting_slot_samples") or 0) >= int(config.HISTORY_MIN_SAMPLES_BAT_ORDER):
            fe_raw = fr.get("batting_position_ema")
            if fe_raw is not None:
                try:
                    fe = float(fe_raw)
                    if 0 < fe <= 15:
                        unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
                        if ema >= unk - 1e-6:
                            ema = fe
                            slot_source = "cricsheet_ballbyball_ema"
                        else:
                            ema = 0.52 * ema + 0.48 * fe
                            slot_source = f"{slot_source}_cricsheet_blend"
                except (TypeError, ValueError):
                    pass

        h2h_pr = [r for r in pr_chrono if int(r["match_id"]) in h2h_ids]
        h2h_venue_mids = {int(x["match_id"]) for x in h2h_venue_fx}
        h2h_v_pr = [r for r in pr_chrono if int(r["match_id"]) in h2h_venue_mids]
        h2h_used_bat = False
        h2h_used_venue_slot = False
        ema_h2h = _weighted_h2h_batting_ema(h2h_pr, cur_season) if h2h_ids else None
        if ema_h2h is not None and len(h2h_fixtures) >= int(config.HISTORY_MIN_SAMPLES_BAT_ORDER):
            cap = float(config.HISTORY_BAT_ORDER_H2H_BLEND_MAX)
            per = float(config.HISTORY_BAT_ORDER_H2H_BLEND_PER_MATCH)
            w_h = min(cap, per * float(len(h2h_fixtures)))
            unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
            if ema >= unk - 1e-6:
                ema = float(ema_h2h)
                slot_source = f"h2h_opponent_primary_{slot_source}"
            else:
                ema = (1.0 - w_h) * float(ema) + w_h * float(ema_h2h)
                slot_source = f"{slot_source}_h2h_opponent_blend"
            h2h_used_bat = True
        ema_h2h_venue = (
            _weighted_h2h_batting_ema(h2h_v_pr, cur_season)
            if len(h2h_venue_fx) >= 3
            else None
        )
        if ema_h2h_venue is not None and h2h_venue_mids:
            w_v = min(0.38, 0.07 * float(len(h2h_venue_fx)))
            ema = (1.0 - w_v) * float(ema) + w_v * float(ema_h2h_venue)
            slot_source = f"{slot_source}_h2h_venue_slot"
            h2h_used_venue_slot = True
        if h2h_used_bat or h2h_used_venue_slot:
            bat_h2h_players += 1

        r5 = _rate(player_mid_set, _recent5_mid_set)
        vnr = _rate(player_mid_set, _venue_mid_set)
        ps = _rate(player_mid_set, _prior_mid_set)
        op = _rate(player_mid_set, _overseas_mid_set)
        cd = _rate(player_mid_set, _chase_mid_set)
        bw = _usage_rate(player_mid_set, _all_mid_order, 12)

        op_h = _rate(player_mid_set, _h2h_overseas_mid_set) if h2h_ids else 0.0
        cd_h = _rate(player_mid_set, _h2h_chase_mid_set) if h2h_ids else 0.0
        bw_h = _rate(player_mid_set, _h2h_bw_mids) if h2h_ids else 0.0
        if len(h2h_fixtures) >= 2 and h2h_ids:
            hb = min(1.0, 0.30 + 0.07 * float(len(h2h_fixtures)))
            op = (1.0 - hb) * op + hb * op_h
            cd = (1.0 - hb) * cd + hb * cd_h
            bw = (1.0 - 0.88 * hb) * bw + 0.88 * hb * bw_h
        elif len(h2h_fixtures) == 1 and h2h_ids:
            op = 0.76 * op + 0.24 * op_h
            cd = 0.76 * cd + 0.24 * cd_h
            bw = 0.72 * bw + 0.28 * bw_h

        if fr and (
            float(fr.get("pp_overs_bowled") or 0)
            + float(fr.get("middle_overs_bowled") or 0)
            + float(fr.get("death_overs_bowled") or 0)
        ) > 0.01:
            ph_rates = {
                "powerplay": float(fr.get("phase_bowl_rate_pp") or 0),
                "middle": float(fr.get("phase_bowl_rate_middle") or 0),
                "death": float(fr.get("phase_bowl_rate_death") or 0),
            }
        else:
            ph_rates = _phase_rates_batch.get(sql_key, {"powerplay": 0.0, "middle": 0.0, "death": 0.0})
        ph_blend = (
            0.34 * float(ph_rates.get("powerplay", 0.0))
            + 0.33 * float(ph_rates.get("middle", 0.0))
            + 0.33 * float(ph_rates.get("death", 0.0))
        )
        ph_h2h: dict[str, float] = {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
        ph_mix = ph_blend
        if h2h_ids and len(h2h_ids) >= 2:
            mids_list = sorted(h2h_ids, reverse=True)[: min(40, len(h2h_ids))]
            ph_h2h = (
                db.player_phase_bowl_rates_in_match_ids(sql_key, canonical_key, mids_list)
                if sql_key
                else {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
            )
            ph_blend_h = (
                0.34 * float(ph_h2h.get("powerplay", 0.0))
                + 0.33 * float(ph_h2h.get("middle", 0.0))
                + 0.33 * float(ph_h2h.get("death", 0.0))
            )
            m_ph = 0.44 + 0.14 * min(1.0, float(len(h2h_fixtures)) / 8.0)
            ph_mix = (1.0 - m_ph) * ph_blend + m_ph * ph_blend_h
            if abs(ph_mix - ph_blend) > 1e-4:
                phase_h2h_players += 1

        if fr and (int(fr.get("vs_spin_balls") or 0) + int(fr.get("vs_pace_balls") or 0)) > 0:
            spin_pace = {
                "spin_share": float(fr.get("vs_spin_tendency") or 0),
                "pace_share": float(fr.get("vs_pace_tendency") or 0),
                "samples": int(fr.get("vs_spin_balls") or 0) + int(fr.get("vs_pace_balls") or 0),
            }
        else:
            spin_pace = _spin_pace_batch.get(sql_key, {"spin_share": 0.0, "pace_share": 0.0, "samples": 0})

        bat_agg = float(fr.get("batting_aggressor_score") or 0) if fr else 0.0
        bowl_ctrl = float(fr.get("bowling_control_score") or 0) if fr else 0.0

        h2h_xi_rate, _h2h_fx_n = _h2h_weighted_xi_rate_for_player(
            pr_chrono, h2h_fixtures, cur_season
        )
        h2h_v_rate, _h2h_vn = (
            _h2h_weighted_xi_rate_for_player(pr_chrono, h2h_venue_fx, cur_season)
            if h2h_venue_fx
            else (0.0, 0)
        )
        if h2h_fixtures and h2h_xi_rate > 1e-5:
            xi_h2h_signal_players += 1

        w_h2h_xi = float(getattr(config, "HISTORY_XI_W_H2H_XI", 0.0))
        w_h2h_v = float(getattr(config, "HISTORY_XI_W_H2H_VENUE", 0.0))
        hx_base = (
            float(config.HISTORY_XI_W_RECENT5) * r5
            + float(config.HISTORY_XI_W_VENUE) * vnr
            + float(config.HISTORY_XI_W_PRIOR_SEASON) * ps
            + float(config.HISTORY_XI_W_OVERSEAS_PATTERN) * op
            + float(config.HISTORY_XI_W_CHASE_DEFEND) * cd
            + float(config.HISTORY_XI_W_BOWL_USAGE) * bw
            + float(config.HISTORY_XI_W_PHASE_BOWL) * ph_mix
            + float(config.HISTORY_XI_W_BAT_AGGRESSOR) * bat_agg
            + float(config.HISTORY_XI_W_BOWL_CONTROL) * bowl_ctrl
            + w_h2h_xi * h2h_xi_rate
            + w_h2h_v * h2h_v_rate
        )
        bump = min(
            float(getattr(config, "HISTORY_XI_ROW_COUNT_BUMP_CAP", 0.06)),
            len(pr) * float(getattr(config, "HISTORY_XI_ROW_COUNT_BUMP", 0.00035)),
        )
        hx = hx_base + bump

        used_cur = bool(cur_season_rows)
        used_prior_rows = bool(prior_rows)
        used_venue = vnr > 0 and bool(venue_keys)
        heuristic_only = len(all_rows) == 0
        bat_prior_fallback = bool(prior_rows) and slot_source.startswith("historical_ema_prior")
        if used_cur:
            xi_selection_tier = "current_season"
        elif used_prior_rows:
            xi_selection_tier = "prior_season_team_usage"
        elif used_venue and venue_keys:
            xi_selection_tier = "venue_only_sparse"
        elif not heuristic_only:
            xi_selection_tier = "sparse_other_matches"
        else:
            xi_selection_tier = "no_stored_history"

        pos_hist = _batting_slot_series(pr_chrono)
        n_pbp_fill = len(pbp_by_mid)
        dist_matches = len({int(r["match_id"]) for r in pr})
        venue_pr = [
            r for r in pr_chrono if _venue_matches_key(str(r.get("venue") or ""), venue_keys)
        ]
        venue_spec_n = len({int(r["match_id"]) for r in venue_pr})
        recent_dates: list[str] = []
        seen_d: set[str] = set()
        for r in reversed(pr_chrono):
            md = r.get("match_date")
            if not md:
                continue
            ds = str(md)[:12]
            if ds in seen_d:
                continue
            seen_d.add(ds)
            recent_dates.append(str(md)[:10] if len(str(md)) >= 10 else ds)
            if len(recent_dates) >= 20:
                break

        wr5 = float(config.HISTORY_XI_W_RECENT5)
        wv = float(config.HISTORY_XI_W_VENUE)
        wps = float(config.HISTORY_XI_W_PRIOR_SEASON)
        wos = float(config.HISTORY_XI_W_OVERSEAS_PATTERN)
        wcd = float(config.HISTORY_XI_W_CHASE_DEFEND)
        wbw = float(config.HISTORY_XI_W_BOWL_USAGE)
        wph = float(config.HISTORY_XI_W_PHASE_BOWL)
        wba = float(config.HISTORY_XI_W_BAT_AGGRESSOR)
        wbc = float(config.HISTORY_XI_W_BOWL_CONTROL)
        batting_position_score = wps * ps + wr5 * r5 * 0.55 + wv * vnr * 0.35
        bowling_usage_score = wbw * bw + wph * ph_mix
        direct_h2h_score = w_h2h_xi * h2h_xi_rate + w_h2h_v * h2h_v_rate
        scoring_breakdown = {
            "history_xi_score": round(float(hx), 5),
            "batting_position_score": round(batting_position_score, 5),
            "bowling_usage_score": round(bowling_usage_score, 5),
            "direct_h2h_score": round(direct_h2h_score, 5),
            "venue_score": round(wv * vnr, 5),
            "weather_score": None,
            "overseas_chase_context_score": round(wos * op + wcd * cd, 5),
            "cricsheet_agg_style_score": round(wba * bat_agg + wbc * bowl_ctrl, 5),
            "history_xi_row_bump": round(bump, 6),
            "history_xi_base_before_bump": round(hx_base, 5),
            "captain_boost_applied": 0.0,
            "wicketkeeper_boost_applied": 0.0,
        }

        p.history_xi_score = float(hx)
        p.history_batting_ema = float(ema)

        gxi = _glob_xi.get(qk_global) or {"tmx_rows": 0, "distinct_matches": 0, "distinct_teams": 0}
        slot_t = _glob_pbp.get(qk_global)
        slot_ema_g = float(slot_t[0]) if slot_t else 0.0
        slot_n_g = int(slot_t[1]) if slot_t else 0
        prof_g = _glob_prof.get(qk_global) or {}
        merged_global = {
            "tmx_rows": int(gxi.get("tmx_rows") or 0),
            "distinct_matches": int(gxi.get("distinct_matches") or 0),
            "distinct_teams": int(gxi.get("distinct_teams") or 0),
            "global_batting_slot_ema": slot_ema_g,
            "global_batting_slot_samples": slot_n_g,
            "profile": prof_g,
        }
        prior_fc, prior_dbg = first_choice_prior.compute_probable_first_choice_prior(
            player=p,
            franchise_distinct_matches=dist_matches,
            franchise_team_match_xi_rows=len(pr),
            global_stats=merged_global,
        )
        fc_cap = int(getattr(config, "FIRST_CHOICE_GLOBAL_FRANCHISE_MATCHES_CAP", 5))
        fc_min_g = int(getattr(config, "FIRST_CHOICE_GLOBAL_MIN_DISTINCT_MATCHES", 2))
        fc_min_prior = float(getattr(config, "FIRST_CHOICE_USED_GLOBAL_PRIOR_MIN", 0.22))
        gdm = int(merged_global.get("distinct_matches") or 0)
        thin_franchise = dist_matches <= fc_cap
        strong_global = gdm >= fc_min_g
        other_fr_rows = int(_glob_other_fr.get(qk_global, 0))
        glob_pres_bool = bool(prior_dbg.get("global_ipl_history_presence"))
        movement_new_signal = (
            glob_pres_bool
            and dist_matches < fc_min_g
            and (other_fr_rows > 0 or gdm >= fc_min_g)
        )
        eff_prior_floor = fc_min_prior * (0.88 if movement_new_signal else 1.0)
        used_global_prior = bool(
            thin_franchise and strong_global and float(prior_fc) >= eff_prior_floor
        )
        selected_franchise_history_presence = bool(dist_matches > 0 or len(pr) > 0)
        history_for_other_franchises_presence = other_fr_rows > 0
        valid_current_squad_new_to_franchise = bool(
            (not wrong_side_squad_assignment)
            and in_selected_team_fetched_squad
            and glob_pres_bool
            and dist_matches < fc_min_g
            and (history_for_other_franchises_presence or gdm >= fc_min_g)
        )
        cap_raw = (captain_display_name or "").strip()
        wk_raw = (wicketkeeper_display_name or "").strip()
        pn = str(getattr(p, "name", "") or "").strip()

        def _manual_pick_matches(raw: str) -> bool:
            if not raw:
                return False
            if raw.casefold() == pn.casefold():
                return True
            return learner.normalize_player_key(raw) == pk

        captain_selected_for_team = _manual_pick_matches(cap_raw)
        wicketkeeper_selected_for_team = _manual_pick_matches(wk_raw)

        dbg: dict[str, Any] = {
            "in_current_squad": True,
            "history_enrichment_current_squad_only": True,
            "canonical_franchise_label": canon_label,
            "canonical_team_key": canonical_key,
            "canonical_player_key": pk,
            "fetched_from_team_slug": _fetch_slug or None,
            "in_selected_team_fetched_squad": in_selected_team_fetched_squad,
            "in_opposite_team_fetched_squad": in_opposite_team_fetched_squad,
            "wrong_side_squad_assignment": wrong_side_squad_assignment,
            "stale_cached_entry_detected": stale_cached_entry_detected,
            "selected_franchise_history_presence": selected_franchise_history_presence,
            "history_for_other_franchises_presence": history_for_other_franchises_presence,
            "valid_current_squad_new_to_franchise": valid_current_squad_new_to_franchise,
            "captain_selected_for_team": captain_selected_for_team,
            "wicketkeeper_selected_for_team": wicketkeeper_selected_for_team,
            "captain_boost_applied": 0.0,
            "wicketkeeper_boost_applied": 0.0,
            "global_resolved_history_key": grk_link or None,
            "global_alias_resolution_type": _lk.get("global_alias_resolution_type"),
            "global_alias_confidence": _lk.get("global_alias_confidence"),
            "global_alias_layer_used": _lk.get("global_alias_layer_used"),
            "used_global_resolved_key_for_prior": bool(_lk.get("used_global_resolved_key_for_prior")),
            "likely_first_ipl_player": bool(_lk.get("likely_first_ipl_player")),
            "debutant_alias_suppression_applied": bool(_lk.get("debutant_alias_suppression_applied")),
            "debutant_alias_rejection_reason": _lk.get("debutant_alias_rejection_reason"),
            "history_linkage": {
                **{
                    k: v
                    for k, v in _lk.items()
                    if k
                    not in (
                        "team_match_xi_rows",
                        "player_match_stats_rows",
                        "player_batting_positions_rows",
                        "latest_match_date",
                        "h2h_rows_vs_opponent",
                        "player_name",
                        "squad_display_name",
                    )
                },
                "squad_display_name": str(getattr(p, "name", "") or "").strip(),
                "player_name": str(getattr(p, "name", "") or "").strip(),
                "squad_canonical_player_key": pk,
                "canonical_player_key": pk,
                "history_lookup_key": sql_key or None,
                "history_status": _lk.get("history_status"),
                "resolution_type": _lk.get("resolution_type"),
                "rolled_up_interpretation": _lk.get("rolled_up_interpretation"),
                "resolution_layer_debug": _lk.get("resolution_layer_debug"),
                "matched_history_player_name": _lk.get("matched_history_player_name"),
                "alias_confidence": _lk.get("alias_confidence"),
                "team_name": _lk.get("team_name", canon_label),
                "canonical_team_key": _lk.get("canonical_team_key", canonical_key),
                "team_match_xi_rows": int(_lk.get("team_match_xi_rows", 0)),
                "player_match_stats_rows": int(_lk.get("player_match_stats_rows", 0)),
                "player_batting_positions_rows": int(_lk.get("player_batting_positions_rows", 0)),
                "latest_match_date": _lk.get("latest_match_date"),
                "h2h_rows_vs_opponent": int(_lk.get("h2h_rows_vs_opponent", 0)),
            },
            "history_linkage_team_summary": _link_summary,
            "history_rows_found": len(pr),
            "history_distinct_matches": dist_matches,
            "batting_position_rows_found": len(pos_hist),
            "pbp_primary_slot_matches": n_pbp_fill,
            "batting_positions_history": [round(x, 2) for x in pos_hist[-24:]],
            "batting_position_ema": round(float(ema), 3),
            "recent5_xi_rate": round(r5, 4),
            "venue_xi_rate": round(vnr, 4),
            "prior_season_xi_rate": round(ps, 4),
            "overseas_pattern_xi_rate": round(op, 4),
            "chase_defend_xi_rate": round(cd, 4),
            "bowl_usage_rate": round(bw, 4),
            "phase_bowl_rates": {k: round(float(v), 4) for k, v in ph_rates.items()},
            "phase_bowl_blend_general": round(ph_blend, 4),
            "phase_bowl_blend_with_h2h": round(ph_mix, 4),
            "phase_bowl_rates_h2h_matches": {k: round(float(v), 4) for k, v in ph_h2h.items()},
            "vs_spin_pace_faced": {
                "spin_share": round(float(spin_pace.get("spin_share", 0.0)), 4),
                "pace_share": round(float(spin_pace.get("pace_share", 0.0)), 4),
                "balls_tagged": int(spin_pace.get("samples", 0) or 0),
            },
            "cricsheet_franchise_features": (
                {
                    "batting_position_ema": fr.get("batting_position_ema"),
                    "batting_slot_samples": int(fr.get("batting_slot_samples") or 0),
                    "pp_overs_bowled": float(fr.get("pp_overs_bowled") or 0),
                    "middle_overs_bowled": float(fr.get("middle_overs_bowled") or 0),
                    "death_overs_bowled": float(fr.get("death_overs_bowled") or 0),
                    "pp_bowl_ball_share": float(fr.get("pp_bowl_ball_share") or 0),
                    "middle_bowl_ball_share": float(fr.get("middle_bowl_ball_share") or 0),
                    "death_bowl_ball_share": float(fr.get("death_bowl_ball_share") or 0),
                    "batting_aggressor_score": round(bat_agg, 4),
                    "bowling_control_score": round(bowl_ctrl, 4),
                }
                if fr
                else None
            ),
            "history_xi_score_base": round(hx_base, 5),
            "history_xi_row_bump": round(bump, 6),
            "history_xi_score": round(hx, 5),
            "batting_slot_ema": round(ema, 3),
            "batting_order_source": slot_source,
            "used_current_season_history": used_cur,
            "used_prior_season_fallback": bat_prior_fallback,
            "xi_used_prior_season_rows": used_prior_rows and not used_cur,
            "xi_selection_tier": xi_selection_tier,
            "used_venue_history": used_venue,
            "fallback_heuristics_only": heuristic_only,
            "stored_rows_for_franchise": len(all_rows),
            "h2h_opponent_canonical": opp or None,
            "h2h_fixtures_in_layer": len(h2h_fixtures),
            "h2h_distinct_matches_player_rows": len({int(r["match_id"]) for r in h2h_pr}),
            "h2h_weighted_xi_rate": round(h2h_xi_rate, 4),
            "h2h_venue_xi_rate": round(h2h_v_rate, 4),
            "h2h_venue_fixtures_in_layer": len(h2h_venue_fx),
            "h2h_batting_order_used_opponent_history": h2h_used_bat,
            "h2h_batting_order_used_venue_matchup_history": h2h_used_venue_slot,
            "h2h_xi_signal_in_score": (w_h2h_xi * h2h_xi_rate + w_h2h_v * h2h_v_rate) > 1e-6,
            "h2h_phase_blend_applied": abs(ph_mix - ph_blend) > 1e-4,
            "weights_h2h_xi_venue": {"h2h_xi": w_h2h_xi, "h2h_venue": w_h2h_v},
            "scoring_breakdown": scoring_breakdown,
            "franchise_history_distinct_matches": dist_matches,
            "probable_first_choice_prior": round(float(prior_fc), 5),
            "used_global_fallback_prior": used_global_prior,
            "global_ipl_history_presence": prior_dbg.get("global_ipl_history_presence"),
            "global_selection_frequency": prior_dbg.get("global_selection_frequency"),
            "global_batting_position_pattern": prior_dbg.get("global_batting_position_pattern"),
            "global_role_strength": prior_dbg.get("global_role_strength"),
            "first_choice_prior_debug": prior_dbg,
            "history_usage_debug": {
                "squad_display_name": str(getattr(p, "name", "") or "").strip(),
                "player_name": str(getattr(p, "name", "") or "").strip(),
                "canonical_player_key": pk,
                "normalized_full_name_key": _lk.get("normalized_full_name_key", pk),
                "resolved_history_key": _lk.get("resolved_history_key"),
                "history_lookup_key": sql_key or None,
                "resolution_type": _lk.get("resolution_type"),
                "history_status": _lk.get("history_status"),
                "resolution_type": _lk.get("resolution_type"),
                "rolled_up_interpretation": _lk.get("rolled_up_interpretation"),
                "resolution_layer_debug": _lk.get("resolution_layer_debug"),
                "matched_history_player_name": _lk.get("matched_history_player_name"),
                "alias_confidence": _lk.get("alias_confidence", _lk.get("confidence")),
                "ambiguous_candidates": _lk.get("ambiguous_candidates"),
                "team_match_xi_rows_found": int(_lk.get("team_match_xi_rows", 0)),
                "player_match_stats_rows_found": int(_lk.get("player_match_stats_rows", 0)),
                "player_batting_positions_rows_found": int(
                    _lk.get("player_batting_positions_rows", 0)
                ),
                "latest_match_date_found": _lk.get("latest_match_date"),
                "current_team_name": team_name,
                "history_match_count": dist_matches,
                "team_match_xi_row_count": len(pr),
                "player_match_stats_row_count": None,
                "batting_positions_found": len(pos_hist),
                "batting_position_rows_found": len(pos_hist),
                "pbp_primary_slot_matches": n_pbp_fill,
                "batting_position_ema": round(float(ema), 3),
                "batting_order_source": slot_source,
                "recent_match_dates_found": recent_dates,
                "direct_head_to_head_match_count_vs_opponent": len(
                    {int(r["match_id"]) for r in h2h_pr}
                ),
                "venue_specific_match_count": venue_spec_n,
                "probable_first_choice_prior": round(float(prior_fc), 5),
                "used_global_fallback_prior": used_global_prior,
                "global_ipl_history_presence": prior_dbg.get("global_ipl_history_presence"),
                "global_selection_frequency": prior_dbg.get("global_selection_frequency"),
                "global_batting_position_pattern": prior_dbg.get("global_batting_position_pattern"),
                "global_role_strength": prior_dbg.get("global_role_strength"),
                "global_resolved_history_key": grk_link or None,
                "global_alias_resolution_type": _lk.get("global_alias_resolution_type"),
                "global_alias_confidence": _lk.get("global_alias_confidence"),
                "global_alias_layer_used": _lk.get("global_alias_layer_used"),
                "used_global_resolved_key_for_prior": bool(_lk.get("used_global_resolved_key_for_prior")),
                "likely_first_ipl_player": bool(_lk.get("likely_first_ipl_player")),
                "debutant_alias_suppression_applied": bool(_lk.get("debutant_alias_suppression_applied")),
                "debutant_alias_rejection_reason": _lk.get("debutant_alias_rejection_reason"),
                "fetched_from_team_slug": _fetch_slug or None,
                "in_selected_team_fetched_squad": in_selected_team_fetched_squad,
                "in_opposite_team_fetched_squad": in_opposite_team_fetched_squad,
                "wrong_side_squad_assignment": wrong_side_squad_assignment,
                "stale_cached_entry_detected": stale_cached_entry_detected,
                "selected_franchise_history_presence": selected_franchise_history_presence,
                "history_for_other_franchises_presence": history_for_other_franchises_presence,
                "valid_current_squad_new_to_franchise": valid_current_squad_new_to_franchise,
                "captain_selected_for_team": captain_selected_for_team,
                "wicketkeeper_selected_for_team": wicketkeeper_selected_for_team,
            },
        }
        p.history_debug = dbg

    if h2h_explain is not None:
        if not h2h_explain.get("_h2h_layer_meta_done"):
            h2h_explain["_h2h_layer_meta_done"] = True
            h2h_explain["h2h_fixture_count"] = len(h2h_fixtures)
            h2h_explain["venue_specific_h2h_fixture_count"] = len(h2h_venue_fx)
            pair = [canon_label, opp] if opp and canon_label else []
            h2h_explain["pair_canonical_sorted"] = sorted(pair) if len(pair) == 2 else pair
            h2h_explain["weights_applied"] = {
                "HISTORY_XI_W_H2H_XI": float(getattr(config, "HISTORY_XI_W_H2H_XI", 0.0)),
                "HISTORY_XI_W_H2H_VENUE": float(getattr(config, "HISTORY_XI_W_H2H_VENUE", 0.0)),
                "WIN_ENG_WEIGHT_HEAD_TO_HEAD": float(
                    getattr(config, "WIN_ENG_WEIGHT_HEAD_TO_HEAD", 0.0)
                ),
            }
        scope = (h2h_explain_scope or "").strip()
        if scope in ("team_a", "team_b"):
            w_h2h_sum = float(getattr(config, "HISTORY_XI_W_H2H_XI", 0.0)) + float(
                getattr(config, "HISTORY_XI_W_H2H_VENUE", 0.0)
            )
            h2h_explain[scope] = {
                "franchise_canonical": canon_label,
                "opponent_canonical": opp or None,
                "general_history_xi_rows_used": len(all_rows),
                "players_batting_order_used_h2h": int(bat_h2h_players),
                "players_xi_received_h2h_rate_signal": int(xi_h2h_signal_players),
                "players_phase_bowl_used_h2h_blend": int(phase_h2h_players),
                "batting_order_used_h2h_data": bat_h2h_players > 0,
                "xi_used_h2h_data": bool(h2h_fixtures) and w_h2h_sum > 0,
                "impact_prediction_used_h2h_data": bool(h2h_fixtures),
            }


def _parse_team_selection_xi_freq_weights(xi_frequency_json: Any) -> dict[str, float]:
    if isinstance(xi_frequency_json, list):
        raw = xi_frequency_json
    elif isinstance(xi_frequency_json, str) and xi_frequency_json.strip():
        try:
            raw = json.loads(xi_frequency_json)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    if not isinstance(raw, list):
        return {}
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pk = str(item.get("player_key") or "").strip()
        if not pk:
            continue
        try:
            c = int(item.get("count") or item.get("xi_matches") or 0)
        except (TypeError, ValueError):
            c = 0
        pairs.append((pk, c))
    if not pairs:
        return {}
    mx = max(c for _, c in pairs) or 1
    return {pk: max(0.0, min(1.0, c / float(mx))) for pk, c in pairs}


def _derive_core_signal_norm(prof: Optional[dict[str, Any]]) -> float:
    if not prof:
        return 0.5
    try:
        xi_f = float(prof.get("xi_selection_frequency") or 0.0)
        ru = float(prof.get("recent_usage_score") or 0.0)
        rs = float(prof.get("role_stability_score") or 0.0)
    except (TypeError, ValueError):
        return 0.5
    core = 0.46 * max(0.0, min(1.0, xi_f)) + 0.30 * max(0.0, min(1.0, ru)) + 0.24 * max(0.0, min(1.0, rs))
    return max(0.0, min(1.0, core))


def _role_likelihood_vector(prof: Optional[dict[str, Any]]) -> dict[str, float]:
    out = {
        "opener_likelihood": 0.5,
        "finisher_likelihood": 0.5,
        "powerplay_bowler_likelihood": 0.5,
        "death_bowler_likelihood": 0.5,
    }
    if not prof:
        return out
    for k in out:
        try:
            out[k] = max(0.0, min(1.0, float(prof.get(k) or 0.5)))
        except (TypeError, ValueError):
            pass
    return out


def _history_source_used_label(hd: dict[str, Any]) -> str:
    lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
    rt = str(lk.get("resolution_type") or "").strip() or "unknown"
    grt = str(lk.get("global_alias_resolution_type") or "").strip()
    if rt == "no_match" and grt and grt != "no_match":
        return f"global_fallback:{grt}"
    if bool(lk.get("used_global_resolved_key_for_prior")):
        return f"franchise_primary+global_prior:{rt}"
    return f"franchise_sqlite:{rt}"


def _sparse_debutante_player(hd: dict[str, Any]) -> bool:
    if bool(hd.get("likely_first_ipl_player")):
        return True
    if bool(hd.get("debutant_alias_suppression_applied")):
        return True
    lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
    if str(lk.get("rolled_up_interpretation") or "") == "likely_new_or_sparse":
        return True
    if bool(lk.get("likely_first_ipl_player")):
        return True
    if bool(lk.get("debutant_alias_suppression_applied")):
        return True
    return False


def _selection_reason_summary(
    *,
    has_hist: bool,
    derive_norm: float,
    venue_b: float,
    h2h_rate: float,
    weather_proxy: float,
    fallback_used: bool,
    debut_damp: bool,
) -> str:
    parts: list[str] = []
    parts.append("Franchise SQLite history is strong for ordering." if has_hist else "Franchise SQLite history is thin; composite and derive fill the gap.")
    if derive_norm >= 0.62:
        parts.append("Stage-2 player profile (XI frequency / usage / stability) supports a high pick probability.")
    elif derive_norm <= 0.42:
        parts.append("Derive profile is muted — role/composite and manual priors matter more.")
    if venue_b > 0.02:
        parts.append("Small lift from venue-specific XI frequency in derive patterns.")
    if h2h_rate > 0.08:
        parts.append("Head-to-head usage signal nudged the score.")
    if weather_proxy >= 0.58:
        parts.append("Conditions analyst/coach blend slightly favour this skill mix.")
    if fallback_used:
        parts.append("Global IPL fallback informed priors where franchise rows were sparse.")
    if debut_damp:
        parts.append("Debut / sparse guard limited how much weak-alias or noisy derive data could move the rank.")
    return " ".join(parts)


def _scenario_xi_branch_delta(
    p: Any,
    derive_snap: Optional[dict[str, Any]],
    *,
    bf: float,
    dew: float,
    spin_f: float,
    pace_b: float,
    swing_proxy: float,
    is_night: bool,
    team_bats_first: bool,
) -> tuple[float, dict[str, float], str, str]:
    """
    Small deterministic adjustment (raw, before cap) for one branch: this franchise bats first vs bowls first.
    Returns (raw_delta, breakdown, bat_first_reason_line, bowl_first_reason_line) — last two are non-empty only for matching branch.
    """
    ds = derive_snap or {}
    try:
        opener = float(ds.get("opener_likelihood") or 0.0)
    except (TypeError, ValueError):
        opener = 0.0
    try:
        finisher = float(ds.get("finisher_likelihood") or 0.0)
    except (TypeError, ValueError):
        finisher = 0.0
    try:
        pp = float(ds.get("powerplay_bowler_likelihood") or 0.0)
    except (TypeError, ValueError):
        pp = 0.0
    try:
        death = float(ds.get("death_bowler_likelihood") or 0.0)
    except (TypeError, ValueError):
        death = 0.0
    try:
        bpe = float(ds.get("batting_position_ema") or 0.0)
    except (TypeError, ValueError):
        bpe = 0.0

    bat_skill = float(getattr(p, "bat_skill", 0.5) or 0.5)
    bowl_skill = float(getattr(p, "bowl_skill", 0.5) or 0.5)
    rb = str(getattr(p, "role_bucket", "") or "")
    btype = str(getattr(p, "bowling_type", "") or "").lower()
    spin_like = any(x in btype for x in ("spin", "slow", "orthodox", "finger", "wrist"))
    seam_like = not spin_like and rb == "Bowler"

    bd: dict[str, float] = {}
    raw = 0.0
    r_bat = ""
    r_bowl = ""

    if team_bats_first:
        t_top = 0.048 * opener * (0.55 + 0.45 * bat_skill)
        if rb in ("Batter", "WK-Batter"):
            raw += t_top
            bd["top_order_bat_first"] = t_top
            if t_top > 0.012:
                r_bat = "Top-order / opener likelihood fits setting a target when batting first."
        if bpe > 0.5 and bpe < 4.2 and rb in ("Batter", "WK-Batter", "All-Rounder"):
            t_slot = 0.032 * max(0.0, (4.0 - bpe) / 3.5) * bat_skill
            raw += t_slot
            bd["early_slot_ema_bat_first"] = t_slot
        t_death = 0.042 * death * bowl_skill * (0.65 + 0.35 * float(is_night))
        if rb in ("Bowler", "All-Rounder"):
            raw += t_death
            bd["death_defend_second_innings"] = t_death
            if t_death > 0.012:
                r_bat = (r_bat + " " if r_bat else "") + "Death-phase bowling history valued for defending a total later."
        t_spin = 0.036 * pp * 0.15
        t_spin += 0.038 * (spin_f - 0.48) * (1.0 if spin_like else 0.35 * bowl_skill)
        if spin_f >= 0.55:
            raw += max(0.0, t_spin)
            bd["spin_friendly_surface"] = max(0.0, t_spin)
        if bf >= 0.58 and rb in ("Batter", "WK-Batter", "All-Rounder"):
            t_depth = 0.034 * (bf - 0.52) * bat_skill
            raw += t_depth
            bd["batting_depth_high_scoring_venue"] = t_depth
            if t_depth > 0.01:
                r_bat = (r_bat + " " if r_bat else "") + "Batting depth uplift on a high-scoring venue profile."
        if dew >= 0.62 and spin_like:
            pen = -0.044 * (dew - 0.55) * (1.0 - 0.55 * max(pp, death))
            raw += pen
            bd["dew_spin_penalty"] = pen
        chase_anchor = 0.028 * finisher * bat_skill * 0.35
        raw -= chase_anchor
        bd["chase_finisher_deprioritize_bat_first"] = -chase_anchor
    else:
        t_pp = 0.052 * pp * bowl_skill * (0.42 + 0.58 * pace_b) * (0.55 + 0.45 * swing_proxy)
        if seam_like or rb == "Bowler":
            raw += t_pp
            bd["powerplay_seam_bowl_first"] = t_pp
            if t_pp > 0.014:
                r_bowl = "Powerplay seam/swing profile fits bowling first in these conditions."
        t_fin = 0.05 * finisher * bat_skill
        if rb in ("Batter", "WK-Batter", "All-Rounder"):
            raw += t_fin
            bd["chase_anchor_finisher_bowl_first"] = t_fin
            if t_fin > 0.014:
                r_bowl = (r_bowl + " " if r_bowl else "") + "Chase / finisher-shaped history matters more when batting second."
        t_death = 0.039 * death * bowl_skill * (0.55 + 0.45 * (1.0 - bf * 0.35))
        if rb in ("Bowler", "All-Rounder"):
            raw += t_death
            bd["death_defend_target_bowl_first"] = t_death
        if spin_f >= 0.56 and spin_like:
            ts = 0.041 * (spin_f - 0.5) * bowl_skill
            raw += ts
            bd["spin_reserve_dry_venue"] = ts
        if dew >= 0.62 and spin_like:
            pen = -0.041 * (dew - 0.55) * (1.0 - 0.5 * max(pp, death))
            raw += pen
            bd["dew_spin_penalty_bowl_first"] = pen

    strength = float(getattr(config, "SCENARIO_XI_BRANCH_STRENGTH", 1.0))
    if strength != 1.0 and strength > 0:
        raw *= strength
        bd = {k: v * strength for k, v in bd.items()}

    cap = float(getattr(config, "SCENARIO_XI_MAX_ABS_DELTA", 0.078))
    if raw > cap:
        scale = cap / raw
        raw = cap
        bd = {k: v * scale for k, v in bd.items()}
    elif raw < -cap:
        scale = -cap / raw if raw != 0 else 1.0
        raw = -cap
        bd = {k: v * scale for k, v in bd.items()}

    return raw, bd, r_bat.strip(), r_bowl.strip()


def _scenario_xi_package_for_player(
    p: Any,
    base_sel: float,
    derive_snap: Optional[dict[str, Any]],
    cond: dict[str, Any],
    *,
    is_night: bool,
) -> dict[str, Any]:
    bf = float(cond.get("batting_friendliness", 0.5))
    dew = float(cond.get("dew_risk", 0.5))
    spin_f = float(cond.get("spin_friendliness", 0.5))
    pace_b = float(cond.get("pace_bias", 0.5))
    swing_proxy = float(cond.get("swing_seam_proxy", 0.5))

    d_bf, bd_bf, r_bf, _ = _scenario_xi_branch_delta(
        p,
        derive_snap,
        bf=bf,
        dew=dew,
        spin_f=spin_f,
        pace_b=pace_b,
        swing_proxy=swing_proxy,
        is_night=is_night,
        team_bats_first=True,
    )
    d_bl, bd_bl, _, r_bl = _scenario_xi_branch_delta(
        p,
        derive_snap,
        bf=bf,
        dew=dew,
        spin_f=spin_f,
        pace_b=pace_b,
        swing_proxy=swing_proxy,
        is_night=is_night,
        team_bats_first=False,
    )
    s_bf = max(0.0, min(1.0, base_sel + d_bf))
    s_bl = max(0.0, min(1.0, base_sel + d_bl))
    return {
        "if_team_bats_first": {
            "scenario_selection_score": round(s_bf, 5),
            "scenario_adjustment_total": round(d_bf, 5),
            "scenario_adjustment_breakdown": {k: round(v, 5) for k, v in bd_bf.items()},
            "selected_for_batting_first_reason": r_bf or "Scenario weights neutral; base selection_score dominates.",
            "selected_for_bowling_first_reason": "",
        },
        "if_team_bowls_first": {
            "scenario_selection_score": round(s_bl, 5),
            "scenario_adjustment_total": round(d_bl, 5),
            "scenario_adjustment_breakdown": {k: round(v, 5) for k, v in bd_bl.items()},
            "selected_for_batting_first_reason": "",
            "selected_for_bowling_first_reason": r_bl or "Scenario weights neutral; base selection_score dominates.",
        },
    }


def compute_selection_scores(
    players: list[Any],
    *,
    conditions: Optional[dict[str, Any]] = None,
    venue_key_candidates: Optional[list[str]] = None,
    fixture_context: Optional[dict[str, Any]] = None,
) -> None:
    """
    Blend history_xi_score with composite for ordering / repairs (current squad only).

    Optionally folds Stage-2 ``player_profiles`` + ``team_selection_patterns`` (SQLite only).
    """
    cond = conditions or {}
    bf = float(cond.get("batting_friendliness", 0.5))
    dew = float(cond.get("dew_risk", 0.5))
    rain = float(cond.get("rain_disruption_risk", 0.0))

    franchise_team_key = ""
    if players:
        franchise_team_key = str(
            getattr(players[0], "canonical_team_key", "") or ""
        ).strip()[:80]

    profiles: dict[str, dict[str, Any]] = {}
    venue_weights: dict[str, float] = {}
    pattern_row: Optional[dict[str, Any]] = None
    if franchise_team_key:
        pkeys = [
            str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
            for p in players
        ]
        pkeys = [k for k in pkeys if k]
        profiles = db.batch_player_profiles_for_franchise(pkeys, franchise_team_key)
        cands = [str(v).strip()[:80] for v in (venue_key_candidates or []) if str(v).strip()]
        pattern_row = db.fetch_team_selection_pattern(franchise_team_key, cands)
        if pattern_row:
            venue_weights = _parse_team_selection_xi_freq_weights(pattern_row.get("xi_frequency_json"))

    scores = [float(getattr(p, "history_xi_score", 0.0)) for p in players]
    mx = max(scores) if scores else 0.0
    mn = min(scores) if scores else 0.0
    span = max(1e-6, mx - mn)
    w_strong = float(getattr(config, "HISTORY_SELECTION_HISTORY_WEIGHT_STRONG", 0.94))
    w_weak = float(getattr(config, "HISTORY_SELECTION_HISTORY_WEIGHT_WEAK", 0.76))
    thr = int(getattr(config, "HISTORY_SELECTION_STRONG_ROWS_THRESHOLD", 2))
    blend_max = float(getattr(config, "STAGE3_DERIVE_HN_BLEND_MAX", 0.24))
    debut_damp_f = float(getattr(config, "STAGE3_DERIVE_DEBUT_DAMP", 0.3))
    conf_floor = float(getattr(config, "STAGE3_PROFILE_CONFIDENCE_FLOOR", 0.14))
    vfit_w = float(getattr(config, "STAGE3_VENUE_FIT_CONDITIONS_WEIGHT", 0.07))
    vx_cap = float(getattr(config, "STAGE3_VENUE_TEAM_XI_FREQ_BOOST_CAP", 0.055))

    for p in players:
        hx = float(getattr(p, "history_xi_score", 0.0))
        hn = (hx - mn) / span
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        rows_n = int(hd.get("history_rows_found") or 0)
        fr = hd.get("cricsheet_franchise_features") or {}
        slot_n = int(fr.get("batting_slot_samples") or 0) if isinstance(fr, dict) else 0
        has_hist = rows_n >= thr or slot_n >= thr
        w = w_strong if has_hist else w_weak
        hn += min(0.035, rows_n * float(getattr(config, "HISTORY_XI_ROW_COUNT_BUMP", 0.00055)))
        dist_fr = int(hd.get("franchise_history_distinct_matches") or rows_n)
        prior_fc = float(hd.get("probable_first_choice_prior") or 0.0)
        used_g = bool(hd.get("used_global_fallback_prior"))
        fc_cap = int(getattr(config, "FIRST_CHOICE_GLOBAL_FRANCHISE_MATCHES_CAP", 5))
        max_boost = float(getattr(config, "FIRST_CHOICE_PRIOR_MAX_HN_BOOST", 0.16))
        thin_fr = dist_fr <= fc_cap
        hn_boost = 0.0
        if thin_fr and prior_fc > 0.2:
            hn_boost = max_boost * min(1.0, prior_fc * (1.12 if used_g else 0.82))
        elif prior_fc > 0.58 and dist_fr <= fc_cap + 4:
            hn_boost = max_boost * 0.42 * prior_fc
        if bool(hd.get("valid_current_squad_new_to_franchise")):
            hn_boost *= float(getattr(config, "FIRST_CHOICE_NEW_TO_FRANCHISE_HN_FACTOR", 1.28))
        cap_boost = float(getattr(config, "CAPTAIN_SELECTION_HN_BOOST", 0.14))
        wk_boost = float(getattr(config, "WICKETKEEPER_SELECTION_HN_BOOST", 0.19))
        captain_applied = 0.0
        wk_applied = 0.0
        if bool(hd.get("captain_selected_for_team")):
            hn += cap_boost
            captain_applied = cap_boost
        if bool(hd.get("wicketkeeper_selected_for_team")):
            hn += wk_boost
            wk_applied = wk_boost
        hn += hn_boost

        pk = str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
        prof = profiles.get(pk)
        derive_snap: Optional[dict[str, Any]] = None
        if prof:
            derive_snap = {
                "xi_selection_frequency": float(prof.get("xi_selection_frequency") or 0.0),
                "recent_usage_score": float(prof.get("recent_usage_score") or 0.0),
                "role_stability_score": float(prof.get("role_stability_score") or 0.0),
                "venue_fit_score": float(prof.get("venue_fit_score") or 0.0),
                "batting_position_ema": float(prof.get("batting_position_ema") or 0.0),
                "profile_confidence": float(prof.get("profile_confidence") or 0.0),
                "sample_matches": int(prof.get("sample_matches") or 0),
            }
            derive_snap.update(_role_likelihood_vector(prof))
        hd["derive_player_profile"] = derive_snap

        derive_norm = _derive_core_signal_norm(prof)
        try:
            p_conf = float(prof.get("profile_confidence") or 0.0) if prof else 0.0
        except (TypeError, ValueError):
            p_conf = 0.0
        conf_scale = max(0.0, min(1.0, (p_conf - conf_floor) / max(1e-6, 1.0 - conf_floor)))
        sparse_debut = _sparse_debutante_player(hd)
        damp = debut_damp_f if sparse_debut else 1.0
        hd["__stage3_debut_damp"] = damp
        gamma = blend_max * conf_scale * damp
        hn_before_derive = hn
        hn_mixed = (1.0 - gamma) * hn + gamma * derive_norm
        hn = hn_mixed

        v_fit = float(prof.get("venue_fit_score") or 0.5) if prof else 0.5
        weather_align = vfit_w * (v_fit - 0.5) * (bf - 0.48 + 0.12 * dew - 0.14 * rain)
        hn += weather_align

        venue_pat_boost = 0.0
        if pk and venue_weights:
            venue_pat_boost = vx_cap * float(venue_weights.get(pk, 0.0))
        hn += venue_pat_boost

        hn = max(0.0, min(1.0, hn))

        comp = float(getattr(p, "composite", 0.5))
        hd["__stage3_hn_by_pk"] = hn
        hd["__stage3_hw_by_pk"] = w
        hd["__stage3_comp"] = comp
        hd["__stage3_has_hist"] = has_hist
        hd["__stage3_derive_norm"] = derive_norm
        hd["__stage3_venue_pat"] = venue_pat_boost
        hd["__stage3_hn_before"] = hn_before_derive
        hd["__stage3_gamma"] = gamma
        hd["__stage3_weather_align"] = weather_align
        hd["__stage3_captain_applied"] = captain_applied
        hd["__stage3_wk_applied"] = wk_applied
        hd["__stage3_hn_boost"] = hn_boost
        hd["__stage3_p_conf"] = p_conf
        hd["__stage3_sparse_debut"] = sparse_debut

    hn_by_pk: dict[str, float] = {}
    hw_by_pk: dict[str, float] = {}
    comp_by_name: dict[str, float] = {}
    for p in players:
        hd = p.history_debug
        pk = str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
        hn_by_pk[pk] = float(hd.pop("__stage3_hn_by_pk"))
        hw_by_pk[pk] = float(hd.pop("__stage3_hw_by_pk"))
        comp_by_name[p.name] = float(hd.pop("__stage3_comp"))

    selection_model.apply_selection_model(
        players,
        conditions=cond,
        franchise_team_key=franchise_team_key,
        profiles=profiles,
        venue_weights=venue_weights,
        pattern_row=pattern_row,
        fixture_context=fixture_context or {},
        hn_by_player=hn_by_pk,
        history_weights_by_pk=hw_by_pk,
        composite_by_player=comp_by_name,
    )

    for p in players:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        pk = str(getattr(p, "player_key", "") or learner.normalize_player_key(getattr(p, "name", ""))).strip()[:80]
        w = float(hw_by_pk.get(pk, 0.82))
        comp = float(comp_by_name.get(p.name, 0.5))
        has_hist = bool(hd.pop("__stage3_has_hist"))
        derive_norm = float(hd.pop("__stage3_derive_norm"))
        venue_pat_boost = float(hd.pop("__stage3_venue_pat"))
        hn_before_derive = float(hd.pop("__stage3_hn_before"))
        gamma = float(hd.pop("__stage3_gamma"))
        weather_align = float(hd.pop("__stage3_weather_align"))
        captain_applied = float(hd.pop("__stage3_captain_applied"))
        wk_applied = float(hd.pop("__stage3_wk_applied"))
        hn_boost = float(hd.pop("__stage3_hn_boost"))
        p_conf = float(hd.pop("__stage3_p_conf"))
        sparse_debut = bool(hd.pop("__stage3_sparse_debut"))
        debut_damp_factor = float(hd.pop("__stage3_debut_damp"))
        derive_snap = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        hn = float(hn_by_pk.get(pk, 0.5))
        sel = float(p.selection_score)
        scen_pkg = hd.get("scenario_xi") or {}

        h2h_rate = float(hd.get("h2h_weighted_xi_rate") or 0.0)
        an = float((getattr(p, "perspectives", None) or {}).get("analyst", 0.5))
        co = float((getattr(p, "perspectives", None) or {}).get("coach", 0.5))
        weather_proxy = 0.58 * an + 0.42 * co
        role_fb = (1.0 - w) * comp
        used_g = bool(hd.get("used_global_fallback_prior"))
        fallback_used = bool(
            used_g
            or (
                str((hd.get("history_linkage") or {}).get("resolution_type") or "")
                == "no_match"
                and str((hd.get("history_linkage") or {}).get("global_alias_resolution_type") or "")
                not in ("", "no_match")
            )
        )
        src_used = _history_source_used_label(hd)
        reason = _selection_reason_summary(
            has_hist=has_hist,
            derive_norm=derive_norm,
            venue_b=venue_pat_boost,
            h2h_rate=h2h_rate,
            weather_proxy=weather_proxy,
            fallback_used=fallback_used,
            debut_damp=sparse_debut and debut_damp_factor < 0.99,
        )
        sm = hd.get("selection_model_debug") or {}
        ex = sm.get("explainability") or {}
        if ex:
            reason = (
                f"{reason} | Model: recent/IPL/team-balance/venue blend with tactical modifiers. "
                f"{ex.get('tactical_modifiers_reason', '')}"
            ).strip()[:520]
        hd["selection_reason_summary"] = reason
        hd["history_source_used"] = src_used
        hd["fallback_used"] = fallback_used

        sb = hd.get("scoring_breakdown")
        if isinstance(sb, dict):
            sb = dict(sb)
            sb["weather_score"] = round(weather_proxy, 5)
            sb["composite_score"] = round(comp, 5)
            sb["role_fallback_score"] = round(role_fb, 5)
            sb["final_selection_score"] = round(sel, 5)
            sb["captain_boost_applied"] = round(captain_applied, 5)
            sb["wicketkeeper_boost_applied"] = round(wk_applied, 5)
            sb["derive_core_signal_norm"] = round(derive_norm, 5)
            sb["venue_pattern_boost"] = round(venue_pat_boost, 5)
            sb["venue_fit_conditions_align"] = round(weather_align, 5)
            hd["scoring_breakdown"] = sb
        hd["captain_boost_applied"] = round(captain_applied, 5)
        hd["wicketkeeper_boost_applied"] = round(wk_applied, 5)
        hd["selection_score_components"] = {
            "history_normalized_pre_derive": round(max(0.0, min(1.0, hn_before_derive)), 5),
            "history_normalized": round(hn, 5),
            "derive_blend_gamma_applied": round(gamma, 5),
            "derive_core_signal_norm": round(derive_norm, 5),
            "venue_team_pattern_boost": round(venue_pat_boost, 5),
            "venue_fit_conditions_align": round(weather_align, 5),
            "xi_selection_frequency": round(float(derive_snap.get("xi_selection_frequency", 0.0)), 5) if derive_snap else None,
            "recent_usage_score": round(float(derive_snap.get("recent_usage_score", 0.0)), 5) if derive_snap else None,
            "role_stability_score": round(float(derive_snap.get("role_stability_score", 0.0)), 5) if derive_snap else None,
            "venue_fit_score": round(float(derive_snap.get("venue_fit_score", 0.0)), 5) if derive_snap else None,
            "opener_likelihood": round(float(derive_snap.get("opener_likelihood", 0.0)), 5) if derive_snap else None,
            "finisher_likelihood": round(float(derive_snap.get("finisher_likelihood", 0.0)), 5) if derive_snap else None,
            "powerplay_bowler_likelihood": round(float(derive_snap.get("powerplay_bowler_likelihood", 0.0)), 5)
            if derive_snap
            else None,
            "death_bowler_likelihood": round(float(derive_snap.get("death_bowler_likelihood", 0.0)), 5)
            if derive_snap
            else None,
            "h2h_boost_rate": round(h2h_rate, 5),
            "weather_score_proxy": round(weather_proxy, 5),
            "history_weight_applied": round(w, 5),
            "history_weight_strong": w_strong,
            "history_weight_weak": w_weak,
            "has_usable_sqlite_or_cricsheet_history": has_hist,
            "probable_first_choice_prior": round(float(hd.get("probable_first_choice_prior") or 0.0), 5),
            "used_global_fallback_prior": bool(hd.get("used_global_fallback_prior")),
            "first_choice_hn_boost_applied": round(hn_boost, 5),
            "captain_boost_applied": round(captain_applied, 5),
            "wicketkeeper_boost_applied": round(wk_applied, 5),
            "valid_current_squad_new_to_franchise": bool(hd.get("valid_current_squad_new_to_franchise")),
            "composite": round(comp, 5),
            "selection_score": round(sel, 5),
            "role_fallback_score": round(role_fb, 5),
            "profile_confidence": round(p_conf, 5),
            "derive_debut_damp_applied": sparse_debut,
            "team_selection_pattern_venue_hits": bool(pattern_row),
            "scenario_xi_if_team_bats_first": scen_pkg.get("if_team_bats_first"),
            "scenario_xi_if_team_bowls_first": scen_pkg.get("if_team_bowls_first"),
            "selection_model": sm,
        }
