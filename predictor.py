"""Playing XI, subs, toss effects, and win probability from multi-perspective scoring."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, FrozenSet, Optional

import audit_profile
import canonical_keys
import config
import db
import history_rules
import history_sync
import history_xi
import impact_subs_engine
import ipl_squad
import ipl_teams
import learner
import time_utils
import win_probability_engine
from history_context import HistoryContext, build_history_context, venue_lookup_keys
from ipl_squad import (
    ALL_ROUNDER,
    BATTER,
    BOWLER,
    WK_BATTER,
    role_bucket_to_predictor_role,
)
from venues import VenueProfile, venue_conditions_summary

logger = logging.getLogger(__name__)
_perf_logger = logging.getLogger("ipl_predictor.perf")


def _slim_prediction_layer_debug(pld: dict[str, Any]) -> dict[str, Any]:
    """Drop the largest lists (full bench impact + per-player history usage) for lighter JSON/UI."""
    slimmed: dict[str, Any] = {}
    for side in ("team_a", "team_b"):
        block = dict(pld.get(side) or {})
        block["impact_sub_ranking"] = []
        block["history_usage_per_player"] = []
        block["_light_debug_omitted_large_lists"] = True
        slimmed[side] = block
    return slimmed

# Toss keys for UI ↔ engine (Team B bowls first ≡ Team A bats first in T20).
TOSS_SCENARIO_OPTIONS: tuple[tuple[str, str], ...] = (
    ("unknown", "Unknown toss"),
    ("a_bats_first", "Team A bats first"),
    ("a_bowls_first", "Team A bowls first"),
    ("b_bats_first", "Team B bats first"),
    ("b_bowls_first", "Team B bowls first"),
)


def resolve_a_bats_first_toss(toss_key: str) -> Optional[bool]:
    """Return True if Team A bats first, False if Team B bats first, None if unknown."""
    k = (toss_key or "unknown").strip().lower()
    if k in ("unknown", "", "none"):
        return None
    if k in ("a_bats_first", "b_bowls_first"):
        return True
    if k in ("a_bowls_first", "b_bats_first"):
        return False
    return None


def franchise_xi_scenario_branch(for_team_a: bool, a_bats_first: Optional[bool]) -> str:
    """SQLite history_xi scenario key used to rank this franchise's XI."""
    if a_bats_first is None:
        return "if_team_bats_first"
    a_first = bool(a_bats_first)
    this_bats_first = a_first if for_team_a else not a_first
    return "if_team_bats_first" if this_bats_first else "if_team_bowls_first"


def _franchise_name_match(a: str, b: str) -> bool:
    na, nb = learner.normalize_player_key(a), learner.normalize_player_key(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def chase_context_for_team(
    team_name: str,
    team_a_name: str,
    team_b_name: str,
    a_bats_first: Optional[bool],
) -> Optional[bool]:
    """
    True if this team is expected to **chase** (bat second); False if bat first; None if toss unknown.
    """
    if a_bats_first is None:
        return None
    if _franchise_name_match(team_name, team_a_name):
        return not a_bats_first
    if _franchise_name_match(team_name, team_b_name):
        return a_bats_first
    return None


def _batting_order_diagnostic_source(p: SquadPlayer) -> str:
    """
    Coarse UI label: direct_history | prior_season | h2h_history | role_fallback.
    Derived from stored scorecard EMA path (``batting_order_source``) and unknown-slot sentinel.
    """
    hd = getattr(p, "history_debug", None) or {}
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    ema = float(getattr(p, "history_batting_ema", unk))
    src = str(hd.get("batting_order_source") or "")
    if ema >= unk - 1e-6:
        return "role_fallback"
    low = src.lower()
    if "h2h" in low:
        return "h2h_history"
    if "prior_season" in src:
        return "prior_season"
    return "direct_history"


def _batting_order_sources_summary(xi: list[SquadPlayer]) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for p in xi:
        c[_batting_order_diagnostic_source(p)] += 1
    return dict(c)


def _batting_order_team_lineup_summary(xi: list[SquadPlayer]) -> str:
    if not xi:
        return "No XI to summarise."
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    anchored = sum(1 for p in xi if float(getattr(p, "history_batting_ema", unk)) < unk - 1e-6)
    derive_touch = sum(
        1
        for p in xi
        if isinstance((getattr(p, "history_debug", None) or {}).get("derive_player_profile"), dict)
    )
    return (
        f"{anchored}/{len(xi)} batters anchored on scorecard EMA; "
        f"{derive_touch}/{len(xi)} picked up Stage-2 batting-role hints; "
        "remainder ordered via role-bucket fallback when EMA is missing."
    )


def _xi_batting_order_diagnostics_rows(
    xi: list[SquadPlayer], order: list[str]
) -> list[dict[str, Any]]:
    by_name = {p.name: p for p in xi}
    rows: list[dict[str, Any]] = []
    for i, n in enumerate(order):
        p = by_name.get(n)
        if p is None:
            continue
        hd = getattr(p, "history_debug", None) or {}
        rows.append(
            {
                "name": n,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "batting_order_position": i + 1,
                "batting_positions_history": hd.get("batting_positions_history"),
                "batting_position_ema": hd.get("batting_slot_ema"),
                "batting_order_source": hd.get("batting_order_source"),
                "batting_order_diagnostic_source": hd.get("batting_order_diagnostic_source"),
                "batting_order_final": hd.get("batting_order_final"),
                "batting_order_reason_summary": hd.get("batting_order_reason_summary"),
                "batting_order_signal_source_ranked": hd.get("batting_order_signal_source_ranked"),
            }
        )
    return rows


@dataclass
class SquadPlayer:
    name: str
    role: str  # bat, bowl, all, wk — scoring / history
    is_overseas: bool
    player_key: str = ""  # canonical identity (truncated); mirrors ``canonical_player_key``
    team_display_name: str = ""
    canonical_team_key: str = ""
    canonical_player_key: str = ""  # same normalization as ``player_key`` when populated
    role_bucket: str = "Batter"  # IPL: Batter, WK-Batter, All-Rounder, Bowler
    bat_skill: float = 0.55
    bowl_skill: float = 0.45
    is_wicketkeeper: bool = False
    batting_roles: list[str] = field(default_factory=list)
    bowling_type: Optional[str] = None
    is_opener_candidate: bool = False
    is_finisher_candidate: bool = False
    history_xi_score: float = 0.0
    history_batting_ema: float = 99.0
    history_debug: dict[str, Any] = field(default_factory=dict)
    selection_score: float = 0.0
    perspectives: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    history_delta: float = 0.0
    history_notes: list[str] = field(default_factory=list)


def _annotate_squad_canonical_keys(squad: list[SquadPlayer], franchise_canonical_label: str) -> None:
    lab = (franchise_canonical_label or "").strip()
    ctk = ipl_teams.canonical_team_key_for_franchise(lab)
    for p in squad:
        p.team_display_name = lab
        p.canonical_team_key = ctk
        pk = (p.player_key or "").strip() or learner.normalize_player_key(p.name)
        p.player_key = pk
        p.canonical_player_key = pk


def _micro_noise(name: str) -> float:
    """Deterministic tiny tie-breaker; not a popularity prior (stable across Python runs)."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10007 / 10007.0) * 0.012


def _role_skills(role: str) -> tuple[float, float, bool]:
    r = role.lower().strip()
    if r == "wk":
        return 0.62, 0.22, True
    if r == "bowl":
        return 0.30, 0.84, False
    if r == "all":
        return 0.58, 0.62, False
    if r == "bat":
        return 0.80, 0.24, False
    return 0.55, 0.40, False


def _squad_player_from_ipl(
    name: str,
    role_bucket: str,
    *,
    overseas: bool = False,
) -> Optional[SquadPlayer]:
    ok, _reason = ipl_squad.validate_clean_name(name)
    if not ok:
        return None
    pk = learner.normalize_player_key(name)
    role = role_bucket_to_predictor_role(role_bucket)
    bat, bowl, is_wk = _role_skills(role)
    bat += _micro_noise(name)
    bowl += _micro_noise(name[::-1])
    br = ipl_squad.default_batting_roles_for_bucket(
        role_bucket, is_keeper=(role_bucket == WK_BATTER)
    )
    return SquadPlayer(
        name=name,
        role=role,
        role_bucket=role_bucket,
        is_overseas=overseas,
        player_key=pk,
        canonical_player_key=pk,
        bat_skill=max(0.05, min(0.98, bat)),
        bowl_skill=max(0.05, min(0.98, bowl)),
        is_wicketkeeper=is_wk or (role_bucket == WK_BATTER),
        batting_roles=list(br),
        bowling_type=None,
    )


def _log_role_bucket_distribution(squad: list[SquadPlayer], tag: str) -> dict[str, int]:
    from collections import Counter

    c = Counter(p.role_bucket for p in squad)
    dist = dict(c)
    logger.info("role_bucket_distribution %s: %s (n=%d)", tag, dist, len(squad))
    ar = c.get(ALL_ROUNDER, 0)
    if len(squad) >= 12 and ar > max(7, int(len(squad) * 0.32)):
        logger.warning(
            "%s: unusually high All-Rounder count=%d of %d — check IPL source roles",
            tag,
            ar,
            len(squad),
        )
    return dist


def _squad_player_key_set(squad: list[SquadPlayer]) -> set[str]:
    """Normalized keys for every player in the current fetched squad."""
    return {learner.normalize_player_key(p.name) for p in squad if learner.normalize_player_key(p.name)}


def _validate_xi_in_current_squad(
    xi: list[SquadPlayer], squad: list[SquadPlayer], side: str
) -> tuple[bool, list[str]]:
    """Hard gate: XI must be a subset of the current structured squad (inner-join semantics)."""
    sk = _squad_player_key_set(squad)
    bad: list[str] = []
    for p in xi:
        pk = (getattr(p, "player_key", None) or "").strip() or learner.normalize_player_key(p.name)
        if pk not in sk:
            bad.append(p.name)
    logger.info(
        "%s strict XI⊆squad: squad_keys=%d xi=%d subset_ok=%s xi_names=%s offenders=%s",
        side,
        len(sk),
        len(xi),
        len(bad) == 0,
        [p.name for p in xi],
        bad,
    )
    return (len(bad) == 0, bad)


def _batting_order_strict_names_for_xi(
    xi: list[SquadPlayer], candidate: list[str]
) -> tuple[list[str], list[str]]:
    """
    Batting order may only reference **selected XI** names (no historical or foreign names).

    Returns ``(order, warnings)``. Missing XI members are appended using bat_skill order.
    """
    xi_set = {p.name for p in xi}
    warnings: list[str] = []
    seen: set[str] = set()
    out: list[str] = []
    for n in candidate:
        if n in xi_set and n not in seen:
            out.append(n)
            seen.add(n)
    dropped = [n for n in candidate if n not in xi_set]
    if dropped:
        msg = (
            f"Batting order ignored {len(dropped)} name(s) not in the selected XI "
            f"(strict scope): {', '.join(dropped[:8])}"
            + (" …" if len(dropped) > 8 else "")
        )
        warnings.append(msg)
        logger.warning("batting_order strict: dropped non-XI names team context dropped=%s", dropped)
    missing = [p.name for p in xi if p.name not in seen]
    if missing:
        rest = sorted(
            [p for p in xi if p.name in missing],
            key=lambda q: (-q.bat_skill, q.name),
        )
        for p in rest:
            out.append(p.name)
            seen.add(p.name)
        warnings.append(
            "Batting order was repaired to include every XI player (by bat_skill): "
            + ", ".join(missing[:10])
            + (" …" if len(missing) > 10 else "")
        )
    if len(xi) >= 11 and len(out) != 11:
        warnings.append(f"Batting order length {len(out)} after strict XI repair (expected 11).")
    return out, warnings


def _collect_strict_validation_warnings(
    *,
    team_a_label: str,
    team_b_label: str,
    xi_a: list[SquadPlayer],
    xi_b: list[SquadPlayer],
    scored_squad_a: list[SquadPlayer],
    scored_squad_b: list[SquadPlayer],
    history_sync_debug: dict[str, Any],
    batting_order_warnings: list[str],
) -> list[str]:
    """User-visible warnings: sparse local SQLite history, batting-order repairs."""
    out: list[str] = []
    thr = int(getattr(config, "HISTORY_VALIDATION_SPARSE_ROWS_WARN", 2))
    frac = float(getattr(config, "HISTORY_VALIDATION_SPARSE_XI_FRACTION", 0.35))
    db_thr = int(getattr(config, "HISTORY_VALIDATION_SPARSE_REQUIRE_DB_MATCHES_BELOW", 14))

    def _franchise_db_matches(key: str) -> int:
        block = (history_sync_debug or {}).get(key) or {}
        return int(block.get("distinct_matches_local_db") or 0)

    def sparse_block(label: str, xi: list[SquadPlayer], side_key: str) -> None:
        if not xi:
            return
        block = (history_sync_debug or {}).get(side_key) or {}
        if not history_sync.raw_stage1_tables_near_empty(block):
            return
        db_n = _franchise_db_matches(side_key)
        if db_n >= db_thr:
            return
        sparse_n = sum(
            1
            for p in xi
            if int((getattr(p, "history_debug", None) or {}).get("history_rows_found") or 0) < thr
        )
        if sparse_n / max(1, len(xi)) >= frac:
            out.append(
                f"{label}: Thin SQLite history — {db_n} distinct stored matches for this franchise "
                f"(warning threshold: <{db_thr} matches), and {sparse_n}/{len(xi)} XI players have fewer than "
                f"{thr} team_match_xi rows. {history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )

    sparse_block(team_a_label, xi_a, "team_a")
    sparse_block(team_b_label, xi_b, "team_b")

    for label, squad in (
        (team_a_label, scored_squad_a),
        (team_b_label, scored_squad_b),
    ):
        if not squad:
            continue
        summ = (getattr(squad[0], "history_debug", None) or {}).get("history_linkage_team_summary") or {}
        health = (summ or {}).get("stage_f_team_health") or ""
        detail = (summ or {}).get("stage_f_team_health_detail") or {}
        if health == "major_linkage_failure":
            out.append(
                f"{label}: Stage F major_linkage_failure — unresolved core players "
                f"{detail.get('core_unresolved', '?')}/{detail.get('core_players', '?')} "
                f"(Batter/WK/AR). Inspect history_usage_per_player and resolution_layer_debug."
            )
        elif health == "partial_linkage_issue":
            out.append(
                f"{label}: Stage F partial_linkage_issue — {detail.get('core_unresolved', '?')} core "
                "player(s) still lack a SQLite key; others linked OK."
            )
        elif health == "healthy_with_sparse_new_players":
            out.append(
                f"{label}: Stage F healthy_with_sparse_new_players — linkage healthy; remaining gaps "
                "skew toward newer/non-core squad names."
            )

    def batting_slot_coverage_warn(label: str, xi: list[SquadPlayer], side_key: str) -> None:
        if len(xi) != 11:
            return
        block = (history_sync_debug or {}).get(side_key) or {}
        if not history_sync.raw_stage1_tables_near_empty(block):
            return
        n_missing = 0
        for p in xi:
            hd = getattr(p, "history_debug", None) or {}
            n_slot = int(
                hd.get("batting_position_rows_found")
                or hd.get("batting_positions_found")
                or (hd.get("history_usage_debug") or {}).get("batting_position_rows_found")
                or 0
            )
            if n_slot < 1:
                n_missing += 1
        if n_missing / 11.0 > 0.5:
            out.append(
                f"{label}: {n_missing}/11 XI players have no stored batting-slot rows in SQLite. "
                f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )

    batting_slot_coverage_warn(team_a_label, xi_a, "team_a")
    batting_slot_coverage_warn(team_b_label, xi_b, "team_b")

    def alias_resolution_core_warn(label: str, xi: list[SquadPlayer], squad_scored: list[SquadPlayer]) -> None:
        """Only escalate when Stage F health is already partial/major (avoid noisy warnings)."""
        if not xi or not squad_scored:
            return
        summ = (getattr(squad_scored[0], "history_debug", None) or {}).get("history_linkage_team_summary") or {}
        health = summ.get("stage_f_team_health") or ""
        if health not in ("major_linkage_failure", "partial_linkage_issue"):
            return
        core_roles = (BATTER, WK_BATTER, ALL_ROUNDER)
        bad: list[str] = []
        for p in xi:
            if p.role_bucket not in core_roles:
                continue
            hd = getattr(p, "history_debug", None) or {}
            lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
            rt = lk.get("resolution_type")
            n_tmx = int(lk.get("team_match_xi_rows", 0) or 0)
            rolled = lk.get("rolled_up_interpretation") or ""
            if rt == "ambiguous_alias":
                bad.append(f"{p.name} (ambiguous_alias / {rolled})")
            elif rt == "no_match" and n_tmx == 0:
                if bool(hd.get("valid_current_squad_new_to_franchise")) and bool(
                    hd.get("global_ipl_history_presence")
                ):
                    continue
                bad.append(f"{p.name} (no_match / {rolled})")
        if bad:
            out.append(
                f"{label}: Core players still unresolved under Stage F ({health}) — "
                + ", ".join(bad[:8])
                + (" …" if len(bad) > 8 else "")
            )

    alias_resolution_core_warn(team_a_label, xi_a, scored_squad_a)
    alias_resolution_core_warn(team_b_label, xi_b, scored_squad_b)

    def wrong_side_squad_warn(label: str, squad: list[SquadPlayer]) -> None:
        leaked = [
            p.name
            for p in squad
            if bool((getattr(p, "history_debug", None) or {}).get("wrong_side_squad_assignment"))
        ]
        if leaked:
            out.append(
                f"{label}: Wrong-side or stale squad leak suspected (player in opposite official fetch "
                f"or stale cache vs selected franchise): {', '.join(leaked[:14])}"
                + (" …" if len(leaked) > 14 else "")
            )

    wrong_side_squad_warn(team_a_label, scored_squad_a)
    wrong_side_squad_warn(team_b_label, scored_squad_b)

    def new_to_franchise_note(label: str, squad: list[SquadPlayer]) -> None:
        names = [
            p.name
            for p in squad
            if bool((getattr(p, "history_debug", None) or {}).get("valid_current_squad_new_to_franchise"))
        ]
        if names:
            out.append(
                f"{label}: Current squad player(s) new to franchise in stored history; "
                f"global IPL fallback prior applied: {', '.join(names[:10])}"
                + (" …" if len(names) > 10 else "")
            )

    new_to_franchise_note(team_a_label, scored_squad_a)
    new_to_franchise_note(team_b_label, scored_squad_b)

    for label, key in ((team_a_label, "team_a"), (team_b_label, "team_b")):
        block = (history_sync_debug or {}).get(key) or {}
        if block.get("failsafe"):
            out.append(
                f"{label}: Local SQLite history snapshot failed; continuing with heuristics. "
                f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
            )
            continue
        for note in (block.get("local_history_notes") or [])[:4]:
            if isinstance(note, str) and note.strip():
                out.append(f"{label}: {note.strip()}")

    out.extend(batting_order_warnings)
    return out


def parse_squad_text(text: str) -> list[SquadPlayer]:
    """
    IPL squad lines:

    - Preferred: ``Name | Batter`` or ``Name | WK-Batter | overseas``
    - Legacy: ``Name, bat`` / ``Name, bowl`` / ``Name, wk`` / ``Name, all`` (+ optional overseas)
    """
    out: list[SquadPlayer] = []
    seen_player_keys: set[str] = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if not parts:
                continue
            base, emb = ipl_squad.split_embedded_role_from_name(parts[0])
            name = (base or parts[0]).strip()
            role_bucket = BATTER
            if len(parts) >= 2:
                nb = ipl_squad.normalize_role_bucket_label(parts[1])
                if nb:
                    role_bucket = nb
                else:
                    low = parts[1].lower().replace("_", " ")
                    if low in ("bat", "batsman", "batting"):
                        role_bucket = BATTER
                    elif low in ("bowl", "bowler", "bowling"):
                        role_bucket = BOWLER
                    elif low in ("wk", "keeper", "wk-batter", "wk batter", "wkbat"):
                        role_bucket = WK_BATTER
                    elif low in ("all", "allrounder", "all-rounder", "all rounder", "ar"):
                        role_bucket = ALL_ROUNDER
            if emb:
                role_bucket = emb
            overseas = any(p.lower() in ("overseas", "os", "foreign", "int", "intl") for p in parts[2:])
            sp = _squad_player_from_ipl(name, role_bucket, overseas=overseas)
            if sp and sp.player_key and sp.player_key not in seen_player_keys:
                seen_player_keys.add(sp.player_key)
                out.append(sp)
            continue

        parts = [p.strip() for p in re.split(r"[,|]", line) if p.strip()]
        if not parts:
            continue
        base, emb = ipl_squad.split_embedded_role_from_name(parts[0])
        name = (base or parts[0]).strip()
        role_bucket = BATTER
        overseas = False
        for p in parts[1:]:
            low = p.lower()
            if low in ("bat", "batsman", "batting", "batter"):
                role_bucket = BATTER
            elif low in ("bowl", "bowler", "bowling"):
                role_bucket = BOWLER
            elif low in ("all", "ar", "allrounder", "all-rounder"):
                role_bucket = ALL_ROUNDER
            elif low in ("wk", "keeper", "wicketkeeper", "wkbat", "wk-batter"):
                role_bucket = WK_BATTER
            elif low in ("overseas", "os", "foreign", "int", "intl"):
                overseas = True
        if emb:
            role_bucket = emb
        sp = _squad_player_from_ipl(name, role_bucket, overseas=overseas)
        if sp and sp.player_key and sp.player_key not in seen_player_keys:
            seen_player_keys.add(sp.player_key)
            out.append(sp)

    logger.info("parse_squad_text: loaded %d structured players (deduped by player_key)", len(out))
    return out


def _normalize_name_set(text: str) -> set[str]:
    return {learner.normalize_player_key(x) for x in text.splitlines() if x.strip()}


def filter_unavailable(squad: list[SquadPlayer], unavailable_blob: str) -> list[SquadPlayer]:
    banned = _normalize_name_set(unavailable_blob)
    if not banned:
        return list(squad)
    return [p for p in squad if learner.normalize_player_key(p.name) not in banned]


def _squad_shape(squad: list[SquadPlayer]) -> dict[str, float]:
    n = max(1, len(squad))
    wk = sum(1 for p in squad if p.is_wicketkeeper)
    bow = sum(
        1
        for p in squad
        if p.role_bucket in (BOWLER, ALL_ROUNDER)
        or p.role in ("bowl", "all")
        or p.bowl_skill >= 0.55
    )
    bat = sum(
        1
        for p in squad
        if p.role_bucket in (BATTER, WK_BATTER)
        or p.role == "bat"
        or p.bat_skill >= 0.65
    )
    os = sum(1 for p in squad if p.is_overseas)
    return {
        "wk_density": wk / n,
        "bowl_depth": bow / n,
        "bat_depth": bat / n,
        "overseas_density": os / n,
    }


def _score_player(
    p: SquadPlayer,
    *,
    self_shape: dict[str, float],
    opp_shape: dict[str, float],
    conditions: dict[str, Any],
    learned_map: dict[str, dict[str, Any]],
    franchise_canonical: Optional[str] = None,
) -> SquadPlayer:
    bf = float(conditions["batting_friendliness"])
    pace_bias = float(conditions["pace_bias"])
    dew = float(conditions["dew_risk"])
    swing = float(conditions["swing_seam_proxy"])
    spin_f = float(conditions["spin_friendliness"])
    rain = float(conditions["rain_disruption_risk"])
    boundary = float(conditions["boundary_size"])

    learned = learner.learned_boost_for_player(p.name, learned_map)

    # Coach: fit to venue + squad balance needs
    need_bowling = max(0.0, 0.55 - self_shape["bowl_depth"])
    need_bat = max(0.0, 0.52 - self_shape["bat_depth"])
    coach = (
        0.34 * (p.bat_skill * bf + p.bowl_skill * (1.0 - bf))
        + 0.22 * (p.bowl_skill * need_bowling * 2.2 + p.bat_skill * need_bat * 2.2)
        + 0.18 * (p.is_wicketkeeper * 0.25 + (1.0 if p.is_wicketkeeper else 0.0) * (0.65 - self_shape["wk_density"]))
        + 0.16 * (p.bowl_skill * pace_bias * swing + p.bowl_skill * (1.0 - pace_bias) * spin_f)
        + 0.10 * (1.0 - rain) * (p.bat_skill * 0.55 + p.bowl_skill * 0.45)
    )
    coach = max(0.0, min(1.0, coach))

    # Player (self): matchup vs opposition squad shape
    player = (
        0.45 * (p.bat_skill * (0.55 + opp_shape["bowl_depth"]) + p.bowl_skill * (0.45 + opp_shape["bat_depth"]))
        + 0.25 * (p.bowl_skill * swing * pace_bias + p.bat_skill * (1.0 - swing) * 0.35)
        + 0.18 * (p.bat_skill * (1.0 - boundary) * 0.4 + p.bowl_skill * boundary * 0.35)
        + 0.12 * (p.bowl_skill * dew * 0.45)
    )
    player = max(0.0, min(1.0, player))

    # Analyst: conditions-heavy read
    analyst = (
        0.30 * (p.bat_skill * bf + p.bowl_skill * (1.1 - bf) * 0.85)
        + 0.22 * (p.bowl_skill * (spin_f * (1.0 - pace_bias) + swing * pace_bias))
        + 0.20 * (p.bat_skill * (1.0 - 0.55 * rain))
        + 0.16 * (p.bowl_skill * (0.5 + 0.5 * dew))
        + 0.12 * learned
    )
    analyst = max(0.0, min(1.0, analyst))

    # Opposition: how much the opponent would dislike this pick (mirror shapes)
    opposition = (
        0.40 * (p.bowl_skill * (0.45 + opp_shape["bat_depth"]) + p.bat_skill * (0.5 + opp_shape["bowl_depth"]))
        + 0.28 * (p.bat_skill * pace_bias * (1.0 - boundary) + p.bowl_skill * boundary)
        + 0.18 * (p.bowl_skill * (1.0 - opp_shape["bowl_depth"]) * 1.15)
        + 0.14 * (p.bat_skill * spin_f * 0.45 + p.bowl_skill * (1.0 - spin_f) * 0.35)
    )
    opposition = max(0.0, min(1.0, opposition))

    composite = (
        config.WEIGHT_COACH * coach
        + config.WEIGHT_PLAYER * player
        + config.WEIGHT_ANALYST * analyst
        + config.WEIGHT_OPPOSITION * opposition
        + config.WEIGHT_LEARNED * learned
    )
    p.perspectives = {
        "coach": coach,
        "player": player,
        "analyst": analyst,
        "opposition": opposition,
        "learned": learned,
    }
    p.composite = composite
    if franchise_canonical:
        lab = ipl_teams.canonical_franchise_label(franchise_canonical) or franchise_canonical.strip()
        fk = ipl_teams.canonical_team_key_for_franchise(lab)
        mfeat = db.get_player_franchise_features(learner.normalize_player_key(p.name), fk)
        if mfeat:
            agg = float(mfeat.get("batting_aggressor_score") or 0)
            ctrl = float(mfeat.get("bowling_control_score") or 0)
            spin_t = float(mfeat.get("vs_spin_tendency") or 0)
            mu = (
                0.012 * agg * float(p.bat_skill)
                + 0.012 * ctrl * float(p.bowl_skill)
                + 0.006 * spin_t * float(p.bat_skill)
            )
            p.composite = max(0.0, min(1.0, float(p.composite) + mu))
    return p


def _history_adjust_for_player(
    p: SquadPlayer,
    team_name: str,
    shape: dict[str, float],
    venue_keys: list[str],
    is_night: bool,
    dew_risk: float,
    ctx: HistoryContext,
) -> None:
    """Optional small composite bump from legacy history rules (disabled when history-primary XI)."""
    bump_scale = float(getattr(config, "HISTORY_COMPOSITE_HISTORY_BUMP_SCALE", 1.0))
    if getattr(config, "HISTORY_PRIMARY_XI_SELECTION", False) and bump_scale <= 0:
        p.history_delta = 0.0
        p.history_notes = []
        return
    pk = learner.normalize_player_key(p.name)
    tk = learner.normalize_player_key(team_name)[:80]
    proxy_slot = int(config.HISTORY_PROXY_BAT_SLOT.get(p.role, 6))

    t_xi = history_rules.xi_frequency_term(
        ctx.xi_by_player.get(pk, 0),
        max(1, ctx.max_xi_picks),
        ctx.db_match_count,
    )

    slot_row = ctx.avg_slot_by_player.get(pk)
    t_slot = history_rules.batting_slot_term(
        slot_row[0] if slot_row else None,
        proxy_slot,
        slot_row[1] if slot_row else 0,
    )

    bowl_row = ctx.bowl_balls_avg_by_player.get(pk)
    t_bowl = history_rules.bowling_usage_term(
        bowl_row[0] if bowl_row else 0.0,
        p.role,
        bowl_row[1] if bowl_row else 0,
    )

    picks_v, vtm = 0, 0
    for vk in venue_keys:
        key3 = (vk, tk, pk)
        if key3 in ctx.venue_team_player_xi:
            picks_v = ctx.venue_team_player_xi[key3]
            vtm = ctx.venue_team_matches.get((vk, tk), 0)
            break
    t_vt = history_rules.venue_team_xi_term(picks_v, vtm)

    os_guess = min(
        config.MAX_OVERSEAS,
        max(0, int(round(shape["overseas_density"] * 11 + 0.001))),
    )
    mix = ctx.pick_overseas_mix(venue_keys, tk)
    t_os = history_rules.overseas_mix_term(os_guess, mix)

    nx = ctx.night_xi_by_player.get(pk, 0)
    dx = ctx.day_xi_by_player.get(pk, 0)
    t_nd = history_rules.night_day_term(is_night, nx, dx)
    t_dew = history_rules.dew_context_term(dew_risk, nx, dx, p.role)

    terms = [t_xi, t_slot, t_bowl, t_vt, t_os, t_nd, t_dew]
    weights = [
        config.LEARN_WEIGHT_XI_FREQUENCY,
        config.LEARN_WEIGHT_BATTING_SLOT,
        config.LEARN_WEIGHT_BOWLING_USAGE,
        config.LEARN_WEIGHT_VENUE_TEAM_XI,
        config.LEARN_WEIGHT_OVERSEAS_MIX,
        config.LEARN_WEIGHT_DAY_NIGHT,
        config.LEARN_WEIGHT_DEW_CONTEXT,
    ]
    d, lines = history_rules.blend_history_deltas(terms, weights)
    p.history_delta = float(d * bump_scale)
    p.history_notes = lines[:8]
    p.composite = max(0.001, p.composite + float(config.HISTORY_BLEND_SCALE) * d * bump_scale)


def _resolve_chase_prior(
    venue_keys: list[str], ctx: HistoryContext
) -> tuple[float, int]:
    for vk in venue_keys:
        row = ctx.chase_share_by_venue.get(vk)
        if row:
            return row[0], row[1]
    return 0.5, 0


def _linkage_confidence_adjustments(
    scored_a: list[SquadPlayer],
    scored_b: list[SquadPlayer],
) -> dict[str, Any]:
    core_roles = frozenset({"Batter", "WK-Batter", "All-Rounder"})
    alias_quality_hits = 0
    global_fb = 0
    collision_n = 0
    sparse_core = 0
    ambiguous_n = 0
    no_match_core = 0

    for p in scored_a + scored_b:
        hd = getattr(p, "history_debug", None) or {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        rt = str(lk.get("resolution_type") or "")
        coll = str(lk.get("collision_resolution_outcome") or "")
        if coll == "lost_collision" or rt == "ambiguous_alias_collision":
            collision_n += 1
        if rt == "ambiguous_alias":
            ambiguous_n += 1
        rolled = str(lk.get("rolled_up_interpretation") or "")
        if bool(lk.get("used_global_resolved_key_for_prior")) or "global_alias" in rolled:
            global_fb += 1
        if (p.role_bucket or "") in core_roles:
            if rt == "no_match":
                no_match_core += 1
            if str(lk.get("rolled_up_interpretation") or "") == "likely_new_or_sparse":
                sparse_core += 1
        if rt in ("exact_match", "alias_match") and coll != "lost_collision":
            alias_quality_hits += 1

    npl = max(1, len(scored_a) + len(scored_b))
    alias_quality_component = min(1.0, alias_quality_hits / float(npl))
    global_fallback_penalty = min(0.22, 0.028 * global_fb)
    collision_penalty = min(0.28, 0.065 * collision_n)
    sparse_core_penalty = min(0.2, 0.035 * sparse_core)
    ambiguous_penalty = min(0.14, 0.022 * ambiguous_n)
    no_match_core_penalty = min(0.18, 0.04 * no_match_core)
    total_pen = min(
        0.55,
        global_fallback_penalty
        + collision_penalty
        + sparse_core_penalty
        + ambiguous_penalty
        + no_match_core_penalty,
    )
    return {
        "alias_quality_component": round(alias_quality_component, 4),
        "global_fallback_penalty": round(global_fallback_penalty, 4),
        "collision_penalty": round(collision_penalty, 4),
        "sparse_core_penalty": round(sparse_core_penalty, 4),
        "ambiguous_alias_penalty": round(ambiguous_penalty, 4),
        "no_match_core_penalty": round(no_match_core_penalty, 4),
        "linkage_penalty_total": round(total_pen, 4),
        "counts": {
            "collision_losers": collision_n,
            "global_fallback_signals": global_fb,
            "ambiguous_alias_players": ambiguous_n,
            "no_match_core_players": no_match_core,
            "likely_sparse_core_players": sparse_core,
        },
    }


def _prediction_confidence(
    xi_a: list[SquadPlayer],
    xi_b: list[SquadPlayer],
    hctx: HistoryContext,
    str_a: float,
    str_b: float,
    *,
    scored_a: Optional[list[SquadPlayer]] = None,
    scored_b: Optional[list[SquadPlayer]] = None,
) -> dict[str, Any]:
    n = hctx.db_match_count
    db_s = min(1.0, n / max(1, config.CONF_MIN_MATCHES_FOR_FULL_DB))

    def xi_cov(xi: list[SquadPlayer]) -> float:
        if len(xi) != 11:
            return 0.0
        hit = sum(
            1 for p in xi if hctx.xi_by_player.get(learner.normalize_player_key(p.name), 0) > 0
        )
        return min(1.0, (hit / 11.0) / max(0.05, config.CONF_IDEAL_XI_HISTORY_COVERAGE))

    cov_s = (xi_cov(xi_a) + xi_cov(xi_b)) / 2.0

    persp_agree: list[float] = []
    for p in xi_a + xi_b:
        vals = [
            float(p.perspectives[k])
            for k in ("coach", "player", "analyst", "opposition")
            if k in p.perspectives
        ]
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            spread = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
            persp_agree.append(max(0.0, min(1.0, 1.0 - spread / 0.32)))
    agree_s = sum(persp_agree) / max(1, len(persp_agree))

    sep = abs(str_a - str_b) / max(1e-6, 11.0)
    sep_s = min(1.0, sep / max(1e-6, config.CONF_SEPARATION_TARGET * 11.0))

    wsum = (
        config.CONF_WEIGHT_DB_DEPTH
        + config.CONF_WEIGHT_HISTORY_COVERAGE
        + config.CONF_WEIGHT_PERSPECTIVE_AGREEMENT
        + config.CONF_WEIGHT_SCORE_SEPARATION
    )
    score = (
        config.CONF_WEIGHT_DB_DEPTH * db_s
        + config.CONF_WEIGHT_HISTORY_COVERAGE * cov_s
        + config.CONF_WEIGHT_PERSPECTIVE_AGREEMENT * agree_s
        + config.CONF_WEIGHT_SCORE_SEPARATION * sep_s
    ) / max(1e-6, wsum)

    link_adj: dict[str, Any] = {}
    if scored_a is not None and scored_b is not None:
        link_adj = _linkage_confidence_adjustments(scored_a, scored_b)
        score = max(0.0, min(1.0, score - float(link_adj.get("linkage_penalty_total") or 0.0)))

    out: dict[str, Any] = {
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "components": {
            "database_depth": round(db_s, 4),
            "xi_history_coverage": round(cov_s, 4),
            "perspective_agreement": round(agree_s, 4),
            "strength_separation": round(sep_s, 4),
        },
        "raw": {
            "stored_matches": n,
            "strength_gap_per_slot": round(sep, 6),
        },
    }
    if link_adj:
        out["linkage_adjustments"] = link_adj
    return out


def _is_bowling_option(p: SquadPlayer) -> bool:
    if p.role_bucket in (BOWLER, ALL_ROUNDER):
        return True
    if p.role in ("bowl", "all"):
        return True
    return p.bowl_skill >= config.BOWLING_OPTION_THRESHOLD


def _set_player_ipl_flags(p: SquadPlayer) -> None:
    """After scoring: opener / finisher flags for XI validation and batting order."""
    p.is_opener_candidate = p.role_bucket in (BATTER, WK_BATTER)
    p.is_finisher_candidate = (
        p.role_bucket == ALL_ROUNDER
        or (p.bat_skill >= 0.54 and p.bowl_skill >= 0.48)
        or (p.role_bucket == BOWLER and p.bat_skill >= 0.43)
    )


def _xi_role_validation_counts(xi: list[SquadPlayer]) -> dict[str, Any]:
    """
    Counts on the **selected XI** for selection-debug UI (derive likelihood thresholds are heuristic).
    """
    if not xi:
        return {
            "xi_size": 0,
            "wicketkeeper_count": 0,
            "bowling_options_count": 0,
            "powerplay_bowlers_count": 0,
            "death_bowlers_count": 0,
            "opener_candidates_count": 0,
            "finisher_candidates_count": 0,
            "overseas_count": 0,
        }

    def _prof(p: SquadPlayer) -> dict[str, Any]:
        hd = getattr(p, "history_debug", None) or {}
        d = hd.get("derive_player_profile")
        return d if isinstance(d, dict) else {}

    pp_thr = 0.45
    death_thr = 0.45
    pp_n = 0
    de_n = 0
    for p in xi:
        d = _prof(p)
        try:
            if float(d.get("powerplay_bowler_likelihood") or 0) >= pp_thr:
                pp_n += 1
        except (TypeError, ValueError):
            pass
        try:
            if float(d.get("death_bowler_likelihood") or 0) >= death_thr:
                de_n += 1
        except (TypeError, ValueError):
            pass

    return {
        "xi_size": len(xi),
        "wicketkeeper_count": sum(1 for p in xi if p.is_wicketkeeper),
        "bowling_options_count": sum(1 for p in xi if _is_bowling_option(p)),
        "powerplay_bowlers_count": pp_n,
        "death_bowlers_count": de_n,
        "opener_candidates_count": sum(1 for p in xi if getattr(p, "is_opener_candidate", False)),
        "finisher_candidates_count": sum(1 for p in xi if getattr(p, "is_finisher_candidate", False)),
        "overseas_count": sum(1 for p in xi if p.is_overseas),
        "powerplay_death_threshold_note": f"PP/death counts use derive likelihood ≥ {pp_thr:.2f}",
    }


def _validate_xi(xi: list[SquadPlayer]) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if len(xi) != 11:
        errs.append(f"XI size {len(xi)} != 11")
    os = sum(1 for p in xi if p.is_overseas)
    if os > config.MAX_OVERSEAS:
        errs.append(f"Overseas {os} > {config.MAX_OVERSEAS}")
    wk = sum(1 for p in xi if p.is_wicketkeeper)
    if wk < config.MIN_WICKETKEEPERS:
        errs.append("No wicketkeeper")
    bow_opts = sum(1 for p in xi if _is_bowling_option(p))
    if bow_opts < config.MIN_BOWLING_OPTIONS:
        errs.append(f"Bowling options {bow_opts} < {config.MIN_BOWLING_OPTIONS}")
    non_bowl = sum(1 for p in xi if p.role_bucket != BOWLER)
    if non_bowl < config.MIN_NON_BOWLERS_IN_XI:
        errs.append(
            f"Batting depth: need ≥{config.MIN_NON_BOWLERS_IN_XI} non-bowlers, have {non_bowl}"
        )
    opener_pool = sum(1 for p in xi if p.role_bucket in (BATTER, WK_BATTER))
    if opener_pool < config.MIN_OPENER_BUCKET_IN_XI:
        errs.append(
            f"Opener candidates: need ≥{config.MIN_OPENER_BUCKET_IN_XI} Batter/WK-Batter, have {opener_pool}"
        )
    if not any(p.is_finisher_candidate for p in xi):
        errs.append(
            "Finisher: need at least one All-Rounder or dual-skill player (see model flags)"
        )
    return (len(errs) == 0, errs)


def _scenario_xi_rank_value(p: SquadPlayer, scenario_branch: Optional[str]) -> float:
    """Rank key for XI build: scenario-adjusted when branch is set, else base selection_score."""
    if scenario_branch not in ("if_team_bats_first", "if_team_bowls_first"):
        return float(getattr(p, "selection_score", 0.0))
    sx = (getattr(p, "history_debug", None) or {}).get("scenario_xi") or {}
    sub = sx.get(scenario_branch) or {}
    v = sub.get("scenario_selection_score")
    if v is not None:
        return float(v)
    return float(getattr(p, "selection_score", 0.0))


def _try_build_xi(
    sorted_pool: list[SquadPlayer],
    full_pool: list[SquadPlayer],
    penalties: Optional[dict[str, float]] = None,
    *,
    scenario_branch: Optional[str] = None,
) -> Optional[list[SquadPlayer]]:
    pen = penalties or {}
    rs: Callable[[SquadPlayer], float] = lambda x: _scenario_xi_rank_value(x, scenario_branch)
    xi: list[SquadPlayer] = []
    used = set()
    for p in sorted_pool:
        if len(xi) >= 11:
            break
        xi.append(p)
        used.add(p.name)
    ok, _ = _validate_xi(xi)
    if ok:
        return xi

    # Repair loop
    pool_by_name = {q.name: q for q in full_pool}
    order = sorted(
        full_pool,
        key=lambda x: (
            rs(x) - pen.get(x.name, 0.0),
            _xi_selection_tier(x),
            x.composite,
        ),
        reverse=True,
    )

    def rebuild_from(names: list[str]) -> list[SquadPlayer]:
        return [pool_by_name[n] for n in names if n in pool_by_name]

    names = [p.name for p in xi]

    for _ in range(240):
        cur = rebuild_from(names)
        ok, errs = _validate_xi(cur)
        if ok:
            return cur
        # Remove weakest fixable player and try replacements
        if any("Overseas" in e for e in errs):
            os_players = [p for p in cur if p.is_overseas]
            if not os_players:
                break
            drop = min(os_players, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name]
            nxt = next((q.name for q in order if q.name not in names and not q.is_overseas), None)
            if nxt:
                names.append(nxt)
            continue
        if any("wicketkeeper" in e for e in errs):
            wks = [q for q in order if q.is_wicketkeeper and q.name not in names]
            if not wks:
                break
            add = wks[0].name
            non_essential = [p for p in cur if not p.is_wicketkeeper]
            if not non_essential:
                break
            drop = min(non_essential, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name] + [add]
            continue
        if any("Bowling options" in e for e in errs):
            bow_candidates = [q for q in order if _is_bowling_option(q) and q.name not in names]
            if not bow_candidates:
                break
            add = bow_candidates[0].name
            non_bowlers = [p for p in cur if not _is_bowling_option(p) and not p.is_wicketkeeper]
            if not non_bowlers:
                non_bowlers = [p for p in cur if not _is_bowling_option(p)]
            if not non_bowlers:
                break
            drop = min(non_bowlers, key=lambda x: rs(x))
            names = [n for n in names if n != drop.name] + [add]
            continue
        if any("Batting depth" in e for e in errs):
            bowlers_in = [p for p in cur if p.role_bucket == BOWLER]
            nb_outside = [q for q in order if q.role_bucket != BOWLER and q.name not in names]
            if bowlers_in and nb_outside:
                drop = min(bowlers_in, key=lambda x: rs(x)).name
                add = max(nb_outside, key=lambda x: rs(x)).name
                names = [n for n in names if n != drop] + [add]
            continue
        if any("Opener candidates" in e for e in errs):
            pool = [q for q in order if q.role_bucket in (BATTER, WK_BATTER) and q.name not in names]
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if p.role_bucket == BOWLER]
            if not drop_candidates:
                drop_candidates = [p for p in cur if p.role_bucket not in (BATTER, WK_BATTER)]
            if not drop_candidates:
                break
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("Finisher:" in e for e in errs):
            pool = [q for q in order if q.role_bucket == ALL_ROUNDER and q.name not in names]
            if not pool:
                pool = [
                    q
                    for q in order
                    if q.role_bucket == BOWLER and q.bat_skill >= 0.42 and q.name not in names
                ]
            if not pool:
                break
            add = pool[0].name
            drop_candidates = [p for p in cur if p.role_bucket == BATTER]
            if not drop_candidates:
                drop_candidates = [p for p in cur if p.role_bucket not in (ALL_ROUNDER, BOWLER)]
            if not drop_candidates:
                break
            drop = min(drop_candidates, key=lambda x: rs(x)).name
            names = [n for n in names if n != drop] + [add]
            continue
        if any("XI size" in e for e in errs):
            if len(names) < 11:
                for q in order:
                    if q.name not in names:
                        names.append(q.name)
                        break
            else:
                names = names[:11]
            continue
        break

    cur = rebuild_from(names)
    ok, _ = _validate_xi(cur)
    return cur if ok and len(cur) == 11 else None


def _xi_selection_tier(p: SquadPlayer) -> int:
    """Prefer specialists for initial fill; all-rounders next; bowlers last (IPL balance)."""
    if p.role_bucket == BATTER:
        return 3
    if p.role_bucket == WK_BATTER:
        return 3
    if p.role_bucket == ALL_ROUNDER:
        return 2
    return 1


def _derive_pattern_venue_keys(venue: VenueProfile, base_keys: list[str]) -> list[str]:
    """Match Stage-2 ``team_selection_patterns.venue_key`` (canonical hash of venue strings)."""
    out: list[str] = []
    seen: set[str] = set()
    for k in base_keys:
        s = str(k or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for attr in ("key", "display_name", "city"):
        v = getattr(venue, attr, None)
        if v and str(v).strip():
            ck = canonical_keys.canonical_player_key(str(v).strip())[:80]
            if ck and ck not in seen:
                seen.add(ck)
                out.append(ck)
    return out


def _precompute_xi_build_penalties(players: list[SquadPlayer]) -> dict[str, float]:
    """
    Soft penalties before greedy XI fill: dull the 5th+ similar bowler type so specialists
    win realistic construction ties without breaking overseas / WK repairs.
    """
    from collections import defaultdict

    rate = float(getattr(config, "STAGE3_SIMILAR_BOWLER_PENALTY", 0.026))
    by_type: dict[str, list[SquadPlayer]] = defaultdict(list)
    for p in players:
        if p.role_bucket != BOWLER:
            continue
        bt = (p.bowling_type or "pace_like").lower().strip() or "pace_like"
        by_type[bt].append(p)
    pen: dict[str, float] = {}
    for group in by_type.values():
        group.sort(key=lambda x: x.selection_score, reverse=True)
        for i, p in enumerate(group):
            if i >= 4:
                pen[p.name] = pen.get(p.name, 0.0) + rate * float(i - 3)
    return pen


def _refine_opener_finisher_from_derive(players: list[SquadPlayer]) -> None:
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        if not prof:
            continue
        try:
            ol = float(prof.get("opener_likelihood") or 0.0)
            fl = float(prof.get("finisher_likelihood") or 0.0)
        except (TypeError, ValueError):
            continue
        if p.role_bucket in (BATTER, WK_BATTER) and ol >= 0.55:
            p.is_opener_candidate = True
        if fl >= 0.57:
            p.is_finisher_candidate = True


def _annotate_bench_xi_margins(squad: list[SquadPlayer], xi: list[SquadPlayer]) -> None:
    xi_set = {p.name for p in xi}
    if len(xi) != 11:
        return
    xi_sorted = sorted(xi, key=lambda p: p.selection_score)
    worst = float(xi_sorted[0].selection_score) if xi_sorted else 0.0
    for p in squad:
        if p.name in xi_set:
            continue
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        p.history_debug["bench_near_xi_margin"] = float(p.selection_score - worst)


def _batting_order_signal_source_ranked(p: SquadPlayer) -> list[str]:
    hd = getattr(p, "history_debug", None) or {}
    src = str(hd.get("batting_order_source") or "").lower()
    final = str(hd.get("batting_order_final") or "")
    ranks: list[str] = []
    if final == "historical_ema_primary":
        ranks.append("batting_position_ema")
    if "h2h" in src:
        ranks.append("head_to_head_slot_blend")
    if "global" in src or "all_franchise" in src:
        ranks.append("global_batting_slot_history")
    prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
    if prof:
        try:
            if float(prof.get("opener_likelihood") or 0) >= 0.55:
                ranks.append("derive_opener_likelihood")
            if float(prof.get("finisher_likelihood") or 0) >= 0.55:
                ranks.append("derive_finisher_likelihood")
            sm = int(prof.get("sample_matches") or 0)
            bpe = float(prof.get("batting_position_ema") or 0)
            if sm >= 3 and bpe > 0.5:
                ranks.append("derive_batting_position_ema")
        except (TypeError, ValueError):
            pass
    if final in ("role_fallback_proxy", "role_bucket_only"):
        ranks.append("role_bucket_fallback")
    if not ranks:
        ranks.append("role_bucket_fallback")
    dedup: list[str] = []
    seen: set[str] = set()
    for r in ranks:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    return dedup


def _batting_order_reason_summary_for_player(p: SquadPlayer, ranked: list[str]) -> str:
    hd = getattr(p, "history_debug", None) or {}
    ema = float(getattr(p, "history_batting_ema", float(config.HISTORY_BAT_SLOT_UNKNOWN)))
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    parts = [
        f"Ordered using: {' > '.join(ranked)}.",
    ]
    if ema < unk - 1e-6:
        parts.append(f"Primary numeric slot signal ≈ {ema:.2f} (lower bats earlier).")
    else:
        parts.append("No stable scorecard EMA for this XI player; derive opener/finisher hints and role buckets fill gaps.")
    if isinstance(hd.get("derive_player_profile"), dict):
        prof = hd["derive_player_profile"]
        try:
            ol = float(prof.get("opener_likelihood") or 0)
            fl = float(prof.get("finisher_likelihood") or 0)
            if ol >= 0.55 or fl >= 0.55:
                parts.append(
                    f"Stage-2 profile suggests opener={ol:.2f}, finisher={fl:.2f} (0–1 scale)."
                )
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def select_playing_xi(
    scored: list[SquadPlayer],
    *,
    scenario_branch: Optional[str] = None,
) -> list[SquadPlayer]:
    pen = _precompute_xi_build_penalties(scored)
    rs = lambda x: _scenario_xi_rank_value(x, scenario_branch)
    order = sorted(
        scored,
        key=lambda x: (
            rs(x) - pen.get(x.name, 0.0),
            _xi_selection_tier(x),
            x.composite,
        ),
        reverse=True,
    )
    xi = _try_build_xi(order, scored, pen, scenario_branch=scenario_branch)
    if xi:
        sk = _squad_player_key_set(scored)
        for p in xi:
            pk = (getattr(p, "player_key", None) or "").strip() or learner.normalize_player_key(p.name)
            if pk not in sk:
                logger.error(
                    "select_playing_xi: strict invariant broken — %r not in current squad keys",
                    p.name,
                )
        logger.info(
            "select_playing_xi: selected XI names=%s role_buckets=%s",
            [p.name for p in xi],
            [p.role_bucket for p in xi],
        )
        return xi
    if len(scored) < 11:
        fb = sorted(scored, key=lambda x: (rs(x), x.composite), reverse=True)
        logger.warning("select_playing_xi: short squad len=%d", len(scored))
        return fb
    fb = order[:11]
    _, errs = _validate_xi(fb)
    logger.warning("select_playing_xi: constraint repair failed; using top-11 fallback errs=%s", errs)
    return fb


def _build_batting_order_role_fallback(xi: list[SquadPlayer], conditions: dict[str, Any]) -> list[str]:
    """
    Legacy IPL role-bucket batting stack: openers → top batters → batting AR → WK slot
    → lower AR / finishers → bowlers last. Used when scorecard history is missing per player.
    """
    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])

    if len(xi) != 11:
        out = [p.name for p in sorted(xi, key=lambda x: (-x.bat_skill, x.name))]
        logger.warning("batting_order: incomplete XI len=%d", len(xi))
        return out

    specialists = [p for p in xi if p.role_bucket == BATTER]
    wks = [p for p in xi if p.role_bucket == WK_BATTER]
    ars = [p for p in xi if p.role_bucket == ALL_ROUNDER]
    bows = [p for p in xi if p.role_bucket == BOWLER]

    openers: list[SquadPlayer] = []
    used: set[str] = set()

    def _opener_rank_key(p: SquadPlayer) -> tuple[float, float, str]:
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        try:
            ol = float(prof.get("opener_likelihood") or 0.0) if prof else 0.0
        except (TypeError, ValueError):
            ol = 0.0
        return (-ol, -p.bat_skill, p.name)

    if specialists:
        openers = sorted(specialists, key=_opener_rank_key)[:2]
    for p in openers:
        used.add(p.name)

    if len(openers) < 2:
        for p in sorted([x for x in wks if x.name not in used], key=lambda q: q.bat_skill, reverse=True):
            if len(openers) >= 2:
                break
            openers.append(p)
            used.add(p.name)

    if len(openers) < 2 and not specialists:
        openers = sorted(wks, key=lambda p: p.bat_skill, reverse=True)[:2]
        used = {p.name for p in openers}

    rest_spec = sorted(
        [p for p in specialists if p.name not in used],
        key=lambda p: p.bat_skill,
        reverse=True,
    )

    ar_middle = sorted(
        [p for p in ars if p.bat_skill >= p.bowl_skill + 0.04],
        key=lambda p: p.bat_skill,
        reverse=True,
    )
    ar_lower = [p for p in ars if p not in ar_middle]

    rest_wk = sorted([p for p in wks if p.name not in used], key=lambda p: p.bat_skill, reverse=True)

    middle_line: list[SquadPlayer] = []
    middle_line.extend(openers)
    middle_line.extend(rest_spec)
    middle_line.extend(ar_middle)

    insert_at = min(max(3, int(3 + bf)), len(middle_line))
    for w in rest_wk:
        idx = min(insert_at, len(middle_line))
        middle_line.insert(idx, w)
        insert_at = idx + 2

    def fin_key(p: SquadPlayer) -> float:
        return p.bat_skill * (0.55 + 0.2 * dew) + p.bowl_skill * (0.45 + 0.15 * (1.0 - dew))

    ar_tail = sorted(ar_lower, key=fin_key, reverse=True)
    bow_tail = sorted(bows, key=lambda p: p.bowl_skill, reverse=True)

    ordered_players = middle_line + ar_tail + bow_tail

    logger.info(
        "batting_order: groups openers=%s spec_mid=%s ar_mid=%s ar_tail=%s bowlers=%s",
        [p.name for p in openers],
        [p.name for p in rest_spec],
        [p.name for p in ar_middle],
        [p.name for p in ar_tail],
        [p.name for p in bow_tail],
    )

    seen: set[str] = set()
    final: list[str] = []
    for p in ordered_players:
        if p.name not in seen:
            seen.add(p.name)
            final.append(p.name)
    for p in xi:
        if p.name not in seen:
            final.append(p.name)
            seen.add(p.name)
    return final


def _batting_order_for_short_xi(
    xi: list[SquadPlayer],
    *,
    team_name: str,
    out_warnings: Optional[list[str]] = None,
) -> list[str]:
    names = [p.name for p in sorted(xi, key=lambda x: (-x.bat_skill, x.name))]
    strict, w = _batting_order_strict_names_for_xi(xi, names)
    if out_warnings is not None:
        out_warnings.extend(w)
    for i, n in enumerate(strict):
        p = next((q for q in xi if q.name == n), None)
        if p is None:
            continue
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        p.history_debug["batting_order_strict_xi_scope"] = True
        p.history_debug["batting_order_rank_final"] = i + 1
        p.history_debug["batting_order_diagnostic_source"] = "role_fallback"
    logger.warning("batting_order: incomplete XI len=%d team=%s", len(xi), team_name)
    return strict


def build_batting_order(
    xi: list[SquadPlayer],
    conditions: dict[str, Any],
    *,
    team_name: str,
    venue_keys: list[str],
    out_warnings: Optional[list[str]] = None,
) -> list[str]:
    """
    Primary sort key is historical batting slot (EMA from stored scorecards) for **XI players only**.

    Output names are strictly a permutation of the selected XI (no non-squad / historical-only names).
    """
    unk = float(config.HISTORY_BAT_SLOT_UNKNOWN)
    if len(xi) != 11:
        return _batting_order_for_short_xi(xi, team_name=team_name, out_warnings=out_warnings)

    fb = _build_batting_order_role_fallback(xi, conditions)
    fr = {n: i for i, n in enumerate(fb)}

    def slot_key(p: SquadPlayer) -> tuple[float, str]:
        ema = float(getattr(p, "history_batting_ema", unk))
        hd = getattr(p, "history_debug", None) or {}
        prof = hd.get("derive_player_profile") if isinstance(hd.get("derive_player_profile"), dict) else None
        ol, fl, w_prof, d_ema = 0.5, 0.5, 0.0, 0.0
        if prof:
            try:
                ol = float(prof.get("opener_likelihood") or 0.5)
                fl = float(prof.get("finisher_likelihood") or 0.5)
                w_prof = max(0.0, min(0.22, float(prof.get("profile_confidence") or 0.0) * 0.35))
                d_ema = float(prof.get("batting_position_ema") or 0.0)
            except (TypeError, ValueError):
                pass
        if ema < unk - 1e-6:
            adj = ema - 0.62 * max(0.0, ol - 0.53) * 2.85
            adj += 0.52 * max(0.0, fl - 0.54) * 2.55
            if d_ema > 0.5 and w_prof > 0.02:
                adj = (1.0 - w_prof) * adj + w_prof * d_ema
            return (adj, p.name)
        fb_rank = float(fr.get(p.name, 99)) + 1.0
        if p.role_bucket == BOWLER:
            fb_rank = max(fb_rank, 8.0)
        elif p.role_bucket == ALL_ROUNDER:
            fb_rank = max(fb_rank, 3.25)
        fb_rank -= 0.48 * max(0.0, ol - 0.53)
        fb_rank += 0.44 * max(0.0, fl - 0.54)
        if d_ema > 0.5 and w_prof > 0.02:
            fb_rank = (1.0 - w_prof) * fb_rank + w_prof * d_ema
        return (fb_rank, p.name)

    ordered = sorted(xi, key=slot_key)
    any_hist = any(float(getattr(p, "history_batting_ema", unk)) < unk - 1e-6 for p in xi)
    candidate = [p.name for p in ordered]
    strict_names, bo_w = _batting_order_strict_names_for_xi(xi, candidate)
    if out_warnings is not None:
        out_warnings.extend(bo_w)

    by_name = {p.name: p for p in xi}
    n_role_fb = 0
    for i, name in enumerate(strict_names):
        p = by_name.get(name)
        if p is None:
            continue
        ema = float(getattr(p, "history_batting_ema", unk))
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        if ema < unk - 1e-6:
            p.history_debug["batting_order_final"] = "historical_ema_primary"
        elif any_hist:
            p.history_debug["batting_order_final"] = "role_fallback_proxy"
        else:
            p.history_debug["batting_order_final"] = "role_bucket_only"
        p.history_debug["batting_order_rank_final"] = i + 1
        p.history_debug["batting_order_strict_xi_scope"] = True
        diag = _batting_order_diagnostic_source(p)
        p.history_debug["batting_order_diagnostic_source"] = diag
        ranked = _batting_order_signal_source_ranked(p)
        p.history_debug["batting_order_signal_source_ranked"] = ranked
        p.history_debug["batting_order_reason_summary"] = _batting_order_reason_summary_for_player(p, ranked)
        if diag == "role_fallback":
            n_role_fb += 1

    logger.info(
        "batting_order: team=%s venue_keys=%s any_hist=%s strict_order=%s role_fallback_xi_players=%d",
        team_name,
        (venue_keys or [])[:3],
        any_hist,
        strict_names,
        n_role_fb,
    )
    if n_role_fb >= 6:
        msg = (
            f"{team_name}: {n_role_fb}/11 XI players lack stored batting-slot history (role_fallback). "
            "SQLite ``team_match_xi`` slot coverage is thin for this lineup. "
            f"{history_sync.HISTORY_MISSING_USER_MESSAGE}"
        )
        if out_warnings is not None:
            out_warnings.append(msg)
        logger.warning("batting_order: %s", msg)
    return strict_names


def impact_subs(
    squad: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    is_chasing: Optional[bool] = None,
    conditions: Optional[dict[str, Any]] = None,
    team_bats_first: Optional[bool] = None,
    team_display_name: str = "",
    canonical_team_key: str = "",
    venue_key: str = "",
    venue_key_candidates: Optional[list[str]] = None,
) -> tuple[list[SquadPlayer], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Bench ranking for impact subs: team patterns, game-state scenarios, tactical role cases,
    plus history/H2H/phase signals (see ``impact_subs_engine``).

    ``team_bats_first``: True if this **squad's** team bats first in the hypothetical fixture.

    Returns ``(top5_players, top5_debug_rows, all_bench_debug_rows)``.
    """
    subs, dbg_rows, dbg_all = impact_subs_engine.rank_impact_sub_candidates(
        squad,
        xi,
        team_display_name=team_display_name,
        canonical_team_key=canonical_team_key or "",
        venue_key=venue_key or "",
        venue_key_candidates=venue_key_candidates,
        is_chasing=is_chasing,
        conditions=conditions or {},
        team_bats_first=team_bats_first,
    )
    return subs, dbg_rows, dbg_all


def _impact_subs_payload(
    subs: list[SquadPlayer], dbg: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, p in enumerate(subs):
        row: dict[str, Any] = {
            "name": p.name,
            "role": p.role,
            "role_bucket": p.role_bucket,
            "composite": round(p.composite, 4),
            "is_overseas": p.is_overseas,
            "selection_score": round(p.selection_score, 5),
        }
        if i < len(dbg):
            row.update(dbg[i])
        out.append(row)
    return out


def _merge_player_match_stats_into_debug(players: list[SquadPlayer], team_key: str) -> None:
    fk = (team_key or "").strip()
    if not fk:
        return
    qkeys: list[str] = []
    for p in players:
        pk = learner.normalize_player_key(p.name)
        hd = getattr(p, "history_debug", None) or {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        hk = lk.get("history_lookup_key") or lk.get("resolved_history_key")
        qk = str(hk).strip() if hk else pk
        qkeys.append(qk)
    uniq = list(dict.fromkeys([k for k in qkeys if k]))
    counts = db.batch_player_match_stats_counts(uniq, fk)
    for p, qk in zip(players, qkeys):
        pk = learner.normalize_player_key(p.name)
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd = p.history_debug
        n = int(counts.get(qk, 0))
        hu = hd.get("history_usage_debug")
        if isinstance(hu, dict):
            hu2 = dict(hu)
            hu2["player_match_stats_row_count"] = n
            hd["history_usage_debug"] = hu2
        else:
            hd["player_match_stats_row_count"] = n


def _squad_post_linkage_history_rows(players: list[SquadPlayer]) -> list[dict[str, Any]]:
    """
    Post–Stage F / post-scoring squad: merges ``history_usage_debug`` with final
    ``history_linkage`` and top-level linkage flags (same objects attached after
    global fallback and collision handling).
    """
    rows: list[dict[str, Any]] = []
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        hu = hd.get("history_usage_debug")
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        row: dict[str, Any] = {}
        if isinstance(hu, dict):
            row.update(dict(hu))
        row["squad_display_name"] = p.name
        row["player_name"] = p.name
        row["canonical_player_key"] = p.player_key or learner.normalize_player_key(p.name)
        for k in (
            "history_lookup_key",
            "collision_resolution_outcome",
            "collided_history_key",
            "collision_group_members",
            "rolled_up_interpretation",
            "history_status",
            "resolution_type",
        ):
            if k in lk:
                row[k] = lk.get(k)
        row["global_resolved_history_key"] = lk.get("global_resolved_history_key") or hd.get(
            "global_resolved_history_key"
        )
        row["used_global_resolved_key_for_prior"] = lk.get("used_global_resolved_key_for_prior")
        if row.get("used_global_resolved_key_for_prior") is None:
            row["used_global_resolved_key_for_prior"] = hd.get("used_global_resolved_key_for_prior")
        row["collision_resolution_winner_name"] = lk.get("collision_winner_player_name")
        row["likely_first_ipl_player"] = lk.get("likely_first_ipl_player")
        if row.get("likely_first_ipl_player") is None:
            row["likely_first_ipl_player"] = hd.get("likely_first_ipl_player")
        row["debutant_alias_suppression_applied"] = lk.get("debutant_alias_suppression_applied")
        if row.get("debutant_alias_suppression_applied") is None:
            row["debutant_alias_suppression_applied"] = hd.get("debutant_alias_suppression_applied")
        row["debutant_alias_rejection_reason"] = lk.get("debutant_alias_rejection_reason")
        if row.get("debutant_alias_rejection_reason") is None:
            row["debutant_alias_rejection_reason"] = hd.get("debutant_alias_rejection_reason")
        rows.append(row)
    return rows


def _history_linkage_squad_rollup(players: list[SquadPlayer]) -> dict[str, Any]:
    from collections import Counter

    res_t = Counter()
    coll_o = Counter()
    for p in players:
        lk = (getattr(p, "history_debug", None) or {}).get("history_linkage")
        if not isinstance(lk, dict):
            continue
        rt = str(lk.get("resolution_type") or "unknown")
        res_t[rt] += 1
        co = str(lk.get("collision_resolution_outcome") or "unknown")
        coll_o[co] += 1
    summ = None
    if players:
        hd0 = getattr(players[0], "history_debug", None) or {}
        summ = hd0.get("history_linkage_team_summary")
    return {
        "player_count": len(players),
        "resolution_type_counts": dict(res_t),
        "collision_resolution_outcome_counts": dict(coll_o),
        "history_linkage_team_summary": summ,
    }


def _squad_scoring_breakdown_rows(players: list[SquadPlayer]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in players:
        hd = getattr(p, "history_debug", None) or {}
        sb = hd.get("scoring_breakdown") if isinstance(hd.get("scoring_breakdown"), dict) else {}
        ssc = hd.get("selection_score_components") if isinstance(hd.get("selection_score_components"), dict) else {}
        row: dict[str, Any] = {
            "squad_display_name": p.name,
            "player_name": p.name,
            "canonical_player_key": p.player_key or learner.normalize_player_key(p.name),
        }
        row.update(sb)
        row["final_selection_score"] = row.get("final_selection_score", ssc.get("selection_score"))
        row["history_weight_applied"] = ssc.get("history_weight_applied")
        row["has_usable_history"] = ssc.get("has_usable_sqlite_or_cricsheet_history")
        row["probable_first_choice_prior"] = hd.get("probable_first_choice_prior")
        row["global_ipl_history_presence"] = hd.get("global_ipl_history_presence")
        row["global_selection_frequency"] = hd.get("global_selection_frequency")
        row["used_global_fallback_prior"] = hd.get("used_global_fallback_prior")
        row["captain_boost_applied"] = (sb.get("captain_boost_applied") if isinstance(sb, dict) else None) or hd.get(
            "captain_boost_applied"
        )
        row["wicketkeeper_boost_applied"] = (
            (sb.get("wicketkeeper_boost_applied") if isinstance(sb, dict) else None)
            or hd.get("wicketkeeper_boost_applied")
        )
        row["valid_current_squad_new_to_franchise"] = hd.get("valid_current_squad_new_to_franchise")
        row["wrong_side_squad_assignment"] = hd.get("wrong_side_squad_assignment")
        row["selection_reason_summary"] = hd.get("selection_reason_summary")
        row["history_source_used"] = hd.get("history_source_used")
        row["fallback_used"] = hd.get("fallback_used")
        row["derive_player_profile"] = hd.get("derive_player_profile")
        sm_dbg = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else None
        if sm_dbg is not None:
            row["tactical_adjustment_total"] = sm_dbg.get("tactical_adjustment_total")
            rf_det = sm_dbg.get("recent_form_detail") if isinstance(sm_dbg.get("recent_form_detail"), dict) else {}
            comps = rf_det.get("competitions_used")
            if isinstance(comps, list) and comps:
                row["recent_form_competitions_used"] = ", ".join(str(c) for c in comps[:20])
            elif isinstance(comps, list):
                row["recent_form_competitions_used"] = ""
            ex = sm_dbg.get("explainability") if isinstance(sm_dbg.get("explainability"), dict) else {}
            line = " | ".join(
                str(x).strip()
                for x in (
                    ex.get("role_reason"),
                    ex.get("recent_form_reason"),
                    ex.get("ipl_role_history_reason"),
                    ex.get("team_balance_reason"),
                    ex.get("venue_reason"),
                )
                if x
            )
            row["selection_model_explain_line"] = (line[:420] + "…") if len(line) > 420 else line
        out.append(row)
    return out


def _omitted_xi_report(
    scored: list[SquadPlayer],
    xi: list[SquadPlayer],
    *,
    scenario_branch: Optional[str] = None,
    xi_baseline: Optional[list[SquadPlayer]] = None,
    top_n: int = 14,
) -> list[dict[str, Any]]:
    """
    Omitted players ranked with the same effective key as ``select_playing_xi``:
    scenario branch score minus precomputed XI-build penalties.
    """
    xi_names = {p.name for p in xi}
    pen = _precompute_xi_build_penalties(scored)

    def eff_rank(p: SquadPlayer) -> float:
        return _scenario_xi_rank_value(p, scenario_branch) - pen.get(p.name, 0.0)

    order = sorted(
        scored,
        key=lambda x: (eff_rank(x), _xi_selection_tier(x), x.composite),
        reverse=True,
    )
    rank_by_name = {p.name: i + 1 for i, p in enumerate(order)}
    cut_eff = eff_rank(order[10]) if len(order) >= 11 else 0.0
    first11_scen = {p.name for p in order[:11]}
    base_xi_names = {p.name for p in xi_baseline} if xi_baseline else set()
    pool = [p for p in scored if p.name not in xi_names]
    pool.sort(key=lambda p: (eff_rank(p), p.composite), reverse=True)
    out: list[dict[str, Any]] = []
    for p in pool[:top_n]:
        hd = getattr(p, "history_debug", None) or {}
        lk = hd.get("history_linkage") if isinstance(hd.get("history_linkage"), dict) else {}
        rt = lk.get("resolution_type") or ""
        rk = rank_by_name.get(p.name)
        scen_gap = float(cut_eff - eff_rank(p)) if len(order) >= 11 else 0.0
        omitted_due_to_constraint = (p.name in first11_scen) and (p.name not in xi_names)
        omitted_due_to_scenario_rank = (
            scenario_branch in ("if_team_bats_first", "if_team_bowls_first")
            and bool(base_xi_names)
            and p.name in base_xi_names
            and p.name not in xi_names
        )
        reasons: list[str] = []
        if rk is not None and rk > 11:
            reasons.append(f"squad_selection_rank_{rk}")
        if len(order) >= 11 and scen_gap > 1e-7:
            reasons.append(f"scenario_eff_rank_gap_vs_rank11_{scen_gap:.5f}")
        if omitted_due_to_constraint:
            reasons.append("omitted_due_to_xi_constraint_repair")
        if omitted_due_to_scenario_rank:
            reasons.append("omitted_due_to_scenario_branch_vs_baseline_xi")
        if int(hd.get("history_rows_found") or 0) < 2:
            reasons.append("low_team_match_xi_rows")
        if int(hd.get("franchise_history_distinct_matches") or 0) <= 2:
            reasons.append("low_franchise_history_distinct_matches")
        if rt in ("no_match", "ambiguous_alias"):
            reasons.append("stage_f_alias_miss_or_ambiguous")
        if p.is_overseas:
            reasons.append("overseas_player_constraints_may_apply")
        if p.role_bucket == BOWLER and sum(1 for x in xi if x.role_bucket == BOWLER) >= 5:
            reasons.append("bowling_slot_competition")
        p_prior = float(hd.get("probable_first_choice_prior") or 0.0)
        if p_prior >= 0.48 and bool(hd.get("used_global_fallback_prior")):
            reasons.append("notable_strong_probable_first_choice_prior_still_omitted")
        if bool(hd.get("captain_selected_for_team")):
            reasons.append("captain_omitted_despite_manual_selection")
        if bool(hd.get("wicketkeeper_selected_for_team")):
            reasons.append("wicketkeeper_omitted_despite_manual_selection")
        omitted_reason_summary = "; ".join(reasons) if reasons else "not_in_top_11_by_xi_model"
        out.append(
            {
                "name": p.name,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "role_bucket": p.role_bucket,
                "selection_score": round(p.selection_score, 5),
                "scenario_rank_used": round(_scenario_xi_rank_value(p, scenario_branch), 5),
                "effective_xi_rank_score": round(eff_rank(p), 5),
                "composite": round(p.composite, 5),
                "squad_selection_rank": rk,
                "scenario_score_gap_vs_cutoff": round(scen_gap, 5),
                "score_gap_vs_cutoff": round(scen_gap, 5),
                "omitted_due_to_constraint": omitted_due_to_constraint,
                "omitted_due_to_scenario_rank": omitted_due_to_scenario_rank,
                "omitted_reason_summary": omitted_reason_summary,
                "omission_reason": omitted_reason_summary,
                "probable_first_choice_prior": round(p_prior, 5),
                "global_ipl_history_presence": hd.get("global_ipl_history_presence"),
                "global_selection_frequency": hd.get("global_selection_frequency"),
                "used_global_fallback_prior": bool(hd.get("used_global_fallback_prior")),
                "reasons_not_in_playing_xi": reasons,
            }
        )
    return out


def _xi_strength(xi: list[SquadPlayer]) -> float:
    if not xi:
        return 0.0
    w = float(getattr(config, "STAGE3_XI_STRENGTH_SELECTION_BLEND", 0.34))
    w = max(0.0, min(0.78, w))
    acc = 0.0
    for p in xi:
        comp = float(p.composite)
        sel = float(getattr(p, "selection_score", comp))
        acc += (1.0 - w) * comp + w * sel
    return acc / max(1, len(xi)) * 11.0


def toss_matchup_effects(
    team_a: str,
    team_b: str,
    *,
    strength_a: float,
    strength_b: float,
    conditions: dict[str, Any],
    venue_key: str,
) -> dict[str, Any]:
    bf = float(conditions["batting_friendliness"])
    dew = float(conditions["dew_risk"])
    rain = float(conditions["rain_disruption_risk"])

    edge_a = learner.venue_toss_edge(venue_key, team_a)
    edge_b = learner.venue_toss_edge(venue_key, team_b)

    # Relative chase/defend strength (unitless)
    bat_first_boost = 0.18 * (bf - 0.5) - 0.12 * dew - 0.22 * rain
    chase_boost = -bat_first_boost + 0.14 * dew

    rel = (strength_a - strength_b) / 11.0
    bat_first_a = rel + bat_first_boost + edge_a["bat_first_logit"] - edge_b["bowl_first_logit"]
    bowl_first_a = -rel + chase_boost + edge_a["bowl_first_logit"] - edge_b["bat_first_logit"]

    return {
        "bat_first_edge_team_a": float(bat_first_a),
        "bowl_first_edge_team_a": float(bowl_first_a),
        "venue_bat_first_prior": bat_first_boost,
        "dew_chase_factor": chase_boost,
        "team_a_bat_first_samples": edge_a["sample_bat"],
        "team_a_bowl_first_samples": edge_a["sample_bowl"],
    }


def win_probability(
    strength_a: float,
    strength_b: float,
    conditions: dict[str, Any],
    toss_effects: dict[str, Any],
    *,
    assume_team_a_bats_first: Optional[bool] = None,
    chase_boost_logit: float = 0.0,
) -> dict[str, Any]:
    """
    If assume_team_a_bats_first is None, marginalize toss with 0.5/0.5.
    chase_boost_logit: venue chase prior applied when A chases; half applied when toss unknown.
    """
    rain = float(conditions["rain_disruption_risk"])
    base = (strength_a - strength_b) / config.WIN_MODEL_TEMPERATURE
    base -= 0.35 * rain * math.copysign(1.0, strength_a - strength_b)

    if assume_team_a_bats_first is None:
        e_bf = float(toss_effects["bat_first_edge_team_a"])
        e_ch = float(toss_effects["bowl_first_edge_team_a"])
        logit = base + 0.5 * e_bf + 0.5 * e_ch + 0.5 * chase_boost_logit
    elif assume_team_a_bats_first:
        logit = base + float(toss_effects["bat_first_edge_team_a"])
    else:
        logit = base + float(toss_effects["bowl_first_edge_team_a"]) + chase_boost_logit

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    p = sigmoid(logit)
    return {
        "team_a_win": float(p),
        "team_b_win": float(1.0 - p),
        "logit": float(logit),
    }


def run_prediction(
    team_a_name: str,
    team_b_name: str,
    squad_a_text: str,
    squad_b_text: str,
    unavailable_text: str,
    venue: VenueProfile,
    match_time: datetime,
    weather: dict[str, Any],
    *,
    toss_scenario_key: str = "unknown",
    team_a_captain_display_name: str = "",
    team_b_captain_display_name: str = "",
    team_a_wicketkeeper_display_name: str = "",
    team_b_wicketkeeper_display_name: str = "",
    team_a_fetched_squad_player_keys: Optional[set[str]] = None,
    team_b_fetched_squad_player_keys: Optional[set[str]] = None,
    team_a_stale_cached_player_keys: Optional[set[str]] = None,
    team_b_stale_cached_player_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    _audit_par = audit_profile.PredictionRunAudit() if audit_profile.audit_enabled() else None
    try:
        return _run_prediction_inner(
            team_a_name,
            team_b_name,
            squad_a_text,
            squad_b_text,
            unavailable_text,
            venue,
            match_time,
            weather,
            toss_scenario_key=toss_scenario_key,
            team_a_captain_display_name=team_a_captain_display_name,
            team_b_captain_display_name=team_b_captain_display_name,
            team_a_wicketkeeper_display_name=team_a_wicketkeeper_display_name,
            team_b_wicketkeeper_display_name=team_b_wicketkeeper_display_name,
            team_a_fetched_squad_player_keys=team_a_fetched_squad_player_keys,
            team_b_fetched_squad_player_keys=team_b_fetched_squad_player_keys,
            team_a_stale_cached_player_keys=team_a_stale_cached_player_keys,
            team_b_stale_cached_player_keys=team_b_stale_cached_player_keys,
            _audit_par=_audit_par,
        )
    except BaseException:
        if _audit_par:
            _audit_par.close_failure()
        raise


def _run_prediction_inner(
    team_a_name: str,
    team_b_name: str,
    squad_a_text: str,
    squad_b_text: str,
    unavailable_text: str,
    venue: VenueProfile,
    match_time: datetime,
    weather: dict[str, Any],
    *,
    toss_scenario_key: str = "unknown",
    team_a_captain_display_name: str = "",
    team_b_captain_display_name: str = "",
    team_a_wicketkeeper_display_name: str = "",
    team_b_wicketkeeper_display_name: str = "",
    team_a_fetched_squad_player_keys: Optional[set[str]] = None,
    team_b_fetched_squad_player_keys: Optional[set[str]] = None,
    team_a_stale_cached_player_keys: Optional[set[str]] = None,
    team_b_stale_cached_player_keys: Optional[set[str]] = None,
    _audit_par: Any = None,
) -> dict[str, Any]:
    _run_t0 = time.perf_counter()
    prediction_timing_ms: dict[str, float] = {}

    def _ap_phase(name: str, t0: float) -> None:
        if audit_profile.audit_enabled():
            audit_profile.record_prediction_phase(name, (time.perf_counter() - t0) * 1000.0)

    def _fetched_key_set(keys: Optional[set[str]]) -> FrozenSet[str]:
        if not keys:
            return frozenset()
        out: set[str] = set()
        for k in keys:
            s = str(k).strip()
            if s:
                out.add(s)
        return frozenset(out)

    _t_early = time.perf_counter()
    conditions = venue_conditions_summary(venue, weather)

    sa = filter_unavailable(parse_squad_text(squad_a_text), unavailable_text)
    sb = filter_unavailable(parse_squad_text(squad_b_text), unavailable_text)
    role_dist_a = _log_role_bucket_distribution(sa, "team_a")
    role_dist_b = _log_role_bucket_distribution(sb, "team_b")
    canon_a = ipl_teams.franchise_label_for_storage(team_a_name) or team_a_name.strip()
    canon_b = ipl_teams.franchise_label_for_storage(team_b_name) or team_b_name.strip()
    slug_a = (ipl_teams.slug_for_canonical_label(canon_a) or "").strip()
    slug_b = (ipl_teams.slug_for_canonical_label(canon_b) or "").strip()
    fetch_sel_a = _fetched_key_set(team_a_fetched_squad_player_keys)
    fetch_sel_b = _fetched_key_set(team_b_fetched_squad_player_keys)
    fetch_opp_a = fetch_sel_b
    fetch_opp_b = fetch_sel_a
    stale_a = _fetched_key_set(team_a_stale_cached_player_keys)
    stale_b = _fetched_key_set(team_b_stale_cached_player_keys)
    _annotate_squad_canonical_keys(sa, canon_a)
    _annotate_squad_canonical_keys(sb, canon_b)
    # Bulk Cricsheet JSON → SQLite runs in the **ingest** stage only (see ``cricsheet_ingest``).
    # ``run_prediction`` reads existing SQLite rows via ``history_xi`` / ``db`` — no on-the-fly JSON ingest.
    history_sync_debug: dict[str, Any] = {
        "local_history_warning": None,
        "team_a": {},
        "team_b": {},
        "h2h_layer": {},
    }

    try:
        history_sync_debug["team_a"] = history_sync.local_history_debug_for_prediction(
            canon_a,
            squad_player_names=[p.name for p in sa],
            include_squad_report=False,
        )
        history_sync_debug["team_b"] = history_sync.local_history_debug_for_prediction(
            canon_b,
            squad_player_names=[p.name for p in sb],
            include_squad_report=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Local history snapshot failed; continuing with heuristics: %s", exc)
        history_sync_debug["local_history_warning"] = (
            "Could not read local SQLite history metadata; prediction continues using squad heuristics. "
            + history_sync.HISTORY_MISSING_USER_MESSAGE
        )
        history_sync_debug["sync_exception"] = f"{type(exc).__name__}: {exc}"
        history_sync_debug["team_a"] = history_sync.failsafe_history_debug(canon_a, exc)
        history_sync_debug["team_b"] = history_sync.failsafe_history_debug(canon_b, exc)

    thin_msgs: list[str] = []
    for side in ("team_a", "team_b"):
        block = history_sync_debug.get(side) or {}
        for w in block.get("warnings") or []:
            if isinstance(w, str) and w.strip():
                thin_msgs.append(f"{side}: {w.strip()}")
    if thin_msgs and not history_sync_debug.get("local_history_warning"):
        a_ne = history_sync.raw_stage1_tables_near_empty(history_sync_debug.get("team_a") or {})
        b_ne = history_sync.raw_stage1_tables_near_empty(history_sync_debug.get("team_b") or {})
        if a_ne or b_ne:
            history_sync_debug["local_history_warning"] = history_sync.HISTORY_MISSING_USER_MESSAGE

    _ap_phase("parse_squads_franchise_slug_and_history_sync_ms", _t_early)

    _t_learn = time.perf_counter()
    learned_map = learner.load_learned_map()
    logger.info(
        "run_prediction: structured squads team_a=%d team_b=%d canonical=(%s,%s) preview_a=%s",
        len(sa),
        len(sb),
        canon_a,
        canon_b,
        [(p.name, p.role_bucket) for p in sa[:6]],
    )
    shape_a = _squad_shape(sa)
    shape_b = _squad_shape(sb)

    scored_a: list[SquadPlayer] = []
    for p in sa:
        sp = _score_player(
            p,
            self_shape=shape_a,
            opp_shape=shape_b,
            conditions=conditions,
            learned_map=learned_map,
            franchise_canonical=canon_a,
        )
        _set_player_ipl_flags(sp)
        scored_a.append(sp)
    scored_b = []
    for p in sb:
        sp = _score_player(
            p,
            self_shape=shape_b,
            opp_shape=shape_a,
            conditions=conditions,
            learned_map=learned_map,
            franchise_canonical=canon_b,
        )
        _set_player_ipl_flags(sp)
        scored_b.append(sp)

    _ap_phase("learned_map_load_shape_and_score_players_both_teams_ms", _t_learn)

    _t_hctx = time.perf_counter()
    hctx = build_history_context()
    vkeys = venue_lookup_keys(venue)
    pattern_vkeys = _derive_pattern_venue_keys(venue, vkeys)
    is_night = time_utils.ist_hour(match_time) >= config.NIGHT_START_HOUR_LOCAL
    dew_risk = float(conditions["dew_risk"])
    a_bats_first = resolve_a_bats_first_toss(toss_scenario_key)
    chase_ctx_a = chase_context_for_team(team_a_name, team_a_name, team_b_name, a_bats_first)
    chase_ctx_b = chase_context_for_team(team_b_name, team_a_name, team_b_name, a_bats_first)

    for p in scored_a:
        _history_adjust_for_player(p, team_a_name, shape_a, vkeys, is_night, dew_risk, hctx)
    for p in scored_b:
        _history_adjust_for_player(p, team_b_name, shape_b, vkeys, is_night, dew_risk, hctx)

    _ap_phase("build_history_context_and_history_rules_bump_ms", _t_hctx)

    h2h_layer: dict[str, Any] = {}
    _t_hist = time.perf_counter()
    history_xi.attach_primary_history_to_squad(
        scored_a,
        team_a_name,
        vkeys,
        shape=shape_a,
        chase_context=chase_ctx_a,
        opponent_canonical_label=canon_b,
        h2h_explain=h2h_layer,
        h2h_explain_scope="team_a",
        fetched_team_slug=slug_a,
        selected_fetched_player_keys=fetch_sel_a if fetch_sel_a else None,
        opposite_fetched_player_keys=fetch_opp_a,
        stale_cached_player_keys=stale_a,
        captain_display_name=team_a_captain_display_name or "",
        wicketkeeper_display_name=team_a_wicketkeeper_display_name or "",
    )
    fixture_ctx_a = {"is_night": is_night, "reference_iso_date": match_time.date().isoformat()}
    fixture_ctx_b = {"is_night": is_night, "reference_iso_date": match_time.date().isoformat()}
    if a_bats_first is not None:
        fixture_ctx_a["xi_scenario_branch_for_tactical"] = (
            "if_team_bats_first" if a_bats_first else "if_team_bowls_first"
        )
        fixture_ctx_b["xi_scenario_branch_for_tactical"] = (
            "if_team_bats_first" if not a_bats_first else "if_team_bowls_first"
        )
    history_xi.compute_selection_scores(
        scored_a,
        conditions=conditions,
        venue_key_candidates=pattern_vkeys,
        fixture_context=fixture_ctx_a,
    )
    _refine_opener_finisher_from_derive(scored_a)
    tk_a = ipl_teams.canonical_team_key_for_franchise(canon_a)
    _merge_player_match_stats_into_debug(scored_a, tk_a)
    history_xi.attach_primary_history_to_squad(
        scored_b,
        team_b_name,
        vkeys,
        shape=shape_b,
        chase_context=chase_ctx_b,
        opponent_canonical_label=canon_a,
        h2h_explain=h2h_layer,
        h2h_explain_scope="team_b",
        fetched_team_slug=slug_b,
        selected_fetched_player_keys=fetch_sel_b if fetch_sel_b else None,
        opposite_fetched_player_keys=fetch_opp_b,
        stale_cached_player_keys=stale_b,
        captain_display_name=team_b_captain_display_name or "",
        wicketkeeper_display_name=team_b_wicketkeeper_display_name or "",
    )
    history_xi.compute_selection_scores(
        scored_b,
        conditions=conditions,
        venue_key_candidates=pattern_vkeys,
        fixture_context=fixture_ctx_b,
    )
    _refine_opener_finisher_from_derive(scored_b)
    tk_b = ipl_teams.canonical_team_key_for_franchise(canon_b)
    _merge_player_match_stats_into_debug(scored_b, tk_b)
    _ap_phase("history_xi_attach_selection_scores_merge_pms_both_teams_ms", _t_hist)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["history_xi_and_selection_both_teams_ms"] = round(
            (time.perf_counter() - _t_hist) * 1000.0, 2
        )
    h2h_layer.pop("_h2h_layer_meta_done", None)
    history_sync_debug["h2h_layer"] = h2h_layer

    _t_xi_pick = time.perf_counter()
    br_a = franchise_xi_scenario_branch(True, a_bats_first)
    br_b = franchise_xi_scenario_branch(False, a_bats_first)
    xi_a_base = select_playing_xi(scored_a, scenario_branch=None)
    xi_a = select_playing_xi(scored_a, scenario_branch=br_a)
    _delta_scen_a = {p.name for p in xi_a}.symmetric_difference({p.name for p in xi_a_base})
    for p in scored_a:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd_a = p.history_debug
        hd_a["xi_scenario_branch_for_xi_build"] = br_a
        hd_a["scenario_rank_used"] = round(_scenario_xi_rank_value(p, br_a), 5)
        hd_a["selection_changed_due_to_scenario"] = p.name in _delta_scen_a

    xi_b_base = select_playing_xi(scored_b, scenario_branch=None)
    xi_b = select_playing_xi(scored_b, scenario_branch=br_b)
    _delta_scen_b = {p.name for p in xi_b}.symmetric_difference({p.name for p in xi_b_base})
    for p in scored_b:
        if not isinstance(getattr(p, "history_debug", None), dict):
            p.history_debug = {}
        hd_b = p.history_debug
        hd_b["xi_scenario_branch_for_xi_build"] = br_b
        hd_b["scenario_rank_used"] = round(_scenario_xi_rank_value(p, br_b), 5)
        hd_b["selection_changed_due_to_scenario"] = p.name in _delta_scen_b

    xi_a_if_bats_first: Optional[list[SquadPlayer]] = None
    xi_a_if_bowls_first: Optional[list[SquadPlayer]] = None
    xi_b_if_bats_first: Optional[list[SquadPlayer]] = None
    xi_b_if_bowls_first: Optional[list[SquadPlayer]] = None
    if a_bats_first is None:
        xi_a_if_bats_first = select_playing_xi(scored_a, scenario_branch="if_team_bats_first")
        xi_a_if_bowls_first = select_playing_xi(scored_a, scenario_branch="if_team_bowls_first")
        xi_b_if_bats_first = select_playing_xi(scored_b, scenario_branch="if_team_bats_first")
        xi_b_if_bowls_first = select_playing_xi(scored_b, scenario_branch="if_team_bowls_first")

    sub_ok_a, sub_err_a = _validate_xi_in_current_squad(xi_a, sa, "team_a")
    sub_ok_b, sub_err_b = _validate_xi_in_current_squad(xi_b, sb, "team_b")
    if not sub_ok_a:
        raise ValueError(
            f"Team A predicted XI includes player(s) not in the current squad (inner-join violation): {sub_err_a}"
        )
    if not sub_ok_b:
        raise ValueError(
            f"Team B predicted XI includes player(s) not in the current squad (inner-join violation): {sub_err_b}"
        )

    v_ok_a, v_err_a = _validate_xi(xi_a)
    v_ok_b, v_err_b = _validate_xi(xi_b)
    if not v_ok_a:
        logger.warning("team_a XI constraints failed: %s", v_err_a)
    if not v_ok_b:
        logger.warning("team_b XI constraints failed: %s", v_err_b)
    batting_order_warnings: list[str] = []
    order_a = (
        build_batting_order(
            xi_a,
            conditions,
            team_name=team_a_name,
            venue_keys=vkeys,
            out_warnings=batting_order_warnings,
        )
        if len(xi_a) == 11
        else _batting_order_for_short_xi(xi_a, team_name=team_a_name, out_warnings=batting_order_warnings)
    )
    order_b = (
        build_batting_order(
            xi_b,
            conditions,
            team_name=team_b_name,
            venue_keys=vkeys,
            out_warnings=batting_order_warnings,
        )
        if len(xi_b) == 11
        else _batting_order_for_short_xi(xi_b, team_name=team_b_name, out_warnings=batting_order_warnings)
    )
    if len(xi_a) == 11:
        _map_a = {p.name: p for p in xi_a}
        xi_a = [_map_a[n] for n in order_a if n in _map_a]
    if len(xi_b) == 11:
        _map_b = {p.name: p for p in xi_b}
        xi_b = [_map_b[n] for n in order_b if n in _map_b]

    strict_validation_warnings = _collect_strict_validation_warnings(
        team_a_label=team_a_name,
        team_b_label=team_b_name,
        xi_a=xi_a,
        xi_b=xi_b,
        scored_squad_a=scored_a,
        scored_squad_b=scored_b,
        history_sync_debug=history_sync_debug,
        batting_order_warnings=batting_order_warnings,
    )
    _annotate_bench_xi_margins(scored_a, xi_a)
    _annotate_bench_xi_margins(scored_b, xi_b)

    _ap_phase("xi_scenario_select_batting_order_validate_and_strict_checks_ms", _t_xi_pick)

    _t_imp = time.perf_counter()
    subs_a, impact_dbg_a, impact_dbg_all_a = impact_subs(
        scored_a,
        xi_a,
        is_chasing=chase_ctx_a,
        conditions=conditions,
        team_bats_first=a_bats_first,
        team_display_name=team_a_name,
        canonical_team_key=tk_a,
        venue_key=venue.key,
        venue_key_candidates=pattern_vkeys,
    )
    subs_b, impact_dbg_b, impact_dbg_all_b = impact_subs(
        scored_b,
        xi_b,
        is_chasing=chase_ctx_b,
        conditions=conditions,
        team_bats_first=(not a_bats_first) if a_bats_first is not None else None,
        team_display_name=team_b_name,
        canonical_team_key=tk_b,
        venue_key=venue.key,
        venue_key_candidates=pattern_vkeys,
    )
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["impact_subs_both_teams_ms"] = round(
            (time.perf_counter() - _t_imp) * 1000.0, 2
        )
    _ap_phase("impact_subs_both_teams_ms", _t_imp)

    _t_win_tail = time.perf_counter()
    str_a = _xi_strength(xi_a) if len(xi_a) == 11 else _xi_strength(xi_a) * 11 / max(1, len(xi_a))
    str_b = _xi_strength(xi_b) if len(xi_b) == 11 else _xi_strength(xi_b) * 11 / max(1, len(xi_b))

    toss = toss_matchup_effects(
        team_a_name,
        team_b_name,
        strength_a=str_a,
        strength_b=str_b,
        conditions=conditions,
        venue_key=venue.key,
    )
    chase_share, chase_n = _resolve_chase_prior(vkeys, hctx)
    chase_boost_logit = 0.0
    if chase_n >= config.LEARN_MIN_SAMPLES_CHASE:
        chase_boost_logit = (
            config.LEARN_WEIGHT_CHASE_BIAS_LOGIT * (chase_share - 0.5) * 2.0
        )
    toss["venue_chase_win_share"] = float(chase_share)
    toss["venue_chase_sample_wins"] = int(chase_n)
    toss["chase_boost_logit_applied"] = float(chase_boost_logit)

    win_marginal_logistic = win_probability(
        str_a,
        str_b,
        conditions,
        toss,
        assume_team_a_bats_first=a_bats_first,
        chase_boost_logit=chase_boost_logit,
    )
    win_a_bf_logistic = win_probability(
        str_a, str_b, conditions, toss, assume_team_a_bats_first=True, chase_boost_logit=0.0
    )
    win_a_ch_logistic = win_probability(
        str_a,
        str_b,
        conditions,
        toss,
        assume_team_a_bats_first=False,
        chase_boost_logit=chase_boost_logit,
    )

    _t_win = time.perf_counter()
    match_rows = db.fetch_match_results_meta(450)
    win_eng = win_probability_engine.compute_win_probability(
        team_a_name,
        team_b_name,
        xi_a,
        xi_b,
        order_a,
        order_b,
        venue,
        conditions,
        venue_keys=vkeys,
        match_rows=match_rows,
        toss_scenario_key=toss_scenario_key,
        a_bats_first_selected=a_bats_first,
        chase_share_by_venue=hctx.chase_share_by_venue,
        is_night_fixture=is_night,
    )
    eng_d = win_eng.to_dict()
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["win_engine_fetch_matches_and_compute_ms"] = round(
            (time.perf_counter() - _t_win) * 1000.0, 2
        )
    _ap_phase("win_engine_fetch_match_meta_and_compute_ms", _t_win)
    headline_a = eng_d["team_a_win_pct_selected_toss"] / 100.0
    neutral_a = eng_d["marginal_team_a_win_pct"] / 100.0
    sf0 = eng_d.get("scenario_factors") or {}
    fac_a = sf0.get("a_bats_first") or {}
    xi_f = fac_a.get("xi_strength") or {}
    vn_f = fac_a.get("venue") or {}
    h2_f = fac_a.get("head_to_head") or {}
    cw_f = fac_a.get("conditions_weather") or {}
    gap_xi = abs(float(xi_f.get("team_a", 0)) - float(xi_f.get("team_b", 0)))
    tilt_v = float(vn_f.get("team_a", 0)) - float(vn_f.get("team_b", 0))
    tilt_h = float(h2_f.get("team_a", 0)) - float(h2_f.get("team_b", 0))
    tilt_cw = float(cw_f.get("team_a", 0)) - float(cw_f.get("team_b", 0))
    dew_c = float(conditions.get("dew_risk", 0.5))
    driver_extra = (
        f" Weight readout (A bats first scenario slice): XI strength separation≈{gap_xi:.2f}; "
        f"venue history tilt A−B≈{tilt_v:.2f}; head-to-head tilt≈{tilt_h:.2f}; "
        f"conditions/weather tilt≈{tilt_cw:.2f} (fixture dew_risk≈{dew_c:.2f})."
    )
    win_marginal = {
        "team_a_win": float(headline_a),
        "team_b_win": float(1.0 - headline_a),
        "model": "rule_engine",
        "uses_selected_toss": a_bats_first is not None,
        "neutral_toss_team_a_win": float(neutral_a),
        "driver_summary": (str(eng_d.get("explanation") or "").strip() + driver_extra).strip()[:520],
    }
    win_a_bf = {
        "team_a_win": float(eng_d["team_a_win_pct_if_a_bats_first"] / 100.0),
        "team_b_win": float(eng_d["team_b_win_pct_if_a_bats_first"] / 100.0),
        "model": "rule_engine",
    }
    win_a_ch = {
        "team_a_win": float(eng_d["team_a_win_pct_if_b_bats_first"] / 100.0),
        "team_b_win": float(eng_d["team_b_win_pct_if_b_bats_first"] / 100.0),
        "model": "rule_engine",
    }

    confidence = _prediction_confidence(
        xi_a, xi_b, hctx, str_a, str_b, scored_a=scored_a, scored_b=scored_b
    )
    _ap_phase(
        "toss_logistic_win_probability_marginals_and_confidence_ms",
        _t_win_tail,
    )

    def _structured_squad_rows(squad: list[SquadPlayer]) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "squad_display_name": p.name,
                "player_name": p.name,
                "player_key": p.player_key or learner.normalize_player_key(p.name),
                "canonical_player_key": p.canonical_player_key or p.player_key or learner.normalize_player_key(p.name),
                "team_name": p.team_display_name,
                "canonical_team_key": p.canonical_team_key,
                "role_bucket": p.role_bucket,
                "overseas": p.is_overseas,
                "role": p.role,
            }
            for p in squad
        ]

    def xi_payload(
        xi: list[SquadPlayer], order: list[str], scenario_branch: str
    ) -> list[dict[str, Any]]:
        pos = {n: i + 1 for i, n in enumerate(order)}
        rows = []
        for p in xi:
            hd = dict(getattr(p, "history_debug", {}) or {})
            sx_br = (hd.get("scenario_xi") or {}).get(scenario_branch) or {}
            rows.append(
                {
                    "name": p.name,
                    "squad_display_name": p.name,
                    "player_key": p.player_key or learner.normalize_player_key(p.name),
                    "in_current_squad": hd.get("in_current_squad", True),
                    "role": p.role,
                    "role_bucket": p.role_bucket,
                    "batting_roles": list(p.batting_roles),
                    "bowling_type": p.bowling_type,
                    "overseas": p.is_overseas,
                    "bat_order": pos.get(p.name),
                    "batting_order_rank_final": hd.get("batting_order_rank_final"),
                    "composite": round(p.composite, 4),
                    "selection_score": round(float(getattr(p, "selection_score", 0.0)), 5),
                    "history_xi_score": round(float(getattr(p, "history_xi_score", 0.0)), 5),
                    "history_rows_found": hd.get("history_rows_found"),
                    "history_distinct_matches": hd.get("history_distinct_matches"),
                    "batting_positions_history": hd.get("batting_positions_history"),
                    "batting_position_rows_found": hd.get("batting_position_rows_found"),
                    "batting_position_ema": hd.get("batting_position_ema"),
                    "pbp_primary_slot_matches": hd.get("pbp_primary_slot_matches"),
                    "canonical_player_key": hd.get("canonical_player_key"),
                    "selection_score_components": hd.get("selection_score_components"),
                    "history_batting_ema": round(float(getattr(p, "history_batting_ema", 99.0)), 3),
                    "history_delta": round(p.history_delta, 5),
                    "history_notes": p.history_notes[:4],
                    "batting_order_source": hd.get("batting_order_source", ""),
                    "batting_order_final": hd.get("batting_order_final", ""),
                    "history_enrichment_current_squad_only": hd.get(
                        "history_enrichment_current_squad_only"
                    ),
                    "batting_order_strict_xi_scope": hd.get("batting_order_strict_xi_scope"),
                    "used_current_season_history": hd.get("used_current_season_history"),
                    "used_prior_season_fallback": hd.get("used_prior_season_fallback"),
                    "used_venue_history": hd.get("used_venue_history"),
                    "fallback_heuristics_only": hd.get("fallback_heuristics_only"),
                    "xi_selection_tier": hd.get("xi_selection_tier", ""),
                    "xi_used_prior_season_rows": hd.get("xi_used_prior_season_rows"),
                    "recent5_xi_rate": hd.get("recent5_xi_rate"),
                    "venue_xi_rate": hd.get("venue_xi_rate"),
                    "prior_season_xi_rate": hd.get("prior_season_xi_rate"),
                    "h2h_weighted_xi_rate": hd.get("h2h_weighted_xi_rate"),
                    "h2h_venue_xi_rate": hd.get("h2h_venue_xi_rate"),
                    "h2h_fixtures_in_layer": hd.get("h2h_fixtures_in_layer"),
                    "h2h_batting_order_used_opponent_history": hd.get(
                        "h2h_batting_order_used_opponent_history"
                    ),
                    "h2h_phase_blend_applied": hd.get("h2h_phase_blend_applied"),
                    "weights_h2h_xi_venue": hd.get("weights_h2h_xi_venue"),
                    "scoring_breakdown": hd.get("scoring_breakdown"),
                    "probable_first_choice_prior": hd.get("probable_first_choice_prior"),
                    "global_ipl_history_presence": hd.get("global_ipl_history_presence"),
                    "global_selection_frequency": hd.get("global_selection_frequency"),
                    "global_batting_position_pattern": hd.get("global_batting_position_pattern"),
                    "global_role_strength": hd.get("global_role_strength"),
                    "used_global_fallback_prior": hd.get("used_global_fallback_prior"),
                    "first_choice_prior_debug": hd.get("first_choice_prior_debug"),
                    "history_usage_debug": hd.get("history_usage_debug"),
                    "batting_order_diagnostic_source": hd.get("batting_order_diagnostic_source"),
                    "fetched_from_team_slug": hd.get("fetched_from_team_slug"),
                    "in_selected_team_fetched_squad": hd.get("in_selected_team_fetched_squad"),
                    "in_opposite_team_fetched_squad": hd.get("in_opposite_team_fetched_squad"),
                    "wrong_side_squad_assignment": hd.get("wrong_side_squad_assignment"),
                    "stale_cached_entry_detected": hd.get("stale_cached_entry_detected"),
                    "valid_current_squad_new_to_franchise": hd.get("valid_current_squad_new_to_franchise"),
                    "selected_franchise_history_presence": hd.get("selected_franchise_history_presence"),
                    "history_for_other_franchises_presence": hd.get("history_for_other_franchises_presence"),
                    "captain_selected_for_team": hd.get("captain_selected_for_team"),
                    "wicketkeeper_selected_for_team": hd.get("wicketkeeper_selected_for_team"),
                    "captain_boost_applied": hd.get("captain_boost_applied"),
                    "wicketkeeper_boost_applied": hd.get("wicketkeeper_boost_applied"),
                    "selection_reason_summary": hd.get("selection_reason_summary"),
                    "history_source_used": hd.get("history_source_used"),
                    "fallback_used": hd.get("fallback_used"),
                    "derive_player_profile": hd.get("derive_player_profile"),
                    "batting_order_reason_summary": hd.get("batting_order_reason_summary"),
                    "batting_order_signal_source_ranked": hd.get("batting_order_signal_source_ranked"),
                    "xi_selection_frequency": (hd.get("selection_score_components") or {}).get("xi_selection_frequency"),
                    "recent_usage_score": (hd.get("selection_score_components") or {}).get("recent_usage_score"),
                    "role_stability_score": (hd.get("selection_score_components") or {}).get("role_stability_score"),
                    "venue_team_pattern_boost": (hd.get("selection_score_components") or {}).get(
                        "venue_team_pattern_boost"
                    ),
                    "venue_fit_conditions_align": (hd.get("selection_score_components") or {}).get(
                        "venue_fit_conditions_align"
                    ),
                    "h2h_boost_rate": (hd.get("selection_score_components") or {}).get("h2h_boost_rate"),
                    "weather_score_proxy": (hd.get("selection_score_components") or {}).get("weather_score_proxy"),
                    "perspectives": {k: round(v, 4) for k, v in p.perspectives.items()},
                    "selection_model_debug": hd.get("selection_model_debug"),
                    "scenario_xi_branch_used": scenario_branch,
                    "scenario_selection_score": sx_br.get("scenario_selection_score"),
                    "scenario_adjustment_total": sx_br.get("scenario_adjustment_total"),
                    "scenario_adjustment_breakdown": sx_br.get("scenario_adjustment_breakdown"),
                    "selected_for_batting_first_reason": sx_br.get(
                        "selected_for_batting_first_reason"
                    ),
                    "selected_for_bowling_first_reason": sx_br.get(
                        "selected_for_bowling_first_reason"
                    ),
                    "scenario_xi_if_team_bats_first": (hd.get("scenario_xi") or {}).get(
                        "if_team_bats_first"
                    ),
                    "scenario_xi_if_team_bowls_first": (hd.get("scenario_xi") or {}).get(
                        "if_team_bowls_first"
                    ),
                    "scenario_rank_used": hd.get("scenario_rank_used"),
                    "selection_changed_due_to_scenario": hd.get(
                        "selection_changed_due_to_scenario"
                    ),
                    "xi_scenario_branch_for_xi_build": hd.get("xi_scenario_branch_for_xi_build"),
                }
            )
        rows.sort(key=lambda r: r["bat_order"] or 99)
        return rows

    xi_scenario_alternates: Optional[dict[str, Any]] = None
    if a_bats_first is None and xi_a_if_bats_first is not None:
        w_alt: list[str] = []

        def _alt_bo(xi: list[SquadPlayer], tnm: str) -> list[str]:
            if len(xi) != 11:
                return [p.name for p in xi]
            return build_batting_order(
                xi, conditions, team_name=tnm, venue_keys=vkeys, out_warnings=w_alt
            )

        xi_scenario_alternates = {
            "team_a": {
                "xi_if_team_bats_first": xi_payload(
                    xi_a_if_bats_first,
                    _alt_bo(xi_a_if_bats_first, team_a_name),
                    "if_team_bats_first",
                ),
                "xi_if_team_bowls_first": xi_payload(
                    xi_a_if_bowls_first,
                    _alt_bo(xi_a_if_bowls_first, team_a_name),
                    "if_team_bowls_first",
                ),
            },
            "team_b": {
                "xi_if_team_bats_first": xi_payload(
                    xi_b_if_bats_first,
                    _alt_bo(xi_b_if_bats_first, team_b_name),
                    "if_team_bats_first",
                ),
                "xi_if_team_bowls_first": xi_payload(
                    xi_b_if_bowls_first,
                    _alt_bo(xi_b_if_bowls_first, team_b_name),
                    "if_team_bowls_first",
                ),
            },
        }

    os_squad_a = sum(1 for p in sa if p.is_overseas)
    os_squad_b = sum(1 for p in sb if p.is_overseas)
    os_xi_a = sum(1 for p in xi_a if p.is_overseas)
    os_xi_b = sum(1 for p in xi_b if p.is_overseas)

    omitted_a = _omitted_xi_report(
        scored_a, xi_a, scenario_branch=br_a, xi_baseline=xi_a_base
    )
    omitted_b = _omitted_xi_report(
        scored_b, xi_b, scenario_branch=br_b, xi_baseline=xi_b_base
    )
    prediction_layer_debug: dict[str, Any] = {
        "team_a": {
            "history_usage_per_player": _squad_post_linkage_history_rows(scored_a),
            "history_linkage_squad_rollup": _history_linkage_squad_rollup(scored_a),
            "scoring_breakdown_per_player": _squad_scoring_breakdown_rows(scored_a),
            "omitted_from_playing_xi": omitted_a,
            "xi_batting_order_diagnostics": _xi_batting_order_diagnostics_rows(xi_a, order_a),
            "impact_sub_ranking": impact_dbg_all_a,
            "impact_sub_ranking_top5": impact_dbg_a,
        },
        "team_b": {
            "history_usage_per_player": _squad_post_linkage_history_rows(scored_b),
            "history_linkage_squad_rollup": _history_linkage_squad_rollup(scored_b),
            "scoring_breakdown_per_player": _squad_scoring_breakdown_rows(scored_b),
            "omitted_from_playing_xi": omitted_b,
            "xi_batting_order_diagnostics": _xi_batting_order_diagnostics_rows(xi_b, order_b),
            "impact_sub_ranking": impact_dbg_all_b,
            "impact_sub_ranking_top5": impact_dbg_b,
        },
    }

    if not getattr(config, "PREDICTION_FULL_DEBUG_PAYLOAD", False):
        prediction_layer_debug = _slim_prediction_layer_debug(prediction_layer_debug)

    _t_asm = time.perf_counter()
    _timing_out = bool(
        getattr(config, "PREDICTION_TIMING_LOG", False) or audit_profile.audit_enabled()
    )

    _out = {
        "conditions": conditions,
        "weather": weather,
        "squad_debug": {
            "team_a_canonical_franchise": canon_a,
            "team_b_canonical_franchise": canon_b,
            "team_a_role_bucket_counts": role_dist_a,
            "team_b_role_bucket_counts": role_dist_b,
            "structured_squad_team_a": _structured_squad_rows(sa),
            "structured_squad_team_b": _structured_squad_rows(sb),
        },
        "history_sync_debug": history_sync_debug,
        "prediction_layer_debug": prediction_layer_debug,
        "prediction_timing_ms": prediction_timing_ms if _timing_out else None,
        "xi_validation": {
            "team_a_in_squad": sub_ok_a,
            "team_b_in_squad": sub_ok_b,
            "team_a_xi_names": [p.name for p in xi_a],
            "team_b_xi_names": [p.name for p in xi_b],
            "constraints_team_a_ok": v_ok_a and len(xi_a) == 11,
            "constraints_team_b_ok": v_ok_b and len(xi_b) == 11,
            "strict_xi_subset_of_fetched_squad": bool(sub_ok_a and sub_ok_b),
            "history_enrichment_limited_to_fetched_squad_only": True,
            "batting_order_names_subset_of_selected_xi": True,
            "strict_validation_warnings": strict_validation_warnings,
        },
        "selection_debug": {
            "team_a": {"xi_validation": _xi_role_validation_counts(xi_a)},
            "team_b": {"xi_validation": _xi_role_validation_counts(xi_b)},
        },
        "toss_scenario": {
            "key": toss_scenario_key,
            "team_a_bats_first": a_bats_first,
            "chase_context_team_a": chase_ctx_a,
            "chase_context_team_b": chase_ctx_b,
            "team_a_xi_scenario_branch_used": br_a,
            "team_b_xi_scenario_branch_used": br_b,
        },
        "xi_scenario_alternates": xi_scenario_alternates,
        "overseas_counts": {
            "squad_team_a": os_squad_a,
            "squad_team_b": os_squad_b,
            "xi_team_a": os_xi_a,
            "xi_team_b": os_xi_b,
        },
        "batting_order_summary": {
            "team_a": {
                **_batting_order_sources_summary(xi_a),
                "lineup_summary": _batting_order_team_lineup_summary(xi_a),
            },
            "team_b": {
                **_batting_order_sources_summary(xi_b),
                "lineup_summary": _batting_order_team_lineup_summary(xi_b),
            },
        },
        "chase_defend_bias": eng_d.get("chase_defend_context") or {},
        "team_a": {
            "name": team_a_name,
            "xi": xi_payload(xi_a, order_a, br_a),
            "batting_order": order_a,
            "impact_subs": _impact_subs_payload(subs_a, impact_dbg_a),
            "squad_size": len(sa),
            "squad_overseas": os_squad_a,
            "xi_overseas": os_xi_a,
            "valid_xi": v_ok_a and len(xi_a) == 11,
            "xi_constraint_errors": v_err_a if not v_ok_a else [],
        },
        "team_b": {
            "name": team_b_name,
            "xi": xi_payload(xi_b, order_b, br_b),
            "batting_order": order_b,
            "impact_subs": _impact_subs_payload(subs_b, impact_dbg_b),
            "squad_size": len(sb),
            "squad_overseas": os_squad_b,
            "xi_overseas": os_xi_b,
            "valid_xi": v_ok_b and len(xi_b) == 11,
            "xi_constraint_errors": v_err_b if not v_ok_b else [],
        },
        "toss_effects": toss,
        "win_probability": win_marginal,
        "win_probability_engine": eng_d,
        "win_probability_logistic": {
            "marginal": win_marginal_logistic,
            "if_team_a_bats_first": win_a_bf_logistic,
            "if_team_a_bowls_first": win_a_ch_logistic,
        },
        "win_if_team_a_bats_first": win_a_bf,
        "win_if_team_a_bowls_first": win_a_ch,
        "strength": {"team_a": str_a, "team_b": str_b},
        "prediction_confidence": confidence,
        "learning_context": {
            "stored_matches": hctx.db_match_count,
            "venue_keys_tried": vkeys,
            "derive_team_selection_pattern_keys_tried": pattern_vkeys,
            "fixture_night": is_night,
            "toss_scenario_used": toss_scenario_key,
            "history_weights_note": (
                "History bump = weighted blend of XI frequency, slot, bowling load, "
                "venue–team picks, overseas mix, day/night, dew (see config.py)."
            ),
            "stage3_note": (
                "Selection_score blends franchise SQLite history, Stage-2 player_profiles, "
                "and venue team_selection_patterns (when keys match derive)."
            ),
        },
    }
    _ap_phase("prediction_result_dict_assembly_ms", _t_asm)
    if getattr(config, "PREDICTION_TIMING_LOG", False):
        prediction_timing_ms["run_prediction_total_ms"] = round(
            (time.perf_counter() - _run_t0) * 1000.0, 2
        )
        _perf_logger.info("run_prediction timing_ms=%s", prediction_timing_ms)
    if _audit_par:
        _audit_par.close_success(_out)
    return _out
