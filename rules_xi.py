from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import config
from player_role_classifier import (
    classify_player,
    pace_only_override_from_conditions,
    role_counts,
)
from ipl_squad import (
    BATTER,
    WK_BATTER,
)
from rules_spec import CANONICAL_RULE_SPEC


@dataclass(frozen=True)
class RuleViolation:
    code: str
    message: str
    actual: Any = None
    expected: Any = None


@dataclass(frozen=True)
class RuleWarning:
    code: str
    message: str
    actual: Any = None
    expected: Any = None


@dataclass(frozen=True)
class XIValidationResult:
    hard_ok: bool
    violations: list[RuleViolation]
    warnings: list[RuleWarning]
    summary: dict[str, Any]
    pace_only_override_used: bool


def assign_designated_keeper_name(xi: list[Any]) -> Optional[str]:
    """
    Deterministic designated-keeper assignment for a frozen XI.

    Note: this function does not mutate player objects; it only returns a name.
    """
    wk = [p for p in xi if classify_player(p).is_designated_keeper_candidate]
    if not wk:
        return None

    def _wk_priority(p: Any) -> tuple[float, float, float, str]:
        hd = getattr(p, "history_debug", None) or {}
        sm = hd.get("selection_model_debug") if isinstance(hd.get("selection_model_debug"), dict) else {}
        bb = sm.get("base_score_breakdown") if isinstance(sm.get("base_score_breakdown"), dict) else {}
        continuity = float(bb.get("last_match_continuity_score") or 0.0)
        role_stability = float(bb.get("stable_role_identity_score") or 0.0)
        experience = float(bb.get("ipl_history_and_role_score") or 0.0)
        pm = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
        meta_conf = float(pm.get("confidence") or 0.0)
        return (continuity, role_stability, experience + 0.2 * meta_conf, str(getattr(p, "name", "") or ""))

    chosen = max(wk, key=_wk_priority)
    return str(getattr(chosen, "name", "") or "") or None


def validate_xi(
    xi: list[Any],
    *,
    conditions: Optional[dict[str, Any]] = None,
    squad: Optional[list[Any]] = None,
) -> XIValidationResult:
    spec = (CANONICAL_RULE_SPEC.get("xi") or {}).get("hard_constraints") or {}
    semi = (CANONICAL_RULE_SPEC.get("xi") or {}).get("semi_hard_constraints") or {}
    violations: list[RuleViolation] = []
    warnings: list[RuleWarning] = []

    pool = squad if squad is not None else xi
    pool_counts = role_counts(list(pool))
    pool_overseas = sum(1 for p in pool if bool(getattr(p, "is_overseas", False)))
    counts = role_counts(xi)

    xi_size = int(spec.get("xi_size") or 11)
    if len(xi) != xi_size:
        violations.append(RuleViolation("xi_size", f"XI size {len(xi)} != {xi_size}", actual=len(xi), expected=xi_size))

    overseas = sum(1 for p in xi if bool(getattr(p, "is_overseas", False)))
    os_spec = spec.get("overseas") or {}
    os_min_spec = int(os_spec.get("min") or int(getattr(config, "MIN_OVERSEAS_IN_XI", 3)))
    os_max_spec = int(os_spec.get("max") or int(getattr(config, "MAX_OVERSEAS", 4)))
    os_min = min(os_min_spec, pool_overseas, os_max_spec)
    os_max = min(os_max_spec, pool_overseas) if pool_overseas else os_max_spec
    if pool_overseas < os_min_spec:
        warnings.append(
            RuleWarning(
                "conflict_squad_overseas_insufficient",
                f"Conflict: squad has {pool_overseas} overseas player(s); cannot enforce overseas_min={os_min_spec}.",
                actual=pool_overseas,
                expected=os_min_spec,
            )
        )
    if overseas < os_min:
        violations.append(RuleViolation("overseas_min", f"Overseas {overseas} < {os_min}", actual=overseas, expected=os_min))
    if overseas > os_max:
        violations.append(RuleViolation("overseas_max", f"Overseas {overseas} > {os_max}", actual=overseas, expected=os_max))

    designated_keeper = assign_designated_keeper_name(xi)
    if bool(spec.get("designated_keeper_required", True)) and not designated_keeper:
        violations.append(RuleViolation("designated_keeper", "No designated keeper", actual=None, expected=True))

    if counts["wk_role_players"] > 2:
        violations.append(RuleViolation("wk_max", f"Max 2 wicketkeepers allowed, found {counts['wk_role_players']}", actual=counts["wk_role_players"], expected=2))

    n_top = sum(1 for p in xi if classify_player(p).is_top_order_batter)
    pool_top = sum(1 for p in pool if classify_player(p).is_top_order_batter)
    # Target top-order players: min 4, or fewer if squad doesn't have enough.
    target_top = min(4, pool_top)
    if n_top < target_top:
        violations.append(RuleViolation("top_order_min", f"Top-order players {n_top} < {target_top}", actual=n_top, expected=target_top))

    n_t1 = sum(1 for p in xi if str(getattr(p, 'history_debug', {}).get('marquee_tier') or "").lower() == 'tier_1')
    pool_t1 = sum(1 for p in pool if str(getattr(p, 'history_debug', {}).get('marquee_tier') or "").lower() == 'tier_1')
    target_t1 = min(3, pool_t1)
    if n_t1 < target_t1:
        violations.append(RuleViolation("tier1_min", f"Tier 1 players {n_t1} < {target_t1}", actual=n_t1, expected=target_t1))

    bowl_min_spec = int(spec.get("bowling_options_min") or int(getattr(config, "MIN_BOWLING_OPTIONS", 5)))
    bowl_min = min(bowl_min_spec, int(pool_counts.get("bowling_options") or 0))
    if int(pool_counts.get("bowling_options") or 0) < bowl_min_spec:
        warnings.append(
            RuleWarning(
                "conflict_squad_bowling_options_insufficient",
                f"Conflict: squad has {int(pool_counts.get('bowling_options') or 0)} bowling option(s); cannot enforce bowling_options_min={bowl_min_spec}.",
                actual=int(pool_counts.get("bowling_options") or 0),
                expected=bowl_min_spec,
            )
        )
    if counts["bowling_options"] < bowl_min:
        violations.append(
            RuleViolation(
                "bowling_options_min",
                f"Bowling options {counts['bowling_options']} < {bowl_min}",
                actual=counts["bowling_options"],
                expected=bowl_min,
            )
        )
    pacers_min_spec = int(spec.get("pacers_min") or int(getattr(config, "MIN_PACE_OPTIONS_IN_XI", 3)))
    pacers_min = min(pacers_min_spec, int(pool_counts.get("pacers") or 0))
    if int(pool_counts.get("pacers") or 0) < pacers_min_spec:
        warnings.append(
            RuleWarning(
                "conflict_squad_pacers_insufficient",
                f"Conflict: squad has {int(pool_counts.get('pacers') or 0)} pacer(s); cannot enforce pacers_min={pacers_min_spec}.",
                actual=int(pool_counts.get("pacers") or 0),
                expected=pacers_min_spec,
            )
        )
    if counts["pacers"] < pacers_min:
        violations.append(
            RuleViolation(
                "pacers_min",
                f"Pacers {counts['pacers']} < {pacers_min}",
                actual=counts["pacers"],
                expected=pacers_min,
            )
        )

    pace_only = pace_only_override_from_conditions(conditions)
    spinners_min_spec = int(spec.get("spinners_min") or int(getattr(config, "MIN_SPINNER_OPTIONS_IN_XI", 1)))
    spinners_min = min(spinners_min_spec, int(pool_counts.get("spinners") or 0))
    if int(pool_counts.get("spinners") or 0) < spinners_min_spec and not pace_only:
        warnings.append(
            RuleWarning(
                "conflict_squad_spinners_insufficient",
                f"Conflict: squad has {int(pool_counts.get('spinners') or 0)} spinner(s); cannot enforce spinners_min={spinners_min_spec}.",
                actual=int(pool_counts.get("spinners") or 0),
                expected=spinners_min_spec,
            )
        )
    if counts["spinners"] < spinners_min and not pace_only:
        violations.append(
            RuleViolation(
                "spinners_min",
                f"Spinners {counts['spinners']} < {spinners_min}",
                actual=counts["spinners"],
                expected=spinners_min,
            )
        )

    # 5. TEAM STRUCTURE CONSTRAINT (LIGHT CONSTRAINT)
    bat_min_spec = 5
    n_proper_batters = sum(1 for p in xi if getattr(p, 'role_bucket', '') in (BATTER, WK_BATTER))
    if n_proper_batters < bat_min_spec:
        warnings.append(
            RuleWarning("batters_min", f"Proper batters {n_proper_batters} < {bat_min_spec}", actual=n_proper_batters, expected=bat_min_spec)
        )

    n_t12 = sum(1 for p in xi if str(getattr(p, 'history_debug', {}).get('marquee_tier') or "").lower() in ('tier_1', 'tier_2'))
    pool_t12 = sum(1 for p in pool if str(getattr(p, 'history_debug', {}).get('marquee_tier') or "").lower() in ('tier_1', 'tier_2'))
    if n_t12 < min(5, pool_t12):
        warnings.append(
            RuleWarning("tier12_pref", f"Prefer at least 5 T1+T2 players, found {n_t12}")
        )

    wk_cap = int(semi.get("wk_role_players_max") or 2)
    if counts["wk_role_players"] > wk_cap:
        non_marquee_extra = []
        if bool(semi.get("wk_role_players_allow_marquee_override", True)):
            for p in xi:
                if not classify_player(p).is_wk_role_player:
                    continue
                hd = getattr(p, "history_debug", None) or {}
                tier = str((hd.get("marquee_tier") or "")).strip().lower()
                if tier not in ("tier_1", "tier_2"):
                    non_marquee_extra.append(str(getattr(p, "name", "") or ""))
        if non_marquee_extra:
            warnings.append(
                RuleWarning(
                    "wk_role_players_cap",
                    f"Semi: wicketkeeper role players {counts['wk_role_players']} > {wk_cap}",
                    actual=counts["wk_role_players"],
                    expected=wk_cap,
                )
            )

    ar_cap = int(semi.get("structural_all_rounders_max") or 3)
    if counts["structural_all_rounders"] > ar_cap:
        warnings.append(
            RuleWarning(
                "structural_all_rounders_cap",
                f"Semi: structural all-rounders {counts['structural_all_rounders']} > {ar_cap}",
                actual=counts["structural_all_rounders"],
                expected=ar_cap,
            )
        )

    summary = {
        "xi_size": len(xi),
        "overseas": overseas,
        "designated_keeper_name": designated_keeper or "",
        "bowling_options": counts["bowling_options"],
        "pacers": counts["pacers"],
        "spinners": counts["spinners"],
        "wk_role_players": counts["wk_role_players"],
        "structural_all_rounders": counts["structural_all_rounders"],
        "hard_ok": len(violations) == 0 and len(xi) == xi_size,
        "semi_warnings": [w.code for w in warnings],
    }

    return XIValidationResult(
        hard_ok=(len(violations) == 0),
        violations=violations,
        warnings=warnings,
        summary=summary,
        pace_only_override_used=bool(pace_only),
    )
