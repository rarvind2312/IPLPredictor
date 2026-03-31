"""
Explicit join between the **current squad** and SQLite history rows using a safe alias resolver.

Maps squad display names to stored ``player_key`` via ``player_alias_resolve``. **Ambiguous** names
never pick a random SQLite row. Squad display strings are never replaced.

Used by ``history_xi.attach_primary_history_to_squad``, pipeline audit Stage F, ``history_sync``, and
validation warnings.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import db
import history_key_collision
import ipl_teams
import learner
import player_alias_resolve


def link_current_squad_to_history(
    current_squad: list[Any],
    team_name: str,
    *,
    opponent_canonical_label: Optional[str] = None,
) -> dict[str, Any]:
    """
    For each current-squad player, resolve a SQLite ``player_key`` (when unambiguous), then count
    history rows for that key under ``canonical_team_key``.
    """
    canon_team = ipl_teams.franchise_label_for_storage(team_name) or (team_name or "").strip()
    canonical_team_key = ipl_teams.canonical_team_key_for_franchise(canon_team)

    snap = db.franchise_history_snapshot(canon_team)
    distinct_m = int(snap.get("distinct_match_count") or 0)
    xi_rows = int(snap.get("xi_row_count") or 0)

    h2h_mids: list[int] = []
    if opponent_canonical_label:
        opp = (
            ipl_teams.franchise_label_for_storage(opponent_canonical_label)
            or (opponent_canonical_label or "").strip()
        )
        if opp and canon_team:
            h2h_mids = [
                int(x["match_id"])
                for x in db.h2h_fixtures_between_franchises(canon_team, opp, limit=120)
            ]

    franchise_keys = db.franchise_distinct_history_player_keys(canonical_team_key)
    fk_count = len(franchise_keys)
    global_keys = db.global_distinct_history_player_keys()

    eff_keys: list[Optional[str]] = []
    resolutions: list[player_alias_resolve.PlayerHistoryResolution] = []
    global_resolutions: list[Optional[player_alias_resolve.PlayerHistoryResolution]] = []
    for p in current_squad:
        name = str(getattr(p, "name", "") or "").strip()
        pk0 = learner.normalize_player_key(name)
        if not pk0:
            continue
        res = player_alias_resolve.resolve_player_to_history_key(name, franchise_keys)
        resolutions.append(res)
        eff_keys.append(player_alias_resolve.history_lookup_key_from_resolution(res))
        gres: Optional[player_alias_resolve.PlayerHistoryResolution] = None
        if res.resolution_type == "no_match":
            gres = player_alias_resolve.resolve_player_to_history_key(name, global_keys)
        global_resolutions.append(gres)

    global_lookup_keys = [
        player_alias_resolve.history_lookup_key_from_resolution(g)
        for g in global_resolutions
        if g is not None
    ]
    unique_global_gks = list(dict.fromkeys([k for k in global_lookup_keys if k]))
    global_tmx_stats = (
        db.batch_global_team_match_xi_stats(unique_global_gks) if unique_global_gks else {}
    )

    unique_hist = sorted({k for k in eff_keys if k})
    all_keys_global_depth = sorted(set(unique_hist) | set(unique_global_gks))
    global_depth_stats = (
        db.batch_global_team_match_xi_stats(all_keys_global_depth) if all_keys_global_depth else {}
    )
    pms = db.batch_player_match_stats_counts(unique_hist, canonical_team_key)
    pbp = db.batch_player_batting_positions_counts(unique_hist, canonical_team_key)
    tmx = db.batch_team_match_xi_counts(unique_hist, canonical_team_key)
    latest = db.batch_team_match_xi_latest_dates(unique_hist, canonical_team_key)
    h2h_c = (
        db.batch_team_match_xi_h2h_counts(unique_hist, canonical_team_key, h2h_mids)
        if h2h_mids
        else {}
    )

    per_player: list[dict[str, Any]] = []

    res_idx = 0
    for p in current_squad:
        name = str(getattr(p, "name", "") or "").strip()
        pk = learner.normalize_player_key(name)
        if not pk:
            continue
        res = resolutions[res_idx]
        ek = eff_keys[res_idx]
        gres = global_resolutions[res_idx]
        res_idx += 1
        role_bucket = str(getattr(p, "role_bucket", "") or "").strip()

        grk = player_alias_resolve.history_lookup_key_from_resolution(gres) if gres else None
        g_dist_raw = int((global_tmx_stats.get(grk) or {}).get("distinct_matches") or 0) if grk else 0
        gdist_ek = int((global_depth_stats.get(ek) or {}).get("distinct_matches") or 0) if ek else 0
        gdist_grk = int((global_depth_stats.get(grk) or {}).get("distinct_matches") or 0) if grk else 0

        n_tmx_raw = int(tmx.get(ek, 0)) if ek else 0
        n_pms_raw = int(pms.get(ek, 0)) if ek else 0
        n_pbp_raw = int(pbp.get(ek, 0)) if ek else 0
        total_fr_raw = n_tmx_raw + n_pms_raw + n_pbp_raw

        debut = player_alias_resolve.apply_debutant_alias_suppression(
            franchise_res=res,
            global_res=gres,
            history_lookup_key=ek,
            global_resolved_key=grk,
            franchise_history_row_count=total_fr_raw,
            global_distinct_for_franchise_key=gdist_ek,
            global_distinct_for_global_key=gdist_grk,
        )
        eff_ek = debut.effective_history_lookup_key
        eff_grk = debut.effective_global_resolved_key
        disp_res = debut.franchise_resolution_effective
        disp_gres = debut.global_resolution_effective
        used_global_resolved_key_for_prior = bool(eff_grk and not eff_ek)

        n_tmx = int(tmx.get(eff_ek, 0)) if eff_ek else 0
        n_pms = int(pms.get(eff_ek, 0)) if eff_ek else 0
        n_pbp = int(pbp.get(eff_ek, 0)) if eff_ek else 0
        lv = latest.get(eff_ek) if eff_ek else None
        h2h_n = int(h2h_c.get(eff_ek, 0)) if eff_ek else 0

        matched_hist_name = (
            db.sample_stored_player_name_for_key(canonical_team_key, eff_ek) if eff_ek else None
        )
        hist_status = player_alias_resolve.history_status_from_resolution(disp_res)
        total_hist_rows = n_tmx + n_pms + n_pbp
        franchise_rolled = player_alias_resolve.rolled_up_history_interpretation(
            disp_res,
            distinct_franchise_matches=distinct_m,
            franchise_xi_row_count=xi_rows,
            franchise_key_count=fk_count,
            role_bucket=role_bucket or None,
            usable_history_rows=total_hist_rows,
        )
        g_dist_eff = (
            int((global_depth_stats.get(eff_grk) or {}).get("distinct_matches") or 0) if eff_grk else 0
        )
        rolled = player_alias_resolve.rolled_up_with_global_alias_fallback(
            disp_res,
            franchise_rolled,
            disp_gres if disp_res.resolution_type == "no_match" else None,
            g_dist_eff,
        )
        if debut.suppression_applied:
            rolled = "likely_new_or_sparse"
        global_alias_resolution_type = (
            (disp_gres.resolution_type if disp_gres else None)
            if disp_res.resolution_type == "no_match"
            else None
        )
        global_alias_confidence = (
            round(float(disp_gres.confidence), 4)
            if disp_gres is not None and disp_res.resolution_type == "no_match"
            else None
        )
        global_alias_layer_used = (
            f"global_pass:{disp_gres.resolution_layer_used}"
            if disp_gres is not None and disp_res.resolution_type == "no_match"
            else None
        )

        layer_debug = {
            "resolution_layer_used": disp_res.resolution_layer_used,
            "surname_bucket_size": disp_res.surname_bucket_size,
            "surname_candidates_checked": list(disp_res.surname_candidates_checked),
            "relaxed_unique_surname_rule_applied": disp_res.layer_d_branch
            == "relaxed_unique_surname",
            "layer_d_branch": disp_res.layer_d_branch or None,
            "layer_d_reason": disp_res.layer_d_reason or None,
        }

        per_player.append(
            {
                "squad_display_name": name,
                "player_name": name,
                "role_bucket": role_bucket,
                "normalized_full_name_key": pk,
                "canonical_player_key": pk,
                "resolution_type": disp_res.resolution_type,
                "resolved_history_key": disp_res.resolved_history_key,
                "history_lookup_key": eff_ek,
                "surname_bucket_size": disp_res.surname_bucket_size,
                "history_status": hist_status,
                "rolled_up_interpretation": rolled,
                "alias_confidence": round(float(disp_res.confidence), 4),
                "confidence": round(float(disp_res.confidence), 4),
                "matched_history_player_name": matched_hist_name,
                "ambiguous_candidates": list(disp_res.ambiguous_candidates),
                "layer_b_variant_hits": list(disp_res.layer_b_variant_hits),
                "resolution_layer_debug": layer_debug,
                "team_name": canon_team,
                "canonical_team_key": canonical_team_key,
                "team_match_xi_rows": n_tmx,
                "player_match_stats_rows": n_pms,
                "player_batting_positions_rows": n_pbp,
                "latest_match_date": lv,
                "h2h_rows_vs_opponent": h2h_n,
                "global_resolved_history_key": eff_grk,
                "global_alias_resolution_type": global_alias_resolution_type,
                "global_alias_confidence": global_alias_confidence,
                "global_alias_layer_used": global_alias_layer_used,
                "used_global_resolved_key_for_prior": used_global_resolved_key_for_prior,
                "likely_first_ipl_player": debut.likely_first_ipl_player,
                "debutant_alias_suppression_applied": debut.suppression_applied,
                "debutant_alias_rejection_reason": debut.debutant_alias_rejection_reason or None,
                "pre_suppression_history_lookup_key": ek,
                "pre_suppression_global_resolved_key": grk,
                "global_distinct_matches_raw_for_global_key": g_dist_raw,
                "collided_history_key": None,
                "collision_group_members": [],
                "collision_resolution_outcome": "no_collision",
                "collision_winner_player_name": None,
                "collision_winner_resolution_type": None,
                "collision_winner_confidence": None,
            }
        )

    collision_stats = history_key_collision.apply_intrasquad_effective_key_collisions(per_player)

    usable = 0
    no_tmx_no_pms = 0
    ambiguous_n = 0
    no_match_n = 0
    collision_alias_n = 0
    for r in per_player:
        n_tmx_r = int(r.get("team_match_xi_rows") or 0)
        n_pms_r = int(r.get("player_match_stats_rows") or 0)
        n_pbp_r = int(r.get("player_batting_positions_rows") or 0)
        if n_tmx_r > 0 or n_pms_r > 0 or n_pbp_r > 0:
            usable += 1
        if n_tmx_r == 0 and n_pms_r == 0:
            no_tmx_no_pms += 1
        rt = str(r.get("resolution_type") or "")
        if rt == "ambiguous_alias":
            ambiguous_n += 1
        elif rt == "ambiguous_alias_collision":
            collision_alias_n += 1
            ambiguous_n += 1
        elif rt == "no_match":
            no_match_n += 1

    for r in per_player:
        ld = r.get("resolution_layer_debug") if isinstance(r.get("resolution_layer_debug"), dict) else {}
        amb_cands = r.get("ambiguous_candidates")
        if not isinstance(amb_cands, (list, tuple)):
            amb_cands = tuple(amb_cands or ())
        else:
            amb_cands = tuple(amb_cands)
        lb = r.get("layer_b_variant_hits")
        if not isinstance(lb, (list, tuple)):
            lb = tuple(lb or ())
        else:
            lb = tuple(lb)
        scc = ld.get("surname_candidates_checked")
        if isinstance(scc, list):
            scc_t = tuple(scc)
        else:
            scc_t = tuple(scc or ())
        res_obj = player_alias_resolve.PlayerHistoryResolution(
            squad_full_name=str(r.get("player_name") or ""),
            normalized_full_name_key=str(r.get("canonical_player_key") or ""),
            resolved_history_key=r.get("resolved_history_key"),
            resolution_type=str(r.get("resolution_type") or "no_match"),
            confidence=float(r.get("alias_confidence") or 0.0),
            ambiguous_candidates=amb_cands,
            layer_b_variant_hits=lb,
            resolution_layer_used=str(ld.get("resolution_layer_used") or "unresolved"),
            surname_candidates_checked=scc_t,
            layer_d_reason=str(ld.get("layer_d_reason") or ""),
            layer_d_branch=str(ld.get("layer_d_branch") or ""),
            surname_bucket_size=int(r.get("surname_bucket_size") or 0),
        )
        amb_json = player_alias_resolve.ambiguous_candidates_json(res_obj)
        db.upsert_player_alias_resolution(
            canonical_team_key,
            str(r.get("player_name") or ""),
            str(r.get("canonical_player_key") or ""),
            r.get("resolved_history_key"),
            str(r.get("resolution_type") or "no_match"),
            float(r.get("alias_confidence") or 0.0),
            amb_json,
        )

    stage_f_team_health, stage_f_health_detail = player_alias_resolve.classify_stage_f_team_health(
        per_player_rows=per_player
    )

    n = len(per_player)
    frac_broken = (no_tmx_no_pms / float(n)) if n else 0.0
    pct_usable = (100.0 * usable / float(n)) if n else 0.0
    summary = {
        "franchise_label": canon_team,
        "canonical_team_key": canonical_team_key,
        "current_squad_players": n,
        "matched_with_usable_history_rows": usable,
        "unmatched_or_zero_history_rows": n - usable,
        "players_zero_tmx_and_zero_pms": no_tmx_no_pms,
        "fraction_zero_tmx_and_zero_pms": round(frac_broken, 4),
        "pct_current_squad_with_usable_history": round(pct_usable, 2),
        "linkage_failed_majority": bool(n > 0 and frac_broken > 0.5),
        "stage_f_team_health": stage_f_team_health,
        "stage_f_team_health_detail": stage_f_health_detail,
        "ambiguous_name_players": ambiguous_n,
        "ambiguous_alias_collision_players": collision_alias_n,
        "no_match_players": no_match_n,
        "franchise_distinct_history_player_keys": fk_count,
        "global_distinct_history_player_keys": len(global_keys),
        "franchise_distinct_matches_snapshot": distinct_m,
        "franchise_xi_row_count_snapshot": xi_rows,
        **collision_stats,
    }

    return {"per_player": per_player, "summary": summary}


def linkage_summary_json_for_debug(summary: dict[str, Any]) -> str:
    """Compact JSON for logs (optional)."""
    return json.dumps(summary, separators=(",", ":"), default=str)


__all__ = ["link_current_squad_to_history", "linkage_summary_json_for_debug"]
