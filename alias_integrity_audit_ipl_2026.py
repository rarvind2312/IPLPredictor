"""
League-wide alias + metadata + candidate-visibility audit for IPL 2026 squads.

This script is **audit-only**: it uses the same runtime modules the Streamlit app uses
(`squad_fetch`, `history_xi`, `predictor._annotate_player_metadata`, `impact_subs_engine`)
to verify that every squad player:
  - is present in squad truth
  - resolves to a history key (or is correctly flagged as new/sparse/ambiguous)
  - receives player_metadata (Cricinfo/curated/DB)
  - remains visible to XI and impact-sub candidate evaluation

Outputs:
  - /tmp/ipl_2026_alias_audit_full.csv
  - /tmp/ipl_2026_alias_audit_problems.csv

Run:
  PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python alias_integrity_audit_ipl_2026.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config
import history_linkage
import history_xi
import impact_subs_engine
import ipl_teams
import learner
import predictor
import squad_fetch
import venues


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_bool(x: Any) -> bool:
    return bool(x) is True


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _load_alias_override_map() -> dict[str, list[str]]:
    raw_path = str(getattr(config, "PLAYER_ALIAS_OVERRIDES_PATH", "") or "").strip()
    if not raw_path:
        return {}
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / raw_path
    if not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in payload.items():
        canon = learner.normalize_player_key(str(k or ""))
        if not canon:
            continue
        aliases: list[str] = []
        if isinstance(v, str):
            v = [v]
        if isinstance(v, (list, tuple)):
            for a in v:
                ak = learner.normalize_player_key(str(a or ""))
                if ak and ak not in aliases and ak != canon:
                    aliases.append(ak)
        if aliases:
            out[canon] = aliases
    return out


def _alias_override_hit(
    normalized_key: str, resolution_layer_used: str, resolved_history_key: Optional[str]
) -> tuple[bool, str]:
    """
    Report whether curated alias overrides were used.

    We intentionally treat `resolution_layer_used == curated_alias_override` as the signal,
    and surface `resolved_history_key` as the "override value" (the concrete history key selected).
    """
    hit = str(resolution_layer_used or "") == "curated_alias_override"
    return hit, (str(resolved_history_key or "") if hit else "")


def _metadata_fields_from_player(p: Any) -> dict[str, Any]:
    hd = getattr(p, "history_debug", None) or {}
    pm = hd.get("player_metadata") if isinstance(hd.get("player_metadata"), dict) else {}
    return {
        "metadata_attached": bool(pm),
        "metadata_source": _as_str(pm.get("source") or ""),
        "metadata_confidence": pm.get("confidence"),
        "batting_hand": _as_str(pm.get("batting_hand") or ""),
        "bowling_style_raw": _as_str(pm.get("bowling_style_raw") or ""),
        "bowling_type_bucket": _as_str(pm.get("bowling_type_bucket") or ""),
        "primary_role": _as_str(pm.get("primary_role") or ""),
        "secondary_role": _as_str(pm.get("secondary_role") or ""),
    }


def _selection_visibility(
    scored: list[Any],
    xi: list[Any],
    impact_dbg_all: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Per-player visibility signals:
      - considered_for_xi: in `scored` (post-squad parse + filtering)
      - considered_for_impact_sub: appears in impact bench debug-all rows (bench evaluation)
    """
    by_name: dict[str, dict[str, Any]] = {}
    scored_names = {getattr(p, "name", "") for p in scored}
    xi_names = {getattr(p, "name", "") for p in xi}
    impact_names = {str(r.get("name") or "") for r in (impact_dbg_all or [])}
    for nm in sorted({*scored_names, *impact_names, *xi_names}):
        if not nm:
            continue
        by_name[nm] = {
            "considered_for_xi": nm in scored_names,
            "in_selected_xi": nm in xi_names,
            "considered_for_impact_sub": nm in impact_names,
            "excluded_before_selection": False,
            "exclusion_reason": "",
        }
    return by_name


def _problem_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    rt = str(row.get("resolution_type") or "")
    linkage = str(row.get("linkage_classification") or "")
    unresolved = rt in ("no_match", "ambiguous_alias", "ambiguous_alias_collision")
    # Treat "no_match" as acceptable for genuine debutants / sparse players; flag only likely alias misses.
    if unresolved and (linkage.startswith("likely_alias_miss") or linkage == "history_key_collision_loser"):
        tags.append("alias_not_resolved")
    if not bool(row.get("metadata_attached")):
        tags.append("metadata_missing")
    # Only flag collisions where this player *lost* the collision and was stripped of its key.
    if str(row.get("collision_resolution_outcome") or "") == "lost_collision":
        tags.append("collision_present")
    if not bool(row.get("considered_for_xi")):
        tags.append("not_visible_to_xi_candidate_pool")
    if not bool(row.get("considered_for_impact_sub")) and not bool(row.get("in_selected_xi")):
        # Non-XI player should usually appear in bench evaluation.
        tags.append("not_visible_to_impact_candidate_pool")
    # Heuristic: obvious alias miss if we have a surname bucket but no resolved key.
    if unresolved and linkage.startswith("likely_alias_miss") and int(row.get("surname_bucket_size") or 0) > 0:
        tags.append("likely_alias_miss_initials_or_spelling")
    return tags


def main() -> int:
    t0 = time.perf_counter()
    out_full = Path("/tmp/ipl_2026_alias_audit_full.csv")
    out_prob = Path("/tmp/ipl_2026_alias_audit_problems.csv")

    venue = venues.resolve_venue("Wankhede Stadium, Mumbai")
    weather_stub = {
        "ok": False,
        "temperature_c": 28.0,
        "relative_humidity_pct": 60.0,
        "precipitation_mm": 0.0,
        "precipitation_probability_pct": 0.0,
        "cloud_cover_pct": 30.0,
        "wind_kmh": 10.0,
    }
    cond = venues.venue_conditions_summary(venue, weather_stub)

    alias_override_map = _load_alias_override_map()
    # Invert for convenience: alias -> canon.
    alias_to_canon: dict[str, str] = {}
    for canon, aliases in alias_override_map.items():
        for a in aliases:
            if a and a not in alias_to_canon:
                alias_to_canon[a] = canon

    learned_map = learner.load_learned_map()

    slugs = list(ipl_teams.TEAM_SLUGS)
    if len(slugs) != 10:
        print(f"warning: expected 10 IPL team slugs, got {len(slugs)}", file=sys.stderr)

    all_rows: list[dict[str, Any]] = []
    squad_load_errors: list[dict[str, Any]] = []

    for i, slug in enumerate(slugs):
        team_name = ipl_teams.label_for_slug(slug)
        canon_team = ipl_teams.franchise_label_for_storage(team_name) or team_name.strip()
        opponent_slug = slugs[(i + 1) % len(slugs)] if slugs else slug
        opponent_name = ipl_teams.label_for_slug(opponent_slug)

        members, err, dbg = squad_fetch.fetch_squad_for_slug(slug)
        if err:
            squad_load_errors.append({"team": team_name, "slug": slug, "error": err})
            continue

        # Task 1 baseline squad truth rows (from official squad page parse).
        base_by_norm: dict[str, dict[str, Any]] = {}
        for m in members:
            nm = str(getattr(m, "name", "") or "").strip()
            pk = learner.normalize_player_key(nm)
            if not nm or not pk:
                continue
            base_by_norm[pk] = {
                "team_name": team_name,
                "team_slug": slug,
                "canonical_team": canon_team,
                "squad_display_name": nm,
                "normalized_player_key": pk,
                "role_bucket": str(getattr(m, "role_bucket", "") or ""),
                "overseas": bool(getattr(m, "overseas", False)),
            }

        # Build SquadPlayer list through the same parsing path the app uses.
        squad_text = squad_fetch.format_squad_text(members)
        parsed = predictor.parse_squad_text(squad_text)
        predictor._annotate_squad_canonical_keys(parsed, canon_team)

        # Alias resolution audit (same resolver + collision logic used by runtime).
        link_pkg = history_linkage.link_current_squad_to_history(
            parsed, team_name, opponent_canonical_label=opponent_name
        )
        link_by_pk = {str(r.get("canonical_player_key") or ""): r for r in (link_pkg.get("per_player") or [])}

        shape_self = predictor._squad_shape(parsed)
        # For scoring we only need opponent shape; use a neutral proxy.
        shape_opp = shape_self

        scored: list[Any] = []
        for p in parsed:
            sp = predictor._score_player(
                p,
                self_shape=shape_self,
                opp_shape=shape_opp,
                conditions=cond,
                learned_map=learned_map,
                franchise_canonical=canon_team,
            )
            predictor._set_player_ipl_flags(sp)
            scored.append(sp)

        vkeys = predictor.venue_lookup_keys(venue)
        chase_ctx = None
        history_xi.attach_primary_history_to_squad(
            scored,
            team_name,
            vkeys,
            shape=shape_self,
            chase_context=chase_ctx,
            opponent_canonical_label=opponent_name,
        )
        history_xi.compute_selection_scores(
            scored,
            conditions=cond,
            venue_key_candidates=predictor._derive_pattern_venue_keys(venue, vkeys),
            fixture_context={"reference_iso_date": datetime.now().date().isoformat()},
        )
        predictor._refine_opener_finisher_from_derive(scored)
        predictor._annotate_player_metadata(scored)
        tk = ipl_teams.canonical_team_key_for_franchise(canon_team)
        predictor._annotate_phase_bowling_signals(scored, tk)
        predictor._annotate_role_bands(scored)

        # XI + impact-sub bench evaluation (for visibility flags only).
        xi = predictor.select_base_playing_xi(scored, conditions=cond)
        picked, top5_dbg, dbg_all = impact_subs_engine.rank_impact_sub_candidates(
            scored,
            xi,
            team_display_name=team_name,
            canonical_team_key=tk,
            venue_key=str(venue.key),
            venue_key_candidates=predictor._derive_pattern_venue_keys(venue, vkeys),
            is_chasing=None,
            conditions=cond,
            team_bats_first=None,
        )
        _vis = _selection_visibility(scored, xi, dbg_all)

        # Alias/linkage rows are stored inside each player's history_debug by attach_primary_history_to_squad.
        # We surface those fields from `history_usage_debug`.
        for p in scored:
            nm = str(getattr(p, "name", "") or "").strip()
            pk = learner.normalize_player_key(nm)
            hd = getattr(p, "history_debug", None) or {}
            link_row = link_by_pk.get(pk, {}) if pk else {}
            resolution_type = _as_str(link_row.get("resolution_type") or "no_match")
            resolved_history_key = link_row.get("resolved_history_key")
            history_lookup_key = link_row.get("history_lookup_key")
            global_resolved_history_key = link_row.get("global_resolved_history_key")
            matched_hist_name = link_row.get("matched_history_player_name")
            rolled = link_row.get("rolled_up_interpretation") or link_row.get("rolled_up_with_global") or ""
            res_layer_dbg = link_row.get("resolution_layer_debug") if isinstance(link_row.get("resolution_layer_debug"), dict) else {}
            layer_used = _as_str(res_layer_dbg.get("resolution_layer_used") or "")
            alias_hit, alias_val = _alias_override_hit(pk, layer_used, history_lookup_key or resolved_history_key)

            meta = _metadata_fields_from_player(p)
            vis = _vis.get(nm, {})

            # Override mapping info: show canonical target if the *normalized key itself* is an alias.
            canon_override_target = alias_to_canon.get(pk) or ""
            row: dict[str, Any] = {
                "audit_run_utc": _now_iso(),
                "team_name": team_name,
                "team_slug": slug,
                "canonical_team": canon_team,
                "squad_display_name": nm,
                "normalized_player_key": pk,
                "role_bucket": str(getattr(p, "role_bucket", "") or ""),
                "overseas": bool(getattr(p, "is_overseas", False)),
                "alias_override_hit": alias_hit,
                "alias_override_value": alias_val,
                "alias_override_canon_target": canon_override_target,
                "resolution_type": resolution_type,
                "alias_resolution_type": resolution_type,
                "resolution_layer_used": layer_used,
                "resolved_history_key": _as_str(resolved_history_key),
                "history_lookup_key": _as_str(history_lookup_key),
                "matched_history_player_name": _as_str(matched_hist_name),
                "global_resolved_history_key": _as_str(global_resolved_history_key),
                "collision": bool(str(link_row.get("collision_resolution_outcome") or "") not in ("", "no_collision")),
                "collision_resolution_outcome": _as_str(link_row.get("collision_resolution_outcome") or ""),
                "unresolved": resolution_type in ("no_match", "ambiguous_alias", "ambiguous_alias_collision"),
                "linkage_classification": _as_str(rolled),
                "surname_bucket_size": int(link_row.get("surname_bucket_size") or 0),
                **meta,
                **vis,
                "bench_rank_top5": bool(any(str(r.get("name") or "") == nm for r in (top5_dbg or []))),
            }
            row["problem_tags"] = "|".join(_problem_tags(row))
            all_rows.append(row)

    # Write full output.
    if all_rows:
        cols = sorted({k for r in all_rows for k in r.keys()})
        with out_full.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in cols})

    # Problems.
    prob_rows = [r for r in all_rows if str(r.get("problem_tags") or "").strip()]
    if prob_rows:
        cols2 = sorted({k for r in prob_rows for k in r.keys()})
        with out_prob.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols2)
            w.writeheader()
            for r in prob_rows:
                w.writerow({k: r.get(k, "") for k in cols2})

    # Summary print.
    print("ipl_2026_alias_audit")
    print(f"- run_utc: {_now_iso()}")
    print(f"- teams_attempted: {len(slugs)}")
    print(f"- squad_load_errors: {len(squad_load_errors)}")
    if squad_load_errors:
        print(f"- squad_load_error_sample: {squad_load_errors[:2]}")
    print(f"- total_players_audited: {len(all_rows)}")
    clean = sum(1 for r in all_rows if not str(r.get("problem_tags") or "").strip())
    print(f"- clean_mapping_rows: {clean}")
    print(f"- problem_rows: {len(prob_rows)}")
    print(f"- full_csv: {out_full}")
    print(f"- problems_csv: {out_prob}")
    print(f"- elapsed_s: {round(time.perf_counter() - t0, 2)}")

    # Explicit checks for requested example players (case-insensitive contains match).
    examples = [
        "vaibhav sooryavanshi",
        "suryakumar yadav",
        "surya kumar yadav",
        "n tilak varma",
        "khaleel ahmed",
        "kk ahmed",
        "rahul chahar",
        "quinton de kock",
        "matthew william short",
        "zak foulkes",
    ]
    lower_index = {str(r.get("normalized_player_key") or "").lower(): r for r in all_rows}
    print("- example_checks:")
    for e in examples:
        ek = learner.normalize_player_key(e)
        hit = lower_index.get(ek)
        if hit:
            print(
                f"  - {e}: team={hit.get('team_name')} resolved={hit.get('resolved_history_key')} "
                f"meta={hit.get('metadata_attached')} xi_visible={hit.get('considered_for_xi')} "
                f"impact_visible={hit.get('considered_for_impact_sub')} tags={hit.get('problem_tags')}"
            )
        else:
            # Try fuzzy contains on display name.
            found = [
                r
                for r in all_rows
                if ek and ek in str(r.get("normalized_player_key") or "").lower()
            ]
            if found:
                r0 = found[0]
                print(
                    f"  - {e}: fuzzy_hit={r0.get('squad_display_name')} team={r0.get('team_name')} "
                    f"tags={r0.get('problem_tags')}"
                )
            else:
                print(f"  - {e}: NOT_FOUND_IN_SQUADS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
