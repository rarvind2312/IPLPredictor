"""
Real-world validation audit for recent IPL 2026 matches.

This intentionally uses the same runtime path as the Streamlit app:
- parsers.router.parse_scorecard (fetch + parse IPLT20 scorecards)
- squad_fetch.fetch_squad_for_slug (fetch + parse current official squads)
- squad_fetch.format_squad_text + predictor.run_prediction (full prediction pipeline)
- venues.resolve_venue + weather.fetch_weather (conditions path)

Output: compact per-match blocks + cross-match summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, time
from typing import Any, Optional

import config
import ipl_teams
import learner
import predictor
import squad_fetch
import venues
import weather
from parsers.router import parse_scorecard
from player_role_classifier import classify_player


DEFAULT_MATCH_URLS: list[str] = [
    "https://www.iplt20.com/match/2026/2417",
    "https://www.iplt20.com/match/2026/2418",
    "https://www.iplt20.com/match/2026/2419",
    "https://www.iplt20.com/match/2026/2420",
    "https://www.iplt20.com/match/2026/2421",
]


def _nk(name: str) -> str:
    return learner.normalize_player_key(name or "")


def _dedupe_preserve(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        k = str(x or "").strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _parse_match_time(meta: dict[str, Any]) -> tuple[datetime, str]:
    """
    IPLT20 parser often yields only a date string.
    We mimic the app default by choosing a stable time slot when no time is provided.
    """
    raw = str(meta.get("date") or "").strip()
    # ipl_parser sometimes yields YYYY-MM-DD, sometimes "2 Apr 2026"
    dt: Optional[datetime] = None
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except Exception:
            continue
    if dt is None:
        # Fall back to "now" (still deterministic enough for this audit run).
        return datetime.now(), "fallback_now"
    # Choose the night slot by default; caller can override if they want 15:30.
    return datetime.combine(dt.date(), time(hour=19, minute=30)), "date_only_default_19_30"


def _extract_actual_xi(payload: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for blk in payload.get("playing_xi") or []:
        if not isinstance(blk, dict):
            continue
        team = str(blk.get("team") or "").strip() or "Team"
        players = [str(x or "").strip() for x in (blk.get("players") or [])]
        players = [p for p in players if p]
        if players:
            out[team] = players[:11]
    return out


def _extract_actual_batting_order(payload: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for inn in payload.get("batting_order") or []:
        if not isinstance(inn, dict):
            continue
        team = str(inn.get("team") or "").strip() or "Team"
        order = [str(x or "").strip() for x in (inn.get("order") or [])]
        order = [o for o in order if o]
        if order:
            out[team] = order
    return out


def _canonical_team_label(team_raw: str) -> str:
    return ipl_teams.franchise_label_for_storage(team_raw)


def _slug_for_team_label(team_label: str) -> str:
    slug = ipl_teams.slug_for_canonical_label(team_label)
    if not slug:
        raise ValueError(f"Unknown IPL team label {team_label!r} (cannot map to squad slug).")
    return slug


def _fetch_current_squad(team_label: str) -> tuple[str, list[squad_fetch.IplSquadMember], dict[str, Any]]:
    slug = _slug_for_team_label(team_label)
    players, err, dbg = squad_fetch.fetch_squad_for_slug(slug)
    if err:
        raise RuntimeError(f"squad_fetch failed for {team_label} (slug={slug}): {err}")
    text = squad_fetch.format_squad_text(players)
    return text, players, {"slug": slug, "debug": asdict(dbg)}


def _venue_from_scorecard(meta: dict[str, Any]) -> tuple[venues.VenueProfile, str]:
    v = str(meta.get("venue") or "").strip()
    if not v:
        return venues.resolve_venue(""), "fallback_generic_unknown"
    return venues.resolve_venue(v), "scorecard_meta"


def _name_match_map(squad_members: list[squad_fetch.IplSquadMember]) -> dict[str, squad_fetch.IplSquadMember]:
    out: dict[str, squad_fetch.IplSquadMember] = {}
    for m in squad_members or []:
        k = _nk(getattr(m, "name", "") or "")
        if k and k not in out:
            out[k] = m
    return out


def _materialize_xi_players(
    xi_names: list[str],
    *,
    squad_members: list[squad_fetch.IplSquadMember],
) -> list[predictor.SquadPlayer]:
    by_key = _name_match_map(squad_members)
    out: list[predictor.SquadPlayer] = []
    for nm in xi_names:
        k = _nk(nm)
        m = by_key.get(k)
        if not m:
            # keep the name but mark as Batter/Indian in absence of squad truth
            sp = predictor._squad_player_from_ipl(nm, predictor.BATTER, overseas=False)
            if sp:
                out.append(sp)
            continue
        sp = predictor._squad_player_from_ipl(
            getattr(m, "name", "") or nm,
            getattr(m, "role_bucket", "") or predictor.BATTER,
            overseas=bool(getattr(m, "overseas", False)),
        )
        if sp:
            out.append(sp)
    predictor._annotate_player_metadata(out)
    return out


def _role_counts_from_players(players: list[predictor.SquadPlayer]) -> dict[str, int]:
    c = Counter()
    designated_keeper = None
    for p in players:
        cp = classify_player(p)
        c["overseas"] += int(bool(getattr(p, "is_overseas", False)))
        c["wk_role_players"] += int(cp.is_wk_role_player)
        c["designated_keeper_candidates"] += int(cp.is_designated_keeper_candidate)
        c["bowling_options"] += int(cp.is_bowling_option)
        c["pacers"] += int(cp.is_pacer)
        c["spinners"] += int(cp.is_spinner)
        c["structural_all_rounders"] += int(cp.is_structural_all_rounder)
        if designated_keeper is None and cp.is_designated_keeper_candidate:
            designated_keeper = p.name
    return {**{k: int(v) for k, v in c.items()}, "designated_keeper_name": designated_keeper or ""}


def _order_overlap(a: list[str], b: list[str], n: int) -> int:
    aa = [_nk(x) for x in (a or [])][:n]
    bb = [_nk(x) for x in (b or [])][:n]
    return sum(1 for i in range(min(len(aa), len(bb))) if aa[i] and aa[i] == bb[i])


def _set_overlap(a: list[str], b: list[str], idxs: list[int]) -> int:
    # idxs are 1-based batting positions
    aa = {_nk(a[i - 1]) for i in idxs if 1 <= i <= len(a)}
    bb = {_nk(b[i - 1]) for i in idxs if 1 <= i <= len(b)}
    aa.discard("")
    bb.discard("")
    return len(aa & bb)


def _normalize_team_key_map(keys: list[str]) -> dict[str, str]:
    """
    Map scorecard team labels to canonical franchise labels used by the app.
    Returns dict raw->canonical.
    """
    out: dict[str, str] = {}
    for t in keys:
        out[t] = _canonical_team_label(t)
    return out


def run_audit(match_urls: list[str]) -> dict[str, Any]:
    # keep payload small but include enough for tuning
    config.PREDICTION_FULL_DEBUG_PAYLOAD = True

    match_urls = _dedupe_preserve(match_urls)
    per_match: list[dict[str, Any]] = []
    per_team_prev_actual: dict[str, list[str]] = {}
    per_team_prev_pred: dict[str, list[str]] = {}
    recurring_gap_tags: Counter[str] = Counter()

    for url in match_urls:
        payload = parse_scorecard(url)
        ing = payload.get("ingestion") or {}
        if not ing.get("fetch_ok"):
            per_match.append(
                {
                    "url": url,
                    "error": ing.get("fetch_error") or "fetch_failed",
                    "ingestion": ing,
                }
            )
            continue

        meta = payload.get("meta") or {}
        teams_raw = list(payload.get("teams") or [])
        teams_raw = [str(t or "").strip() for t in teams_raw if str(t or "").strip()]
        team_map = _normalize_team_key_map(teams_raw)
        teams = [team_map[t] for t in teams_raw if team_map.get(t)]
        if len(teams) < 2:
            per_match.append({"url": url, "error": "team_parse_failed", "meta": meta, "ingestion": ing})
            continue

        team_a, team_b = teams[0], teams[1]

        actual_xi_by_team = _extract_actual_xi(payload)
        actual_bat_by_team = _extract_actual_batting_order(payload)
        # re-key actual blocks to canonical labels where possible
        actual_xi: dict[str, list[str]] = {}
        for t, xi in actual_xi_by_team.items():
            actual_xi[_canonical_team_label(t)] = xi
        actual_bo: dict[str, list[str]] = {}
        for t, bo in actual_bat_by_team.items():
            actual_bo[_canonical_team_label(t)] = bo

        match_time, match_time_source = _parse_match_time(meta)
        venue_profile, venue_source = _venue_from_scorecard(meta)
        wx = weather.fetch_weather(venue_profile, match_time)

        squad_a_text, squad_a_members, squad_a_src = _fetch_current_squad(team_a)
        squad_b_text, squad_b_members, squad_b_src = _fetch_current_squad(team_b)

        result = predictor.run_prediction(
            team_a,
            team_b,
            squad_a_text,
            squad_b_text,
            "",
            venue_profile,
            match_time,
            wx,
            toss_scenario_key="unknown",
        )

        pld = result.get("prediction_layer_debug") or {}
        xi_pred_a = ((result.get("xi_validation") or {}).get("team_a_xi_names")) or []
        xi_pred_b = ((result.get("xi_validation") or {}).get("team_b_xi_names")) or []

        # Predicted order from rule_trace (final_position sorting)
        rt_a = ((pld.get("team_a") or {}).get("rule_trace")) or {}
        rt_b = ((pld.get("team_b") or {}).get("rule_trace")) or {}
        pred_bo_a_rows = rt_a.get("batting_order") or []
        pred_bo_b_rows = rt_b.get("batting_order") or []
        pred_bo_a_rows = [r for r in pred_bo_a_rows if isinstance(r, dict) and r.get("name")]
        pred_bo_b_rows = [r for r in pred_bo_b_rows if isinstance(r, dict) and r.get("name")]
        pred_bo_a_rows.sort(key=lambda r: (r.get("final_position") is None, int(r.get("final_position") or 999)))
        pred_bo_b_rows.sort(key=lambda r: (r.get("final_position") is None, int(r.get("final_position") or 999)))
        pred_bo_a = [str(r.get("name") or "") for r in pred_bo_a_rows]
        pred_bo_b = [str(r.get("name") or "") for r in pred_bo_b_rows]

        act_xi_a = actual_xi.get(team_a) or []
        act_xi_b = actual_xi.get(team_b) or []
        act_bo_a = actual_bo.get(team_a) or []
        act_bo_b = actual_bo.get(team_b) or []

        # pick overlap stats
        pred_a_keys = {_nk(x) for x in xi_pred_a}
        pred_b_keys = {_nk(x) for x in xi_pred_b}
        act_a_keys = {_nk(x) for x in act_xi_a}
        act_b_keys = {_nk(x) for x in act_xi_b}
        pred_a_keys.discard("")
        pred_b_keys.discard("")
        act_a_keys.discard("")
        act_b_keys.discard("")

        def _xi_diff(act_keys: set[str], pred_keys: set[str], act: list[str], pred: list[str]) -> dict[str, Any]:
            correct = len(act_keys & pred_keys)
            missed = [x for x in act if _nk(x) in (act_keys - pred_keys)]
            extra = [x for x in pred if _nk(x) in (pred_keys - act_keys)]
            return {
                "correct_picks": correct,
                "missed_actual": missed,
                "extra_predicted": extra,
            }

        xi_diff_a = _xi_diff(act_a_keys, pred_a_keys, act_xi_a, xi_pred_a)
        xi_diff_b = _xi_diff(act_b_keys, pred_b_keys, act_xi_b, xi_pred_b)

        # team structure counts (computed for actual XI using the same canonical classifier + runtime metadata attachment)
        act_players_a = _materialize_xi_players(act_xi_a, squad_members=squad_a_members)
        act_players_b = _materialize_xi_players(act_xi_b, squad_members=squad_b_members)
        pred_players_a = _materialize_xi_players(xi_pred_a, squad_members=squad_a_members)
        pred_players_b = _materialize_xi_players(xi_pred_b, squad_members=squad_b_members)
        struct_act_a = _role_counts_from_players(act_players_a)
        struct_act_b = _role_counts_from_players(act_players_b)
        struct_pred_a = _role_counts_from_players(pred_players_a)
        struct_pred_b = _role_counts_from_players(pred_players_b)

        # conditions influence
        conditions = result.get("conditions") or {}
        cond_changes_a = (pld.get("team_a") or {}).get("base_to_final_condition_changes") or []
        cond_changes_b = (pld.get("team_b") or {}).get("base_to_final_condition_changes") or []

        def _changes_from_prev(team: str, xi_now: list[str], store: dict[str, list[str]]) -> Optional[int]:
            if not xi_now:
                return None
            prev = store.get(team)
            store[team] = list(xi_now)
            if not prev:
                return None
            prev_k = {_nk(x) for x in prev}
            now_k = {_nk(x) for x in xi_now}
            prev_k.discard("")
            now_k.discard("")
            return len(prev_k ^ now_k) // 2

        real_changes_a = _changes_from_prev(team_a, act_xi_a, per_team_prev_actual)
        real_changes_b = _changes_from_prev(team_b, act_xi_b, per_team_prev_actual)
        pred_changes_a = _changes_from_prev(team_a, xi_pred_a, per_team_prev_pred)
        pred_changes_b = _changes_from_prev(team_b, xi_pred_b, per_team_prev_pred)

        # conditions verdict heuristic: 0–2 changes typical
        def _cond_verdict(n_swaps: int) -> str:
            if n_swaps <= 0:
                return "conditions under-weighted"
            if n_swaps <= 2:
                return "conditions realistic"
            return "conditions over-weighted"

        # tag likely mismatch causes from omission summaries (lightweight)
        def _gap_tags(side_block: dict[str, Any], missed: list[str], extra: list[str]) -> list[str]:
            tags: list[str] = []
            omitted = side_block.get("omitted_from_playing_xi") or []
            reasons_by = {str(r.get("name") or ""): str(r.get("omitted_reason_summary") or r.get("omission_reason") or "") for r in omitted if isinstance(r, dict)}
            for nm in missed[:8]:
                rs = reasons_by.get(nm, "").lower()
                if "overseas" in rs:
                    tags.append("overseas logic issue")
                elif "pacer" in rs or "pace" in rs or "spinner" in rs or "spin" in rs:
                    tags.append("pacer/spinner classification issue")
                elif "conditions" in rs:
                    tags.append("conditions over-weighted")
                elif "repair" in rs:
                    tags.append("repair overreach")
                elif "form" in rs:
                    tags.append("fringe player overvalued")
                elif "continuity" in rs or "last match" in rs:
                    tags.append("continuity too weak")
            for _ in extra[:8]:
                pass
            return _dedupe_preserve(tags)

        tags_a = _gap_tags(pld.get("team_a") or {}, xi_diff_a["missed_actual"], xi_diff_a["extra_predicted"])
        tags_b = _gap_tags(pld.get("team_b") or {}, xi_diff_b["missed_actual"], xi_diff_b["extra_predicted"])
        for t in tags_a + tags_b:
            recurring_gap_tags[t] += 1

        block = {
            "url": url,
            "meta": {
                "venue": meta.get("venue") or "",
                "date": meta.get("date") or "",
                "winner": meta.get("winner") or "",
                "margin": meta.get("margin") or "",
            },
            "runtime_path_confirmation": {
                "scorecard_parser": "parsers.router.parse_scorecard",
                "squad_source": "squad_fetch.fetch_squad_for_slug (iplt20.com/teams/{slug}/squad)",
                "prediction": "predictor.run_prediction",
                "weather": "weather.fetch_weather",
            },
            "squad_sources": {
                team_a: squad_a_src,
                team_b: squad_b_src,
            },
            "conditions_used": {
                "venue_profile": venue_profile.display_name,
                "venue_source": venue_source,
                "match_time": match_time.isoformat(),
                "match_time_source": match_time_source,
                "weather": wx,
                "conditions": conditions,
            },
            "actual_vs_predicted": {
                team_a: {
                    "actual_xi": act_xi_a,
                    "predicted_xi": xi_pred_a,
                    **xi_diff_a,
                    "actual_batting_order": act_bo_a,
                    "predicted_batting_order": pred_bo_a,
                    "batting_order_overlap": {
                        "openers_positional": _order_overlap(act_bo_a, pred_bo_a, 2),
                        "top4_positional": _order_overlap(act_bo_a, pred_bo_a, 4),
                        "lower_8_11_set": _set_overlap(act_bo_a, pred_bo_a, [8, 9, 10, 11]),
                    },
                    "team_structure": {
                        "actual": struct_act_a,
                        "predicted": struct_pred_a,
                    },
                    "conditions_influence": {
                        "real_changes_from_prev_xi": real_changes_a,
                        "pred_changes_from_prev_xi": pred_changes_a,
                        "condition_swaps_count": int(len(cond_changes_a)),
                        "condition_swaps": cond_changes_a[:6],
                        "verdict": _cond_verdict(int(len(cond_changes_a))),
                    },
                    "likely_gap_tags": tags_a,
                },
                team_b: {
                    "actual_xi": act_xi_b,
                    "predicted_xi": xi_pred_b,
                    **xi_diff_b,
                    "actual_batting_order": act_bo_b,
                    "predicted_batting_order": pred_bo_b,
                    "batting_order_overlap": {
                        "openers_positional": _order_overlap(act_bo_b, pred_bo_b, 2),
                        "top4_positional": _order_overlap(act_bo_b, pred_bo_b, 4),
                        "lower_8_11_set": _set_overlap(act_bo_b, pred_bo_b, [8, 9, 10, 11]),
                    },
                    "team_structure": {
                        "actual": struct_act_b,
                        "predicted": struct_pred_b,
                    },
                    "conditions_influence": {
                        "real_changes_from_prev_xi": real_changes_b,
                        "pred_changes_from_prev_xi": pred_changes_b,
                        "condition_swaps_count": int(len(cond_changes_b)),
                        "condition_swaps": cond_changes_b[:6],
                        "verdict": _cond_verdict(int(len(cond_changes_b))),
                    },
                    "likely_gap_tags": tags_b,
                },
            },
        }
        per_match.append(block)

    summary = {
        "matches_requested": len(match_urls),
        "matches_audited": sum(1 for m in per_match if not m.get("error")),
        "matches_failed": [m for m in per_match if m.get("error")],
        "recurring_gap_tags_top5": recurring_gap_tags.most_common(5),
    }
    return {"per_match": per_match, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="*", default=None, help="Match URLs (iplt20.com).")
    ap.add_argument("--json", action="store_true", help="Print raw JSON.")
    args = ap.parse_args()
    urls = args.urls if args.urls else DEFAULT_MATCH_URLS
    out = run_audit(urls)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    # compact human-readable summary
    print("IPL 2026 validation audit")
    print("matches_audited:", out["summary"]["matches_audited"], "of", out["summary"]["matches_requested"])
    if out["summary"]["matches_failed"]:
        print("matches_failed:", len(out["summary"]["matches_failed"]))
        for m in out["summary"]["matches_failed"][:6]:
            print(" -", m.get("url"), "error=", m.get("error"))
    print("recurring_gap_tags_top5:", out["summary"]["recurring_gap_tags_top5"])


if __name__ == "__main__":
    main()

