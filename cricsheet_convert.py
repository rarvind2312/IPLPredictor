"""
Convert Cricsheet IPL JSON (v1.0 / v1.1) into the internal ``insert_parsed_match`` payload
shape plus extended per-player rows for ``player_match_stats`` / ``player_phase_usage``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import ingest_normalize
import utils

logger = logging.getLogger(__name__)


def competition_label_from_info(info: dict[str, Any]) -> str:
    """Human-readable competition string for multi-league Cricsheet files."""
    ev = info.get("event")
    if isinstance(ev, dict):
        n = str(ev.get("name") or "").strip()
        if n:
            return n[:160]
    evs = info.get("events")
    if isinstance(evs, list) and evs:
        e0 = evs[0]
        if isinstance(e0, dict):
            n = str(e0.get("name") or "").strip()
            if n:
                return n[:160]
    c = str(info.get("competition") or "").strip()
    if c:
        return c[:160]
    return "Unknown"


def normalize_match_format_from_info(info: dict[str, Any]) -> str:
    """
    Map Cricsheet ``info.match_type`` to a short normalized code (T20, T20I, ODI, TEST, …).
    Empty when unknown — ingest should still store competition for audit.
    """
    mt = str(info.get("match_type") or "").strip().upper().replace(" ", "")
    if not mt:
        return ""
    aliases: dict[str, str] = {
        "T20": "T20",
        "T20I": "T20I",
        "IT20": "T20I",
        "WT20": "T20",
        "ODM": "ODI",
        "ODI": "ODI",
        "MDM": "ODM",
        "TEST": "TEST",
        "MSL": "T20",
    }
    return aliases.get(mt, mt[:16])

_SPIN_TOKENS: tuple[str, ...] = (
    "chahal",
    "kuldeep",
    "ashwin",
    "jadeja",
    "rashid",
    "hasaranga",
    "zampa",
    "santner",
    "mishra",
    "bishnoi",
    "washington",
    "chawla",
    "krunal",
    "axar",
    "gopal",
    "shamsi",
    "tabraiz",
    "noor ahmad",
    "ahmad",
    "finger",
    "mujeeb",
    "nabi",
    "theekshana",
    "sodhi",
    "murugan",
    "varun",
    "markande",
    "lamichhane",
    "karn sharma",
)

_PACE_STRONG: tuple[str, ...] = (
    "bumrah",
    "bolt",
    "rabada",
    "ferguson",
    "archer",
    "wood",
    "milne",
    "nortje",
    "siraj",
    "shami",
)


def classify_bowler_style(name: str) -> Optional[str]:
    """Heuristic spin vs pace for vs-spin / vs-pace aggregates (best-effort)."""
    n = (name or "").lower()
    if not n:
        return None
    for t in _SPIN_TOKENS:
        if t in n:
            return "spin"
    for t in _PACE_STRONG:
        if t in n:
            return "pace"
    if re.search(r"\b(lf|rf|lm|rm|lws|rws)\b", n):
        return "pace"
    return None


def _phase_for_over(over_zero: int) -> str:
    """
    Map Cricsheet 0-based over index to phase (T20 IPL):
    overs 1–6 → powerplay (indices 0–5), 7–15 → middle (6–14), 16–20 → death (15–19).
    """
    if over_zero <= 5:
        return "powerplay"
    if over_zero <= 14:
        return "middle"
    return "death"


def _delivery_batter(delivery: dict[str, Any]) -> str:
    """
    Striker name for this ball. Cricsheet uses ``batter`` (v1.1+); older files may use ``batsman``.
    """
    b = delivery.get("batter")
    if b is None or (isinstance(b, str) and not str(b).strip()):
        b = delivery.get("batsman")
    return str(b or "").strip()


def _outcome_margin(outcome: dict[str, Any]) -> str:
    if not outcome:
        return ""
    if "result" in outcome and outcome.get("result"):
        return str(outcome.get("result"))
    by = outcome.get("by") or {}
    parts: list[str] = []
    for k in ("runs", "wickets"):
        if k in by:
            parts.append(f"{k} {by[k]}")
    if "innings" in by:
        parts.append(f"innings {by['innings']}")
    if "method" in outcome:
        parts.append(str(outcome["method"]))
    return " | ".join(parts) if parts else ""


def _collect_replacements(
    innings: list[dict[str, Any]],
    stor: Callable[[str], str],
) -> list[tuple[str, str, str]]:
    """List of (team_stored_label, player_out, player_in) from impact-style replacements."""
    swaps: list[tuple[str, str, str]] = []
    for inn in innings:
        for ov in inn.get("overs") or []:
            for d in ov.get("deliveries") or []:
                rep = d.get("replacements") or {}
                for block in rep.get("match") or []:
                    po = (block.get("out") or "").strip()
                    pi = (block.get("in") or "").strip()
                    tm_raw = (block.get("team") or "").strip()
                    if po and pi and tm_raw:
                        swaps.append((stor(tm_raw), po, pi))
    return swaps


def _other_team_raw(teams_raw: list[str], batting_raw: str) -> str:
    for t in teams_raw:
        if t != batting_raw:
            return t
    return ""


def _resolve_playing_xi(
    team: str,
    listed: list[str],
    swaps: list[tuple[str, str, str]],
    participants: set[str],
) -> list[str]:
    """
    Build an 11-name XI: apply in/out swaps for this team, then trim/pad using participants.
    """
    names = [str(x).strip() for x in listed if str(x).strip()]
    for tm, po, pi in swaps:
        if tm != team:
            continue
        if po in names:
            i = names.index(po)
            names[i] = pi
        elif pi not in names:
            names.append(pi)
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    names = seen
    if len(names) == 11:
        return names
    if len(names) > 11:
        idx_map = {n: i for i, n in enumerate(names)}
        scored: list[tuple[int, int, str]] = []
        for n in names:
            scored.append((0 if n in participants else 1, idx_map.get(n, 99), n))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [t[2] for t in scored[:11]]
    for p in participants:
        if p not in names:
            names.append(p)
        if len(names) >= 11:
            break
    return names[:11]


def cricsheet_json_to_payload(
    data: dict[str, Any],
    *,
    cricsheet_match_id: str,
    url_scheme: str = "ipl",
) -> dict[str, Any]:
    info = data.get("info") or {}
    teams_raw = [str(x).strip() for x in (info.get("teams") or []) if str(x).strip()]
    if len(teams_raw) < 2:
        raise ValueError("Cricsheet JSON missing info.teams")
    team_resolved = [ingest_normalize.normalize_team_display_for_ingest(t) for t in teams_raw]
    raw_to_resolved: dict[str, str] = {
        teams_raw[i]: team_resolved[i] for i in range(min(len(teams_raw), len(team_resolved)))
    }

    def stor(tm_raw: str) -> str:
        t = (tm_raw or "").strip()
        if t in raw_to_resolved:
            return raw_to_resolved[t]
        return ingest_normalize.normalize_team_display_for_ingest(t)

    teams = team_resolved
    dates = list(info.get("dates") or [])
    match_date = dates[0] if dates else ""
    season = str(info.get("season") or (match_date[:4] if match_date else ""))
    venue = (info.get("venue") or "").strip() or None
    city = (info.get("city") or "").strip() or None
    toss = info.get("toss") or {}
    outcome = info.get("outcome") or {}
    winner = (outcome.get("winner") or "").strip() or None
    winner_stored = stor(winner) if winner else None

    innings_list = list(data.get("innings") or [])
    swaps = _collect_replacements(innings_list, stor)

    # --- Participation and delivery aggregates (team keys = storage labels) ---
    bat_runs: dict[tuple[str, str], int] = {}
    bat_balls: dict[tuple[str, str], int] = {}
    bat_fours: dict[tuple[str, str], int] = {}
    bat_sixes: dict[tuple[str, str], int] = {}
    # Primary batting order per team = first innings that team bats (main innings; not super-over).
    bat_order: dict[str, list[str]] = {teams[0]: [], teams[1]: []}
    primary_order_locked: set[str] = set()
    innings_batting_orders: list[dict[str, Any]] = []
    dismissal_kind: dict[tuple[str, str], str] = {}

    bowl_balls: dict[tuple[str, str], int] = {}
    bowl_runs: dict[tuple[str, str], int] = {}
    bowl_wk: dict[tuple[str, str], int] = {}

    phase_bat_runs: dict[tuple[str, str, str], int] = {}
    phase_bat_balls: dict[tuple[str, str, str], int] = {}
    phase_bowl_balls: dict[tuple[str, str, str], int] = {}
    phase_bowl_wk: dict[tuple[str, str, str], int] = {}
    vs_spin_balls: dict[tuple[str, str], int] = {}
    vs_pace_balls: dict[tuple[str, str], int] = {}

    participants: dict[str, set[str]] = {teams[0]: set(), teams[1]: set()}

    batting_first_team: Optional[str] = None
    for inn_idx, inn in enumerate(innings_list, start=1):
        bteam_raw = str(inn.get("team") or "").strip()
        if not bteam_raw:
            continue
        bteam = stor(bteam_raw)
        if bteam not in participants:
            participants[bteam] = set()
        if bteam not in bat_order:
            bat_order[bteam] = []
        if batting_first_team is None:
            batting_first_team = bteam
        fteam_raw = _other_team_raw(teams_raw, bteam_raw)
        fteam = stor(fteam_raw) if fteam_raw else ""
        if fteam and fteam not in participants:
            participants[fteam] = set()
        seen_this_innings: set[str] = set()
        order_this_innings: list[str] = []
        for ov in inn.get("overs") or []:
            over_idx = int(ov.get("over", 0))
            phase = _phase_for_over(over_idx)
            for d in ov.get("deliveries") or []:
                batter = _delivery_batter(d)
                bowler = str(d.get("bowler") or "").strip()
                runs = d.get("runs") or {}
                br = int(runs.get("batter") or 0)
                tot = int(runs.get("total") or 0)
                extras = d.get("extras") or {}
                is_wide = isinstance(extras, dict) and "wides" in extras
                legal = not is_wide

                if batter and bteam:
                    if batter not in seen_this_innings:
                        seen_this_innings.add(batter)
                        order_this_innings.append(batter)
                    participants[bteam].add(batter)
                    key_b = (bteam, batter)
                    bat_runs[key_b] = bat_runs.get(key_b, 0) + br
                    if br == 4:
                        bat_fours[key_b] = bat_fours.get(key_b, 0) + 1
                    if br == 6:
                        bat_sixes[key_b] = bat_sixes.get(key_b, 0) + 1
                    if legal:
                        bat_balls[key_b] = bat_balls.get(key_b, 0) + 1
                    pk = (bteam, batter, phase)
                    phase_bat_runs[pk] = phase_bat_runs.get(pk, 0) + br
                    if legal:
                        phase_bat_balls[pk] = phase_bat_balls.get(pk, 0) + 1
                    sty = classify_bowler_style(bowler)
                    if legal and sty == "spin":
                        vs_spin_balls[key_b] = vs_spin_balls.get(key_b, 0) + 1
                    elif legal and sty == "pace":
                        vs_pace_balls[key_b] = vs_pace_balls.get(key_b, 0) + 1

                if bowler and fteam:
                    participants[fteam].add(bowler)
                    key_o = (fteam, bowler)
                    bowl_runs[key_o] = bowl_runs.get(key_o, 0) + tot
                    if legal:
                        bowl_balls[key_o] = bowl_balls.get(key_o, 0) + 1
                    pk2 = (fteam, bowler, phase)
                    if legal:
                        phase_bowl_balls[pk2] = phase_bowl_balls.get(pk2, 0) + 1
                    for w in d.get("wickets") or []:
                        kind = str(w.get("kind") or "").lower()
                        player_out = str(w.get("player_out") or "").strip()
                        if kind == "run out":
                            continue
                        bowl_wk[key_o] = bowl_wk.get(key_o, 0) + 1
                        if player_out:
                            dismissal_kind[(bteam, player_out)] = kind or "unknown"
                        if legal:
                            phase_bowl_wk[pk2] = phase_bowl_wk.get(pk2, 0) + 1
                        break

                for w in d.get("wickets") or []:
                    po = str(w.get("player_out") or "").strip()
                    if po:
                        participants[bteam].add(po)

        if bteam and order_this_innings:
            innings_batting_orders.append(
                {
                    "innings_number": inn_idx,
                    "team": bteam,
                    "order": list(order_this_innings),
                }
            )
            if bteam not in primary_order_locked:
                bat_order[bteam] = list(order_this_innings)
                primary_order_locked.add(bteam)
            logger.info(
                "cricsheet id=%s innings=%d team=%s batting_order=%s",
                cricsheet_match_id,
                inn_idx,
                bteam,
                order_this_innings,
            )

    playing_xi: list[dict[str, Any]] = []
    players_block = info.get("players") or {}
    for tm_raw, tm_stored in zip(teams_raw, team_resolved):
        listed = list(players_block.get(tm_raw) or players_block.get(tm_stored) or [])
        xi = _resolve_playing_xi(tm_stored, listed, swaps, participants.get(tm_stored, set()))
        playing_xi.append({"team": tm_stored, "players": xi})

    xi_by_team = {str(s.get("team") or ""): list(s.get("players") or []) for s in playing_xi}

    batting_rows_by_team: dict[str, list[dict[str, Any]]] = {teams[0]: [], teams[1]: []}
    for tm in teams:
        order = bat_order.get(tm) or []
        xi_names_for_tm = xi_by_team.get(tm, [])
        for pos, name in enumerate(order, start=1):
            key = (tm, name)
            runs = bat_runs.get(key, 0)
            balls = bat_balls.get(key, 0)
            if runs == 0 and balls == 0 and name not in xi_names_for_tm:
                continue
            sr = round(100.0 * runs / max(1, balls), 3) if balls else None
            batting_rows_by_team[tm].append(
                {
                    "player": name,
                    "position": pos,
                    "runs": runs,
                    "balls": balls,
                    "fours": bat_fours.get(key, 0),
                    "sixes": bat_sixes.get(key, 0),
                    "strike_rate": sr,
                    "dismissal": dismissal_kind.get((tm, name)),
                }
            )

    bowling_rows_by_team: dict[str, list[dict[str, Any]]] = {teams[0]: [], teams[1]: []}
    for tm in teams:
        keys = [k for k in bowl_balls if k[0] == tm]
        for _t, name in sorted(keys, key=lambda x: x[1]):
            balls = bowl_balls.get((_t, name), 0)
            if balls <= 0:
                continue
            overs = round(balls / 6.0, 2)
            rc = bowl_runs.get((_t, name), 0)
            wk = bowl_wk.get((_t, name), 0)
            econ = round(6.0 * rc / max(1, balls), 3)
            bowling_rows_by_team[tm].append(
                {
                    "player": name,
                    "overs": overs,
                    "maidens": 0,
                    "runs": rc,
                    "wickets": wk,
                    "economy": econ,
                }
            )

    batting_payload = [{"team": tm, "rows": batting_rows_by_team[tm]} for tm in teams]
    bowling_payload = [{"team": tm, "rows": bowling_rows_by_team[tm]} for tm in teams]

    batting_order_payload = [{"team": tm, "order": bat_order.get(tm, [])} for tm in teams]
    bowlers_used_payload = [
        {
            "team": tm,
            "bowlers": [r["player"] for r in bowling_rows_by_team[tm]],
        }
        for tm in teams
    ]

    scheme = (url_scheme or "ipl").strip().lower()
    if scheme == "all":
        url = f"cricsheet://all/{cricsheet_match_id}"
        mf = normalize_match_format_from_info(info) or "T20"
        comp_l = competition_label_from_info(info)
        meta = {
            "url": url,
            "source": "cricsheet_all_archive",
            "date": match_date,
            "venue": venue,
            "city": city,
            "season": season,
            "cricsheet_match_id": cricsheet_match_id,
            "winner": winner_stored or winner,
            "toss_winner": stor((toss.get("winner") or "").strip()) if (toss.get("winner") or "").strip() else None,
            "toss_decision": (toss.get("decision") or "").strip() or None,
            "batting_first": batting_first_team,
            "margin": _outcome_margin(outcome),
            "competition": comp_l,
            "match_format": mf,
            "result_text": _outcome_margin(outcome),
        }
    else:
        url = f"cricsheet://ipl/{cricsheet_match_id}"
        meta = {
            "url": url,
            "source": "cricsheet",
            "date": match_date,
            "venue": venue,
            "city": city,
            "season": season,
            "cricsheet_match_id": cricsheet_match_id,
            "winner": winner_stored or winner,
            "toss_winner": stor((toss.get("winner") or "").strip()) if (toss.get("winner") or "").strip() else None,
            "toss_decision": (toss.get("decision") or "").strip() or None,
            "batting_first": batting_first_team,
            "margin": _outcome_margin(outcome),
            "competition": "IPL",
            "match_format": "T20",
            "result_text": _outcome_margin(outcome),
        }

    payload: dict[str, Any] = {
        "meta": meta,
        "teams": teams,
        "playing_xi": playing_xi,
        "batting": batting_payload,
        "bowling": bowling_payload,
        "batting_order": batting_order_payload,
        "bowlers_used": bowlers_used_payload,
        "innings_batting_orders": innings_batting_orders,
    }

    # --- Extended rows for SQLite detail tables (match_id filled in by db layer) ---
    player_stats_extended: list[dict[str, Any]] = []
    player_phase_extended: list[dict[str, Any]] = []

    for tm in teams:
        tk = ingest_normalize.normalize_team_key_for_ingest(tm)
        xi_names = xi_by_team.get(tm, [])
        xi_set = set(xi_names)

        by_name: dict[str, dict[str, Any]] = {}
        for row in batting_rows_by_team[tm]:
            nm = row["player"]
            by_name[nm] = {
                "team_name": tm,
                "team_key": tk,
                "canonical_team_key": tk,
                "player_name": nm,
                "player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                "canonical_player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                "runs": row.get("runs"),
                "balls": row.get("balls"),
                "fours": row.get("fours"),
                "sixes": row.get("sixes"),
                "strike_rate": row.get("strike_rate"),
                "batting_position": row.get("position"),
                "dismissal_type": row.get("dismissal"),
                "selected_in_xi": 1 if nm in xi_set else 0,
                "season": season,
                "vs_spin_balls_faced": vs_spin_balls.get((tm, nm), 0),
                "vs_pace_balls_faced": vs_pace_balls.get((tm, nm), 0),
            }
        for row in bowling_rows_by_team[tm]:
            nm = row["player"]
            st = by_name.setdefault(
                nm,
                {
                    "team_name": tm,
                    "team_key": tk,
                    "canonical_team_key": tk,
                    "player_name": nm,
                    "player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                    "canonical_player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                    "season": season,
                    "selected_in_xi": 1 if nm in xi_set else 0,
                },
            )
            st["overs_bowled"] = row.get("overs")
            st["wickets"] = row.get("wickets")
            st["runs_conceded"] = row.get("runs")
            st["economy"] = row.get("economy")
            if "selected_in_xi" not in st:
                st["selected_in_xi"] = 1 if nm in xi_set else 0

        for nm in xi_names:
            if nm not in by_name:
                by_name[nm] = {
                    "team_name": tm,
                    "team_key": tk,
                    "canonical_team_key": tk,
                    "player_name": nm,
                    "player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                    "canonical_player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                    "selected_in_xi": 1,
                    "season": season,
                }

        for st in by_name.values():
            player_stats_extended.append(st)

        for phase in ("powerplay", "middle", "death"):
            for nm in bat_order.get(tm, []):
                pk = (tm, nm, phase)
                bb = phase_bat_balls.get(pk, 0)
                br = phase_bat_runs.get(pk, 0)
                if bb == 0 and br == 0:
                    continue
                player_phase_extended.append(
                    {
                        "team_name": tm,
                        "team_key": tk,
                        "canonical_team_key": tk,
                        "player_name": nm,
                        "player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                        "canonical_player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                        "role": "bat",
                        "phase": phase,
                        "balls": bb,
                        "runs": br,
                        "wickets": 0,
                        "vs_spin_balls": 0,
                        "vs_pace_balls": 0,
                    }
                )
            keys = [k for k in phase_bowl_balls if k[0] == tm and k[2] == phase]
            for _t, nm, ph in keys:
                bb = phase_bowl_balls.get((_t, nm, ph), 0)
                wk = phase_bowl_wk.get((_t, nm, ph), 0)
                if bb == 0 and wk == 0:
                    continue
                player_phase_extended.append(
                    {
                        "team_name": tm,
                        "team_key": tk,
                        "canonical_team_key": tk,
                        "player_name": nm,
                        "player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                        "canonical_player_key": ingest_normalize.normalize_player_key_for_ingest(nm),
                        "role": "bowl",
                        "phase": phase,
                        "balls": bb,
                        "runs": 0,
                        "wickets": wk,
                        "vs_spin_balls": 0,
                        "vs_pace_balls": 0,
                    }
                )

    payload["player_stats_extended"] = player_stats_extended
    payload["player_phase_extended"] = player_phase_extended
    payload["meta"]["canonical_match_key"] = utils.canonical_match_identity_key(
        teams[0], teams[1], match_date
    )
    return payload


def load_cricsheet_payload(
    json_path: Path | str,
    *,
    cricsheet_match_id: str,
    url_scheme: str = "ipl",
) -> dict[str, Any]:
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return cricsheet_json_to_payload(data, cricsheet_match_id=cricsheet_match_id, url_scheme=url_scheme)
