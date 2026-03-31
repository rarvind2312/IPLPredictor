"""IPLT20.com match scorecard parser — structured ``__NEXT_DATA__`` / JSON first, minimal HTML."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Callable, Optional

from bs4 import BeautifulSoup, Tag

import config

from parsers._common import base_meta, clean_player_name, make_soup

logger = logging.getLogger(__name__)

_JUNK_VENUE_RE = re.compile(r"^#([0-9a-f]{3,8});?\s*$", re.I)
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.I,
)


def _extract_next_data_json(html: str) -> Optional[dict[str, Any]]:
    m = re.search(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, re.S | re.I)
    if not m:
        return None
    raw = (m.group(1) or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_ld_json_blobs(html: str) -> list[Any]:
    out: list[Any] = []
    try:
        soup = make_soup(html)
    except Exception:  # noqa: BLE001
        return out
    for script in soup.find_all("script"):
        typ = (script.get("type") or "").lower()
        if "ld+json" not in typ:
            continue
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            blob = json.loads(raw)
            if isinstance(blob, list):
                out.extend(blob)
            else:
                out.append(blob)
        except json.JSONDecodeError:
            continue
    return out


def _is_valid_venue_text(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 4:
        return False
    low = t.lower()
    if low in ("null", "none", "tbc", "n/a", "-", "undefined"):
        return False
    if _JUNK_VENUE_RE.match(t):
        return False
    if t.startswith("#") and len(t) <= 9:
        return False
    if "rgb(" in low or "rgba(" in low or "hsl(" in low:
        return False
    if not re.search(r"[A-Za-z\u00c0-\u024f]", t):
        return False
    # Reject obvious CSS fragments
    if ";" in t and len(t) < 30 and not re.search(r"\b(stadium|ground|arena|park)\b", low):
        return False
    return True


def _coerce_text_date(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # epoch ms
        if val > 1_000_000_000_000:
            try:
                return datetime.utcfromtimestamp(val / 1000.0).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                return None
        if val > 1_000_000_000:
            try:
                return datetime.utcfromtimestamp(float(val)).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                return None
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    m = _ISO_DATE_RE.search(s)
    if m:
        return m.group(1)
    m2 = _MONTH_DATE_RE.search(s)
    if m2:
        return m2.group(1).strip()
    return None


def _entity_name(obj: Any) -> str:
    if isinstance(obj, str):
        return clean_player_name(obj)
    if not isinstance(obj, dict):
        return ""
    for key in (
        "fullName",
        "teamFullName",
        "teamName",
        "name",
        "longName",
        "displayName",
        "shortName",
        "playerName",
        "title",
    ):
        v = obj.get(key)
        if isinstance(v, str) and len(v.strip()) > 1:
            return clean_player_name(v)
    return ""


def _team_pair_from_dict(d: dict[str, Any]) -> tuple[str, str]:
    pairs = (
        ("team1", "team2"),
        ("homeTeam", "awayTeam"),
        ("teamA", "teamB"),
        ("firstTeam", "secondTeam"),
        ("battingFirst", "bowlingFirst"),
    )
    for a, b in pairs:
        if a in d and b in d:
            t1, t2 = _entity_name(d[a]), _entity_name(d[b])
            if t1 and t2:
                return t1, t2
    return "", ""


def _venue_from_value(val: Any) -> tuple[Optional[str], str]:
    if isinstance(val, str):
        v = val.strip()
        return (v if _is_valid_venue_text(v) else None, "string")
    if isinstance(val, dict):
        for key in ("name", "fullName", "venueName", "stadiumName", "ground", "title", "longName"):
            v = val.get(key)
            if isinstance(v, str) and _is_valid_venue_text(v):
                return clean_player_name(v), f"dict.{key}"
        city = val.get("city")
        ground = val.get("ground") or val.get("stadium")
        parts = []
        if isinstance(ground, str) and _is_valid_venue_text(ground):
            parts.append(clean_player_name(ground))
        if isinstance(city, str) and len(city.strip()) > 1:
            parts.append(clean_player_name(city))
        if parts:
            return ", ".join(parts), "dict.composite"
    return None, ""


def _date_from_dict(d: dict[str, Any]) -> tuple[Optional[str], str]:
    for key in (
        "matchDate",
        "matchDateTime",
        "matchDateTimeLocal",
        "fixtureDate",
        "scheduledStart",
        "startDate",
        "date",
        "gameDate",
        "matchDateUtc",
    ):
        if key not in d:
            continue
        norm = _coerce_text_date(d.get(key))
        if norm:
            return norm, key
    # Nested schedule
    sched = d.get("schedule") or d.get("fixture")
    if isinstance(sched, dict):
        for key in ("matchDate", "date", "startDate"):
            norm = _coerce_text_date(sched.get(key))
            if norm:
                return norm, f"schedule.{key}"
    return None, ""


def _player_name_from_bat_row(row: dict[str, Any]) -> str:
    for key in ("name", "playerName", "fullName", "batsmanName", "strikerName"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return clean_player_name(v)
    for nest in ("player", "batsman", "striker", "batting", "batter"):
        v = row.get(nest)
        n = _entity_name(v)
        if n:
            return n
    return ""


def _batting_rows_from_list(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pos = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        nm = _player_name_from_bat_row(item)
        if not nm:
            continue
        low = nm.lower()
        if low in ("extras", "total", "did not bat", "yet to bat"):
            continue
        pos += 1
        runs = item.get("runs") if item.get("runs") is not None else item.get("r")
        balls = item.get("balls") if item.get("balls") is not None else item.get("b")
        try:
            ri = int(runs) if runs is not None and str(runs).strip() not in ("", "-") else None
        except (TypeError, ValueError):
            ri = None
        try:
            bi = int(balls) if balls is not None and str(balls).strip() not in ("", "-") else None
        except (TypeError, ValueError):
            bi = None
        out.append({"player": nm, "position": pos, "runs": ri, "balls": bi})
    return out


def _bowling_rows_from_list(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        nm = ""
        for key in ("name", "playerName", "fullName", "bowlerName"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                nm = clean_player_name(v)
                break
        if not nm:
            nm = _entity_name(item.get("bowler") or item.get("player"))
        if not nm:
            continue
        def _pf(x: Any) -> Optional[float]:
            if x is None:
                return None
            m = re.search(r"\d+(\.\d+)?", str(x))
            return float(m.group(0)) if m else None

        def _pi(x: Any) -> Optional[int]:
            if x is None:
                return None
            m = re.search(r"\d+", str(x))
            return int(m.group(0)) if m else None

        overs = item.get("overs") if item.get("overs") is not None else item.get("o")
        maidens = item.get("maidens") if item.get("maidens") is not None else item.get("m")
        runs = item.get("runs") if item.get("runs") is not None else item.get("r")
        wk = item.get("wickets") if item.get("wickets") is not None else item.get("w")
        out.append(
            {
                "player": nm,
                "overs": _pf(overs),
                "maidens": _pi(maidens) or 0,
                "runs": _pi(runs),
                "wickets": _pi(wk),
            }
        )
    return out


def _innings_team_name(inn: dict[str, Any]) -> str:
    for key in ("team", "battingTeam", "batting_team", "teamDetail", "teamDetails"):
        v = inn.get(key)
        n = _entity_name(v)
        if n:
            return n
    return ""


def _batting_list_from_innings(inn: dict[str, Any]) -> list[Any]:
    for key in (
        "batsmen",
        "batsmans",
        "batsman",
        "batting",
        "battingRows",
        "battingCard",
        "batters",
        "teamBatting",
    ):
        v = inn.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def _bowling_list_from_innings(inn: dict[str, Any]) -> list[Any]:
    for key in ("bowlers", "bowler", "bowling", "bowlingRows", "bowlingCard", "teamBowling"):
        v = inn.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def _parse_innings_list(innings: list[Any], teams_hint: list[str]) -> tuple[list[dict], list[dict]]:
    batting: list[dict[str, Any]] = []
    bowling: list[dict[str, Any]] = []
    for inn in innings:
        if not isinstance(inn, dict):
            continue
        team = _innings_team_name(inn)
        if not team and teams_hint:
            team = teams_hint[len(batting) % len(teams_hint)]
        br = _batting_rows_from_list(_batting_list_from_innings(inn))
        if br:
            batting.append({"team": team, "rows": br})
        bwl = _bowling_rows_from_list(_bowling_list_from_innings(inn))
        if bwl:
            bowling.append({"team": team, "rows": bwl})
    return batting, bowling


def _playing_xi_flag_active(p: dict[str, Any]) -> bool:
    if p.get("isPlaying") is False or p.get("playingXI") is False:
        return False
    for k in ("isPlaying", "playingXI", "onPlayingXI", "inPlayingXI"):
        v = p.get(k)
        if v is True or v == 1 or v == "1":
            return True
    return False


def _playing_xi_from_players_list(players: list[Any], team_name: str) -> Optional[list[str]]:
    flag_keys = ("isPlaying", "playingXI", "onPlayingXI", "inPlayingXI")
    squad_uses_flags = any(isinstance(p, dict) and any(k in p for k in flag_keys) for p in players)
    names = [_entity_name(p) for p in players if isinstance(p, dict)]
    names = [n for n in names if n]

    if squad_uses_flags:
        out: list[str] = []
        for p in players:
            if not isinstance(p, dict):
                continue
            if not _playing_xi_flag_active(p):
                continue
            nm = _entity_name(p)
            if nm and nm.lower() not in {x.lower() for x in out}:
                out.append(nm)
            if len(out) >= 11:
                break
        if len(out) >= 5:
            return out[:11]
        return None

    if 11 <= len(names) <= 14:
        return names[:11]
    if 5 <= len(names) <= 11:
        return names
    return None


def _playing_xi_blocks_from_match_dict(d: dict[str, Any], teams: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for key in ("playingXI", "playing11", "playingX1", "lineUps", "lineups", "squads", "teamSquad"):
        v = d.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            if "players" in v[0] or "playerName" in v[0]:
                for i, block in enumerate(v[:2]):
                    if not isinstance(block, dict):
                        continue
                    pl = block.get("players") or block.get("squad") or block.get("playingXI")
                    tname = _entity_name(block.get("team")) or (
                        teams[i] if i < len(teams) else ""
                    )
                    if isinstance(pl, list):
                        xi = _playing_xi_from_players_list(pl, tname)
                        if xi:
                            blocks.append({"team": tname or (teams[i] if i < len(teams) else "Team"), "players": xi})
                if blocks:
                    return blocks[:2]
    for tk in ("team1", "team2", "homeTeam", "awayTeam"):
        if tk not in d:
            continue
        tobj = d.get(tk)
        if not isinstance(tobj, dict):
            continue
        pl = tobj.get("players") or tobj.get("squad") or tobj.get("playingXI") or tobj.get("playing11")
        if isinstance(pl, list):
            tname = _entity_name(tobj)
            xi = _playing_xi_from_players_list(pl, tname)
            if xi:
                blocks.append({"team": tname, "players": xi})
    return blocks[:2]


def _result_from_dict(d: dict[str, Any]) -> tuple[str, str]:
    for key in (
        "result",
        "matchResult",
        "outcomeDescription",
        "winningSummary",
        "status",
        "matchStatus",
        "resultDescription",
    ):
        v = d.get(key)
        if isinstance(v, str) and len(v.strip()) > 8:
            low = v.lower()
            if any(x in low for x in ("won", "tied", "no result", "abandoned", "tie")):
                return clean_player_name(v), key
    return "", ""


def _merge_meta(meta: dict[str, Any], d: dict[str, Any]) -> None:
    v, vs = _venue_from_value(d.get("venue"))
    if v and not meta.get("venue"):
        meta["venue"] = v
        meta["ipl_venue_source"] = vs
    dt, dk = _date_from_dict(d)
    if dt and not meta.get("date"):
        meta["date"] = dt
        meta["ipl_date_source"] = dk
    margin, mk = _result_from_dict(d)
    if margin and not (meta.get("margin") or "").strip():
        meta["margin"] = margin
        meta["ipl_margin_source"] = mk


def _score_candidate(
    teams: list[str],
    batting: list[dict],
    bowling: list[dict],
    playing_xi: list[dict],
) -> int:
    s = 0
    s += len(teams) * 5
    for inn in batting:
        s += len(inn.get("rows") or []) * 3
    for inn in bowling:
        s += len(inn.get("rows") or []) * 2
    for blk in playing_xi:
        s += len(blk.get("players") or []) * 2
    return s


def _extract_from_subtree(
    obj: Any,
    path: str,
    depth: int = 0,
) -> Optional[dict[str, Any]]:
    if depth > 18 or not isinstance(obj, dict):
        return None
    teams: list[str] = []
    t1, t2 = _team_pair_from_dict(obj)
    if t1 and t2:
        teams = [t1, t2]
    meta: dict[str, Any] = {}
    _merge_meta(meta, obj)

    innings_raw = obj.get("innings") or obj.get("inningsDetail") or obj.get("inningsDetails")
    batting: list[dict[str, Any]] = []
    bowling: list[dict[str, Any]] = []
    if isinstance(innings_raw, list) and innings_raw:
        batting, bowling = _parse_innings_list(innings_raw, teams)

    playing_xi = _playing_xi_blocks_from_match_dict(obj, teams)

    # Scorecard nested
    sc = obj.get("scorecard") or obj.get("scoreCard") or obj.get("fullScorecard")
    if isinstance(sc, dict):
        inn2 = sc.get("innings") or sc.get("inningsDetail")
        if isinstance(inn2, list) and inn2:
            b2, bw2 = _parse_innings_list(inn2, teams)
            if _score_candidate(teams, b2, bw2, playing_xi) > _score_candidate(teams, batting, bowling, playing_xi):
                batting, bowling = b2, bw2
        _merge_meta(meta, sc)
        if not playing_xi:
            playing_xi = _playing_xi_blocks_from_match_dict(sc, teams)

    if teams or batting or bowling or playing_xi:
        return {
            "path": path,
            "teams": teams,
            "meta": meta,
            "batting": batting,
            "bowling": bowling,
            "playing_xi": playing_xi,
            "_score": _score_candidate(teams, batting, bowling, playing_xi),
        }

    # Recurse into likely containers
    best: Optional[dict[str, Any]] = None
    for key in (
        "matchDetail",
        "matchDetails",
        "match",
        "fixture",
        "data",
        "content",
        "pageData",
        "scorecard",
        "scoreCard",
        "details",
        "matchInfo",
        "matchCentre",
    ):
        child = obj.get(key)
        if isinstance(child, dict):
            sub = _extract_from_subtree(child, f"{path}.{key}", depth + 1)
            if sub and (best is None or int(sub.get("_score") or 0) > int(best.get("_score") or 0)):
                best = sub
    return best


def _iter_structured_roots(next_data: dict[str, Any]) -> list[tuple[str, Any]]:
    roots: list[tuple[str, Any]] = []
    props = next_data.get("props")
    if isinstance(props, dict):
        pp = props.get("pageProps")
        if isinstance(pp, dict):
            roots.append(("props.pageProps", pp))
            dq = pp.get("dehydratedState")
            if isinstance(dq, dict):
                for i, q in enumerate(dq.get("queries") or []):
                    if not isinstance(q, dict):
                        continue
                    data = (q.get("state") or {}).get("data")
                    if data is not None:
                        roots.append((f"props.pageProps.dehydratedState.queries[{i}].data", data))
    return roots


def _parse_next_data_match(
    next_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns (payload_fragment, debug_dict).
    payload_fragment keys: teams, playing_xi, batting, bowling, meta_partial
    """
    debug: dict[str, Any] = {
        "scorecard_json_found": False,
        "roots_scanned": 0,
        "chosen_path": None,
        "teams_source": None,
        "innings_parsed": 0,
        "xi_players_count": 0,
        "batting_rows_count": 0,
        "bowling_rows_count": 0,
    }
    best: Optional[dict[str, Any]] = None
    roots = _iter_structured_roots(next_data)
    debug["roots_scanned"] = len(roots)
    for rpath, root in roots:
        cand = _extract_from_subtree(root, rpath, 0)
        if cand and (best is None or int(cand.get("_score") or 0) > int(best.get("_score") or 0)):
            best = cand

    if not best:
        return (
            {"teams": [], "playing_xi": [], "batting": [], "bowling": [], "meta": {}},
            debug,
        )

    debug["scorecard_json_found"] = True
    debug["chosen_path"] = best.get("path")
    teams = list(best.get("teams") or [])
    if len(teams) >= 2:
        debug["teams_source"] = f"structured_pair:{best.get('path')}"
    batting = list(best.get("batting") or [])
    bowling = list(best.get("bowling") or [])
    playing_xi = list(best.get("playing_xi") or [])
    meta_partial = dict(best.get("meta") or {})

    debug["innings_parsed"] = len(batting)
    debug["batting_rows_count"] = sum(len(inn.get("rows") or []) for inn in batting)
    debug["bowling_rows_count"] = sum(len(inn.get("rows") or []) for inn in bowling)
    debug["xi_players_count"] = sum(len(b.get("players") or []) for b in playing_xi)

    return (
        {
            "teams": teams,
            "playing_xi": playing_xi,
            "batting": batting,
            "bowling": bowling,
            "meta": meta_partial,
        },
        debug,
    )


def _scorecard_tables_from_main(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse tables only under ``main`` or scorecard-like sections (avoid site chrome)."""
    scope: Optional[Tag] = soup.find("main") or soup.find(attrs={"role": "main"})
    if not isinstance(scope, Tag):
        scope = soup.find("body") or soup
    assert scope is not None
    batting: list[dict[str, Any]] = []
    bowling: list[dict[str, Any]] = []
    for table in scope.find_all("table"):
        header_cells = [clean_player_name(th.get_text()).lower() for th in table.find_all("th")]
        htxt = " ".join(header_cells)
        if "bat" in htxt and "run" in htxt:
            rows_out: list[dict[str, Any]] = []
            pos = 0
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                link = tds[0].find("a")
                raw = link.get_text() if link else tds[0].get_text()
                nm = clean_player_name(raw)
                if not nm or nm.lower() in ("extras", "total"):
                    continue
                nums = []
                for td in tds[1:5]:
                    m = re.search(r"\d+", td.get_text())
                    if m:
                        nums.append(int(m.group(0)))
                pos += 1
                rows_out.append(
                    {
                        "player": nm,
                        "position": pos,
                        "runs": nums[0] if nums else None,
                        "balls": nums[1] if len(nums) > 1 else None,
                    }
                )
            if rows_out:
                batting.append({"team": "", "rows": rows_out})
        looks_like_bowling = ("over" in htxt or htxt.split()[0:1] == ["o"]) and (
            "wkt" in htxt or "wickets" in htxt or re.search(r"\bw\b", htxt)
        )
        if looks_like_bowling:
            rows_b: list[dict[str, Any]] = []
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                nm = clean_player_name(tds[0].get_text())
                if not nm:
                    continue

                def pf(i: int) -> Optional[float]:
                    m = re.search(r"\d+(\.\d+)?", tds[i].get_text())
                    return float(m.group(0)) if m else None

                def pi(i: int) -> Optional[int]:
                    m = re.search(r"\d+", tds[i].get_text())
                    return int(m.group(0)) if m else None

                rows_b.append(
                    {
                        "player": nm,
                        "overs": pf(1),
                        "maidens": pi(2) or 0,
                        "runs": pi(3),
                        "wickets": pi(4) if len(tds) > 4 else None,
                    }
                )
            if rows_b:
                bowling.append({"team": "", "rows": rows_b})
    return batting, bowling


def parse(html: str, url: str) -> dict[str, Any]:
    meta: dict[str, Any] = {**base_meta(url, "ipl")}
    meta.setdefault("competition", getattr(config, "IPL_COMPETITION_LABEL", "IPL"))
    meta["ipl_structured_attempted"] = True

    sm = re.search(r"/(20\d{2})/", url)
    if sm:
        meta.setdefault("season", sm.group(1))

    debug: dict[str, Any] = {
        "parser_path": "empty",
        "teams_source": None,
        "venue_source": meta.get("ipl_venue_source"),
        "date_source": meta.get("ipl_date_source"),
        "scorecard_container_found": False,
        "innings_parsed": 0,
        "xi_players_total": 0,
        "batting_rows_total": 0,
        "bowling_rows_total": 0,
    }

    try:
        soup = make_soup(html)
    except Exception:  # noqa: BLE001
        meta["ipl_parse_debug"] = {**debug, "parser_path": "soup_failed"}
        logger.warning("ipl_parser: BeautifulSoup failed url=%s", url)
        return {
            "meta": meta,
            "teams": [],
            "playing_xi": [],
            "batting": [],
            "bowling": [],
        }

    teams: list[str] = []
    playing_xi: list[dict[str, Any]] = []
    batting: list[dict[str, Any]] = []
    bowling: list[dict[str, Any]] = []

    next_data = _extract_next_data_json(html)
    if next_data:
        frag, nd_dbg = _parse_next_data_match(next_data)
        teams = list(frag.get("teams") or [])
        playing_xi = list(frag.get("playing_xi") or [])
        batting = list(frag.get("batting") or [])
        bowling = list(frag.get("bowling") or [])
        for k, v in (frag.get("meta") or {}).items():
            if v is not None and (k not in meta or not str(meta.get(k) or "").strip()):
                meta[k] = v
        debug["parser_path"] = "next_data"
        debug["scorecard_container_found"] = bool(nd_dbg.get("scorecard_json_found"))
        debug["innings_parsed"] = int(nd_dbg.get("innings_parsed") or 0)
        debug["xi_players_total"] = int(nd_dbg.get("xi_players_count") or 0)
        debug["batting_rows_total"] = int(nd_dbg.get("batting_rows_count") or 0)
        debug["bowling_rows_total"] = int(nd_dbg.get("bowling_rows_count") or 0)
        debug["teams_source"] = nd_dbg.get("teams_source") or (
            "structured_pair" if len(teams) >= 2 else None
        )
        debug["venue_source"] = meta.get("ipl_venue_source")
        debug["date_source"] = meta.get("ipl_date_source")
        meta["ipl_next_data_debug"] = nd_dbg
    else:
        debug["parser_path"] = "no_next_data"
        logger.info("ipl_parser: no __NEXT_DATA__ in HTML url=%s", url)

    # Optional: JSON-LD SportsEvent (venue / teams / date) — structured only
    for blob in _extract_ld_json_blobs(html):
        if not isinstance(blob, dict):
            continue
        typ = blob.get("@type")
        types = typ if isinstance(typ, list) else [typ]
        if not any(str(t).lower() in ("sportsevent", "sports event", "event") for t in types if t):
            continue
        h = blob.get("homeTeam") or blob.get("competitor")
        a = blob.get("awayTeam")
        if isinstance(h, dict) and isinstance(a, dict):
            hn, an = _entity_name(h), _entity_name(a)
            if hn and an and len(teams) < 2:
                teams = [hn, an]
                debug["teams_source"] = debug.get("teams_source") or "json_ld.SportsEvent"
        loc = blob.get("location")
        v, vs = _venue_from_value(loc)
        if v and not meta.get("venue"):
            meta["venue"] = v
            meta["ipl_venue_source"] = f"json_ld.{vs}"
        start = blob.get("startDate") or blob.get("endDate")
        ds = _coerce_text_date(start)
        if ds and not meta.get("date"):
            meta["date"] = ds
            meta["ipl_date_source"] = "json_ld.startDate"
        break

    # HTML tables only inside main (never title / og for teams)
    if not batting and not bowling:
        b2, bw2 = _scorecard_tables_from_main(soup)
        if b2 or bw2:
            batting, bowling = b2, bw2
            debug["parser_path"] = f"{debug.get('parser_path')}+main_tables"
            debug["scorecard_container_found"] = True
            debug["batting_rows_total"] = sum(len(x.get("rows") or []) for x in batting)
            debug["bowling_rows_total"] = sum(len(x.get("rows") or []) for x in bowling)
            debug["innings_parsed"] = max(len(batting), len(bowling))

    # Result: prefer structured meta margin; else short text in main only
    if not (meta.get("margin") or "").strip():
        scope = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("body")
        if isinstance(scope, Tag):
            for tag in scope.find_all(["p", "div", "span"]):
                tx = tag.get_text(" ", strip=True)
                if 15 < len(tx) < 220 and re.search(r"won by|won the match|match tied|no result", tx, re.I):
                    meta["margin"] = clean_player_name(tx)
                    meta["ipl_margin_source"] = "main_text"
                    break

    if not (meta.get("venue") or "").strip() or not _is_valid_venue_text(str(meta.get("venue") or "")):
        meta.pop("venue", None)
        meta.pop("ipl_venue_source", None)

    if playing_xi:
        debug["xi_players_total"] = sum(len(b.get("players") or []) for b in playing_xi)

    meta["ipl_parse_debug"] = debug

    logger.info(
        "ipl_parser: path=%s teams=%d xi_blocks=%s batting_innings=%d bowling_innings=%d url=%s",
        debug.get("parser_path"),
        len(teams),
        len(playing_xi),
        len(batting),
        len(bowling),
        url,
    )

    return {
        "meta": meta,
        "teams": teams,
        "playing_xi": playing_xi,
        "batting": batting,
        "bowling": bowling,
    }
