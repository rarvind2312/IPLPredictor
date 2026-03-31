"""Cricbuzz scorecard HTML parser."""

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from parsers._common import base_meta, clean_player_name, make_soup


def _parse_int(val: str) -> Optional[int]:
    val = (val or "").strip()
    if not val or val in ("-", ""):
        return None
    m = re.search(r"-?\d+", val.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_float(val: str) -> Optional[float]:
    val = (val or "").strip()
    if not val or val == "-":
        return None
    m = re.search(r"\d+(\.\d+)?", val)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _teams(soup: BeautifulSoup) -> list[str]:
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string
    m = re.search(
        r"^\s*([^|]+?)\s+vs\s+([^|]+?)(?:\s*\||$)",
        title,
        flags=re.I,
    )
    if m:
        return [clean_player_name(m.group(1)), clean_player_name(m.group(2))]
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        c = og["content"]
        m2 = re.search(r"^(.+?)\s+vs\s+(.+?)(?:\s*\||$)", c, flags=re.I)
        if m2:
            return [clean_player_name(m2.group(1)), clean_player_name(m2.group(2))]
    return []


def _header_meta(soup: BeautifulSoup) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for div in soup.find_all("div"):
        tx = div.get_text(" ", strip=True)
        if len(tx) > 240:
            continue
        if re.search(r"won by|won the match|Match tied|No result", tx, re.I):
            out["margin"] = tx
        if tx.lower().startswith("toss"):
            out["toss_line"] = tx
            mw = re.search(
                r"toss[:\s]+(.+?)(?:\s+elected|\s+opted|\s+chose|$)",
                tx,
                flags=re.I,
            )
            if mw:
                out["toss_winner"] = clean_player_name(mw.group(1))
            if re.search(r"bat(?:t)?ing first", tx, re.I):
                out["toss_decision"] = "bat"
            elif re.search(r"bowl(?:ing)? first", tx, re.I):
                out["toss_decision"] = "bowl"
        if tx.lower().startswith("venue"):
            out["venue"] = clean_player_name(re.sub(r"^Venue[:\s]+", "", tx, flags=re.I))
        if re.fullmatch(r"\d{1,2}\s+\w+\s+\d{4}", tx.strip()):
            out["date"] = tx.strip()
    return out


def _playing_xi(soup: BeautifulSoup, teams: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for div in soup.find_all("div"):
        cls = " ".join(div.get("class") or [])
        if "cb-play11" not in cls and "play11" not in cls.lower():
            continue
        hdr = div.find(["div", "span"], class_=re.compile(r"hdr|title|team", re.I))
        team_name = ""
        if hdr:
            team_name = clean_player_name(hdr.get_text())
        players: list[str] = []
        for a in div.find_all("a", href=True):
            href = a.get("href") or ""
            if "/profiles/" in href or "/player/" in href:
                nm = clean_player_name(a.get_text())
                if nm and nm not in players and len(nm) < 80:
                    players.append(nm)
        if not players:
            for li in div.find_all("li"):
                nm = clean_player_name(li.get_text())
                if nm and re.match(r"^[A-Za-z]", nm) and nm not in players:
                    players.append(nm)
        if players:
            sections.append({"team": team_name, "players": players[:11]})
    # Name association by order if two team names known
    if len(sections) == 2 and teams:
        for i, s in enumerate(sections):
            if not s.get("team"):
                s["team"] = teams[i] if i < len(teams) else s["team"]
    return sections[:2]


def _batting_innings(soup: BeautifulSoup, teams: list[str]) -> list[dict[str, Any]]:
    innings: list[dict[str, Any]] = []
    for idx, head in enumerate(soup.find_all(string=re.compile(r"Innings\b", re.I))):
        parent = head.parent
        if not parent:
            continue
        block = parent.find_parent("div")
        if not block:
            block = parent
        label = clean_player_name(parent.get_text(" ", strip=True))
        team_guess = ""
        for t in teams:
            if t and t.lower() in label.lower():
                team_guess = t
                break
        if not team_guess:
            team_guess = label.split("Innings")[0].strip() or f"Innings {idx+1}"

        table = block.find_next("table")
        if not table:
            continue
        rows_out: list[dict[str, Any]] = []
        pos = 0
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            link = tds[0].find("a", href=True)
            if not link:
                continue
            nm = clean_player_name(link.get_text())
            if not nm:
                continue
            nums: list[int] = []
            for td in tds[1:6]:
                v = _parse_int(td.get_text())
                if v is not None:
                    nums.append(v)
            runs = nums[0] if nums else None
            balls = nums[1] if len(nums) > 1 else None
            pos += 1
            rows_out.append({"player": nm, "position": pos, "runs": runs, "balls": balls})
        if rows_out:
            innings.append({"team": team_guess, "rows": rows_out})
    return innings


def _bowling_wrap(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for head in soup.find_all(string=re.compile(r"Bowling", re.I)):
        p = head.parent
        if not p:
            continue
        block = p.find_parent("div") or p
        table = block.find_next("table")
        if not table:
            continue
        team_guess = ""
        prev = table.find_previous(string=re.compile(r"Innings", re.I))
        if prev and prev.parent:
            team_guess = clean_player_name(prev.parent.get_text(" ", strip=True)).split("Innings")[0].strip()
        rows_out: list[dict[str, Any]] = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            link = tds[0].find("a", href=True)
            if not link:
                continue
            nm = clean_player_name(link.get_text())
            if not nm:
                continue
            overs = _parse_float(tds[1].get_text())
            maidens = _parse_int(tds[2].get_text()) or 0
            runs = _parse_int(tds[3].get_text())
            wk = _parse_int(tds[4].get_text()) if len(tds) > 4 else None
            rows_out.append(
                {
                    "player": nm,
                    "overs": overs,
                    "maidens": maidens,
                    "runs": runs,
                    "wickets": wk,
                }
            )
        if rows_out:
            out.append({"team": team_guess, "rows": rows_out})
    return out


def _infer_winner(teams: list[str], meta: dict[str, Any]) -> Optional[str]:
    margin = (meta.get("margin") or "").lower()
    for t in teams:
        if t and t.lower() in margin and "won" in margin:
            return t
    return None


def _fallback_batting_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    innings: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        rows_out: list[dict[str, Any]] = []
        pos = 0
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            link = tds[0].find("a", href=True)
            if not link:
                continue
            href = link.get("href") or ""
            if "/profiles/" not in href and "/player/" not in href:
                continue
            nm = clean_player_name(link.get_text())
            if not nm or nm.lower() in ("extras", "total"):
                continue
            nums: list[int] = []
            for td in tds[1:6]:
                v = _parse_int(td.get_text())
                if v is not None:
                    nums.append(v)
            runs = nums[0] if nums else None
            balls = nums[1] if len(nums) > 1 else None
            pos += 1
            rows_out.append({"player": nm, "position": pos, "runs": runs, "balls": balls})
        if len(rows_out) >= 3:
            innings.append({"team": "", "rows": rows_out})
    return innings[:4]


def _fallback_bowling_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        rows_out: list[dict[str, Any]] = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            link = tds[0].find("a", href=True)
            if not link:
                continue
            nm = clean_player_name(link.get_text())
            if not nm:
                continue
            overs = _parse_float(tds[1].get_text())
            if overs is None:
                continue
            maidens = _parse_int(tds[2].get_text()) or 0
            runs = _parse_int(tds[3].get_text())
            wk = _parse_int(tds[4].get_text()) if len(tds) > 4 else None
            rows_out.append(
                {"player": nm, "overs": overs, "maidens": maidens, "runs": runs, "wickets": wk}
            )
        if len(rows_out) >= 2:
            out.append({"team": "", "rows": rows_out})
    return out[:4]


def parse(html: str, url: str) -> dict[str, Any]:
    try:
        soup = make_soup(html)
    except Exception:  # noqa: BLE001
        return {
            "meta": {**base_meta(url, "cricbuzz")},
            "teams": [],
            "playing_xi": [],
            "batting": [],
            "bowling": [],
        }
    teams = _teams(soup)
    meta = _header_meta(soup)
    meta.update(base_meta(url, "cricbuzz"))

    playing_xi = _playing_xi(soup, teams)
    batting = _batting_innings(soup, teams)
    bowling = _bowling_wrap(soup)

    if not batting:
        batting = _fallback_batting_tables(soup)
    if not bowling:
        bowling = _fallback_bowling_tables(soup)

    if not playing_xi and batting:
        seen: set[str] = set()
        for inn in batting:
            plist: list[str] = []
            for row in inn.get("rows") or []:
                nm = row.get("player")
                if nm and nm not in seen:
                    seen.add(str(nm))
                    plist.append(str(nm))
                if len(plist) >= 11:
                    break
            if plist:
                playing_xi.append({"team": inn.get("team") or "Team", "players": plist[:11]})

    meta["winner"] = _infer_winner(teams, meta)
    meta["batting_first"] = batting[0]["team"] if batting else None

    return {
        "meta": meta,
        "teams": teams,
        "playing_xi": playing_xi,
        "batting": batting,
        "bowling": bowling,
    }
