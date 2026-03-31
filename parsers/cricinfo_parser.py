"""ESPNcricinfo scorecard HTML parser."""

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag

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


def _team_names(soup: BeautifulSoup) -> list[str]:
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string
    m = re.search(
        r"^\s*([^,|]+?)\s+vs\.?\s+([^,|]+?)(?:\s*[-,|]|$)",
        title,
        flags=re.I,
    )
    if m:
        return [clean_player_name(m.group(1)), clean_player_name(m.group(2))]
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        content = og["content"]
        m2 = re.search(r"^(.+?)\s+vs\s+(.+?)(?:\s*[-,]|$)", content, flags=re.I)
        if m2:
            return [clean_player_name(m2.group(1)), clean_player_name(m2.group(2))]
    return []


def _match_result_meta(soup: BeautifulSoup) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag in soup.find_all(string=re.compile(r"won by|won the match|Match tied|No result", re.I)):
        parent = getattr(tag, "parent", None)
        if parent:
            t = clean_player_name(parent.get_text())
            if len(t) < 200:
                out["margin"] = t
            break
    # Toss
    for tag in soup.find_all(string=re.compile(r"Toss", re.I)):
        p = tag.parent
        if p:
            line = clean_player_name(p.get_text())
            if "toss" in line.lower() and len(line) < 220:
                out["toss_line"] = line
                mw = re.search(
                    r"toss[:\s]+(.+?)(?:\s+elected|\s+chose|\s+opted|$)",
                    line,
                    flags=re.I,
                )
                if mw:
                    out["toss_winner"] = clean_player_name(mw.group(1))
                if re.search(r"bat(?:t)?ing first", line, re.I):
                    out["toss_decision"] = "bat"
                elif re.search(r"bowl(?:ing)? first", line, re.I):
                    out["toss_decision"] = "bowl"
    # Venue / date from header chips
    for li in soup.find_all(["span", "div", "li"]):
        tx = li.get_text(" ", strip=True)
        if re.search(r"^Venue[:\s]", tx, re.I):
            out["venue"] = clean_player_name(re.sub(r"^Venue[:\s]+", "", tx, flags=re.I))
        if re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", tx):
            if "match" in tx.lower() or len(tx) < 40:
                out["date"] = tx
    return out


def _find_playing_xi_blocks(soup: BeautifulSoup) -> list[tuple[str, list[str]]]:
    """Return list of (team_hint, players)."""
    blocks: list[tuple[str, list[str]]] = []
    for hdr in soup.find_all(["h2", "h3", "h4", "span", "div"]):
        t = hdr.get_text(strip=True).lower()
        if "playing xi" not in t and "playing 11" not in t:
            continue
        team_hint = ""
        hm = re.search(r"([A-Za-z0-9\s'.-]+)\s*-\s*playing", hdr.get_text(), re.I)
        if hm:
            team_hint = clean_player_name(hm.group(1))
        players: list[str] = []
        sib = hdr
        for _ in range(40):
            sib = sib.find_next_sibling()
            if sib is None:
                break
            if isinstance(sib, Tag) and sib.name in ("h2", "h3", "h4") and "innings" in sib.get_text(
                "", strip=True
            ).lower():
                break
            for a in sib.find_all("a", href=True):
                href = a.get("href") or ""
                if "/player/" in href or "/ci/content/player/" in href:
                    nm = clean_player_name(a.get_text())
                    if nm and nm not in players and len(nm) < 80:
                        players.append(nm)
            if len(players) >= 22:
                break
        if players:
            blocks.append((team_hint, players[:11]))
    return blocks


def _innings_headers(soup: BeautifulSoup) -> list[tuple[str, Tag]]:
    """Return (innings_label, header_tag) for batting sections."""
    found: list[tuple[str, Tag]] = []
    for tag in soup.find_all(["h2", "h3", "span", "div"]):
        tx = tag.get_text(" ", strip=True)
        low = tx.lower()
        if " innings" not in low and not low.endswith("innings"):
            continue
        if "fall of" in low or "partnership" in low:
            continue
        if "bowling" in low or "batting" == low.lower():
            continue
        label = clean_player_name(tx)
        if label:
            found.append((label, tag))
    return found


def _table_after(tag: Tag) -> Optional[Tag]:
    cur: Optional[Tag] = tag
    for _ in range(60):
        cur = cur.find_next()
        if cur is None:
            return None
        if isinstance(cur, Tag) and cur.name == "table":
            return cur
    return None


def _extract_batting_table(table: Tag, team_guess: str) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    pos = 0
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        first = tds[0]
        link = first.find("a", href=True)
        if not link:
            continue
        href = link.get("href") or ""
        if "/player/" not in href and "/ci/content/player/" not in href:
            continue
        name = clean_player_name(link.get_text())
        if not name:
            continue
        # Typical columns: R, B, 4s, 6s, SR — take first int-like after name column
        nums: list[int] = []
        for td in tds[1:6]:
            v = _parse_int(td.get_text())
            if v is not None:
                nums.append(v)
        runs = nums[0] if len(nums) > 0 else None
        balls = nums[1] if len(nums) > 1 else None
        pos += 1
        rows_out.append(
            {
                "player": name,
                "position": pos,
                "runs": runs,
                "balls": balls,
            }
        )
    return {"team": team_guess, "rows": rows_out}


def _extract_bowling_table(table: Tag, team_guess: str) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        link = tds[0].find("a", href=True)
        if not link:
            continue
        href = link.get("href") or ""
        if "/player/" not in href:
            continue
        name = clean_player_name(link.get_text())
        if not name:
            continue
        overs = _parse_float(tds[1].get_text())
        maidens = _parse_int(tds[2].get_text())
        runs = _parse_int(tds[3].get_text()) if len(tds) > 3 else None
        wk = _parse_int(tds[4].get_text()) if len(tds) > 4 else None
        rows_out.append(
            {
                "player": name,
                "overs": overs,
                "maidens": maidens or 0,
                "runs": runs,
                "wickets": wk,
            }
        )
    return {"team": team_guess, "rows": rows_out}


def _split_team_from_innings(label: str, teams: list[str]) -> str:
    low = label.lower()
    for t in teams:
        if t and t.lower() in low:
            return t
    parts = label.replace(" Innings", "").replace(" innings", "").strip()
    return clean_player_name(parts.split("(")[0])


def _bowling_table_near_batting(soup: BeautifulSoup, bat_table: Tag) -> Optional[Tag]:
    cur: Optional[Tag] = bat_table
    for _ in range(25):
        cur = cur.find_next("table")
        if cur is None:
            return None
        # bowling table has OVERS / O header sometimes
        head = cur.find("th")
        hx = head.get_text(strip=True).lower() if head else ""
        if "over" in hx or hx in ("o", "m", "r", "w", "econ"):
            return cur
    return None


def _infer_winner(teams: list[str], meta: dict[str, Any]) -> Optional[str]:
    margin = (meta.get("margin") or "").lower()
    for t in teams:
        if t and t.lower() in margin and "won" in margin:
            return t
    return None


def _infer_batting_first(teams: list[str], batting: list[dict[str, Any]]) -> Optional[str]:
    if not batting:
        return None
    first = batting[0].get("team")
    if isinstance(first, str) and first:
        return first
    if teams:
        return teams[0]
    return None


def _fallback_batting_scan_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """If innings headers fail, collect any table with multiple /player/ batting rows."""
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
            if "/player/" not in href and "/ci/content/player/" not in href:
                continue
            name = clean_player_name(link.get_text())
            if not name or name.lower() in ("extras", "total"):
                continue
            nums: list[int] = []
            for td in tds[1:6]:
                v = _parse_int(td.get_text())
                if v is not None:
                    nums.append(v)
            runs = nums[0] if nums else None
            balls = nums[1] if len(nums) > 1 else None
            pos += 1
            rows_out.append({"player": name, "position": pos, "runs": runs, "balls": balls})
        if len(rows_out) >= 3:
            innings.append({"team": "", "rows": rows_out})
    return innings[:4]


def _fallback_bowling_scan_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        ths = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        joined = " ".join(ths)
        if "over" not in joined and "o " not in joined:
            continue
        rows_out: list[dict[str, Any]] = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            link = tds[0].find("a", href=True)
            if not link:
                continue
            href = link.get("href") or ""
            if "/player/" not in href:
                continue
            name = clean_player_name(link.get_text())
            if not name:
                continue
            overs = _parse_float(tds[1].get_text())
            if overs is None:
                continue
            maidens = _parse_int(tds[2].get_text())
            runs = _parse_int(tds[3].get_text()) if len(tds) > 3 else None
            wk = _parse_int(tds[4].get_text()) if len(tds) > 4 else None
            rows_out.append(
                {
                    "player": name,
                    "overs": overs,
                    "maidens": maidens or 0,
                    "runs": runs,
                    "wickets": wk,
                }
            )
        if len(rows_out) >= 2:
            out.append({"team": "", "rows": rows_out})
    return out[:4]


def parse(html: str, url: str) -> dict[str, Any]:
    try:
        soup = make_soup(html)
    except Exception:  # noqa: BLE001
        return {
            "meta": {**base_meta(url, "cricinfo")},
            "teams": [],
            "playing_xi": [],
            "batting": [],
            "bowling": [],
        }

    teams = _team_names(soup)
    meta = _match_result_meta(soup)
    meta.update(base_meta(url, "cricinfo"))

    xi_blocks = _find_playing_xi_blocks(soup)
    playing_xi: list[dict[str, Any]] = []
    if len(xi_blocks) >= 1 and teams:
        for i, (hint, players) in enumerate(xi_blocks[:2]):
            team_name = hint or (teams[i] if i < len(teams) else f"Team {i+1}")
            playing_xi.append({"team": team_name, "players": players})
    elif xi_blocks and not teams:
        for i, (hint, players) in enumerate(xi_blocks[:2]):
            playing_xi.append({"team": hint or f"Side {i+1}", "players": players})

    batting: list[dict[str, Any]] = []
    bowling: list[dict[str, Any]] = []
    for label, hdr in _innings_headers(soup):
        team_guess = _split_team_from_innings(label, teams)
        bat_tbl = _table_after(hdr)
        if not bat_tbl:
            continue
        binns = _extract_batting_table(bat_tbl, team_guess)
        if binns["rows"]:
            batting.append(binns)
        bow_tbl = _bowling_table_near_batting(soup, bat_tbl)
        if bow_tbl:
            bowling.append(_extract_bowling_table(bow_tbl, team_guess))

    if not batting:
        batting = _fallback_batting_scan_tables(soup)
    if not bowling:
        bowling = _fallback_bowling_scan_tables(soup)

    # Fallback XI from batting lineups (first 11 distinct)
    if playing_xi == [] and batting:
        seen: set[str] = set()
        for inn in batting:
            plist: list[str] = []
            for row in inn.get("rows") or []:
                nm = row.get("player")
                if nm and nm not in seen:
                    seen.add(nm)
                    plist.append(str(nm))
                if len(plist) >= 11:
                    break
            if plist:
                playing_xi.append({"team": inn.get("team") or "Team", "players": plist[:11]})

    winner = _infer_winner(teams, meta)
    meta["winner"] = winner
    meta["batting_first"] = _infer_batting_first(teams, batting)

    return {
        "meta": meta,
        "teams": teams,
        "playing_xi": playing_xi,
        "batting": batting,
        "bowling": bowling,
    }
