"""
Parse ESPN Cricinfo squad pages and player profiles for IPL metadata curation.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CRICINFO_2026_SQUAD_URLS: tuple[str, ...] = (
    "https://www.espncricinfo.com/series/ipl-2026-1510719/chennai-super-kings-squad-1511148/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/mumbai-indians-squad-1511109/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/punjab-kings-squad-1511082/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/rajasthan-royals-squad-1511089/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/royal-challengers-bengaluru-squad-1511134/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/sunrisers-hyderabad-squad-1511114/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/gujarat-titans-squad-1511094/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/delhi-capitals-squad-1511107/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/kolkata-knight-riders-squad-1511092/series-squads",
    "https://www.espncricinfo.com/series/ipl-2026-1510719/lucknow-super-giants-squad-1511235/series-squads",
)


@dataclass
class PlayerMetadata:
    player_name: str
    batting_hand: str
    bowling_style_raw: str
    bowling_type_bucket: str
    primary_role: str
    source: str = "cricinfo_curated"
    confidence: float = 0.95


def _normalize_key(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"\s+", " ", s)[:80]


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IPLMetadataBot/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=25) as r:  # noqa: S310
        return r.read().decode("utf-8", errors="ignore")


def _strip_tags(html: str) -> str:
    txt = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    txt = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", txt)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = txt.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", txt).strip()


def _extract_squad_block(text: str) -> str:
    m = re.search(r"(?:BATTERS|BATTING)\s+(.*?)(?:Indian Premier League|Full Table|Top Wicket Takers)", text, flags=re.I)
    return (m.group(1).strip() if m else text)


def _normalize_batting_hand(v: str) -> str:
    s = (v or "").strip().lower()
    if "right-hand bat" in s or "right hand bat" in s:
        return "right"
    if "left-hand bat" in s or "left hand bat" in s:
        return "left"
    return "unknown"


def _derive_bowling_type_bucket(v: str) -> str:
    s = (v or "").strip().lower()
    if not s:
        return "unknown"
    if "left-arm orthodox" in s or "left arm orthodox" in s:
        return "left_arm_orthodox"
    if "legbreak" in s or "leg break" in s:
        return "wrist_spin"
    if "offbreak" in s or "off break" in s:
        return "finger_spin"
    if "fast" in s or "medium" in s:
        return "pace"
    return "unknown"


def _infer_primary_role(*, playing_role: str, batting_hand: str, bowling_type_bucket: str) -> str:
    r = (playing_role or "").lower()
    if "wicketkeeper" in r or "keeper" in r:
        return "wk_batter"
    if bowling_type_bucket != "unknown" and batting_hand != "unknown":
        return "all_rounder"
    if bowling_type_bucket != "unknown":
        return "bowler"
    return "batter"


def _iter_player_chunks(squad_text: str) -> list[tuple[str, str, str, str]]:
    """
    Parse text chunks that look like:
    Name ... <role> ... Age:... Batting:<...> Bowling:<...optional...>
    """
    role_pat = (
        r"(Wicketkeeper Batter|Opening Batter|Top order Batter|Middle order Batter|"
        r"Batting Allrounder|Bowling Allrounder|Allrounder|Bowler|Batter)"
    )
    pat = re.compile(
        rf"([A-Z][A-Za-z .'\-]+?)\s+(?:\(c\)|\(vc\)|†|\(c\)\s*|\(vc\)\s*)*"
        rf"{role_pat}\s+Age:[^B]{{0,40}}\s+Batting:([^B]{{1,80}})"
        rf"(?:\s+Bowling:([^A]{{1,120}}))?"
        rf"(?=\s+[A-Z][A-Za-z .'\-]+?\s+(?:\(c\)|\(vc\)|†|\(c\)\s*|\(vc\)\s*)*{role_pat}\s+Age:|$)",
        flags=re.I,
    )
    out: list[tuple[str, str, str, str]] = []
    for m in pat.finditer(squad_text):
        name = str(m.group(1) or "").strip()
        role = str(m.group(2) or "").strip()
        batting = str(m.group(3) or "").strip()
        bowling = str(m.group(4) or "").strip() if m.group(4) else ""
        if name:
            out.append((name, role, batting, bowling))
    return out


def parse_cricinfo_squad_page(url: str) -> list[PlayerMetadata]:
    html = _fetch_text(url)
    text = _strip_tags(html)
    squad_block = _extract_squad_block(text)
    out: list[PlayerMetadata] = []
    seen_keys: set[str] = set()
    for name, role_raw, batting_style, bowling_style in _iter_player_chunks(squad_block):
        pm = PlayerMetadata(
            player_name=name,
            batting_hand=_normalize_batting_hand(batting_style),
            bowling_style_raw=bowling_style or "unknown",
            bowling_type_bucket=_derive_bowling_type_bucket(bowling_style),
            primary_role=_infer_primary_role(
                playing_role=role_raw,
                batting_hand=_normalize_batting_hand(batting_style),
                bowling_type_bucket=_derive_bowling_type_bucket(bowling_style),
            ),
            source="cricinfo_curated",
            confidence=0.95,
        )
        k = _normalize_key(pm.player_name)
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)
        out.append(pm)
    return out


def build_cricinfo_metadata_for_urls(urls: Iterable[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for u in urls:
        for pm in parse_cricinfo_squad_page(str(u)):
            k = _normalize_key(pm.player_name)
            out[k] = asdict(pm)
    return out


def save_cricinfo_metadata_json(output_path: str, urls: Iterable[str]) -> dict[str, dict]:
    data = build_cricinfo_metadata_for_urls(urls)
    p = Path(output_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / output_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


__all__ = [
    "PlayerMetadata",
    "CRICINFO_2026_SQUAD_URLS",
    "parse_cricinfo_squad_page",
    "build_cricinfo_metadata_for_urls",
    "save_cricinfo_metadata_json",
]

