"""
Fetch and parse official IPL squad pages only: https://www.iplt20.com/teams/{slug}/squad

Produces structured IplSquadMember rows (name + role_bucket). Does not concatenate
internal predictor roles like 'all' into the displayed name.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup, Tag

import config
import ipl_teams
import learner
from ipl_squad import (
    ALL_ROUNDER,
    BATTER,
    BOWLER,
    IplSquadMember,
    WK_BATTER,
    build_ipl_squad_member,
    normalize_role_bucket_label,
    overseas_from_api_record,
    role_bucket_from_api_record,
    split_embedded_role_from_name,
    validate_clean_name,
)

logger = logging.getLogger(__name__)

SQUAD_URL_TEMPLATE = "https://www.iplt20.com/teams/{slug}/squad"

# --- Navigation / footer noise (lowercase) for raw text before structuring ---
_BLOCKED_TOKENS: frozenset[str] = frozenset(
    {
        "about",
        "guidelines",
        "contact",
        "terms",
        "privacy",
        "policy",
        "cookies",
        "cookie",
        "home",
        "menu",
        "search",
        "follow",
        "subscribe",
        "newsletter",
        "twitter",
        "facebook",
        "instagram",
        "youtube",
        "linkedin",
        "shop",
        "tickets",
        "fantasy",
        "watch",
        "more",
        "read",
        "next",
        "prev",
        "previous",
        "copyright",
        "rights",
        "reserved",
        "sitemap",
        "accessibility",
        "careers",
        "advertise",
        "advertising",
        "partners",
        "legal",
        "disclaimer",
        "help",
        "support",
        "faq",
        "faqs",
        "login",
        "sign",
        "register",
        "account",
        "settings",
        "download",
        "app",
        "iplt20",
        "ipl",
        "official",
        "website",
        "venues",
        "schedule",
        "fixtures",
        "results",
        "standings",
        "points",
        "table",
        "videos",
        "photos",
        "archive",
        "teams",
        "season",
        "auction",
        "draft",
        "wpl",
        "accept",
        "reject",
        "close",
        "agree",
        "ok",
        "yes",
        "no",
        "view",
        "all",
        "see",
        "click",
        "here",
        "learn",
        "india",
        "indian",
        "premier",
        "league",
        "board",
        "bcci",
        "tata",
        "sponsor",
    }
)

_BLOCKED_EXACT_PHRASES: frozenset[str] = frozenset(
    {
        "what are cookies",
        "privacy policy",
        "terms of use",
        "terms & conditions",
        "terms and conditions",
        "code of conduct",
        "anti corruption",
        "match playing conditions",
    }
)

_EXCLUDE_NAME_PATTERN = re.compile(
    r"^(batters?|bowlers?|all[-\s]?rounders?|squad|fixtures|results|videos|news|archive|"
    r"home|teams|season|schedule|captain|coach|owner|venue|official|accept|cookies|"
    r"what are cookies|iplt20|indian premier league)\b",
    re.I,
)


@dataclass
class SquadParseDebug:
    source: str = "iplt20"
    raw_candidate_count: int = 0
    cleaned_count: int = 0
    rejected_sample: list[str] = field(default_factory=list)
    methods_used: list[str] = field(default_factory=list)
    raw_extracted_preview: list[str] = field(default_factory=list)
    foreign_player_icon_hits: int = 0

    def log_summary(self) -> None:
        logger.debug(
            "ipl_squad_parse source=%s raw=%d cleaned=%d methods=%s preview=%s rejected=%s",
            self.source,
            self.raw_candidate_count,
            self.cleaned_count,
            self.methods_used,
            self.raw_extracted_preview[:8],
            self.rejected_sample[:12],
        )


RawRow = tuple[str, str, Optional[dict[str, Any]]]  # name_raw, role_bucket_hint, api_obj


def _pre_filter_raw_name(text: str) -> Optional[str]:
    """Fast reject before structured parse (nav/footer)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) < 3 or len(t) > 80:
        return None
    low = t.lower()
    if low in _BLOCKED_EXACT_PHRASES:
        return None
    if low in _BLOCKED_TOKENS:
        return None
    tokens = low.split()
    if tokens and all(tok in _BLOCKED_TOKENS for tok in tokens):
        return None
    if t.isupper() and len(t) <= 22 and " " not in t:
        return None
    if _EXCLUDE_NAME_PATTERN.search(t):
        return None
    if t.isdigit():
        return None
    return t


def _section_heading_to_bucket(section_lower: str) -> Optional[str]:
    s = section_lower.strip()
    if re.search(r"wicket[-\s]?keeper|wk[-\s]?bat", s, re.I):
        return WK_BATTER
    if re.search(r"^batters?$", s, re.I) and "wk" not in s:
        return BATTER
    if "all" in s and "round" in s:
        return ALL_ROUNDER
    if "bowl" in s:
        return BOWLER
    return None


def _player_record_name(obj: dict[str, Any]) -> Optional[str]:
    n = obj.get("playerName") or obj.get("fullName")
    if isinstance(n, str) and n.strip():
        return n.strip()
    return None


def _player_record_strength(obj: dict[str, Any]) -> int:
    score = 0
    for k in (
        "jerseyNo",
        "jerseyNumber",
        "jersey",
        "playerId",
        "playerImage",
        "imageUrl",
        "headshot",
        "nationality",
        "country",
        "battingStyle",
        "bowlingStyle",
        "playingRole",
        "playerSkill",
        "teamId",
        "teamName",
    ):
        v = obj.get(k)
        if v is None or v == "":
            continue
        if isinstance(v, (int, float)) and v != 0:
            score += 2
        elif isinstance(v, str) and len(v.strip()) > 1:
            score += 1
    return score


def _looks_like_squad_player_list(items: list[Any]) -> bool:
    if len(items) < 5 or len(items) > 40:
        return False
    if not all(isinstance(x, dict) for x in items):
        return False
    named = 0
    strong = 0
    for it in items:
        assert isinstance(it, dict)
        n = _player_record_name(it)
        if not n or len(n) < 3:
            continue
        named += 1
        st = _player_record_strength(it)
        if st >= 2:
            strong += 1
        elif " " in n.strip() and len(n) >= 8:
            strong += 1
    if strong >= max(4, int(len(items) * 0.82)):
        return True
    return named >= max(5, int(len(items) * 0.7)) and strong >= max(4, int(len(items) * 0.45))


def _collect_squad_like_lists(obj: Any, found: list[list[dict[str, Any]]]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_squad_like_lists(v, found)
    elif isinstance(obj, list) and obj:
        if all(isinstance(x, dict) for x in obj) and _looks_like_squad_player_list(obj):
            found.append(obj)  # type: ignore[arg-type]
        else:
            for v in obj:
                _collect_squad_like_lists(v, found)


def _strict_nav_player_dicts(obj: Any, acc: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        name = _player_record_name(obj)
        if name and _player_record_strength(obj) >= 3:
            acc.append(obj)
        for v in obj.values():
            _strict_nav_player_dicts(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _strict_nav_player_dicts(v, acc)


def _parse_next_data_raw(html: str) -> list[RawRow]:
    m = re.search(
        r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    try:
        root = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    lists: list[list[dict[str, Any]]] = []
    _collect_squad_like_lists(root, lists)

    out: list[RawRow] = []
    seen: set[str] = set()

    for squad_list in lists:
        for obj in squad_list:
            name = _player_record_name(obj)
            if not name:
                continue
            pf = _pre_filter_raw_name(name)
            if not pf:
                continue
            bucket = role_bucket_from_api_record(obj)
            key = pf.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((pf, bucket, obj))

    if len(out) >= 8:
        return out

    acc: list[dict[str, Any]] = []
    _strict_nav_player_dicts(root, acc)
    for obj in acc:
        name = _player_record_name(obj)
        if not name:
            continue
        pf = _pre_filter_raw_name(name)
        if not pf:
            continue
        key = pf.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((pf, role_bucket_from_api_record(obj), obj))

    return out


_CHROME_ANCESTOR_NAMES = frozenset({"footer", "nav", "header", "aside"})


def _tag_under_chrome(tag: Tag) -> bool:
    for p in tag.parents:
        if not isinstance(p, Tag):
            continue
        if p.name and p.name.lower() in _CHROME_ANCESTOR_NAMES:
            return True
        role = (p.get("role") or "").lower()
        if role in ("navigation", "contentinfo", "banner"):
            return True
        classes = p.get("class")
        if isinstance(classes, list):
            cl = " ".join(classes).lower()
        else:
            cl = str(classes or "").lower()
        if any(
            x in cl
            for x in (
                "footer",
                "site-footer",
                "page-footer",
                "navbar",
                "nav-bar",
                "navigation",
                "sidebar",
                "cookie",
                "consent",
                "mega-menu",
                "site-header",
            )
        ):
            return True
    return False


def _main_or_body_scope(soup: BeautifulSoup) -> Tag:
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if isinstance(main, Tag):
        return main
    body = soup.find("body")
    if isinstance(body, Tag):
        return body
    html = soup.find("html")
    if isinstance(html, Tag):
        return html
    return soup


def _heading_inside_player_card(tag: Tag) -> bool:
    for p in tag.parents:
        if not isinstance(p, Tag):
            continue
        classes = p.get("class")
        if isinstance(classes, list):
            cl = " ".join(classes).lower()
        else:
            cl = str(classes or "").lower()
        if any(
            s in cl
            for s in (
                "player-card",
                "playercard",
                "squad-player",
                "team-player",
                "member-card",
                "player-tile",
                "player-item",
                "squad-member",
            )
        ):
            return True
    return False


def _parse_headings_raw(
    scope: Tag,
    *,
    franchise_lower: set[str],
    only_player_cards: bool,
) -> list[RawRow]:
    current_bucket = BATTER
    out: list[RawRow] = []
    seen: set[str] = set()

    for tag in scope.find_all(["h2", "h3", "h4", "h5"]):
        if not isinstance(tag, Tag):
            continue
        if _tag_under_chrome(tag):
            continue
        if only_player_cards and not _heading_inside_player_card(tag):
            continue
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        low = text.lower()
        if low in franchise_lower:
            continue
        sec = _section_heading_to_bucket(low)
        if sec is not None:
            current_bucket = sec
            continue
        if len(text) > 70:
            continue
        if low in ("squad", "team", "players", "overview", "coaching", "support", "staff"):
            continue
        pf = _pre_filter_raw_name(text)
        if not pf:
            continue
        key = pf.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((pf, current_bucket, None))
    return out


def _parse_player_links_raw(scope: Tag) -> list[RawRow]:
    out: list[RawRow] = []
    seen: set[str] = set()
    for a in scope.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        if _tag_under_chrome(a):
            continue
        href = (a.get("href") or "").lower()
        if "/players/" not in href and "/player/" not in href:
            continue
        text = a.get_text(" ", strip=True)
        pf = _pre_filter_raw_name(text)
        if not pf:
            continue
        key = pf.lower()
        if key in seen:
            continue
        seen.add(key)
        # Role unknown from link alone — default Batter; bucket may refine from page context
        out.append((pf, BATTER, None))
    return out


def clean_candidate_player_names(
    candidates: list[tuple[str, str]],
    *,
    max_rejected_logged: int = 60,
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Back-compat helper: treat second field as ignored role hint; validate names only.
    Returns [(name, 'ipl')...] for tests / legacy callers.
    """
    accepted: list[tuple[str, str]] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw_name, _role in candidates:
        pf = _pre_filter_raw_name(raw_name or "")
        if not pf:
            if len(rejected) < max_rejected_logged:
                rejected.append(f"{str(raw_name)[:64]!r}: prefilter")
            continue
        ok, reason = validate_clean_name(pf)
        if not ok:
            if len(rejected) < max_rejected_logged:
                rejected.append(f"{pf[:64]!r}: {reason}")
            continue
        base, _emb = split_embedded_role_from_name(pf)
        final_name = base or pf
        ok2, r2 = validate_clean_name(final_name)
        if not ok2:
            if len(rejected) < max_rejected_logged:
                rejected.append(f"{final_name[:64]!r}: {r2}")
            continue
        k = final_name.lower()
        if k in seen:
            continue
        seen.add(k)
        accepted.append((final_name, "ipl"))
    return accepted, rejected


def _apply_overseas_from_foreign_player_icons(soup: BeautifulSoup, members: list[IplSquadMember], debug: SquadParseDebug) -> None:
    """
    IPL squad cards mark overseas players with ``teams-foreign-player-icon.svg`` (or similar src).
    Walk from each icon to the nearest player link and mark that squad member overseas.
    """
    if not members:
        return
    key_to_member: dict[str, IplSquadMember] = {}
    for m in members:
        k = m.name.strip().lower()
        if k:
            key_to_member[k] = m

    hits = 0
    for img in soup.find_all("img", src=True):
        if not isinstance(img, Tag):
            continue
        src = (img.get("src") or "").lower()
        if "teams-foreign-player-icon" not in src and "foreign-player-icon" not in src:
            continue
        cur: Optional[Tag] = img
        matched = False
        for _ in range(24):
            if not isinstance(cur, Tag):
                break
            for a in cur.find_all("a", href=True):
                if not isinstance(a, Tag):
                    continue
                href = (a.get("href") or "").lower()
                if "/players/" not in href and "/player/" not in href:
                    continue
                raw = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
                base, _emb = split_embedded_role_from_name(raw)
                clean = (base or raw).strip()
                ck = clean.lower()
                if ck in key_to_member:
                    key_to_member[ck].overseas = True
                    matched = True
                    break
                for nk, mem in key_to_member.items():
                    if nk in ck or ck in nk:
                        mem.overseas = True
                        matched = True
                        break
                if matched:
                    break
            if matched:
                hits += 1
                break
            cur = cur.parent if hasattr(cur, "parent") else None

    debug.foreign_player_icon_hits = hits


def finalize_structured_squad(
    raw_rows: list[RawRow],
    *,
    debug: Optional[SquadParseDebug] = None,
    max_rejected: int = 80,
) -> list[IplSquadMember]:
    """
    Split embedded role suffixes, validate names, dedupe, build IplSquadMember rows.
    """
    members: list[IplSquadMember] = []
    seen: set[str] = set()
    rejected: list[str] = []

    for name_raw, bucket_hint, api_obj in raw_rows:
        base, embedded_bucket = split_embedded_role_from_name(name_raw)
        candidate = (base or name_raw).strip()
        rb = embedded_bucket or normalize_role_bucket_label(bucket_hint) or BATTER
        if api_obj is not None:
            api_b = role_bucket_from_api_record(api_obj)
            if embedded_bucket is None and api_b:
                rb = api_b
        ok, reason = validate_clean_name(candidate)
        if not ok:
            if len(rejected) < max_rejected:
                rejected.append(f"{name_raw[:72]!r}: {reason}")
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        api_overseas = overseas_from_api_record(api_obj) if api_obj is not None else False
        members.append(
            build_ipl_squad_member(name=candidate, role_bucket=rb, overseas=api_overseas, api_obj=api_obj)
        )

    if debug is not None:
        debug.rejected_sample = rejected
        debug.cleaned_count = len(members)

    logger.debug(
        "ipl_squad_finalize structured_players=%d rejected=%d",
        len(members),
        len(rejected),
    )
    return members


def extract_squad_players_from_html(html: str, source: str = "iplt20") -> tuple[list[IplSquadMember], SquadParseDebug]:
    debug = SquadParseDebug(source=source)
    methods: list[str] = []
    raw_rows: list[RawRow] = []

    nd = _parse_next_data_raw(html)
    if nd:
        methods.append(f"next_data:rows={len(nd)}")
        raw_rows.extend(nd)

    soup = BeautifulSoup(html, "html.parser")
    scope = _main_or_body_scope(soup)

    lk = _parse_player_links_raw(scope)
    if lk:
        methods.append(f"player_links:rows={len(lk)}")
        raw_rows.extend(lk)

    use_card_only = len(nd) + len(lk) >= 10
    hd = _parse_headings_raw(
        scope,
        franchise_lower={lab.lower() for _, lab in ipl_teams.IPL_TEAMS},
        only_player_cards=use_card_only,
    )
    if hd:
        methods.append(f"headings:rows={len(hd)}(player_cards_only={use_card_only})")
        raw_rows.extend(hd)

    debug.raw_candidate_count = len(raw_rows)
    debug.methods_used = methods
    debug.raw_extracted_preview = [f"{r[0]} | {r[1]}" for r in raw_rows[:25]]

    logger.info(
        "ipl_squad raw_extracted count=%d methods=%s preview=%s",
        len(raw_rows),
        methods,
        debug.raw_extracted_preview[:6],
    )

    members = finalize_structured_squad(raw_rows, debug=debug)
    _apply_overseas_from_foreign_player_icons(soup, members, debug)
    if debug.foreign_player_icon_hits:
        methods.append(f"foreign_player_icon:{debug.foreign_player_icon_hits}")
        debug.methods_used = methods
    debug.log_summary()

    logger.info(
        "ipl_squad cleaned_structured count=%d sample=%s",
        len(members),
        [m.to_dict() for m in members[:4]],
    )

    return members, debug


def parse_squad_html(html: str, source: str = "iplt20") -> list[IplSquadMember]:
    players, _dbg = extract_squad_players_from_html(html, source=source)
    return players


def format_squad_text(members: list[IplSquadMember]) -> str:
    """Serialize for textarea: Name | RoleBucket [| overseas]."""
    lines: list[str] = []
    for m in members:
        line = f"{m.name} | {m.role_bucket}"
        if m.overseas:
            line += " | overseas"
        lines.append(line)
    return "\n".join(lines)


def fetch_squad_for_slug(
    slug: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 25.0,
) -> tuple[list[IplSquadMember], Optional[str], SquadParseDebug]:
    empty_debug = SquadParseDebug(source=f"iplt20:{slug}")

    if slug not in ipl_teams.TEAM_SLUGS:
        return [], f"Unknown team slug: {slug!r}", empty_debug

    url = SQUAD_URL_TEMPLATE.format(slug=slug)
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": config.USER_AGENT})
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        html = r.text
    except Exception as exc:  # noqa: BLE001
        return [], f"Network error loading squad: {type(exc).__name__}: {exc}", empty_debug

    if not html or len(html) < 500:
        return [], "Empty or very short response from IPL server.", empty_debug

    players, dbg = extract_squad_players_from_html(html, source=f"iplt20:{slug}")

    team_lab = ipl_teams.franchise_label_for_storage(ipl_teams.label_for_slug(slug)) or ipl_teams.label_for_slug(
        slug
    )
    team_ck = ipl_teams.canonical_team_key_for_franchise(team_lab)
    for m in players:
        m.team_name = team_lab
        m.canonical_team_key = team_ck
        m.canonical_player_key = learner.normalize_player_key(m.name)

    if not players:
        return [], (
            "Could not parse any players from the squad page. "
            "The site layout may have changed — enter the squad manually."
        ), dbg

    logger.info(
        "squad_fetch ok slug=%s raw_rows=%d structured=%d methods=%s",
        slug,
        dbg.raw_candidate_count,
        len(players),
        dbg.methods_used,
    )

    if len(players) < 12:
        logger.warning(
            "squad_parse few_players slug=%s count=%d (expected ~25)",
            slug,
            len(players),
        )

    return players, None, dbg
