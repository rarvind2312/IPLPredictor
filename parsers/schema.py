"""Normalized scorecard payload shape, enrichment, and ingestion metadata."""

from __future__ import annotations

import re
from typing import Any, Optional

from parsers._common import base_meta, clean_player_name


def empty_payload(url: str, source: str) -> dict[str, Any]:
    return {
        "meta": dict(base_meta(url, source)),
        "teams": [],
        "playing_xi": [],
        "batting": [],
        "bowling": [],
        "batting_order": [],
        "bowlers_used": [],
    }


def _has_text(meta: dict[str, Any], key: str) -> bool:
    v = meta.get(key)
    return isinstance(v, str) and bool(v.strip())


def scorecard_core_nonempty(payload: dict[str, Any]) -> bool:
    """
    True if we have at least one of: playing XI names, batting order, or bowlers-used lists
    (after enrichment / derivation).
    """
    xi_any = any(len(s.get("players") or []) >= 1 for s in (payload.get("playing_xi") or []))
    bo = payload.get("batting_order") or []
    bu = payload.get("bowlers_used") or []
    bo_any = any(len(s.get("order") or []) > 0 for s in bo)
    bu_any = any(len(s.get("bowlers") or []) > 0 for s in bu)
    return bool(xi_any or bo_any or bu_any)


def scorecard_core_empty(payload: dict[str, Any]) -> bool:
    return not scorecard_core_nonempty(payload)


def derive_batting_order(batting: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per innings: ordered player names by batting position."""
    out: list[dict[str, Any]] = []
    for inn in batting or []:
        team = inn.get("team") or ""
        rows = list(inn.get("rows") or [])
        rows.sort(key=lambda r: (r.get("position") is None, r.get("position") or 999))
        order = [r.get("player") for r in rows if r.get("player")]
        out.append({"team": team, "order": order})
    return out


def derive_bowlers_used(bowling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per bowling innings: distinct bowler names (usage list)."""
    out: list[dict[str, Any]] = []
    for inn in bowling or []:
        team = inn.get("team") or ""
        names: list[str] = []
        seen: set[str] = set()
        for row in inn.get("rows") or []:
            p = row.get("player")
            if not p or not str(p).strip():
                continue
            s = str(p).strip()
            if s.lower() not in seen:
                seen.add(s.lower())
                names.append(s)
        out.append({"team": team, "bowlers": names})
    return out


def fallback_teams_from_html(html: str) -> list[str]:
    """Last-resort team names from <title> or og:title-like snippets."""
    if not html:
        return []
    m = re.search(r"<title[^>]*>([^<]{0,200})</title>", html, re.I | re.S)
    if m:
        title = clean_player_name(re.sub(r"\s+", " ", m.group(1)))
        for pattern in (
            r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[-,|]|$)",
            r"^(.+?)\s+vs\s+(.+?)(?:\s*[-,|]|$)",
        ):
            mm = re.match(pattern, title, flags=re.I)
            if mm:
                a, b = clean_player_name(mm.group(1)), clean_player_name(mm.group(2))
                if len(a) > 1 and len(b) > 1:
                    return [a, b]
    m2 = re.search(r'property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', html, re.I)
    if m2:
        title = clean_player_name(m2.group(1))
        mm = re.match(r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[-,|]|$)", title, flags=re.I)
        if mm:
            return [clean_player_name(mm.group(1)), clean_player_name(mm.group(2))]
    return []


def fallback_venue_date_from_html(html: str, meta: dict[str, Any]) -> None:
    """Mutates meta in place when venue/date missing."""
    if not html:
        return
    if not _has_text(meta, "venue"):
        mv = re.search(
            r'(?:Venue|Ground)[:\s]+([^<"\n]{5,120})',
            html,
            re.I,
        )
        if mv:
            meta["venue"] = clean_player_name(mv.group(1))
    if not _has_text(meta, "date"):
        md = re.search(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
            html,
            re.I,
        )
        if md:
            meta["date"] = md.group(1).strip()
        else:
            md2 = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", html)
            if md2:
                meta["date"] = md2.group(1)


def fallback_result_from_html(html: str, meta: dict[str, Any]) -> None:
    if not html or _has_text(meta, "margin"):
        return
    m = re.search(
        r"([^<\n]{10,180}?(?:won by|won the match|Match tied|No result)[^<\n]{0,120})",
        html,
        re.I,
    )
    if m:
        meta["margin"] = clean_player_name(m.group(1))


def infer_playing_xi_from_batting(
    batting: list[dict[str, Any]],
    teams: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Returns (playing_xi list, inferred flag)."""
    if not batting:
        return [], False
    out: list[dict[str, Any]] = []
    for inn in batting:
        seen: set[str] = set()
        plist: list[str] = []
        rows = sorted(
            inn.get("rows") or [],
            key=lambda r: (r.get("position") is None, r.get("position") or 999),
        )
        for row in rows:
            nm = row.get("player")
            if nm and str(nm).strip():
                s = str(nm).strip()
                if s.lower() not in seen:
                    seen.add(s.lower())
                    plist.append(s)
            if len(plist) >= 11:
                break
        if plist:
            team = inn.get("team") or ""
            if not team and teams:
                team = teams[0] if len(out) == 0 else (teams[1] if len(teams) > 1 else teams[0])
            out.append({"team": team or "Team", "players": plist[:11]})
    return out, True


def infer_winner_generic(teams: list[str], meta: dict[str, Any]) -> None:
    margin = (meta.get("margin") or "").lower()
    if meta.get("winner") or not margin:
        return
    for t in teams:
        if t and t.lower() in margin and ("won" in margin or "tied" in margin):
            meta["winner"] = t
            return


def xi_source_flag(playing_xi: list[dict[str, Any]], inferred: bool) -> str:
    if not playing_xi:
        return "none"
    return "inferred_batting" if inferred else "explicit"


def enrich_payload(
    payload: dict[str, Any],
    html: str,
    source: str,
    warnings: list[str],
) -> dict[str, Any]:
    """
    Apply cross-source fallbacks and derived sections. Mutates and returns payload.
    """
    meta = payload.get("meta") or {}
    payload["meta"] = meta
    meta.setdefault("url", "")
    meta.setdefault("source", source)

    original_xi = [dict(s) for s in (payload.get("playing_xi") or [])]
    teams: list[str] = list(payload.get("teams") or [])

    # IPL official pages: never recover teams/venue/date/margin from generic page text —
    # those produce title abbreviations, CSS junk, and false positives.
    ipl_strict = source == "ipl" or (meta.get("source") == "ipl")

    if len(teams) < 2 and not ipl_strict:
        fb = fallback_teams_from_html(html)
        if len(fb) >= 2:
            teams = fb
            payload["teams"] = teams
            warnings.append("Team names recovered via HTML title fallback.")

    if not ipl_strict:
        fallback_venue_date_from_html(html, meta)
        fallback_result_from_html(html, meta)
    infer_winner_generic(teams, meta)

    playing_xi = list(payload.get("playing_xi") or [])
    had_explicit_xi = any(len((s.get("players") or [])) >= 1 for s in original_xi)
    inferred = False
    if not playing_xi or all(len((s.get("players") or [])) < 11 for s in playing_xi):
        xi2, _ = infer_playing_xi_from_batting(payload.get("batting") or [], teams)
        if xi2:
            inferred = True
            if not playing_xi:
                playing_xi = xi2
                payload["playing_xi"] = playing_xi
                warnings.append("Playing XI inferred from batting card (partial or missing block).")
            else:
                for i, block in enumerate(xi2):
                    if i >= len(playing_xi):
                        playing_xi.append(block)
                    elif len((playing_xi[i].get("players") or [])) < len(block.get("players") or []):
                        playing_xi[i] = block
                payload["playing_xi"] = playing_xi
                warnings.append("Playing XI augmented from batting order.")

    batting = payload.get("batting") or []
    payload["batting_order"] = derive_batting_order(batting)
    payload["bowlers_used"] = derive_bowlers_used(payload.get("bowling") or [])

    if not meta.get("batting_first") and batting:
        first_team = batting[0].get("team")
        if first_team:
            meta["batting_first"] = first_team

    payload["meta"] = meta
    payload.setdefault("ingestion_hints", {})
    if had_explicit_xi and any(len(s.get("players") or []) >= 11 for s in original_xi):
        payload["ingestion_hints"]["xi_source"] = "explicit"
    elif inferred:
        payload["ingestion_hints"]["xi_source"] = "inferred_batting"
    else:
        payload["ingestion_hints"]["xi_source"] = xi_source_flag(playing_xi, False)
    return payload


def compute_completeness(payload: dict[str, Any]) -> dict[str, bool]:
    meta = payload.get("meta") or {}
    teams = payload.get("teams") or []
    playing_xi = payload.get("playing_xi") or []
    batting = payload.get("batting") or []
    bowling = payload.get("bowling") or []
    bo = payload.get("batting_order") or []
    bu = payload.get("bowlers_used") or []

    xi_ok = any(len((s.get("players") or [])) >= 11 for s in playing_xi) or any(
        len((s.get("players") or [])) >= 1 for s in playing_xi
    )
    bat_ord_ok = any(len((s.get("order") or [])) > 0 for s in bo)
    bowlers_ok = any(len((s.get("bowlers") or [])) > 0 for s in bu)

    margin = (meta.get("margin") or "").strip().lower()
    result_ok = bool(meta.get("winner")) or ("won" in margin or "tied" in margin or "no result" in margin)

    return {
        "team_names": len(teams) >= 2,
        "venue": _has_text(meta, "venue"),
        "date": _has_text(meta, "date"),
        "playing_xi": xi_ok,
        "batting_order": bat_ord_ok,
        "bowlers_used": bowlers_ok,
        "result": result_ok,
    }


def has_storable_content(payload: dict[str, Any]) -> bool:
    """True if we should persist a row (usable scorecard slice + two teams)."""
    teams = payload.get("teams") or []
    if len(teams) < 2:
        return False
    if scorecard_core_empty(payload):
        return False
    return True


def attach_ingestion_meta(
    payload: dict[str, Any],
    *,
    source: str,
    fetch_ok: bool,
    fetch_error: Optional[str],
    parse_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    comp = compute_completeness(payload)
    parse_ok = len(parse_errors) == 0
    ipl_missing_scorecard = (
        source == "ipl"
        and fetch_ok
        and scorecard_core_empty(payload)
        and not parse_errors
    )
    if ipl_missing_scorecard:
        parse_ok = False
        warnings.append(
            "Official IPL match page fetched successfully but scorecard data was not found in parseable structured form"
        )

    ing: dict[str, Any] = {
        "source": source,
        "fetch_ok": fetch_ok,
        "fetch_error": fetch_error,
        "parse_ok": parse_ok,
        "errors": list(parse_errors),
        "warnings": list(warnings),
        "completeness": comp,
        "has_storable_content": has_storable_content(payload),
        "ipl_scorecard_missing": bool(source == "ipl" and fetch_ok and scorecard_core_empty(payload)),
    }
    payload["ingestion"] = ing
    return payload
