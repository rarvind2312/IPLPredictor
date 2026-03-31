"""
Automatic scorecard ingestion: detect source from URL, fetch safely, parse with fallbacks.

Never raises to callers — returns a payload with `ingestion.errors` / `completeness` instead.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

import config
from parsers._common import detect_source
from parsers.schema import attach_ingestion_meta, empty_payload, enrich_payload

from . import cricbuzz_parser, cricinfo_parser, ipl_parser


def fetch_html_safe(
    url: str,
    *,
    session: Optional[requests.Session] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Returns (html, error_message). HTML is None on failure."""
    try:
        sess = session or requests.Session()
        sess.headers.update({"User-Agent": config.USER_AGENT})
        r = sess.get(url.strip(), timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text, None
    except Exception as exc:  # noqa: BLE001
        return None, f"fetch_failed: {type(exc).__name__}: {exc}"


def _normalize_url(url: str) -> tuple[str, Optional[str]]:
    u = (url or "").strip()
    if not u:
        return "", "empty_url"
    try:
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            return u, "url_must_be_http_or_https"
        if not parsed.netloc:
            return u, "url_missing_host"
    except Exception:  # noqa: BLE001
        return u, "url_parse_error"
    return u, None


def _run_parser(
    fn: Callable[[str, str], dict[str, Any]], html: str, url: str
) -> Tuple[dict[str, Any], List[str]]:
    errs: list[str] = []
    try:
        data = fn(html, url)
        if not isinstance(data, dict):
            errs.append("parser_returned_non_dict")
            return empty_payload(url, detect_source(url)), errs
        return data, errs
    except Exception as exc:  # noqa: BLE001
        errs.append(f"parse_exception: {type(exc).__name__}: {exc}")
        return empty_payload(url, detect_source(url)), errs


def _parser_fn(source: str) -> Optional[Callable[[str, str], dict[str, Any]]]:
    """Resolved at call time so tests can patch submodule `parse` functions."""
    if source == "cricbuzz":
        return cricbuzz_parser.parse
    if source == "cricinfo":
        return cricinfo_parser.parse
    if source == "ipl":
        return ipl_parser.parse
    return None


def parse_scorecard(
    url: str,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """
    Fetch and parse a scorecard URL. Always returns a dict suitable for `db.insert_parsed_match`
    (when `ingestion.has_storable_content` is True).
    """
    warnings: list[str] = []
    parse_errors: list[str] = []

    norm, verr = _normalize_url(url)
    if verr == "empty_url":
        pl = empty_payload("", "unknown")
        pl["meta"]["url"] = ""
        return attach_ingestion_meta(
            pl,
            source="unknown",
            fetch_ok=False,
            fetch_error="empty_url",
            parse_errors=["empty_url"],
            warnings=warnings,
        )

    src = detect_source(norm)
    if src == "unknown":
        pl = empty_payload(norm, "unknown")
        return attach_ingestion_meta(
            pl,
            source="unknown",
            fetch_ok=False,
            fetch_error=None,
            parse_errors=["unsupported_domain"],
            warnings=warnings,
        )

    html, ferr = fetch_html_safe(norm, session=session)
    if ferr or html is None:
        pl = empty_payload(norm, src)
        return attach_ingestion_meta(
            pl,
            source=src,
            fetch_ok=False,
            fetch_error=ferr,
            parse_errors=[ferr or "fetch_failed"],
            warnings=warnings,
        )

    parser_fn = _parser_fn(src)
    if not parser_fn:
        pl = empty_payload(norm, src)
        parse_errors.append("no_parser_registered")
        pl = enrich_payload(pl, html, src, warnings)
        return attach_ingestion_meta(
            pl,
            source=src,
            fetch_ok=True,
            fetch_error=None,
            parse_errors=parse_errors,
            warnings=warnings,
        )

    payload, perrs = _run_parser(parser_fn, html, norm)
    parse_errors.extend(perrs)

    # Ensure base keys exist
    payload.setdefault("meta", {})
    payload["meta"].setdefault("url", norm)
    payload["meta"].setdefault("source", src)
    payload.setdefault("teams", [])
    payload.setdefault("playing_xi", [])
    payload.setdefault("batting", [])
    payload.setdefault("bowling", [])

    enrich_payload(payload, html, src, warnings)

    return attach_ingestion_meta(
        payload,
        source=src,
        fetch_ok=True,
        fetch_error=None,
        parse_errors=parse_errors,
        warnings=warnings,
    )
