"""Shared helpers for scorecard HTML fetching and cleanup."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config


def fetch_html(url: str, *, session: Optional[requests.Session] = None) -> str:
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": config.USER_AGENT})
    r = sess.get(url, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def clean_player_name(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"\(c\)|\(wk\)|†|\*", "", t, flags=re.I).strip()
    return t


def detect_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "cricbuzz.com" in host:
        return "cricbuzz"
    if "espncricinfo.com" in host or "cricinfo.com" in host:
        return "cricinfo"
    if "iplt20.com" in host:
        return "ipl"
    return "unknown"


def base_meta(url: str, source: str) -> dict:
    return {"url": url.strip(), "source": source}
