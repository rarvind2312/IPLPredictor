"""Shared Streamlit DB bootstrap (used by main app and multipage admin)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
import db


@st.cache_resource(show_spinner=False)
def ensure_db_schema_initialized(db_signature: tuple[str, int, int]) -> None:
    """
    Run ``db.init_schema()`` once per process for a given on-disk DB identity.

    Signature is ``(resolved path, mtime_ns, size_bytes)`` so wipes/recreates pick up a new key;
    stable DBs skip repeated migration work on Streamlit reruns.
    """
    db.init_schema()


def db_init_signature() -> tuple[str, int, int]:
    bp = Path(config.DB_PATH).resolve()
    if not bp.is_file():
        return (str(bp), 0, 0)
    st_info = bp.stat()
    mtime_ns = int(getattr(st_info, "st_mtime_ns", int(st_info.st_mtime * 1e9)))
    return (str(bp), mtime_ns, int(st_info.st_size))
