"""
Optional profiling / audit hooks.

Enable with env **IPL_AUDIT_PROFILING=true** (and optionally **IPL_PREDICTION_TIMING=true** for logger lines).

**Streamlit:** ``app.py`` records per-rerun startup splits, weather + ``run_prediction`` wall times, tuning-debug
button work, and ``predict_ui_render`` time. Open the **Audit profiling** expander at the bottom of the page.

**Prediction:** ``predictor.run_prediction`` records phase wall times and every ``db.connection()`` ``execute`` /
``executemany`` (rowcount may be -1 for SELECTs until fetched).

Read-only: does not change scores or XI outputs.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator, Optional

import config

_sql_events: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar("audit_sql_events", default=None)
_prediction_phases: ContextVar[Optional[list[tuple[str, float]]]] = ContextVar(
    "audit_prediction_phases", default=None
)

# Thread-local fallback if contextvar not set (e.g. tests)
_tls = threading.local()


def audit_enabled() -> bool:
    return bool(getattr(config, "AUDIT_PROFILING", False))


def sql_capture_active() -> bool:
    if _sql_events.get() is not None:
        return True
    ev = getattr(_tls, "sql_events", None)
    return ev is not None


def _append_sql_event(entry: dict[str, Any]) -> None:
    bucket = _sql_events.get()
    if bucket is None:
        bucket = getattr(_tls, "sql_events", None)
    if bucket is not None:
        bucket.append(entry)


def _tables_from_sql(sql: str) -> list[str]:
    s = re.sub(r"\s+", " ", sql.strip().lower())
    out: list[str] = []
    for m in re.finditer(
        r"\b(from|join)\s+([a-z_][a-z0-9_]*)",
        s,
        re.IGNORECASE,
    ):
        t = m.group(2)
        if t not in ("select", "where", "on", "and", "or", "inner", "left", "outer", "cross"):
            out.append(t)
    return list(dict.fromkeys(out))


def _index_heuristic(sql: str) -> str:
    sl = sql.strip().lower()
    if sl.startswith("pragma"):
        return "pragma"
    if " where " not in sl and sl.startswith("select"):
        return "no_where_clause_full_scan_risk"
    if re.search(r"\bwhere\b.+\b=\s*\?", sl):
        return "likely_point_lookup_or_indexed_filter"
    if " join " in sl:
        return "join_plan_depends_on_indexes"
    return "unknown"


def wrap_sqlite_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Return a proxy that logs ``execute`` / ``executemany`` when SQL capture is active."""

    class _ConnProxy:
        __slots__ = ("_c",)

        def __init__(self, c: sqlite3.Connection) -> None:
            object.__setattr__(self, "_c", c)

        def execute(
            self, sql: str, parameters: Any = ()
        ) -> sqlite3.Cursor:  # type: ignore[override]
            t0 = time.perf_counter()
            cur = self._c.execute(sql, parameters)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if sql_capture_active():
                params_preview = ""
                try:
                    if parameters is not None and not isinstance(parameters, (bytes, str)):
                        params_preview = repr(parameters)[:120]
                    else:
                        params_preview = repr(parameters)[:120]
                except Exception:  # noqa: BLE001
                    params_preview = "?"
                try:
                    rc = cur.rowcount
                except Exception:  # noqa: BLE001
                    rc = -1
                _append_sql_event(
                    {
                        "ms": round(dt_ms, 3),
                        "sql_preview": (sql or "")[:500],
                        "params_preview": params_preview,
                        "rowcount_after_execute": int(rc) if rc is not None else -1,
                        "tables_guess": _tables_from_sql(sql or ""),
                        "index_heuristic": _index_heuristic(sql or ""),
                    }
                )
            return cur

        def executemany(self, sql: str, seq_of_parameters: Any) -> sqlite3.Cursor:  # type: ignore[override]
            t0 = time.perf_counter()
            cur = self._c.executemany(sql, seq_of_parameters)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if sql_capture_active():
                try:
                    n = len(seq_of_parameters)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    n = -1
                _append_sql_event(
                    {
                        "ms": round(dt_ms, 3),
                        "sql_preview": (sql or "")[:500] + f" [executemany batches={n}]",
                        "params_preview": "",
                        "rowcount_after_execute": -1,
                        "tables_guess": _tables_from_sql(sql or ""),
                        "index_heuristic": "executemany",
                    }
                )
            return cur

        def __getattr__(self, name: str) -> Any:
            return getattr(self._c, name)

    return _ConnProxy(conn)  # type: ignore[return-value]


def begin_prediction_audit() -> None:
    events: list[dict[str, Any]] = []
    phases: list[tuple[str, float]] = []
    _sql_events.set(events)
    _prediction_phases.set(phases)
    _tls.sql_events = events


def end_prediction_audit() -> dict[str, Any]:
    events = _sql_events.get() or []
    phases = _prediction_phases.get() or []
    _sql_events.set(None)
    _prediction_phases.set(None)
    _tls.sql_events = None

    total_sql_ms = round(sum(float(e.get("ms", 0)) for e in events), 2)
    top5 = sorted(events, key=lambda e: float(e.get("ms", 0)), reverse=True)[:5]
    by_kind: dict[str, float] = {}
    for e in events:
        key = (e.get("tables_guess") or ["?"])[0] if e.get("tables_guess") else "?"
        by_kind[key] = by_kind.get(key, 0.0) + float(e.get("ms", 0))

    return {
        "sql_query_count": len(events),
        "sql_total_ms": total_sql_ms,
        "sql_top5_slowest": top5,
        "sql_time_by_first_table_ms": {k: round(v, 2) for k, v in sorted(by_kind.items(), key=lambda x: -x[1])[:12]},
        "phases_ms": {name: round(ms, 2) for name, ms in phases},
    }


def record_prediction_phase(name: str, ms: float) -> None:
    ph = _prediction_phases.get()
    if ph is not None:
        ph.append((name, ms))


@contextmanager
def prediction_phase(name: str) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if audit_enabled():
            record_prediction_phase(name, (time.perf_counter() - t0) * 1000.0)


def merge_prediction_audit_into_result(result: dict[str, Any], audit: dict[str, Any]) -> None:
    result["audit_prediction"] = audit
    _pt = result.get("prediction_timing_ms")
    existing = dict(_pt) if isinstance(_pt, dict) else {}
    for k, v in (audit.get("phases_ms") or {}).items():
        existing[f"audit_{k}"] = v
    existing["audit_sql_total_ms"] = audit.get("sql_total_ms")
    existing["audit_sql_query_count"] = audit.get("sql_query_count")
    result["prediction_timing_ms"] = existing


class PredictionRunAudit:
    """Begin SQL/phase capture at prediction start; finalize into ``result`` on success."""

    def __init__(self) -> None:
        self._on = audit_enabled()
        if self._on:
            begin_prediction_audit()

    def close_success(self, result: dict[str, Any]) -> None:
        if not self._on:
            return
        bundle = end_prediction_audit()
        merge_prediction_audit_into_result(result, bundle)

    def close_failure(self) -> None:
        if self._on:
            end_prediction_audit()


def summarize_startup_phases(phases: list[tuple[str, float]]) -> dict[str, float]:
    return {k: round(v, 2) for k, v in phases}


def append_session_audit_event(
    kind: str,
    name: str,
    ms: float,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append a UI-side timing row (Streamlit ``session_state``) when audit is on."""
    if not audit_enabled():
        return
    try:
        import streamlit as st

        ev = list(st.session_state.get("_audit_streamlit_events") or [])
        ev.append(
            {
                "kind": kind,
                "name": name,
                "ms": round(ms, 2),
                "extra": extra or {},
            }
        )
        st.session_state["_audit_streamlit_events"] = ev[-200:]
    except Exception:  # noqa: BLE001
        pass


def record_tuning_action(name: str, ms: float, *, extra: Optional[dict[str, Any]] = None) -> None:
    """Store tuning-debug button timing + mirror into the general session audit list."""
    payload = {"name": name, "ms": round(ms, 2), "extra": extra or {}}
    append_session_audit_event("prediction_tuning_debug", name, payload["ms"], extra=payload.get("extra"))
    try:
        import streamlit as st

        hist = list(st.session_state.get("_audit_tuning_history") or [])
        hist.append(payload)
        st.session_state["_audit_tuning_history"] = hist[-20:]
        st.session_state["_audit_tuning_last"] = payload
    except Exception:  # noqa: BLE001
        pass
