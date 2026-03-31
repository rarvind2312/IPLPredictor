"""Fetch weather for match time using Open-Meteo (no API key)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import requests

import config
import time_utils
from venues import VenueProfile


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def fetch_weather(
    venue: VenueProfile,
    match_time: datetime,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """
    Hourly forecast closest to match_time at venue coordinates.
    Match time is interpreted as IST (Asia/Kolkata) for IPL — use IST wall clock with Open-Meteo.
    """
    sess = session or _session()
    mt_ist = time_utils.ist_naive_wall_clock(
        match_time if isinstance(match_time, datetime) else datetime.combine(match_time, datetime.min.time())
    )
    params = {
        "latitude": venue.latitude,
        "longitude": venue.longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "timezone": "Asia/Kolkata",
        "start_date": mt_ist.strftime("%Y-%m-%d"),
        "end_date": mt_ist.strftime("%Y-%m-%d"),
    }
    try:
        r = sess.get(
            config.OPEN_METEO_URL,
            params=params,
            timeout=config.REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "temperature_c": 28.0,
            "relative_humidity_pct": 55.0,
            "precipitation_mm": 0.0,
            "precipitation_probability_pct": 0.0,
            "cloud_cover_pct": 40.0,
            "wind_kmh": 12.0,
            "hour_iso": mt_ist.isoformat(),
        }

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {
            "ok": False,
            "error": "No hourly data in response",
            "temperature_c": 28.0,
            "relative_humidity_pct": 55.0,
            "precipitation_mm": 0.0,
            "precipitation_probability_pct": 0.0,
            "cloud_cover_pct": 40.0,
            "wind_kmh": 12.0,
            "hour_iso": mt_ist.isoformat(),
        }

    target = mt_ist
    best_i = 0
    best_delta = None
    for i, ts in enumerate(times):
        try:
            raw = str(ts).replace("Z", "+00:00")
            t = datetime.fromisoformat(raw)
            if t.tzinfo is not None:
                t = t.astimezone(time_utils.IST).replace(tzinfo=None)
        except ValueError:
            continue
        delta = abs((t - target).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_i = i

    def series(name: str) -> list:
        return list(hourly.get(name) or [])

    temps = series("temperature_2m")
    hums = series("relative_humidity_2m")
    pprob = series("precipitation_probability")
    precip = series("precipitation")
    clouds = series("cloud_cover")
    wind = series("wind_speed_10m")

    def at(idx: int, arr: list, default: float) -> float:
        if idx < len(arr) and arr[idx] is not None:
            try:
                return float(arr[idx])
            except (TypeError, ValueError):
                return default
        return default

    return {
        "ok": True,
        "error": None,
        "temperature_c": at(best_i, temps, 28.0),
        "relative_humidity_pct": at(best_i, hums, 55.0),
        "precipitation_mm": at(best_i, precip, 0.0),
        "precipitation_probability_pct": at(best_i, pprob, 0.0),
        "cloud_cover_pct": at(best_i, clouds, 40.0),
        "wind_kmh": at(best_i, wind, 12.0),
        "hour_iso": times[best_i] if best_i < len(times) else mt_ist.isoformat(),
        "timezone_note": "Asia/Kolkata (IST)",
    }
