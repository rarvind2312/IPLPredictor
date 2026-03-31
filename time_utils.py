"""Match time in India Standard Time (IST) for weather, dew, and day/night logic."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def combine_date_time_ist(d: date, t: time) -> datetime:
    """Build timezone-aware datetime in Asia/Kolkata."""
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second, tzinfo=IST)


def as_ist_datetime(match_time: datetime) -> datetime:
    """Normalize to aware IST (naive inputs are treated as IST wall clock)."""
    if match_time.tzinfo is None:
        return match_time.replace(tzinfo=IST)
    return match_time.astimezone(IST)


def ist_naive_wall_clock(match_time: datetime) -> datetime:
    """Naive datetime representing the same IST wall-clock time (for Open-Meteo hourly keys)."""
    aware = as_ist_datetime(match_time)
    return aware.replace(tzinfo=None)


def ist_hour(match_time: datetime) -> int:
    return as_ist_datetime(match_time).hour
