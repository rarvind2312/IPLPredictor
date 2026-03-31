"""IPL venue metadata: coordinates, typical conditions, chase bias."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class VenueProfile:
    key: str
    display_name: str
    city: str
    latitude: float
    longitude: float
    # 0 = low scores / bowling friendly, 1 = high scores / batting friendly
    batting_friendliness: float
    # 0 = spin friendly, 1 = pace friendly
    pace_bias: float
    # 0 = little dew, 1 = heavy dew risk (evening)
    dew_risk: float
    # Typical boundary size proxy: 0 small, 1 large
    boundary_size: float
    notes: str


# Curated IPL venues (approximate city coordinates / major stadiums)
VENUES: dict[str, VenueProfile] = {}


def _add(v: VenueProfile) -> None:
    VENUES[v.key] = v


_add(
    VenueProfile(
        key="wankhede",
        display_name="Wankhede Stadium, Mumbai",
        city="Mumbai",
        latitude=18.9389,
        longitude=72.8258,
        batting_friendliness=0.72,
        pace_bias=0.55,
        dew_risk=0.78,
        boundary_size=0.35,
        notes="Often high scoring; sea breeze; significant dew under lights.",
    )
)
_add(
    VenueProfile(
        key="eden",
        display_name="Eden Gardens, Kolkata",
        city="Kolkata",
        latitude=22.5646,
        longitude=88.3433,
        batting_friendliness=0.62,
        pace_bias=0.42,
        dew_risk=0.55,
        boundary_size=0.45,
        notes="Large ground; spin can play a big role; balanced par scores.",
    )
)
_add(
    VenueProfile(
        key="chinnaswamy",
        display_name="M. Chinnaswamy Stadium, Bengaluru",
        city="Bengaluru",
        latitude=12.9784,
        longitude=77.5996,
        batting_friendliness=0.88,
        pace_bias=0.62,
        dew_risk=0.65,
        boundary_size=0.22,
        notes="Short boundaries; high run rate venue; pace can be expensive.",
    )
)
_add(
    VenueProfile(
        key="arun_jaitley",
        display_name="Arun Jaitley Stadium, Delhi",
        city="Delhi",
        latitude=28.6379,
        longitude=77.2432,
        batting_friendliness=0.58,
        pace_bias=0.48,
        dew_risk=0.48,
        boundary_size=0.52,
        notes="Slower than it looks; spinners often handy; variable pace bounce.",
    )
)
_add(
    VenueProfile(
        key="chepauk",
        display_name="MA Chidambaram Stadium, Chennai",
        city="Chennai",
        latitude=13.0629,
        longitude=80.2792,
        batting_friendliness=0.48,
        pace_bias=0.35,
        dew_risk=0.42,
        boundary_size=0.58,
        notes="Traditionally spin friendly; harder to chase big totals.",
    )
)
_add(
    VenueProfile(
        key="narendra_modi",
        display_name="Narendra Modi Stadium, Ahmedabad",
        city="Ahmedabad",
        latitude=23.0917,
        longitude=72.5976,
        batting_friendliness=0.55,
        pace_bias=0.58,
        dew_risk=0.52,
        boundary_size=0.62,
        notes="Large outfield; black soil can aid pace; day games differ from night.",
    )
)
_add(
    VenueProfile(
        key="barsapara",
        display_name="Barsapara Stadium, Guwahati",
        city="Guwahati",
        latitude=26.1445,
        longitude=91.7362,
        batting_friendliness=0.65,
        pace_bias=0.52,
        dew_risk=0.60,
        boundary_size=0.48,
        notes="Used as secondary home; humidity and swing can matter.",
    )
)
_add(
    VenueProfile(
        key="dharamsala",
        display_name="HPCA Stadium, Dharamsala",
        city="Dharamsala",
        latitude=32.2190,
        longitude=76.3264,
        batting_friendliness=0.60,
        pace_bias=0.75,
        dew_risk=0.45,
        boundary_size=0.55,
        notes="Altitude assists pace; cool evenings; outfield often quick.",
    )
)
_add(
    VenueProfile(
        key="mohali",
        display_name="Punjab Cricket Association IS Bindra Stadium, Mohali",
        city="Mohali",
        latitude=30.6900,
        longitude=76.7372,
        batting_friendliness=0.57,
        pace_bias=0.55,
        dew_risk=0.50,
        boundary_size=0.50,
        notes="Good pace carry; balanced IPL venue historically.",
    )
)
_add(
    VenueProfile(
        key="hyderabad",
        display_name="Rajiv Gandhi Intl Stadium, Hyderabad",
        city="Hyderabad",
        latitude=17.4065,
        longitude=78.5502,
        batting_friendliness=0.63,
        pace_bias=0.50,
        dew_risk=0.58,
        boundary_size=0.48,
        notes="Generally true surface; spin and cutters effective in middle overs.",
    )
)
_add(
    VenueProfile(
        key="jaipur",
        display_name="Sawai Mansingh Stadium, Jaipur",
        city="Jaipur",
        latitude=26.8940,
        longitude=75.8033,
        batting_friendliness=0.64,
        pace_bias=0.48,
        dew_risk=0.52,
        boundary_size=0.46,
        notes="Good for stroke play; evening dew can tilt chasing.",
    )
)
_add(
    VenueProfile(
        key="lucknow",
        display_name="BRSABV Ekana Stadium, Lucknow",
        city="Lucknow",
        latitude=26.8467,
        longitude=80.9462,
        batting_friendliness=0.52,
        pace_bias=0.45,
        dew_risk=0.48,
        boundary_size=0.68,
        notes="Larger square; can be slower; spin often in play.",
    )
)


def list_venue_choices() -> list[tuple[str, str]]:
    """(key, display_name) sorted by city."""
    items = [(v.key, v.display_name) for v in VENUES.values()]
    items.sort(key=lambda x: x[1].lower())
    return items


def resolve_venue(user_input: str) -> VenueProfile:
    """
    Map free-text venue selection to a profile.
    Falls back to a neutral generic profile if unknown.
    """
    raw = (user_input or "").strip().lower()
    if not raw:
        return _generic("unknown", user_input or "Unknown venue")

    for key, v in VENUES.items():
        if key == raw or key.replace("_", " ") in raw:
            return v
        dn = v.display_name.lower()
        if raw in dn or dn in raw:
            return v
        if v.city.lower() in raw:
            return v
        stadium = dn.split(",")[0].strip()
        if len(stadium) >= 6 and stadium in raw:
            return v

    for key, v in VENUES.items():
        if key in raw:
            return v

    return _generic(raw, user_input.strip())


def _generic(key: str, display: str) -> VenueProfile:
    return VenueProfile(
        key=key,
        display_name=display,
        city="",
        latitude=20.5937,
        longitude=78.9629,
        batting_friendliness=0.55,
        pace_bias=0.50,
        dew_risk=0.50,
        boundary_size=0.50,
        notes="Generic Indian T20 conditions (unknown venue).",
    )


def venue_conditions_summary(venue: VenueProfile, weather: dict[str, Any]) -> dict[str, Any]:
    """Merge static venue priors with live weather into analyst-facing features."""
    precip = float(weather.get("precipitation_mm", 0) or 0)
    wind = float(weather.get("wind_kmh", 0) or 0)
    cloud = float(weather.get("cloud_cover_pct", 0) or 0)
    temp_c = float(weather.get("temperature_c", 28) or 28)
    rh = float(weather.get("relative_humidity_pct", 55) or 55)

    # Derived scalars 0–1
    swing_proxy = min(1.0, (rh / 100.0) * 0.55 + (cloud / 100.0) * 0.35 + min(1.0, wind / 40.0) * 0.25)
    spin_grip = max(0.0, 1.0 - swing_proxy * 0.85)
    rain_risk = min(1.0, precip / 8.0)
    heat_fatigue = max(0.0, min(1.0, (temp_c - 26) / 14.0))

    dew_effective = min(1.0, venue.dew_risk * (0.55 + rh / 200.0))

    return {
        "venue": venue.display_name,
        "batting_friendliness": venue.batting_friendliness,
        "pace_bias": venue.pace_bias,
        "dew_risk": dew_effective,
        "boundary_size": venue.boundary_size,
        "swing_seam_proxy": swing_proxy,
        "spin_friendliness": spin_grip,
        "rain_disruption_risk": rain_risk,
        "heat_fatigue": heat_fatigue,
        "notes": venue.notes,
        "weather_snapshot": weather,
    }
