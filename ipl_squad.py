"""
IPL-only squad semantics: canonical role buckets and structured squad members.

Used by squad_fetch (IPLT20 pages) and predictor (XI / batting order).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import learner

# Canonical IPL display buckets (match IPL site copy)
BATTER = "Batter"
WK_BATTER = "WK-Batter"
ALL_ROUNDER = "All-Rounder"
BOWLER = "Bowler"

ROLE_BUCKETS: tuple[str, ...] = (BATTER, WK_BATTER, ALL_ROUNDER, BOWLER)

_ROLE_BUCKET_LOWER: dict[str, str] = {
    "batter": BATTER,
    "batters": BATTER,
    "wk-batter": WK_BATTER,
    "wk batter": WK_BATTER,
    "wicket-keeper batter": WK_BATTER,
    "wicketkeeper batter": WK_BATTER,
    "wk-bat": WK_BATTER,
    "all-rounder": ALL_ROUNDER,
    "all rounder": ALL_ROUNDER,
    "allrounder": ALL_ROUNDER,
    "bowler": BOWLER,
    "bowlers": BOWLER,
}

# Names must not be exactly these (lowercase)
_INVALID_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "all",
        "bat",
        "bowl",
        "wk",
        "batter",
        "bowler",
        "about",
        "contact",
        "guidelines",
        "policy",
        "terms",
        "privacy",
        "home",
        "menu",
        "squad",
        "players",
    }
)

_EMBEDDED_ROLE_SUFFIX = re.compile(
    r"\s+(Batter|WK[-\s]?Batter|Bowler|All[-\s]?Rounder)\s*$",
    re.IGNORECASE,
)


@dataclass
class IplSquadMember:
    """Structured row from IPL squad source (before scoring)."""

    name: str
    role_bucket: str
    overseas: bool = False
    batting_roles: list[str] = field(default_factory=list)
    bowling_type: Optional[str] = None  # "pace", "spin", "mixed" when known
    is_keeper: bool = False
    team_name: str = ""
    canonical_team_key: str = ""
    canonical_player_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "player_name": self.name,
            "role_bucket": self.role_bucket,
            "overseas": self.overseas,
            "batting_roles": list(self.batting_roles),
            "bowling_type": self.bowling_type,
            "is_keeper": self.is_keeper,
            "team_name": self.team_name,
            "canonical_team_key": self.canonical_team_key,
            "canonical_player_key": self.canonical_player_key,
        }


def normalize_role_bucket_label(raw: str) -> Optional[str]:
    """Map free text to one of ROLE_BUCKETS, or None if unknown."""
    if not raw:
        return None
    s = raw.strip()
    low = s.lower().replace("_", " ")
    low = re.sub(r"\s+", " ", low)
    if low in _ROLE_BUCKET_LOWER:
        return _ROLE_BUCKET_LOWER[low]
    if "wicket" in low and "bat" in low:
        return WK_BATTER
    if "wk" in low and "bat" in low:
        return WK_BATTER
    if re.search(r"all[-\s]*round", low):
        return ALL_ROUNDER
    if low.startswith("bowl") or re.search(r"\bbowler\b", low):
        return BOWLER
    if low.startswith("bat") or "batsman" in low:
        return BATTER
    return None


def split_embedded_role_from_name(name_raw: str) -> tuple[str, Optional[str]]:
    """
    If IPL concatenated role into the name (e.g. 'Shubham Dubey Batter'),
    return (clean_name, bucket_or_none).
    """
    t = re.sub(r"\s+", " ", (name_raw or "").strip())
    if not t:
        return "", None
    m = _EMBEDDED_ROLE_SUFFIX.search(t)
    if not m:
        return t, None
    suffix = m.group(1)
    base = t[: m.start()].strip()
    bucket = normalize_role_bucket_label(suffix)
    return base, bucket


def infer_bowling_type_from_styles(obj: dict[str, Any]) -> Optional[str]:
    b = str(obj.get("bowlingStyle") or obj.get("bowling_style") or "").lower()
    if not b:
        return None
    if "spin" in b or "leg" in b or "off" in b or "orthodox" in b:
        return "spin"
    if "pace" in b or "fast" in b or "medium" in b or "seam" in b:
        return "pace"
    return "mixed"


def default_batting_roles_for_bucket(role_bucket: str, *, is_keeper: bool) -> list[str]:
    """Default batting-order hints (refined later in predictor after scoring)."""
    if role_bucket == BATTER:
        return ["top_order", "middle"]
    if role_bucket == WK_BATTER:
        return ["middle", "flex"]
    if role_bucket == ALL_ROUNDER:
        return ["middle", "lower_middle", "finisher"]
    if role_bucket == BOWLER:
        return ["tail"]
    return ["middle"]


def role_bucket_to_predictor_role(role_bucket: str) -> str:
    """Map IPL bucket to internal bat | wk | all | bowl."""
    if role_bucket == WK_BATTER:
        return "wk"
    if role_bucket == BOWLER:
        return "bowl"
    if role_bucket == ALL_ROUNDER:
        return "all"
    return "bat"


def overseas_from_api_record(obj: dict[str, Any]) -> bool:
    """
    Infer overseas from IPL JSON when present. Conservative: unknown nationality → not overseas.
    """
    if obj.get("isForeignPlayer") is True or obj.get("isOverseasPlayer") is True:
        return True
    if obj.get("isOverseas") is True or obj.get("overseas") is True:
        return True
    nat = str(obj.get("nationality") or obj.get("country") or "").strip().lower()
    if not nat:
        return False
    if "india" in nat:
        return False
    return True


def role_bucket_from_api_record(obj: dict[str, Any]) -> str:
    """
    Prefer official IPL JSON. Conservative defaults: **Batter** unless source clearly says otherwise.
    Do not treat vague substrings (e.g. ``bowl`` inside ``batsman``) as Bowler.
    """
    playing = str(obj.get("playingRole") or obj.get("role") or "").strip()
    skill = str(obj.get("playerSkill") or obj.get("category") or "").strip()
    low = f"{playing} {skill}".lower()
    low = re.sub(r"\s+", " ", low)

    if re.search(r"\bwk\b", low) or "wicket" in low:
        return WK_BATTER
    if re.search(r"all[-\s]*round", low):
        return ALL_ROUNDER
    if re.search(r"\bbowler\b", low) or low.strip() in ("bowl", "bowling"):
        return BOWLER

    for chunk in (playing, skill, f"{playing} {skill}"):
        b = normalize_role_bucket_label(chunk)
        if b == ALL_ROUNDER and not re.search(r"all[-\s]*round", chunk.lower()):
            continue
        if b:
            return b
    return BATTER


def validate_clean_name(name: str) -> tuple[bool, str]:
    """Return (ok, reason_if_bad)."""
    t = re.sub(r"\s+", " ", (name or "").strip())
    if len(t) < 3:
        return False, "too_short"
    if len(t) > 80:
        return False, "too_long"
    low = t.lower()
    if low in _INVALID_NAME_TOKENS:
        return False, "invalid_token_name"
    if t.isupper() and len(t) <= 22 and " " not in t:
        return False, "all_caps_nav"
    if not re.search(r"[a-zA-Z]", t):
        return False, "no_letters"
    words = re.findall(r"[A-Za-z][A-Za-z'\-.]*", t)
    if len(words) < 2 and len(words[0]) < 6 if words else True:
        return False, "not_name_shaped"
    return True, ""


def build_ipl_squad_member(
    *,
    name: str,
    role_bucket: str,
    overseas: bool = False,
    api_obj: Optional[dict[str, Any]] = None,
    team_name: str = "",
    canonical_team_key: str = "",
    canonical_player_key: str = "",
) -> IplSquadMember:
    rb = role_bucket if role_bucket in ROLE_BUCKETS else BATTER
    is_k = rb == WK_BATTER
    bowling = infer_bowling_type_from_styles(api_obj) if api_obj else None
    roles = default_batting_roles_for_bucket(rb, is_keeper=is_k)
    cpk = (canonical_player_key or "").strip() or learner.normalize_player_key(name)
    ctk = (canonical_team_key or "").strip()[:80]
    return IplSquadMember(
        name=name,
        role_bucket=rb,
        overseas=overseas,
        batting_roles=roles,
        bowling_type=bowling,
        is_keeper=is_k,
        team_name=team_name,
        canonical_team_key=ctk,
        canonical_player_key=cpk,
    )
