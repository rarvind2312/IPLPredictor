"""Canonical merged player registry for runtime metadata, aliases, and marquee tags."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Optional

import config
import ipl_teams
import learner

logger = logging.getLogger(__name__)

_REGISTRY_CACHE: Optional[dict[str, Any]] = None
_REGISTRY_CACHE_MTIME: Optional[float] = None
_REGISTRY_METADATA_LOOKUP_CACHE: Optional[dict[str, dict[str, Any]]] = None
_REGISTRY_MARQUEE_LOOKUP_CACHE: Optional[dict[str, dict[str, Any]]] = None
_REGISTRY_ALIAS_OVERRIDE_CACHE: Optional[tuple[dict[str, list[str]], dict[str, str]]] = None

_SOURCE_PRIORITY = {
    "marquee_override": 100,
    "alias_override": 90,
    "curated_manual": 80,
    "squad_json": 50,
    "cricinfo_curated": 30,
    "previous_registry": 20,
    "db_linkage_enrichment": 10,
    "raw_cricsheet_fallback": 10,
    "registry_slot_defaults": 5,
}


def _resolve_path(raw_path: str) -> Path:
    p = Path(str(raw_path or "").strip())
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _registry_path() -> Path:
    return _resolve_path(
        str(getattr(config, "PLAYER_REGISTRY_MASTER_PATH", "") or "data/player_registry_master.json")
    )


def _registry_linkage_audit_path() -> Path:
    return _resolve_path(
        str(
            getattr(config, "PLAYER_REGISTRY_LINKAGE_AUDIT_PATH", "")
            or "data/player_registry_linkage_enrichment_audit.json"
        )
    )


def _db_path() -> Path:
    return _resolve_path(str(getattr(config, "DB_PATH", "") or "data/ipl_predictor.sqlite"))


def _squads_dir_path() -> Path:
    return _resolve_path(str(getattr(config, "PLAYER_SQUADS_DIR", "") or "data/squads"))


def _load_json_file(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)


def _normalize(raw: str) -> str:
    return learner.normalize_player_key(str(raw or ""))


def _compact_key(raw: str) -> str:
    return "".join(_normalize(raw).split())


def _title_from_key(key: str) -> str:
    tokens = [t for t in _normalize(key).split() if t]
    if not tokens:
        return ""
    return " ".join(t.capitalize() if len(t) > 1 else t.upper() for t in tokens)


def _blank_player_record(registry_key: str, *, seed_display_name: str = "") -> dict[str, Any]:
    display_name = seed_display_name.strip() or _title_from_key(registry_key)
    return {
        "registry_key": registry_key,
        "canonical_name": display_name,
        "display_name": display_name,
        "history_canonical_key": "",
        "aliases": [],
        "team": "",
        "squad_status": "",
        "status": "",
        "batting_hand": "",
        "batting_style": "",
        "bowling_style_raw": "",
        "bowling_style": "",
        "bowling_type_bucket": "",
        "primary_role": "",
        "secondary_role": "",
        "role_description": "",
        "allrounder_type": "",
        "is_captain": False,
        "is_vice_captain": False,
        "is_wicketkeeper": False,
        "age": "",
        "likely_batting_band": "",
        "likely_bowling_phases": "",
        "allowed_batting_slots": [],
        "preferred_batting_slots": [],
        "opener_eligible": False,
        "finisher_eligible": False,
        "floater_eligible": False,
        "allrounder_subtype_hint": "",
        "marquee_tier": "",
        "marquee_reason": "",
        "notes": "",
        "metadata_source_summary": {
            "sources_seen": [],
            "preferred_metadata_source": "",
            "alias_override_applied": False,
            "marquee_override_applied": False,
            "history_canonical_source": "",
        },
        "field_sources": {},
        "confidence": 0.0,
    }


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        nk = _normalize(item)
        if nk and nk not in out:
            out.append(nk)
    return out


def _dedupe_exact_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def _record_lookup_keys(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    candidates = [
        record.get("registry_key"),
        record.get("display_name"),
        record.get("canonical_name"),
        record.get("history_canonical_key"),
    ]
    candidates.extend(record.get("aliases") or [])

    for raw in candidates:
        nk = _normalize(str(raw or ""))
        if not nk:
            continue
        if nk not in out:
            out.append(nk)

        # Universal variant indexing
        tokens = nk.split()
        if len(tokens) >= 2:
            # 1. Initials variant (e.g. "MW Short")
            initials = "".join(t[0] for t in tokens[:-1])
            v_init = f"{initials} {tokens[-1]}"
            if v_init not in out:
                out.append(v_init)
            # 2. First-Last variant (e.g. "Matthew Short" from "Matthew William Short")
            if len(tokens) >= 3:
                v_fl = f"{tokens[0]} {tokens[-1]}"
                if v_fl not in out:
                    out.append(v_fl)
    return _dedupe_keep_order(out)


def _build_indexes(records: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    compact: dict[str, str] = {}
    for registry_key, record in records.items():
        for candidate in _record_lookup_keys(record):
            exact.setdefault(candidate, registry_key)
            ck = _compact_key(candidate)
            if ck:
                compact.setdefault(ck, registry_key)
    return exact, compact


def _resolve_existing_registry_key(
    raw_candidates: list[str],
    records: dict[str, dict[str, Any]],
) -> str:
    exact, compact = _build_indexes(records)
    
    # 1. Highest confidence: Exact normalized or compact match
    for raw in raw_candidates:
        nk = _normalize(raw)
        if nk and nk in exact:
            return exact[nk]
    for raw in raw_candidates:
        ck = _compact_key(raw)
        if ck and ck in compact:
            return compact[ck]
            
    # 2. fuller-name variant resolution (subset/superset matching)
    # e.g. "Matthew Short" should resolve to "Matthew William Short" and vice versa.
    for raw in raw_candidates:
        nk = _normalize(raw)
        if not nk:
            continue
        tokens = set(nk.split())
        if len(tokens) < 2:
            continue
        
        nk_surname = _surname_signature(nk)
        for registry_key, record in records.items():
            for candidate in _record_lookup_keys(record):
                cand_tokens = set(candidate.split())
                if len(cand_tokens) < 2:
                    continue
                if (tokens.issubset(cand_tokens) or cand_tokens.issubset(tokens)) and _surname_signature(candidate) == nk_surname:
                    return registry_key
    return ""

def _capture_aliases(record: dict[str, Any], candidates: list[str]) -> None:
    registry_key = _normalize(str(record.get("registry_key") or ""))
    merged_aliases = list(record.get("aliases") or [])
    for cand in candidates:
        nk = _normalize(cand)
        if nk and nk != registry_key and nk not in merged_aliases:
            merged_aliases.append(nk)
    if merged_aliases:
        record["aliases"] = _dedupe_keep_order(merged_aliases)


def _ensure_record(
    registry_key: str,
    records: dict[str, dict[str, Any]],
    *,
    seed_display_name: str = "",
) -> dict[str, Any]:
    nk = _normalize(registry_key)
    if not nk:
        raise ValueError("registry_key must be non-empty")
    record = records.get(nk)
    if record is None:
        record = _blank_player_record(nk, seed_display_name=seed_display_name)
        records[nk] = record
    elif seed_display_name and not str(record.get("display_name") or "").strip():
        record["display_name"] = seed_display_name.strip()
        record["canonical_name"] = seed_display_name.strip()
    return record


def _set_field(record: dict[str, Any], field: str, value: Any, source_label: str) -> None:
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return
    elif isinstance(value, list):
        if not value:
            return
    elif isinstance(value, dict):
        if not value:
            return

    # Task 6: Preserve stronger curated/master metadata by checking source priority
    field_sources = record.get("field_sources", {})
    current_source = field_sources.get(field)
    if current_source:
        current_prio = _SOURCE_PRIORITY.get(current_source, 0)
        new_prio = _SOURCE_PRIORITY.get(source_label, 0)
        if current_prio > new_prio:
            # Current value is from a more authoritative source, do not override
            return

    record[field] = value
    record.setdefault("field_sources", {})[field] = source_label


def _append_source(record: dict[str, Any], source_label: str) -> None:
    summary = record.setdefault("metadata_source_summary", {})
    seen = list(summary.get("sources_seen") or [])
    if source_label and source_label not in seen:
        seen.append(source_label)
    summary["sources_seen"] = seen


def _apply_metadata_payload(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_label: str,
) -> None:
    _append_source(record, source_label)
    display_name = str(payload.get("display_name") or payload.get("player_name") or "").strip()
    if display_name:
        _set_field(record, "display_name", display_name, source_label)
        _set_field(record, "canonical_name", display_name, source_label)
    for field in (
        "team",
        "squad_status",
        "status",
        "batting_hand",
        "batting_style",
        "bowling_style_raw",
        "bowling_style",
        "bowling_type_bucket",
        "primary_role",
        "secondary_role",
        "role_description",
        "allrounder_type",
        "is_captain",
        "is_vice_captain",
        "is_wicketkeeper",
        "age",
        "likely_batting_band",
        "likely_bowling_phases",
        "allowed_batting_slots",
        "preferred_batting_slots",
        "opener_eligible",
        "finisher_eligible",
        "floater_eligible",
        "allrounder_subtype_hint",
        "notes",
        "marquee_tier",
    ):
        _set_field(record, field, payload.get(field), source_label)

    try:
        conf_val = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf_val = 0.0
    if conf_val > float(record.get("confidence") or 0.0):
        record["confidence"] = max(0.0, min(1.0, conf_val))
        record.setdefault("field_sources", {})["confidence"] = source_label

    summary = record.setdefault("metadata_source_summary", {})
    preferred = str(summary.get("preferred_metadata_source") or "").strip()
    if not preferred or source_label in ("curated_manual", "squad_json"):
        summary["preferred_metadata_source"] = source_label


def _normalize_slot_list(raw: Any) -> list[int]:
    values: list[int] = []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    elif isinstance(raw, str):
        items = [part.strip() for part in raw.replace(";", ",").split(",")]
    else:
        items = []
    for item in items:
        try:
            slot = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= slot <= 11 and slot not in values:
            values.append(slot)
    return values


def _slot_defaults_from_role(record: dict[str, Any]) -> tuple[list[int], list[int], bool, bool, bool]:
    likely_band = _registry_band_hint(record)
    role_description = str(record.get("role_description") or "").strip().lower()
    allrounder_type = str(record.get("allrounder_type") or "").strip().lower()
    primary_role = str(record.get("primary_role") or "").strip().lower()
    is_wicketkeeper = bool(record.get("is_wicketkeeper"))

    label = likely_band or role_description or allrounder_type or primary_role
    if label in ("opener", "opening_batter"):
        return [1, 2, 3], [1, 2], True, False, False
    if label == "top_order":
        return [1, 2, 3, 4], [2, 3, 4], False, False, False
    if label == "middle_order":
        return [5, 6, 7], [5, 6], False, False, False
    if label == "lower_middle":
        return [5, 6, 7, 8], [6, 7], False, True, True
    if label == "lower_order":
        return [9, 10, 11], [9, 10], False, False, False
    if label == "wicketkeeper_batter" or (primary_role == "wk_batter" and not likely_band):
        return [5, 6, 7], [5, 6], False, False, True
    if label == "batting_allrounder":
        return [4, 5, 6, 7], [5, 6], False, True, True
    if label in ("balanced_allrounder", "allrounder", "all_rounder"):
        return [5, 6, 7, 8], [6, 7], False, True, True
    if label == "bowling_allrounder":
        return [7, 8, 9, 10, 11], [8, 9], False, False, True
    if label == "bowler" or primary_role == "bowler":
        return [8, 9, 10, 11], [9, 10], False, False, False
    if label == "batter" or primary_role == "batter":
        return [1, 2, 3, 4, 5, 6, 7], [3, 4, 5], False, False, False
    if is_wicketkeeper:
        return [5, 6, 7], [5, 6], False, False, True
    return [5, 6, 7], [5, 6], False, False, False


def _apply_slot_constraint_defaults(record: dict[str, Any]) -> None:
    normalized_band = _registry_band_hint(record)
    if normalized_band and normalized_band != str(record.get("likely_batting_band") or "").strip().lower():
        record["likely_batting_band"] = normalized_band
    allowed = _normalize_slot_list(record.get("allowed_batting_slots"))
    preferred = _normalize_slot_list(record.get("preferred_batting_slots"))
    opener_eligible = bool(record.get("opener_eligible"))
    finisher_eligible = bool(record.get("finisher_eligible"))
    floater_eligible = bool(record.get("floater_eligible"))
    if not allowed or not preferred:
        d_allowed, d_preferred, d_opener, d_finisher, d_floater = _slot_defaults_from_role(record)
        if not allowed:
            allowed = d_allowed
            _set_field(record, "allowed_batting_slots", allowed, "registry_slot_defaults")
        if not preferred:
            preferred = d_preferred
            _set_field(record, "preferred_batting_slots", preferred, "registry_slot_defaults")
        if d_opener:
            opener_eligible = opener_eligible or d_opener
        if d_finisher:
            finisher_eligible = finisher_eligible or d_finisher
        if d_floater:
            floater_eligible = floater_eligible or d_floater
    preferred = [slot for slot in preferred if slot in allowed] or list(allowed[:2])
    record["allowed_batting_slots"] = allowed
    record["preferred_batting_slots"] = preferred
    record["opener_eligible"] = bool(opener_eligible and any(slot <= 3 for slot in allowed))
    record["finisher_eligible"] = bool(finisher_eligible and any(5 <= slot <= 8 for slot in allowed))
    record["floater_eligible"] = bool(floater_eligible)
    field_sources = record.setdefault("field_sources", {})
    field_sources.setdefault("allowed_batting_slots", "registry_slot_defaults")
    field_sources.setdefault("preferred_batting_slots", "registry_slot_defaults")
    field_sources.setdefault("opener_eligible", "registry_slot_defaults")
    field_sources.setdefault("finisher_eligible", "registry_slot_defaults")
    field_sources.setdefault("floater_eligible", "registry_slot_defaults")


def _normalize_marquee_players(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    src = raw.get("players") if isinstance(raw.get("players"), dict) else raw
    out: dict[str, dict[str, Any]] = {}
    for key, value in (src.items() if isinstance(src, dict) else []):
        nk = _normalize(str(key or ""))
        if nk and isinstance(value, dict):
            out[nk] = dict(value)
    return out


_SQUAD_TEAM_CODE_TO_LABEL: dict[str, str] = {
    "CSK": "Chennai Super Kings",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "RCB": "Royal Challengers Bengaluru",
    "SRH": "Sunrisers Hyderabad",
    "RR": "Rajasthan Royals",
    "GT": "Gujarat Titans",
    "DC": "Delhi Capitals",
    "LSG": "Lucknow Super Giants",
    "PBKS": "Punjab Kings",
}


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def _normalize_primary_role_from_squad(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in {"allrounder", "all_rounder"}:
        return "all_rounder"
    if s in {"batter", "batsman", "bat"}:
        return "batter"
    if s in {"bowler", "bowl"}:
        return "bowler"
    if s in {"wk_batter", "wicketkeeper_batter", "wicketkeeper", "keeper"}:
        return "wk_batter"
    return s


def _normalize_secondary_role_from_squad(raw: Any, *, is_wicketkeeper: bool, primary_role: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"allrounder", "all_rounder"}:
        return "all_rounder"
    if s in {"batting_allrounder", "balanced_allrounder", "bowling_allrounder"}:
        return s
    if s in {"wk_batter", "wicketkeeper_batter", "wicketkeeper", "keeper"}:
        return "wk_batter"
    if not s and is_wicketkeeper and primary_role in {"batter", "wk_batter"}:
        return "wk_batter"
    return s


def _normalize_batting_hand_from_squad(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s.startswith("right"):
        return "right"
    if s.startswith("left"):
        return "left"
    return s


def _normalize_bowling_style_from_squad(raw: Any) -> str:
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_team_label_from_squad(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.upper() in _SQUAD_TEAM_CODE_TO_LABEL:
        return _SQUAD_TEAM_CODE_TO_LABEL[s.upper()]
    return (
        ipl_teams.canonical_franchise_label(s)
        or ipl_teams.canonical_franchise_label(s.replace("_", " ").replace("-", " "))
        or ""
    )


def _load_previous_registry_players() -> dict[str, dict[str, Any]]:
    payload = _load_json_file(_registry_path())
    players = payload.get("players") if isinstance(payload, dict) else {}
    return dict(players) if isinstance(players, dict) else {}


def _preserve_previous_registry_fields(
    records: dict[str, dict[str, Any]],
    previous_players: dict[str, dict[str, Any]],
) -> None:
    preserve_fields = (
        "history_canonical_key",
        "aliases",
        "likely_batting_band",
        "likely_bowling_phases",
    )
    for registry_key, prev in previous_players.items():
        if registry_key not in records or not isinstance(prev, dict):
            continue
        record = records[registry_key]
        for field in preserve_fields:
            value = prev.get(field)
            if field == "aliases":
                aliases = _dedupe_keep_order(list(value or []))
                if aliases:
                    record["aliases"] = aliases
                    record.setdefault("field_sources", {})["aliases"] = str(
                        ((prev.get("field_sources") or {}).get("aliases") or "previous_registry")
                    )
            else:
                _set_field(
                    record,
                    field,
                    value,
                    str(((prev.get("field_sources") or {}).get(field) or "previous_registry")),
                )


def _load_squad_json_records() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    squads_dir = _squads_dir_path()
    out: dict[str, dict[str, Any]] = {}
    summary = {
        "squads_dir": str(squads_dir),
        "files_seen": [],
        "files_used": [],
        "duplicate_team_files": {},
    }
    if not squads_dir.is_dir():
        return out, summary
    team_file_seen: dict[str, str] = {}
    for path in sorted(squads_dir.glob("*.json")):
        summary["files_seen"].append(str(path))
        payload = _load_json_file(path)
        if not isinstance(payload, dict):
            continue
        team_label = _canonical_team_label_from_squad(payload.get("team"))
        players = payload.get("players")
        if not isinstance(players, list):
            continue
        team_key = _normalize(team_label)
        if team_key and team_key in team_file_seen:
            summary["duplicate_team_files"].setdefault(team_key, []).append(str(path))
        elif team_key:
            team_file_seen[team_key] = str(path)
        summary["files_used"].append(str(path))
        for player in players:
            if not isinstance(player, dict):
                continue
            name = str(player.get("name") or player.get("display_name") or "").strip()
            registry_key = _normalize(name)
            if not registry_key:
                continue
            primary_role = _normalize_primary_role_from_squad(player.get("primary_role"))
            is_wk = _normalize_bool(player.get("is_wicketkeeper"))
            squad_record = {
                "display_name": name,
                "canonical_name": name,
                "team": team_label,
                "squad_status": str(player.get("status") or "").strip().lower(),
                "status": str(player.get("status") or "").strip().lower(),
                "primary_role": primary_role,
                "secondary_role": _normalize_secondary_role_from_squad(
                    player.get("allrounder_type"),
                    is_wicketkeeper=is_wk,
                    primary_role=primary_role,
                ),
                "role_description": str(player.get("role_description") or "").strip().lower(),
                "allrounder_type": str(player.get("allrounder_type") or "").strip().lower(),
                "allrounder_subtype_hint": str(player.get("allrounder_type") or "").strip().lower(),
                "is_captain": _normalize_bool(player.get("is_captain")),
                "is_vice_captain": _normalize_bool(player.get("is_vice_captain")),
                "is_wicketkeeper": is_wk,
                "age": str(player.get("age") or "").strip(),
                "batting_style": str(player.get("batting_style") or "").strip().lower(),
                "batting_hand": _normalize_batting_hand_from_squad(player.get("batting_hand")),
                "bowling_style": _normalize_bowling_style_from_squad(player.get("bowling_style")),
                "bowling_style_raw": _normalize_bowling_style_from_squad(player.get("bowling_style")),
                "bowling_type_bucket": str(player.get("bowling_type_bucket") or "").strip().lower(),
                "allowed_batting_slots": _normalize_slot_list(player.get("allowed_batting_slots")),
                "preferred_batting_slots": _normalize_slot_list(player.get("preferred_batting_slots")),
                "opener_eligible": _normalize_bool(player.get("opener_eligible")),
                "finisher_eligible": _normalize_bool(player.get("finisher_eligible")),
                "floater_eligible": _normalize_bool(player.get("floater_eligible")),
                "marquee_tier": str(player.get("marquee_tier") or "").strip().lower(),
                "notes": str(player.get("notes") or "").strip(),
                "source": "squad_json",
                "confidence": 0.99,
            }
            existing = out.get(registry_key)
            if existing and existing.get("team") and squad_record.get("team") and existing.get("team") != squad_record.get("team"):
                continue
            out[registry_key] = squad_record
    return out, summary


def _split_given_surname_tokens(norm_name: str) -> tuple[list[str], list[str]]:
    tokens = [t for t in _normalize(norm_name).split() if t]
    if not tokens:
        return [], []
    if len(tokens) == 1:
        return [], tokens[:]
    surname_particles = {
        "de",
        "van",
        "der",
        "den",
        "da",
        "di",
        "du",
        "von",
        "st",
        "ste",
        "le",
        "la",
        "el",
        "al",
        "ibn",
        "ben",
        "bin",
        "del",
        "della",
        "ter",
        "ten",
        "op",
        "mac",
        "mc",
    }
    if len(tokens) >= 3 and tokens[-2] in surname_particles:
        return tokens[:-2], tokens[-2:]
    return tokens[:-1], tokens[-1:]


def _surname_signature(norm_name: str) -> str:
    _given, surname = _split_given_surname_tokens(norm_name)
    return " ".join(surname).strip()


def _edit_distance_le_one(a: str, b: str) -> bool:
    if a == b:
        return True
    la = len(a)
    lb = len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if la > lb:
            i += 1
        elif lb > la:
            j += 1
        else:
            i += 1
            j += 1
    if i < la or j < lb:
        edits += 1
    return edits <= 1


def _safe_full_name_variant_match(expected_name: str, observed_name: str) -> bool:
    exp_given, exp_sur = _split_given_surname_tokens(expected_name)
    obs_given, obs_sur = _split_given_surname_tokens(observed_name)
    if not exp_given or not obs_given or exp_sur != obs_sur or len(exp_given) != len(obs_given):
        return False
    diff_n = 0
    for exp_tok, obs_tok in zip(exp_given, obs_given):
        if exp_tok == obs_tok:
            continue
        if exp_tok[:1] != obs_tok[:1] or not _edit_distance_le_one(exp_tok, obs_tok):
            return False
        diff_n += 1
    return diff_n == 1


def _initials_pool_candidate_match(expected_name: str, observed_name: str) -> bool:
    exp_given, exp_sur = _split_given_surname_tokens(expected_name)
    obs_given, obs_sur = _split_given_surname_tokens(observed_name)
    if not exp_given or not obs_given or exp_sur != obs_sur:
        return False
    first_obs = str(obs_given[0] or "")
    if len(first_obs) > 3:
        return False
    return exp_given[0][:1] == first_obs[:1]


def _safe_surname_variant_initial_match(expected_name: str, observed_name: str) -> bool:
    exp_given, exp_sur = _split_given_surname_tokens(expected_name)
    obs_given, obs_sur = _split_given_surname_tokens(observed_name)
    if not exp_given or not obs_given:
        return False
    exp_first = str(exp_given[0] or "")
    obs_first = str(obs_given[0] or "")
    if not exp_first or not obs_first or exp_first[:1] != obs_first[:1]:
        return False
    if len(obs_first) > 3:
        return False
    exp_surname = " ".join(exp_sur).strip()
    obs_surname = " ".join(obs_sur).strip()
    if not exp_surname or not obs_surname:
        return False
    def consonant_signature(value: str) -> str:
        return "".join(ch for ch in value if ch not in "aeiou")
    if consonant_signature(exp_surname) == consonant_signature(obs_surname):
        return True
    return _edit_distance_le_one(exp_surname, obs_surname)


def _batting_band_from_positions(position_counts: dict[int, int]) -> str:
    total = sum(int(v or 0) for v in position_counts.values())
    if total <= 0:
        return ""
    opener_n = sum(position_counts.get(pos, 0) for pos in (1, 2))
    top_n = sum(position_counts.get(pos, 0) for pos in (1, 2, 3, 4))
    mid_n = sum(position_counts.get(pos, 0) for pos in (5, 6, 7))
    lower_mid_n = int(position_counts.get(8, 0) or 0)
    lower_n = sum(position_counts.get(pos, 0) for pos in (9, 10, 11))
    if opener_n >= 8 and opener_n / total >= 0.7:
        return "opener"
    if top_n >= 12 and top_n / total >= 0.7:
        return "top_order"
    if mid_n >= 12 and mid_n / total >= 0.7:
        return "middle_order"
    if lower_mid_n >= 6 and lower_mid_n / total >= 0.7:
        return "lower_middle"
    if lower_n >= 6 and lower_n / total >= 0.7:
        return "lower_order"
    return ""


def _batting_band_rank(raw: str) -> int:
    order = {"opener": 0, "top_order": 1, "middle_order": 2, "lower_middle": 3, "lower_order": 4}
    return order.get(str(raw or "").strip().lower(), 99)


def _registry_band_hint(record: dict[str, Any]) -> str:
    band = str(record.get("likely_batting_band") or "").strip().lower()
    if band == "middle":
        return "middle_order"
    if band == "finisher":
        return "lower_middle"
    if band == "tail":
        return "lower_order"
    return band


def _profile_role_fit(record: dict[str, Any], profile: dict[str, Any]) -> float:
    counts = dict(profile.get("role_counts") or {})
    total = float(sum(int(v or 0) for v in counts.values()) or 0.0)
    if total <= 0:
        return 0.0
    batter_share = (float(counts.get("Batter", 0) or 0.0) + float(counts.get("WK-Batter", 0) or 0.0)) / total
    bowler_share = float(counts.get("Bowler", 0) or 0.0) / total
    ar_share = float(counts.get("All-Rounder", 0) or 0.0) / total
    primary = str(record.get("primary_role") or "").strip().lower()
    secondary = str(record.get("secondary_role") or "").strip().lower()
    if primary in ("batter", "wk_batter", "wicketkeeper_batter"):
        if batter_share >= 0.65:
            return 0.12
        if batter_share >= 0.45:
            return 0.06
        if bowler_share >= 0.65:
            return -0.12
        return -0.04
    if primary == "bowler":
        if bowler_share >= 0.65:
            return 0.12
        if bowler_share >= 0.45:
            return 0.06
        if batter_share >= 0.65:
            return -0.12
        return -0.04
    if primary == "all_rounder":
        if secondary in ("batting_allrounder", "batter", "wk_batter", "wicketkeeper_batter") or _registry_band_hint(record) in (
            "opener",
            "top_order",
        ):
            if batter_share >= 0.35 or (batter_share + ar_share) >= 0.75:
                return 0.08
            if bowler_share >= 0.75 and batter_share <= 0.12:
                return -0.08
        if secondary in ("bowling_allrounder", "bowler"):
            if bowler_share >= 0.45 or (bowler_share + ar_share) >= 0.75:
                return 0.08
        if ar_share >= 0.25 or (batter_share >= 0.2 and bowler_share >= 0.2):
            return 0.04
    return 0.0


def _profile_batting_band_fit(record: dict[str, Any], profile: dict[str, Any]) -> float:
    registry_band = _registry_band_hint(record)
    if not registry_band:
        return 0.0
    inferred_band = _batting_band_from_positions(dict(profile.get("batting_positions") or {}))
    if not inferred_band:
        return 0.0
    if inferred_band == registry_band:
        return 0.1
    if abs(_batting_band_rank(inferred_band) - _batting_band_rank(registry_band)) <= 1:
        return 0.04
    return -0.08


def _profile_team_fit(record: dict[str, Any], profile: dict[str, Any], *, latest_ipl_year: int) -> float:
    team_label = ipl_teams.canonical_franchise_label(str(record.get("team") or ""))
    if not team_label:
        return 0.0
    team_key = ipl_teams.canonical_team_key_for_franchise(team_label)
    if not team_key:
        return 0.0
    team_entry = dict(profile.get("teams") or {}).get(team_key) or {}
    if not team_entry:
        return 0.0
    matches = int(team_entry.get("matches") or 0)
    latest_date = str(team_entry.get("latest_date") or "")
    latest_year = int(latest_date[:4]) if latest_date[:4].isdigit() else 0
    is_recent = bool(latest_year and latest_ipl_year and latest_ipl_year - latest_year <= 1)
    if is_recent and matches >= 5:
        return 0.2
    if is_recent and matches >= 2:
        return 0.18
    if is_recent:
        return 0.14
    if matches >= 8:
        return 0.08
    if matches >= 3:
        return 0.05
    return 0.02


def _infer_bowling_phase_hint(profile: dict[str, Any]) -> str:
    phase_balls = {str(k): float(v or 0.0) for k, v in dict(profile.get("phase_balls") or {}).items()}
    total = sum(phase_balls.values())
    if total < 120:
        return ""
    ordered = sorted(
        ((phase, balls / total) for phase, balls in phase_balls.items() if phase in ("powerplay", "middle", "death")),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ordered:
        return ""
    top_phase, top_share = ordered[0]
    if top_share >= 0.72:
        return top_phase
    if len(ordered) >= 2 and top_share >= 0.45 and ordered[1][1] >= 0.25:
        return "multiple"
    return ""


def _blank_history_profile(player_key: str) -> dict[str, Any]:
    return {
        "player_key": player_key,
        "team_match_xi_rows": 0,
        "player_match_stats_rows": 0,
        "player_batting_positions_rows": 0,
        "player_phase_usage_rows": 0,
        "global_distinct_matches": 0,
        "latest_ipl_date": "",
        "names": defaultdict(int),
        "teams": {},
        "role_counts": defaultdict(int),
        "batting_positions": defaultdict(int),
        "phase_balls": defaultdict(float),
    }


def _load_ipl_history_evidence() -> dict[str, Any]:
    db_path = _db_path()
    team_key_to_label = {
        ipl_teams.canonical_team_key_for_franchise(label): label for _, label in ipl_teams.IPL_TEAMS
    }
    if not db_path.is_file():
        return {
            "db_path": str(db_path),
            "profiles": {},
            "name_to_keys": {},
            "surname_to_name_keys": {},
            "claimed_global_keys": frozenset(),
            "team_key_to_label": team_key_to_label,
            "latest_ipl_date": "",
        }
    placeholders = ",".join("?" for _ in team_key_to_label)
    params = tuple(team_key_to_label)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        profiles: dict[str, dict[str, Any]] = {}
        name_to_keys: dict[str, set[str]] = defaultdict(set)
        surname_to_name_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)

        def ensure_profile(raw_key: str) -> dict[str, Any]:
            pk = _normalize(raw_key)
            if not pk:
                raise ValueError("player_key must be non-empty")
            prof = profiles.get(pk)
            if prof is None:
                prof = _blank_history_profile(pk)
                profiles[pk] = prof
            return prof

        for row in conn.execute(
            f"""
            SELECT player_key, player_name, canonical_team_key, COUNT(*) AS row_count
            FROM team_match_xi
            WHERE trim(player_key) != '' AND canonical_team_key IN ({placeholders})
            GROUP BY player_key, player_name, canonical_team_key
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            prof["team_match_xi_rows"] += int(row["row_count"] or 0)
            player_name = _normalize(str(row["player_name"] or ""))
            if player_name:
                prof["names"][player_name] += int(row["row_count"] or 0)
                name_to_keys[player_name].add(prof["player_key"])
                surname_to_name_keys[_surname_signature(player_name)].add((player_name, prof["player_key"]))

        for row in conn.execute(
            f"""
            SELECT t.player_key, t.canonical_team_key, COUNT(DISTINCT t.match_id) AS match_count,
                   MAX(COALESCE(m.match_date, '')) AS latest_date
            FROM team_match_xi t
            LEFT JOIN match_results m ON m.id = t.match_id
            WHERE trim(t.player_key) != '' AND t.canonical_team_key IN ({placeholders})
            GROUP BY t.player_key, t.canonical_team_key
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            team_key = str(row["canonical_team_key"] or "").strip()
            team_entry = prof["teams"].setdefault(team_key, {"matches": 0, "latest_date": ""})
            team_entry["matches"] += int(row["match_count"] or 0)
            latest = str(row["latest_date"] or "").strip()
            if latest and latest > str(team_entry.get("latest_date") or ""):
                team_entry["latest_date"] = latest

        for row in conn.execute(
            f"""
            SELECT t.player_key, COUNT(DISTINCT t.match_id) AS match_count,
                   MAX(COALESCE(m.match_date, '')) AS latest_date
            FROM team_match_xi t
            LEFT JOIN match_results m ON m.id = t.match_id
            WHERE trim(t.player_key) != '' AND t.canonical_team_key IN ({placeholders})
            GROUP BY t.player_key
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            prof["global_distinct_matches"] = int(row["match_count"] or 0)
            prof["latest_ipl_date"] = str(row["latest_date"] or "").strip()

        for row in conn.execute(
            f"""
            SELECT player_key, role_bucket, COUNT(*) AS row_count
            FROM team_match_xi
            WHERE trim(player_key) != '' AND canonical_team_key IN ({placeholders})
            GROUP BY player_key, role_bucket
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            prof["role_counts"][str(row["role_bucket"] or "").strip()] += int(row["row_count"] or 0)

        for row in conn.execute(
            f"""
            SELECT player_key, COUNT(*) AS row_count
            FROM player_match_stats
            WHERE trim(player_key) != '' AND canonical_team_key IN ({placeholders})
            GROUP BY player_key
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            prof["player_match_stats_rows"] = int(row["row_count"] or 0)

        for row in conn.execute(
            f"""
            SELECT player_key, batting_position, COUNT(*) AS row_count
            FROM player_batting_positions
            WHERE trim(player_key) != '' AND canonical_team_key IN ({placeholders}) AND batting_position IS NOT NULL AND batting_position > 0
            GROUP BY player_key, batting_position
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            pos = int(float(row["batting_position"] or 0))
            c = int(row["row_count"] or 0)
            prof["player_batting_positions_rows"] += c
            if pos > 0:
                prof["batting_positions"][pos] += c

        for row in conn.execute(
            f"""
            SELECT player_key, phase, SUM(balls) AS ball_count, COUNT(*) AS row_count
            FROM player_phase_usage
            WHERE trim(player_key) != '' AND canonical_team_key IN ({placeholders}) AND role = 'bowl'
            GROUP BY player_key, phase
            """,
            params,
        ).fetchall():
            prof = ensure_profile(str(row["player_key"] or ""))
            phase = str(row["phase"] or "").strip().lower()
            prof["player_phase_usage_rows"] += int(row["row_count"] or 0)
            if phase in ("powerplay", "middle", "death"):
                prof["phase_balls"][phase] += float(row["ball_count"] or 0.0)

        latest_ipl_date = ""
        for prof in profiles.values():
            if str(prof.get("latest_ipl_date") or "") > latest_ipl_date:
                latest_ipl_date = str(prof.get("latest_ipl_date") or "")
        return {
            "db_path": str(db_path),
            "profiles": profiles,
            "name_to_keys": {k: set(v) for k, v in name_to_keys.items()},
            "surname_to_name_keys": {k: set(v) for k, v in surname_to_name_keys.items()},
            "claimed_global_keys": frozenset(profiles.keys()),
            "team_key_to_label": team_key_to_label,
            "latest_ipl_date": latest_ipl_date,
        }
    finally:
        conn.close()


def _load_raw_cricsheet_name_evidence() -> dict[str, Any]:
    from cricsheet_readme import filter_last_n_seasons, load_readme_rows, resolve_readme_path

    readme_path = resolve_readme_path()
    json_dir = Path(getattr(config, "CRICSHEET_JSON_DIR", config.DATA_DIR / "ipl_json"))
    if readme_path is None or not json_dir.is_dir():
        return {
            "readme_path": str(readme_path or ""),
            "json_dir": str(json_dir),
            "latest_season_year": 0,
            "recent_years": [],
            "rows_indexed": 0,
            "files_scanned": 0,
            "names": {},
            "surname_to_names": {},
        }

    rows = load_readme_rows(readme_path, competition="IPL", genders={"male"})
    if not rows:
        return {
            "readme_path": str(readme_path),
            "json_dir": str(json_dir),
            "latest_season_year": 0,
            "recent_years": [],
            "rows_indexed": 0,
            "files_scanned": 0,
            "names": {},
            "surname_to_names": {},
        }

    latest_season_year = max(row.season_year for row in rows)
    recent_rows = filter_last_n_seasons(
        rows,
        current_season_year=latest_season_year,
        n_seasons=int(getattr(config, "CRICSHEET_HISTORY_SEASON_COUNT", 5) or 5),
    )
    recent_match_ids = {str(row.match_id) for row in recent_rows}
    names: dict[str, dict[str, Any]] = {}
    surname_to_names: dict[str, set[str]] = defaultdict(set)
    files_scanned = 0
    rows_indexed = 0

    def ensure_entry(raw_name: str) -> dict[str, Any]:
        nk = _normalize(raw_name)
        entry = names.get(nk)
        if entry is None:
            entry = {
                "normalized_name": nk,
                "display_names": set(),
                "person_ids": set(),
                "teams": defaultdict(int),
                "recent_teams": defaultdict(int),
                "match_count": 0,
                "recent_match_count": 0,
                "matched_files": [],
                "latest_date": "",
            }
            names[nk] = entry
            surname_to_names[_surname_signature(nk)].add(nk)
        return entry

    for row in rows:
        match_id = str(row.match_id)
        path = json_dir / f"{match_id}.json"
        if not path.is_file():
            continue
        rows_indexed += 1
        files_scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        info = payload.get("info") if isinstance(payload, dict) else {}
        if not isinstance(info, dict):
            continue
        players_by_team = info.get("players")
        if not isinstance(players_by_team, dict):
            continue
        people_map = {}
        registry_payload = info.get("registry")
        if isinstance(registry_payload, dict) and isinstance(registry_payload.get("people"), dict):
            people_map = dict(registry_payload.get("people") or {})
        is_recent = match_id in recent_match_ids
        for team_name, raw_players in players_by_team.items():
            if not isinstance(raw_players, list):
                continue
            team_label = (
                ipl_teams.canonical_franchise_label_from_history_name(str(team_name or ""))
                or ipl_teams.canonical_franchise_label(str(team_name or ""))
                or str(team_name or "").strip()
            )
            seen_in_match: set[str] = set()
            for raw_name in raw_players:
                nk = _normalize(str(raw_name or ""))
                if not nk or nk in seen_in_match:
                    continue
                seen_in_match.add(nk)
                entry = ensure_entry(nk)
                entry["display_names"].add(str(raw_name or "").strip())
                person_id = str(people_map.get(raw_name) or "").strip()
                if person_id:
                    entry["person_ids"].add(person_id)
                entry["teams"][team_label] += 1
                entry["match_count"] += 1
                if is_recent:
                    entry["recent_teams"][team_label] += 1
                    entry["recent_match_count"] += 1
                if path.name not in entry["matched_files"] and len(entry["matched_files"]) < 12:
                    entry["matched_files"].append(path.name)
                if str(row.match_date) > str(entry.get("latest_date") or ""):
                    entry["latest_date"] = str(row.match_date)

    serializable_names: dict[str, dict[str, Any]] = {}
    for nk, entry in names.items():
        serializable_names[nk] = {
            "display_names": sorted(entry["display_names"]),
            "person_ids": sorted(entry["person_ids"]),
            "teams": dict(entry["teams"]),
            "recent_teams": dict(entry["recent_teams"]),
            "match_count": int(entry["match_count"] or 0),
            "recent_match_count": int(entry["recent_match_count"] or 0),
            "matched_files": list(entry["matched_files"]),
            "latest_date": str(entry.get("latest_date") or ""),
        }

    return {
        "readme_path": str(readme_path),
        "json_dir": str(json_dir),
        "latest_season_year": int(latest_season_year or 0),
        "recent_years": sorted({row.season_year for row in recent_rows}),
        "rows_indexed": rows_indexed,
        "files_scanned": files_scanned,
        "names": serializable_names,
        "surname_to_names": {k: sorted(v) for k, v in surname_to_names.items()},
    }


def _suggest_team_label(profile: dict[str, Any], team_key_to_label: dict[str, str]) -> tuple[str, str]:
    teams = [
        {
            "team_key": team_key,
            "matches": int((payload or {}).get("matches") or 0),
            "latest_date": str((payload or {}).get("latest_date") or ""),
        }
        for team_key, payload in dict(profile.get("teams") or {}).items()
        if team_key in team_key_to_label
    ]
    if not teams:
        return "", "no_ipl_team_history"
    if len(teams) == 1:
        only = teams[0]
        return team_key_to_label.get(str(only["team_key"]) or "", ""), "single_ipl_team"
    teams.sort(key=lambda row: (row["latest_date"], row["matches"]), reverse=True)
    top = teams[0]
    nxt = teams[1]
    if top["latest_date"] and nxt["latest_date"] and top["latest_date"][:4].isdigit() and nxt["latest_date"][:4].isdigit():
        if int(top["latest_date"][:4]) - int(nxt["latest_date"][:4]) >= 2:
            return team_key_to_label.get(str(top["team_key"]) or "", ""), "most_recent_team_clear"
    if top["matches"] >= max(6, 2 * max(1, int(nxt["matches"]))):
        return team_key_to_label.get(str(top["team_key"]) or "", ""), "dominant_team_match_count"
    return "", "team_ambiguous"


def _safe_alias_additions(
    record: dict[str, Any],
    chosen_history_key: str,
    profile: dict[str, Any],
    name_to_keys: dict[str, set[str]],
    registry_keys: set[str],
) -> list[str]:
    out: list[str] = []
    base_names = {
        _normalize(str(record.get("registry_key") or "")),
        _normalize(str(record.get("display_name") or "")),
        _normalize(str(record.get("canonical_name") or "")),
    }
    for alias in list(record.get("aliases") or []):
        base_names.add(_normalize(alias))
    for alias, count in sorted((profile.get("names") or {}).items(), key=lambda kv: (-int(kv[1] or 0), kv[0])):
        nk = _normalize(alias)
        if not nk or nk in base_names or nk in registry_keys:
            continue
        owners = name_to_keys.get(nk) or set()
        if owners != {chosen_history_key}:
            continue
        if count < 2 and len(nk.split()) <= 2:
            continue
        out.append(nk)
    return _dedupe_keep_order(out)


def _collect_raw_alias_candidates(
    record: dict[str, Any],
    raw_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    names = dict(raw_evidence.get("names") or {})
    surname_to_names = {
        str(k): set(v or []) for k, v in dict(raw_evidence.get("surname_to_names") or {}).items()
    }
    if not names:
        return []
    current_team = ipl_teams.canonical_franchise_label(str(record.get("team") or ""))
    expected_names = _dedupe_keep_order(
        [
            str(record.get("registry_key") or ""),
            str(record.get("display_name") or ""),
            str(record.get("canonical_name") or ""),
            *list(record.get("aliases") or []),
        ]
    )
    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(raw_name: str, *, mode: str, score: float, note: str) -> None:
        nk = _normalize(raw_name)
        entry = names.get(nk) or {}
        if not entry:
            return
        payload = candidates.setdefault(
            nk,
            {
                "raw_alias": nk,
                "score": 0.0,
                "modes": set(),
                "notes": [],
                "matched_files": list(entry.get("matched_files") or []),
                "match_count": int(entry.get("match_count") or 0),
                "recent_match_count": int(entry.get("recent_match_count") or 0),
                "team_match_count": 0,
                "recent_team_match_count": 0,
                "person_id_count": len(list(entry.get("person_ids") or [])),
            },
        )
        payload["score"] = max(float(payload.get("score") or 0.0), float(score))
        payload["modes"].add(mode)
        payload["notes"].append(note)
        if current_team:
            payload["team_match_count"] = max(
                int(payload.get("team_match_count") or 0),
                int((entry.get("teams") or {}).get(current_team, 0) or 0),
            )
            payload["recent_team_match_count"] = max(
                int(payload.get("recent_team_match_count") or 0),
                int((entry.get("recent_teams") or {}).get(current_team, 0) or 0),
            )

    for expected in expected_names:
        nk = _normalize(expected)
        if not nk:
            continue
        if nk in names:
            add_candidate(nk, mode="exact_raw_name", score=0.98, note=f"exact raw match {nk}")
        surname_sig = _surname_signature(nk)
        for observed_name in surname_to_names.get(surname_sig) or set():
            if observed_name == nk:
                continue
            if _safe_full_name_variant_match(nk, observed_name):
                add_candidate(
                    observed_name,
                    mode="safe_raw_variant",
                    score=0.94,
                    note=f"safe raw name variant {observed_name}",
                )
            elif _safe_surname_variant_initial_match(nk, observed_name):
                add_candidate(
                    observed_name,
                    mode="safe_raw_surname_variant",
                    score=0.9,
                    note=f"safe raw surname variant {observed_name}",
                )
            elif _initials_pool_candidate_match(nk, observed_name):
                add_candidate(
                    observed_name,
                    mode="raw_initials_pool",
                    score=0.56,
                    note=f"raw initials candidate {observed_name}",
                )
        for observed_name in names:
            if observed_name == nk:
                continue
            if _safe_surname_variant_initial_match(nk, observed_name):
                add_candidate(
                    observed_name,
                    mode="safe_raw_surname_variant",
                    score=0.9,
                    note=f"safe raw surname variant {observed_name}",
                )

    ranked: list[dict[str, Any]] = []
    for raw_alias, payload in candidates.items():
        score = float(payload.get("score") or 0.0)
        recent_team_match_count = int(payload.get("recent_team_match_count") or 0)
        team_match_count = int(payload.get("team_match_count") or 0)
        recent_match_count = int(payload.get("recent_match_count") or 0)
        match_count = int(payload.get("match_count") or 0)
        person_id_count = int(payload.get("person_id_count") or 0)
        if recent_team_match_count >= 2:
            score += 0.25
        elif recent_team_match_count == 1:
            score += 0.18
        elif team_match_count >= 2:
            score += 0.1
        elif team_match_count == 1:
            score += 0.06
        if recent_match_count >= 5:
            score += 0.1
        elif recent_match_count >= 2:
            score += 0.06
        elif match_count >= 3:
            score += 0.04
        if person_id_count == 1:
            score += 0.04
        payload["score"] = score
        ranked.append(payload)

    ranked.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            int(item.get("recent_team_match_count") or 0),
            int(item.get("recent_match_count") or 0),
            int(item.get("match_count") or 0),
        ),
        reverse=True,
    )
    return ranked


def _apply_raw_cricsheet_linkage_fallback(
    records: dict[str, dict[str, Any]],
    audit_payload: dict[str, Any],
    stats: dict[str, Any],
    db_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_evidence = _load_raw_cricsheet_name_evidence()
    audit_players = audit_payload.setdefault("players", OrderedDict())
    raw_names = dict(raw_evidence.get("names") or {})
    profiles = dict(db_evidence.get("profiles") or {})
    name_to_keys = {k: set(v or []) for k, v in dict(db_evidence.get("name_to_keys") or {}).items()}
    registry_keys = set(records)
    existing_claims: dict[str, str] = {}
    existing_lookup_owners: dict[str, str] = {}
    for registry_key, record in records.items():
        hk = _normalize(str(record.get("history_canonical_key") or ""))
        if hk:
            existing_claims.setdefault(hk, registry_key)
        for lookup_key in _record_lookup_keys(record):
            existing_lookup_owners.setdefault(lookup_key, registry_key)

    proposals: dict[str, dict[str, Any]] = {}
    alias_claims: dict[str, set[str]] = defaultdict(set)
    for registry_key, record in records.items():
        current_history_key = _normalize(str(record.get("history_canonical_key") or ""))
        current_aliases = set(_dedupe_keep_order(list(record.get("aliases") or [])))
        needs_history = not current_history_key
        needs_aliases = bool(current_history_key and not current_aliases)
        if not (needs_history or needs_aliases):
            continue
        ranked_aliases = _collect_raw_alias_candidates(record, raw_evidence)
        if not ranked_aliases:
            audit_rec = audit_players.setdefault(registry_key, {})
            if not current_history_key:
                audit_rec["reason_if_left_blank"] = "no_safe_raw_match"
            audit_rec["raw_search_reason"] = "no_safe_raw_match"
            audit_rec["matched_files"] = []
            audit_rec["evidence_source"] = str(audit_rec.get("evidence_source") or "")
            continue

        top = ranked_aliases[0]
        runner = ranked_aliases[1] if len(ranked_aliases) > 1 else None
        accepted_aliases: list[str] = []
        matched_files: list[str] = []
        notes: list[str] = []
        raw_candidate_keys: set[str] = set()
        for payload in ranked_aliases[:3]:
            raw_alias = _normalize(str(payload.get("raw_alias") or ""))
            modes = set(payload.get("modes") or set())
            score = float(payload.get("score") or 0.0)
            gap = score - float(runner.get("score") or 0.0) if runner and payload is top else score
            recent_team_match_count = int(payload.get("recent_team_match_count") or 0)
            recent_match_count = int(payload.get("recent_match_count") or 0)
            team_match_count = int(payload.get("team_match_count") or 0)
            acceptable = False
            if "exact_raw_name" in modes and score >= 0.9:
                acceptable = True
            elif "safe_raw_variant" in modes and score >= 0.88:
                acceptable = True
            elif (
                "raw_initials_pool" in modes
                and score >= 0.82
                and (recent_team_match_count >= 1 or team_match_count >= 2 or recent_match_count >= 2)
                and gap >= 0.08
            ):
                acceptable = True
            elif "safe_raw_surname_variant" in modes and score >= 0.86:
                acceptable = True
            if not acceptable:
                continue
            accepted_aliases.append(raw_alias)
            matched_files.extend(list(payload.get("matched_files") or []))
            notes.extend(list(payload.get("notes") or []))
            if raw_alias in profiles:
                raw_candidate_keys.add(raw_alias)
            raw_candidate_keys.update(name_to_keys.get(raw_alias) or set())

        accepted_aliases = _dedupe_keep_order(accepted_aliases)
        matched_files = _dedupe_exact_keep_order(matched_files)

        proposed_history_key = current_history_key
        inferred = False
        unresolved_collision = False
        raw_reason = "no_safe_raw_match"
        if needs_history and accepted_aliases:
            temp_record = dict(record)
            temp_record["aliases"] = _dedupe_keep_order(list(record.get("aliases") or []) + accepted_aliases)
            decision = _infer_history_key_for_record(temp_record, db_evidence, freeze_existing=False)
            proposed_history_key = _normalize(str(decision.get("history_key") or ""))
            inferred = bool(decision.get("auto_inferred") and proposed_history_key)
            notes.extend(list(decision.get("candidate_notes") or []))
            if proposed_history_key and raw_candidate_keys and proposed_history_key in raw_candidate_keys:
                prior_owner = existing_claims.get(proposed_history_key)
                if prior_owner and prior_owner != registry_key:
                    proposed_history_key = ""
                    unresolved_collision = True
                    raw_reason = "collision_risk"
                else:
                    existing_claims[proposed_history_key] = registry_key
                    raw_reason = "raw_alias_supported_history_key"
            else:
                proposed_history_key = ""
                raw_reason = "ambiguous_alias"
        elif current_history_key and accepted_aliases:
            supported = []
            for alias in accepted_aliases:
                alias_keys = set()
                if alias in profiles:
                    alias_keys.add(alias)
                alias_keys.update(name_to_keys.get(alias) or set())
                if current_history_key in alias_keys:
                    supported.append(alias)
            accepted_aliases = _dedupe_keep_order(supported)
            if accepted_aliases:
                raw_reason = "raw_alias_supported_existing_key"
            else:
                raw_reason = "ambiguous_alias"

        proposals[registry_key] = {
            "accepted_aliases": accepted_aliases,
            "matched_files": matched_files,
            "notes": _dedupe_keep_order(notes),
            "proposed_history_key": proposed_history_key,
            "inferred": inferred,
            "unresolved_collision": unresolved_collision,
            "raw_reason": raw_reason,
        }
        for alias in accepted_aliases:
            alias_claims[alias].add(registry_key)

    alias_collisions = {alias for alias, owners in alias_claims.items() if len(owners) > 1}

    for registry_key, proposal in proposals.items():
        record = records[registry_key]
        audit_rec = audit_players.setdefault(registry_key, {})
        current_history_key = _normalize(str(record.get("history_canonical_key") or ""))
        accepted_aliases = [
            alias
            for alias in proposal.get("accepted_aliases") or []
            if alias not in alias_collisions and alias not in set(_dedupe_keep_order(list(record.get("aliases") or [])))
        ]
        accepted_aliases = [
            alias
            for alias in accepted_aliases
            if existing_lookup_owners.get(alias, registry_key) == registry_key
        ]
        proposed_history_key = _normalize(str(proposal.get("proposed_history_key") or ""))
        raw_reason = str(proposal.get("raw_reason") or "")
        raw_enriched = False
        already_enriched = bool(
            current_history_key
            or list(audit_rec.get("alias_additions") or [])
            or bool(audit_rec.get("team_filled"))
            or (record.get("field_sources") or {}).get("likely_batting_band") == "db_linkage_enrichment"
            or (record.get("field_sources") or {}).get("likely_bowling_phases") == "db_linkage_enrichment"
        )

        if proposed_history_key and not current_history_key:
            _append_source(record, "raw_cricsheet_fallback")
            summary = record.setdefault("metadata_source_summary", {})
            summary["history_canonical_source"] = "raw_cricsheet_fallback"
            summary["linkage_enrichment_applied"] = True
            _set_field(record, "history_canonical_key", proposed_history_key, "raw_cricsheet_fallback")
            current_history_key = proposed_history_key
            raw_enriched = True
            if stats["players_left_blank"] > 0:
                stats["players_left_blank"] -= 1

        if not current_history_key:
            accepted_aliases = []
        if proposal.get("unresolved_collision") and not current_history_key:
            accepted_aliases = []

        if accepted_aliases:
            _append_source(record, "raw_cricsheet_fallback")
            summary = record.setdefault("metadata_source_summary", {})
            summary["linkage_enrichment_applied"] = True
            merged_aliases = list(record.get("aliases") or [])
            merged_aliases.extend(accepted_aliases)
            record["aliases"] = _dedupe_keep_order(merged_aliases)
            record.setdefault("field_sources", {})["aliases"] = "raw_cricsheet_fallback"
            stats["aliases_added"] += len(accepted_aliases)
            raw_enriched = True

        prof = profiles.get(current_history_key) or {}
        if current_history_key and not str(record.get("likely_batting_band") or "").strip():
            inferred_band = _batting_band_from_positions(dict(prof.get("batting_positions") or {}))
            if inferred_band:
                _append_source(record, "raw_cricsheet_fallback")
                _set_field(record, "likely_batting_band", inferred_band, "raw_cricsheet_fallback")
                raw_enriched = True
        if current_history_key and not str(record.get("likely_bowling_phases") or "").strip():
            inferred_phase = _infer_bowling_phase_hint(prof)
            if inferred_phase:
                _append_source(record, "raw_cricsheet_fallback")
                _set_field(record, "likely_bowling_phases", inferred_phase, "raw_cricsheet_fallback")
                raw_enriched = True

        if raw_enriched and not already_enriched:
            stats["players_enriched"] += 1
        if proposal.get("unresolved_collision") or any(alias in alias_collisions for alias in proposal.get("accepted_aliases") or []):
            stats["unresolved_collisions"] += 1

        old_history_key = _normalize(str(audit_rec.get("old_history_canonical_key") or ""))
        audit_rec["old_history_canonical_key"] = old_history_key
        audit_rec["new_history_canonical_key"] = current_history_key
        audit_rec["inferred"] = bool(proposal.get("inferred") and current_history_key)
        existing_evidence = list(audit_rec.get("evidence_summary") or [])
        existing_evidence.extend(list(proposal.get("notes") or []))
        audit_rec["evidence_summary"] = _dedupe_keep_order(existing_evidence)
        existing_alias_additions = list(audit_rec.get("alias_additions") or [])
        existing_alias_additions.extend(accepted_aliases)
        audit_rec["alias_additions"] = _dedupe_keep_order(existing_alias_additions)
        audit_rec["confidence"] = max(float(audit_rec.get("confidence") or 0.0), 0.86 if current_history_key and accepted_aliases else float(audit_rec.get("confidence") or 0.0))
        audit_rec["evidence_source"] = "raw_cricsheet_fallback" if raw_enriched else str(audit_rec.get("evidence_source") or "")
        audit_rec["matched_files"] = _dedupe_exact_keep_order(list(audit_rec.get("matched_files") or []) + list(proposal.get("matched_files") or []))
        audit_rec["unresolved_collision"] = bool(
            audit_rec.get("unresolved_collision")
            or proposal.get("unresolved_collision")
            or any(alias in alias_collisions for alias in proposal.get("accepted_aliases") or [])
            or any(existing_lookup_owners.get(alias, registry_key) != registry_key for alias in proposal.get("accepted_aliases") or [])
        )
        if not current_history_key:
            if raw_reason == "no_safe_raw_match" and str(record.get("team") or "").strip():
                raw_reason = "likely_new_player"
            audit_rec["reason_if_left_blank"] = raw_reason
        else:
            audit_rec["reason_if_left_blank"] = ""

    audit_payload["raw_cricsheet_search_summary"] = {
        "readme_path": str(raw_evidence.get("readme_path") or ""),
        "json_dir": str(raw_evidence.get("json_dir") or ""),
        "latest_season_year": int(raw_evidence.get("latest_season_year") or 0),
        "recent_years": list(raw_evidence.get("recent_years") or []),
        "rows_indexed": int(raw_evidence.get("rows_indexed") or 0),
        "files_scanned": int(raw_evidence.get("files_scanned") or 0),
    }
    audit_payload["summary"] = dict(stats)
    return audit_payload, stats


def _infer_history_key_for_record(
    record: dict[str, Any],
    evidence: dict[str, Any],
    *,
    freeze_existing: bool = False,
) -> dict[str, Any]:
    profiles = dict(evidence.get("profiles") or {})
    name_to_keys = dict(evidence.get("name_to_keys") or {})
    surname_to_name_keys = dict(evidence.get("surname_to_name_keys") or {})
    latest_ipl_date = str(evidence.get("latest_ipl_date") or "")
    latest_year = int(latest_ipl_date[:4]) if latest_ipl_date[:4].isdigit() else 0
    registry_key = _normalize(str(record.get("registry_key") or ""))
    display_name = str(record.get("display_name") or record.get("canonical_name") or registry_key)
    existing_key = _normalize(str(record.get("history_canonical_key") or ""))
    if freeze_existing:
        return {
            "history_key": existing_key,
            "reason": "preserved_previous_registry",
            "confidence": 1.0 if existing_key else 0.0,
            "candidate_notes": [f"preserved previous registry history key {existing_key or '(blank)'}"],
            "auto_inferred": False,
        }
    if existing_key and existing_key in profiles:
        return {
            "history_key": existing_key,
            "reason": "existing_registry_key_kept",
            "confidence": 1.0,
            "candidate_notes": [f"preserved existing history key {existing_key}"],
            "auto_inferred": False,
        }

    candidate_names = _dedupe_keep_order(
        [
            registry_key,
            str(record.get("display_name") or ""),
            str(record.get("canonical_name") or ""),
            *list(record.get("aliases") or []),
        ]
    )
    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(history_key: str, *, source: str, score: float, note: str) -> None:
        hk = _normalize(history_key)
        if not hk or hk not in profiles:
            return
        payload = candidates.setdefault(hk, {"score": 0.0, "sources": [], "notes": []})
        payload["score"] = max(float(payload.get("score") or 0.0), float(score))
        payload["sources"].append(source)
        payload["notes"].append(note)

    for cand_name in candidate_names:
        nk = _normalize(cand_name)
        if not nk:
            continue
        if nk in profiles:
            add_candidate(nk, source="exact_player_key", score=0.98, note=f"exact IPL player_key match for {nk}")
        for hk in sorted(name_to_keys.get(nk) or set()):
            add_candidate(hk, source="exact_player_name", score=0.96, note=f"exact scorecard name match {nk}")

    for cand_name in candidate_names:
        nk = _normalize(cand_name)
        if not nk:
            continue
        surname_sig = _surname_signature(nk)
        for observed_name, history_key in sorted(surname_to_name_keys.get(surname_sig) or set()):
            if _safe_full_name_variant_match(nk, observed_name):
                prof = profiles.get(history_key) or {}
                if int(prof.get("global_distinct_matches") or 0) >= 3:
                    add_candidate(
                        history_key,
                        source="safe_name_variant",
                        score=0.93,
                        note=f"safe one-edit scorecard variant {observed_name}",
                    )

    for cand_name in candidate_names:
        nk = _normalize(cand_name)
        if not nk:
            continue
        surname_sig = _surname_signature(nk)
        for observed_name, history_key in sorted(surname_to_name_keys.get(surname_sig) or set()):
            if _initials_pool_candidate_match(nk, observed_name):
                add_candidate(
                    history_key,
                    source="surname_initial_pool",
                    score=0.52,
                    note=f"same-surname initials pool candidate {observed_name}",
                )

    if not candidates:
        return {
            "history_key": "",
            "reason": "no_safe_ipl_candidate",
            "confidence": 0.0,
            "candidate_notes": ["no IPL history candidate met safe-link criteria"],
            "auto_inferred": False,
        }

    ranked: list[dict[str, Any]] = []
    for hk, payload in candidates.items():
        prof = profiles.get(hk) or {}
        score = float(payload.get("score") or 0.0)
        matches = int(prof.get("global_distinct_matches") or 0)
        latest_date = str(prof.get("latest_ipl_date") or "")
        role_fit = _profile_role_fit(record, prof)
        band_fit = _profile_batting_band_fit(record, prof)
        team_fit = _profile_team_fit(record, prof, latest_ipl_year=latest_year)
        latest_score = 0.0
        if latest_date[:4].isdigit() and latest_year:
            year_gap = latest_year - int(latest_date[:4])
            if year_gap <= 0:
                latest_score = 0.16
            elif year_gap == 1:
                latest_score = 0.12
            elif year_gap == 2:
                latest_score = 0.06
            else:
                latest_score = -0.08
        score += latest_score
        if matches >= 20:
            score += 0.12
        elif matches >= 8:
            score += 0.08
        elif matches >= 5:
            score += 0.05
        else:
            score -= 0.04
        score += role_fit
        score += band_fit
        score += team_fit
        ranked.append(
            {
                "history_key": hk,
                "score": score,
                "matches": matches,
                "latest_date": latest_date,
                "fit_score": role_fit + band_fit + team_fit,
                "sources": list(payload.get("sources") or []),
                "notes": list(payload.get("notes") or []),
            }
        )
    ranked.sort(key=lambda row: (row["score"], row["matches"], row["latest_date"]), reverse=True)
    top = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    top_sources = set(str(src or "") for src in top.get("sources") or [])
    strong_source = bool(top_sources & {"exact_player_key", "exact_player_name", "safe_name_variant"})
    if strong_source and (runner is None or float(top["score"]) - float(runner["score"]) >= 0.08):
        return {
            "history_key": str(top["history_key"] or ""),
            "reason": ",".join(sorted(top_sources)),
            "confidence": min(0.99, max(0.84, float(top["score"]))),
            "candidate_notes": list(top.get("notes") or []),
            "auto_inferred": True,
        }
    top_year = int(str(top.get("latest_date") or "")[:4]) if str(top.get("latest_date") or "")[:4].isdigit() else 0
    runner_year = int(str(runner.get("latest_date") or "")[:4]) if runner and str(runner.get("latest_date") or "")[:4].isdigit() else 0
    if (
        runner is None
        and "surname_initial_pool" in top_sources
        and (
            int(top["matches"]) >= 10
            or (top_year and latest_year and latest_year - top_year <= 1 and int(top["matches"]) >= 3)
        )
        and float(top.get("fit_score") or 0.0) >= 0.03
    ):
        return {
            "history_key": str(top["history_key"] or ""),
            "reason": "unique_recent_initials_candidate",
            "confidence": min(0.9, max(0.82, float(top["score"]))),
            "candidate_notes": list(top.get("notes") or []),
            "auto_inferred": True,
        }
    if (
        float(top["score"]) >= 0.78
        and int(top["matches"]) >= 5
        and top_year
        and latest_year
        and latest_year - top_year <= 1
        and float(top.get("fit_score") or 0.0) >= 0.14
        and (
            runner is None
            or not runner_year
            or (top_year - runner_year) >= 3
            or float(top["score"]) - float(runner["score"]) >= 0.18
        )
    ):
        return {
            "history_key": str(top["history_key"] or ""),
            "reason": "recent_ipl_dominance",
            "confidence": min(0.94, max(0.8, float(top["score"]))),
            "candidate_notes": list(top.get("notes") or []),
            "auto_inferred": True,
        }
    notes = list(top.get("notes") or [])
    if runner:
        notes.append(
            f"left blank: {top['history_key']} score={top['score']:.2f} vs {runner['history_key']} score={runner['score']:.2f}"
        )
    else:
        notes.append(f"left blank: top candidate {top['history_key']} lacked strong evidence")
    return {
        "history_key": "",
        "reason": "candidate_ambiguous_or_too_weak",
        "confidence": 0.0,
        "candidate_notes": notes,
        "auto_inferred": False,
    }


def _apply_db_linkage_enrichment(
    records: dict[str, dict[str, Any]],
    *,
    frozen_registry_keys: Optional[set[str]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _load_ipl_history_evidence()
    profiles = dict(evidence.get("profiles") or {})
    team_key_to_label = dict(evidence.get("team_key_to_label") or {})
    name_to_keys = dict(evidence.get("name_to_keys") or {})
    registry_keys = set(records)
    audit_players: dict[str, dict[str, Any]] = OrderedDict()
    stats = {
        "players_enriched": 0,
        "players_left_blank": 0,
        "aliases_added": 0,
        "teams_filled": 0,
        "unresolved_collisions": 0,
    }

    existing_claims: dict[str, str] = {}
    for registry_key, record in records.items():
        hk = _normalize(str(record.get("history_canonical_key") or ""))
        if hk:
            existing_claims.setdefault(hk, registry_key)

    tentative: dict[str, dict[str, Any]] = {}
    for registry_key, record in records.items():
        old_history_key = _normalize(str(record.get("history_canonical_key") or ""))
        freeze_existing = bool(frozen_registry_keys and registry_key in frozen_registry_keys)
        decision = _infer_history_key_for_record(record, evidence, freeze_existing=freeze_existing)
        new_history_key = _normalize(str(decision.get("history_key") or ""))
        unresolved_collision = False
        reason_if_left_blank = ""
        if new_history_key and old_history_key != new_history_key:
            prior_owner = existing_claims.get(new_history_key)
            if prior_owner and prior_owner != registry_key:
                new_history_key = ""
                unresolved_collision = True
                reason_if_left_blank = f"history key already claimed by {prior_owner}"
            else:
                existing_claims[new_history_key] = registry_key
        tentative[registry_key] = {
            "old_history_canonical_key": old_history_key,
            "new_history_canonical_key": new_history_key or old_history_key,
            "decision_reason": str(decision.get("reason") or ""),
            "candidate_notes": list(decision.get("candidate_notes") or []),
            "confidence": float(decision.get("confidence") or 0.0),
            "inferred": bool(decision.get("auto_inferred") and new_history_key),
            "unresolved_collision": unresolved_collision,
            "reason_if_left_blank": reason_if_left_blank,
        }

    key_to_registry: dict[str, list[str]] = defaultdict(list)
    for registry_key, payload in tentative.items():
        hk = _normalize(str(payload.get("new_history_canonical_key") or ""))
        if hk:
            key_to_registry[hk].append(registry_key)
    for hk, registry_list in key_to_registry.items():
        if len(registry_list) <= 1:
            continue
        protected = [
            rk for rk in registry_list if _normalize(str(records.get(rk, {}).get("history_canonical_key") or "")) == hk
        ]
        keep = protected[:1]
        for rk in registry_list:
            if rk in keep:
                continue
            tentative[rk]["new_history_canonical_key"] = _normalize(str(records.get(rk, {}).get("history_canonical_key") or ""))
            tentative[rk]["inferred"] = False
            tentative[rk]["unresolved_collision"] = True
            tentative[rk]["reason_if_left_blank"] = f"shared collision on {hk}"

    proposed_aliases: dict[str, list[str]] = {}
    alias_claims: dict[str, set[str]] = defaultdict(set)
    for registry_key, record in records.items():
        final_history_key = _normalize(str(tentative[registry_key].get("new_history_canonical_key") or ""))
        prof = profiles.get(final_history_key) or {}
        alias_additions = _safe_alias_additions(record, final_history_key, prof, name_to_keys, registry_keys) if final_history_key else []
        proposed_aliases[registry_key] = alias_additions
        for alias in alias_additions:
            alias_claims[alias].add(registry_key)
    alias_collisions = {alias for alias, owners in alias_claims.items() if len(owners) > 1}

    for registry_key, record in records.items():
        payload = tentative[registry_key]
        freeze_existing = bool(frozen_registry_keys and registry_key in frozen_registry_keys)
        old_history_key = _normalize(str(payload.get("old_history_canonical_key") or ""))
        final_history_key = _normalize(str(payload.get("new_history_canonical_key") or ""))
        inferred = bool(payload.get("inferred"))
        collision = bool(payload.get("unresolved_collision"))
        reason_if_left_blank = str(payload.get("reason_if_left_blank") or "")
        alias_additions = [] if freeze_existing else [alias for alias in proposed_aliases.get(registry_key) or [] if alias not in alias_collisions]
        prof = profiles.get(final_history_key) or {}

        team_filled = False
        inferred_team = ""
        if not freeze_existing and not str(record.get("team") or "").strip() and final_history_key:
            inferred_team, _team_reason = _suggest_team_label(prof, team_key_to_label)
            if inferred_team:
                _append_source(record, "db_linkage_enrichment")
                _set_field(record, "team", inferred_team, "db_linkage_enrichment")
                team_filled = True
                stats["teams_filled"] += 1

        batting_band_filled = False
        if not freeze_existing and not str(record.get("likely_batting_band") or "").strip() and final_history_key:
            inferred_band = _batting_band_from_positions(dict(prof.get("batting_positions") or {}))
            if inferred_band:
                _append_source(record, "db_linkage_enrichment")
                _set_field(record, "likely_batting_band", inferred_band, "db_linkage_enrichment")
                batting_band_filled = True

        bowling_phase_filled = False
        if not freeze_existing and not str(record.get("likely_bowling_phases") or "").strip() and final_history_key:
            inferred_phase = _infer_bowling_phase_hint(prof)
            if inferred_phase:
                _append_source(record, "db_linkage_enrichment")
                _set_field(record, "likely_bowling_phases", inferred_phase, "db_linkage_enrichment")
                bowling_phase_filled = True

        if not freeze_existing and final_history_key and final_history_key != old_history_key:
            _append_source(record, "db_linkage_enrichment")
            summary = record.setdefault("metadata_source_summary", {})
            summary["history_canonical_source"] = str(payload.get("decision_reason") or "db_linkage_enrichment")
            summary["linkage_enrichment_applied"] = True
            _set_field(record, "history_canonical_key", final_history_key, "db_linkage_enrichment")

        if alias_additions:
            _append_source(record, "db_linkage_enrichment")
            summary = record.setdefault("metadata_source_summary", {})
            summary["linkage_enrichment_applied"] = True
            merged_aliases = list(record.get("aliases") or [])
            merged_aliases.extend(alias_additions)
            record["aliases"] = _dedupe_keep_order(merged_aliases)
            record.setdefault("field_sources", {})["aliases"] = "db_linkage_enrichment"
            stats["aliases_added"] += len(alias_additions)

        enriched = bool(
            (final_history_key and final_history_key != old_history_key)
            or alias_additions
            or team_filled
            or batting_band_filled
            or bowling_phase_filled
        )
        if enriched:
            stats["players_enriched"] += 1
        if collision:
            stats["unresolved_collisions"] += 1
        if not final_history_key:
            stats["players_left_blank"] += 1

        audit_players[registry_key] = {
            "registry_key": registry_key,
            "canonical_name": str(record.get("canonical_name") or record.get("display_name") or registry_key),
            "old_history_canonical_key": old_history_key,
            "new_history_canonical_key": final_history_key,
            "inferred": inferred,
            "evidence_summary": list(payload.get("candidate_notes") or []),
            "alias_additions": alias_additions,
            "team_filled": team_filled,
            "team_after": str(record.get("team") or ""),
            "confidence": float(payload.get("confidence") or 0.0),
            "unresolved_collision": collision or any(alias in alias_collisions for alias in proposed_aliases.get(registry_key) or []),
            "reason_if_left_blank": reason_if_left_blank or ("" if final_history_key else str(payload.get("decision_reason") or "")),
        }

    return {
        "schema_version": 1,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": str(evidence.get("db_path") or ""),
        "latest_ipl_date": str(evidence.get("latest_ipl_date") or ""),
        "summary": stats,
        "players": audit_players,
    }, stats


def build_player_registry(
    *,
    output_path: str | Path | None = None,
    audit_output_path: str | Path | None = None,
) -> dict[str, Any]:
    cricinfo_path = _resolve_path(
        str(getattr(config, "PLAYER_METADATA_CRICINFO_PATH", "") or "data/player_metadata_cricinfo.json")
    )
    curated_path = _resolve_path(
        str(getattr(config, "PLAYER_METADATA_CURATED_PATH", "") or "data/player_metadata_curated.json")
    )
    alias_path = _resolve_path(
        str(getattr(config, "PLAYER_ALIAS_OVERRIDES_PATH", "") or "data/player_alias_overrides.json")
    )
    marquee_path = _resolve_path(
        str(getattr(config, "PLAYER_MARQUEE_OVERRIDES_PATH", "") or "data/player_marquee_overrides.json")
    )
    squad_json_records, squad_json_summary = _load_squad_json_records()

    cricinfo = _load_json_file(cricinfo_path)
    curated = _load_json_file(curated_path)
    alias_overrides = _load_json_file(alias_path)
    marquee_overrides = _normalize_marquee_players(_load_json_file(marquee_path))
    records: dict[str, dict[str, Any]] = {}

    for raw_key, payload in (cricinfo.items() if isinstance(cricinfo, dict) else []):
        if not isinstance(payload, dict):
            continue
        registry_key = _normalize(str(raw_key or ""))
        if not registry_key:
            continue
        record = _ensure_record(
            registry_key,
            records,
            seed_display_name=str(payload.get("player_name") or payload.get("display_name") or ""),
        )
        _apply_metadata_payload(
            record,
            payload,
            source_label=str(payload.get("source") or "cricinfo_curated"),
        )

    for raw_key, payload in (curated.items() if isinstance(curated, dict) else []):
        if not isinstance(payload, dict):
            continue
        normalized_key = _normalize(str(raw_key or ""))
        candidates = [normalized_key, str(payload.get("display_name") or ""), str(payload.get("player_name") or "")]
        registry_key = _resolve_existing_registry_key(candidates, records) or normalized_key
        if not registry_key:
            continue
        
        record = _ensure_record(
            registry_key,
            records,
            seed_display_name=str(payload.get("display_name") or payload.get("player_name") or ""),
        )
        _capture_aliases(record, candidates)
        _apply_metadata_payload(
            record,
            payload,
            source_label=str(payload.get("source") or "curated_manual"),
        )

    for raw_canon, raw_aliases in (alias_overrides.items() if isinstance(alias_overrides, dict) else []):
        canon_key = _normalize(str(raw_canon or ""))
        if not canon_key:
            continue
        if isinstance(raw_aliases, str):
            alias_list = [raw_aliases]
        elif isinstance(raw_aliases, (list, tuple)):
            alias_list = [str(x or "") for x in raw_aliases]
        else:
            alias_list = []
        alias_list = _dedupe_keep_order(alias_list)
        registry_key = _resolve_existing_registry_key([canon_key, *alias_list], records)
        if not registry_key:
            registry_key = _normalize(alias_list[0] if alias_list else canon_key)
        record = _ensure_record(registry_key, records, seed_display_name=alias_list[0] if alias_list else "")
        _append_source(record, "alias_override")
        summary = record.setdefault("metadata_source_summary", {})
        summary["alias_override_applied"] = True
        summary["history_canonical_source"] = "alias_override"
        _set_field(record, "history_canonical_key", canon_key, "alias_override")
        merged_aliases = list(record.get("aliases") or [])
        if registry_key != canon_key:
            merged_aliases.append(registry_key)
        merged_aliases.extend(alias_list)
        record["aliases"] = _dedupe_keep_order(merged_aliases)

    for raw_key, payload in marquee_overrides.items():
        if not isinstance(payload, dict):
            continue
        registry_key = _resolve_existing_registry_key([raw_key], records) or _normalize(raw_key)
        record = _ensure_record(registry_key, records, seed_display_name=raw_key)
        _append_source(record, "marquee_override")
        summary = record.setdefault("metadata_source_summary", {})
        summary["marquee_override_applied"] = True
        merged_aliases = list(record.get("aliases") or [])
        nk = _normalize(raw_key)
        if nk and nk != registry_key and nk not in merged_aliases:
            merged_aliases.append(nk)
        record["aliases"] = _dedupe_keep_order(merged_aliases)
        tier = str(payload.get("marquee_tier") or "").strip().lower()
        reason = str(payload.get("marquee_reason") or "").strip()
        if tier:
            _set_field(record, "marquee_tier", tier, "marquee_override")
        if reason:
            _set_field(record, "marquee_reason", reason, "marquee_override")

    for squad_key, payload in squad_json_records.items():
        # Task 1 & 3: Ensure squad-imported names resolve to existing curated records
        registry_key = _resolve_existing_registry_key([squad_key], records) or squad_key

        record = _ensure_record(
            registry_key,
            records,
            seed_display_name=str(payload.get("display_name") or payload.get("canonical_name") or ""),
        )

        if registry_key != squad_key:
            # Explicitly capture merged squad name as an alias for runtime lookup
            _capture_aliases(record, [squad_key])
            if "short" in squad_key:
                logger.info("Registry identity merge: squad_name='%s' -> registry_key='%s'", payload.get("display_name"), registry_key)

        _apply_metadata_payload(record, payload, source_label="squad_json")

    linkage_audit_payload, linkage_stats = _apply_db_linkage_enrichment(records)
    linkage_audit_payload, linkage_stats = _apply_raw_cricsheet_linkage_fallback(
        records,
        linkage_audit_payload,
        linkage_stats,
        _load_ipl_history_evidence(),
    )

    for registry_key, record in records.items():
        display_name = str(record.get("display_name") or record.get("canonical_name") or "").strip()
        if not display_name:
            display_name = _title_from_key(registry_key)
        record["display_name"] = display_name
        record["canonical_name"] = str(record.get("canonical_name") or display_name).strip() or display_name
        aliases = _dedupe_keep_order(list(record.get("aliases") or []))
        record["aliases"] = [alias for alias in aliases if alias != registry_key]
        _apply_slot_constraint_defaults(record)
        summary = record.setdefault("metadata_source_summary", {})
        summary["sources_seen"] = list(summary.get("sources_seen") or [])
        summary.setdefault(
            "preferred_metadata_source",
            str(
                summary.get("preferred_metadata_source")
                or record.get("field_sources", {}).get("primary_role")
                or record.get("field_sources", {}).get("display_name")
                or ""
            ),
        )
        summary.setdefault("alias_override_applied", False)
        summary.setdefault("marquee_override_applied", False)
        summary.setdefault("linkage_enrichment_applied", False)
        summary.setdefault("history_canonical_source", "")
        record["confidence"] = max(0.0, min(1.0, float(record.get("confidence") or 0.0)))

    out_path = _resolve_path(str(output_path or _registry_path()))
    audit_path = _resolve_path(str(audit_output_path or _registry_linkage_audit_path()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_files": {
            "player_metadata_cricinfo": str(cricinfo_path),
            "player_metadata_curated": str(curated_path),
            "player_alias_overrides": str(alias_path),
            "player_marquee_overrides": str(marquee_path),
            "player_squads_dir": str(_squads_dir_path()),
            "player_registry_linkage_audit": str(audit_path),
        },
        "squad_json_summary": squad_json_summary,
        "linkage_enrichment_summary": linkage_stats,
        "players": OrderedDict(sorted(records.items(), key=lambda kv: kv[0])),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(linkage_audit_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _invalidate_registry_cache()
    return payload


def _invalidate_registry_cache() -> None:
    global _REGISTRY_CACHE
    global _REGISTRY_CACHE_MTIME
    global _REGISTRY_METADATA_LOOKUP_CACHE
    global _REGISTRY_MARQUEE_LOOKUP_CACHE
    global _REGISTRY_ALIAS_OVERRIDE_CACHE
    _REGISTRY_CACHE = None
    _REGISTRY_CACHE_MTIME = None
    _REGISTRY_METADATA_LOOKUP_CACHE = None
    _REGISTRY_MARQUEE_LOOKUP_CACHE = None
    _REGISTRY_ALIAS_OVERRIDE_CACHE = None


def load_player_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    global _REGISTRY_CACHE_MTIME
    path = _registry_path()
    if not path.is_file():
        logger.warning("player_registry: missing %s", path)
        return {"schema_version": 1, "players": {}}
    mtime = float(path.stat().st_mtime)
    if _REGISTRY_CACHE is not None and _REGISTRY_CACHE_MTIME == mtime:
        return _REGISTRY_CACHE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("player_registry: failed reading %s (%s)", path, exc)
        return {"schema_version": 1, "players": {}}
    if not isinstance(payload, dict):
        payload = {"schema_version": 1, "players": {}}
    if not isinstance(payload.get("players"), dict):
        payload["players"] = {}
    _REGISTRY_CACHE = payload
    _REGISTRY_CACHE_MTIME = mtime
    _REGISTRY_METADATA_LOOKUP_CACHE = None
    _REGISTRY_MARQUEE_LOOKUP_CACHE = None
    _REGISTRY_ALIAS_OVERRIDE_CACHE = None
    return payload


def registry_players() -> dict[str, dict[str, Any]]:
    payload = load_player_registry()
    players = payload.get("players")
    if isinstance(players, dict):
        return players
    return {}


def registry_active() -> bool:
    return bool(registry_players())


def registry_metadata_lookup_map() -> dict[str, dict[str, Any]]:
    global _REGISTRY_METADATA_LOOKUP_CACHE
    if _REGISTRY_METADATA_LOOKUP_CACHE is not None:
        return _REGISTRY_METADATA_LOOKUP_CACHE
    built_at = str(load_player_registry().get("built_at") or "")
    out: dict[str, dict[str, Any]] = {}
    for registry_key, record in registry_players().items():
        metadata_record = {
            "player_key": registry_key,
            "display_name": str(record.get("display_name") or record.get("canonical_name") or ""),
            "batting_hand": str(record.get("batting_hand") or ""),
            "batting_style": str(record.get("batting_style") or ""),
            "bowling_style_raw": str(record.get("bowling_style_raw") or ""),
            "bowling_style": str(record.get("bowling_style") or ""),
            "bowling_type_bucket": str(record.get("bowling_type_bucket") or ""),
            "primary_role": str(record.get("primary_role") or ""),
            "secondary_role": str(record.get("secondary_role") or ""),
            "role_description": str(record.get("role_description") or ""),
            "allrounder_type": str(record.get("allrounder_type") or ""),
            "likely_batting_band": str(record.get("likely_batting_band") or ""),
            "likely_bowling_phases": str(record.get("likely_bowling_phases") or ""),
            "allowed_batting_slots": list(record.get("allowed_batting_slots") or []),
            "preferred_batting_slots": list(record.get("preferred_batting_slots") or []),
            "opener_eligible": bool(record.get("opener_eligible")),
            "finisher_eligible": bool(record.get("finisher_eligible")),
            "floater_eligible": bool(record.get("floater_eligible")),
            "source": str(((record.get("metadata_source_summary") or {}).get("preferred_metadata_source") or "")),
            "confidence": float(record.get("confidence") or 0.0),
            "last_updated": built_at,
            "team": str(record.get("team") or ""),
            "squad_status": str(record.get("squad_status") or ""),
            "status": str(record.get("status") or ""),
            "is_captain": bool(record.get("is_captain")),
            "is_vice_captain": bool(record.get("is_vice_captain")),
            "is_wicketkeeper": bool(record.get("is_wicketkeeper")),
            "age": str(record.get("age") or ""),
            "allrounder_subtype_hint": str(record.get("allrounder_subtype_hint") or ""),
            "history_canonical_key": str(record.get("history_canonical_key") or ""),
            "marquee_tier": str(record.get("marquee_tier") or ""),
            "notes": str(record.get("notes") or ""),
            "registry_key": registry_key,
        }
        for lookup_key in _record_lookup_keys(record):
            existing = out.get(lookup_key)
            if existing and lookup_key != registry_key and existing.get("registry_key") != registry_key:
                continue
            out[lookup_key] = dict(metadata_record)
    _REGISTRY_METADATA_LOOKUP_CACHE = out
    return out


def registry_marquee_lookup_map() -> dict[str, dict[str, Any]]:
    global _REGISTRY_MARQUEE_LOOKUP_CACHE
    if _REGISTRY_MARQUEE_LOOKUP_CACHE is not None:
        return _REGISTRY_MARQUEE_LOOKUP_CACHE
    out: dict[str, dict[str, Any]] = {}
    for registry_key, record in registry_players().items():
        tier = str(record.get("marquee_tier") or "").strip().lower()
        reason = str(record.get("marquee_reason") or "").strip()
        if not tier:
            continue
        payload = {"marquee_tier": tier, "marquee_reason": reason}
        for lookup_key in _record_lookup_keys(record):
            existing = out.get(lookup_key)
            if existing and lookup_key != registry_key and existing != payload:
                continue
            out[lookup_key] = dict(payload)
    _REGISTRY_MARQUEE_LOOKUP_CACHE = out
    return out


def registry_alias_override_maps() -> tuple[dict[str, list[str]], dict[str, str]]:
    global _REGISTRY_ALIAS_OVERRIDE_CACHE
    if _REGISTRY_ALIAS_OVERRIDE_CACHE is not None:
        return _REGISTRY_ALIAS_OVERRIDE_CACHE

    canon_to_aliases: dict[str, list[str]] = defaultdict(list)
    alias_to_canon: dict[str, str] = {}
    collisions: set[str] = set()
    for registry_key, record in registry_players().items():
        summary = record.get("metadata_source_summary") or {}
        canon = _normalize(str(record.get("history_canonical_key") or registry_key))
        if not canon:
            continue
        if not (
            bool(summary.get("alias_override_applied"))
            or bool(summary.get("linkage_enrichment_applied"))
            or bool(record.get("aliases"))
            or canon != registry_key
        ):
            continue
        aliases = _dedupe_keep_order(
            [
                registry_key,
                str(record.get("display_name") or ""),
                str(record.get("canonical_name") or ""),
                *list(record.get("aliases") or []),
            ]
        )
        aliases = [alias for alias in aliases if alias and alias != canon]
        for alias in aliases:
            if alias in alias_to_canon and alias_to_canon.get(alias) != canon:
                collisions.add(alias)
                continue
            alias_to_canon[alias] = canon
            if alias not in canon_to_aliases[canon]:
                canon_to_aliases[canon].append(alias)

    for alias in collisions:
        alias_to_canon.pop(alias, None)
    for canon, aliases in list(canon_to_aliases.items()):
        canon_to_aliases[canon] = [alias for alias in aliases if alias not in collisions]
        if not canon_to_aliases[canon]:
            canon_to_aliases.pop(canon, None)

    _REGISTRY_ALIAS_OVERRIDE_CACHE = (dict(canon_to_aliases), alias_to_canon)
    return _REGISTRY_ALIAS_OVERRIDE_CACHE


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical merged player registry.")
    parser.add_argument("--output", default=str(_registry_path()), help="Output JSON path")
    parser.add_argument(
        "--audit-output",
        default=str(_registry_linkage_audit_path()),
        help="Linkage enrichment audit JSON path",
    )
    args = parser.parse_args(argv)
    payload = build_player_registry(output_path=args.output, audit_output_path=args.audit_output)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(_resolve_path(args.output)),
                "audit_output": str(_resolve_path(args.audit_output)),
                "player_count": len((payload.get("players") or {})),
                "linkage_enrichment_summary": payload.get("linkage_enrichment_summary") or {},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
