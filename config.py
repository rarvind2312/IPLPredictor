"""Application configuration: API endpoints, scoring weights, paths."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (directory containing this file)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = os.environ.get("IPL_PREDICTOR_DB", str(DATA_DIR / "ipl_predictor.sqlite"))

# Local Cricsheet IPL bundle (readme index + per-match JSON). Official zip uses
# ``data/ipl_json/README.txt``; also accept ``data/readme.txt`` if you copy the index.
CRICSHEET_JSON_DIR = Path(os.environ.get("IPL_CRICSHEET_JSON_DIR", str(DATA_DIR / "ipl_json")))
# Full multi-competition Cricsheet JSON archive (ingest-only; prediction reads SQLite).
CRICSHEET_ALL_JSON_DIR = Path(os.environ.get("IPL_CRICSHEET_ALL_JSON_DIR", str(DATA_DIR / "all_json")))
CRICSHEET_README_CANDIDATES: tuple[Path, ...] = (
    DATA_DIR / "readme.txt",
    DATA_DIR / "README.txt",
    CRICSHEET_JSON_DIR / "README.txt",
    CRICSHEET_JSON_DIR / "readme.txt",
)
# Ingest only the most recent N IPL seasons (by calendar year / season field) for prediction prep.
CRICSHEET_HISTORY_SEASON_COUNT = int(os.environ.get("IPL_CRICSHEET_HISTORY_SEASONS", "5"))
# When True, ``run_rebuild_raw_cricsheet_ingest`` ingests every IPL readme row (JSON on disk),
# not ``CRICSHEET_HISTORY_SEASON_COUNT``. Overridable per-run from the Streamlit checkbox.
CRICSHEET_FULL_ARCHIVE_INGEST = os.environ.get("IPL_CRICSHEET_FULL_ARCHIVE_INGEST", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Open-Meteo (no API key)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# HTTP
REQUEST_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Multi-perspective weights (must sum to 1.0 for interpretability)
WEIGHT_COACH = 0.22
WEIGHT_PLAYER = 0.20
WEIGHT_ANALYST = 0.22
WEIGHT_OPPOSITION = 0.18
WEIGHT_LEARNED = 0.18

# Squad constraints (IPL-style)
MAX_OVERSEAS = 4
MIN_WICKETKEEPERS = 1
MIN_BOWLING_OPTIONS = 5
# Realistic balance (role buckets — see ipl_squad / predictor)
MIN_NON_BOWLERS_IN_XI = 5
MIN_OPENER_BUCKET_IN_XI = 1  # at least one Batter or WK-Batter

# Heuristic: count as bowling option if bowling subscore >= this (0–1 scale)
BOWLING_OPTION_THRESHOLD = 0.38

# Learning: EMA factor for updating player impact after each parsed match
LEARN_EMA_ALPHA = 0.35

# Win model calibration (legacy logistic path in predictor.py)
WIN_MODEL_TEMPERATURE = 1.15

# --- Rule-based win probability engine (weighted factors, deterministic) ---
# Weights must sum to 1.0. Head-to-head is recency-weighted direct A-vs-B from SQLite.
WIN_ENG_WEIGHT_HEAD_TO_HEAD = 0.25
WIN_ENG_WEIGHT_VENUE = 0.15
WIN_ENG_WEIGHT_XI_STRENGTH = 0.20
WIN_ENG_WEIGHT_BATTING_ORDER = 0.15
WIN_ENG_WEIGHT_BOWLING_PHASES = 0.15
WIN_ENG_WEIGHT_MATCHUP = 0.03
WIN_ENG_WEIGHT_CONDITIONS = 0.04
# Innings-role fit from team chase vs defend records for the hypothetical toss
WIN_ENG_WEIGHT_TOSS_ROLE = 0.02
# Venue chase prior + dew + night + rain tilt toward the chasing side
WIN_ENG_WEIGHT_CHASE_ENVIRONMENT = 0.01

WIN_ENG_PROB_BASE = 50.0
WIN_ENG_PROB_MIN = 30.0
WIN_ENG_PROB_MAX = 70.0
# Each team's composite score is 0–100; difference maps linearly to probability offset
WIN_ENG_SCORE_TO_PROB_SCALE = 1.0

WIN_ENG_H2H_MAX_MATCHES = 5

# --- Rule-based learning weights (multi-perspective "learned" channel is separate) ---
# Each signal produces a raw score in ~[-1,1]; these are relative weights before HISTORY_BLEND_SCALE.
LEARN_WEIGHT_XI_FREQUENCY = 0.22
LEARN_WEIGHT_BATTING_SLOT = 0.14
LEARN_WEIGHT_BOWLING_USAGE = 0.16
LEARN_WEIGHT_VENUE_TEAM_XI = 0.18
LEARN_WEIGHT_OVERSEAS_MIX = 0.12
LEARN_WEIGHT_DAY_NIGHT = 0.10
LEARN_WEIGHT_DEW_CONTEXT = 0.08

# Multiply blended history raw score before capping per player
HISTORY_BLEND_SCALE = 0.85
# Max absolute bump applied to a player's composite from all history signals combined
HISTORY_ADJ_PER_PLAYER_CAP = 0.09

# Role proxy batting slot (1=opener) when refining before final order is known
HISTORY_PROXY_BAT_SLOT = {"bat": 3, "wk": 6, "bowl": 10, "all": 7}

# Minimum samples before a history rule fires (explainability / avoid noise)
LEARN_MIN_SAMPLES_SLOT = 2
LEARN_MIN_SAMPLES_BOWL = 2
LEARN_MIN_SAMPLES_VENUE_TEAM = 3
LEARN_MIN_SAMPLES_OVERSEAS_MIX = 4
LEARN_MIN_SAMPLES_DAY_NIGHT = 3
LEARN_MIN_SAMPLES_CHASE = 5

# Night vs day split for the *current* fixture (used with stored match_context)
NIGHT_START_HOUR_LOCAL = 18

# Dew: only blend night-history heuristic when model dew risk exceeds this
LEARN_DEW_RISK_THRESHOLD = 0.45

# Ingest-time dew proxy when hour is known (used only for learning tables)
DEW_PROXY_DAY = 0.38
DEW_PROXY_NIGHT = 0.70

# Ingest defaults when start hour unknown (no update to day/night tables)
INGEST_UNKNOWN_START_HOUR = -1

# Win model: extra logit from venue chase prior (bowl-first wins / all wins at venue)
LEARN_WEIGHT_CHASE_BIAS_LOGIT = 0.28

# --- Prediction confidence (0–1, rule-based, no ML) ---
CONF_WEIGHT_DB_DEPTH = 0.22
CONF_WEIGHT_HISTORY_COVERAGE = 0.28
CONF_WEIGHT_PERSPECTIVE_AGREEMENT = 0.28
CONF_WEIGHT_SCORE_SEPARATION = 0.22
CONF_MIN_MATCHES_FOR_FULL_DB = 12
CONF_IDEAL_XI_HISTORY_COVERAGE = 0.72
CONF_SEPARATION_TARGET = 0.045

# --- IPL history-first XI (scorecards in SQLite `matches` / `team_match_xi`) ---
IPL_CURRENT_SEASON_YEAR = 2026
IPL_COMPETITION_LABEL = "IPL"

# Blend: final selection score = history_weight * normalized(history_xi_score)
#   + (1 - history_weight) * composite. Stronger history when SQLite rows exist.
HISTORY_SELECTION_HISTORY_WEIGHT = 0.90
HISTORY_SELECTION_HISTORY_WEIGHT_STRONG = 0.94
HISTORY_SELECTION_HISTORY_WEIGHT_WEAK = 0.76
# Min team_match_xi rows (or cricsheet slot samples) to treat as "has usable history"
HISTORY_SELECTION_STRONG_ROWS_THRESHOLD = 2

# Probable first-choice XI prior (global IPL fallback when current-franchise rows are thin).
FIRST_CHOICE_GLOBAL_FRANCHISE_MATCHES_CAP = 5
FIRST_CHOICE_GLOBAL_MIN_DISTINCT_MATCHES = 2
FIRST_CHOICE_USED_GLOBAL_PRIOR_MIN = 0.22
FIRST_CHOICE_PRIOR_MAX_HN_BOOST = 0.16
FIRST_CHOICE_PRIOR_COMPOSITE_WEIGHT = 0.1
# Extra multiplier on first-choice HN boost when ``valid_current_squad_new_to_franchise``.
FIRST_CHOICE_NEW_TO_FRANCHISE_HN_FACTOR = 1.28

# Manual captain / wicketkeeper XI priors (not hard locks).
CAPTAIN_SELECTION_HN_BOOST = 0.14
WICKETKEEPER_SELECTION_HN_BOOST = 0.19

# Components for history_xi_score (sum of weighted normalized terms, ~0–1+ scale)
HISTORY_XI_W_RECENT5 = 0.27
HISTORY_XI_W_VENUE = 0.13
HISTORY_XI_W_PRIOR_SEASON = 0.08
HISTORY_XI_W_OVERSEAS_PATTERN = 0.06
HISTORY_XI_W_CHASE_DEFEND = 0.05
HISTORY_XI_W_BOWL_USAGE = 0.04
HISTORY_XI_W_PHASE_BOWL = 0.08
HISTORY_XI_W_BAT_AGGRESSOR = 0.03
HISTORY_XI_W_BOWL_CONTROL = 0.03
# Direct head-to-head vs selected opponent (recency-weighted XI / role signals)
HISTORY_XI_W_H2H_XI = 0.14
HISTORY_XI_W_H2H_VENUE = 0.09
# Blend batting-order EMA: weight on H2H-derived slot when samples exist (cap)
HISTORY_BAT_ORDER_H2H_BLEND_MAX = 0.58
HISTORY_BAT_ORDER_H2H_BLEND_PER_MATCH = 0.052

# Rows scanned when recomputing ``player_franchise_features`` from ball-by-ball tables
MATCHUP_FEATURE_MAX_INNINGS_ROWS = 90
MATCHUP_FEATURE_MAX_MATCHES_PHASE = 45

# --- Stage 2 derive (SQLite → profiles / patterns; no raw JSON) ---
DERIVE_HISTORY_SEASONS = int(os.environ.get("IPL_DERIVE_HISTORY_SEASONS", "5"))
DERIVE_RECENCY_HALFLIFE_MATCHES = float(os.environ.get("IPL_DERIVE_RECENCY_HALFLIFE", "20"))
DERIVE_SPARSE_PLAYER_SAMPLES = int(os.environ.get("IPL_DERIVE_SPARSE_SAMPLES", "3"))
DERIVE_FALLBACK_CONFIDENCE_MAX = float(os.environ.get("IPL_DERIVE_FALLBACK_CONF", "0.35"))

# Batting-order EMA from stored positions (recent alpha)
HISTORY_BAT_SLOT_EMA_ALPHA = 0.42
HISTORY_BAT_SLOT_UNKNOWN = 99.0
HISTORY_MIN_SAMPLES_BAT_ORDER = 2

# When True, do not add history_rules delta onto composite (history drives XI via history_xi_score).
HISTORY_PRIMARY_XI_SELECTION = True
HISTORY_COMPOSITE_HISTORY_BUMP_SCALE = 0.0

# Tiny additive bump per stored scorecard row for this player+franchise (breaks flat hx=0 ties).
HISTORY_XI_ROW_COUNT_BUMP = 0.00035
# Max extra score from row-count bump (keeps ordering stable).
HISTORY_XI_ROW_COUNT_BUMP_CAP = 0.06

# --- Local SQLite history depth (sufficiency hints in debug / warnings) ---
# Rows are loaded outside **Run prediction** (ingest stage, manual Parse & store). No pre-match internet sync.
HISTORY_SYNC_MIN_RECENT_MATCHES = 5
HISTORY_SYNC_TARGET_RECENT_MATCHES = 10
HISTORY_SYNC_MIN_PRIOR_SEASON_MATCHES = 2
HISTORY_SYNC_STALE_DAYS = 10.0
# Build expensive per-player squad↔history report during prediction-time history snapshot.
# Keep off by default; enable only for deep linkage debugging.
HISTORY_SYNC_INCLUDE_SQUAD_REPORT_ON_PREDICTION = (
    os.environ.get("IPL_HISTORY_SYNC_INCLUDE_SQUAD_REPORT", "").strip().lower() in ("1", "true", "yes")
)

# Warn in UI when a franchise has fewer than this many distinct usable matches in SQLite.
LOCAL_HISTORY_MIN_DISTINCT_MATCHES_WARN = 2

# Only attach the strong "load Cricsheet ingest" message when raw history is effectively empty.
STAGE_1_RAW_NEAR_EMPTY_MATCHES = 3
STAGE_1_RAW_NEAR_EMPTY_XI_ROWS = 40

# Debutant / first-IPL guard: reject weak franchise or global alias matches when SQLite depth
# for the resolved key is too thin (avoids inheriting wrong initials/surname links).
DEBUTANT_ALIAS_CONFIDENCE_STRONG = 0.915  # below layer_b (0.92) and layer_c (0.78) treated as weak/medium
DEBUTANT_MIN_GLOBAL_DISTINCT_FOR_WEAK_ALIAS = 9
DEBUTANT_MIN_FRANCHISE_HISTORY_ROWS_FOR_WEAK_ALIAS = 4

# Stage F rollup / health (per franchise, from linkage summary).
STAGE_F_FRANCHISE_DEPTH_OK_MATCHES = 10
STAGE_F_FRANCHISE_KEY_INDEX_OK = 25
STAGE_F_FRANCHISE_XI_ROWS_OK = 40
# ``major_linkage_failure`` only when most core slots lack a key *and* few cores linked.
STAGE_F_MAJOR_CORE_UNRESOLVED_FRAC = 0.6
STAGE_F_MAJOR_MIN_CORE_LINKED = 5
# ``partial_linkage_issue`` when unresolved core fraction exceeds this (and not major).
STAGE_F_PARTIAL_CORE_UNRESOLVED_FRAC = 0.3
STAGE_F_HEALTHY_UNRES_NON_CORE_FRAC = 0.6
# Among unresolved players, this fraction with no franchise surname keys → "mostly new".
STAGE_F_UNRESOLVED_MOSTLY_NEW_FRAC = 0.65
# Alias linked with history row counts in (0, this] → rolled_up ``linked_low_sample``.
STAGE_F_LINKED_LOW_SAMPLE_MAX_ROWS = 3

# --- Strict XI / history validation (warnings + batting-order scope) ---
# Warn when this many XI players have fewer stored history rows (per player).
HISTORY_VALIDATION_SPARSE_ROWS_WARN = 2
# Warn if at least this fraction of the XI is below the sparse threshold.
HISTORY_VALIDATION_SPARSE_XI_FRACTION = 0.35
# Also require franchise DB depth low before XI sparse warning (avoid false alarms).
HISTORY_VALIDATION_SPARSE_REQUIRE_DB_MATCHES_BELOW = 14

# --- Stage 3 prediction (SQLite derive profiles/patterns; no raw JSON) ---
# Blend normalized history (hn) with Stage-2 ``player_profiles`` core signal before final selection_score.
STAGE3_DERIVE_HN_BLEND_MAX = 0.24
STAGE3_DERIVE_DEBUT_DAMP = 0.3
STAGE3_PROFILE_CONFIDENCE_FLOOR = 0.14
STAGE3_VENUE_TEAM_XI_FREQ_BOOST_CAP = 0.055
STAGE3_VENUE_FIT_CONDITIONS_WEIGHT = 0.07
# Scenario-aware XI: cap additive tweak on base selection_score (0–1).
SCENARIO_XI_MAX_ABS_DELTA = 0.105
# Multiplier on raw branch deltas before capping (toss / innings role should move fringe picks).
SCENARIO_XI_BRANCH_STRENGTH = 1.42
# Penalize similar bowler types beyond depth-4 when ranking the squad for XI construction.
STAGE3_SIMILAR_BOWLER_PENALTY = 0.026
# Legacy logistic strength: blend composite with post-derive selection_score.
STAGE3_XI_STRENGTH_SELECTION_BLEND = 0.34
# Win engine: XI factor uses selection_score as well as composite.
WIN_ENG_XI_STRENGTH_SELECTION_SCORE_WEIGHT = 0.36
# Blend derive-time head_to_head_patterns with match-row H2H when sample is healthy.
WIN_ENG_DERIVED_H2H_BLEND = 0.2
WIN_ENG_DERIVED_H2H_MIN_SAMPLES = 6

# --- Selection model (Stage 3 XI ranking): base blend + tactical modifiers ---
# Base selection_score = weighted sum (each term in [0,1]); then tactical deltas (capped).
SELECTION_WEIGHT_RECENT_FORM = 0.40
SELECTION_WEIGHT_IPL_HISTORY_ROLE = 0.30
SELECTION_WEIGHT_TEAM_BALANCE_FIT = 0.20
SELECTION_WEIGHT_VENUE_EXPERIENCE = 0.10
# Recent form: last N T20-family matches + rolling months window (see selection_model).
SELECTION_RECENT_FORM_LAST_N_MATCHES = 5
SELECTION_RECENT_FORM_MONTHS = 5
# Cap total tactical adjustment (sum of pitch/weather/toss/opponent/squad_need) on [0,1] score axis.
SELECTION_TACTICAL_ADJUST_CAP = 0.11
# Prediction payload: full per-bench impact + history_usage tables (large). Default off for faster UI/JSON.
PREDICTION_FULL_DEBUG_PAYLOAD = os.environ.get("IPL_PREDICTION_FULL_DEBUG", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Log timing for startup / prediction sub-steps (see logger ``ipl_predictor.perf``).
PREDICTION_TIMING_LOG = os.environ.get("IPL_PREDICTION_TIMING", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Deep audit: phase + per-query SQLite timings during prediction, Streamlit startup breakdown (see ``audit_profile``).
AUDIT_PROFILING = os.environ.get("IPL_AUDIT_PROFILING", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# On first page load, auto-fetching squads can block first paint due to network latency.
# Keep off by default; users can click "Load squads from IPLT20 now".
AUTO_FETCH_SQUADS_ON_START = os.environ.get("IPL_AUTO_FETCH_SQUADS_ON_START", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Substrings on ``matches.competition`` (case-insensitive) for T20-family when ``match_format`` unset.
T20_FAMILY_COMPETITION_SUBSTRINGS: tuple[str, ...] = (
    "ipl",
    "indian premier league",
    "t20",
    "twenty20",
    "bbl",
    "big bash",
    "blast",
    "vitality",
    "psl",
    "cpl",
    "sa20",
    "ilt20",
    "hundred",
)
# Exclude obvious non-T20 competitions when format column is empty.
T20_EXCLUDE_COMPETITION_SUBSTRINGS: tuple[str, ...] = ("test", "first-class", "fc ", "odi", "one day")
