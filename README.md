# IPL Playing XI Predictor (Streamlit)

End-to-end **Playing XI** projection for IPL-style squads with **live weather**, **venue priors**, **multi-perspective scoring** (coach, player, analyst, opposition), **impact subs**, **toss / innings leverage**, and **win probability**. **IPL-only:** stored **scorecards are the primary signal** for XI selection and batting order (recent XI rates, venue usage, overseas patterns, bowling usage, chase/defend when known).

### Match history (local SQLite only)

**Run prediction** does **not** scrape IPL results pages, Cricbuzz, or Cricinfo. XI and batting-order priors read **only** from local SQLite (`matches`, `team_match_xi`, `team_match_summary`). Populate history via **Cricsheet JSON backfill** (loader to be added alongside this repo) or occasional **Parse & store** in the sidebar for single URLs.

If SQLite is thin or stale, the UI surfaces **local** warnings (Cricsheet backfill / manual ingest) — not “internet sync failed”.

## Setup

```bash
cd ipl-xi-predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: point the database elsewhere:

```bash
export IPL_PREDICTOR_DB=/path/to/ipl_predictor.sqlite
```

## Inputs

1. **Team A (Home)** and **Team B (Away)** — dropdowns with official IPL slugs (`chennai-super-kings`, `mumbai-indians`, …). Team B cannot match Team A.
2. **Venue** — curated list plus optional free-text override.
3. **Match date** and **Match Time (IST)** — all fixture times and **weather** use **India Standard Time** (`Asia/Kolkata`).
4. **Unavailable players** — one name per line (loose match).

**Squads** load automatically from `https://www.iplt20.com/teams/{slug}/squad` when you change teams (see `squad_fetch.py`). Text areas stay **editable** (roles `bat` / `wk` / `all` / `bowl` from the site where possible; add `overseas` manually if needed). If the fetch fails, a warning is shown and you can paste the squad manually. Use **Refresh squads from IPLT20** to retry.

## Rules enforced in XI selection

- At most **4 overseas** players.
- At least **1 wicketkeeper** (role `wk`).
- At least **5 bowling options** (role `bowl` / `all`, or bowling skill above an internal threshold).
- **Batting order** is driven by **historical batting positions** from ingested scorecards (recent-weighted EMA), with **role-bucket ordering** only as a fallback when a player has no usable history. The lineup table in the UI includes `selection_score`, `history_xi_score`, and explainability flags.

## Learning

Parsed scorecards populate:

- `match_results`, `match_xi`, `match_batting`, `match_bowling`
- **`matches`** — one row per stored match (venue, date, teams, result, scorecard URL, source) keyed to `match_results.id`
- **`team_match_xi`** — one row per player per team per match: actual XI membership, batting position / order, keeper, overs bowled, role bucket, impact/overseas flags where inferable
- **`team_match_summary`** — JSON snapshots (`playing_xi_json`, `batting_order_json`, `bowlers_used_json`, overseas/impact blobs) for debugging and future models
- `learned_player` — EMA **impact** from batting/bowling lines
- `learned_venue_team` — coarse **bat-first vs bowl-first win** counts for future toss edges

Re-pasting the same URL is rejected (deduped by URL). The same **fixture** from a **different** URL (e.g. IPL vs Cricbuzz) is deduped by **canonical match key** (teams + date) — no duplicate `match_results` rows.

**Backfill:** older DBs may lack `matches` / `team_match_*` for past ingests. Call `db.backfill_history_tables_from_results(limit=400)` (e.g. from a one-off script or the optional control in the app) to rebuild history tables from stored `raw_payload`. With an **empty** history, the predictor falls back to composite + role heuristics (`config.HISTORY_*`).

## Scorecard ingestion

- **`parsers/router.py`** — `parse_scorecard(url)` selects the parser from the **URL host** (Cricbuzz, ESPNcricinfo, IPLT20), **fetches safely**, and **never raises**: failures land in `payload["ingestion"]["errors"]`.
- **`parsers/schema.py`** — Cross-source **fallbacks** (title-based team names, venue/date/result regexes), **`batting_order`** and **`bowlers_used`** derived lists, and **`ingestion.completeness`** booleans for: team names, venue, date, playing XI, batting order, bowlers used, result.
- **Partial data** is written to SQLite whenever `ingestion.has_storable_content` is true; empty player rows are skipped.
- **Tests:** `python -m unittest discover -s tests -v` (install `requirements.txt` first). Includes `tests/test_history_sync.py` (dedupe, local history debug, prediction failsafe).

## Architecture

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `predictor.py` | Scoring, XI selection, batting order, subs, win model |
| `history_sync.py` | Local SQLite history snapshot for prediction; optional `fetch_and_store_scorecard`; `get_cached_match_count` |
| `providers/ipl_provider.py` | Placeholder (automatic URL discovery removed; use Cricsheet backfill) |
| `utils.py` | Scorecard URL + canonical match identity helpers |
| `history_xi.py` | IPL: history-first XI scores + batting EMA from `team_match_xi` |
| `learner.py` | Read/write learned signals from stored matches |
| `weather.py` | Open-Meteo hourly snapshot |
| `venues.py` | IPL venue coordinates + condition priors |
| `db.py` | SQLite schema and CRUD |
| `config.py` | Weights, thresholds, paths |
| `parsers/router.py` | URL → fetch → source parser → enrichment + ingestion meta |
| `parsers/schema.py` | Fallbacks, derived fields, completeness |
| `parsers/*.py` | Site-specific HTML / JSON extraction |
| `ipl_teams.py` | IPL franchise slugs ↔ display labels |
| `squad_fetch.py` | Official IPLT20 squad page fetch + parse |
| `time_utils.py` | IST (`Asia/Kolkata`) helpers for weather & prediction |

## Rule-based learning & confidence

- **Weights** for history signals (XI frequency, batting slot, bowling load, venue–team picks, overseas mix, day/night, dew) and **confidence** tuning live in `config.py` (`LEARN_*`, `HISTORY_*`, `CONF_*`).
- **XI ordering** blends normalized `history_xi_score` with `composite` (`HISTORY_SELECTION_HISTORY_WEIGHT`). With `HISTORY_PRIMARY_XI_SELECTION` and `HISTORY_COMPOSITE_HISTORY_BUMP_SCALE = 0`, the legacy per-player composite **bump** from `history_rules` is skipped so the squad list is **eligibility-only** for pattern decisions.
- **History bump** (`history_delta` / `history_notes`) still reflects venue–team priors for transparency; primary pick signal is `selection_score` + stored XI history.
- **Venue chase prior**: aggregated from `learned_venue_team` (bowl-first wins vs bat-first wins) and nudges win probability when Team A **chases** (see toss section in the UI).
- **Optional ingest hints**: local start hour and overseas counts improve `match_context` and `learned_overseas_mix` (see expander under scorecard URL).

## Disclaimer

Scraping is **fragile** if a site redesigns markup. Parsers use multiple heuristics and fall back to **inferring XI from batting tables** when a dedicated Playing XI block is missing. This tool is for **analysis and experimentation**, not betting or official team selection.
