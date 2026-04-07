# Repo streamline audit (Phase 1)

**Scope:** Trace from `app.py` (Streamlit Predict page) plus `pages/1_Admin_and_maintenance.py` (second entrypoint). **Excluded:** `.vendor/`, `tests/`, and generated/cache paths.

**Method:** Static import graph from top-level `import` / `from` statements and targeted `rg` for symbol references. Lazy / function-local imports (e.g. `db` → `learner`) are called out where found.

---

## Runtime entrypoints

| Entrypoint | Role |
|------------|------|
| `app.py` | Primary Streamlit app (prediction UX, sidebar ingest hooks). |
| `pages/1_Admin_and_maintenance.py` | Multi-page app: ingest, derive, audits, maintenance. |

---

## Classification buckets

- **DIRECT_RUNTIME** — Imported (transitively) from `app.py` on normal startup.
- **INDIRECT_RUNTIME** — Not imported by `app.py` directly, but imported by a module in the `app.py` closure (prediction / squad / DB / UI render path).
- **BUILD_TOOLS** — Offline or CI-oriented builders (CLI `__main__`), not needed for `import app`.
- **INGESTION_TOOLS** — Cricsheet / scorecard / archive ingest; used from app sidebar, admin page, or manual runs.
- **AUDIT_DEBUG_TOOLS** — Audits, validation harnesses, logging helpers; optional at runtime.
- **UNUSED_OR_OPTIONAL** — No importers found in production code paths (may still be useful manually or in tests).

For each category below: **move root?** / **archive?** / **duplicate?** / **referenced by runtime entrypoints?**

---

## A. Files safe to move to `tools/` (standalone scripts / audits)

Evidence: **no** `import` of these modules elsewhere in repo (only `__main__` or docs).

| File | Notes |
|------|--------|
| `alias_integrity_audit_ipl_2026.py` | CLI audit; uses squad/history/predictor stack. |
| `validate_last_ipl_2026_matches.py` | CLI validation harness (URLs + full prediction). |
| `pipeline_audit.py` | Full-pipeline audit runner; not imported by app or predictor. |
| `build_player_registry.py` | Thin wrapper → `player_registry.main()`; safe to colocate with other CLIs. |

After move: run as `python tools/<name>.py` from repo root with a small `sys.path` bootstrap (see batch 1).

---

## B. Files safe to move to `archive/` (later, not in batch 1)

Only after confirming no Streamlit/docs/scripts reference paths. **Not moved in batch 1** to avoid churn.

- None identified with **zero** risk without checking CI and personal workflows. Candidates for *future* review: dead packaging experiments, superseded one-off notebooks (none in tree).

---

## C. Files safe to delete later (not now)

| File / area | Why “later” |
|-------------|-------------|
| ~~`providers/`~~ → **`archive/providers/`** (batch 2) | Confirmed zero Python importers before archival; README updated. |
| Duplicate audit entrypoints | If `pipeline_audit.py` and overlapping reports in admin converge, one path could retire after merge. |

---

## D. Duplicate logic paths to merge later

1. **Selection debug table** — `_selection_debug_top15_for_side` in `app.py` and `_selection_debug_top15_for_side_admin` in `pages/1_Admin_and_maintenance.py` are nearly the same (diverged copies).
2. **History linkage / alias resolution** — `history_linkage.link_current_squad_to_history`, `player_alias_resolve` (including collision loser flags), `history_key_collision.apply_intrasquad_effective_key_collisions`, and strings like `history_key_collision_loser` span multiple layers; single “resolution record” shape would reduce drift.
3. **Marquee tier** — `history_debug["marquee_tier"]`, `player_registry.registry_marquee_lookup_map()`, JSON overrides (`data/player_marquee_overrides.json`), and `full_pipeline_audit.marquee_tier_str` read the same conceptual field at different times.
4. **Player metadata** — Curated JSON (`db._load_curated_player_metadata_*`), DB table `player_metadata`, Cricinfo JSON, and `predictor._annotate_player_metadata` / `history_xi` attachment; multiple load/merge paths.
5. **Cricsheet payload path** — `cricsheet_convert.load_cricsheet_payload` used from `cricsheet_all_ingest`, `cricsheet_ingest`, `cricsheet_recent_api`, `pipeline_audit`; normalization overlaps with `ingest_normalize`.
6. **XI vs impact subs** — `impact_subs_engine` vs XI selection inside `predictor` share scoring/debug fields (`selection_model_debug`, `probable_first_choice_prior`); easy to double-apply or display inconsistent “why omitted” reasons without a single narrative builder.
7. **URL normalization** — use `utils.normalize_scorecard_url` only (`history_sync` alias removed in batch 2; zero callers).

---

## E. Circular dependencies to break later

- **`learner` ↔ `db`:** `learner` imports `db` at module level; `db` uses **lazy** `import learner` inside functions. Usually safe at runtime but fragile for typing/tests and obscures ownership.
- **No strong cycle found** for `predict_ui_render` ↔ `predictor` (one-way: UI imports predictor only).

---

## F. Runtime hotspots (recompute / overwrite / repeated JSON)

1. **Repeated JSON file reads** — `player_alias_resolve`, `player_registry`, `config`-adjacent data files, curated metadata in `db`; many use `lru_cache` or load-once patterns but not unified.
2. **`history_rules` after `history_xi`** — `config` documents optional suppression of stacking history onto composite; risk of “history applied twice” depending on flags (`config` comments near history_rules blend).
3. **`full_pipeline_audit`** — Imported by `predictor`; emits when env var set; read-only but adds branching in hot path.
4. **SQLite + batch fetches** — `db.fetch_*_batch` used from UI (`predict_ui_render`) and engine; same entities sometimes re-fetched in adjacent layers.
5. **Squad row construction** — `predict_ui_render._build_squad_row_maps` merges structured squad, scoring breakdown, omitted list, XI rows, then attaches `_meta`; parallel “player dict” shapes may exist in `predictor` output.

---

## G. First safest cleanup batch only

**Applied (Phase 2):** see `docs/streamline_batch1_results.md`.

Summary: the four CLI/audit scripts listed below were moved under `tools/` with a repo-root `sys.path` bootstrap. No runtime modules were renamed; no app/predictor imports required updates.

- `tools/alias_integrity_audit_ipl_2026.py`
- `tools/validate_last_ipl_2026_matches.py`
- `tools/pipeline_audit.py`
- `tools/build_player_registry.py`

**Still do not move** without a dedicated batch: `stage1_audit.py`, `full_pipeline_audit.py`, `player_registry.py`, or any module imported by `app.py` / `predictor.py` / `pages/`.

---

## Module inventory (root + packages, non-test, non-vendor)

### DIRECT_RUNTIME (from `app.py` import list)

| Module | Bucket note |
|--------|-------------|
| `app.py` | Entry |
| `audit_profile` | Perf / timing |
| `config` | Paths, flags |
| `cricsheet_all_ingest` | Sidebar full-archive ingest |
| `db` | SQLite |
| `ipl_squad`, `ipl_teams` | Domain constants / labels |
| `learner` | Weights / keys |
| `predict_ui_render` | Stored results UI |
| `predictor` | Core prediction |
| `recent_form_cache` | Cache rebuild UI hooks |
| `squad_fetch` | Official squads |
| `streamlit_db_init` | DB init on startup |
| `time_utils`, `weather`, `venues` | Match time / weather / venue |

### INDIRECT_RUNTIME (transitive from above; representative)

| Module | Why indirect |
|--------|----------------|
| `batting_order_whatif` | `predict_ui_render` |
| `canonical_keys` | `predictor`, `learner` |
| `first_choice_prior` | `history_xi` |
| `full_pipeline_audit` | `predictor` (audit emit) |
| `h2h_history` | `win_probability_engine`, UI |
| `history_context` | `predictor`, UI |
| `history_key_collision` | `history_linkage` |
| `history_linkage` | `history_xi`, `stage1_audit` (admin) |
| `history_rules` | `predictor` |
| `history_sync` | `predictor`, UI |
| `history_xi` | `predictor`, `impact_subs_engine` |
| `impact_subs_engine` | `predictor` |
| `matchup_features` | `selection_model`, `recent_form_cache` |
| `player_alias_resolve` | `predictor`, `history_sync`, `player_role_classifier` |
| `player_registry` | `db`, `player_alias_resolve` |
| `player_role_classifier` | `predictor`, UI |
| `rules_spec`, `rules_xi` | XI rules |
| `selection_model` | `history_xi` |
| `utils` | `db`, `history_sync`, `cricsheet_convert`, … |
| `win_probability_engine` | `predictor`, UI |
| `cricsheet_convert` | `cricsheet_all_ingest` |
| `ingest_normalize` | `cricsheet_convert` |
| `cricsheet_readme` | `cricsheet_convert` / ingest chain |

### INGESTION_TOOLS (admin page and/or manual; some overlap app)

| Module | Entry |
|--------|--------|
| `cricsheet_ingest` | Admin |
| `cricsheet_recent_api` | Admin |
| `stage_derive` | Admin |
| `cricinfo_squad_parser` | Admin |
| `parsers/*` | `history_sync`, admin scorecard parse |

### AUDIT_DEBUG_TOOLS

| Module | Entry |
|--------|--------|
| `stage1_audit` | Admin page |
| `alias_integrity_audit_ipl_2026.py` | CLI only |
| `validate_last_ipl_2026_matches.py` | CLI only |
| `pipeline_audit.py` | CLI only |

### BUILD_TOOLS

| Module | Entry |
|--------|--------|
| `build_player_registry.py` | CLI → `player_registry.main()` |
| `player_registry` | Library + `python -m player_registry` style main |

### UNUSED_OR_OPTIONAL (no importers in app/predictor/admin path)

| Module | Note |
|--------|------|
| `archive/providers/ipl_provider.py` | Archived placeholder; still unused by runtime. |

---

## Tests & CI

All under `tests/` — **BUILD_TOOLS** / quality gate; not loaded by Streamlit unless you run pytest.

---

*Generated for streamline pass Phase 1. Phase 2 applies batch 1 moves only; Phase 3 is planning-only.*
