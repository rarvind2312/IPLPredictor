# Runtime streamlining plan (Phase 3 — plan only)

**Status:** Proposal for later implementation. **Do not execute** as part of batch 1.

Principles: one source of truth per concern, remove duplicate UI/debug copies, break only **lazy** cycles with explicit interfaces, keep behavior identical until covered by tests.

---

## 1. Single metadata source of truth

**Today:** Curated JSON (`db` loaders), SQLite `player_metadata`, Cricinfo JSON, runtime attachment in `history_xi` / `predictor._annotate_player_metadata`, and UI merges in `predict_ui_render._build_squad_row_maps`.

**Target:**

- Define a small immutable `PlayerMetadataView` (dict-like or dataclass) built in **one** function used by both prediction and UI.
- `db.fetch_player_metadata_batch` remains the read path; merge rules (curated > cricinfo > defaults) live in one module, called from that builder.
- UI only **displays** `row["_meta"]` derived from the same builder output attached on the prediction result payload (avoid second merge).

---

## 2. Single alias / identity resolution path

**Today:** `player_alias_resolve`, `history_linkage`, `history_key_collision`, linkage flags on `history_debug`, and string statuses like `history_key_collision_loser`.

**Target:**

- One `resolve_player_identity(name, team_context) -> Resolution` returning canonical key, linkage type, collision group id, and human-readable reason.
- `history_xi` and `history_sync` both call into that API; collision application stays in one place after resolution.

---

## 3. Single marquee tier source of truth

**Today:** Registry map in `player_registry`, JSON overrides file, `history_debug["marquee_tier"]`, and audit helpers in `full_pipeline_audit`.

**Target:**

- Resolve tier once when building `history_debug` (or metadata view), store only the final string/enum.
- Audits read the same field; remove duplicate string normalization helpers where possible.

---

## 4. Single squad / `SquadPlayer` construction path

**Today:** Squad fetch → parse → predictor internal scoring attaches many `history_debug` keys incrementally.

**Target:**

- Keep a single factory `squad_row_to_squad_player(row, team_key) -> SquadPlayer` used everywhere squads enter the engine.
- Avoid attaching overlapping keys in different stages without a schema (document required `history_debug` keys in one place).

---

## 5. Remove redundant wrappers

**Today (after batch 2):** Only `utils.normalize_scorecard_url` remains; the unused `history_sync` forwarder was removed (zero in-repo callers).

**Target:**

- No further action needed for this specific alias unless external scripts still expected `history_sync.normalize_scorecard_url`.

---

## 6. Reduce recompute / overwrite loops

**Today:** `history_rules` blend vs `history_xi` scores; config flag to avoid double-counting; selection debug recomputed in UI from `prediction_layer_debug`.

**Target:**

- Emit a single `selection_explain` / `history_bump_summary` on each player at scoring time.
- UI becomes read-only for that narrative (no second scoring interpretation).

---

## 7. Centralize repeated JSON loads

**Today:** Multiple modules load JSON from `data/` with ad hoc caching.

**Target:**

- Small `data_files.py` (or extend `config`) with `get_json(path) -> dict` using `lru_cache` + mtime or explicit version keys.
- Register known files (aliases, marquee overrides, curated metadata paths) in one registry.

---

## 8. Clearer XI vs impact-sub story

**Today:** `impact_subs_engine` + XI selection in `predictor`; shared `selection_model_debug` and `probable_first_choice_prior`.

**Target:**

- After XI is frozen, run impact pipeline with explicit inputs (XI set, bench set) and attach `impact_subs_trace` separate from `xi_trace`.
- Ensure “omitted despite high prior” explanations are generated once, not independently in two code paths.

---

## 9. Suggested order of execution (future batches)

1. ~~Dedupe `utils.normalize_scorecard_url` / `history_sync` wrapper~~ (done in streamline batch 2).
2. Extract shared `_selection_debug_top15_for_side` used by `app.py` and admin page (UI-only, behavior parity).
3. Metadata view builder shared by `predict_ui_render` and prediction payload (higher risk; needs golden JSON snapshot tests).
4. Identity resolution consolidation (highest risk; needs alias/collision test suite green).

Each batch: run `import app`, one predictor unit module, and a minimal `run_prediction` scenario with a **valid** synthetic squad that satisfies bowling constraints.
