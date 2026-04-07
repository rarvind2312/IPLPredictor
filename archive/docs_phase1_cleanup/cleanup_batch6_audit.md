# Cleanup batch 6 — audit

## Method

- `rg` over `*.py` (excluding `.vendor/`) for import usage and single-use symbols.
- No changes to predictor scoring, XI/impact paths, alias resolution semantics, metadata merge, history linkage, or caches.

## A. Proven-dead imports

| Location | Finding |
|----------|---------|
| `tools/alias_integrity_audit_ipl_2026.py` | **`import json`** had exactly **one** use: `json.loads(p.read_text(encoding="utf-8"))` inside `_load_alias_override_map`. After routing through `utils.read_json_utf8`, **`json` is unused** — removed (verified with `rg json` on the file). |

**Spot-checks (no change):** `predict_ui_render`, `batting_order_whatif`, `full_pipeline_audit`, `matchup_features`, `weather`, `first_choice_prior` — imports appear referenced; left as-is.

## B. Proven-unused helpers

- **None removed.** No tiny private function found that is unreferenced repo-wide without touching prediction/UI paths.

## C. Trivial wrappers

- **None removed.** No additional forward-only wrappers identified at lower risk than leaving them.

## D. Safest micro-cleanup target (implemented)

- **`_load_alias_override_map`** in `tools/alias_integrity_audit_ipl_2026.py` now uses **`utils.read_json_utf8(p)`** instead of `is_file` + `try` / `json.loads` / `except`.
- **Semantics:** Missing file, unreadable file, or invalid JSON → `read_json_utf8` returns `None` → `not isinstance(payload, dict)` → `{}`, same as before. Non-dict JSON (e.g. list) → `{}`, same as before.

## E. Risky areas intentionally untouched

- `predictor`, `history_xi`, `history_linkage`, `player_alias_resolve` (runtime alias pipeline), `db` metadata merge / `player_registry` loaders, `cricsheet_convert` (must raise on bad JSON), `impact_subs_engine`, selection/scoring, Streamlit pages beyond the one tool file above.
