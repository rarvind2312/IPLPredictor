# Streamline batch 3 — results

## Goal

Deduplicate the selection-debug DataFrame helper used by **Predict** (`app.py` → `predict_ui_render`) and **Admin** (`pages/1_Admin_and_maintenance.py`) without changing engine logic or altering each surface’s displayed table shape.

## Duplicated helpers removed

| Location | Removed |
|----------|---------|
| `app.py` | `_selection_debug_top15_for_side` (~65 lines) |
| `pages/1_Admin_and_maintenance.py` | `_selection_debug_top15_for_side_admin` (~43 lines) |

## New shared helper

| Module | Symbol |
|--------|--------|
| `selection_debug_ui.py` | `selection_debug_top15_dataframe_for_side(r, side, *, include_reason_columns=True)` |

- **`include_reason_columns=True` (default):** Same rows/columns as the former **Predict** helper: includes `recent_form_competitions` and `reason_summary` (with the same truncation / newline handling).
- **`include_reason_columns=False`:** Same rows/columns as the former **Admin** helper (nine core columns only).

`xi_validation` extraction from `r["selection_debug"][side]` is unchanged and shared.

## Files modified

- `selection_debug_ui.py` (**new**)
- `app.py` — import module; pass `selection_debug_ui.selection_debug_top15_dataframe_for_side` into `render_stored_prediction_results`
- `pages/1_Admin_and_maintenance.py` — `functools.partial(..., include_reason_columns=False)` bound as `_SELECTION_DEBUG_TOP15_ADMIN`
- `tests/test_selection_debug_ui.py` (**new**) — column-set and per-column parity for one sample row
- `docs/repo_streamline_audit.md` — note dedupe under duplicate list
- `docs/streamline_batch3_results.md` (this file)

## Other UI/debug duplicates

No additional merges in this batch: only the audit-listed pair was clearly identical modulo the two optional columns.

## Intentionally left untouched

- `predict_ui_render` callback contract `(r, side) -> (DataFrame, dict)` — unchanged; callables satisfy it.
- `predictor`, history linkage, metadata, marquee, squad construction, scoring, impact-subs, caching.
- Admin page still ends with `main()` on import; no change to Streamlit lifecycle.

## Validation

| Step | Result |
|------|--------|
| `python -c "import app"` | **OK** |
| `python -m unittest tests.test_selection_debug_ui` | **OK** |
| Admin page | **Not imported** as a module (file calls `main()` at load). Validated with `compile(..., 'exec')` on `pages/1_Admin_and_maintenance.py` — **OK** |

## Revert

Remove `selection_debug_ui.py` and the test; restore the two local functions from git history; wire call sites back to the in-file helpers.
