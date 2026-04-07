# Cleanup batch 6 — results

## Changes

| File | Change |
|------|--------|
| `tools/alias_integrity_audit_ipl_2026.py` | `_load_alias_override_map`: use `utils.read_json_utf8(p)`; drop redundant `is_file` + `try/json.loads`; remove unused `import json`; add `import utils`. |

## Duplicated helpers / wrappers

- None removed beyond the inline load pattern above (same helper as batch 5).

## Docs

- `docs/cleanup_batch6_audit.md` (new)
- `docs/cleanup_batch6_results.md` (this file)

## Validation

| Step | Command | Outcome |
|------|---------|---------|
| App | `python -c "import app"` | OK |
| Selection debug | `python -m unittest tests.test_selection_debug_ui -q` | OK |
| Utils JSON helper | `python -m unittest tests.test_history_sync.TestHistorySyncDb.test_utils_canonical_key_order_insensitive -q` | OK |
| Admin | `compile(open('pages/1_Admin_and_maintenance.py').read(), ..., 'exec')` | OK |

## Revert

Restore `_load_alias_override_map` body and `import json` from git history.

## Stop criteria

Batch 6 completes the agreed low-risk cleanup line; further work should shift to prediction accuracy per user plan.
