# Cleanup batch 5 — results

## Implemented (single step)

**New:** `utils.read_json_utf8(path: Path) -> Optional[Any]` — UTF-8 file read + `json.loads`, returns `None` when the path is not a file, read/parse fails, or JSON is invalid.

**Wired (behavior-matched):**

1. **`db._load_curated_player_metadata_file`** — Uses `read_json_utf8` after path resolution; keeps `try/except OSError` around `is_file()`; logs the same warning when the load yields `None`; unchanged dict key normalization.
2. **`predictor._metadata_dependency_report` inner `_json_status`** — Uses `read_json_utf8` after the existing `is_file()` gate; same return shapes for missing vs unreadable files.

## Files modified

- `utils.py` — `read_json_utf8`, imports (`json`, `Path`, `Any`)
- `db.py` — curated metadata file load path
- `predictor.py` — `import utils`; `_json_status` load
- `docs/cleanup_batch5_audit.md` (new)
- `docs/cleanup_batch5_results.md` (this file)

## Dead imports / helpers

- None removed (no proven-unused imports in touched blocks; `json` remains required in `predictor` elsewhere).

## Intentionally left untouched

- `player_registry._load_json_file`, `cricsheet_convert.load_cricsheet_payload`, tools, admin-only readers.
- All alias, metadata precedence, XI, impact-sub, and scoring logic.

## Validation

| Check | Command | Outcome |
|--------|---------|---------|
| App import | `python -c "import app"` | OK |
| Selection debug UI | `python -m unittest tests.test_selection_debug_ui -q` | OK |
| Utils / history | `python -m unittest tests.test_history_sync.TestHistorySyncDb.test_utils_canonical_key_order_insensitive -q` | OK |

## Revert

Remove `read_json_utf8` and restore the previous `try/json.loads` bodies in `db.py` and `predictor._json_status`.
