# Streamline batch 2 — results

Low-risk cleanup only: dead imports, one dead wrapper, archival of a zero-reference package. No prediction, UI, or output behavior changes intended.

## Evidence gathered (before edits)

| Item | Search | Result |
|------|--------|--------|
| `providers` / `ipl_provider` | `rg` over `*.py` | **No** imports (only self-doc + README + audit docs). |
| `history_sync.normalize_scorecard_url` | `rg` over repo | **Only** definition in `history_sync.py`; internal `fetch_and_store_scorecard` already calls `utils.normalize_scorecard_url`. |
| `datetime` in `app.py` | `rg` `\bdatetime\b` | **Only** the import line — symbol unused. |

## A. Dead imports

| File | Change |
|------|--------|
| `app.py` | Removed unused `datetime` from `from datetime import date, datetime, time as dt_time` → `date, time as dt_time`. |

## B. Dead helpers

- **None removed** this batch (no other helper proven unused repo-wide without automated lint).

## C. Redundant wrappers

| File | Change |
|------|--------|
| `history_sync.py` | Removed `normalize_scorecard_url` (3-line forwarder to `utils.normalize_scorecard_url`). Call sites already use `utils` directly inside this module. |

**Intentionally not changed:** No re-export added; public surface shrinks only for a name that had **zero** external references.

## D. Zero-reference files archived

| Action | Detail |
|--------|--------|
| `git mv providers archive/providers` | Entire package moved (empty `__init__.py` + deprecated `ipl_provider.py` placeholder). **Batch 4:** re-homed to `archive/source_deprecated/providers/` (still unused). |

**README.md:** Architecture table row updated (see **cleanup batch 4** for current path `archive/source_deprecated/providers/`).

## Files modified (summary)

- `app.py`
- `history_sync.py`
- `README.md`
- `docs/repo_streamline_audit.md` (kept in sync with batch 2 facts)
- `docs/streamline_batch2_results.md` (this file)
- `archive/source_deprecated/providers/__init__.py` (moved; batch 4 path)
- `archive/source_deprecated/providers/ipl_provider.py` (moved; batch 4 path)

## Functions removed

- `history_sync.normalize_scorecard_url`

## Intentionally left untouched

- `utils.normalize_scorecard_url` implementation
- All alias resolution, metadata merge, XI / impact-sub / marquee / squad construction
- `docs/runtime_streamlining_plan.md` content (still valid roadmap; batch 2 only partially checks off the URL-wrapper item)
- Broad F401 sweeps without `ruff`/`pyflakes` in the venv

## Validation outcomes

| Check | Command | Outcome |
|--------|---------|---------|
| App import | `python -c "import app"` | **OK** |
| Predictor unit test | `python -m unittest tests.test_predictor_pipeline_stages.TestPredictorPipelineStages.test_assign_batting_order_stage_uses_final_xi_only_and_writes_reasons` | **OK** |
| History / sync path | `python -m unittest tests.test_history_sync.TestLocalHistoryDebug.test_local_history_debug_is_sqlite_only` | **OK** |

### Full `run_prediction` flow

`TestPredictionFailsafe.test_run_prediction_continues_when_local_history_raises` was **not** re-run as the “normal prediction” check: it still fails on the existing fixture (`Bowling options 2 < 5`), same as batch 1 — **out of scope** for this cleanup.

## Revert notes

- Restore `providers` at repo root: `git mv archive/source_deprecated/providers providers` (current path after batch 4)
- Restore `history_sync.normalize_scorecard_url` from history if an external script depended on it (none found in-repo).
