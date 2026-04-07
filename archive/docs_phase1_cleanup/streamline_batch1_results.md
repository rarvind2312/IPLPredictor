# Streamline batch 1 — results

## Goal

Move clearly non-runtime CLI / audit scripts out of the repo root without deleting anything, changing prediction logic, or changing Streamlit UI code.

## What changed

| Action | Path |
|--------|------|
| Created package | `tools/__init__.py` (empty; marks `tools` as a package for clarity) |
| Moved | `alias_integrity_audit_ipl_2026.py` → `tools/alias_integrity_audit_ipl_2026.py` |
| Moved | `validate_last_ipl_2026_matches.py` → `tools/validate_last_ipl_2026_matches.py` |
| Moved | `pipeline_audit.py` → `tools/pipeline_audit.py` |
| Moved | `build_player_registry.py` → `tools/build_player_registry.py` |

Each moved script now prepends the **repository root** to `sys.path` when the file lives under `tools/`, so imports like `import predictor` and `import config` still resolve when you run:

```bash
# from repo root
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/build_player_registry.py --help
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/alias_integrity_audit_ipl_2026.py
```

The alias audit docstring was updated to show the new path.

**Imports elsewhere:** No production modules imported these files (verified via search before the move), so no `import` rewrites were required in `app.py`, `predictor.py`, or `pages/`.

## Verification

| Check | Command / outcome |
|--------|-------------------|
| App module import | `python -c "import app"` → **OK** (`import app OK`) |
| Moved CLI | `python tools/build_player_registry.py --help` → **OK** |
| Predictor pipeline unit test | `python -m unittest tests.test_predictor_pipeline_stages.TestPredictorPipelineStages.test_assign_batting_order_stage_uses_final_xi_only_and_writes_reasons` → **OK** |

### Full `run_prediction` smoke test

`tests.test_history_sync.TestPredictionFailsafe.test_run_prediction_continues_when_local_history_raises` was attempted; it **errored** with `ValueError: Team A hard constraints unsatisfied after repair: ['Bowling options 2 < 5']`. This appears tied to the **fixture squad** / constraint repair path, not to batch 1 file moves. No code was changed to “fix” that test in this pass (out of scope).

### Streamlit process

Batch 1 did not start a long-lived `streamlit run` server here. **Module-level import** of `app` is the practical startup check that all top-level dependencies (including `predict_ui_render` → `predictor`) still resolve.

## Git

Use `git status` to review; moves were done with `git mv` so history is preserved for the four files.

## Next

See `docs/runtime_streamlining_plan.md` for a **plan-only** follow-up (no execution in this pass).
