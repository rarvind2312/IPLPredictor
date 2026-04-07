# Cleanup batch 4 — results

## Scope

- Audit source-specific / parser / ingest-related modules for **proven** unused status.
- Archive only what is **PROVEN_UNUSED**; no deletions; no prediction or UI logic changes.

## Finding

**Cricbuzz, Cricinfo scorecard, and IPL HTML parsers** (`parsers/cricbuzz_parser.py`, `parsers/cricinfo_parser.py`, `parsers/ipl_parser.py`) are **ACTIVE_RUNTIME** via `parsers/router.py` → `parse_scorecard`, used from `history_sync` and the Admin **Parse & store** flow. They were **not** archived.

**`cricinfo_squad_parser.py`** is **ACTIVE_ADMIN**. **Not** archived.

The only codebase area with **zero** Python importers (already archived in batch 2) was the legacy **`providers`** placeholder package.

## Applied change (safe archive batch)

| Action | Detail |
|--------|--------|
| Re-home | `archive/providers/` → `archive/source_deprecated/providers/` |
| Rationale | Same proven-unused content; groups deprecated source scaffolding under one tree. |

## Files modified

- `archive/source_deprecated/providers/ipl_provider.py` — docstring path text
- `README.md` — architecture table path
- `docs/repo_streamline_audit.md` — path references
- `docs/streamline_batch2_results.md` — historical note + current paths for moved files / revert
- `docs/unused_source_modules_audit.md` — **new** (full classification)
- `docs/cleanup_batch4_results.md` — this file

## Functions / logic

- **None** removed or changed in runtime code.

## Validation

| Check | Result |
|--------|--------|
| `python -c "import app"` | **OK** |
| `python -m unittest tests.test_selection_debug_ui -q` | **OK** (representative unit test) |
| `compile(open('pages/1_Admin_and_maintenance.py').read(), ..., 'exec')` | **OK** (admin calls `main()` on import; syntax-only check) |

## Revert

```bash
git mv archive/source_deprecated/providers archive/providers
```

Then restore README / doc path strings from git history if needed.
