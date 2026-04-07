# Unused / source-specific modules audit (batch 4)

**Method:** `rg` over `*.py` (excluding `.vendor/`) for `import <module>`, `from <module>`, and string references that imply runtime wiring. **Date:** streamline batch 4.

**Legend**

| Class | Meaning |
|-------|---------|
| **ACTIVE_RUNTIME** | Imported on prediction / SQLite / squad paths from `app.py` closure |
| **ACTIVE_ADMIN** | Used from `pages/1_Admin_and_maintenance.py` or maintenance Streamlit only |
| **ACTIVE_TOOL** | Used from `tools/` scripts only |
| **TEST_ONLY** | Referenced from `tests/` only (still required for CI) |
| **DOCS_ONLY** | Mentioned in markdown only |
| **PROVEN_UNUSED** | No `*.py` importer in app, admin, tools, tests, or runtime libs (may be self + docs) |
| **LIKELY_UNUSED_REVIEW** | Possible dead API surface; removing would need design review |

---

## A. Proven unused files safe to archive

| Path | Evidence | Action (batch 4) |
|------|----------|------------------|
| `providers/` package | Batch 2: zero `import providers` / `ipl_provider` in `*.py`. | Already removed from root in batch 2; **re-homed** to `archive/source_deprecated/providers/` (no code imports; path-only change). |

**No other** root-level or `parsers/` modules qualified as PROVEN_UNUSED without breaking `parsers.router.parse_scorecard`, `history_sync`, admin ingest, or tests.

---

## B. Likely unused files needing review

| Item | Notes |
|------|--------|
| `parsers/__init__.py` re-exports `parse_cricbuzz`, `parse_cricinfo`, `parse_ipl` | Runtime uses `parsers.router` and direct submodule imports in tests (`from parsers import cricbuzz_parser`). Public `parse_*` aliases may be redundant for in-repo callers only — **do not remove** without a deprecation pass and grep for external notebooks/scripts. |

---

## C. Still-used source-specific files

| Module | Classification | Importers / notes |
|--------|----------------|-------------------|
| `parsers/cricbuzz_parser.py` | **ACTIVE_RUNTIME** | `parsers/router.py` (`source == "cricbuzz"`). |
| `parsers/cricinfo_parser.py` | **ACTIVE_RUNTIME** | `parsers/router.py` (`source == "cricinfo"`). |
| `parsers/ipl_parser.py` | **ACTIVE_RUNTIME** | `parsers/router.py` (`source == "ipl"`). |
| `parsers/router.py` | **ACTIVE_RUNTIME** | `history_sync.fetch_and_store_scorecard`; `pages/1_Admin_and_maintenance.py` (parse & store). |
| `parsers/_common.py` | **ACTIVE_RUNTIME** | `parsers/router.py`, site parsers (`detect_source`, soup helpers). |
| `parsers/schema.py` | **ACTIVE_RUNTIME** | `parsers/router.py`, `history_sync` (`has_storable_content`). |
| `parsers/__init__.py` | **ACTIVE_RUNTIME** + **TEST_ONLY** | Re-exports; `tests/test_ingestion.py` imports submodules via `from parsers import …`. |
| `cricinfo_squad_parser.py` | **ACTIVE_ADMIN** | `pages/1_Admin_and_maintenance.py` (Cricinfo metadata JSON rebuild). |
| `cricsheet_all_ingest.py` | **ACTIVE_RUNTIME** | `app.py` sidebar ingest. |
| `cricsheet_ingest.py` | **ACTIVE_ADMIN** + **ACTIVE_RUNTIME** (indirect) | Admin; `stage1_audit`; ingest pipeline. |
| `cricsheet_recent_api.py` | **ACTIVE_ADMIN** | `pages/1_Admin_and_maintenance.py`. |
| `cricsheet_convert.py` | **ACTIVE_RUNTIME** + **ACTIVE_TOOL** | `cricsheet_all_ingest`, `cricsheet_ingest`, `tools/pipeline_audit.py`, etc. |
| `cricsheet_readme.py` | **ACTIVE_RUNTIME** + **ACTIVE_ADMIN** + **ACTIVE_TOOL** + **TEST_ONLY** | `cricsheet_ingest`, `stage1_audit`, `player_registry` (lazy import), `tools/pipeline_audit`, tests. |
| `ingest_normalize.py` | **ACTIVE_RUNTIME** | `cricsheet_convert.py` only (normalization for ingest). |
| `squad_fetch.py` | **ACTIVE_RUNTIME** | `app.py`, `predictor`, `tools/*`, etc. |

---

## D. Files intentionally left untouched

- All of **`parsers/*.py`** except the package already archived — required for multi-host scorecard URLs.
- **`stage_derive.py`**, **`stage1_audit.py`** — admin / audit stack.
- **`data/player_metadata_cricinfo.json`** — data artifact, not Python.
- **`.vendor/`** — vendored deps.

---

## E. Safe archive batch (applied in batch 4)

1. Create `archive/source_deprecated/`.
2. `git mv archive/providers archive/source_deprecated/providers`
3. Update docstrings / README / streamline docs that pointed at `archive/providers/`.

**Not done:** Deleting any file; archiving Cricbuzz/Cricinfo/IPL parsers; moving `cricinfo_squad_parser.py` (active admin).

---

## Search commands (reproducibility)

```bash
rg 'cricbuzz|cricinfo_parser|parse_cricbuzz' --glob '*.py'
rg 'cricinfo_squad_parser' --glob '*.py'
rg 'parse_scorecard|parsers\.router' --glob '*.py'
rg 'ipl_parser|parse_ipl' --glob '*.py'
rg 'import providers|ipl_provider' --glob '*.py'
```
