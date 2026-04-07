# Cleanup batch 5 — audit

## A. Repeated loader paths

| Pattern | Locations | Notes |
|---------|-----------|--------|
| `json.loads(path.read_text(encoding="utf-8"))` | `db._load_curated_player_metadata_file`, `cricsheet_convert.load_cricsheet_payload`, `player_registry._load_json_file`, `player_registry` CLI paths, `tools/alias_integrity_audit`, etc. | Mixed semantics: some must **raise** on bad JSON (Cricsheet ingest), some return `{}`, some use `OrderedDict`. |
| `Path.read_text` + JSON | `cricsheet_readme`, `cricsheet_recent_api`, SQLite `json.loads(raw)` rows | Not all are `data/*.json` config files. |
| `predictor._metadata_dependency_report._json_status` | Nested helper: resolve path → `is_file` → `json.loads` → key counts | Same UTF-8 + swallow parse errors pattern as curated metadata load (without dict filtering). |
| `impact_subs_engine._safe_json_obj` | Parses **strings** / dict passthrough | Different contract. |
| `player_alias_resolve._load_alias_overrides` | Uses `player_registry.registry_alias_override_maps()` | Not a raw file read in alias module. |
| `db` / `learner` / `stage_derive` | `json.loads` on **SQLite column blobs** | Not file paths. |

## B. Safest consolidation target (chosen)

**Single helper:** `utils.read_json_utf8(path: Path) -> Optional[Any]`

- **Why safe:** `utils` already sits below `db` and does not import `db` / `learner` (avoids cycles).
- **Call sites (batch 5):**
  1. **`db._load_curated_player_metadata_file`** — replace inline `json.loads(read_text)` with `read_json_utf8`; preserve `OSError` handling on `is_file()`, preserve warning log when load returns `None`, preserve downstream dict filtering.
  2. **`predictor._metadata_dependency_report._json_status`** — replace inner `try/json.loads` with `read_json_utf8`; preserve outer `is_file()` branch and returned status dict for missing vs present-but-invalid JSON.

**Not consolidated (same batch):** `cricsheet_convert` (must propagate parse errors), `player_registry._load_json_file` (`object_pairs_hook=OrderedDict`), tools scripts, marquee/alias **precedence** (unchanged).

## C. Risky areas left untouched

- Metadata merge / registry lookup precedence (`player_registry.registry_metadata_lookup_map`, `db._load_curated_player_metadata_with_priority`).
- Any `OrderedDict` JSON loads.
- Cricsheet / ingest pipelines that rely on **exceptions** for control flow.
- `lru_cache` redesign or cross-module shared caches for registry rows.
- `player_alias_resolve` resolution layers.
