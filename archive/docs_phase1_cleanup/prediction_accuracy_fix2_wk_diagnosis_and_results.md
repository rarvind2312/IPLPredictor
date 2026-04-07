# Prediction accuracy fix 2 — wicketkeeper-role handling for `wk_max`

## A. Current wicketkeeper-role path (before conceptual split)

1. **`player_role_classifier.classify_player`** computed a single flag **`is_wk_role_player`** as:

   - `SquadPlayer.is_wicketkeeper` **(squad boolean)**, **or**
   - `role_bucket == WK-Batter`, **or**
   - Metadata primary/secondary role strings indicating keeper (`_meta_role_indicates_keeper`).

2. **`is_designated_keeper_candidate`** was set **equal** to that same flag.

3. **`rules_xi.validate_xi`** incremented **`wk_role_players`** via **`role_counts` → `classify_player(...).is_wk_role_player`** and raised **`wk_max`** if **> 2**.

4. **`rules_xi.assign_designated_keeper_name`** chose the designated keeper from players with **`is_wk_role_player`**.

5. **Repair** in **`predictor._repair_xi_if_needed`** used **`classify_player(p).is_wk_role_player`** for wicketkeeper-related swaps.

**Effect:** Any squad-listed WK-Batter with full keeper metadata (e.g. MS Dhoni, Sanju Samson, **Kartik Sharma** on Chennai Super Kings) all counted toward the **hard cap**, even when one is clearly a **young backup** listing. That produced **`Max 2 wicketkeepers allowed, found 3`** and **`repair_exhausted_no_constraint_safe_swaps_remaining`** on the known GT vs CSK-style path.

---

## B. Exact root cause of over-counting

- **All three** OR conditions could apply simultaneously for multiple CSK players: IPL squad **`WK-Batter`** bucket plus **`is_wicketkeeper`** plus **primary `wk_batter` / `wicketkeeper_batter`** metadata.
- There was **no distinction** between:
  - **Primary / franchise keeper** (should consume a `wk_max` slot), and
  - **Squad-depth WK-Batter** (flex / rookie backup) who should **not** each consume a cap slot when tier/history signals say “backup.”
- **Secondary-only keeper metadata** (primary `batter`, secondary `wk_*`) also charged a full cap slot even when the player is primarily a batter in the data model.

---

## C. Code change made

### `player_role_classifier.py`

- **`_broad_designated_keeper_candidate`**: preserves the **old OR-of-three** semantics for **who may be named designated keeper**.
- **`_counts_toward_wk_max_cap`**: **narrower** logic for **`is_wk_role_player`** (the field **`rules_xi` / repair still use for the hard cap**):
  1. **`_secondary_only_keeper_meta`**: primary role is **non-keeper** but secondary is keeper → **excluded** from cap (batter-primary flex).
  2. **`_wk_backup_wk_batter_excluded_from_wk_cap`**: **`WK-Batter`** with **`marquee_tier` in (`tier_2`, `tier_3`)** and **no** `last_match_is_keeper` / `is_keeper` in `selection_model_debug.last_match_detail` → **excluded** from cap (backup without recent keeping signal).
  3. **`Batter`** bucket + **`is_wicketkeeper`** but primary metadata is **not** keeper-indicating → **excluded** from cap.

- **`classify_player`**:  
  - `is_designated_keeper_candidate = _broad_designated_keeper_candidate(...)`  
  - `is_wk_role_player = _counts_toward_wk_max_cap(...)`

- **`wk_cap_exclusion_reason(p)`**: returns a short machine-readable reason when broad keeper signals exist but the player does **not** count toward `wk_max`.

- **`wicketkeeper_xi_debug_rows`**: adds **`marquee_tier`**, **`classify_is_designated_keeper_candidate`**, **`classify_counts_toward_wk_max_cap`**, **`wk_cap_exclusion_reason`**, and keeps **`classify_is_wk_role_player`** as an alias of the **cap** flag for backward compatibility.

### `rules_xi.py`

- **`assign_designated_keeper_name`**: builds the candidate pool from **`is_designated_keeper_candidate`** (broad), **not** the cap-narrow flag, so a backup WK can still be chosen designated keeper if the XI truly needs it.

**Not changed:** overseas, bowling constraints, batting order, impact subs, repair search breadth, scoring weights, metadata precedence order in ingest.

---

## D. Before / after on known failing repro

**Setup:** `predictor.run_prediction("Gujarat Titans", "Chennai Super Kings", ...)` with live `squad_fetch` squads, venue Ahmedabad, date **2023-05-26** (historical smoke).

| | Before fix | After fix |
|---|------------|-----------|
| Outcome | **`ValueError`**: `Team B hard constraints unsatisfied after repair: ['Max 2 wicketkeepers allowed, found 3']` | **`SUCCESS`** — prediction completes; CSK XI has 11 names. |

**Mechanism:** **Kartik Sharma** (tier_2 WK-Batter, no recent keeper flag in selection debug) no longer increments **`wk_max`** while **Dhoni** and **Samson** still do.

---

## E. Validation results

1. **`python -c "import app"`** — OK.

2. **`tools/generate_prediction_vs_actual_report.py --from-sqlite 3`** — completes (small sample); no `wk_max` hard-stop observed in that slice.

3. **GT vs CSK direct repro** — completes successfully (see §D).

4. **Scope check** — diffs limited to **`player_role_classifier.py`** and **`rules_xi.assign_designated_keeper_name`**; **`predictor.py`** unchanged (existing `wicketkeeper_xi_debug_rows` log lines now emit richer JSON).

---

## F. Risks intentionally not addressed

- **Tier-2/3 keeper who is the real match-day keeper** but has **no** `last_match_is_keeper` yet may be **excluded from the cap** and could allow a pathological XI to pass validation while understating keeper slots. Mitigation later: stronger keeper usage signals from history.

- **`marquee_tier` missing** on a player → backup rule does **not** strip the cap (safe default: still counts).

- **Three tier-1 WK-Batters** on one squad would still violate `wk_max` — this fix targets **backup / secondary-metadata** patterns, not impossible squad shapes.

- **Soft** wicketkeeper preference logic elsewhere (e.g. predictor snippets using raw `is_wicketkeeper` only) is **unchanged**; only **`classify_player`-driven** hard/semi-hard paths use the new split.
