# Prediction accuracy fix 4 — low-history / new-player handling (results)

## A. Current low-history path

1. **Franchise history → normalized `hn`:** In `history_xi.compute_selection_scores`, each player’s `history_xi_score` is min–max normalized across the squad. `has_usable_sqlite_or_cricsheet_history` is true when row counts (or batting-slot samples) meet `HISTORY_SELECTION_STRONG_ROWS_THRESHOLD` (default 2). Below that, a **weaker** history weight `HISTORY_SELECTION_HISTORY_WEIGHT_WEAK` applies when blending `hn` with composite later.

2. **Derive blend and debut damp:** Stage-2 profile rows feed `derive_norm` and a blend `gamma = STAGE3_DERIVE_HN_BLEND_MAX * conf_scale * damp`. For **sparse/debut** players (`_sparse_debutante_player`), `damp = STAGE3_DERIVE_DEBUT_DAMP` (0.3), which **shrinks** how much derive can correct a low `hn`.

3. **Selection model:** `selection_model.apply_selection_model` builds  
   `base = 0.40*recent_form + 0.30*ipl_history_role + 0.20*team_balance + 0.10*venue`.  
   **`_ipl_history_role_score`** is anchored on **`hn`** (~55%) plus derive XI frequency / role stability / role shape. No T20 cache row forces **recent form** onto a composite-heavy fallback (~0.62*`recent_usage` + 0.28*`composite` + neutral).

4. **Downstream XI ranking:** `_run_xi_selection_stage` orders largely by tier / `selection_score` / composite. Marquee tiering runs **before** this fix’s nudge and still uses pre-nudge `selection_model_debug` for impact-style signals.

## B. Exact cause of under-selection

- Thin franchise SQLite history yields **low `hn`**, so **`ipl_history_role_score`** stays low even when **derive** shows meaningful **`recent_usage_score`** or **`xi_selection_frequency`** (continuity at the franchise).
- **`debut_damp`** further reduces derive’s ability to pull `hn` up for flagged new/sparse players.
- **Team-balance** and **venue** terms are squad-relative; veterans with strong `hn` and form cache rows remain ahead, so **youngsters / churn-era picks** who are **plausibly in the XI** but **under-sampled in SQLite** sit too low in the ordered squad.

## C. Code change made

**Single narrow adjustment:** After `history_xi.compute_selection_scores`, full derive/metadata/marquee annotation, and `_set_player_ipl_flags`, `predictor._apply_low_history_continuity_nudge` runs per team. It **adds a capped increment** to `selection_score` only when:

- `selection_score_components.has_usable_sqlite_or_cricsheet_history` is **false** (thin franchise history per existing flag), and  
- `selection_score` is below `LOW_HISTORY_NUDGE_ONLY_BELOW_SCORE` (default 0.72), and  
- At least one **continuity** gate fires: strong/medium **`recent5_xi_rate`**, strong/medium **derive `recent_usage_score` / `xi_selection_frequency`**, or **`valid_current_squad_new_to_franchise`** with **`probable_first_choice_prior`** above a floor.

Bump tiers (max default **0.042**): full max for strong recent-XI rate; ~78% or ~68% of max for derive/recent gates; ~55% for new-franchise prior path.

**Config knobs** (env overrides in `config.py`):  
`LOW_HISTORY_CONTINUITY_NUDGE_ENABLE`, `LOW_HISTORY_CONTINUITY_NUDGE_MAX`, `LOW_HISTORY_CONTINUITY_RECENT5_MIN`, `LOW_HISTORY_CONTINUITY_RECENT_USAGE_MIN`, `LOW_HISTORY_CONTINUITY_XI_FREQ_MIN`, `LOW_HISTORY_NUDGE_ONLY_BELOW_SCORE`, `LOW_HISTORY_NUDGE_NEW_FRANCHISE_PRIOR_MIN`.

**Diagnostics:**

- Per player: `history_debug["low_history_continuity_nudge"]` (bump, tier, before/after score, signals).  
- Mirrored into `selection_model_debug`, `selection_score_components.low_history_continuity_nudge_bump`, and `scoring_breakdown` where present.  
- Rollup: `history_sync_debug["low_history_continuity_nudge"]` with `players_helped_team_a` / `_b`, `players_helped_count`, and short **rank_before / rank_after** in each row (pure `selection_score` ordering within the squad).

**Not changed:** `selection_model` formulas, `history_xi` hn/derive math, impact subs, batting order, repair, parsers, squad fetch, WK-cap logic.

## D. Before/after examples

- **Qualitative:** On a **3-match** SQLite audit run (same harness as `tools/generate_prediction_vs_actual_report.py --from-sqlite 3`), **Kolkata Knight Riders**’ predicted XI included **Angkrish Raghuvanshi** alongside established names (e.g. Narine, Green, Varun), with **9/11** overlap vs the scorecard XI for that fixture — useful as a **spot-check** that a thin-history youngster can surface when continuity signals justify it. This run is **not** a controlled A/B vs the pre-fix commit; treat as **sanity** only.
- **How to verify locally:** Run a prediction with full debug and inspect `history_sync_debug.low_history_continuity_nudge.players_helped_*` and any player’s `history_debug.low_history_continuity_nudge`.

## E. Validation results

| Check | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c "import app; import predictor"` | OK |
| Small prediction-vs-actual sample (`generate_prediction_vs_actual_report.py --from-sqlite 3`) | OK (full permissions; first attempt hit `database is locked` under sandbox — environment-specific) |
| Youngster/fringe spot-check | KKR block showed Raghuvanshi in predicted XI on that sample |
| Broad scoring rewrite | **No** — one post-pass additive nudge + config flags |

## F. Risks intentionally not addressed

- **Marquee tier** is computed **before** the nudge; tier-driven ordering does not automatically reflect the bump (only `selection_score` does).
- Players who are **low-history and low on all continuity gates** are **unchanged**; the fix does not promote unknown bench players.
- **`has_usable_sqlite_or_cricsheet_history` false** can include heterogeneous cases; tightening eligibility further would require a new signal or threshold (out of scope).
- Slightly higher `selection_score` can interact with **overseas / role repair** at the margin; caps and gates limit blast radius.
