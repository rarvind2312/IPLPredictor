# Prediction accuracy fix 3 — repair overreach (results)

## A. Current repair path summary

Post-selection repair lives in `predictor._repair_xi_if_needed`:

1. **Trigger:** `rules_xi.validate_xi(xi, …)` on the XI entering repair. If there are **hard violations** or **semi-hard warnings**, repair runs; if neither, repair is skipped (early return with `repair_diagnostics.repair_skipped`).

2. **Main loop:** Up to **120** iterations. Each iteration re-validates the current `repaired` XI, reads `hard_codes` / `semi_codes`, builds an **add pool** (scored players not in the XI), and walks a **fixed priority** of rule buckets: `designated_keeper` → `wk_max` → `bowling_options_min` → `pacers_min` → `spinners_min` → overseas bounds → `wk_role_players_cap` (semi) → `structural_all_rounders_cap` (semi) → `top_order_min`. The first bucket that applies and yields a swap wins for that iteration.

3. **Candidate sets:** Each bucket restricts **adds** (e.g. keepers for designated keeper, non-WKs for `wk_max`, bowlers for bowling minima) and **drops** (e.g. non-keepers when adding a keeper), gated by `_drop_safe` (locks elites / anchors / certain top-order signals unless `allow_locked` widens the pool).

4. **Swap choice:** `_best_swap` enumerates add×drop pairs (capped), filters by **min_gain** / **max_quality_drop** on `_q_rank`, requires **hard-code set** to remain a subset of the iteration’s hard set (no fixing one hard rule by introducing another), ranks surviving trials by **hard improvement**, **semi improvement**, **gain**, then tie-breakers.

5. **Fallback:** If the loop exits still hard-invalid, the code may replace the XI with `select_playing_xi` + overseas optimization when that **reduces violation count** (`used_select_playing_xi_fallback` in diagnostics).

6. **Output:** Final `repair_enforce` includes `hard_constraints_satisfied`, `semi_hard_failed`, `repair_swaps` (net out/in vs **pre-repair** XI), and **`repair_diagnostics`** (see below). `selection_debug.*.xi_validation.repair_diagnostics` mirrors the same payload for UI/JSON consumers.

## B. Exact root cause of overreach

Two compounding behaviors:

1. **Late phase was too permissive:** For `iter_idx >= 80`, repair allowed **large** negative `min_gain` and **large** `max_quality_drop` (legacy: `-0.12` / `0.2`). That let `_best_swap` pick moves that fixed a constraint but **moved the XI far** from the ranked selection’s intent.

2. **No preference to preserve stronger original XI members on ties:** `_best_swap` ordered candidates by `(hard_improve, semi_improve, gain, add_name, drop_name)`. When multiple swaps had the same hard/semi/gain, the choice could drop a **better** original XI player because lexical ordering on names is arbitrary with respect to “who was higher in the pre-repair XI.”

Realism loss = **constraint satisfaction** at the expense of **composition continuity** relative to the pre-repair XI.

## C. Code change made

All changes are in `predictor.py`, confined to `_repair_xi_if_needed` and debug surfacing.

1. **Least-disruptive tie-break:** Build `orig_xi_drop_rank` from the **input** `xi` sorted by `_q_rank` descending. In `_best_swap`, after `gain`, sort prefers **higher** `orig_xi_drop_rank(drop)` — i.e. drop the player who was **weaker** in the original XI when hard/semi/gain tie.

```4552:4634:predictor.py
    # Higher index = lower original XI quality rank (prefer dropping these in tie-breaks).
    orig_xi_drop_rank = {
        p.name: i for i, p in enumerate(sorted(xi, key=lambda p: (-_q_rank(p), str(p.name or ""))))
    }
    swap_records: list[dict[str, Any]] = []
    used_fallback_reselect = False
    ...
                drop_tr = float(orig_xi_drop_rank.get(drop.name, -1))
                rank_key = (
                    float(hard_improve),
                    float(semi_improve),
                    float(gain),
                    drop_tr,
                    str(add.name),
                    str(drop.name),
                )
```

2. **More conservative late phase:** Extend strict phase to **`iter_idx < 100`** and tighten late bounds to **`min_gain = -0.06`**, **`max_quality_drop = 0.10`** (was `< 80` with `-0.12` / `0.2`).

```4667:4670:predictor.py
        # Longer strict phase + tighter late bounds reduce "anything goes" swaps (repair overreach).
        strict_phase = iter_idx < 100
        min_gain = -0.015 if strict_phase else -0.06
        max_quality_drop = 0.05 if strict_phase else 0.10
```

3. **Diagnostics:** Per-swap `swap_records` (out/in, `constraint_rule`, `quality_gain_approx`); `_repair_diagnostics_payload` with initial violation codes, `rules_applied_in_order`, net players in/out vs original XI, `used_select_playing_xi_fallback`, policy strings, and `all_swaps_non_negative_quality_gain`. Exposed under `repair_enforce["repair_diagnostics"]` and `selection_debug.*.xi_validation["repair_diagnostics"]`.

**Intentionally unchanged:** Overseas emergency `_best_swap` (`min_gain=-0.30`, `max_quality_drop=0.45`), rule priority order, `_drop_safe`, batting order, impact subs, aliases, metadata, squad fetch, and `wk_max` classifier split from fix 2.

## D. Before / after on known overreach examples

- **Heuristic tag:** `tools/validate_last_ipl_2026_matches._gap_tags` labels “repair overreach” when a **missed actual** player’s bench omission reason text contains `"repair"`. That tag is **not** the same as “post-selection repair swap count”; it can remain at **1** even when repair becomes more conservative.

- **Sample audit (SQLite, last 5 `match_results` rows, 2026-04-07):** Regenerated with  
  `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/generate_prediction_vs_actual_report.py --from-sqlite 5`.  
  **KKR vs SRH** (`cricsheet://all/1527679`) still shows overlap **9 / 11** and the same heuristic tag on KKR — the dominant gaps (e.g. Tyagi / Raghuvanshi vs predicted fringe) are **not** explained solely by post-selection repair; the fix targets **which** players repair removes when multiple swaps are tied or when late-phase loosening was too aggressive.

- **Expected delta vs pre-fix behavior:** When repair must run, swaps should **prefer dropping lower original-XI ranks** on ties and **avoid** the largest allowed quality regressions in iterations 100–119. Net XI drift vs the ranked XI should shrink in those edge cases; **valid XI** remains the goal of the same rule buckets and fallback.

## E. Validation results

| Check | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c "import app; import predictor"` | OK |
| Small prediction-vs-actual sample (`generate_prediction_vs_actual_report.py --from-sqlite 5`) | Completed; 5/5 matches audited; no pipeline crash |
| Known repair-overreach rows | Heuristic count unchanged (1); overlap on cited KKR block unchanged — no regression signal on that single aggregate; structural repair is more conservative by construction |
| Broad prediction rewrite | **No** — only `_repair_xi_if_needed` tie-break, phase thresholds, diagnostics, and debug wiring |

## F. Risks intentionally not addressed

- **Tighter late phase** may increase cases where the loop stops progressing before all semi-hard warnings clear; hard validity and `select_playing_xi` fallback behavior are unchanged in intent but may be exercised slightly differently.

- **Tie-break on `orig_xi_drop_rank`** uses the same `_q_rank` as swap gain; it does **not** encode separate “marquee” or “last match XI” signals beyond what already flows into `selection_score` / tier.

- **Emergency overseas** path remains a large allowed drop; extreme squad/constraint corners can still produce aggressive swaps.

- **Audit tag “repair overreach”** still keys off omission-summary text; improving that tag would be a **reporting** change, not done here.
