# Prediction accuracy fix 5 — batting order (slot / guardrail realism)

## A. Current batting-order conflict path

1. **Entry:** `_assign_batting_order_stage` calls `build_batting_order` for an **already selected** XI of 11.

2. **Heuristic ordering:** Openers and band-based sorts produce a candidate list; `_batting_order_strict_names_for_xi` enforces a permutation of the XI.

3. **Elite / band swaps:** Generic min–max **band** guardrails (`allowed_min` / `allowed_max` from slot eligibility) swap players; adjacent strength tie-break; **role regroup** (bat vs lower_order vs specialist).

4. **Tail structure:** `_enforce_specialist_bowler_tail` and `_enforce_lower_order_overflow` may reorder before discrete optimization.

5. **Discrete slot DP:** `_optimize_slot_constraints` runs a **11!-style** DP assignment minimizing `_slot_cost` with **hard** legality (`allow_illegal=False`) first. If **no** full permutation has finite cost, it logs `slot_constraint_legal_assignment_unavailable` and previously **returned the pre-DP order unchanged**.

6. **Revert:** If the order after tail+DP was not **fully** discrete-legal (`_order_discrete_legal`), the pipeline **reverted to `bo_pre_tail_optimize`**, i.e. **before specialist-bowler tail and lower-order overflow** — wiping both tail fixes and any partial slot progress.

7. **Post-pass:** Per-player `history_debug` fields and `rule_conflict: batting_order_guardrails_unsatisfied` when **band** min/max still fail after final positions.

## B. Exact root cause addressed

- **No feasible legal discrete assignment:** For many real XIs, registry **allowed_slots** sets are tight enough that **no** permutation satisfies all slots simultaneously under the old hard-DP gate. The code then **skipped** a principled fallback and often **reverted past tail enforcement**, amplifying mismatch vs the intended XI shape.

- **Revert too coarse:** Reverting to **pre-tail** order discarded **useful** tail/overflow structure when only the **slot-DP** step was problematic.

**Fix 5** adds: (1) **soft DP** (finite penalties for discrete violations, stronger **preservation of the incoming order** into the DP step); (2) **smarter revert** — only revert the slot step when discrete violations **increase** vs **post-tail pre-slot** order; otherwise **keep** the DP result; (3) **DP tie-break** on equal cost (lexicographic name tail) for stability; (4) **team diagnostics** on every XI player.

**Not addressed here:** Band min/max conflicts (`rule_conflict`) when profiles are intrinsically inconsistent with the XI — those need profile or classifier follow-up, not order DP alone.

## C. Code change made

All edits in `predictor.build_batting_order` (`predictor.py`).

1. **`slot_optimization_debug`** — tracks legal vs soft DP, input/output orders, discrete legality counts, `revert_steps`.

2. **`_slot_cost`:** For `allow_illegal=True`, weight on preserving **pre-DP slot index** raised (`prior_w = 0.72` vs `0.35`) so soft assignments stay closer to the tail-shaped order.

3. **`_solve`:** Tie-break when `total` is equal — prefer lexicographically smaller name permutation.

4. **After hard DP fails:** `soft = _solve_assignment(True)`; on success set `soft_dp_fallback_used`, log `batting_order: slot_constraint_soft_dp_fallback_applied` (INFO).

5. **Snapshot `after_tail_before_slot_opt`** after tail + overflow, **before** DP.

6. **Revert logic:** If not fully discrete-legal after DP, **keep** the DP order when `_discrete_violation_count(final) <= _discrete_violation_count(after_tail_before_slot_opt)`; else revert to **post-tail pre-slot** first, then legacy chain to `bo_pre_tail_optimize` / `bo_after_fallback_snapshot`.

7. **Diagnostics:** After band-conflict detection, each `p` in `xi` gets `history_debug["batting_order_slot_pipeline_debug"]` with `moves_post_tail_to_final`, `band_guardrail_conflicts_remaining`, and the slot-optimization flags.

`rules_spec.py` was **not** changed.

## D. Before / after examples on recent fixtures

**Evidence source:** [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md) (baseline per-match batting-order lines) vs fresh `run_audit(load_match_payloads_from_sqlite(3))` after this change.

**Aggregate batting-order metrics (6 team-innings, same three `cricsheet://all/152767*` fixtures):**

| Metric (sum over 6 innings) | Baseline report | After fix 5 |
| --- | --- | --- |
| top-3 positional matches | 4 | 4 |
| openers (2) positional | 3 | 3 |
| middle slots 4–7 set overlap | 6 | 6 |

So on this slice, **positional overlap vs scorecard is unchanged** at the summed level (expected: scorecard order and squad frictions dominate; fix targets **internal** legality and **fewer destructive reverts**, not oracle batting order).

**Qualitative:** Logs still show `slot_constraint_legal_assignment_unavailable` when hard DP is infeasible; **soft DP** then runs (see `logger.info` `slot_constraint_soft_dp_fallback_applied` when root logger shows INFO). **GT / LSG / DC / KKR** still emit `rule_conflict: batting_order_guardrails_unsatisfied` where **band** ranges disagree with slot families — out of scope for this narrow pass.

**Example internal change:** Gujarat Titans predicted order after soft DP can reorder relative to the strict heuristic (e.g. Rashid / top-order cluster) while **XI membership is unchanged** (still a permutation of the same 11 names).

## E. Validation results

| Check | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c "import app; import predictor"` | OK |
| `python -m unittest tests.test_predictor_pipeline_stages` | OK (4 tests) |
| 3-match SQLite audit — summed `batting_order_overlap` | Same totals as baseline report aggregate (section D) |
| XI membership | Still a **permutation** of the selected XI; no XI-selection code touched |

## F. Risks intentionally not addressed

- **Band guardrail errors** (`allowed` `[lo, hi]` vs final position) can remain; soft DP optimizes **discrete registry slots** and priors, not the separate min/max loop outcomes.

- **Residual discrete violations** may be **kept on purpose** when they are no worse than post-tail — could leave rare illegal registry slot assignments if the math forces it; cost function still penalizes them heavily.

- **Multiple `build_batting_order` calls** per prediction (e.g. scenario branches) may duplicate log lines; not changed here.

- **Impact subs, repair, scoring, low-history nudge, WK cap** — untouched per scope.
