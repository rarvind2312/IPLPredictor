# Prediction accuracy diagnosis

**Scope:** XI selection, batting order, bowling-usage comparison, and impact-sub outputs versus scorecard ground truth. Based on `docs/prediction_vs_actual_report.md` (2026-04-07 run), `tools/validate_last_ipl_2026_matches.py`, `tools/generate_prediction_vs_actual_report.py`, and the live prediction path in `predictor.py` / related modules.

**Non-goals (per request):** No engine rewrites; no broad cleanup proposals.

---

## A. Top recurring mismatch patterns

### 1. Name / identity mismatch (Cricsheet vs squad display) — **measurement + downstream history**

**Evidence:** In the report, many “missed” and “extra” pairs are the same human player under different strings, e.g. KKR vs SRH: `FH Allen` vs `Finn Allen`, `SP Narine` vs `Sunil Narine`, `CV Varun` vs `Varun Chakaravarthy`, `RG Sharma` vs `Rohit Sharma`, `N. Tilak Varma` vs `Tilak Varma`. A quick key check shows `learner.normalize_player_key` does **not** unify these pairs (e.g. `fh allen` ≠ `finn allen`; `cv varun` ≠ `varun chakaravarthy`).

**Effect:** Per-match overlap counts and “mispredicted player” aggregates are **pessimistic**: true XI accuracy is understated wherever Cricsheet uses initials and the model/squad uses full names (or vice versa). Bowling-usage rows that say “bowled but not in predicted XI” are **partly the same issue** when bowler names differ only by initials.

**Distinct from model logic:** This hits the **audit layer** first; it may also reduce **history / continuity** quality if joins use the same naive keys.

---

### 2. Squad snapshot vs match date (historical fixtures with **current** IPLT20 squad)

**Evidence:** Match 5 pairs **2023-05-26** scorecard (GT vs MI, `cricsheet://ipl/1370352`) with **`squad_fetch.fetch_squad_for_slug`** (today’s official pages), as documented in the report header. Predicted MI XI includes players not in the 2023 actual XI (e.g. Trent Boult, Quinton de Kock, Ryan Rickelton); overlap **0 / 11** for MI.

**Effect:** This is largely **wrong eligible pool**, not a fair test of “stale history weight” alone. Any conclusion that “the model always misses MI” for that row confounds **roster drift** with selection quality.

---

### 3. Hard XI constraints vs real teams (overseas, bowling mix, **wicketkeeper cap**)

**Evidence (logs / code path):** Console output during the same audit includes `select_playing_xi: unable to build valid XI under hard constraints errs=['Overseas 5 > 4']` and, for one fixture, **pipeline failure**: `Team B hard constraints unsatisfied after repair: ['Max 2 wicketkeepers allowed, found 3'] | reason=repair_exhausted_no_constraint_safe_swaps_remaining` (CSK as team B, `cricsheet://ipl/1370353`). The raised error is emitted in `predictor._run_prediction_inner` when `xi_*_repair_enforce["hard_constraints_satisfied"]` is false after `_repair_xi_if_needed`.

**Rules:** `rules_spec.CANONICAL_RULE_SPEC` fixes overseas min/max, min bowling options, pacers, spinners, and `rules_xi.validate_xi` enforces **`Max 2 wicketkeepers`** via `classify_player(p).is_wk_role_player` counts (`rules_xi.py`).

**Effect:** When the **official squad** lists multiple players as WK-Batter (or classifier marks them as wicketkeeper-role), the engine may be unable to satisfy **wk_max** while also keeping marquee locks — repair exhausts (`predictor._repair_xi_if_needed`). That is a **constraint + role classification + repair** interaction, not a single “bad score” bug.

---

### 4. Batting-order guardrails vs realized orders

**Evidence:** Standard logs during runs: `batting_order: slot_constraint_legal_assignment_unavailable`, `discrete_slots_still_illegal_after_revert`, `rule_conflict: batting_order_guardrails_unsatisfied` with conflicts such as players at slot 4 with `allowed: [5, 7]`. Spec: `rules_spec.CANONICAL_RULE_SPEC["batting_order"]`; implementation: `predictor._assign_batting_order_stage` and `_optimize_slot_constraints` (~5774+).

**Effect:** Predicted batting order is **forced** into discrete slot families; real IPL lineups often violate those abstractions. Report columns “top-3 positional matches”, “middle 4–7 set overlap” will systematically underperform even when XI membership is good.

---

### 5. Repair-driven XI distortion (“repair overreach” tag)

**Evidence:** Report section A tags **`repair overreach`** (count 1 on this run). The audit derives tags from `prediction_layer_debug` omission summaries when the reason text contains `"repair"` (`tools/validate_last_ipl_2026_matches.py`, `_gap_tags`). KKR block lists that tag alongside a predicted XI that is far from actual — consistent with **post-selection swaps** in `_repair_xi_if_needed` fixing hard/semi-hard violations at the cost of plausibility.

**Effect:** **Constraint repair** can replace players to satisfy `rules_xi.validate_xi`, producing XIs that are **valid under rules** but **not** close to the team’s real pick.

---

### 6. Impact subs: model-only vs **no** scorecard ground truth

**Evidence:** Every per-team block notes actual Impact Player is **not** in the parser schema; only `predictor` → `impact_subs` → `impact_subs_engine.rank_impact_sub_candidates` output is shown.

**Effect:** Discrepancies here are **unmeasurable** in this pipeline; any “mismatch” is speculative. The engine uses venue/team patterns and roles (`impact_subs_engine.py`, DB-backed `team_selection_patterns`), not the same signals captains use on the day.

---

### 7. Young / low-history players (where data exists)

**Evidence:** SRH actual XI includes names like **E Malinga**, **Harsh Dubey**, **Shivang Kumar** with large “missed” lists vs a predicted XI anchored on **Travis Head**, **Pat Cummins**, etc. KKR impact list includes **Angkrish Raghuvanshi** while the model’s predicted XI omitted several actual youngsters.

**Effect:** Where **history and tiers** dominate `selection_score` (`select_playing_xi` ordering by tier + `selection_score`), newcomers are easy to under-rank. This is **scoring + data coverage**, not a single bug.

---

## B. Root-cause mapping

| Pattern | Primary files / functions | Category | Team scope |
|--------|---------------------------|----------|------------|
| Initials vs full names in overlap stats | `tools/validate_last_ipl_2026_matches.py` (`_nk` / `_xi_diff`); `learner.normalize_player_key` | **Data** (comparison + possibly history joins) | All teams on Cricsheet-style cards |
| Current squad vs past match | `squad_fetch.fetch_squad_for_slug`, `validate_last_ipl_2026_matches._fetch_current_squad` | **Data** (audit setup) | Any historical row |
| Overseas / bowling / top-order / tier mins | `rules_spec.py`, `rules_xi.validate_xi`, `predictor.select_playing_xi`, `_try_build_xi` / `_build_xi_with_hard_role_quotas` | **Constraints** | All teams; worse when squad shape is extreme |
| Wicketkeeper cap + repair failure | `rules_xi.validate_xi` (`wk_max`), `player_role_classifier.classify_player`, `predictor._repair_xi_if_needed`, raise in `_run_prediction_inner` ~7020–7055 | **Constraints + role classification + fallback/repair** | Observed: **CSK** (3 WK-role classified); principle applies to any multi-WK squad |
| Condition branches + overseas tuning | `predictor._run_xi_selection_stage`, `_apply_condition_adjustments_from_base`, `_optimize_overseas_preference` | **Scoring + constraints** | All |
| Batting order vs reality | `rules_spec` batting_order hard constraints, `predictor._assign_batting_order_stage`, `_optimize_slot_constraints` | **Constraints** (ordering) | All |
| Repair swaps | `predictor._repair_xi_if_needed` | **Fallback / repair** | All when violations present |
| Impact sub ordering | `predictor.impact_subs`, `impact_subs_engine.rank_impact_sub_candidates` | **Scoring** (separate module + DB patterns) | All; **not validated** vs actual in report |

---

## C. Failure bucket analysis

### `prediction_pipeline_failed` fixtures

- **Count (this report):** 1 — `cricsheet://ipl/1370353`, GT vs CSK.
- **Mechanism:** After XI selection and repair, `predictor._run_prediction_inner` requires `xi_b_repair_enforce["hard_constraints_satisfied"]`; CSK fails with **`wk_max`** (`Max 2 wicketkeepers allowed, found 3`) and `repair_failure_reason=repair_exhausted_no_constraint_safe_swaps_remaining`.
- **Implication:** Not a random crash — **hard stop by design** when repair cannot fix violations. Actual XIs in the error section show **2023-era** names (e.g. Conway, Rayudu on CSK) while the **predictor uses the current squad graph**, compounding difficulty.

### Keeper constraint failures

- **Rule:** `rules_xi.validate_xi` counts `classify_player(p).is_wk_role_player` and violates if `> 2`.
- **Observed debug (console):** Multiple CSK list players with `classify_is_wk_role_player: true` (e.g. Sanju Samson, Kartik Sharma, MS Dhoni) — consistent with **squad role_bucket / meta** marking several as WK-Batter.
- **Repair:** `_repair_xi_if_needed` must swap under `_drop_safe` / marquee locks; if too many WK-role players are “locked”, **no safe swap** remains → exhaustion.

### Bowling-options / pace / spin constraint failures

- **Spec:** `MIN_BOWLING_OPTIONS`, `MIN_PACE_OPTIONS_IN_XI`, `MIN_SPINNER_OPTIONS_IN_XI` in `rules_spec` / `config`.
- **Evidence:** Logs show `select_playing_xi: unable to build valid XI under hard constraints errs=['Overseas 5 > 4']` (and combined with WK in other traces). **Overseas cap** interacts with the same ranked list as bowling composition.
- **Report symptom:** “Predicted XI bowling options with no recorded overs” often mixes **true** tactical non-use with **name-key mismatch** for the same bowler.

### Low-history / new-player cases

- **Evidence:** SRH, PBKS, and KKR rows show actual youngsters or fringe names missed while high-salary or high-tier names appear in predicted XIs.
- **Mechanism:** `select_playing_xi` sorts on `_tier_val`, `selection_score`, `composite` (`predictor.py`); thin history → lower scores → rarely in top 11 unless constraints force them.

---

## D. First low-risk accuracy improvement (single recommendation)

**Use registry-aware identity for audit overlap (and optionally logging), not a full engine change.**

- **What:** When comparing predicted vs actual in `tools/validate_last_ipl_2026_matches.py`, resolve each name to a stable id if possible (e.g. `player_registry` canonical key / alias lists) before computing overlap, missed, and extras — in addition to `learner.normalize_player_key`.
- **Why it’s high value:** Demonstrated key failures (`FH Allen`/`Finn Allen`, `SP Narine`/`Sunil Narine`, etc.) **inflate** apparent error; fixing measurement gives a trustworthy signal for real model work.
- **Why it’s low risk:** Touches **validation/reporting** only; does not change `select_playing_xi`, repair, or rules.
- **Files:** `tools/validate_last_ipl_2026_matches.py` (`_xi_diff`, `_nk` usage); `player_registry.py` (existing alias machinery).

---

## E. Accuracy roadmap (next five)

1. **Easiest + highest value:** **Audit-only** name resolution (section D) + **explicit flag** in generated reports when `match_date` is not in the same IPL season as “current squad” snapshot (`generate_prediction_vs_actual_report.py` / audit meta), so historical rows are not misread as pure model error.

2. **Next best:** Tighten **when a squad-listed WK-Batter counts as `is_wk_role_player`** for XI caps (e.g. secondary signals from recent `team_match_xi` / designated keeper usage) in `player_role_classifier` / attachment of `SquadPlayer` metadata — targets **CSK-style** `wk_max` repair failures without loosening rules globally.

3. **Medium risk:** Broaden **repair search** for `wk_max` / overseas conflicts (e.g. more candidates, or controlled unlock of non-playing WK labels) inside `predictor._repair_xi_if_needed` — improves feasibility but can change many XIs; needs golden tests.

4. **Higher risk:** Revisit **batting-order discrete slots** vs real data (`rules_spec` + `_assign_batting_order_stage`): relaxing or re-parameterizing guardrails changes almost every user-visible order.

5. **Deep refactor later:** **Impact Player** end-to-end: parse actual impact from feed, align `impact_subs_engine` weights with observed substitution patterns, and optionally train or calibrate from outcomes.

---

## Appendix: Evidence pointers

| Item | Where |
|------|--------|
| Report overlap / bowling / failures | `docs/prediction_vs_actual_report.md` |
| Audit pipeline + tags + comparisons | `tools/validate_last_ipl_2026_matches.py` |
| Markdown aggregation | `tools/generate_prediction_vs_actual_report.py` |
| XI selection + repair | `predictor.py` — `select_playing_xi`, `_run_xi_selection_stage`, `_repair_xi_if_needed` |
| Hard failure raise | `predictor.py` `_run_prediction_inner` ~7020–7055 |
| Rule numbers | `rules_spec.py`, enforced in `rules_xi.py` |
| Impact ranking | `predictor.impact_subs`, `impact_subs_engine.py` |
