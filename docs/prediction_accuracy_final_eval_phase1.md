# Prediction accuracy — final phase-1 evaluation (after fixes 1–5)

**Eval run:** 2026-04-07 UTC (wall ~7 minutes), same harness as `tools/validate_last_ipl_2026_matches.run_audit` and `tools/generate_prediction_vs_actual_report.load_match_payloads_from_sqlite`: full `predictor.run_prediction`, current IPLT20 squads, registry-aware XI diff.

**Fixes in scope:** (1) audit identity matching, (2) wicketkeeper cap refinement, (3) repair overreach reduction, (4) low-history continuity nudge, (5) batting-order softening (slot DP / revert behavior).

**No code changes** were made for this document.

---

## Method (tasks 1–4)

### 1. Prediction vs actual — recent fixtures only

- **Command pattern:** `load_match_payloads_from_sqlite(3)` → `run_audit(payloads_by_url=...)`.
- **Payloads:** Same family as [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md): `cricsheet://all/1527679`, `1527678`, `1527677` (order follows DB `id DESC` scan until three IPL `all/*` rows are collected).
- **Outcome:** **3** matches audited, **0** `prediction_pipeline_failed`.

### 2. Separate summaries (recent slice)

| Area | Recent 3-match slice (6 team-innings) |
| --- | --- |
| **XI overlap** | Mean **9.33 / 11** (sum **56**); min **8** (Sunrisers Hyderabad). |
| **Failed predictions** | **0** hard failures. |
| **Wicketkeeper-related** | Classifier-based **`wk_role_players`** count on materialized actual vs predicted XI: **2 / 6** innings differ (definition is coarse; not the same as “wrong named keeper”). |
| **Low-history / new-player** | Kolkata Knight Riders predicted XI now includes **Angkrish Raghuvanshi** in the playing XI (aligned with fix 4 continuity nudge); overlap remains **9 / 11** vs scorecard — swap changed **who** is wrong (e.g. **Umran Malik**, **Blessing Muzarabani** vs **Kartik Tyagi**, **AS Roy**) rather than lifting overlap on this row. Sunrisers Hyderabad remains **8 / 11** with the same structural miss pattern (stars vs actual fringe bowlers). |
| **Batting-order metrics** | Summed over 6 innings: **top-3 positional = 4**, **openers(2) = 3**, **middle 4–7 set overlap = 6**, **lower 8–11 set = 1** — **identical** to the per-match figures already recorded in the committed [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md) for these fixtures. |

### 2b. Mixed SQLite sample (explicitly **not** “recent 2026-only”)

To align with [`docs/prediction_accuracy_post_fix_eval.md`](prediction_accuracy_post_fix_eval.md) and to show **squad-year confounding** separately:

- **`load_match_payloads_from_sqlite(15)`:** **14** matches OK, **1** `prediction_pipeline_failed`.
- **Failure (this run):** `cricsheet://ipl/1359540` — team B **`Top-order players 3 < 4`** after repair (`repair_exhausted_no_constraint_safe_swaps_remaining`). (The older diagnosis cited a **wk_max** failure on a different historical URL; the **class** of problem remains **hard constraints + repair exhaustion**, not a new crash type.)
- **Pooled 28 innings:** mean XI overlap **4.0 / 11** — **dominated** by `cricsheet://ipl/*` rows vs **2026** squads (see post-fix eval stratification). **Do not** interpret **4.0** as the “true” post-fix accuracy; use the **recent 3-match** block above for model-facing signal.

| Area | 15-row mixed sample (28 innings) |
| --- | --- |
| **XI overlap** | Mean **4.0 / 11** (pooled); recent `all/152767*` subsample still **~9.33** when isolated (see post-fix eval math). |
| **Failed predictions** | **1 / 15** matches. |
| **Wicketkeeper-related** | **`wk_role_players`** pred vs actual count mismatch: **21 / 28** innings (vs **20 / 28** in the 2026-04-07 post-fix eval — **no material change**). |
| **Batting-order metrics (pooled)** | top-3 sum **4**, openers **3**, middle 4–7 **9**, lower 8–11 **4** — **not** comparable to the 3-match slice because innings count and team mix differ. |

### 3. Comparison to the earliest **trustworthy** baseline

| Baseline | What it measures | vs current recent slice |
| --- | --- | --- |
| **Pre–fix 1 (literal keys)** | [`docs/prediction_accuracy_fix1_results.md`](prediction_accuracy_fix1_results.md): same-style fixtures showed **~1 / 11**-scale overlap for sides like KKR when only `normalize_player_key` was used. | **Large gain** from fix 1 alone: registry-aware audit keys report **mean 9.33 / 11** on the same **2026 `cricsheet://all/*`** family. |
| **Committed report (post–fix 1, 2026-04-07)** | [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md): mean **9.33**, batting sums **4 / 3 / 6 / 1**, **1** `repair overreach` tag, **0** failures. | **Aggregate overlap and batting-order report metrics unchanged** after fixes 2–5 on this slice; **composition** on some teams shifted (e.g. KKR XI names). |

### 4. What clearly improved vs barely changed vs defer

- **Clearly improved (phase arc):** **Trustworthy measurement** (fix 1) and **WK-cap feasibility** on multi–WK-Batter squads (fix 2 — see fix 2 doc; avoids the narrow “three listed keepers = hard stop” failure mode).
- **Barely changed on the recent 3-match scorecard test:** **Mean XI overlap**, **coarse batting-order tallies**, **repair-overreach tag count (1)**, **one mixed-sample pipeline failure**.
- **Defer:** Historical **`cricsheet://ipl/*`** rows vs **current** squads; **impact sub** validation without parser ground truth; **broad** batting **band** vs **discrete slot** reconciliation; **WK count** as an eval metric until pred/actual **designated keeper** is defined consistently with scorecards.

---

## A. What improved meaningfully

1. **Measurement and interpretability (fix 1):** Overlap, missed/extra, and bowling dedupe use **`audit_player_identity_key`** — the dominant step-change vs pre-fix literal keys (documented in fix 1 and [`docs/prediction_accuracy_diagnosis.md`](prediction_accuracy_diagnosis.md)).
2. **Wicketkeeper hard-cap behavior (fix 2):** Separates **cap-relevant** WK role from **broad keeper candidacy**, reducing pathological **`wk_max`** + repair exhaustion on stacked WK-Batter listings (fix 2 write-up).
3. **Honest stratification of eval (fix 1 + docs):** Reports and post-fix eval explicitly separate **2026 `all/*` recent** rows from **`ipl/*` historical** rows where **mean overlap ~2.5/11** is expected under **roster drift** — avoiding false “regression” narratives from pooled means.

---

## B. What improved slightly

1. **Repair conservatism (fix 3):** Heuristic **`repair overreach`** count stayed **1** on the recent 3-match report pattern — order of magnitude unchanged; intent was narrower swaps, not zero tags.
2. **Low-history continuity (fix 4):** Observable on **KKR** as **Angkrish Raghuvanshi** in the **predicted XI** on the fresh audit; **9/11** overlap **unchanged** vs the committed report — nudge shifts **which** fringe players compete for slots, not a guaranteed overlap jump on one fixture.
3. **Batting-order softening (fix 5):** **Reported** positional aggregates on the **same three fixtures** match the pre–fix 5 committed markdown; internal order and logging can differ without moving these coarse scorecard-facing tallies. **Band** `rule_conflict: batting_order_guardrails_unsatisfied` lines still appear frequently in console output (expected per fix 5 scope).

---

## C. What did not improve enough

1. **Fringe vs star composition** on **SRH** (and similar): **8 / 11** with recurring missed **young bowlers** and extra **marquee** names — not solved by a small continuity bump alone.
2. **Pooled historical eval:** Mean **4.0 / 11** on 15 SQLite rows remains **misleading** without **year-aligned squads** or filtering.
3. **Hard pipeline failures:** **One** failure per 15-row scan persists (**constraint + repair exhaustion**; this run **top_order_min** on team B).
4. **WK “accuracy” as count match:** **21 / 28** pred vs actual **`wk_role_players`** mismatches on the mixed sample — still **no** clean KPI without a tighter keeper definition (called out in [`docs/prediction_accuracy_post_fix_eval.md`](prediction_accuracy_post_fix_eval.md)).
5. **Batting-order realism vs scorecard:** Coarse **top-3 / openers / middle** sums stay low; **fix 5** targeted **internal** slot friction, not full alignment to realized orders.

---

## D. Recommended stop / go for more fixes

**Recommendation: STOP** the current **narrow, single-issue** predictor patch series for phase 1.

**Rationale:** On the **only** slice that fairly tests the last five fixes together (**recent `cricsheet://all/*` + current squads**), **aggregate XI overlap and report batting-order buckets are stable** vs the post–fix 1 baseline; remaining gaps are **structural** (squad-year, fringe selection, band vs discrete order, incomplete ground truth for impact). Further **small** code tweaks risk **interaction debt** without a new **evaluation contract** (time-aligned rosters or IPL-season-scoped fixtures).

**GO again when:** You accept a **new scope** — e.g. **squad snapshot by season**, **expanded golden fixtures**, or a **single** deep refactor from section E — not another ad-hoc line edit.

---

## E. Top 3 future deeper refactors (only if stopping narrow work)

1. **Time-aligned squad and eval harness:** Ingest or pin **official XI / squad as-of match date** (or restrict audits to **current-season** fixtures only) so overlap metrics are not dominated by **roster drift** on `cricsheet://ipl/*` rows.
2. **Batting order: unify or relax guardrails with data:** Reconcile **registry discrete slots**, **band min/max**, and **real IPL** orders (or learn/order from `team_match_xi` / scorecard priors) instead of alternating **DP**, **revert**, and **rule_conflict** patches.
3. **Impact Player end-to-end:** Parse **actual** impact from feed (or manual labels), then align **`impact_subs_engine`** with observed substitution patterns — today the report **cannot** score impact accuracy.

---

## Evidence pointers

| Artifact | Role |
| --- | --- |
| This run | `load_match_payloads_from_sqlite(3|15)` + `run_audit` (2026-04-07); metrics summarized above. |
| Recent baseline markdown | [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md) |
| Pre–fix 1 / stratified narrative | [`docs/prediction_accuracy_fix1_results.md`](prediction_accuracy_fix1_results.md), [`docs/prediction_accuracy_post_fix_eval.md`](prediction_accuracy_post_fix_eval.md) |
| Per-fix detail | [`docs/prediction_accuracy_fix2_wk_diagnosis_and_results.md`](prediction_accuracy_fix2_wk_diagnosis_and_results.md) … [`docs/prediction_accuracy_fix5_batting_order_results.md`](docs/prediction_accuracy_fix5_batting_order_results.md) |
