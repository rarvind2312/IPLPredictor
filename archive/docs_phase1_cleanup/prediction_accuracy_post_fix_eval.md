# Prediction accuracy — post-fix evaluation (fixes 1–4)

**Eval run:** 2026-04-07 UTC, local audit harness (`tools.validate_last_ipl_2026_matches.run_audit` + `load_match_payloads_from_sqlite(15)`), full DB/network permissions, ~7.9 minutes wall time.

**Baseline reference:** Committed report [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md) (generated 2026-04-07, **3** SQLite preloaded fixtures, URLs `cricsheet://all/1527679`, `1527678`, `1527677`).

---

## A. Baseline vs current summary

### A.1 Same three “recent preload” fixtures (apples-to-apples)

Metrics recomputed on **only** those three URLs from the fresh audit (6 team-innings):

| Metric | Baseline report (`prediction_vs_actual_report.md`) | Current re-audit (same 3 URLs) |
| --- | --- | --- |
| Mean XI overlap (/11) | **9.33** (9+8+9+9+11+10 over 6 innings) | **9.33** |
| Min overlap | **8** (Sunrisers Hyderabad) | **8** |
| Heuristic tag **repair overreach** | **1** team-inning | **1** |
| Pipeline failures | 0 | 0 |

**Interpretation:** On the same scorecard payloads the aggregate **set overlap is unchanged** at the mean level. Fixes 2–4 do not show up as a large shift in this coarse metric on this tiny slice; some **per-player** composition changed (e.g. Kolkata Knight Riders XI now includes **Angkrish Raghuvanshi** in the predicted XI on current code, whereas the baseline report listed **Anukul Roy** / **Sarthak Ranjan** among predictions — consistent with **fix 4** nudging thin-history continuity cases without moving the 9/11 count for that match).

**Fix 1 (audit identity):** Still see high **registry-bridged pair** counts in per-match text (expected); the audit layer continues to normalize aliases for overlap math without changing the predictor.

### A.2 Expanded sample (15 SQLite fixtures requested)

| Metric | Value |
| --- | --- |
| Matches requested | **15** |
| Matches audited (full prediction) | **14** |
| Matches failed | **1** (`prediction_pipeline_failed`: **team B** `hard_constraints_satisfied` false — `Top-order players 3 < 4`, `repair_exhausted_no_constraint_safe_swaps_remaining`) |
| Team-innings in aggregate | **28** |

**Mean XI overlap across all 28 innings:** **4.0 / 11**.

**Critical caveat:** The 15-row SQLite scan mixes:

- **`cricsheet://all/152767*`** — same family as the baseline (6 innings, **mean overlap 9.33**).
- **`cricsheet://ipl/13…`** — older IPL rows (22 innings) where **mean overlap is ~2.5** (derived from the split: \((28 \times 4.0 - 6 \times 9.33) / 22 \approx 2.5\)).

Those older rows are dominated by **2026 IPLT20 squad pages vs historical playing XIs** (see `squad_temporal_confound` in audit blocks), not by a sudden model regression. **Do not** use the pooled **4.0** figure as a “post-fix accuracy score” without stratifying.

### A.3 Heuristic tags (full 28-inning sample)

- **repair overreach:** still **1** (same order of magnitude as baseline; tag keys off bench **omission** text containing `"repair"`, not swap counts).

### A.4 Wicketkeeper role counts (predicted vs actual XI, classifier)

Across **28** audited innings, **20** had `wk_role_players` **≠** between actual and predicted materialized XIs (mostly **predicted &gt; actual**, e.g. predicted 3 vs actual 1–2). This reflects **definition drift** (part-time WK vs designated keeper, scorecard line-ups vs squad metadata), not necessarily hard `wk_max` repair failures. **Fix 2** cannot be validated as a simple “fewer mismatches” counter here without a tighter WK ground-truth definition.

### A.5 Missed / extra players (full 28-inning sample)

- **Top missed** (recurring): **N Wadhera (5)**, **WP Saha, HH Pandya, V Shankar, DA Miller, Noor Ahmad, Mohammed Shami, C Green, TH David, … (4 each)** — many are **stars / multi-season names** on historical scorecards absent from “current” squads.
- **Top extras** (recurring): **Avesh Khan, Matthew Breetzke, Glenn Phillips, Washington Sundar, Kagiso Rabada, Jos Buttler, Prasidh Krishna, Jasprit Bumrah, … (4 each)** — **2026 squad staples** over-selected vs those historical XIs.

On the **3-match slice**, missed/extras remain **single-count** fringe names (e.g. **Harsh Dubey**, **Shivang Kumar**, **DA Payne**; extras **Harshal Patel**, **Travis Head**, **Pat Cummins**) — i.e. **youngsters and overseas / pace assets** on specific teams, not the bulk historical list above.

---

## B. Matches / teams with biggest improvement

1. **Kolkata Knight Riders on `cricsheet://all/1527679` (within the 3-match slice):** Predicted XI now aligns more closely with **young continuity** ( **Angkrish Raghuvanshi** present post–fix 4), while overlap stays **9/11**; **Kartik Tyagi** remains missed, **Blessing Muzarabani** remains an extra — remaining gap is **bowling pick / overseas**, not only repair.
2. **Gujarat Titans on the same slice:** **11/11** overlap in both baseline and current — no regression.
3. **No clear global “winner” across the older `cricsheet://ipl/*` rows:** Overlap stays very low; “improvement” is not measurable there until **squad snapshot** matches the match year or historical XIs are filtered to **IPL 2026–only** fixtures.

---

## C. Failures or weak areas still remaining

1. **Hard pipeline failure (1/15):** One fixture leaves **team B** invalid after repair (**`top_order_min`**). This is a **constraint / repair edge case**, not fixed by fixes 1–4 scope.
2. **SRH-style fringe misses (3-match slice):** Three **actual** bowlers/youngsters missed on one innings (**Harsh Dubey**, **Shivang Kumar**, **DA Payne**) while **Head / Cummins / Harshal** are extras — **pace/overseas + depth** still wrong.
3. **Repair / bench narrative:** **repair overreach** tag persists where omission reasons still mention `"repair"`; fix 3 tightens swaps but does not remove that heuristic.
4. **WK count mismatches:** Widespread **pred vs actual** `wk_role_players` differences on the large sample; needs **clearer evaluation definition**, not only classifier tweaks.
5. **Batting-order layer:** Logs show repeated **`slot_constraint_legal_assignment_unavailable`**, **`rule_conflict: batting_order_guardrails_unsatisfied`**, and **`discrete_slots_*`** for many franchises — **ordering stress after XI is chosen** is a major **runtime / UX** signal even when set overlap is high.
6. **Stratified historical rows:** **`cricsheet://ipl/*`** rows with overlap **1–4** dominate the pooled mean; they are **weak tests** of post-fix quality unless squads are time-aligned.

---

## D. Whether a Fix 5 is justified

**Yes.** Fixes 1–4 address **identity**, **WK semantics**, **repair conservatism**, and **low-history scoring**, but the eval shows:

- a **hard failure** on **`top_order_min` / repair exhaustion**,
- persistent **fringe vs star** composition errors on real recent fixtures,
- pervasive **batting-order conflicts** in logs,
- and **unstratified historical data** inflating apparent error rates.

A **narrow Fix 5** is justified **if** it targets **one** of these with measurable effect (e.g. fewer hard failures or fewer guardrail conflicts) without a full engine rewrite.

---

## E. Recommended single next fix (only one)

**Recommend Fix 5:** **Reduce batting-order / slot-guardrail conflict after XI selection** — i.e. when the XI is already fixed, **reconcile discrete slot assignment** (or relax **specific** guardrails that repeatedly conflict with the ranked XI) so the pipeline stops fighting the same order across **slot_constraint_legal_assignment_unavailable** / **rule_conflict** / **revert** loops.

**Why this one (vs overseas-only or squad-year tooling):**  
It is **internal to the prediction path**, **observable in logs on almost every match**, and **orthogonal** to fixes 1–4. Overseas/squad-year issues remain huge on **`cricsheet://ipl/*`** rows but are largely **data / evaluation design**; a Fix 5 there would be “change the audit” or “time-machine squads,” not a small predictor fix.

**Explicitly not recommended as Fix 5 in this pass:** broad **overseas target** redesign, **impact subs**, or **metadata precedence** — scope and interaction risk are too high for a single follow-up.

---

## Evidence pointers

- Baseline markdown: [`docs/prediction_vs_actual_report.md`](prediction_vs_actual_report.md) (3 matches, tag table, per-match overlaps).
- Audit harness: `tools/validate_last_ipl_2026_matches.py` (`run_audit`, `_gap_tags`, `team_structure` for WK counts).
- Raw aggregates for this write-up were produced from `load_match_payloads_from_sqlite(15)` + `run_audit` JSON-style summary (see Cursor terminal capture from eval run, exit 0, **matches_audited: 14**, **matches_failed: 1**).
