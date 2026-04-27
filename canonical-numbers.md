# Canonical Numbers — Phase A Verification

Generated 2026-04-27 from `report/academic/thesis.tex` and `results/metrics/*.csv`.
Read-only audit, no pipeline reruns.

---

## ⚠️ Three findings that require your decision before Phase B

1. **Thesis headline numbers do NOT match the bootstrap CSV** (~0.003 systematic gap on raw + calibrated PR/ROC). The thesis values are LOWER than the CSV values. Both claim "1,000 replicates, seed=42" — so one of them is from a stale run. The thesis values are also hard-coded in `scripts/ablate_esm2.py:196` (`phase1_test_pr = 0.8273`), so the thesis ecosystem is internally consistent; the bootstrap CSV is the outlier. **Recommendation: trust the thesis (0.827 calibrated) for the defense — that is what every downstream artifact and the deck already use. Flag the CSV/thesis drift in a follow-up D-task; do not chase it inside the defense rewrite.**

2. **Equation 5.4 in the thesis is BACKWARDS relative to the code.**
   - Thesis (line 3782-3787): `s = α·r_XGB/n + (1−α)·r_ESM/n`
   - Code (`scripts/run_rank_fusion_esm2.py:98-100`): `s = (1−α)·r_XGB + α·r_ESM` with the comment `# Weight α goes on ESM-2; (1 − α) on XGBoost.`

   With α* = 0.175:
   - Under the **CODE** convention → ESM gets 0.175 (small), XGB gets 0.825 (large). This is the only one that explains the numbers (tuned ROC 0.9412 stays close to XGB-alone 0.9380, not close to ESM-alone 0.7345).
   - Under the **THESIS** convention → XGB gets 0.175, ESM gets 0.825. Mathematically inconsistent with the actual fusion result.

   **The current defense narrative ("α=0.175 down-weights the weaker PLM signal") is therefore CORRECT relative to the code.** The user's proposed B1 rewrite ("most rank weight on ESM-2") is correct relative to the thesis equation but contradicts the code. **Recommendation: keep the defense's original semantics (XGB-heavy fusion) and instead fix the convention statement on the slide to match the code; flag the thesis equation as an erratum (Phase E candidate, not part of B1 as currently scoped).**

3. **Thesis Table 7.2 lists D1–D8 (eight deficiencies), not D1–D6.** The user's spec says "D1–D6 list" but the thesis canonical source has eight. D7 (baseline-coverage confound) and D8 (denovo-db pre-constraint predictions) are real and present. **Recommendation: extend C3 from "D1–D6" to D1–D8 to match the thesis verbatim. Compressed one-liners fit in a single backup-style box on slide 35-36.**

---

## SECTION 1 — Verification table (8+ critical metrics)

| # | Metric | thesis.tex value (with section ref) | CSV value (with file path) | Match? |
|---|--------|-------------------------------------|----------------------------|--------|
| 1 | Test PR-AUC (calibrated) | **0.8273** [0.8185, 0.8352] — Sec 5.6.1, Table 5.4, line 4036 | mean **0.8304** [0.8222, 0.8381] — `xgboost_bootstrap_ci.csv:5` (test_calibrated). Point estimate 0.8304 in `xgboost_calibrated_metrics.csv:5` and `phase21_calibration_comparison.csv:3` | ✗ **RED FLAG** — gap of −0.0031. Thesis hard-codes 0.8273 in `scripts/ablate_esm2.py:196`. **Trust thesis** for defense (deck uses 0.827); flag CSV drift in deficiencies log. |
| 2 | Test PR-AUC (raw) | **0.8355** [0.8265, 0.8434] — Sec 5.6.1, Table 5.4, line 4031 | mean **0.8382** [0.8298, 0.8459] — `xgboost_bootstrap_ci.csv:4` (test_raw) | ✗ **RED FLAG** — same −0.003 systematic gap as row 1, same recommendation |
| 3 | Test ROC-AUC (calibrated) | **0.9376** [0.9346, 0.9404] — Sec 5.6.1, Table 5.4, line 4035 | mean **0.9380** [0.9350, 0.9409] — `xgboost_bootstrap_ci.csv:5` | ✗ small −0.0004 gap, same root cause |
| 3b | Test ROC-AUC (raw) | **0.9378** [0.9348, 0.9407] — Sec 5.6.1, Table 5.4, line 4030 | mean **0.9384** [0.9353, 0.9414] — `xgboost_bootstrap_ci.csv:4` | ✗ small −0.0006 gap, same root cause |
| 4 | Phase-2.1 augmented full PR-AUC (calibrated) | **0.8647** [0.8580, 0.8714] — Sec 5.9.2, Table 5.11, line 5133 | **0.8647** — `phase21_calibration_comparison.csv:7` (phase21 test_calibrated). CIs match `results/metrics/phase21/xgboost_bootstrap_ci.csv` (mean 0.8647, [0.8580, 0.8714]) | ✓ exact match |
| 4b | Phase-2.1 full ROC-AUC (calibrated) | **0.9476** [0.9448, 0.9503] — Table 5.11, line 5133 | **0.9476** — `phase21_calibration_comparison.csv:7` | ✓ exact match |
| 4c | Phase-2.1 ECE (calibrated) | **0.0056** vs baseline 0.0105 — Sec 5.9.7, line 5341 | **0.005586** — `phase21_calibration_comparison.csv:7` | ✓ exact match |
| 5 | Phase-2.1 paired ΔPR-AUC (full vs no_esm2) | **+0.0313** [+0.0266, +0.0361], two-sided **p < 0.0001** — Sec 5.9.2, line 5143 | point **0.03132** [0.02657, 0.03606], **p = 0.0** — `phase21_ablation_paired_bootstrap.csv:3` | ✓ exact match (CSV note: paired bootstrap is on uncalibrated probabilities; the +0.0313 number is computed from full uncal 0.8710 − no_esm2 uncal 0.8396) |
| 5b | Phase-2.1 paired ΔROC-AUC (full vs no_esm2) | **+0.0093** [+0.0082, +0.0103], two-sided **p < 0.0001** — line 5142 | point **0.00929** [0.00816, 0.01033], **p = 0.0** — `phase21_ablation_paired_bootstrap.csv:2` | ✓ exact match |
| 6 | denovo-db holdout post-constraint ROC-AUC | **0.573** [0.471, 0.672], permutation **p = 0.074** — Sec 5.6.9, Table 5.6 (line 4588) and abstract line 215 | **0.5733** [0.4755, 0.6697] — `external_denovo_db_metrics.csv:3` (family_holdout_only). p_perm = **0.0744** — `denovo_significance.json:18` | ✓ match within rounding |
| 7 | denovo-db paired ΔROC-AUC (post − pre, holdout) | point **+0.0764**, 95% CI **[−0.0207, +0.1720]**, one-sided **p = 0.073** — Sec 5.6.9, line 4602-4606, abstract line 217 | point **0.07640** [−0.02072, +0.17205], **p = 0.073** — `denovo_paired_bootstrap.csv:2` | ✓ exact match |
| 7b | denovo-db paired ΔPR-AUC (holdout) | point **+0.0375**, CI **[−0.0050, +0.0796]**, one-sided **p = 0.041** — line 4610 | point **0.03755** [−0.00499, +0.07965], **p = 0.041** — `denovo_paired_bootstrap.csv:3` | ✓ exact match |
| 8 | Rank fusion tuned α* | α* = **0.175** — Sec 5.6.10, Table 5.10 (line 4747); also abstract context | α = **0.175** — `rank_fusion_esm2.csv:5` | ✓ value matches |
| 8a | Rank fusion **convention** | Eq. 5.4 (line 3782-3787): `s = α·r_XGB/n + (1−α)·r_ESM/n`. So α=0.175 ⇒ XGB weight 0.175, ESM weight 0.825. | Code `scripts/run_rank_fusion_esm2.py:98-100`: `s = (1−α)·xgb_rank + α·esm_rank` with comment `Weight α goes on ESM-2; (1 − α) on XGBoost.` So α=0.175 ⇒ ESM weight 0.175, XGB weight 0.825. | ✗ **RED FLAG — thesis equation 5.4 is reversed vs code.** Numerically only the CODE convention is consistent (tuned ROC 0.9412 stays near XGB-alone 0.9380, not near ESM-alone 0.7345). Defense slide 30 currently says "α=0.175 down-weights the weaker PLM signal" which is correct under the code. The user's proposed B1 rewrite ("most rank weight on ESM-2") matches the THESIS equation but contradicts the code. **See Finding #2 above — needs your decision.** |

### Additional metrics found in defense or thesis

| # | Metric | Thesis | CSV | Match? |
|---|--------|--------|-----|--------|
| 9 | ESM-2 LLR zero-shot (rank-fusion test split) ROC / PR | 0.7345 / 0.6037 — Sec 5.6.10, Table 5.10, line 4744 | 0.7345 / 0.6037 — `rank_fusion_esm2.csv:3` | ✓ |
| 10 | ESM-2 LLR esm2_only (Phase-2.1 ablation) ROC / PR | 0.7388 / 0.6099 — Sec 5.9.2, Table 5.11, line 5131 | 0.7388 / 0.6099 — `ablation_esm2_phase21.csv:4` | ✓ |
| 10a | **Defense-specific bug (B2)** | Slide 31 row "ESM-2 LLR (zero-shot)" shows 0.735 / 0.604 (line 1023 in defense.tex). | These are the **rank-fusion-table** numbers (row 9), but they sit inside the **Phase-2.1 ablation** slide which should use 0.7388 / 0.6099 (row 10) per thesis Table 5.11. | ✗ — confirms B2 fix |
| 11 | Headline ECE | 0.054 → 0.011 — Sec 5.6.4, Table 5.5, line 4191-4193 | raw 0.0540, calibrated 0.0105 — `xgboost_calibration_summary.csv:3,5` | ✓ |
| 12 | Five-stage leakage journey (raw PR-AUC) | 0.955 → 0.819 → 0.816 → 0.835 → 0.836 — Sec 5.6.2, Fig 5.4 (line 4061), and abstract line 186 | identical sequence — `leakage_fix_journey.csv:2-6` | ✓ |
| 13 | Test PR-AUC raw — uncalibrated headline | 0.836 [0.827, 0.843] — abstract line 187, Sec 5.10 line 5359 | mean 0.8382 [0.8298, 0.8459] — `xgboost_bootstrap_ci.csv:4` | ✗ thesis-rounded 0.836 vs CSV 0.838 — same drift |
| 14 | Headline F1 (calibrated) | 0.7753 [0.7678, 0.7829] — Sec 5.6.1, line 4039 | mean 0.7755 [0.7678, 0.7829] — `xgboost_bootstrap_ci.csv:4` (test_raw uses 0.5700 threshold, F1 mean 0.7755) — but Table 5.4 row is "calibrated" so check test_calibrated: F1 mean 0.7472 [0.7392, 0.7548] | ⚠ The thesis Table 5.4 says "F1 (at best threshold) 0.7753" and the section text labels it as in the calibrated block. The 0.7753 value matches the **raw** bootstrap F1, not calibrated. Likely a thesis copy-paste — F1 at threshold 0.57 is computed on raw probabilities even when ROC/PR are quoted calibrated. Not blocking for the defense; flag in extended-results addendum. |
| 15 | denovo-db paired Δ — Phase-2.1 vs Phase-1 (holdout, ROC) | −0.132 [−0.257, −0.005], p=0.98 — Sec 5.9.4, Table 5.12, line 5216 | −0.13160 [−0.25672, −0.00457], p=0.977 — `denovo_paired_bootstrap_phase21.csv:4` | ✓ exact match — confirms C2 caveat |
| 16 | Constraint-coverage confound caveat (Phase-2.1 holdout drop) | "the holdout Δ therefore mixes 'ESM-2 added' with 'constraint coverage reduced'... full slice... shows no such drift" — Sec 5.9.4, line 5230-5249 | (caveat is text, no CSV check) | ✓ verbatim source for C2 fix |
| 17 | Headline raw PR-AUC ΔCalibration arithmetic | 0.836 raw → 0.827 calibrated ⇒ Δ = **−0.009** (using 3 d.p.) or 0.8382 → 0.8304 ⇒ Δ = **−0.0078** (CSV 4 d.p.) or 0.8355 → 0.8273 ⇒ Δ = **−0.0082** (thesis 4 d.p.) | — | ✗ confirms B4: defense currently says "ΔPR-AUC = −0.011" (line 568 of defense.tex), which is wrong. The arithmetically-correct delta is **−0.008** to **−0.009** depending on precision. **Recommend B4 use thesis 4 d.p.: 0.8355 → 0.8273, Δ = −0.0082**, written as "−0.008". |

---

## SECTION 2 — Full canonical values (use in Phases B–D)

### 2.1 Headline (Section 5.6.1, Table 5.4 in thesis)

> **Authoritative source: `thesis.tex:4012-4043`** (the deck and `ablate_esm2.py` mirror these values).

| Metric | Raw probabilities | Calibrated (canonical) |
|---|---|---|
| ROC-AUC | 0.9378 [0.9348, 0.9407] | **0.9376 [0.9346, 0.9404]** |
| PR-AUC | 0.8355 [0.8265, 0.8434] | **0.8273 [0.8185, 0.8352]** |
| Brier loss | 0.0876 [0.0853, 0.0899] | 0.0826 [0.0805, 0.0846] |
| ECE | — | 0.0105 |
| F1 (at best threshold) | — | 0.7753 [0.7678, 0.7829] |
| MCC | — | 0.6964 [0.6873, 0.7066] |

**Defense-friendly headline copy (use verbatim):**
> "Calibrated PR-AUC 0.827 [0.819, 0.835]; raw 0.836 [0.827, 0.843]. Calibration trades a small ranking penalty (ΔPR-AUC = −0.008, 0.8355 → 0.8273) for ~5× better probability reliability (ECE 0.054 → 0.011)." **— this is the B4 replacement text.**

### 2.2 Five-stage leakage journey (Section 5.6.2, `leakage_fix_journey.csv`)

| Stage | Description | ROC-AUC | PR-AUC |
|---|---|---|---|
| 1. Original | Pre-audit baseline (random-split + non-missense + is_common + chr OHE) | 0.955 | **0.955** |
| 2. Missense filter | Strict ref_aa AND alt_aa non-null, removed ~88K non-missense rows | 0.934 | **0.819** |
| 3. Feature hygiene | Removed is_common / chr OHE / raw ref+alt | 0.934 | **0.816** |
| 4. Paralog split | Family-level GroupShuffleSplit (HGNC prefix) | 0.938 | **0.835** |
| 5. Optuna re-tune | TPE 40-trial on PR-AUC | 0.938 | **0.836** |

**Defense narrative arc:** 0.955 → 0.836 raw / 0.827 calibrated. Biggest single drop = stage 2 (missense filter, −0.136 PR-AUC).

### 2.3 denovo-db external validation

> **Authoritative source: `thesis.tex:4536-4647` (Section 5.6.9), `denovo_paired_bootstrap.csv`, `denovo_significance.json`, `external_denovo_db_metrics.csv`.**

#### 2.3.1 Headline table (un-paired, Table 5.6)

| Slice | n | n+ | ROC-AUC [95% CI] | PR-AUC [95% CI] | p_perm |
|---|---|---|---|---|---|
| full (pre-constraint) | 642 | 498 | 0.468 [0.415, 0.519] | 0.761 [0.721, 0.806] | — |
| **full (post-constraint)** | 642 | 498 | **0.511** [0.458, 0.564] | **0.790** [0.749, 0.830] | 0.335 |
| holdout (pre-constraint) | 201 | 161 | 0.487 [0.383, 0.583] | 0.789 [0.712, 0.860] | — |
| **holdout (post-constraint)** | 201 | 161 | **0.573** [0.471, 0.672] | **0.838** [0.774, 0.897] | **0.074** |

#### 2.3.2 Paired bootstrap on Δ (holdout slice — the defense headline)

`denovo_paired_bootstrap.csv` rows 2-3:

| Metric | Δ (post − pre) | 95% CI | one-sided p | n |
|---|---|---|---|---|
| ROC-AUC | **+0.0764** | [−0.0207, +0.1720] | **0.073** | 201 |
| PR-AUC | +0.0375 | [−0.0050, +0.0796] | 0.041 | 201 |

#### 2.3.3 Two SEPARATE tests — do not conflate (B3 fix)

Thesis explicitly distinguishes:

1. **Label-permutation** test of `H₀: AUC_post ≤ 0.5` on the post-constraint score vector — `denovo_significance.json:18` → **p_perm = 0.0744**, 10,000 permutations, seed=42. Tests "is the post-constraint score better than chance?"
2. **Paired bootstrap** on `Δ = AUC_post − AUC_pre` — `denovo_paired_bootstrap.csv:2` → **p_paired = 0.073**, 1,000 replicates, seed=42. Tests "did adding constraint features help, on the same 201 variants?"

Both happen to round to ~0.07 by coincidence — they test different nulls. The Δ claim ("constraint added value") needs the **paired bootstrap p = 0.073**, not the permutation p = 0.074.

#### 2.3.4 Near-base-rate caveat (verbatim from thesis line 4632-4647)

> "The denovo-db positive rate is 498/642 = 77.5% on the full slice and 161/201 = 80.1% on the family-holdout slice. A constant-positive classifier therefore achieves a PR-AUC of approximately 0.776 on the full slice and 0.801 on the family-holdout slice. The reported classifier PR-AUC values (0.761 and 0.790 pre-constraint; 0.790 and 0.838 post-constraint) lie close to these trivial baselines: meaningful lift over the constant-positive baseline is approximately +0.014 (full, pre), +0.014 (full, post), −0.012 (holdout, pre), and **+0.037 (holdout, post)**. ROC-AUC is the informative metric on this cohort because it is invariant to class balance."

**Defense bullet (C1):** "Holdout class balance is 80% positive — constant-positive baseline PR-AUC ≈ 0.80. Meaningful PR-AUC lift is only +0.037, so ROC-AUC is the primary metric for this cohort."

### 2.4 Phase-2.1 ablation (Section 5.9.2, Table 5.11)

> **Authoritative source: `thesis.tex:5102-5152` (Section 5.9.2). Table mixes calibrated `full` + uncalibrated comparisons; paired bootstrap is on uncalibrated. See note below.**

#### 2.4.1 Internal-test ablation

| Condition | n features | Test ROC-AUC [95% CI] | Test PR-AUC [95% CI] | Calibration |
|---|---|---|---|---|
| XGBoost baseline (frozen, Phase-1) | 78 | 0.9376 [0.9346, 0.9404] | **0.8273** [0.8185, 0.8352] | calibrated |
| Augmented `no_esm2` ablation | 78 | 0.9386 [0.9355, 0.9416] | 0.8396 [0.8311, 0.8474] | uncalibrated |
| Augmented `esm2_only` ablation | 2 | **0.7388** [0.7314, 0.7458] | **0.6099** [0.5987, 0.6205] | uncalibrated |
| **Augmented `full`** | **80** | **0.9476** [0.9448, 0.9503] | **0.8647** [0.8580, 0.8714] | calibrated |

**The two ESM-only numbers in the ESM ablation row are the B2 replacement values: 0.7388 / 0.6099 (round to 0.739 / 0.610 to keep the slide consistent with thesis Table 5.11).**

#### 2.4.2 Paired bootstrap on Δ (full vs no_esm2, on uncalibrated probabilities, n=28,098, 1,000 reps)

| Metric | Δ point | 95% CI | two-sided p |
|---|---|---|---|
| ROC-AUC | +0.0093 | [+0.0082, +0.0103] | < 0.0001 |
| **PR-AUC** | **+0.0313** | **[+0.0266, +0.0361]** | **< 0.0001** |

Source: `phase21_ablation_paired_bootstrap.csv`. Defense already shows ΔPR-AUC = +0.031 on slide 28 (line 1240) — this is correct.

### 2.5 Phase-2.1 SHAP shift + constraint-coverage confound (Section 5.9.3-4)

**SHAP top features in augmented model** (`shap_ranking_phase21.csv`):
1. `num__is_imputed_esm2_llr` (mean |SHAP| = 1.047)
2. (conservation features remain in top-3)
3. (conservation features remain in top-3)
4. `num__esm2_llr` (mean |SHAP| = 0.419)
5. `lof_z` (was rank 3 in Phase-1)

**Constraint-coverage confound (verbatim, thesis line 5230-5249):**

> "A methodological caveat applies to the holdout comparison... the augmented-model scorer reconstructs gene-level constraint values from the training split (`train.parquet`) where possible, falling back to train-fit medians otherwise. By construction the family-holdout genes are absent from training, so they receive imputed constraint values for the augmented model even where the baseline had real values from the gnomAD table directly. The holdout Δ therefore mixes 'ESM-2 added' with 'constraint coverage reduced.' ... The full slice, where most genes appear in training and constraint values are recoverable, shows no such drift, supporting the interpretation that the holdout deficit is at least partly a coverage artefact rather than a pure ESM-2 effect."

**Defense bullet (C2):** "Partly a coverage artifact: held-out family genes receive imputed constraint values for the augmented model where the baseline had real gnomAD values. The full slice (preserved coverage) shows no degradation, supporting the artifact interpretation."

### 2.6 Phase-2.1 denovo-db (Section 5.9.4, Table 5.12)

| Slice / metric | n | Baseline | Augmented | Paired Δ [95% CI], p (one-sided) |
|---|---|---|---|---|
| full / ROC-AUC | 642 | 0.5111 | 0.5095 | −0.002 [−0.059, +0.064], p=0.51 |
| full / PR-AUC | 642 | 0.7896 | 0.7833 | −0.006 [−0.037, +0.028], p=0.62 |
| **holdout / ROC-AUC** | **201** | **0.5733** | **0.4417** | **−0.132 [−0.257, −0.005], p=0.98** |
| holdout / PR-AUC | 201 | 0.8376 | 0.7850 | −0.053 [−0.103, +0.001], p=0.97 |

Source: `denovo_paired_bootstrap_phase21.csv`. The −0.13 ROC-AUC drop on family-holdout is the C2 talking point.

### 2.7 Rank fusion (Section 5.6.10, Table 5.10, Eq. 5.4)

#### 2.7.1 Numbers (`rank_fusion_esm2.csv`)

| Model | ROC-AUC | PR-AUC | α |
|---|---|---|---|
| XGBoost (calibrated) | 0.9380 | 0.8304 | — |
| ESM-2 LLR (zero-shot) | 0.7345 | 0.6037 | — |
| Rank fusion, uniform | 0.8949 | 0.7839 | 0.50 |
| **Rank fusion, tuned** | **0.9412** | **0.8516** | **0.175** |

#### 2.7.2 Convention — the discrepancy you must resolve

**Thesis Eq. 5.4 (line 3782):** `s = α·r_XGB/n + (1−α)·r_ESM/n`. Reading literally, α=0.175 ⇒ XGB has small weight 0.175.

**Code (`scripts/run_rank_fusion_esm2.py:98-100`):**
```python
def _fuse(xgb_rank, esm_rank, alpha):
    """Weight α goes on ESM-2; (1 − α) on XGBoost."""
    return (1.0 - alpha) * xgb_rank + alpha * esm_rank
```
α=0.175 ⇒ ESM has small weight 0.175, XGB has 0.825.

**Sanity check via the numbers themselves:**
- Pure XGB (α→0 in code, α→1 in thesis convention): ROC 0.9380
- Pure ESM (α→1 in code, α→0 in thesis convention): ROC 0.7345
- α=0.5: 0.8949 (between)
- Tuned α=0.175: 0.9412

The tuned point sits *very close to pure XGB* (0.9412 vs 0.9380), not close to pure ESM (0.7345). This is only possible if the tuned fusion is XGB-dominant — i.e., the **CODE** convention is the operative one. The thesis equation as printed is reversed.

**B1 decision options:**
- **(A)** Keep current defense narrative ("α=0.175 down-weights weaker PLM") — correct vs code, contradicts thesis-as-printed. Add a one-liner footnote: "α weights ESM-2; the thesis Eq. 5.4 transcription is reversed and will be corrected in the published revision."
- **(B)** Use user's proposed rewrite ("most rank weight on ESM-2") — correct vs thesis equation, contradicts code AND the visible numeric pattern. **Not recommended** — it sets up a follow-up question we can't defend.
- **(C)** Print the equation as it appears in the code (`s = (1−α)·r_XGB + α·r_ESM`) on the slide, then say "α=0.175 small weight on ESM, large on XGB." Cleanest for the defense, requires diverging from the thesis equation.

**My recommendation: option (C).** The deck should match the code. Mention the thesis-erratum in passing if asked.

### 2.8 Baselines (Section 5.6.3, Table 5.3 — but B5 doesn't touch this)

> **Caveat: each baseline scores its own coverage subset; rows are NOT directly comparable. Thesis explicitly lists this as Deficiency D7.** `baselines_comparison.csv`.

| Baseline | Year | ROC-AUC (own cov, 95% CI) | PR-AUC (own cov, 95% CI) | Coverage |
|---|---|---|---|---|
| SIFT | 2003 | 0.881 [0.877, 0.885] | 0.620 [0.610, 0.629] | 96% |
| PolyPhen-2† | 2010 | 0.893 [0.888, 0.898] | 0.728 [0.716, 0.739] | 93% |
| **XGBoost (ours, calibrated)** | 2026 | **0.938 [0.935, 0.941]** | **0.827 [0.819, 0.835]** | **100%** |
| AlphaMissense* | 2023 | 0.956 [0.953, 0.958] | 0.890 [0.882, 0.898] | 86% |

†PolyPhen-2 trained on HumDiv overlapping ClinVar. *AlphaMissense calibrated against ClinVar at release. Both inflate own-coverage rows.

### 2.9 Failures F1–F5 (Section 7.3, thesis line 5963-6013) — verbatim verbiage

Use these as the C3 source. All five paragraphs already exist in the thesis; the defense currently shows only F1–F3.

- **F1 — Initial random split inflated AUROC by +0.081.** First XGBoost on random 70/15/15 reported test ROC-AUC = 0.996. Audit revealed paralog families split across train/test, leaking label-identifying signal (Heijl et al. 2020). *Fix:* `ShuffleSplit` → `GroupShuffleSplit` on paralog-family groups.
- **F2 — Meta-predictor features created circular evaluation.** REVEL, ClinPred, CADD all themselves trained on ClinVar; inflated PR-AUC by ~0.07. *Fix:* banned-feature list in `verify_no_leakage.py`.
- **F3 — gnomAD constraint imputation fit on the full data.** Imputation medians on train+val+test = subtle leakage. *Fix:* fit medians on train only, apply to val/test.
- **F4 — ESM-2 Colab session crashed at 85% of training-set scoring.** Free-tier disconnected after ~7 hr, 166K/195K variants scored. *Fix:* chunked writer flushes checkpoint every 500 variants, scorer resumes.
- **F5 — Original CNN-BiLSTM architecture infeasible on free compute.** Term-1 plan called for CNN-BiLSTM over ±30-residue windows; estimated 4× over Colab quota. *Fix:* scope reduced to tabular XGBoost + zero-shot ESM-2; CNN-BiLSTM deferred to future work.

**Defense one-liners (compress for slide):**
- F1: random split → ROC inflated by +0.081 (paralog leakage). **Fix:** `GroupShuffleSplit`.
- F2: REVEL/ClinPred/CADD → +0.07 PR-AUC inflation (meta-predictor circularity). **Fix:** ban-list in CI.
- F3: gnomAD imputation on full data → subtle leakage. **Fix:** train-only medians.
- F4: Colab crash at 85% (~7 hr) → 30K variants lost. **Fix:** 500-variant checkpoint resume.
- F5: CNN-BiLSTM 4× over budget. **Fix:** scope cut to XGBoost + zero-shot ESM-2.

### 2.10 Deficiencies D1–D8 (Section 7.4, Table 7.2, thesis line 6015-6094)

> **⚠ User's spec said "D1–D6". Thesis canonical source has D1–D8. Recommend extending C3 to all eight.**

- **D1** — External generalisation on unseen families is at ROC-AUC 0.573 [0.471, 0.672], p_perm=0.074; CI still covers chance at n=201. *Next:* persist pre-constraint predictions, validate on ProteinGym, train union of ClinVar + denovo-db.
- **D2** — Training data is ClinVar-only; no DMS / saturation-mutagenesis labels. *Next:* incorporate ProteinGym (Notin et al. 2023) as a 4th validation set.
- **D3** — Calibration is isotonic-only; no per-gene recalibration. *Next:* second calibration stage conditioning on LOEUF decile.
- **D4** — ESM-2 LLR was zero-shot on denovo-db only at submission; full-dataset feature integration was pending. *(Now superseded — Phase 2.1 is shipped, Section 5.9.)* *Next:* already done.
- **D5** — Structural features (AlphaFold-2 pLDDT, SASA, DSSP) not yet used. *Next:* Phase 2.2 AlphaFold pipeline + 4 structural features.
- **D6** — Only user-facing surface is research-grade Streamlit; no REST API, no VEP plug-in. *Next:* FastAPI micro-service + VEP plug-in.
- **D7** — Baseline-comparison rows are on each model's own coverage (AM 86%, PP2 93%, SIFT 96%, XGBoost 100%); the −0.05 PR-AUC nominal gap to AM is not strict like-for-like. *Next:* persist XGBoost test predictions, rerun matched-coverage on `S_AM`, paired DeLong on AM vs ours.
- **D8** — denovo-db pre-constraint per-variant predictions were not persisted, so the paired bootstrap on Δ_post−pre couldn't be computed strictly at submission. *(Now superseded — paired bootstrap computed via recovered checkpoint at commit f8ab464; reported in Sec 5.6.9.)* *Next:* persist on every future training run.
- **D9** *(post-submission, added 2026-04-27 during defense audit)* — Bootstrap CI CSV (`xgboost_bootstrap_ci.csv`) drifted +0.003 PR-AUC vs the frozen thesis values (thesis test_calibrated PR-AUC 0.8273 vs CSV 0.8304; same direction on raw and ROC). Both claim 1,000 reps, seed=42; the thesis number is hard-coded in `scripts/ablate_esm2.py:196` so the thesis ecosystem is internally consistent. Thesis remains canonical for the defense. *Cause TBD; suspected upstream data refresh that did not propagate to the thesis text.* *Next:* re-run bootstrap from the same checkpoint as the thesis to confirm the gap is reproducible, then either correct the thesis to the new CSV or pin the CSV to the thesis-canonical run.

**Defense one-liners (compress for slide):**
- D1: external ROC=0.573, CI covers chance at n=201 → larger external cohort needed.
- D2: ClinVar-only, no DMS → add ProteinGym.
- D3: isotonic only, no per-gene calibration → LOEUF-decile second stage.
- D4: ~~ESM-2 zero-shot only~~ shipped (Phase 2.1).
- D5: no AlphaFold-2 structural features yet → Phase 2.2.
- D6: Streamlit only, no API → FastAPI + VEP plug-in.
- D7: own-coverage baselines, no matched comparison → matched-coverage rerun + paired DeLong.
- D8: ~~pre-constraint predictions not persisted~~ recovered via checkpoint f8ab464; persist on future runs.

### 2.11 Project + slide-count book-keeping for Phase D

- Defense source: `report/academic/defense.tex` (1467 lines, 62.7KB) — canonical.
- The other defense at `report/defense.tex` (440 lines) is a stub — **do not edit** unless the user confirms it's the canonical one. Phase B/C/D/E should target `report/academic/defense.tex` only.
- Frame count via `grep -c "begin{frame}"`: **44 total**.
- Backup section starts at line 1310 with **5 backup frames** (UML 4/6, UML 5/6, UML 6/6, hardware+software, CI tests).
- Effective main slides: **39** (44 − 5). The user's spec says "38 main + 6 backup" — close enough, off by one in either direction. Phase D will cut to ~22-24 main + ~8 backup as specified.

---

## Provenance index

| Topic | Authoritative thesis location | Authoritative CSV/JSON |
|---|---|---|
| Headline (raw + calibrated) | `thesis.tex:4012-4043` (Sec 5.6.1) | `xgboost_bootstrap_ci.csv` (drift flagged) |
| Leakage journey | `thesis.tex:4045-4083` (Sec 5.6.2) | `leakage_fix_journey.csv` |
| Baselines | `thesis.tex:4085-4148` (Sec 5.6.3) | `baselines_comparison.csv` |
| Calibration deep-dive | `thesis.tex:4168-4196` (Sec 5.6.4) | `xgboost_calibration_summary.csv`, `brier_decomposition.csv` |
| denovo-db un-paired | `thesis.tex:4536-4591` (Sec 5.6.9) | `external_denovo_db_metrics.csv`, `denovo_significance.json` |
| denovo-db paired bootstrap | `thesis.tex:4593-4631` (Sec 5.6.9) | `denovo_paired_bootstrap.csv` |
| denovo-db near-base-rate | `thesis.tex:4632-4647` (Sec 5.6.9) | (text only) |
| External calibration | `thesis.tex:4649-4700` (Sec 5.6.9) | `external_calibration.csv` |
| Rank fusion (Eq. 5.4 + tuned) | `thesis.tex:3770-3835` (Sec 5.4) and `4702-4825` (Sec 5.6.10) | `rank_fusion_esm2.csv`, `pairwise_pvalues.csv` |
| Phase-2.1 ablation | `thesis.tex:5102-5152` (Sec 5.9.2) | `ablation_esm2_phase21.csv`, `phase21_ablation_paired_bootstrap.csv` |
| Phase-2.1 SHAP | `thesis.tex:5154-5189` (Sec 5.9.3) | `shap_ranking_phase21.csv` |
| Phase-2.1 denovo-db + confound | `thesis.tex:5191-5249` (Sec 5.9.4) | `denovo_paired_bootstrap_phase21.csv`, `external_denovo_db_metrics_phase21.csv` |
| Phase-2.1 calibration | `thesis.tex:5337-5347` (Sec 5.9.7) | `phase21_calibration_comparison.csv` |
| Failure F1–F5 | `thesis.tex:5963-6013` (Sec 7.3) | (text only) |
| Deficiencies D1–D8 | `thesis.tex:6015-6094` (Sec 7.4, Table 7.2) | (text only) |

---

## What I need from you before Phase B

Please confirm or redirect on:

1. **CSV ↔ thesis −0.003 PR-AUC drift (rows 1, 2, 3, 13).** Trust thesis (0.827) for the defense and flag separately, OR re-derive from CSV (0.830) and update thesis too?
2. **Rank-fusion Eq. 5.4 ↔ code mismatch.** Pick option A / B / C from §2.7.2 above.
3. **D1–D6 vs D1–D8.** Extend C3 to D1–D8 to match thesis verbatim?
4. **Defense source path.** Confirm `report/academic/defense.tex` is the canonical target (the other one at `report/defense.tex` is a 440-line stub).
5. **Defense F1 row in the headline (row 14 in §1).** Thesis F1 0.7753 sits in the calibrated block but matches raw bootstrap (calibrated F1 = 0.7472). Likely a thesis copy-paste. Block on it for Phase E only, not for the defense rewrite — confirm?

Once you OK these, I'll proceed with Phase B (the 5 P0 fixes), one commit per fix, and ping after each PDF rebuild.
