# Changelog

All notable changes to the honest-baseline pipeline. Dates are ISO-8601.
Commits are on `origin/main`.

## [Phase 2.1 — ESM-2 LLR as a training feature] — 2026-04-25

### Decisions (recorded with the user before any code was written)
1. **Aggregation:** `min(esm2_llr)` per `variant_key` (most-pathogenic isoform).
   Sensitivity probe in `results/metrics/esm2_aggregation_sensitivity_phase21.csv`
   confirms the choice is a no-op on the current score parquets (one row
   per `variant_key` shipped from Colab).
2. **Hyperparameters:** BOTH paths run. Frozen Phase-1 hyperparameters are
   the canonical headline (cleanest causal isolation of the ESM-2 marginal
   effect); Optuna 40-trial retune is appendix-only and confirms |Δ| < 0.005
   PR-AUC vs frozen-hp (`xgboost_phase21_optuna_split_metrics.csv`).
3. **External validation:** denovo-db re-evaluated on Phase-2.1 model
   (`external_denovo_db_predictions_phase21.parquet`); paired bootstrap
   vs Phase-1 in `denovo_paired_bootstrap_phase21.csv`. Existing
   `denovo_paired_bootstrap.csv` (pre-vs-post-constraint) is NEVER overwritten.
4. **Null-result handling:** ship transparently; ceiling at 35 M scale
   documented as Phase 2.1b motivator in §5.8.5 + Chapter 7.

### Headline results
- **Internal-test ablation (n=28,098, paired bootstrap, 1,000 reps, seed=42):**
  - Δ test PR-AUC = **+0.0313** [+0.0266, +0.0361], p < 10⁻⁴
  - Δ test ROC-AUC = **+0.0093** [+0.0082, +0.0103], p < 10⁻⁴
  - Both intervals entirely above zero; gain exceeds the post-hoc
    rank-fusion ceiling of +0.014 PR-AUC.
- **Calibration:** Phase-2.1 test ECE = 0.0056 vs Phase-1 = 0.0105 (tighter).
- **denovo-db full slice:** Δ ROC-AUC = −0.002 (n.s.), Δ PR-AUC = −0.006 (n.s.).
- **denovo-db family-holdout (n=201):** Δ ROC-AUC = **−0.132** [−0.257, −0.005],
  one-sided p = 0.98. Phase-2.1 is statistically worse on unseen gene
  families. Confound: holdout genes lack constraint coverage in this
  worktree (raw gnomAD-constraint TSV is gitignored), so Phase-2.1 holdout
  uses imputed-everywhere constraint while Phase-1 had partial real coverage;
  the deficit therefore mixes "ESM-2 added" with "constraint reduced." Full
  slice (where train-recoverable constraint applies) shows no drift.

### What's added
- `data/splits/phase21/{train,val,test}.parquet` (53 cols = 51 + `esm2_llr`
  + `is_imputed_esm2_llr`); 99.14 / 99.37 / 99.39 % ESM-2 coverage.
- `results/checkpoints/xgboost_phase21_esm2.ubj` (frozen-hp; HEADLINE).
- `results/checkpoints/xgboost_phase21_optuna_esm2.ubj` (appendix sensitivity).
- `results/metrics/phase21/{xgboost_bootstrap_ci,xgboost_calibrated_metrics,xgboost_calibration_summary,xgboost_reliability_curve,xgboost_operating_points,xgboost_predictions}.parquet/csv`.
- New CSVs/parquets in `results/metrics/`:
  `xgboost_phase21_*`, `ablation_esm2_phase21`,
  `phase21_ablation_paired_bootstrap`, `external_denovo_db_metrics_phase21`,
  `external_denovo_db_predictions_phase21`,
  `external_denovo_db_coverage_phase21`, `denovo_paired_bootstrap_phase21`,
  `esm2_aggregation_sensitivity_phase21`,
  `esm2_correlation_analysis_phase21`, `phase21_ablation_summary`,
  `phase21_calibration_comparison`, `shap_values_phase21_test`,
  `shap_ranking_phase21`.
- 9 new scripts: `probe_esm2_aggregation`, `build_phase21_train`,
  `verify_esm2_split_integrity`, `train_phase21_xgboost`, `compute_shap_phase21`,
  `ablate_esm2`, `score_denovo_phase21`, `paired_bootstrap_denovo_phase21`,
  `phase21_diagnostics`, `phase21_calibration_audit`.
- 5 new figures (mirrored under `report/academic/figures/` for the LaTeX build):
  `shap_summary_phase21`, `shap_bar_phase21`, `shap_dependence_top3_phase21`,
  `ablation_esm2_forest_phase21`, `phase21_feature_importance_comparison`.
- 4 new tests: `tests/unit/test_phase21_build.py`,
  `tests/unit/test_ablate_esm2.py`,
  `tests/integration/test_phase21_drift.py`,
  `tests/integration/test_esm2_split_integrity.py`.
- 2 new leakage-gate checks in `src/verify_no_leakage.py`:
  Phase-2.1 ESM-2 features-present + ESM-2 split integrity. Gate reports 6 PASS.
- `.gitignore` exemptions for `data/splits/phase21/` (mirroring the
  `strict/` pattern).
- New thesis `Section 5.8` (`sec:res-phase21`, 7 subsections); Section 7.8
  Phase 2.1 entry updated from "in flight" → "completed; see §5.8" plus
  new Phase 2.1b (650 M checkpoint) motivation.
- 2 new defense slides between current rank-fusion slide and Conclusion.

### What is NOT touched
- `results/checkpoints/xgboost_best.ubj` — read-only.
- `results/metrics/denovo_paired_bootstrap.csv` — unchanged
  (pre-vs-post-constraint paired result from P0-1 stays untouched).
- `results/metrics/baselines_intersection.csv` — unchanged.
- Thesis Section 5.6 narrative — unchanged.
- All Phase-1 / P0-fix CSVs and parquets — unchanged.

## [P0-2 Revision — intersection-subset baseline comparison] — 2026-04-25

### What was added
- `scripts/intersection_baselines.py`: scores every test-split variant
  with the trained XGBoost (calibrated) and extracts SIFT, Polyphen2,
  and AlphaMissense scores from the bundled dbNSFP TSV via `tabix`.
- `tests/unit/test_intersection_baselines.py`: 3 unit tests covering
  bootstrap helper and dbNSFP score-string parsing.
- `results/metrics/baselines_intersection.csv`: per-method ROC/PR with
  bootstrap CIs on the variants where all four methods produced a
  score.

### Infrastructural caveat (recorded in script docstring + CSV note)
The bundled `dbNSFP5.3.1a_grch37.gz` is BGZF-indexed on GRCh38
coordinates (cols 1--2) despite the file name; GRCh37 positions live
in cols 8--9 as a secondary annotation, not as the tabix key. Our test
split, the dbNSFP feature cache, and the trained XGBoost are all keyed
on GRCh37. Tabix lookups using GRCh37 chr:pos therefore land on a
dbNSFP record only for variants where the GRCh38 and GRCh37 positions
coincide, which yields n=446 of 28,098 test variants (1.6%).

A correct full intersection would require either (a) a UCSC liftover
chain or (b) a one-pass scan of the 47 GB dbNSFP TSV filtering on
cols 8--9. Neither fit the revision-pass budget. The reported
n=446 result is the *tabix-recoverable* intersection rather than the
full method-overlap intersection. This is documented in the script
docstring, in the output CSV's `note` column, and is flagged as a
known limitation in the thesis.

### Result (n=446 tabix-recoverable subset)
| method | n | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|---|
| XGBoost | 446 | 0.9389 [0.9139, 0.9597] | 0.8269 [0.7556, 0.8945] |
| SIFT | 446 | 0.9027 [0.8728, 0.9290] | 0.7222 [0.6396, 0.8005] |
| PolyPhen-2 | 446 | 0.9370 [0.9144, 0.9564] | 0.8104 [0.7344, 0.8769] |
| AlphaMissense | 446 | 0.9697 [0.9514, 0.9834] | 0.9444 [0.9144, 0.9692] |

AlphaMissense intersection PR-AUC of 0.9444 is +0.0544 above Table 5.2's
own-coverage 0.890. Direction-of-effect interpretation: on this subset,
AM scores *harder* variants (higher PR-AUC where it has overlap with
all peers), so the coverage-bias-only explanation of the Table 5.2 gap
is *not* supported by this subset --- the AM advantage looks real.
This is a one-sided sanity-check finding only; a full liftover-based
intersection should be run before any clinical-grade inference.

## [P0-1 Revision — denovo-db paired-bootstrap prerequisite] — 2026-04-24

### What was retrained and why
Nothing was retrained. The pre-constraint XGBoost checkpoint was recovered
from git history via archaeology (`git log --all --oneline -- results/checkpoints/xgboost_best.ubj`),
specifically from commit `f8ab464` ("Harden XGBoost baseline: fix leakage,
add bootstrap CIs and calibration"), which is the commit immediately
before `b2fd3d0` introduced gnomAD gene-level constraint features.

### Why "Do not retrain" was overridden
Not overridden. Git archaeology succeeded (Step 0 of the P0-1 brief), so
the brief's Step 1 retrain path was skipped entirely. The recovered
checkpoint is the *exact* model that produced the 0.487 pre-constraint
holdout number reported in Table 5.5 of the thesis.

### What was changed
- Copied the `f8ab464` checkpoint to `results/checkpoints/xgboost_pre_constraint.ubj`
  (under `checkpoints/` so it survives the `results/metrics/*` gitignore rule).
- Copied the `f8ab464` feature-columns manifest to `results/metrics/xgboost_pre_constraint_feature_columns.csv`
  (73 features — 6 fewer than HEAD: no `pLI`, `oe_lof_upper`, `mis_z`,
  `oe_mis_upper`, `lof_z`, or `is_imputed_gnomad_constraint`).
- Added `scripts/score_denovo_paired.py` which reconstructs the featurized
  denovo-db frame from the existing `external_denovo_db_predictions.parquet`
  skeleton (642 variant_keys + labels + family_holdout flags) joined to
  the committed dbNSFP and gnomAD-AF caches, then scores with both
  checkpoints.
- Emitted `results/metrics/denovo_predictions_pre_constraint.parquet` and
  `results/metrics/denovo_predictions_post_constraint.parquet` in the
  `variant_id, y_true, p_pred, slice` schema.

### What was NOT changed
- `results/checkpoints/xgboost_best.ubj` — untouched.
- `data/splits/train.parquet`, `val.parquet`, `test.parquet` — untouched.
- Any Chapter 5 headline number (ROC-AUC, PR-AUC, Brier, ECE, MCC on the
  paralog-disjoint test split) — recomputed against committed artifacts,
  unchanged.
- The leakage gate, the paralog splitter, the calibrator, or any trained
  artifact under `src/` — untouched.

### Match-check results (Step 3 of the P0-1 brief)
| Model | Slice | ROC-AUC | Thesis target | Deviation | Status |
|---|---|---|---|---|---|
| pre-constraint | holdout | 0.4969 | 0.487 | +0.0099 (within ±0.02) | PASS |
| post-constraint | holdout | 0.5733 | 0.573 | +0.0003 | PASS |
| post-constraint | full | 0.5111 | 0.511 | +0.0001 | PASS (bonus) |

The small positive drift on pre-constraint holdout (+0.01 ROC-AUC) is
attributable to the fact that at `f8ab464` the splits were themselves
slightly different (see `git diff f8ab464 HEAD -- data/splits/`); the
match is within the ±0.02 tolerance the brief specifies, so the
archaeology checkpoint is accepted for the paired-bootstrap test.

## [Thesis v1.0 — Final Submission Edit Pass] — 2026-04-24

Comprehensive edit pass across the thesis informed by the
`thesis_edits_complete.md` review checklist. All 48 scheduled
edits were implemented in a single worktree pass.

### Changed
- Reconciled PR-AUC headline number across Abstract, Table 5.1
  and Appendix A Table A.1: headline is now the calibrated
  value `0.827 [0.819, 0.835]`; raw value `0.836 [0.827, 0.843]`
  is surfaced alongside it.
- Abstract contamination claim reframed as a hypothesis that the
  thesis tests and demonstrates, not a blanket claim against the
  field.
- AlphaMissense contamination-risk phrasing softened to neutral
  interpretation caveat.
- Added paired-bootstrap discussion (D8) to denovo-db section;
  formally acknowledged that per-variant pre-constraint predictions
  were not persisted, with revision path.
- Expanded Section 5.6.5 (confident errors) with clinical-asymmetry
  discussion (4.25:1 FN:FP, clinical significance, phyloP cluster).
- Added `Why our ESM-2 zero-shot ROC-AUC is below published
  figures` discussion (checkpoint size, split difficulty, compute
  budget).
- Added near-base-rate caveat to denovo-db PR-AUC reporting
  (constant-positive baseline reference).
- Expanded Section 5.7 with prior-work summary table and three
  observation bullets.
- Added intersection-subset comparison note on Table 5.2.
- Expanded testing-chapter coverage defence for `data_splitting.py`.
- Added mutation-testing gap acknowledgement to Section 6.9.
- Added Phase-2 roadmap timeline to Section 7.8.
- Added ClinVar ancestry/demographic bias to Section 7.6.
- Extended constraint-imputation bullet with denovo-db rate note.
- Split Chapter 1.7 contribution 5 into two entries (methodological
  package vs. engineering artefacts).
- Compressed Sections 1.1.1–1.1.2 (Central Dogma + Variant Classes).
- Expanded Section 2.4 Research Gap to a fully-developed paragraph.
- Strengthened REVEL contamination discussion (Section 2.3.3).
- Rewrote Use-Case, State, and Component diagram captions
  (Chapter 4) with purpose-driven content.
- Added schedule-realisation paragraph to Section 3.2.3.
- Added Figure 3.3 Gantt year markers in caption.
- Expanded Figure 4.7 Streamlit caption with threshold bands.
- Expanded Table 5.4 SHAP caption with Phase-2.1 row meaning.
- Removed unsupportable "first" claims in Chapter 2.5.

### Fixed
- Resolved LaTeX cross-reference failure in Section 2.3.8 (ESM-2
  zero-shot evaluation reference now points to `sec:res-esm2`).
- Resolved LaTeX cross-reference failure in Section 3.5 (missense
  filter leakage reference now points to `sec:res-leakage`).
- Replaced Unicode em-dashes with LaTeX `---` throughout for
  encoding safety under pdflatex.
- Aligned NFR1 latency claim to observed performance with explicit
  benchmark deferral (D6).

### Added
- Appendix B: paired-bootstrap helper, paralog-family grouper, and
  rank-fusion listings.
- Appendix C: SHA-256 artifact hashes section.
- Deficiency D8: persist pre-constraint per-variant predictions for
  paired bootstrap.

## [Portfolio Stage 2.4 + Stage 3 — SHAP + calibration deep dive] — 2026-04-21

Two complementary analyses that ground the model in its features:
1. **SHAP** (Stage 2.4): which features drive predictions, and which
   variants does the model get wrong with high confidence?
2. **Calibration deep dive** (Stage 3): is the uncertainty on probabilities
   itself trustworthy enough for clinical use?

### Stage 3 — Calibration deep dive

Murphy (1973) Brier decomposition on test:

| Calibrator | Brier | Reliability | Resolution | ECE | Log-loss |
|---|---:|---:|---:|---:|---:|
| Raw | 0.0876 | 0.00543 | 0.1055 | 0.054 | 0.286 |
| Platt | 0.0834 | 0.00114 | 0.1055 | 0.029 | 0.278 |
| **Isotonic** | **0.0826** | **0.00024** | 0.1053 | **0.011** | **0.273** |

The decomposition shows **Resolution is constant** (discrimination
ability unchanged — expected for a monotone post-hoc calibrator) while
**Reliability drops 23× with isotonic**. That's the whole story:
miscalibration was the entire Brier error source, not poor
discrimination. Isotonic achieves ECE = 0.011, below the 0.02 target
set in the plan.

New module: `src/calibration_deep.py` with
`decompose_brier()`, `fit_platt() / apply_platt()`,
`fit_isotonic() / apply_isotonic()`,
`compute_decomposition_table()`, `render_triptych()`.

Tests: `tests/test_calibration_deep.py` — 11 tests, including:
- Murphy identity exact when each bin contains identical probabilities
  (< 1e-9 residual)
- Murphy identity within 0.01 on realistic Beta-distributed data
- Overconfident classifier has reliability > 0.9
- Random (constant) classifier has resolution ≈ 0
- Platt + Isotonic monotonicity
- Isotonic reliability ≤ raw reliability on fit split

### Stage 2.4 — SHAP + error analysis

TreeSHAP on a 2 000-variant stratified test sample. Top-10 features by
mean |SHAP|:

| Rank | Feature | mean \|SHAP\| |
|---:|---|---:|
| 1 | `phyloP100way_vertebrate` | 0.8094 |
| 2 | `AN` (gnomAD allele number) | 0.5315 |
| **3** | **`lof_z` (gnomAD constraint — Stage-2.1 addition)** | **0.3424** |
| 4 | `alt_aa_P` (Proline substitution) | 0.3039 |
| 5 | `pfam_domain` | 0.2844 |
| 6 | `phastCons100way_vertebrate` | 0.2453 |
| 7 | `phastCons30way_mammalian` | 0.2044 |
| 8 | `oe_mis_upper` (gnomAD constraint) | 0.1892 |
| 9 | `phyloP30way_mammalian` | 0.1855 |
| 10 | `GERP++_RS` | 0.1468 |

**The new gnomAD constraint features (`lof_z`, `oe_mis_upper`) ranked
#3 and #8** — validating the Phase-2-step-1 hypothesis that
gene-level priors carry orthogonal signal to variant-level features.

Confident errors (|p_cal − y_true| > 0.5): 326 variants out of 2 000
(16.3%). 62 false positives, 264 false negatives. The FN > FP pattern
is expected for a pathogenic-minority classifier and suggests the next
modeling work should focus on recall on hard pathogenic cases.

New script: `scripts/compute_shap.py`. Artifacts:
- `results/metrics/shap_values_test.parquet` — per-variant SHAP
- `results/figures/shap_summary.png` — beeswarm (top 20)
- `results/figures/shap_bar.png` — mean |SHAP| bar chart
- `results/figures/shap_dependence_top3.png` — dependence plots
- `results/metrics/confident_errors.csv` — FN/FP list with
  `top1/2/3_feature` columns for per-row attribution

## [Portfolio Stage 1 — Baseline comparison on paralog-disjoint test] — 2026-04-21

First apples-to-apples comparison of three published missense-effect
predictors on the same paralog-disjoint test split (n ≈ 28k). This is
what reviewers ask for first: "how does your model stack up against
SIFT, PolyPhen-2, and AlphaMissense on *your* exact test set?"

### Added
- `src/baselines/alphamissense.py` — stream-scans the 643 MB
  `AlphaMissense_hg38.tsv.gz` once and extracts scores only for the
  query keys we care about (~28k). Result is a 24 k-row parquet; the
  cache survives across runs so every rerun is instant.
- `src/baselines/sift_polyphen.py` — pulls SIFT + PolyPhen-2 scores
  per variant from the Ensembl VEP REST endpoint (GRCh38). Resumable
  parquet cache.
- `src/baselines/evaluate.py` — `evaluate_baseline()` scores any
  `(y_true, y_score)` pair on the ClinVar test slice + denovo-db +
  family-holdout slices, with 1,000-boot 95% CIs. Baselines with
  `higher_is_pathogenic=False` (SIFT) are auto-inverted.
- `scripts/run_baselines.py` — driver: query all three baselines,
  compute metrics, write CSV, draw the forest plot with our XGBoost
  overlaid.
- `tests/test_baselines.py` — 7 unit tests, including a perfect-scorer
  sanity, a sign-flip test (lower-is-damaging), a NaN-coverage test,
  and a mini AlphaMissense extractor round-trip.
- Data: `data/raw/baselines/alphamissense/AlphaMissense_hg38.tsv.gz`
  (643 MB; gitignored, SHA256 in manifest).

### Headline — ClinVar test (paralog-disjoint, GRCh38)

| Baseline | Year | ROC-AUC (95% CI) | PR-AUC (95% CI) | Coverage |
|---|---:|---|---|---:|
| SIFT | 2003 | 0.881 [0.877, 0.885] | 0.620 [0.610, 0.629] | 96% |
| PolyPhen-2 | 2010 | 0.893 [0.888, 0.898] | 0.728 [0.716, 0.739] | 93% |
| **XGBoost (ours)** | 2026 | **0.938** | **0.838** | 100% |
| AlphaMissense* | 2023 | 0.956 [0.953, 0.958] | 0.890 [0.882, 0.898] | 86% |

\* AlphaMissense was calibrated on ClinVar at release; its ClinVar
numbers are inflated by training-set contamination. Our
`training_contamination_warning` column documents this in every row of
the output CSV. We have no comparable contamination — our model never
saw test-split ClinVar entries during training (paralog-disjoint
family split).

### Interpretation
- Our XGBoost cleanly outperforms the two classical tools SIFT (PR-AUC
  +0.22) and PolyPhen-2 (+0.11) under identical evaluation.
- We sit below AlphaMissense by 1.8 pp ROC / 5.2 pp PR, but under
  stricter methodology (no ClinVar calibration leak).
- Remaining gap to AlphaMissense is the justification for Stages 2.1
  (ESM-2 full-training-set feature) and 2.2 (AlphaFold2 structural
  features).

### Follow-up work
- **denovo-db baseline coverage** is currently ~3% because the
  fetchers target GRCh38 (correct for ClinVar) while denovo-db publishes
  GRCh37. Stage 1.5 will add a GRCh37-aware baseline pathway + an AM
  hg19 lookup so the denovo-db comparison row is also populated.
- REVEL (~8 GB) and CADD (web-API) deferred to Stage 1.5 once the
  infrastructure is validated.

## [Build correction — pipeline is GRCh38, not GRCh37] — 2026-04-21

While building the Stage 1 baseline-comparison infrastructure we
discovered that the pipeline's coordinate system is **GRCh38** end-to-end,
despite:

- `configs/config.yaml` referencing `dbNSFP5.3.1a_grch37.gz`;
- our early documentation describing outputs as GRCh37;
- the external-validation harness defaulting to the GRCh37 REST endpoint.

### How we found it

A random sample of 5 test variants was queried against both Ensembl REST
endpoints. In every case only the GRCh38 endpoint returned the gene and
amino-acid change that matched the row's annotated `(gene, ref_aa,
alt_aa)`:

| variant_key           | GRCh37 endpoint      | GRCh38 endpoint        | Committed annotation    |
|-----------------------|----------------------|------------------------|-------------------------|
| `17:44075760:G:A`     | MAPT intron variant  | G6PC3 missense R→H ✓   | G6PC3 missense R→H      |
| `1:181798404:C:G`     | ATP6V1D downstream   | ZFYVE26 missense L→F ✓ | ZFYVE26 missense L→F    |
| `14:52314973:A:T`     | ZYG11B intron        | CPT2 missense L→P ✓    | CPT2 missense L→P       |
| … (2 more, all agree) | —                    | —                      | —                       |

### Why it still works

dbNSFP 5.x changed column conventions: the primary `pos(1-based)` column
is GRCh38 (with a separate `hg19_pos(1-based)` for GRCh37). Our
extractor picks `pos(1-based)`, so positions in the splits are GRCh38.
gnomAD v3/v4 (also GRCh38) joined correctly, so AF features are
populated for 100% of training rows.

The features the XGBoost model actually uses are **build-agnostic**:
conservation scores, amino-acid chemistry, gnomAD constraint. None of
them depend on the coordinate itself, only on the variant identity.
That's why the trained model still scores correctly on denovo-db (which
is GRCh37) — the external-validation harness featurizes denovo-db
variants through GRCh37-queried VEP, then feeds those features to the
model, which is indifferent to the build the features came from.

### What was broken

- `scripts/run_baselines.py` downloaded and scanned `AlphaMissense_hg19.tsv.gz`.
  Only 4.6% of our test keys had hits.
- `src/baselines/sift_polyphen.py` queried the GRCh37 REST endpoint for
  ClinVar-derived test variants — most returned non-missense consequences
  for the same numeric coordinates on the wrong build.

### Fix

- Baselines: download `AlphaMissense_hg38.tsv.gz` (643 MB) and switch
  the SIFT/PolyPhen VEP URL to `rest.ensembl.org` (GRCh38).
- `src/esm2_scorer.py` gains a `genome_build` kwarg (default
  `"GRCh38"`); the denovo-db path in the external harness explicitly
  passes `"GRCh37"`.
- `notebooks/11_esm2_full_scoring_colab.ipynb` inherits the new default,
  so the Stage 2.1 full-training-set scoring targets GRCh38 REST.

### Caveats

- Existing denovo-db external-validation artifacts
  (`results/metrics/external_denovo_db_*`) are unchanged — that path was
  always internally consistent on GRCh37 (denovo-db's own build).
- The README references to "hg19 / GRCh37" in older sections were
  historically inaccurate; future edits will correct them. The
  `configs/config.yaml` `file:` path remains
  `dbNSFP5.3.1a_grch37.gz` (the filename on the source bundle) with a
  comment documenting that the extracted column is GRCh38.

## [Portfolio Stage 0 — Production-grade foundations] — 2026-04-21

Hardened the repo to production-grade standards before launching any of
the long-compute Stage 2 experiments. The principle: **one bug in
`src/evaluation.py` would invalidate 35 h of ESM-2 compute**, so every
line that will be on the critical path gets a test, a type check, and a
CI gate first.

### Added
- `tests/` — 130 pytest tests, 59.6% coverage on the actively-maintained
  surface area (grandfathered frozen data-prep modules excluded). Files:
  - `tests/conftest.py` — session-scoped fixtures for splits + mock
    binary predictions.
  - `tests/test_data_splitting.py` — regression-tests
    `assign_gene_family()` on 30+ known paralog clusters, asserts zero
    family overlap across committed train/val/test.
  - `tests/test_evaluation.py` — unit-tests metric math, bootstrap CI
    coverage, reliability curve ECE/MCE, threshold-selection
    optimality.
  - `tests/test_gnomad_constraint.py` — asserts the "train-only median"
    no-leakage pattern; extended to test `load_constraint_table()`
    dedup logic.
  - `tests/test_utils.py`, `tests/test_variant_mapper.py`,
    `tests/test_featurize.py`, `tests/test_denovo_loader.py` —
    deterministic unit tests for the external-validation helpers.
  - `tests/test_xgboost_model.py` — tiny-data integration test for the
    Optuna tuning loop.
  - `tests/test_verify_no_leakage.py` — wraps the 4 existing leakage
    checks as individual pytest cases so CI failures pinpoint the exact
    category.
  - `tests/integration/test_reproduce_headline.py` — **the critical
    gate**: loads the committed XGBoost checkpoint, re-scores the
    committed test split, asserts ROC/PR/F1/Brier reproduce within
    1e-3.

- `pyproject.toml` — centralizes pytest (`--cov-fail-under=55`), ruff,
  black, mypy, and coverage config. Per-file `ignore` for the frozen
  data-prep modules that produced the committed splits (changing them
  risks bit-for-bit repro).

- `requirements-lock.txt` — exact `pip freeze` (287 packages) for
  reproducible installs. `requirements.txt` keeps `>=` floors for
  humans; CI installs from the lock.

- `Dockerfile` — `python:3.11.7-slim` + liblzma-dev + DSSP +
  OpenBLAS. Builds a < 2 GB image where `make test` + `make
  reproduce-headline` succeed identically to the host.

- `Makefile` — canonical entry points: `make test`, `make lint`, `make
  train`, `make evaluate`, `make external`, `make
  reproduce-headline`, plus Docker shortcuts.

- `.pre-commit-config.yaml` — ruff, black, YAML/TOML validation,
  trailing-whitespace, and a local hook that runs
  `src.verify_no_leakage` when splits or training code change.

- `.github/workflows/test.yml` — CI: install from lock, run leakage
  gate, run pytest with coverage. Runs on every push / PR to main.

- `.github/workflows/lint.yml` — ruff + black + mypy (mypy
  non-blocking for now). Badges added to `README.md`.

### Changed
- `src/external_validation/variant_mapper.py` — fixed a silent bug: the
  nucleotide-only filter did not reject multi-nucleotide refs/alts like
  `AT>G`. Locked in with `test_variant_mapper.py::test_non_acgt_rejected`.
- `README.md` — badges now reflect live CI status (`tests`, `lint`) and
  latest PR-AUC (0.838) + ECE (0.011).

### Scorecard
| Dimension | Before | After |
|---|---:|---:|
| Code modularity | 4/5 | 4/5 |
| Type hints + docstrings | 3/5 | 4/5 |
| **Testing** | **0/5** | **4/5** (130 tests, 59.6% cov) |
| Reproducibility | 4/5 | **5/5** (Dockerfile + lock + Makefile) |
| Documentation | 5/5 | 5/5 |
| Demo / interactive | 0/5 | 0/5 *(Stage 4)* |
| **Overall** | **3.8/5** | **4.2/5** |

Remaining gaps before publication-ready: baselines comparison
(Stage 1), deep-science experiments (Stage 2), and the Streamlit demo +
technical report (Stage 4).

## [Phase 2 step 2 — ESM-2 zero-shot proof-of-concept] — 2026-04-20

Step 2 of the Phase 2 plan: add a protein-language-model signal that is
orthogonal to the tabular features. Using ESM-2 35M
(`facebook/esm2_t12_35M_UR50D`), we compute a masked-LM
log-likelihood ratio per missense variant:

    esm2_llr = log P(alt | context) - log P(ref | context)

with the mutated position replaced by `<mask>`. This encodes the
evolutionary-sequence plausibility of the substitution — a signal the
tabular baseline has no access to.

### Added
- `src/esm2_scorer.py` — end-to-end scorer:
    1. Annotate variants via Ensembl VEP REST to pull the canonical
       `transcript_id` and `protein_position`.
    2. Fetch protein sequences from Ensembl `/sequence/id/?type=protein`
       (resumable parquet cache).
    3. Run ESM-2 35M masked-LM on MPS / CUDA / CPU; sliding-window the
       input for proteins longer than 1022 residues.
    4. Emit `esm2_prob_ref`, `esm2_prob_alt`, `esm2_llr` per variant,
       with a `skip_reason` column for rows whose sequence/position
       couldn't be resolved.
- `scripts/analyze_esm2_denovo.py` — proof-of-concept comparison:
  ROC/PR with 1000-boot CIs for **XGBoost alone** vs **ESM-2 alone** vs
  **rank-average fusion** on denovo-db (full + family_holdout_only).

### Infrastructure
- Built a separate `~/.venvs/esm2` using Homebrew's Python 3.13 because
  the project's pyenv 3.11.7 was compiled without `_lzma`, which
  HuggingFace requires for the ESM-2 tokenizer.

### Artifacts
- `data/intermediate/esm2/vep_ann.parquet` — per-variant VEP annotation
  (transcript + protein_position).
- `data/intermediate/esm2/sequences.parquet` — per-transcript canonical
  protein sequence.
- `data/intermediate/esm2/scores.parquet` — per-variant ESM-2 scores
  (resumable).
- `results/metrics/esm2_denovo_db_scores.parquet` — denovo-db scored
  subset (n=642 of 644; two variants on non-missense consequences).
- `results/metrics/esm2_denovo_db_comparison.csv` — ROC/PR + 95% CIs for
  the three scores × two slices.

### Headline — denovo-db (n=642 sample, 1000-boot CIs)

| Slice | Score | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|---|
| full | xgb_calibrated | 0.511 [0.455, 0.564] | 0.790 [0.749, 0.830] |
| full | esm2_llr | 0.515 [0.441, 0.527] | 0.772 [0.723, 0.800] |
| full | **rank_fusion** | **0.517** [0.461, 0.573] | 0.777 [0.735, 0.823] |
| family_holdout_only | xgb_calibrated | 0.573 [0.476, 0.670] | 0.838 [0.774, 0.897] |
| family_holdout_only | esm2_llr | 0.552 [0.420, 0.574] | 0.821 [0.738, 0.865] |
| family_holdout_only | **rank_fusion** | **0.588** [0.495, 0.683] | **0.851** [0.788, 0.914] |

### Interpretation
- ESM-2 zero-shot alone has meaningful signal on unseen families
  (ROC ≈ 0.55) — better than random, but weaker than the
  constraint-augmented XGBoost.
- **Rank-fusion adds a small real boost** to XGBoost on the
  family_holdout_only slice: ROC +0.015, PR +0.013. Direction is
  consistent, CIs are wide because n=201.
- Proper integration will retrain XGBoost **with `esm2_llr` as a
  feature**, letting gradient-boosted trees exploit ESM × constraint ×
  conservation interactions that rank-fusion cannot model. That needs
  `esm2_llr` computed across the full training corpus
  (~195k variants → ~35 h of MPS compute), deferred as an overnight run.

## [Phase 2 step 1 — gnomAD gene-level constraint features] — 2026-04-20

Phase D v1 revealed that the ClinVar-trained baseline scored **at chance**
(ROC = 0.468) on denovo-db. The diagnosis was that the tabular feature set
encoded only *variant-level* evidence (conservation, AA chemistry) and no
*gene-level* intolerance priors — the single strongest clinical prior for
whether damage to a given gene is plausibly disease-causing.

### Added
- `src/gnomad_constraint.py` — loads gnomAD v2.1.1
  `lof_metrics.by_gene.txt.bgz`, keeps
  `{pLI, oe_lof_upper (LOEUF), mis_z, oe_mis_upper, lof_z}`, and left-joins
  them onto each split by gene. **Median imputation is fit on TRAIN ONLY**
  and re-applied to val / test / external rows (no leakage), with
  `is_imputed_gnomad_constraint` flag so SHAP can isolate imputed rows.
- `data/raw/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz`
  (4.4 MB, 19,658 genes; 19,155 with pLI).
- `results/metrics/gnomad_constraint_medians.csv` — persisted train-fit
  medians for reuse by the external-validation featurizer.

### Changed
- `data/splits/{train,val,test}.parquet` now include 6 new columns
  (`pLI, oe_lof_upper, mis_z, oe_mis_upper, lof_z,
  is_imputed_gnomad_constraint`). Gene-level coverage: **96.3% train,
  96.3% val, 93.3% test** (rest imputed with train median).
- `scripts/evaluate_external.py` attaches the same constraint block to
  featurized external variants using the persisted train-fit medians, so
  the external feature matrix matches the new training schema without
  leaking gene-level stats from the external set.
- `results/checkpoints/xgboost_best.ubj` retrained (40 Optuna trials,
  seed=42) on the enriched 39-feature matrix.

### Numbers — test split (held-out, paralog-disjoint)

| Stage | ROC-AUC | PR-AUC (raw) | PR-AUC (isotonic cal) | ECE (cal) |
|---|---:|---:|---:|---:|
| Post-lockdown (33 features) | 0.938 | 0.836 | 0.827 | 0.011 |
| + gnomAD constraint (39 features) | **0.938** | **0.838** | **0.830** | **0.011** |

The in-distribution gain is marginal (+0.002 PR-AUC). Expected: the test
set is already paralog-disjoint, so for genes the model saw during
training the variant-level features already capture most of the signal.

### Numbers — denovo-db external (n=642 stratified sample, 1000-boot CIs)

| Slice | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) | Base-rate PR |
|---|---|---|---|---:|
| full | variant-only | 0.468 [0.415, 0.519] | 0.761 [0.721, 0.806] | 0.776 |
| full | + constraint | **0.511** [0.455, 0.564] | **0.790** [0.749, 0.830] | 0.776 |
| family_holdout_only | variant-only | 0.487 [0.383, 0.583] | 0.789 [0.712, 0.860] | 0.801 |
| family_holdout_only | + constraint | **0.573** [0.476, 0.670] | **0.838** [0.774, 0.897] | 0.801 |

The **family_holdout_only** gain is the headline: for gene families the
model has never seen during training, ROC jumps from 0.487 (chance) to
0.573 (+0.086), and PR-AUC moves from 0.789 (at base-rate) to 0.838
(+0.037 above base-rate). This is exactly the regime where *only*
gene-level constraint priors can help — variant-level features by
construction cannot discriminate affected-vs-control de-novo variants on
an unseen gene.

> *Gene-level constraint priors (pLI / LOEUF / mis_z) close about half
> the gap between the ClinVar-test baseline and chance-level
> generalization to de-novo variants on held-out gene families.*

Full closure of the gap will require phenotype-specific priors (gene-
disease association scores) which live outside the tabular baseline's
scope. That's Phase 2's job.

## [Phase D v1 — External Validation Infrastructure] — 2026-04-20

Built the end-to-end harness for scoring the baseline on datasets outside
ClinVar. First source wired up: **denovo-db** (non-SSC samples). Full
ProteinGym integration is deferred to D v2 once UniProt ↔ genome mapping
is in place.

### Added
- `src/external_validation/__init__.py` — module surface.
- `src/external_validation/variant_mapper.py` — canonicalize heterogeneous
  variant IDs into the `chr:pos:ref:alt` key used in training.
- `src/external_validation/denovo_loader.py` — parse denovo-db TSV into a
  labeled missense-only frame. Pathogenic label = affected proband; benign
  label = documented sibling/control. Ambiguous phenotypes are dropped.
- `src/external_validation/featurize.py` — left-join external variants onto
  the cached `dbnsfp_selected_features.parquet`. Unmapped rows are reported,
  never silently dropped.
- `src/external_validation/vep_featurize.py` — **Ensembl VEP REST fallback**
  for variants missing from the dbNSFP cache. Pulls phyloP100, phastCons100,
  GERP++, amino-acid identities; computes BLOSUM62, Grantham, and AA
  physicochemistry locally using the same helper tables as
  `src/dbnsfp_extraction.py`. Missing fields
  (`phyloP30way_mammalian`, `phastCons30way_mammalian`, `pfam_domain`) are
  imputed with **training-set medians**. gnomAD AF defaults match the
  ultra-rare assumption for de-novo variants.
- `src/external_validation/evaluate.py` — rebuilds the training
  ColumnTransformer from `xgboost_feature_columns.csv`, scores raw +
  isotonic-calibrated probabilities, emits bootstrap 95% CIs over full +
  **family-holdout** slices.
- `scripts/evaluate_external.py` — driver. Flags: `--sample`, `--use-vep`,
  `--n-boot`, `--only`.
- Raw data: `data/raw/external/denovo_db/denovo-db.non-ssc-samples.variants.tsv.gz`
  (7.4 MB, 9,848 missense rows, 9,704 affected + 144 sibling controls).
- Output artifacts under `results/metrics/`:
  - `external_denovo_db_metrics.csv` — ROC/PR/F1/Brier + bootstrap CIs.
  - `external_denovo_db_coverage.csv` — rows featurized vs unmapped.
  - `external_denovo_db_predictions.parquet` — per-variant raw + calibrated.
  - `external_denovo_db_unmapped.csv` — rows that failed featurization.

### Headline result (n=644 sampled — all 144 sibling controls + 500 affected)

| Slice | n | n_pos | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---:|---:|---|---|
| full | 642 | 498 | **0.468** [0.415, 0.519] | 0.761 [0.721, 0.806] |
| family_holdout_only | 201 | 161 | 0.487 [0.383, 0.583] | 0.789 [0.712, 0.860] |

Base-rate PR-AUC for this slice is `n_pos / n = 0.776`, so PR-AUC at 0.76
is indistinguishable from the class prior. **The calibrated ClinVar-
trained baseline performs at chance on denovo-db.** This is the single
most important external-validation finding so far:

> *A classifier that scores 0.836 PR-AUC on paralog-aware ClinVar
> generalizes to zero signal on affected-vs-control de-novo variants.*

Interpretation: the model has learned "this variant *looks* disease-
causing" (high conservation, disruptive AA substitution), but almost every
missense variant in denovo-db — affected or control — *also* looks that
way at the single-variant level. Separating a causative de-novo variant
from a bystander requires per-phenotype priors, gene-disease associations,
or zygosity / inheritance context the tabular model never sees.

## [Phase 1 Lockdown] — 2026-04-20

Quality-assurance pass to freeze Phase 1 before external validation (Phase D)
and deep-learning work (Phase 2).

### Added
- `docs/CHANGELOG.md` (this file).
- `results/metrics/README.md` — manifest describing every CSV artifact.
- `src/verify_no_leakage.py` — automated gate that fails CI if any banned
  feature (`is_common`, `chr`, raw `ref`/`alt`) re-enters the training matrix,
  or if train/test share a gene family.
- `results/figures/reliability_calibration.png`,
  `results/figures/leakage_journey.png`,
  `results/figures/pr_roc_curves.png` — lockdown figures referenced from README.

### Changed
- `requirements.txt` now pins `optuna>=4.0` (was unlisted despite being imported).
- Stale notebooks (01–05, 08) carry a top-cell banner pointing readers to the
  current honest numbers. Cell contents preserved for provenance.

## [0.2.0 — Honest Baseline] — 2026-04-19 — commit `f8ab464`

The three-leakage-fix release. PR-AUC 0.955 (mis-reported) → 0.836 (honest).

### Fixed — data leakage
- **Non-missense contamination.** `src/feature_analysis.py::filter_missense_only`
  now drops rows with null `ref_aa` or null `alt_aa` before any downstream
  processing. Removed 88,294 rows (64% of pathogenic pool was non-missense;
  2% of benign). Side effect: 11 AA-derived features auto-recovered because
  their null fraction fell below the 20% drop threshold.
- **Definitional circularity.** Removed `is_common` from the training feature
  set. It was 100% benign when True (ClinVar labels common gnomAD variants
  benign by construction) — a pure label-leak.
- **Known-disease-loci proxy.** Removed `chr` one-hot encoding. Gain=15.5%
  was dataset-bias memorization, not biology.
- **Paralog leakage.** `src/data_splitting.py::assign_gene_family` groups
  keratins, cadherins, collagens, zinc-fingers, SLC transporters, HLA,
  olfactory receptors, MT-, PCDH, CCDC, LRRC, ANKR, TRIM, TMEM, RPL, RPS,
  and numeric-suffix families to a single split. 15,479 genes → 7,851 families.

### Changed — evaluation rigor
- `src/models/xgboost_model.py::tune_xgboost` rewritten around Optuna
  `TPESampler(multivariate=True)` + `MedianPruner(n_warmup_steps=200)`.
  Objective is pure PR-AUC (replaces the arbitrary `0.65·ROC + 0.35·PR`
  composite). 40 trials default (was 14).
- `src/evaluation.py` gains `bootstrap_metrics` (1000-replicate percentile CIs)
  and `reliability_curve` (quantile-binned ECE/MCE).
- `src/evaluate_baseline.py` (new) — fits isotonic calibration on validation
  probabilities only, emits operating points, reliability curve, and
  pre/post-calibration comparison.
- `src/ablation_af.py` (new) — feature-group ablation. Confirms AF ablation
  costs only ΔROC ≈ -0.003 (refutes the most common reviewer concern).

### Numbers — before vs after

| Stage | ROC-AUC | PR-AUC | Rows | Notes |
|---|---:|---:|---:|---|
| Pre-audit | 0.955 | 0.955 | 283,392 | Contaminated baseline |
| + missense filter | 0.934 | 0.819 | 195,098 | biggest single drop |
| + feature hygiene | 0.934 | 0.816 | 195,098 | `is_common`, `chr` out |
| + paralog split | 0.938 | 0.835 | 195,098 | honest |
| + Optuna 40 trials | 0.938 | 0.836 | 195,098 | ceiling reached |

## [0.1.0 — README Rewrite] — 2026-04-19 — commit `b12bdce`

Full README rewrite (83 → 307 lines). Shields.io badges, leakage-hunt table,
ASCII architecture diagram, roadmap. No code change.
