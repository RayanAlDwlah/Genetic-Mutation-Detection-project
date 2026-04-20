# Changelog

All notable changes to the honest-baseline pipeline. Dates are ISO-8601.
Commits are on `origin/main`.

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
