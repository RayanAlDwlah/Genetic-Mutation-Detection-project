# Changelog

All notable changes to the honest-baseline pipeline. Dates are ISO-8601.
Commits are on `origin/main`.

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
