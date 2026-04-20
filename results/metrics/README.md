# `results/metrics/` — artifact manifest

Every CSV in this folder is generated deterministically from `src/` with `seed=42`.
This file is the **single source of truth** — numbers in notebooks or the main
README that disagree with these CSVs are stale.

## Leakage-fix journey (the project's main story)

| File | Produced by | What it shows |
|---|---|---|
| `leakage_fix_journey.csv` | manual, audited from experiment logs | The 5-stage decline from 0.955 → 0.836 PR-AUC as each leakage source is fixed. The headline honest-baseline table. |

## Current (post-leakage-fix) baseline

| File | Produced by | Contents |
|---|---|---|
| `xgboost_split_metrics.csv` | `src/training.py` | Point metrics (ROC/PR/F1/Brier) on train/val/test for the final model. |
| `xgboost_bootstrap_ci.csv` | `src/evaluate_baseline.py` | 1000-replicate nonparametric bootstrap 95% CIs on val + test for ROC, PR, F1, Brier, MCC, precision, recall. |
| `xgboost_calibrated_metrics.csv` | `src/evaluate_baseline.py` | Same metric block after fitting `IsotonicRegression` on val probabilities. |
| `xgboost_reliability_curve.csv` | `src/evaluate_baseline.py` | 15-bin quantile reliability curve. `df.attrs['ECE']` / `['MCE']` persisted as first-row columns. |
| `xgboost_calibration_summary.csv` | `src/evaluate_baseline.py` | Pre- vs post-calibration ECE/MCE/Brier/log-loss comparison. |
| `xgboost_operating_points.csv` | `src/evaluate_baseline.py` | Threshold, precision, recall, F1 at recall≥{.80,.90,.95,.99} and precision≥{.80,.90,.95,.99}. |
| `xgboost_val_threshold_curve.csv` | `src/training.py` | Validation threshold sweep used to pick the operating threshold. |
| `xgboost_predictions.parquet` | `src/evaluate_baseline.py` | Row-level predictions on val + test with `p_raw`, `p_calibrated`, `label`, `gene`, `split`. |

## Tuning

| File | Produced by | Contents |
|---|---|---|
| `xgboost_best_params.csv` | `src/training.py` | Winning Optuna hyperparameters. |
| `xgboost_tuning_history.csv` | `src/training.py` | All 40 Optuna trials with per-trial PR-AUC, ROC-AUC, best_iteration, and sampled params. |
| `xgboost_feature_columns.csv` | `src/training.py` | Exact ordered feature list the model was trained on (leakage-vetted). |

## External validation

| File | Produced by | Contents |
|---|---|---|
| `external_denovo_db_metrics.csv` | `scripts/evaluate_external.py` | ROC/PR/F1/Brier point + 1000-bootstrap 95% CIs for `full` and `family_holdout_only` slices of denovo-db. |
| `external_denovo_db_coverage.csv` | `scripts/evaluate_external.py` | How many raw external rows were featurized vs unmapped at the dbNSFP / VEP step. |
| `external_denovo_db_predictions.parquet` | `scripts/evaluate_external.py` | Per-variant raw + isotonic-calibrated probability, family-holdout flag. |
| `external_denovo_db_unmapped.csv` | `scripts/evaluate_external.py` | External variants we could not featurize (reported openly, never silently dropped). |

## Feature augmentation

| File | Produced by | Contents |
|---|---|---|
| `gnomad_constraint_medians.csv` | `src/gnomad_constraint.py` | Train-fit median values for `pLI, oe_lof_upper, mis_z, oe_mis_upper, lof_z`. Reused by the external featurizer so val/test/external rows impute from the training distribution only (no leakage). |
| `esm2_denovo_db_scores.parquet` | `src/esm2_scorer.py` | Per-variant ESM-2 35M zero-shot scores on denovo-db (n=642 of 644 resolvable): `esm2_prob_ref`, `esm2_prob_alt`, `esm2_llr`, plus `transcript_id` and `protein_position`. |
| `esm2_denovo_db_comparison.csv` | `scripts/analyze_esm2_denovo.py` | Point + 95% bootstrap CI for XGBoost, ESM-2 alone, and rank-fusion on the `full` and `family_holdout_only` slices of denovo-db. |

## Ablations

| File | Produced by | Contents |
|---|---|---|
| `ablation_af.csv` | `src/ablation_af.py` | ΔROC / ΔPR when removing AF / conservation / AA-property feature groups. Refutes the "AF circularity inflates our score" concern (ΔROC ≈ -0.003). |
| `ablation_feature_groups.csv` | `src/ablation_af.py` (legacy) | Broader feature-group ablation table. |

## Conventions

- **Seeds.** `seed=42` everywhere. Bootstrap uses `np.random.default_rng(42)`.
- **Splits.** Paralog-aware family-level `GroupShuffleSplit` (see `src/data_splitting.py::assign_gene_family`). 7,851 families from 15,479 genes.
- **Rows.** 195,098 missense variants after strict `ref_aa AND alt_aa` non-null filter (see `src/feature_analysis.py::filter_missense_only`).
- **Features.** 33 columns, no `is_common`, no `chr` one-hot, no raw `ref`/`alt` (leakage-vetted in `src/training.py::select_feature_columns`).

## Regenerating everything

```bash
python -m src.training          # refits model, dumps split_metrics, tuning_history, best_params
python -m src.evaluate_baseline # bootstrap CIs + calibration + operating points
python -m src.ablation_af       # feature-group ablation
```

If any CSV here ever disagrees with a number printed in `README.md` or a notebook,
**trust this folder** and file a correction.
