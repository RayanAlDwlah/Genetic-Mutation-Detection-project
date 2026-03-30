# Genetic Variant Pathogenicity Classification

Binary classification of human missense genetic variants as **Pathogenic (1)** or **Benign (0)**.

## Project Overview

This project builds a machine learning pipeline that predicts the clinical significance of
missense variants using conservation scores, amino acid properties, and population allele
frequencies. The final system includes a gradient-boosted baseline (XGBoost) and two deep
learning models (1D CNN and ESM-2 transfer learning) evaluated under a strict gene-level
cross-validation protocol that prevents data leakage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

## Quick Start From a GitHub Clone

This repository intentionally keeps the huge upstream raw sources out of git, but it now
tracks the compact parquet snapshots in `data/intermediate/`, `data/processed/`, and
`data/splits/`. A collaborator can therefore clone the repo and run the baseline training
immediately:

```bash
python -m src.training
```

Artifacts are written to `results/checkpoints/` and `results/metrics/`.

## Data Sources

| Source  | Role                                             | Version   | Genome Build |
|---------|--------------------------------------------------|-----------|--------------|
| ClinVar | Supervised labels (Pathogenic / Benign)          | 2026-02   | GRCh37       |
| gnomAD  | Population allele frequency features             | r2.1.1    | GRCh37       |
| dbNSFP  | Pre-computed conservation & physicochemical features | 5.3.1a | GRCh37       |
| UniProt | Protein sequences (for ESM-2 model — Phase 2)   | 2025_01   | N/A          |

**Label policy:**
- `Pathogenic` / `Likely pathogenic` → `1`
- `Benign` / `Likely benign` → `0`
- Variants of Uncertain Significance (VUS) are excluded from supervised training.

**Excluded predictors:** REVEL, ClinPred, MetaLR, MetaSVM, MetaRNN, BayesDel, VEST4, M-CAP.
These are ClinVar-derived meta-predictors that would introduce circularity into evaluation.

Raw source files under `data/raw/` are intentionally excluded from git because they are too
large for GitHub. Only the cleaned/processed parquet snapshots needed for reproducible runs are
versioned in the repository.

## Directory Structure

```
data/
  raw/           Raw downloaded files for full rebuilds (not committed)
  intermediate/  Versioned cleaned parquet snapshots for ClinVar / gnomAD / dbNSFP
  processed/     Versioned merged and feature-engineered datasets
  splits/        Versioned gene-level train / val / test parquet files
    strict/      High-quality splits (review_stars ≥ 2, no imputation)

results/
  figures/       All plots (EDA, evaluation, SHAP, ablation)
  checkpoints/   Saved model weights (.ubj for XGBoost)
  metrics/       CSV files with evaluation metrics and tuning history

notebooks/       Presentation layer — full analysis story cell by cell
src/             Logic layer — reusable, importable Python modules
configs/         YAML configuration (paths, hyperparameter defaults)
```

## Running the Project

A fresh clone can train the XGBoost baseline immediately:

```bash
python -m src.training
```

To rebuild the full pipeline from raw sources, first download the upstream data into
`data/raw/`, then run each step below or follow the corresponding notebook.

```bash
# Step 1 — Clean ClinVar labels
python -m src.clinvar_cleaning --config configs/config.yaml

# Step 2 — Extract gnomAD allele frequencies
python -m src.gnomad_extraction --input data/raw/gnomad/gnomad.exomes.r2.1.1.sites.vcf.bgz \
    --clinvar-variants data/intermediate/clinvar_labeled_clean.parquet

# Step 3 — Extract dbNSFP features
python -m src.dbnsfp_extraction --config configs/config.yaml

# Step 4 — Merge datasets
python -m src.data_merge --config configs/config.yaml

# Step 5 — Feature analysis (correlation filtering + two dataset versions)
python -m src.feature_analysis --config configs/config.yaml

# Step 6 — Gene-level train/val/test split
python -m src.data_splitting --config configs/config.yaml

# Step 7 — Train XGBoost baseline
python -m src.training
```

## Models

| Model                     | File                          | Status          |
|---------------------------|-------------------------------|-----------------|
| XGBoost baseline          | `src/models/xgboost_model.py` | ✅ Complete     |
| 1D CNN + Attention        | `src/models/cnn_model.py`     | 🔜 Phase 2     |
| ESM-2 Transfer Learning   | `src/models/esm2_model.py`    | 🔜 Phase 2     |

**XGBoost results (test set):** ROC-AUC = 0.955

## Key Design Decisions

- **Gene-level split:** Genes are assigned to only one split. This prevents the model from
  memorizing gene → label patterns and ensures evaluation reflects generalization to unseen genes.
- **Circularity protection:** Meta-predictors trained on ClinVar (REVEL, ClinPred, etc.) are
  excluded from features to avoid inflated performance on ClinVar-based labels.
- **Two dataset versions:** A *balanced* version (283K variants, review ≥ 1) and a *strict*
  version (37K variants, review ≥ 2) allow ablation of data quality vs. quantity trade-offs.

## Team

Genetic Graduation Project — King Khalid University.

## License

Academic/research use only. To be finalized by the team.
