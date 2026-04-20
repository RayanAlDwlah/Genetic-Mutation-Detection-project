<div align="center">

# 🧬 Missense Variant Pathogenicity Classification

### *Finding the honest ceiling of tabular ML for clinical genomics*

**King Khalid University — Computer Science Graduation Project**

![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.938-brightgreen?style=flat-square)
![PR--AUC](https://img.shields.io/badge/PR--AUC-0.836-green?style=flat-square)
![ECE](https://img.shields.io/badge/ECE-0.015-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

</div>

---

## 🎯 TL;DR

We classify human **missense variants** as *pathogenic* or *benign* using ClinVar labels, gnomAD allele frequencies, and dbNSFP conservation/biochemistry features — then we **audited our own baseline** and discovered three leakage sources that had inflated our PR-AUC from an honest **0.836** to an apparent **0.955**. Fixing them is the story this repository tells.

> *The paper nobody writes: "Here's what we thought we had. Here's the bug. Here's the real number."*

---

## 📊 Headline Results

| Metric | Test Set (family-split, missense-only, tuned) |
|---|---|
| **ROC-AUC** | **0.938** [0.935, 0.941] |
| **PR-AUC** | **0.836** [0.827, 0.843] |
| **F1** | 0.773 |
| **Brier (calibrated)** | **0.084** |
| **ECE (calibrated)** | **0.015** |
| **Operating point** — recall ≥ 95% | precision 55%, threshold 0.11 |
| **Operating point** — precision ≥ 99% | *not reachable* (documented limitation) |

All confidence intervals from 1,000 nonparametric bootstrap replicates.

---

## 🕵️ The Leakage Hunt — How a 0.955 Became an Honest 0.836

We started with a baseline that looked too good:

| Stage | PR-AUC | ROC-AUC | What we found |
|---|---:|---:|---|
| **1. Initial baseline** | 0.955 | 0.955 | "Suspicious ceiling" |
| **2. Missense filter** ⚠️ | **0.819** | 0.934 | **64% of pathogenic rows had no `alt_aa`** — stop-gained / splice-site / UTR variants leaking in. Model was learning *"if AA features are null → pathogenic."* |
| **3. Feature hygiene** | 0.816 | 0.934 | `is_common=True` was **100% benign** (definitional circularity). `chr` one-hot contributed 15% gain as a known-disease-loci proxy. Removed. |
| **4. Paralog-aware split** | 0.835 | 0.938 | Plain gene-split shared **52% of gene-prefix families** (ZNF\*, SLC\*, KRT\*, TMEM\*) between train and test. Family-level split closes this. |
| **5. Optuna retuning** | **0.836** | **0.938** | 40 TPE trials landed within 0.001 of each other → **the baseline is at its feature-limited ceiling**. |

📄 Full journey tracked in [`results/metrics/leakage_fix_journey.csv`](results/metrics/leakage_fix_journey.csv)

<p align="center">
  <img src="results/figures/leakage_journey.png" width="720" alt="Leakage-fix journey — PR-AUC and ROC-AUC across the five audit stages">
</p>

---

## 📐 Calibration & Held-Out Performance

<p align="center">
  <img src="results/figures/reliability_calibration.png" width="720" alt="Reliability diagrams — isotonic calibration brings ECE from 0.07 to 0.015">
</p>

<p align="center">
  <img src="results/figures/pr_roc_curves.png" width="720" alt="ROC and Precision-Recall curves on the held-out test set with 95% bootstrap CIs">
</p>

---

## 🔬 Where We Stand vs. Prior Work

A fair comparison requires matching *what was measured*. The table below places our honest number next to headline numbers from recent missense-classification papers, flagging known contamination sources.

| Method (year) | Family | Train size | Test | Reported AUC | Leakage guards |
|---|---|---:|---|---:|---|
| **Ours (2026)** | XGBoost + tabular | **195K missense** | 28K (family-split) | **ROC 0.938 · PR 0.836** | ✅ missense-only · ✅ paralog-aware · ✅ no meta-predictors |
| VARITY (2021) | Gradient boosting | ~35K (HumsaVar) | ~6K ClinVar | ROC ~0.90 | Gene-level split (no paralog guard) |
| mvPPT (2023) | Gradient boosting | ~150K | ClinVar subset | ROC ~0.94 | 🔴 uses REVEL + CADD (ClinVar-trained) as features |
| MAGPIE (2024) | Gradient boosting | ~250K | Multi-benchmark | ROC ~0.92 | Paralog status not documented |
| MVP (2021) | 1D CNN | ~112K | ~12K ClinVar | ROC ~0.88 | HGMD-trained |
| MutFormer (2023) | Transformer | ~230K | ~25K | ROC ~0.93 | HGMD-trained |
| ESM-1b zero-shot (2023) | PLM (no fine-tune) | 250M seq pre-train | 36K ClinVar | ROC ~0.85 | ✅ never sees ClinVar |
| AlphaMissense (2023) | PLM + primate | 250M seq + primates | 18,924 ClinVar | ROC ~0.94 | Proprietary compute (TPU v4 pods) |

**Read this carefully:** headline numbers above 0.94 on ClinVar either (a) use ClinVar-trained meta-predictors as features — that's `mvPPT` — or (b) spend DeepMind-scale compute on protein language models — that's `AlphaMissense`. Our 0.836 PR-AUC is lower because it is **gauged against a harder, leak-free benchmark** of our own making. Phase 2 (ESM-2 35M + hybrid fusion) is where we attempt to match the PLM class on a modest academic budget.

---

## 🧪 Why These Fixes Matter

### 1️⃣ Missense Filter — the biggest catch

```
Before filter:  pathogenic alt_aa null = 64%     benign alt_aa null = 2%
After filter:   both = 0% (by construction)
Dataset size:   283,392 → 195,098 variants
```

The old "20% missingness drop" threshold was silently discarding 11 amino-acid features. The new filter makes the task cleanly *missense-only* and **recovers those features for free**.

### 2️⃣ Paralog Leakage — the subtle one

Gene-level splits pass the basic leakage test (no gene appears in both train and test) but still leak signal through **homologous gene families**. We map every gene to a family identifier using curated HGNC-like prefix patterns (`KRT*`, `ZNF*`, `SLC##A#`, `OR##\w#`, etc.) and split at the family level instead.

```
  15,479 unique genes  →  7,851 unique families
  Families shared between train and test:  0 ✅
```

### 3️⃣ Ablation Proves It's Real Biology

| Feature group removed | Δ ROC-AUC | Δ PR-AUC | Interpretation |
|---|---:|---:|---|
| Allele frequency (AF/AC/AN/log_AF) | **−0.003** | −0.012 | Not circular ✅ |
| Amino-acid physicochemistry | −0.002 | −0.003 | Redundant with AA identity |
| **Conservation (phyloP/phastCons/GERP)** | **−0.141** | **−0.245** | **The real signal** 🎯 |

> Conservation drives the model. AF ablation cost only 0.3 pt, ruling out the most common circularity concern in ClinVar-based studies.

---

## 🏗️ Architecture

```
                  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                  │   ClinVar    │   │    gnomAD    │   │    dbNSFP    │
                  │   (labels)   │   │   (AF/AC)    │   │ (phyloP etc) │
                  └───────┬──────┘   └───────┬──────┘   └───────┬──────┘
                          │                  │                  │
                          └──────────────────┼──────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  src/data_merge.py          │
                              │  variant_key = chr:pos:ref:alt
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  src/feature_analysis.py    │
                              │  ┌─ STEP 0: missense filter │  ← leakage fix
                              │  ├─ STEP 1: drop flagged    │
                              │  ├─ STEP 2: corr > 0.95     │
                              │  └─ STEP 3: impute <20%     │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  src/data_splitting.py      │
                              │  assign_gene_family() +     │  ← leakage fix
                              │  GroupShuffleSplit(family)  │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  src/training.py            │
                              │  Optuna TPE + MedianPruner  │  ← Phase C
                              │  PR-AUC objective, 40 trials│
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  src/evaluate_baseline.py   │
                              │  • Bootstrap 1000× CIs      │
                              │  • Isotonic calibration     │
                              │  • Reliability + ECE/MCE    │
                              │  • Clinical operating points│
                              └─────────────────────────────┘
```

---

## 🚀 Quick Start

A fresh clone can train the full clean baseline in ~3 minutes:

```bash
git clone <repo-url>
cd GenticGraduationProject
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (Optuna TPE, 40 trials)
python -m src.training --trials 40 --seed 42

# Evaluate (bootstrap CIs, calibration, operating points)
python -m src.evaluate_baseline

# Ablation study
python -m src.ablation_af --trials 8

# Automated leakage gate (run before any result ships)
python -m src.verify_no_leakage
```

Outputs land in `results/checkpoints/` and `results/metrics/` — see the
[artifact manifest](results/metrics/README.md). Every change to the pipeline
is tracked in [`docs/CHANGELOG.md`](docs/CHANGELOG.md).

<details>
<summary><b>🔧 Full pipeline from raw sources (click to expand)</b></summary>

```bash
# Step 1 — Clean ClinVar labels
python -m src.clinvar_cleaning --config configs/config.yaml

# Step 2 — Extract gnomAD allele frequencies
python -m src.gnomad_extraction \
    --input data/raw/gnomad/gnomad.exomes.r2.1.1.sites.vcf.bgz \
    --clinvar-variants data/intermediate/clinvar_labeled_clean.parquet

# Step 3 — Extract dbNSFP features
python -m src.dbnsfp_extraction --config configs/config.yaml

# Step 4 — Merge
python -m src.data_merge --config configs/config.yaml

# Step 5 — Feature analysis (missense filter + correlation + impute)
python -m src.feature_analysis --config configs/config.yaml

# Step 6 — Paralog-aware family-level split
python -m src.data_splitting --config configs/config.yaml

# Step 7 — Train + evaluate
python -m src.training --trials 40
python -m src.evaluate_baseline
python -m src.ablation_af
```

</details>

---

## 📚 Data Sources

| Source | Role | Version | Genome Build |
|---|---|---|---|
| **ClinVar** | Pathogenic / Benign labels | 2026-02 | GRCh37 |
| **gnomAD** | Population allele-frequency features | r2.1.1 | GRCh37 |
| **dbNSFP** | Conservation + biochemistry features | 5.3.1a | GRCh37 |
| **UniProt** | Protein sequences *(for Phase 2 ESM-2)* | 2025_01 | — |

**Label policy:**
- `Pathogenic` / `Likely pathogenic` → **1**
- `Benign` / `Likely benign` → **0**
- Variants of Uncertain Significance (VUS) are excluded.

**Excluded predictors** (avoid ClinVar circularity):
REVEL, ClinPred, MetaLR, MetaSVM, MetaRNN, BayesDel, VEST4, M-CAP, `is_common`.

Raw files in `data/raw/` are kept out of git. Cleaned parquet snapshots under `data/intermediate/`, `data/processed/`, and `data/splits/` **are committed** so a fresh clone trains end-to-end immediately.

---

## 📁 Directory Layout

```
data/
├─ raw/              Raw upstream files (gitignored — too large)
├─ intermediate/     Cleaned per-source parquet (ClinVar / gnomAD / dbNSFP)
├─ processed/        Merged + feature-engineered datasets
└─ splits/           Paralog-aware train / val / test parquet

src/
├─ clinvar_cleaning.py        Label parsing + review-star filtering
├─ gnomad_extraction.py       AF/AC/AN extraction from VCF
├─ dbnsfp_extraction.py       Conservation + AA properties
├─ data_merge.py              variant_key unification
├─ feature_analysis.py        ⭐ Missense filter + 3-step feature pipeline
├─ data_splitting.py          ⭐ Paralog-aware family split (assign_gene_family)
├─ training.py                ⭐ Optuna TPE + MedianPruner + PR-AUC objective
├─ evaluation.py              Bootstrap CIs + reliability_curve (ECE/MCE)
├─ evaluate_baseline.py       ⭐ Pro evaluation suite
├─ ablation_af.py             ⭐ Feature-group ablation study
└─ models/
   ├─ xgboost_model.py        Optuna-tuned XGBoost
   ├─ cnn_model.py            🔜 Phase 2 (1D CNN + Attention)
   └─ esm2_model.py           🔜 Phase 2 (ESM-2 35M transfer learning)

notebooks/           Narrative analysis (EDA → results cell by cell)
results/
├─ checkpoints/      Trained model weights (.ubj)
├─ metrics/          All CSVs: bootstrap_ci, reliability_curve,
│                      operating_points, calibration_summary,
│                      ablation_af, leakage_fix_journey, …
└─ figures/          Plots (EDA, SHAP, reliability, PR curves)
```

---

## 🧠 Models

| Model | File | Status | Notes |
|---|---|---|---|
| **XGBoost baseline** | `src/models/xgboost_model.py` | ✅ **Complete** | Optuna TPE, 40 trials, PR-AUC objective |
| 1D CNN + Attention | `src/models/cnn_model.py` | 🔜 Phase 2 | Character-level protein window |
| ESM-2 (35M) transfer | `src/models/esm2_model.py` | 🔜 Phase 2 | Frozen embeddings + small head |
| Hybrid (tabular + ESM-2) | *planned* | 🔜 Phase 2 | Late fusion |

---

## 🗺️ Roadmap

- [x] **Phase 1 — Baseline & leakage hunt** · XGBoost, bootstrap CIs, calibration
- [ ] **Phase D — External validation** · ProteinGym DMS + denovo-db
- [ ] **Phase 2a** · 1D CNN + Attention on protein windows
- [ ] **Phase 2b** · ESM-2 35M frozen embeddings
- [ ] **Phase 2c** · Hybrid tabular + PLM late fusion
- [ ] **Phase 3** · Thesis write-up & slide deck

---

## 🎓 Key Design Decisions

- **Missense-only cohort.** Variants without both `ref_aa` and `alt_aa` are dropped at ingestion. This was the single biggest leakage source.
- **Family-level split.** Genes are grouped into ~7,851 HGNC-like families so paralogs stay on the same side of the split.
- **No ClinVar meta-predictors.** REVEL/ClinPred etc. are excluded; circularity isn't "reduced" — it's removed.
- **Honest reporting.** Every headline metric ships with a 1,000-replicate bootstrap 95% CI. Precision ≥ 99% is documented as *unreachable* rather than hidden.
- **Calibrated probabilities.** Isotonic regression fit on validation only; test ECE drops from 0.074 → 0.015.
- **PR-AUC-first.** Primary objective under class imbalance; ROC-AUC reported secondarily.

---

## 👥 Team

> **Genetic Graduation Project** · Computer Science · **King Khalid University**

- **Rayan AlShahrani** — Technical lead
- *Five additional collaborators — credits to be finalized in thesis.*

Supervisor: **Dr. Shanawaz Ahmed**

---

## 📄 License

Academic / research use only. Final license terms to be set by the team upon thesis submission.

---

<div align="center">

*Built with careful audits, honest numbers, and an obsession for not fooling ourselves.* 🧪

</div>
