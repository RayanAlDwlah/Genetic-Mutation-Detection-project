<div align="center">

# 🧬 Missense Variant Pathogenicity Classification

### *Finding the honest ceiling of tabular ML for clinical genomics*

**B.Sc. Computer Science — Graduation Project · King Khalid University (2025–2026)**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![XGBoost 2.x](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![ROC-AUC 0.938](https://img.shields.io/badge/ROC--AUC-0.938-brightgreen.svg)](#-headline-results)
[![PR-AUC 0.827](https://img.shields.io/badge/PR--AUC%20(calibrated)-0.827-brightgreen.svg)](#-headline-results)
[![ECE 0.0105](https://img.shields.io/badge/ECE-0.0105-brightgreen.svg)](#-method-highlights)
[![Reproducible](https://img.shields.io/badge/reproducibility-Docker%20%2B%20locked%20deps-blue.svg)](#-reproducibility)
[![tests](https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project/actions/workflows/test.yml/badge.svg)](https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project/actions/workflows/test.yml)
[![lint](https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project/actions/workflows/lint.yml/badge.svg)](https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project/actions/workflows/lint.yml)
[![License: Academic](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#-license)

</div>

---

## 🎯 TL;DR

I built an XGBoost model that classifies human **missense variants** as *pathogenic* or *benign* using ClinVar labels, gnomAD allele frequencies, dbNSFP conservation/biochemistry features, and a zero-shot ESM-2 protein-language signal. Then **I audited my own baseline** — and discovered three leakage sources that had inflated test PR-AUC to a misleading **0.955**; the honest tuned value is **0.836**. Fixing them and calibrating outputs gave the headline result: **PR-AUC 0.827 [0.819, 0.835] (calibrated) · ECE 0.0105 · 195,098 missense variants (after the missense filter) · 7,851 paralog-disjoint gene families**. Adding ESM-2 lifts in-distribution PR-AUC further, but does not help on the external de novo set (see below).

> *The paper nobody writes: "Here's what we thought we had. Here's the bug. Here's the real number."*

---

## 📊 Headline Results

| Metric | Value | 95 % CI | Notes |
|:--|:--:|:--:|:--|
| **PR-AUC (calibrated)** — main | **0.8273** | [0.819, 0.835] | Pre-audit baseline misleadingly showed 0.955 |
| PR-AUC (raw, pre-calibration) | 0.8355 | [0.827, 0.843] | |
| ROC-AUC (calibrated) | 0.9376 | [0.935, 0.940] | |
| **ECE (calibrated)** | **0.0105** | — | Down from 0.054 raw → ~5× tighter; well below clinical target 0.02 |
| F1 (at tuned threshold) | 0.7753 | [0.768, 0.783] | |
| Phase 2.1 PR-AUC (+ESM-2 LLR, calibrated) | **0.8647** | [0.858, 0.871] | Internal test split only. Paired Δ **+0.031** [0.027, 0.036] raw PR-AUC vs. the same pipeline retrained without ESM-2 (0.871 vs. 0.840); 0 of 1,000 paired bootstrap replicates at or below zero; n = 28,098 |
| Phase 2.1 external ROC-AUC (denovo-db, n = 642) | 0.510 | — | **ESM-2 does not transfer:** vs. 0.511 without it (Δ −0.002 [−0.059, 0.064]) |
| Phase 2.1 external ROC-AUC (denovo-db, unseen families, n = 201) | 0.442 | — | Falls from 0.573 without ESM-2 (Δ −0.132 [−0.257, −0.005]) |
| External ROC-AUC (denovo-db, unseen families, n = 201), no ESM-2 | 0.573 | [0.476, 0.670] | CI crosses 0.5; honest negative result, reported, not hidden |

Internal rows: 1,000-replicate bootstrap CIs on the paralog-disjoint test split (28,098 variants). External rows: denovo-db, scored by the gnomAD-constraint model with and without ESM-2. The first five rows are the thesis headline (`report/academic/thesis.tex`, Table 5.1). The checkpoint committed in `results/checkpoints/xgboost_best.ubj` is the later gnomAD-constraint retrain; on the same test split it scores calibrated PR-AUC 0.830 [0.822, 0.838] and ROC-AUC 0.938 (`results/metrics/xgboost_bootstrap_ci.csv`). Other sources: `results/metrics/phase21/`, `phase21_ablation_paired_bootstrap.csv`, `external_denovo_db_metrics*.csv`, `denovo_paired_bootstrap_phase21.csv`.

---

## 🔍 The Leakage Audit — The Spine of This Project

The first model I trained scored a suspiciously high test PR-AUC of **0.955**. Instead of celebrating, I audited it.

<div align="center">

![Five-Stage Leakage Journey](results/figures/leakage_journey.png)

*5-stage diagnostic journey: every drop is a real bug found and fixed.*

</div>

| Stage | Fix | ROC-AUC | PR-AUC | Δ PR | What was learned |
|:--:|:--|:--:|:--:|:--:|:--|
| 0 | Random split + non-missense + circular features | 0.955 | **0.955** | — | The "too good to be true" baseline |
| 1 | **Missense-only filter** (require both ref & alt amino acid) | 0.934 | 0.819 | −0.136 | 64 % of "pathogenic" rows had no `alt_aa` — the model was learning *"null AA → pathogenic"* |
| 2 | **Feature hygiene** (drop `is_common`, drop `chr` one-hot) | 0.934 | 0.816 | −0.003 | `is_common` is 100 % benign by construction → circular; chromosome OHE leaks disease-locus identity |
| 3 | **Paralog-aware split** (`HGNC` prefix family-level `GroupShuffleSplit`) | **0.938** | **0.835** | +0.019 | Paralog families had straddled train and test; 15,479 genes grouped into 7,851 families, split as whole families |
| 4 | Optuna TPE retune (40 trials, PR-AUC objective) | 0.938 | **0.836** | +0.001 | Confirms a feature-limited ceiling — the audit was the boost, not the tuning |

**Honest ceiling:** PR-AUC 0.836 raw → **0.827 calibrated** (isotonic, validation-only).
The 0.955 → 0.836 collapse (−12 %) is *the contribution*: the inflated number came from the evaluation setup, not the model.

The **automated leakage gate** (`src/verify_no_leakage.py`) runs in CI on every push and as a pre-commit hook when split or training files change. Six checks: feature hygiene (banned features), missense-only filter, gene and family split disjointness, label balance across splits, Phase 2.1 ESM-2 features present, ESM-2 split integrity.

---

## 🧠 Method Highlights

### Calibration — proven, not assumed

<div align="center">

![Calibration Triptych](results/figures/calibration_triptych.png)

*Reliability diagrams: raw model (left) → Platt scaling (center) → isotonic, validation-only (right).*

</div>

- Isotonic regression fit **only on the validation set** — never on training, never on test.
- **Murphy's Brier-score decomposition** confirms the gain is *pure reliability improvement* (resolution stays constant): reliability dropped from 0.0054 → **0.00024** on the test split.
- ECE 0.054 → **0.0105** (≈5× tighter) on the test split.

### ESM-2 zero-shot signal — Phase 2.1

- Integrated the **35 M-parameter ESM-2 protein language model** as a single zero-shot log-likelihood-ratio feature.
- Ran ESM-2 over the 195,098-variant pool (≈ 35 h end-to-end) with a **checkpoint-every-500-variants resume-aware writer** that survived Colab disconnects mid-run. Variants ESM-2 cannot score are imputed and flagged.
- Internal test split: paired ΔPR-AUC **+0.031** (95 % CI [0.027, 0.036]; 0 of 1,000 paired bootstrap replicates at or below zero; *n* = 28,098) vs. the same pipeline retrained without ESM-2.
- **Honest negative, external set:** on `denovo-db` the gain does **not** transfer. ROC-AUC is 0.510 with ESM-2 vs. 0.511 without (n = 642, Δ −0.002, not significant), and on unseen gene families it falls from 0.573 to 0.442 (n = 201, Δ −0.132 [−0.257, −0.005]).

### Informative missingness — a finding, not a bug

SHAP ranking surfaced `is_imputed_esm2_llr` (the *flag* for "ESM-2 couldn't score this protein") as **2.5×** more important than the raw ESM-2 score itself.

Why this is biology, not leakage:
- ESM-2 systematically fails on **disordered, short, or rare proteins**.
- Those proteins are *themselves* enriched for pathogenic variation.
- The flag captures real biological signal that the raw score can't represent.

### Baselines on the same test split

<div align="center">

![Baselines Forest Plot](results/figures/baselines_forest_plot.png)

*Forest plot: this work vs. SIFT, PolyPhen-2, AlphaMissense on the identical paralog-disjoint test split (1,000-replicate bootstrap CIs).*

</div>

---

## 🔁 Reproducibility

| Layer | What's pinned / enforced |
|:--|:--|
| **Container** | `Dockerfile` (Python 3.11.7-slim-bookworm, dssp + liblzma + openblas system deps, build-time sanity import) |
| **Deps** | `requirements-lock.txt` — 287 pinned packages, frozen on macOS (does not install on Ubuntu; CI installs `requirements.txt`) |
| **Build** | `Makefile` — 19 targets (`install`, `test`, `lint`, `typecheck`, `verify-leakage`, `train`, `evaluate`, `external`, `ablate-esm2`, `reproduce-headline`, `docker-*`, `clean-artifacts`, ...) |
| **Splits** | Committed `data/splits/*.parquet` — paralog-aware family-level split, never re-randomized |
| **Checkpoints** | Committed model artifacts under `results/checkpoints/` — re-scoreable bit-for-bit |
| **Integration test** | `make reproduce-headline` re-scores the committed checkpoint on the committed test split and asserts ROC-AUC, PR-AUC, F1 and Brier (raw scores, `results/metrics/xgboost_split_metrics.csv`) within **1e-3** (< 30 s) |

```bash
make reproduce-headline   # re-scores the committed checkpoint, fails if drift > 1e-3
```

---

## ✅ Code Quality & Engineering

| Aspect | Detail |
|:--|:--|
| **Tests** | 132 test functions across 24 files; `tests/integration/test_reproduce_headline.py` is the canonical regression guard |
| **Coverage gate** | ≥ 55 % on actively-maintained modules; frozen data-prep modules deliberately excluded (bit-for-bit guarantee) |
| **Markers** | `slow`, `integration`, `needs_network`, `needs_gpu` for selective runs |
| **Lint / format** | `ruff` (E/F/W + isort + bugbear + pylint + rules), `black`; `mypy` runs in CI as a non-blocking check |
| **Pre-commit** | Auto-format + a custom `verify-no-leakage` local hook that blocks any commit touching split / training / leakage-critical files unless the gate passes |
| **CI** | `.github/workflows/test.yml` + `lint.yml` on ubuntu-22.04, Python 3.11; leakage gate + pytest; `needs_gpu` tests (they download the ESM-2 checkpoint) are deselected in CI and run locally with `pytest -m needs_gpu`; coverage XML uploaded |

---

## 🧱 Pipeline

```mermaid
flowchart LR
    A[ClinVar VCF] --> B[Missense filter<br/>+ ref/alt AA invariant]
    C[gnomAD<br/>allele freq + constraint] --> D[4-way merge]
    E[dbNSFP<br/>conservation + biochem] --> D
    F[UniProt sequences] --> G[ESM-2 35M<br/>zero-shot LLR]
    B --> D
    G --> D
    D --> H[Paralog-aware<br/>GroupShuffleSplit<br/>70/15/15]
    H --> I[Optuna TPE<br/>40 trials, PR-AUC]
    I --> J[XGBoost]
    J --> K[Isotonic calibration<br/>validation-only]
    K --> L[SHAP + external validation<br/>denovo-db family-holdout]
    L --> M[(Headline:<br/>PR-AUC 0.827)]

    style M fill:#22c55e,stroke:#16a34a,color:#fff
    style A fill:#dbeafe,stroke:#3b82f6
    style C fill:#dbeafe,stroke:#3b82f6
    style E fill:#dbeafe,stroke:#3b82f6
    style F fill:#dbeafe,stroke:#3b82f6
```

---

## 📁 Repository Structure

```text
.
├── src/                          # 29 modules
│   ├── data_splitting.py         # paralog-family-aware GroupShuffleSplit
│   ├── feature_analysis.py       # missense filter + correlation + imputation
│   ├── models/xgboost_model.py   # training + Optuna TPE
│   ├── calibration_deep.py       # isotonic + Murphy Brier decomposition
│   ├── esm2_scorer.py            # checkpoint-resumable ESM-2 LLR writer
│   ├── verify_no_leakage.py      # 5-check automated audit (CI-gated)
│   └── ...                       # baselines, denovo loader, paired statistical tests
├── tests/                        # 132 tests · 24 files (unit/ + integration/)
├── scripts/                      # 29 scripts: analysis & re-run drivers, Streamlit demo
├── configs/                      # config.yaml
├── notebooks/                    # 12 EDA + analysis notebooks (+ legacy/)
├── results/
│   ├── figures/                  # 39 PNG figures
│   ├── metrics/                  # leakage_fix_journey.csv + bootstrap CIs
│   └── checkpoints/              # versioned model artifacts
├── report/                       # academic/thesis.tex (111 pp) + academic/defense.tex (40 pp)
├── docs/                         # CHANGELOG
├── data/                         # raw + intermediate + committed splits
├── Dockerfile · Makefile · pyproject.toml · requirements-lock.txt
└── .pre-commit-config.yaml · .github/workflows/*.yml
```

---

## 🚀 Quick Start

### Run locally (CPU, ~15 min for evaluation)

```bash
git clone https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project.git
cd Genetic-Mutation-Detection-project

# Install. On macOS: make install (pinned requirements-lock.txt).
# On Linux: pip install -r requirements.txt
make install

# Verify the leakage audit passes
make verify-leakage

# Reproduce the headline number from the committed checkpoint
make reproduce-headline
```

### Run in Docker (full bit-for-bit reproducibility)

```bash
make docker-build      # ~5 min, single layer-cached build
make docker-reproduce  # re-scores the committed checkpoint inside the container
```

### Full retrain (≈ 6 h CPU / + 35 h on Apple MPS for ESM-2 scoring)

```bash
make train             # XGBoost + Optuna TPE (40 trials, PR-AUC objective)
make evaluate          # bootstrap CIs + calibration + SHAP
make external          # denovo-db family-holdout validation
make ablate-esm2       # paired bootstrap for Phase 2.1 contribution
```

---

## 📈 Inputs

| Source | Used for | Records (post-filter) |
|:--|:--|:--:|
| **ClinVar** | Gold-standard pathogenic/benign labels | 195,098 missense variants |
| **gnomAD v4** | Allele frequencies + gene-level constraint (`lof_z`, `mis_z`, `oe_lof`) | 15,479 genes |
| **dbNSFP 4.x** | Per-variant conservation (`phyloP100way`, `GERP++`) + biochemistry | full join |
| **UniProt** | Reference protein sequences (for ESM-2 scoring) | — |

Final dataset → 7,851 paralog-disjoint gene families → 70 / 15 / 15 split (train / val / test).

---

## 📝 Citation

```bibtex
@misc{aldwlah2026missense,
  title  = {Missense Variant Pathogenicity Classification:
            Finding the Honest Ceiling of Tabular ML for Clinical Genomics},
  author = {Al-Dwlah, Rayan Saleh},
  year   = {2026},
  school = {King Khalid University, College of Computer Science},
  note   = {B.Sc. Computer Science Graduation Project; supervisor:
            Mr. Makki Akasha Babiker Al-Bashir}
}
```

---

## 👤 Author

**Rayan Saleh Al-Dwlah**
B.Sc. Computer Science (AI / Machine Learning) — King Khalid University, Abha — Class of 2026
📧 rayanaldwlah@gmail.com · 🔗 [LinkedIn](https://www.linkedin.com/in/rayanaldwlah/) · 🐙 [GitHub](https://github.com/RayanAlDwlah)

**Supervisor:** Mr. Makki Akasha Babiker Al-Bashir — King Khalid University, College of Computer Science.

---

## 🏷️ License

Released for **academic and research use**. Industrial / clinical deployment would require SFDA (Saudi Food and Drug Authority) review — see `report/thesis.tex §6.3` for the deployment-readiness discussion.

---

<div align="center">
<sub><i>Built with discipline, audited with honesty, calibrated with proof.</i></sub>
</div>
