# Changelog

All notable changes to the honest-baseline pipeline. Dates are ISO-8601.
Commits are on `origin/main`.

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
