# Speaker Notes — Per-Slide Pacing and Transitions

**Total budget:** 25 minutes presentation + 5-10 minutes Q&A = 30-35 minutes total.

**Slide count after consolidation:** ~30 main + ~10 backup.

**Pacing target:** 50 seconds per main slide on average. Some slides are 30 seconds (transitions, simple visuals); others are 90 seconds (calibration deep-dive, comparison table).

---

## How to use this document

1. The night before: read it through aloud once, timing yourself.
2. Morning of: review section openers and transitions only.
3. During the talk: don't try to memorize verbatim — these are *prompts*, not a script.

**Italicized lines are transitions** — practice these so you don't trip when moving between sections.

---

## PART I — The Problem (target: 5 minutes, 6 slides)

### Slide 1: Title slide [10 seconds]
> Greet the committee in Arabic if culturally appropriate, then in English: "Good morning. My name is Rayan AlShahrani, and I'll be presenting our team's work on Genetic Mutation Detection Using Machine Learning, supervised by Mr. Makki Akasha."

### Slide 2: This talk in one minute [40 seconds]
*(NEW SLIDE — see 02_new_slides.tex Section 7)*

> "Before I dive in, here's the talk in one minute. Published variant predictors report ROC-AUC above 0.92, but most don't survive external validation. We built a classifier for 195,000 missense variants, found three contamination sources in our own pipeline, removed them, and watched our PR-AUC fall from 0.955 to 0.838. Then we re-scored published baselines under the same rules and validated externally. The audit itself is the contribution."

> *"With that framing in mind, let me walk you through how we got there."*

### Slide 3: Agenda [15 seconds]
> "I'll cover the problem and related work, then walk through our pipeline — preprocessing, training and audit, evaluation. Then system design, our results, and conclusions. I'll keep it to about 25 minutes."

### Slide 4: The problem in one slide [60 seconds]
*(REPLACES slides 4 + 5 — see 02_new_slides.tex Section 1)*

> "A missense variant is a point mutation that swaps one amino acid in a protein. They're the most common coding variant in the genome — most are benign, but a clinically important minority cause disease. Distinguishing them is the central problem of clinical genetics."

> "Three failure modes plague the literature. First, definitional circularity — features derived from the label inflate metrics. Second, meta-predictor contamination — using REVEL or CADD as features when they were trained on ClinVar. Third, paralog leakage — KRT1 in train and KRT14 in test means 80% sequence identity, so the model essentially saw the test set."

> *"These three failure modes shape every methodological choice in our pipeline."*

### Slide 5: Project objectives [40 seconds]
*(MODIFIED — see 02_new_slides.tex Section 2)*

> "Seven objectives, in two groups. Data and methodology: assemble a paralog-disjoint corpus, eliminate three leakage sources, calibrate a defensible baseline, re-score published tools. Validation and delivery: external testing on denovo-db, prove ESM-2 orthogonality, publish everything reproducibly. Every objective maps to at least one CI-enforced check."

### Slide 6: Section divider — Related Works [skip if running long]
> *"How does this fit into prior art?"*

### Slide 7: Classical tools (2003-2010) [40 seconds]
> "SIFT in 2003 used conservation only — no amino-acid chemistry. PolyPhen-2 in 2010 added structural features but trained on HumDiv, which overlaps ClinVar. Both are accessible via Ensembl VEP REST. They cover 93-96% of our test variants. SIFT picks its 0.05 threshold arbitrarily — that's a recurring theme in the field."

### Slide 8: Gradient-boosted ensembles (2016-2024) [50 seconds]
> "REVEL stacks 13 base learners — but most of those base learners were ClinVar-trained, creating compound contamination. CADD uses simulated null scores but its outputs aren't calibrated probabilities. VARITY does gene-level splits, which our work shows is insufficient. MAGPIE doesn't document its paralog-handling. All sit above 0.92 ROC-AUC on their own benchmarks. None reports a like-for-like external test. None publishes bootstrap CIs."

### Slide 9: Protein language models (2021-2023) [30 seconds]
> "EVE trains a per-gene VAE — limited to ~3,000 genes. ESM-2 in 2023 is fully unsupervised, trained on 250 million protein sequences with no ClinVar exposure — provably uncontaminated. AlphaMissense uses structural PLM features but is calibrated on ClinVar, so it has the contamination we're trying to avoid."

> *"That motivates Phase 2.1 of our work — integrate ESM-2 as a training-time feature."*

### Slide 10: Section divider — Contribution [skip]

### Slide 11: What we contribute [60 seconds]
> "Five contributions. First, a transparent leakage audit with before-and-after numbers — we'll see this drop from 0.955 to 0.838 in detail. Second, like-for-like re-scoring of three published baselines on our paralog-disjoint test. Third, external validation on denovo-db with pre and post gnomAD-constraint decomposition. Fourth, a zero-shot ESM-2 proof-of-concept under rank fusion. Fifth, a fully reproducible code base — 157 tests, CI-enforced leakage gate, Docker, Streamlit demo."

> *"Everything is on GitHub at the address shown. Let me walk through the pipeline."*

---

## PART II — The Pipeline (target: 7 minutes, 8 slides)

### Slide 12: Section divider — Proposed System [skip]

### Slide 13: Pipeline three parts [40 seconds]
> "Three parts. Part 1: data — ClinVar plus gnomAD goes through missense filtering, dbNSFP merge, and a paralog-aware split. Part 2: training — leakage gate, Optuna hyperparameter search, threshold tuning, isotonic calibration. Part 3: evaluation — bootstrap CIs, SHAP, baseline re-scoring, denovo-db, ESM-2. Note the dashed lines: every stage hands off to the next via committed parquet files, which means any stage can be reproduced independently."

### Slide 14: Preprocessing 1/3 — Sources [40 seconds]
> "Four data sources. ClinVar provides labels — pathogenic and benign — filtered to review-star at least 2 for quality. gnomAD versions 2.1 and 4 provide allele frequency, count, and number. dbNSFP 5.3 provides conservation scores, amino-acid chemistry, and Pfam domain annotations. gnomAD 2.1.1 also provides gene-level constraint scores — pLI, LOEUF, mis_z. Everything joins on canonical chromosome-position-ref-alt on GRCh38."

### Slide 15: Preprocessing 2/3 — The critical filter [70 seconds]
> "This is the single biggest finding of the audit. We dropped every row with null reference or alternate amino acid — about 88,000 rows. But here's the issue: 64% of pathogenic rows had null amino-acid fields, because they weren't actually missense variants — they were nonsense, frameshift, splice variants that ClinVar had labeled pathogenic for unrelated reasons. The model was learning 'null amino acid implies pathogenic' — pure label leakage."

> "Just fixing this one filter dropped PR-AUC from 0.955 to 0.819. That's a 13.6 point drop from a single audit step. It's also a clear lesson — the most expensive contamination is the one that's invisible until you go looking."

### Slide 16: Preprocessing 3/3 — Paralog-aware split [60 seconds]
> "A naive gene-level split gives you the illusion of disjointness. We measured: 52% of prefix families — ZNF, SLC, KRT, TMEM — had members in both train and test. The model was effectively scoring memorized sequences."

> "Our fix: family-level splits. The function `assign_gene_family` uses 16 hand-curated regex patterns plus a trailing-digit fallback. 15,479 genes map to 7,851 families. Zero shared families across train, validation, and test. And it's CI-enforced — every push to main runs the check."

> *"That paralog split is what gives our test numbers their meaning. Let me show you the audit gate that enforces it."*

### Slide 17: Leakage audit 1/4 — Four CI invariants [40 seconds]
> "Four invariants. No banned features in the training matrix — that catches the `is_common` issue and the chromosome/ref/alt fields that would let the model memorize positions. 100% missense-only rows. Zero shared gene families. Pathogenic-rate gap less than 8% across splits. All four run on every push. Any failure blocks the merge to main."

### Slide 18: Leakage audit 2/4 — The journey [70 seconds]
*(This is your money slide — pace it well)*

> "This chart tells the audit story. Stage 1: pre-audit, PR-AUC 0.955 — looks fantastic. Stage 2: missense filter — drops to 0.819. That's the 13-point hit I mentioned. Stage 3: feature hygiene — removed banned features — slight further drop to 0.816. Stage 4: paralog-aware split — actually goes back up to 0.835, because honest data lets the model focus on real signal. Stage 5: Optuna tuning — 0.836."

> "ROC-AUC is more stable across the audit because it's threshold-independent. PR-AUC is more sensitive to class imbalance, which is why it dropped harder."

> "The lesson: the audit cost us 117 PR-AUC points but bought us numbers that mean something."

### Slide 19: Leakage audit 3/4 — Model + calibration [70 seconds]
> "The model is XGBoost with Optuna TPE — 40 trials, multivariate grouping, PR-AUC objective, median pruner after 200 boosting rounds. We evaluated calibration with Murphy's Brier decomposition: Brier equals reliability minus resolution plus uncertainty."

> "Reading the table: raw model has reliability 0.0054, ECE 0.054. Platt scaling brings reliability down to 0.0011 — better but assumes a sigmoid. Isotonic regression brings it to 0.00024 — a 23-fold improvement over raw — and ECE drops to 0.011. Resolution stays constant across all three, meaning calibration improves probability quality without sacrificing discrimination."

### Slide 20: Leakage audit 4/4 — Reliability triptych [40 seconds]
> "Same story visually. Raw — points consistently below the diagonal, model is overconfident. Platt — closer, with sigmoid distortion. Isotonic — points hug the diagonal, indicating calibrated probabilities."

> *"With a clean pipeline and calibrated model, here are the headline results."*

---

## PART III — Results (target: 5 minutes, 4 slides)

### Slide 21: Headline results [70 seconds]
*(MODIFIED — see 02_new_slides.tex Section 4)*

> "Test split — paralog-disjoint, 28,098 variants, ~30% pathogenic prevalence. ROC-AUC 0.938 with a tight confidence interval. PR-AUC: I report two numbers transparently. Uncalibrated 0.838, calibrated 0.827. The calibrated number is the headline because calibration is part of our deployment pipeline; the uncalibrated is shown for transparency."

> "Calibration trades 11 PR-AUC points for a 5-fold improvement in probability reliability. F1 of 0.775 at our chosen threshold. ECE of 0.011 means a predicted probability of 0.7 corresponds to ~70% empirical pathogenicity rate. All confidence intervals from 1,000-replicate nonparametric bootstrap."

### Slide 22: SHAP interpretability [60 seconds]
> "Top 5 features by mean absolute SHAP. PhyloP100-way conservation dominates — that's expected; conservation is the most informative single signal. gnomAD allele number and lof_z (loss-of-function constraint) come next. Proline at the alternate position — biochemically distinctive amino acid, often disrupts secondary structure. Pfam domain membership."

> "Two features highlighted in yellow are gene-level constraint priors we added in Phase 2.1. They're carrying orthogonal signal — meaning they're not just redundant with conservation."

> "Confident errors: 326 of 2,000 sampled test variants — 16.3%. False negatives outnumber false positives 4:1, meaning the model misses hard pathogenic cases in low-conservation regions. That's the next thing we'd improve."

### Slide 23: External validation — denovo-db [70 seconds]
*(This is the slide where the committee will probe. Be ready.)*

> "External validation on denovo-db, 642 variants. Pre-constraint — meaning without gene-level priors — ROC is 0.468, essentially chance. With constraint features it rises to 0.511. Restricting to the holdout — variants from gene families completely unseen in training — pre-constraint ROC is 0.487, post-constraint 0.573."

> "The honest reading: on truly unseen gene families, our model is roughly halfway between chance and our internal test performance. That's a major distribution shift — denovo-db is de novo variants from neurodevelopmental cohorts, very different from curated ClinVar."

> "But notice the trend: gene-level priors close half the generalization gap. The architecture is learning the right kind of signal; it just needs more capacity. That motivates the ESM-2-650M item in future work."

> *"Speaking of ESM-2 — let me show how we integrate it."*

---

## PART IV — System Analysis (target: 4 minutes, 6 slides)

> *Move quickly through this section — it's required by the committee but isn't where the strongest content lives.*

### Slide 24: Functional + non-functional requirements [40 seconds]
> "Eight functional requirements covering ingestion, scoring, attribution, batch mode, reproduction. Seven non-functional: performance under 100 milliseconds on cache hit, reproducibility to 10^-3, scalability to a million variants. Every requirement traces to at least one src module and at least one pytest test case."

### Slides 25-30: UML diagrams [3-4 minutes total — about 30 seconds each]

For UML 1/6 (Use case): "Two human actors, two system actors. Eight use cases. The CI/CD system itself is an actor — it consumes the leakage gate use case."

For UML 2/6 (Class): "Four packages — data, models, eval, interfaces. Composition relationships shown with diamond. The ESM2Scorer is composed into the LeakageGate because the gate verifies its outputs don't pollute training."

For UML 3/6 (Sequence): "A scoring request from Streamlit. Cache hit returns immediately; cache miss falls through to VEP REST for annotation, then assembles features, then predicts, then explains."

For UML 4-6 (Activity, State, Component): one sentence each — pace is more important than depth here.

> *"That covers the design. Let me show the implementation."*

---

## PART V — Implementation (target: 4 minutes, 6 slides)

### Slide 31: Hardware + stack [30 seconds]
> "Apple Silicon laptop for development, Colab T4 for ESM-2 scoring, GitHub Actions for CI. Python 3.11 with pinned dependencies. XGBoost, Optuna, SHAP, transformers, PyTorch. Docker, Makefile, pre-commit. 157 pytest tests."

### Slide 32: Testing strategy [50 seconds]
> "Five-level pyramid. 120 unit tests under 10 seconds. 8 hypothesis-based property tests covering calibrator monotonicity and rank preservation. 25 integration tests over 1-2 minutes. 5 scientific E2E scripts that reproduce the headline metrics within 10^-3. CI gate blocks merge if any of these fail. 59% overall coverage; top modules at 90% or above."

### Slide 33: CI screenshot [20 seconds — backup material if behind schedule]

### Slide 34: Leakage gate output [30 seconds]
> "The gate output. Four checks, all passing: feature hygiene, missense filter, split disjointness at gene and family level, label balance. Any red row blocks merge."

### Slide 35: ESM-2 architecture [50 seconds]
> "ESM-2 small — 35 million parameters, 12 transformer blocks, dimension 480, 20 attention heads. Zero-shot — no fine-tuning, no weight updates. We mask the position of interest and compute the log-likelihood ratio: log probability of alternate amino acid minus log probability of reference, given the masked context. Lower LLR means more pathogenic."

### Slide 36: Rank fusion [50 seconds]
> "Two scores: XGBoost predicted probability and ESM-2 LLR. Convert each to a percentile rank. Fuse linearly with weight alpha. Tuned alpha-star equals 0.175 — meaning ESM-2 contributes about 17% to the final score. Uniform alpha of 0.5 is actually worse than XGBoost alone, because the 35M model's noise dominates equal weighting."

### Slide 37: ESM-2 + rank fusion results [60 seconds]
> "Test split: XGBoost 0.938 ROC-AUC. ESM-2 alone 0.735 — meaningful signal. Uniform fusion 0.895 — worse than XGBoost. Tuned fusion 0.941 — best. External denovo-db: XGBoost 0.573, ESM-2 alone 0.552, fusion 0.588 — small gain that does transfer."

> "Two independent signals showing orthogonality: rank fusion improves over either source alone, *and* the gain transfers externally — even at 35M parameters."

### Slide 38: Training-time ablation [50 seconds]
> "Going beyond rank fusion to training-time integration. Baseline 0.938 ROC, 0.827 PR-AUC. Adding ESM-2 LLR as a feature: 0.948 ROC, 0.865 PR-AUC. Paired bootstrap: PR-AUC delta plus 0.031 with confidence interval 0.027 to 0.036, p less than 10^-4. Ablating the LLR (no_esm2 row) drops to baseline. ESM-2-only row shows the LLR alone is too weak."

### Slide 39: SHAP after ESM-2 [40 seconds]
*(MODIFIED — see 02_new_slides.tex Section 5)*

> "After integration, the imputation flag — a binary indicating whether ESM-2 produced a score — ranks first. The skipped variants live in non-canonical or truncated transcripts, which themselves correlate with pathogenicity. The LLR itself ranks fourth. Conservation features remain dominant — Spearman correlation between LLR and phyloP is at most 0.31 in absolute value, so ESM-2 partly substitutes, partly supplements."

> "Caveat: on the denovo-db holdout, training-time ESM-2 actually loses 0.13 ROC. The gain doesn't transfer at the 35M scale."

---

## PART VI — Comparison & Conclusion (target: 4 minutes, 5 slides)

### Slide 40: Comparison table [60 seconds]
*(MODIFIED — see 02_new_slides.tex Section 3, slide 38a)*

> "All baselines re-scored on our paralog-disjoint test. SIFT 0.620. PolyPhen-2 0.728. AlphaMissense 0.890 — but flagged with a star because it's ClinVar-calibrated. ESM-2 zero-shot 0.604. Our XGBoost baseline 0.827. Our XGBoost plus ESM-2 LLR 0.865 — highlighted."

### Slide 41: Why our numbers are different [60 seconds]
*(MODIFIED — see 02_new_slides.tex Section 3, slide 38b)*

> "Two columns. Left: what only we report — paralog-disjoint split with CI enforcement, three-source audit, external validation, bootstrap CIs, calibration deep-dive, Docker reproduction. Right: honest framing — we're plus 0.245 over SIFT, plus 0.137 over PolyPhen-2, minus 0.025 versus AlphaMissense. AM trains on ClinVar; we test on ClinVar-derived data; the AM gap is likely smaller under matched methodology."

### Slide 42: Critical self-assessment [50 seconds]
> "Four axes. Scientific defensibility: strong — bootstrap CIs everywhere. Methodological rigor: strong — paralog gate plus Murphy decomposition. Reproducibility: excellent — `docker run` reproduces metrics to six decimals. Deployment readiness: limited — research artifact, SFDA out of scope. The OOD ceiling at 35M ESM-2 is the binding constraint."

> "Three failures we caught and fixed during the project: random-split leakage inflated AUROC by 0.081, REVEL/CADD removed for circularity, gnomAD imputation refit train-only."

### Slide 43: Take-home messages [60 seconds]
> "Four. One: audited numbers beat inflated ones — calibrated 0.827 with no contamination versus pre-audit 0.955 with three. Two: like-for-like baselines change the conversation — under stricter methodology we beat classical tools cleanly and sit just below AlphaMissense. Three: training-time PLM integration adds significant in-distribution signal — plus 0.031 PR-AUC, p less than 10^-4. Four: external validation is the real benchmark — partial closure on denovo-db, the OOD ceiling motivates a larger checkpoint."

### Slide 44: Future work [30 seconds]
*(MODIFIED — see 02_new_slides.tex Section 6)*

> "Four directions. ESM-2-650M — 19 times more parameters, expected to lift the OOD ceiling. AlphaFold2 structural features — pLDDT, SASA, DSSP. Deep mutational scanning external validation via ProteinGym. Hybrid stacking with ACMG integration."

### Slide 45: Thank you [20 seconds]
*(MODIFIED — see 02_new_slides.tex Section 8)*

> "Thank you. Code, tests, Docker, notebooks all on GitHub. Reproduce the headline with `docker run`. Streamlit demo via the command shown. With gratitude to Mr. Makki for supervision and to my teammates Eyad, Zahran, Khalid, Saad, and Abdullah. Questions?"

---

## Pacing Discipline

If you're behind schedule at slide 20, **skip the UML detail** (slides 25-30). Show, don't dwell. Say: "I'll skip ahead through the UML diagrams to leave time for results — happy to come back if you have questions."

If you're ahead of schedule at slide 30, **expand on the audit journey** (slide 18) — that's your strongest content.

If you're catastrophically behind at slide 35, **jump straight to slide 43 (take-home)** and skip the comparison detail.

---

## Voice and delivery

- **Speak slower than feels natural.** Defense nerves accelerate speech ~30%. Match the audience's listening pace, not your speaking pace.
- **Pause after each headline number.** "PR-AUC 0.827." [pause one beat] "That's our calibrated number." A pause signals confidence.
- **Look at the committee, not the slides.** You know the slides. They want to see you.
- **Hands visible, not pocketed.** Gesture sparingly to emphasize transitions.
- **Water nearby.** Drink before slide 1 and again at slide 22 (mid-talk reset).

---

## The night before

1. Read this document once aloud, full speed.
2. Read 04_qa_prep.md once aloud.
3. Sleep 7+ hours.
4. Don't look at this document again until 2 hours before defense.

You're ready. Trust the work.
