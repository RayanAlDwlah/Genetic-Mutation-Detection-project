# Q&A Preparation — Defense Committee Likely Questions

15 questions ranked by probability the committee will ask. Memorize the **structure** of each answer (the bullet points), not the exact wording. Speak naturally; the structure keeps you from rambling.

**Time budget:** Defense Q&A is typically 5-10 minutes. Most questions deserve a 30-60 second answer. If a committee member follows up, that's a good sign — engage, don't retreat.

**Default opening for any question:** "That's a good question." (Buys 2 seconds. Don't overuse — once or twice is fine.) Then go straight to the answer.

**Default closing for any question you don't fully know:** "I don't have that exact number / I haven't tested that — but here's how I'd approach it: ..." Honesty is graded higher than confident wrong answers.

---

## Q1 — *Almost certain to be asked*

> **"Why is denovo-db performance close to chance? Doesn't that invalidate the whole project?"**

**Answer structure:**
1. **Distribution shift, not failure.** denovo-db is de novo variants from autism/developmental cohorts; ClinVar is curated germline across many disease categories. Different populations.
2. **By design.** Gene-family disjointness was *enforced* — the model has never seen any of these gene families in training. This is the strictest external validation in the field.
3. **The trend is informative.** Pre-constraint ROC = 0.487 (chance). Post-constraint = 0.573. Gene-level priors close half the generalization gap. The architecture is learning the right kind of signal; it just needs more capacity.
4. **35M parameter ceiling.** ESM-2-35M has a documented OOD ceiling. Future work item #1: upgrade to ESM-2-650M ($19{\times}$ parameters).

**Closer:** "We deliberately chose the hardest possible external test rather than the easiest. The number is a feature of our methodology, not a bug in our model."

---

## Q2 — *Very likely*

> **"What does PR-AUC of 0.865 actually mean clinically?"**

**Answer structure:**
1. **Threshold behavior:** at our chosen operating point, F1 = 0.775. For every 100 variants flagged as pathogenic, ~78 are true pathogenic.
2. **Calibration matters more than rank.** Post-isotonic ECE = 0.011, meaning a predicted probability of 0.7 corresponds to ~70% empirical pathogenicity rate.
3. **Framing:** This is a research artifact, not a clinical tool. We mark "Deployment readiness: Limited" in the self-assessment because clinical use requires SFDA certification, which is out of scope.
4. **Intended use:** As an auxiliary signal in the ACMG framework — one of multiple lines of evidence, not a standalone diagnosis.

---

## Q3 — *Very likely from the technical examiner*

> **"Why XGBoost and not a deep neural network?"**

**Answer structure:**
1. **Data shape fits XGBoost.** 195k samples, mixed numeric and categorical features, tabular structure. XGBoost is state-of-the-art for this regime.
2. **Interpretability for free.** SHAP values are essential for clinical credibility. Deep models require post-hoc methods that are less reliable.
3. **Deep methods are already covered.** EVE, AlphaMissense, ESM-2 — the PLM literature handles the deep approach. Our value-add is methodological rigor on a defensible classical baseline, not architectural novelty.
4. **Empirical justification:** the ESM-2 augmentation (slide 44) shows we already integrate deep signal where it adds value (+0.031 PR-AUC, $p < 10^{-4}$).

---

## Q4 — *Very likely*

> **"You report a PR-AUC drop from 0.955 to 0.838 after the audit. Doesn't that mean your model got worse?"**

**Answer structure:**
1. **The 0.955 was inflated.** It reflected three contamination sources: null-AA rows that correlated with the label, paralog overlap between train and test, and label-derived features.
2. **The 0.838 is honest.** Same model architecture, same training procedure — just on uncontaminated data.
3. **Inflated numbers don't replicate.** External validation on denovo-db confirms this: published methods don't survive the same test.
4. **The audit IS the contribution.** We're not reporting that our model improved; we're reporting that the field's evaluation methodology is broken and showing what an audited number looks like.

---

## Q5 — *Likely from the methodology examiner*

> **"How do you know your paralog split is actually disjoint?"**

**Answer structure:**
1. **Two-layer enforcement.**
   - Layer 1: `assign_gene_family()` — 16 hand-curated regex patterns + trailing-digit fallback. Maps 15,479 genes to 7,851 families.
   - Layer 2: `verify_no_leakage.py` — fail-closed CI gate. Asserts zero family overlap between train/val/test.
2. **CI-enforced.** Every push to main runs the gate. If a gene like KRT14 ended up in test while KRT1 is in train, the build goes red and blocks merge.
3. **Auditable.** The leakage gate output is in slide 40; we can show it live.
4. **Conservative bias.** When the regex is uncertain, it groups genes more aggressively, which makes our split *stricter* (more disjoint) than a perfect oracle would produce.

---

## Q6 — *Likely from the technical examiner*

> **"Why only 35M ESM-2? Why not 650M or 3B?"**

**Answer structure:**
1. **Compute constraint.** Available hardware: Apple Silicon laptop (MPS) and Colab T4 GPU. Estimated ~40 GPU-hours for ESM-2-650M on 195k variants — beyond our budget.
2. **Proof-of-concept goal.** The aim was to establish *whether* PLM signal is orthogonal to existing features, not to maximize its contribution. We proved orthogonality with $+0.031$ PR-AUC paired bootstrap, $p < 10^{-4}$.
3. **Documented as the binding constraint.** The OOD ceiling ($\Delta$ROC = $-0.13$ on unseen families) directly motivates the upgrade path in future work.
4. **Not a missed opportunity.** Without the 35M baseline, we wouldn't know that the gain *doesn't transfer* — which is itself a publishable finding.

---

## Q7 — *Likely*

> **"You excluded REVEL and CADD as features but compare against AlphaMissense. Isn't that inconsistent?"**

**Answer structure:**
1. **Different roles.** REVEL/CADD as *features* would re-introduce ClinVar contamination — both are ClinVar-trained classifiers. Including them would defeat the audit.
2. **AM as a *competitor* doesn't have this problem** — we don't use AM as input, we compare its scores to ours on the same test split.
3. **We flag AM's bias openly.** The footnote on the comparison slide notes AM is ClinVar-calibrated, so its numbers on a ClinVar-derived test set are likely optimistic.
4. **Methodological symmetry would require rerunning AM on a fully unseen test** — which AlphaFold/DeepMind have not released the infrastructure to do reproducibly.

---

## Q8 — *Possible from the software examiner*

> **"59% test coverage seems low. Justify."**

**Answer structure:**
1. **Distribution matters more than headline number.** Top modules (model trainer, calibrator, leakage gate, metric computation) all sit at $\geq 90\%$ coverage. The 41% uncovered code is dominated by data-download scripts and one-off analysis notebooks.
2. **Testing those wouldn't add value.** Mocking external APIs (ClinVar FTP, gnomAD) would test the mocks, not the code. Live testing would be slow and brittle.
3. **CI gate at 55%** is set to catch regressions on the modules that *do* have tests. Coverage going down means a tested module became less tested.
4. **Property-based tests, not just unit tests.** 8 hypothesis-based properties test calibrator monotonicity, rank preservation, etc. These cover infinite input spaces, not just the 59%.

---

## Q9 — *Likely from the clinical/biology examiner*

> **"A clinical geneticist sees probability 0.71. Can they trust it?"**

**Answer structure:**
1. **Within calibration limits, yes.** ECE = 0.011 means predicted probabilities match observed frequencies to within ~1 percentage point on test.
2. **0.71 means roughly 71% empirical pathogenicity rate** for variants in that score range. Murphy decomposition confirms this (reliability dropped 23× post-calibration).
3. **The risk band ('Uncertain' between 0.3 and 0.8) prompts further evidence.** The Streamlit demo shows this — slide 35.
4. **ACMG context.** Our score is one line of evidence under the ACMG framework. Geneticists don't (and shouldn't) decide pathogenicity from a single number. The probability is calibrated; the decision is not.

---

## Q10 — *Sometimes asked as a closer*

> **"What would you do differently if you started over?"**

**Answer structure:**
1. **Build the leakage gate first, then the model.** We started by training and got 0.955 PR-AUC, then spent half the project deconstructing why that number was wrong. Inverting the order saves time.
2. **Budget for ESM-2-650M from day one.** The OOD ceiling at 35M is the largest scientific limitation; we knew it would matter but didn't have the compute provisioned.
3. **Add a third external dataset earlier.** denovo-db alone is one slice; ProteinGym (deep mutational scanning) would give independent validation across the whole project, not just at the end.
4. **Closer note:** "These are improvements, not regrets. The audit-first lessons we learned are the contribution; we wouldn't have learned them without the iteration."

---

## Q11 — *Possible*

> **"Your training-time ESM-2 ablation shows +0.031 PR-AUC. Is that practically significant or just statistically?"**

**Answer structure:**
1. **Statistical significance is the floor, not the ceiling.** Paired bootstrap $p < 10^{-4}$ rules out chance. But practical significance needs a different lens.
2. **Practical: depends on use case.** For a triage tool that filters 20,000 variants down to a manageable shortlist, +0.031 PR-AUC at the high-precision end means meaningfully fewer false positives in the actionable bucket.
3. **The OOD result is the cautionary half.** $\Delta$ROC = $-0.13$ on family-holdout means the gain doesn't transfer. So in-distribution: practically meaningful. Out-of-distribution: not yet.
4. **Honest framing:** the 35M ESM-2 isn't ready for clinical adoption; it's an existence proof that PLM signal can be integrated cleanly.

---

## Q12 — *Possible from the methodology examiner*

> **"Why isotonic calibration and not Platt scaling?"**

**Answer structure:**
1. **Platt assumes a sigmoid;** isotonic is non-parametric and only assumes monotonicity. For a tree ensemble like XGBoost, the calibration curve is rarely a clean sigmoid.
2. **Empirical comparison on slide 20:**
   - Raw ECE: 0.054
   - Platt ECE: 0.029
   - Isotonic ECE: **0.011**
   Isotonic wins by 2.6× over Platt.
3. **Resolution unchanged.** Murphy decomposition shows isotonic improves reliability by 23× while resolution (discrimination) is constant. Calibration is doing exactly what it should.
4. **Standard practice.** Isotonic is the field-standard for tree-based models in calibration literature (Niculescu-Mizil & Caruana, 2005).

---

## Q13 — *Possible*

> **"Your group has six members. What was your specific contribution?"**

**Answer structure:**
1. **Role: Core Model Engineer.** Owned model architecture, training pipeline, calibration, leakage audit, ESM-2 integration, evaluation framework.
2. **Concretely built:** the XGBoost trainer (`src/models/`), the leakage gate (`src/verify_no_leakage.py`), the bootstrap CI computation (`src/eval/bootstrap.py`), the ESM-2 LLR scorer (`src/models/esm2_scorer.py`), and the full ablation framework.
3. **Team contributions:** [name the 1-2 specific contributions of others — e.g., data ingestion, Streamlit UI, LaTeX report, testing infrastructure].
4. **Cross-cutting:** the 157 pytest tests and the CI/CD pipeline were a joint effort across the team.

> **Note:** Have specific commit history ready (`git log --author="Rayan"`) if pressed for evidence. Don't overclaim; do claim what's yours.

---

## Q14 — *Less likely but disarming if asked*

> **"What's the weakest part of your project?"**

**Answer structure:**
1. **External generalization is the binding constraint.** denovo-db ROC = 0.573 even with gene-level priors. Our model is more honest than the literature, but it doesn't yet match what a clinical-grade tool would need.
2. **35M ESM-2 doesn't transfer OOD.** Future work item #1 addresses this; we documented the ceiling but didn't break it.
3. **No deep mutational scanning external validation.** Third-source validation would strengthen the external claims.
4. **Single-supervisor project.** Larger projects typically have an iterative review cycle with multiple senior reviewers; we relied heavily on automated CI gates as a substitute.

> **Why answer this honestly:** Examiners respect candidates who name their own weaknesses before being asked. It signals scientific maturity.

---

## Q15 — *Closing question, sometimes asked*

> **"What's the one thing you want us to remember from this work?"**

**Answer structure (rehearse this exactly):**
1. **The headline:** "Audited numbers are smaller than inflated ones, and they replicate."
2. **The mechanism:** "We removed three specific contamination sources from a standard ML pipeline and watched PR-AUC fall from 0.955 to 0.838."
3. **The implication:** "Most published variant predictors report numbers above 0.92 ROC-AUC. Our work shows what an honest evaluation methodology looks like for one specific problem."
4. **The artifact:** "Everything is reproducible — `docker run` will reproduce every number in this talk to six decimal places."

---

## Pre-Defense Self-Quiz

The night before, ask yourself each question aloud and time your answer. Target: 30-60 seconds.

If your answer goes over 90 seconds, you're rambling — cut to bullet 1 and bullet 4, skip the middle.
If your answer is under 20 seconds, you're under-explaining — add the *why* to each *what*.

## What to do if you genuinely don't know

Honest scripts to memorize:

- **"I don't have that number with me, but I can reproduce it from the bootstrap parquet — would you like me to send a follow-up?"**
- **"That's a question we considered but didn't fully resolve. Here's our partial thinking: ... If I had more time, I'd ..."**
- **"I don't know. My intuition is X, but I haven't tested it."**

The examiner is testing whether you understand the boundary of your own knowledge. "I don't know" with a thoughtful follow-up is graded *higher* than a confident wrong answer.

---

## What you should NEVER do

1. **Don't say "obviously"** — if it were obvious to the committee, they wouldn't ask.
2. **Don't argue with a committee member's framing** — restate it in your own words ("If I understand correctly, you're asking ...") and then answer.
3. **Don't apologize for limitations** — *describe* them. "This is a limitation" is different from "I'm sorry it didn't work better." The first is scientific, the second is undergraduate.
4. **Don't go off-script into territory you didn't prepare** — a clean "good question, I don't have a complete answer" is better than improvising into a wrong claim.

---

## Final note

You have done graduate-level work. Walk in knowing that. The committee will probe — that's their job — but they're not trying to trip you up. They want to see that you understand your own contribution well enough to defend it under pressure.

You do. Now prove it.
