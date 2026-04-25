# Q&A Reference Card - Print This

## 10 Likely Questions - Answer Structure

**Q1: Why denovo-db near chance (0.487-0.573)?**
1. Distribution shift (de novo vs curated germline)
2. Gene-family disjointness enforced (strictest test)
3. 35M ESM-2 ceiling - future work
Closer: "Hardest external test by design"

**Q2: What does PR-AUC 0.865 mean clinically?**
1. F1=0.775 at threshold
2. ECE=0.011 (calibrated to ~1pp)
3. Research artifact, not clinical tool
4. ACMG framework, not standalone

**Q3: Why XGBoost not deep learning?**
1. Tabular + 195k = XGBoost zone
2. SHAP for free
3. Deep covered by EVE/AlphaMissense
4. We DO use ESM-2 LLR (+0.031, p<10^-4)

**Q4: PR-AUC dropped 0.955→0.836. Got worse?**
1. 0.955 was inflated (3 contaminations)
2. 0.836 raw / 0.827 calibrated is honest
3. Inflated numbers don't replicate
4. The audit IS the contribution

**Q5: How disjoint is paralog split?**
1. assign_gene_family - 16 regex (15479→7851)
2. verify_no_leakage - fail-closed CI gate
3. Every push runs the gate
4. Conservative bias

**Q6: Why only 35M ESM-2?**
1. Compute (Colab T4)
2. Proof-of-concept goal
3. Proved orthogonality (+0.031, p<10^-4)
4. OOD ceiling = future work #1

**Q7: Excluded REVEL/CADD but compared AlphaMissense?**
1. REVEL/CADD as features = ClinVar circularity
2. AM as competitor = no circularity
3. We flag AM's ClinVar bias openly

**Q8: 59% test coverage low?**
1. Top modules >=90%
2. 41% uncovered = data download scripts
3. Mocking APIs tests mocks not code
4. CI gate at 55% catches regressions

**Q9: Trust probability 0.71?**
1. ECE=0.011 means calibrated to ~1pp
2. 0.71 -> ~71% empirical rate
3. Risk band 'Uncertain' prompts evidence

**Q10: What would you do differently?**
1. Build leakage gate FIRST
2. Budget ESM-2-650M from day 1
3. Add ProteinGym validation earlier
Closer: "Improvements not regrets"

---

## Crisis Scripts

**Blank out:** "Let me come back to that - moving forward first."
**Don't know:** "I don't have the exact number, but I can reproduce from bootstrap parquet."
**Pressed:** "Yes, that's a known limitation. Marked as future work because [reason]."

## Day-of Checklist
- [ ] PDF on USB + Cloud + Phone
- [ ] Charger
- [ ] Print this card
- [ ] Water
- [ ] Phone silent
- [ ] Read "Talk in one minute" slide 3x before entering
