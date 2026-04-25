# Streamlit Demo — Ready-to-Type Variants

26 variants pre-validated against the Phase-2.1 model
(`xgboost_phase21_optuna_esm2.ubj`, calibrated PR-AUC 0.865). All
guaranteed in the dbNSFP cache (instant scoring, no VEP REST
round-trip).

* **Tiers 1–4 (20 variants)** are from `phase21/test` — the demo
  model's standard held-out test fold (random 70/15/15 split). The
  model never saw these in training.
* **Tier 5 (6 variants)** is from `strict/test` — the *paralog-aware*
  held-out fold. PTEN's paralog group was excluded entirely from
  training **and** these PTEN rows fell into `phase21/test` too, so
  they're double-held-out (paralog-disjoint + standard-held-out). Use
  Tier 5 if a methodology question asks for a paralog-aware example.

Format is `chr:pos:ref:alt`. Paste directly into the Streamlit input
box.

---

## TIER 1 — The "wow" variants (start here)

These give a clean p ≈ 1.00 and a SHAP plot dominated by ESM-2 + phyloP:

| Variant key                | Gene    | True   | p_cal | Notes                              |
|----------------------------|---------|--------|-------|------------------------------------|
| `2:166051955:G:T`          | SCN1A   | path   | 1.000 | SCN1A → Dravet syndrome (epilepsy) |
| `13:51937583:C:A`          | ATP7B   | path   | 0.975 | ATP7B → Wilson disease             |
| `10:87933160:T:G`          | PTEN    | path   | 0.951 | PTEN tumour-suppressor             |
| `3:114339360:T:G`          | ZBTB20  | path   | 0.975 | ESM-2 LLR = −13.8 (very strong)    |

Why these are good: each has `esm2_llr ≤ −12.9`, so when you click
"Score" the SHAP plot will visibly highlight `is_imputed_esm2_llr` and
`esm2_llr` in the top-5. That demonstrates Phase-2.1's contribution
visually.

---

## TIER 2 — The "benign vs pathogenic" pair (best for SHAP contrast)

Run them back-to-back to show the model isn't just predicting "1" all
the time:

| Variant key                | Gene    | True   | p_cal |
|----------------------------|---------|--------|-------|
| `2:166051955:G:T`          | SCN1A   | path   | 1.000 |
| `5:1293552:G:A`            | TERT    | benign | 0.000 |

TERT is famous in cancer biology — committee will recognise it. The
benign call here is robust (p_cal = 0.00, not 0.05).

---

## TIER 3 — The "uncertain" example (shows risk-band feature)

Demonstrates calibrated probability at the edge — useful if asked
"what does the model do when it's not sure?":

| Variant key                | Gene    | True   | p_cal | Notes                                |
|----------------------------|---------|--------|-------|--------------------------------------|
| `19:15192182:G:A`          | NOTCH3  | path   | 0.400 | NOTCH3 → CADASIL; correctly Uncertain|
| `13:51964894:C:T`          | ATP7B   | path   | 0.400 | also flagged Uncertain               |

Talking point: "Risk band 'Uncertain' triggers ACMG-style further
evidence in clinical workflow."

---

## TIER 4 — Backup pool (use any if Tier 1/2 don't ring a bell)

### High-confidence pathogenic (p ≥ 0.95)
- `5:177293844:G:A`  NSD1     1.000  (Sotos syndrome)
- `15:72727978:G:A`  BBS4     1.000  (Bardet-Biedl)
- `1:244055129:G:A`  ZBTB18   1.000
- `X:150660444:G:T`  MTM1     1.000  (myotubular myopathy)
- `3:53186014:G:A`   PRKCD    1.000
- `17:7222837:G:T`   ACADVL   1.000  (VLCAD deficiency)
- `11:126271560:G:C` FOXRED1  1.000

### High-confidence benign (p ≤ 0.03)
- `14:104703397:G:A` INF2     0.000
- `7:2525499:G:A`    LFNG     0.000
- `3:111986998:G:A`  ABHD10   0.000
- `9:5300268:T:C`    RLN2     0.000
- `2:36899540:C:T`   STRN     0.000
- `12:121221931:G:A` P2RX4    0.000
- `6:75840668:C:T`   COL12A1  0.000

---

## TIER 5 — Paralog-aware (`strict/test`) variants

These are the methodologically purest demo rows: each comes from the
paralog-aware held-out fold, where entire gene families (here PTEN's
paralog group) were excluded from training. p_cal computed with the
exact Streamlit-app pipeline (`scripts/streamlit_app.py`) on
2026-04-26.

### Pathogenic — PTEN (paralog-disjoint **and** in `phase21/test`)

| Variant key       | Gene | aa change | True       | p_cal | ClinVar          |
|-------------------|------|-----------|------------|-------|------------------|
| `10:87933127:A:G` | PTEN | H → R     | pathogenic | 0.975 | Pathogenic (3⭐)  |
| `10:87933129:T:C` | PTEN | C → R     | pathogenic | 0.961 | Pathogenic (3⭐)  |
| `10:87952135:T:A` | PTEN | S → R     | pathogenic | 0.919 | Pathogenic (3⭐)  |

### Benign — clean low-p_cal calls across the other strict/test genes

| Variant key       | Gene  | aa change | True   | p_cal | ClinVar         |
|-------------------|-------|-----------|--------|-------|-----------------|
| `5:112843456:C:G` | APC   | S → C     | benign | 0.013 | Benign (3⭐)     |
| `17:7676230:G:A`  | TP53  | P → S     | benign | 0.014 | Benign (3⭐)     |
| `22:28712124:A:C` | CHEK2 | L → R     | benign | 0.037 | Benign (2⭐)     |

> **Talking-point:** "Phase-2.1's PR-AUC of 0.827 on the strict
> paralog-aware test set is conservative — every variant in this tier
> is from a gene family that was excluded from training, so there's no
> 'sequence leakage' from a paralog the model already memorised.
> PTEN H→R (`10:87933127:A:G`) is the cleanest single example."

### Caveat — PTEN benigns the model gets wrong

If the committee asks about failure modes, these three real PTEN
benign variants from `strict/test` **are mis-classified** by Phase-2.1
— useful for an honest discussion about the strict-split AUPR drop:

| Variant key       | Gene | aa change | True   | p_cal | ClinVar         |
|-------------------|------|-----------|--------|-------|-----------------|
| `10:87864461:C:G` | PTEN | L → V     | benign | 0.883 | Benign (3⭐)     |
| `10:87931071:G:A` | PTEN | A → T     | benign | 0.339 | Likely benign (3⭐) |
| `10:87863560:T:C` | PTEN | M → T     | benign | 0.339 | Likely benign (3⭐) |

(Don't paste these unless asked — they're for the failure-mode slide,
not the live demo.)

---

## Suggested live-demo flow (90 seconds)

If a committee member says "show me the demo":

1. **First click:** `2:166051955:G:T` (SCN1A) → p ≈ 1.0
   - Point to: phyloP and ESM-2 LLR in the top-5 SHAP bars
   - Say: "Conservation + the protein-language-model agree this is
     deleterious — that's the orthogonal signal Phase-2.1 adds."

2. **Second click:** `5:1293552:G:A` (TERT) → p ≈ 0.0
   - Point to: how SHAP bars now point negative (push toward benign)
   - Say: "Same model, opposite verdict. Calibrated probability,
     not a binary threshold."

3. **Third click (if asked):** `19:15192182:G:A` (NOTCH3) → p ≈ 0.40
   - Risk band shows "Uncertain"
   - Say: "This is the band where clinical workflow says 'gather more
     evidence' — we don't force a call."

Total time on screen: under 2 minutes. Then move on.

---

## If the demo fails to launch

Backup: open `report/academic/figures/screenshots/07_streamlit_demo.png`
and walk through it as a static screenshot. Slide 31 of the deck has
the same screenshot with annotations.

The static path always works; the live path adds polish but isn't a
hard dependency.
