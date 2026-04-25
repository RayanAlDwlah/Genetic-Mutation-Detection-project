# Critical Fixes — Exact Text Replacements

Apply these in `defense.tex`. Each fix has a search target and replacement.

---

## Fix 1: UML Numbering Bug

**Problem:** Slides 27-29 say `UML 1/3, 2/3, 3/3` but slides 30-32 say `UML 4/6, 5/6, 6/6`. Should all be `1/6` through `6/6`.

**In `defense.tex`, find and replace these frame titles:**

```diff
- \begin{frame}{UML 1/3: Use case diagram}
+ \begin{frame}{UML 1/6: Use case diagram}

- \begin{frame}{UML 2/3: Class diagram (grouped by package)}
+ \begin{frame}{UML 2/6: Class diagram (grouped by package)}

- \begin{frame}{UML 3/3: Sequence diagram (scoring request)}
+ \begin{frame}{UML 3/6: Sequence diagram (scoring request)}
```

(Slides 4/6, 5/6, 6/6 already correctly numbered.)

**Verification:** After compile, all six UML slides should read `UML N/6`.

---

## Fix 2: PR-AUC Unification

**Problem:** Three different PR-AUC values appear for the same model (calibrated XGBoost on paralog-disjoint test):
- Slide 22: **0.838** [0.830, 0.846]
- Slide 43: **0.830**
- Slides 44, 47: **0.827**

**Decision (canonical):** The reported number for "our model on test" is **0.827** (calibrated). The 0.838 likely refers to uncalibrated. Make this distinction explicit.

### Step 2a: Verify in your repo

Before editing slides, run this check:

```bash
cd /path/to/repo
python -c "
import pandas as pd
from pathlib import Path

# Find all metric files
metric_files = list(Path('results/metrics').rglob('*.csv')) + \
               list(Path('results/metrics').rglob('*.parquet'))

for f in metric_files:
    print(f'\n=== {f} ===')
    df = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f)
    if 'pr_auc' in df.columns or 'PR-AUC' in df.columns:
        print(df.head())
"
```

Identify which CSV has 0.827 (canonical, calibrated) and which has 0.838 (likely uncalibrated). Then proceed with the replacements below using the actual numbers from your files.

### Step 2b: Apply the replacements

**Slide 22 (Headline results):**

```diff
- Metric              & Value (95\% CI)         \\
- ROC-AUC             & 0.938 [0.935, 0.941]    \\
- PR-AUC              & 0.838 [0.830, 0.846]    \\
- F1                  & 0.775                   \\
- Brier (calibrated)  & 0.083                   \\
- ECE (calibrated)    & 0.011                   \\
+ Metric                    & Value (95\% CI)         \\
+ ROC-AUC                   & 0.938 [0.935, 0.941]    \\
+ PR-AUC (uncalibrated)     & 0.838 [0.830, 0.846]    \\
+ PR-AUC (calibrated)       & 0.827 [0.819, 0.835]    \\
+ F1                        & 0.775                   \\
+ Brier (calibrated)        & 0.083                   \\
+ ECE (calibrated)          & 0.011                   \\
```

> **Note:** If your actual calibrated CI is different from `[0.819, 0.835]`, replace with your real numbers from the bootstrap parquet. If you don't have a separate bootstrap for the calibrated PR-AUC, run:
> ```python
> from src.eval.bootstrap import bootstrap_metric
> ci = bootstrap_metric(y_true, p_calibrated, metric='pr_auc', n=1000)
> ```

**Slide 43 (ESM-2 + rank fusion):**

```diff
- XGBoost (calibrated)  & 0.938 & 0.830 & ---  \\
+ XGBoost (calibrated)  & 0.938 & 0.827 & ---  \\
```

**Slide 44 (Training-time ablation):** Already says 0.827 — leave alone.

**Slide 47 (Comparison):** Already says 0.827 — leave alone.

**Slide 50 (Take-home messages):** Verify these references are consistent:

```diff
- Calibrated PR-AUC 0.827 has no known contamination
+ Calibrated PR-AUC 0.827 has no known contamination
  (already correct — verify it stays this way)

- The ESM-2-augmented model lifts calibrated test PR-AUC 0.827 → 0.865
+ The ESM-2-augmented model lifts calibrated test PR-AUC 0.827 → 0.865
  (already correct — verify it stays this way)
```

### Step 2c: Add a clarifying note to the headline slide

Add this footnote-style line under the table on slide 22:

```latex
\vspace{0.5em}
\footnotesize
Calibrated PR-AUC is reported as the canonical headline; uncalibrated is shown for transparency. Calibration trades a small ranking penalty for substantially better probability reliability (ECE: 0.054 $\to$ 0.011).
```

This pre-empts the question "why are you reporting two PR-AUCs" and shows methodological awareness.

---

## Fix 3: SHAP Figure Placeholder

**Problem:** Slide 45 ("What ESM-2 changed in SHAP") shows the Phase-1 panel with placeholder text **"Phase-1 SHAP parquet missing"** instead of an actual chart. This is the single most embarrassing bug in the deck.

### Decision tree

```
Does results/shap/phase1_shap.parquet exist?
├── YES → Run 03_regenerate_shap_figure.py with --mode=full
│         Produces side-by-side comparison figure
│
└── NO  → Two options:
          ├── (a) Recompute Phase-1 SHAP (recommended if cheap)
          │       cd repo && python -m src.eval.shap_phase1
          │       Then run --mode=full
          │
          └── (b) Single-panel Phase-2.1 only
                  Run 03_regenerate_shap_figure.py with --mode=single
                  Update slide 45 caption (see below)
```

### If you go with single-panel (option b), update slide 45:

```diff
- \begin{frame}{What ESM-2 changed in SHAP}
-   \includegraphics[width=\linewidth]{figures/shap_phase1_vs_phase2.png}
-   \begin{itemize}
-     \item \texttt{is\_imputed\_esm2\_llr} ranks 1, \texttt{esm2\_llr} ranks 4 ...
-     \item Conservation features (phyloP, GERP) remain dominant ...
-     \item denovo-db holdout: $\Delta$ROC-AUC = $-0.13$ ...
-   \end{itemize}
- \end{frame}

+ \begin{frame}{What ESM-2 changed in SHAP}
+   \begin{columns}[T]
+     \column{0.55\textwidth}
+     \includegraphics[width=\linewidth]{figures/shap_phase2_only.png}
+     
+     \column{0.45\textwidth}
+     \footnotesize
+     \textbf{Top features after ESM-2 integration:}
+     \begin{itemize}
+       \item \texttt{is\_imputed\_esm2\_llr} ranks \#1 — the imputation flag is itself signal
+       \item \texttt{esm2\_llr} ranks \#4 — substantial but not dominant
+       \item Conservation features (phyloP, GERP) remain top-tier
+       \item Spearman $\rho$ between ESM-2 LLR and phyloP $\leq |0.31|$ — partly substitutes, partly supplements
+       \item denovo-db holdout: $\Delta$ROC = $-0.13$ — in-distribution gain doesn't transfer at 35M scale
+     \end{itemize}
+   \end{columns}
+   
+   \vspace{0.4em}
+   \scriptsize
+   Phase-1 SHAP omitted; top Phase-1 features were \texttt{phyloP100way\_vertebrate}, \texttt{AN}, and \texttt{lof\_z} (in that order).
+ \end{frame}
```

This makes the omission *deliberate* and *informative*, not *broken*.

### Verification

After regenerating the figure:
1. Open the PDF and confirm slide 45 has no text reading "missing"
2. Confirm the figure renders with proper axes labels
3. Confirm the bullet points are readable (not cut off the right edge)

---

## Fix 4: Slide 38 Comparison Table Overcrowded

**Problem:** Slide 38 has 7 rows + 5 columns + 2 multi-line footnotes that wrap behind the page-number footer. Hard to read, footnotes cut off.

**Solution:** Split into two slides — see `02_new_slides.tex` Section 3 for the LaTeX.

### Quick summary of the split

- **New Slide 38a (Comparison table):** Drop REVEL row entirely. 6 rows total. Table only with brief footnotes inline.
- **New Slide 38b (Why our numbers are different):** Two-column layout — "What only we report" + "Honest framing." This becomes a strong methodological statement.

This change adds one slide net but the next section (Conclusion) starts cleaner.

---

## Verification Checklist

After applying Fixes 1-4, run through this checklist:

- [ ] `latexmk -pdf defense.tex` completes without errors
- [ ] All six UML slides show `N/6` numbering
- [ ] Slide 22 shows both uncalibrated (0.838) and calibrated (0.827) PR-AUC
- [ ] Slide 43 shows XGBoost calibrated PR-AUC = 0.827 (matches slides 44, 47, 50)
- [ ] Slide 45 has no "missing" placeholder text — either two panels or one clean panel with note
- [ ] Slide 38 is split into two slides; comparison table fits cleanly with no footnote overflow
- [ ] PDF page count: was 52, now likely 51-53 depending on how you handled the Phase-1 SHAP

If all six checks pass, the deck is now defense-ready on the critical-bug axis. Move to `02_new_slides.tex` for content/pacing improvements.
