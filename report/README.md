# Technical report (`main.tex`)

A 10-section LaTeX report that turns the project narrative into a
conference-submission-ready write-up.

## Sections

1. **Abstract** — headline result + methodology summary
2. **Introduction** — the missense classification problem, why
   honest evaluation matters, three kinds of leakage to watch for
3. **Related Work** — VARITY / MVP / MAGPIE / EVE / ESM-1b /
   AlphaMissense / CADD / REVEL comparison table
4. **Methods** — data assembly, paralog-aware splitting, model +
   tuning, calibration, evaluation rigor, interpretability,
   external validation, reproducibility
5. **Results** — leakage audit, headline performance, baseline
   comparison, calibration, SHAP + confident errors, denovo-db,
   ESM-2 zero-shot PoC
6. **Discussion** — honest numbers beat inflated ones; like-for-like
   baselines change the conversation; external generalisation is
   the real benchmark
7. **Limitations + Future Work**
8. **Reproducibility** — how to rebuild all figures from scratch
9. **Conclusion**
10. **References** — 22 BibTeX entries in `references.bib`

## How to compile

### Option A — Overleaf (easiest)

1. Upload `main.tex`, `references.bib`, and the `figures/` directory
   to a fresh Overleaf project.
2. Press the green **Recompile** button.
3. Download the PDF.

This is the recommended path for graduation work — it also lets you
edit the text collaboratively with your supervisor.

### Option B — Local TeXLive

```bash
# macOS (one-time, ~5 GB):
brew install --cask mactex

# Or a lighter alternative:
brew install tectonic

# Then, from the repo root:
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex  # third pass to resolve refs
```

## Editing tips

- The tables pull numbers directly from the committed CSVs in
  `../results/metrics/`; update those (by re-running the
  pipeline) before re-rendering the report so everything stays
  in sync.
- Figures in `figures/` are copies of the ones in
  `../results/figures/` — regenerate via:

  ```bash
  cp ../results/figures/{leakage_journey,calibration_triptych,\
shap_summary,shap_bar,baselines_forest_plot}.png figures/
  ```

- The bibliography style is `plainnat` (natbib). If your institution
  requires IEEE or Nature style, swap `\bibliographystyle{plainnat}`
  at the bottom of `main.tex`.

## Defense presentation (`defense.tex`)

A 20-slide Beamer deck structured as the oral defense of the report.
Uses the `metropolis` theme (clean, modern, widely accepted).

To compile:

```bash
cd report
pdflatex defense.tex
pdflatex defense.tex
```

Or upload to Overleaf alongside `figures/` for editing.

Slides cover:

1. Title
2. One-sentence pitch
3. What is a missense variant (the problem)
4. Why it's hard (three failure modes in prior work)
5. We audited our own baseline
6–8. Three leakage sources, one slide each
9. Five-stage journey (the headline figure)
10. Post-audit numbers
11. Baseline comparison table
12. Forest plot
13. Why calibration matters + Murphy decomposition
14. Calibration results table
15. Reliability triptych
16. SHAP top-10 (highlighting Phase 2.1 additions)
17. SHAP beeswarm
18. Confident errors
19. denovo-db external validation — the sobering result
20. Before vs after gnomAD constraint on denovo-db
21. Phase 2 roadmap
22. Reproducibility
23. Take-home messages
24. Thank you / Q&A

## Target venues

This report is structured like a methods paper suitable for
submission to any of:

- *Bioinformatics* (Oxford University Press)
- *Genome Medicine* (BMC)
- *PLOS Computational Biology*
- *Nature Communications* (harder bar, but the honesty angle fits
  their editorial line)

For a conference-paper version, trim Methods §3.1–§3.3 to two
paragraphs and cite our repository for full detail.
