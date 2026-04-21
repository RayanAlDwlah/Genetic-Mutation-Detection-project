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
