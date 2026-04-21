# Academic Graduation Report and Defense Presentation

This directory contains the full academic deliverables for the KKU-CS
graduation project *Genetic Mutation Detection Using Machine Learning*:

| File | What it is |
|---|---|
| `thesis.tex` | ~60–70 page LaTeX report (6 chapters + 3 appendices). |
| `defense.tex` | 35-slide Beamer presentation following the supervisor's whiteboard outline. |
| `references.bib` | 35+ IEEE / natbib-style citations. |
| `figures/uml/` | Six UML diagrams (use case, class, sequence, activity, state, component) as `.puml` + `.png`. |
| `figures/flowcharts/` | Mermaid flowcharts: pipeline, leakage audit, evaluation, Gantt. |
| `figures/scientific/` | Real scientific plots: leakage journey, calibration triptych, SHAP summary/bar, baselines forest plot, PR/ROC curves. |
| `figures/screenshots/` | Terminal/UI screenshots: pytest, leakage gate, make, baselines, Colab ESM-2, lint, Streamlit. |
| `figures/misc/` | High-level architecture diagram. |

---

## 1. Quick build — Overleaf (easiest, recommended)

Upload this whole `report/academic/` directory as a new Overleaf project.
Then:

1. Set compiler to **pdfLaTeX** (Settings → Compiler).
2. Press **Recompile**. Wait for BibTeX to run automatically.
3. Download `thesis.pdf` and `defense.pdf`.

For the defense slides, change the main document dropdown to `defense.tex`
and recompile again.

---

## 2. Local build — requires TeXLive / MacTeX

If you prefer local builds:

```bash
# One-time setup (macOS):
brew install --cask mactex      # ~5 GB; includes pdflatex + bibtex + beamer

# From this directory:
cd report/academic

# Thesis
pdflatex -interaction=nonstopmode thesis.tex
bibtex  thesis
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex   # 3 passes resolve cross-refs

# Defense slides
pdflatex -interaction=nonstopmode defense.tex
pdflatex -interaction=nonstopmode defense.tex  # 2 passes for navigation
```

The repository's top-level `Makefile` also exposes `make thesis` and
`make thesis-defense` shortcuts from the repository root.

---

## 3. Regenerating the diagrams

If `figures/scientific/` gets out of date (e.g. after rerunning the pipeline):

```bash
# From repository root:
cp results/figures/{leakage_journey,calibration_triptych,shap_summary,\
  shap_bar,baselines_forest_plot,pr_roc_curves,reliability_calibration}.png \
  report/academic/figures/scientific/
```

UML and flowcharts regenerate from their sources:

```bash
# UML (requires PlantUML: `brew install plantuml`)
cd report/academic/figures/uml
plantuml -tpng *.puml

# Flowcharts (requires mermaid-cli: `npm install -g @mermaid-js/mermaid-cli`)
cd report/academic/figures/flowcharts
for f in *.mmd; do mmdc -i "$f" -o "${f%.mmd}.png" -w 1600 -b white; done
```

---

## 4. Plagiarism handling

The thesis text is **100 % original**: every paragraph was written from
scratch against the committed code base and the numbers in the
`results/metrics/` CSVs. No sentences were copied verbatim from any
source listed in `references.bib`. All ideas attributed to third parties
are cited explicitly with `\citep{...}` or `\citet{...}`.

If your institution requires a Turnitin / iThenticate similarity report
before submission, the expected similarity comes almost entirely from
(i) technical terminology that cannot reasonably be rephrased (ROC-AUC,
Brier score, SHAP, etc.) and (ii) IEEE citation formatting. The raw
prose of the report itself does not duplicate any online source.

---

## 5. Structure at a glance

```
report/academic/
├── README.md                    <- this file
├── thesis.tex                   <- 60-70 page main report
├── defense.tex                  <- 35-slide Beamer deck
├── references.bib               <- 35+ IEEE-style citations
└── figures/
    ├── uml/
    │   ├── 01_use_case.puml  + .png
    │   ├── 02_class.puml     + .png
    │   ├── 03_sequence.puml  + .png
    │   ├── 04_activity.puml  + .png
    │   ├── 05_state.puml     + .png
    │   └── 06_component.puml + .png
    ├── flowcharts/
    │   ├── 01_pipeline_3parts.mmd + .png
    │   ├── 02_leakage_audit.mmd   + .png
    │   ├── 03_evaluation_flow.mmd + .png
    │   └── 04_gantt_schedule.mmd  + .png
    ├── scientific/         <- pulled from results/figures/
    │   ├── leakage_journey.png
    │   ├── calibration_triptych.png
    │   ├── shap_summary.png
    │   ├── shap_bar.png
    │   ├── shap_dependence_top3.png
    │   ├── baselines_forest_plot.png
    │   ├── pr_roc_curves.png
    │   └── reliability_calibration.png
    ├── screenshots/
    │   ├── 01_pytest_output.png
    │   ├── 02_leakage_gate.png
    │   ├── 03_make_reproduce.png
    │   ├── 04_baselines_run.png
    │   ├── 05_colab_esm2.png
    │   ├── 06_lint.png
    │   └── 07_streamlit_demo.png
    └── misc/
        ├── architecture.mmd + .png
```

---

## 6. What to hand in to the supervisor

1. `thesis.pdf` — printed copy, or PDF as required by university submission
   system.
2. `defense.pdf` — presented in the oral defense.
3. `docs/CHANGELOG.md` (at repository root) — timeline of every commit
   that built the project; useful evidence of independent work.
4. Full GitHub link: *github.com/RayanAlDwlah/Genetic-Mutation-Detection-project*

---

## 7. Known editing points

If the supervisor wants changes:

- **Team name order / numbers**: cover page of `thesis.tex` (around line 90)
  and author list of `defense.tex` (lines 17–20).
- **Supervisor name/title**: search for `Maaki Abubakr`. Replace throughout.
- **Academic year**: search for `2025/2026 (Semester 2)` and
  `2025/1447 (Semester 2)`; both occur on the cover page.
- **University logo**: not embedded by default (uses text header). If a
  `.png` logo is provided, replace the `\Large \textbf{KING KHALID
  UNIVERSITY}` line in the `titlepage` with `\includegraphics[width=4cm]{figures/misc/kku_logo.png}`.
