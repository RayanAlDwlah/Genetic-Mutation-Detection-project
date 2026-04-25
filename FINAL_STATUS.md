# Defense Status Report

**Date:** 2026-04-26
**Total commits:** 128 (23 added in this defense-prep session)
**Latest:** 74fe186 Defense: rebuild PDF (53p, 1MB) + add QA quick-reference card
**PDF pages:** 53 (48 main + 5 appendix backup)
**PDF size:** 1.0 MB

## DONE (No Action Needed)
- All defense-prep package items applied to main (across 23 commits)
- UML numbering fixed (all six read N/6)
- PR-AUC unified (0.836 raw / 0.827 calibrated; matches thesis exactly)
- SHAP placeholder eliminated, figure regenerated from CSV
- Comparison split into 2 slides (table + methodology framing)
- Coverage column + own-coverage caveat added to comparison table
- AlphaMissense gap narrowing (-0.063 -> -0.025) surfaced in framing
- denovo-db one-sided p=0.073 added to external-validation slide
- SHAP rank shift (lof_z 3 -> 5) made explicit
- Contributions slide expanded to 6 items (Phase-1 + Phase-2.1 split)
- Intro consolidated, headline 2-column, future work reordered
- "This talk in one minute" opener + enhanced Thank-you closer
- 9 narrative section dividers added
- 5 slides moved to backup appendix (UML 4-6, Hardware, CI screenshot)
- Calibration triptych Python source recolored (red/orange/green)
- Footer page-numbers slimmed (scriptsize, kkuSubtle gray)
- Thesis SHAP figure switched to clean Phase-2.1-only chart
- Thesis baselines listing aligned (0.838 -> 0.836)
- 3 PDF-rendered overflows fixed after first build
- Q&A quick-reference card generated for printable handout
- Defense playbook + commit-history evidence committed

## FILE LOCATIONS
- PDF:           report/academic/defense.pdf  (53 pages, 1.0 MB)
- Defense LaTeX: report/academic/defense.tex
- Thesis LaTeX:  report/academic/thesis.tex
- Thesis PDF:    report/academic/thesis.pdf
- Q&A card:      docs/defense_prep/QA_QUICK_REFERENCE.md
- Playbook:      docs/defense_prep/DEFENSE_PLAYBOOK.md
- Speaker notes: docs/defense_prep/05_speaker_notes.md
- Q&A details:   docs/defense_prep/04_qa_prep.md
- Commit log:    docs/defense_prep/rayan_commits.txt

## SANITY CHECKS (all passed)
- 53 pages PDF, no spurious 'missing' placeholders in slide content
- 8x 0.827 calibrated, 3x 0.836 raw, 1x 0.838 (denovo holdout, expected)
- All 6 UML titles read N/6
- All deck-thesis numbers cross-checked and aligned
