# Defense Improvement Package — Apply with Claude Code

This package contains every change needed to take the defense from "good" to "perfect." Apply in this order. Total estimated time: **3-4 focused hours.**

## What's in this package

| File | Purpose | When to use |
|------|---------|-------------|
| `00_README.md` | This file — apply order and decisions | Read first |
| `01_critical_fixes.md` | Numeric inconsistencies + UML bug + SHAP placeholder | Step 1 |
| `02_new_slides.tex` | LaTeX source for rewritten/consolidated slides | Step 2 |
| `03_regenerate_shap_figure.py` | Python script for the missing Phase-1 SHAP panel | Step 1c |
| `04_qa_prep.md` | 15 likely committee questions with answer scripts | Memorize before defense |
| `05_speaker_notes.md` | Per-slide pacing, transitions, what to say | Practice with this open |
| `06_polish_pass.md` | Small text/figure tweaks across the deck | Last pass |

## Apply order (recommended)

```
Step 1: Critical fixes (90 min)
  1a. UML numbering bug          [5 min, see 01]
  1b. PR-AUC unification          [30 min, see 01]
  1c. SHAP figure regeneration    [30 min, see 03]
  1d. Slide 38 split              [25 min, see 02]

Step 2: Slide consolidation (60 min)
  2a. Replace intro slides        [20 min, see 02]
  2b. Move slides to backup       [20 min, see 02]
  2c. Verify slide count ≤ 30     [5 min]
  2d. Test build                  [15 min]

Step 3: Polish (45 min)
  3a. Apply 06_polish_pass.md     [30 min]
  3b. Final compile and visual QA [15 min]

Step 4: Practice (the night before)
  4a. Read 04_qa_prep.md three times
  4b. Time yourself with 05_speaker_notes.md
  4c. Practice the awkward bits aloud
```

## Decisions already made for you

To save time, I've made executive decisions on ambiguous points:

1. **Canonical PR-AUC = 0.827** (calibrated) for "our model on test split." Uncalibrated is 0.838. Both numbers are legitimate; calibrated is what gets reported in the headline because calibration is part of our pipeline.

2. **Final slide count = 30** in main deck, ~12 in backup/appendix.

3. **Comparison slide split into 2** (table + methodology framing) — the current single slide is overcrowded.

4. **SHAP figure: if Phase-1 parquet missing → single-panel figure for Phase-2.1 only.** The script in `03_regenerate_shap_figure.py` handles both cases.

5. **REVEL row removed** from the comparison table because it has no number; mentioned in footnote only.

If you disagree with any of these decisions, the LaTeX in `02_new_slides.tex` is structured so you can edit one block without breaking others.

## What I am NOT changing

I'm leaving these alone because they're already strong:

- The leakage journey chart (slide 19)
- The calibration triptych (slide 21)
- The pipeline flowchart (slide 14)
- The headline results table (slide 22) — only updating numbers
- The denovo-db external validation (slide 24)
- The future work and conclusion slides
- The overall green/gold KKU theme

## Build verification after each step

```bash
cd /path/to/repo
make clean
latexmk -pdf defense.tex   # or: make pdf

# Verify:
# - PDF compiles without errors
# - No "Phase-1 SHAP parquet missing" anywhere
# - All PR-AUC numbers match spec in 01_critical_fixes.md
# - UML titles read 1/6 through 6/6
# - Slide count ≤ 30 in main deck
```

## If something breaks

The LaTeX in `02_new_slides.tex` uses standard Beamer constructs (frame, columns, block, alertblock). If your custom theme uses different environments (e.g.\ a custom `\twocolumn` macro), search the existing `defense.tex` for an equivalent slide and mirror its structure.

The most likely break-points:
- `\rowcolor{kkugold!20}` requires the `colortbl` package — already loaded if your existing tables have any background color.
- `\textsuperscript` works in plain LaTeX but if your theme overrides it, use `${}^*$` in math mode.
- Custom font sizes — adjust `\small` / `\footnotesize` to match what your existing slides use.

---

**Bottom line:** Read `01_critical_fixes.md` first. Those three bugs will cost you points if you don't fix them, and they're the cheapest to fix. Everything else is upside.
