# Polish Pass — Small Tweaks Across the Deck

These are minor improvements that individually don't matter, but collectively raise the deck from "very good" to "polished." Apply *after* the critical fixes and slide consolidations from steps 01-02.

**Time estimate:** 30 minutes total.

---

## Tweak 1: Add a legend to highlighted table rows

**Slides affected:** 23 (SHAP table), 24 (denovo-db).

The yellow-highlighted rows currently mean "Phase 2.1 features" or "holdout post-constraint" but the reader doesn't know that until they read the bullets *below* the table. Add a one-liner above each table.

**Slide 23 — above the table:**
```latex
\footnotesize\emph{Highlighted rows: gene-level constraint features added in Phase 2.1.}
```

**Slide 24 — above the table:**
```latex
\footnotesize\emph{Highlighted rows: holdout from gene families completely unseen in training.}
```

---

## Tweak 2: Disambiguate the LLR formula notation

**Slide 41 (ESM-2 architecture):** the formula uses `s̃_\π` which isn't defined.

**Replace:**
```latex
\mathrm{LLR}_{\text{ESM-2}} = \log \frac{\mathbb{P}_\theta(a_{\text{alt}} \mid \tilde{s}_{\setminus \pi})}{\mathbb{P}_\theta(a_{\text{ref}} \mid \tilde{s}_{\setminus \pi})}
```

**With (clearer):**
```latex
\mathrm{LLR}_{\text{ESM-2}} = \log \frac{\mathbb{P}_\theta(a_{\text{alt}} \mid s_{\setminus \pi})}{\mathbb{P}_\theta(a_{\text{ref}} \mid s_{\setminus \pi})}
\qquad \text{where } s_{\setminus \pi} = \text{sequence with position } \pi \text{ masked}
```

The dropped tilde simplifies; the inline definition removes ambiguity.

---

## Tweak 3: Sharper opening line on every section divider

The current dividers are bare titles ("Introduction", "Related Works", etc.). Add one line that sets up what's coming. This is what tour guides call a "narrative hand-off."

**Section dividers to update:**

```latex
% Slide 3 (Introduction)
\begin{frame}{Introduction}
  \centering
  \vspace{2cm}
  \Large \emph{What we're trying to predict, and why it's harder than it looks.}
\end{frame}

% Slide 7 (Related Works)
\begin{frame}{Related Works}
  \centering
  \vspace{2cm}
  \Large \emph{Three generations of variant predictors. None reports a like-for-like external test.}
\end{frame}

% Slide 11 (Contribution)
\begin{frame}{Contribution}
  \centering
  \vspace{2cm}
  \Large \emph{What this project adds: an audit, a baseline, and a reproducibility commitment.}
\end{frame}

% Slide 13 (Proposed System)
\begin{frame}{Proposed System}
  \centering
  \vspace{2cm}
  \Large \emph{Three pipeline stages, each with a fail-closed CI gate.}
\end{frame}

% Slide 25 (System Analysis & Design)
\begin{frame}{System Analysis \& Design}
  \centering
  \vspace{2cm}
  \Large \emph{Eight functional + seven non-functional requirements, all traceable to code and tests.}
\end{frame}

% Slide 36 (Implementation)
\begin{frame}{Implementation}
  \centering
  \vspace{2cm}
  \Large \emph{The stack, the tests, and the leakage gate that keeps us honest.}
\end{frame}

% Slide 46 (Comparison)
\begin{frame}{Comparison}
  \centering
  \vspace{2cm}
  \Large \emph{How our numbers stack up under matched methodology.}
\end{frame}

% Slide 48 (Conclusion)
\begin{frame}{Conclusion}
  \centering
  \vspace{2cm}
  \Large \emph{What we found, what we didn't, and what comes next.}
\end{frame}
```

> **Trade-off:** these narrative dividers add ~1 second per section to your runtime. Worth it if you're at ≤25 minutes; cut them if you're at the limit.

---

## Tweak 4: Make the ESM-2 zero-shot results slide more readable

**Slide 43** crams two tables and a take-away into one slide. Reorganize:

```latex
\begin{frame}{ESM-2 zero-shot + rank fusion}
  \centering
  \footnotesize

  \textbf{Test split (28{,}098 variants, paralog-disjoint)}
  
  \vspace{0.4em}
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Model                      & ROC-AUC & PR-AUC & $\alpha$ \\
    \midrule
    XGBoost (calibrated)       & 0.938 & 0.827 & ---   \\
    ESM-2 LLR (zero-shot)      & 0.735 & 0.604 & ---   \\
    Rank fusion, uniform       & 0.895 & 0.784 & 0.50  \\
    \rowcolor{kkugold!20}
    \textbf{Rank fusion, tuned} & \textbf{0.941} & \textbf{0.852} & \textbf{0.175} \\
    \bottomrule
  \end{tabular}

  \vspace{0.8em}
  \begin{columns}[T]
    \column{0.5\textwidth}
    \textbf{External (denovo-db, family-holdout):}
    \begin{itemize}
      \footnotesize
      \item XGBoost: ROC 0.573
      \item ESM-2 alone: ROC 0.552
      \item Fusion: ROC \textbf{0.588} ($+0.015$)
    \end{itemize}

    \column{0.5\textwidth}
    \begin{exampleblock}{Two signals of orthogonality}
      \footnotesize
      Fusion improves over either source AND the gain transfers to held-out gene families. Motivates training-time integration of the LLR.
    \end{exampleblock}
  \end{columns}
\end{frame}
```

---

## Tweak 5: Restructure the comparison footnotes

The current footnotes on slide 38 are wrapping in ways that obscure them. After splitting into 38a + 38b (per `02_new_slides.tex`), make sure footnote text:

- Uses `\footnotesize` not `\small`
- Sits in `\vspace{0.5em}` from the table
- Has no `\\` line breaks inside the body of the footnote (let LaTeX flow it)
- Asterisks render as `\textsuperscript{*}` not `^*` (for non-math contexts)

---

## Tweak 6: Improve the Streamlit demo slide

**Slide 35** shows a screenshot but the text is small. Add an annotation overlay.

```latex
\begin{frame}{Streamlit demo --- scoring a variant}
  \centering
  \includegraphics[width=0.92\linewidth]{figures/streamlit_demo.png}

  \vspace{0.4em}
  \footnotesize
  \begin{tabular}{@{}rl@{}}
    \textbf{Input}     & GRCh38 variant key: \texttt{chr:pos:ref:alt} \\
    \textbf{Output}    & Calibrated probability + raw XGBoost score + risk band \\
    \textbf{Explainer} & Top-15 SHAP contributions, color-coded by direction \\
    \textbf{Latency}   & $<$ 100 ms on cache hit, $\sim$ 2 s on cache miss (VEP fetch)
  \end{tabular}
\end{frame}
```

---

## Tweak 7: Color discipline on the calibration triptych

**Slide 21:** Currently three subplots in a row. The "perfect" diagonal line is dashed gray, which is correct, but the three "fraction positive" lines are all the same blue. Differentiate them:

- Raw → red (signals miscalibration)
- Platt → orange (intermediate)
- Isotonic → green (good)

This is a 5-line edit in whatever Python script generated the figure. If the figure is committed as a PNG without source, leave it alone — color regeneration isn't worth the time.

---

## Tweak 8: Sharper title for slide 22 (Headline results)

The current title is "Part 3 -- Headline results on test (1/3)" — bureaucratic.

**Replace with:** `Headline results --- paralog-disjoint test`

The "Part 3" framing is implicit from the section divider; you don't need it again.

---

## Tweak 9: Add slide numbers in a less obtrusive footer

The current footer reads "King Khalid University | College of Computer Science    21/42" on every slide. After consolidation it'll be 21/30 — fine, but the page number font is too prominent.

If your beamer theme exposes the footer template, reduce the page number to `\scriptsize` and right-align without bold.

This is purely cosmetic — skip if your theme doesn't make it easy.

---

## Tweak 10: Pre-flight checklist (read once, day-of)

Before walking into the defense room:

- [ ] PDF compiled with all fixes applied
- [ ] PDF on USB drive AND in cloud (Drive/Dropbox) AND on phone
- [ ] Charger for laptop
- [ ] Print 2 copies of slide deck (4 slides per page) — one for you, one as backup
- [ ] Bottle of water
- [ ] Phone on silent
- [ ] 5 min before: read slide 2 ("This talk in one minute") aloud once

---

## Stop-loss

You are allowed to ship the deck without applying every tweak in this file. If you've done 01-04 cleanly, the deck is already at A+ level. Tweaks 1-10 take it from A+ to "the best defense the committee will see this term."

But "good enough and submitted" beats "perfect and rushed at the last minute." Know when to stop polishing.
