# Defense Playbook — Final 48 Hours

Single-source-of-truth for everything that's left after the deck and
prep package landed in this repo. The code-side work is done; what
remains is rehearsal and physical setup.

---

## 1. Build the PDF (do this first)

The repo doesn't ship a compiled PDF — produce one locally:

```bash
cd report/academic
latexmk -pdf defense.tex          # if you have latexmk
# or:
pdflatex defense.tex && pdflatex defense.tex   # twice for cross-refs
```

If `pdflatex` is not installed:

```bash
brew install --cask mactex        # ~4 GB, takes a while
```

Or use Overleaf: upload `report/academic/defense.tex` plus the
`figures/` folder.

**Sanity check the PDF before practicing:**

- Total pages: ~53 (48 main + 5 appendix)
- Cover slide: KKU ribbon at top, supervisor + date visible
- Slide 18 ("Headline results"): table shows both PR-AUC rows
- Slide 19 ("SHAP interpretability"): yellow rows with legend above
- Slide ~26 ("Streamlit demo"): screenshot + 4-row annotation table
- Slide ~31 ("What ESM-2 changed in SHAP"): proper bar chart, no
  "parquet missing" placeholder
- All UML titles read N/6 (1-3 in main, 4-6 in appendix)

---

## 2. Practice schedule (the night before)

From `00_README.md` Step 4:

- [ ] Read `04_qa_prep.md` three times — focus on Q1, Q2, Q4, Q5, Q15
      (those are 80% of the questions you'll get).
- [ ] Read `05_speaker_notes.md` once aloud at full speed, stopwatch
      running. Target: 25 min through slide 45 (Thank-you).
- [ ] Identify the 2-3 awkward bits and rehearse them isolated.
- [ ] Sleep 7+ hours.

If you go over 28 minutes on the timed run: cut Slides 25-30 detail
(UML 1-3) per the speaker notes' "Pacing Discipline" section. UML 4-6
already moved to backup.

---

## 3. Q&A memorization checklist

15 questions in `04_qa_prep.md`. Memorize the **structure** of each
answer, not the wording. Self-quiz aloud:

- [ ] Q1 (denovo-db close to chance — almost-certain to be asked)
- [ ] Q2 (PR-AUC of 0.865 means what clinically)
- [ ] Q3 (XGBoost vs deep NN)
- [ ] Q4 (drop from 0.955 to 0.838 — does that mean worse?)
- [ ] Q5 (paralog split — how do you know it's disjoint?)
- [ ] Q6 (why only 35M ESM-2)
- [ ] Q7 (excluded REVEL/CADD but compare AlphaMissense?)
- [ ] Q8 (59% test coverage — justify)
- [ ] Q9 (clinical geneticist sees probability 0.71 — trust it?)
- [ ] Q10 (do differently if started over)
- [ ] Q11 (+0.031 PR-AUC — practical or just statistical?)
- [ ] Q12 (isotonic vs Platt)
- [ ] Q13 (your specific contribution — see `rayan_commits.txt`)
- [ ] Q14 (weakest part of the project)
- [ ] Q15 (one thing to remember — rehearse this **exactly**)

If you genuinely don't know mid-defense, use one of:

- "I don't have that number with me, but I can reproduce it from the
  bootstrap parquet — would you like me to send a follow-up?"
- "That's a question we considered but didn't fully resolve. Here's
  our partial thinking: ... If I had more time, I'd ..."
- "I don't know. My intuition is X, but I haven't tested it."

---

## 4. What you should NEVER do (from `04_qa_prep.md`)

1. Don't say "obviously" — if it were obvious, they wouldn't ask.
2. Don't argue with framing — restate it ("If I understand correctly,
   you're asking ...") then answer.
3. Don't apologize for limitations — *describe* them. "This is a
   limitation" is scientific; "I'm sorry it didn't work better" is
   undergraduate.
4. Don't improvise into territory you didn't prepare. A clean "good
   question, I don't have a complete answer" beats a confident wrong
   claim.

---

## 5. Day-of pre-flight (from `06_polish_pass.md` Tweak 10)

Before walking into the defense room:

- [ ] PDF compiled with all fixes applied
- [ ] PDF on USB drive **and** in cloud (Drive/Dropbox) **and** on
      phone — three copies, zero failure modes
- [ ] Charger for laptop
- [ ] 2 printed copies of the deck (4 slides per page) — one for you,
      one as backup
- [ ] Bottle of water, drink before slide 1 and again at slide 18
- [ ] Phone on silent (and screen-down)
- [ ] 5 min before: read slide 2 ("This talk in one minute") aloud
      once for warm-up

---

## 6. Voice and delivery (from `05_speaker_notes.md`)

- Speak **slower** than feels natural. Defense nerves accelerate
  speech ~30%; match the audience's listening pace, not yours.
- Pause after each headline number. "PR-AUC 0.827." [pause] "That's
  our calibrated number." A pause signals confidence.
- Look at the committee, not the slides — you know the slides.
- Hands visible, not pocketed; gesture sparingly to emphasize
  transitions.

---

## 7. Stop-loss principles

- Done is better than perfect. Everything in the prep package is
  applied; nothing is half-finished. Don't keep tweaking the LaTeX in
  the last 12 hours — practice instead.
- If a build error appears at 11pm: the source committed in this
  branch passes structural checks (53 frames, 10 columns, 8 blocks,
  all balanced). If you broke something locally, `git reset --hard
  origin/main` to recover.
- If you blank on a question: silence is fine for 3 seconds. "Let me
  think about that for a moment." is fine for 5 seconds. Beyond that,
  use the "I don't know" script.

---

## 8. After the defense

- [ ] Push your final tagged commit: `git tag defense-final && git
      push --tags` so the version you defended is permanently
      pinned.
- [ ] Save the committee's questions list — useful for the journal
      paper version of this work.

You've put graduate-level effort into this. The committee will see
that. Trust the work, breathe, and walk in.
