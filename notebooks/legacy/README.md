# Legacy notebooks — preserved for provenance only

These notebooks were written during the early (pre-Stage-0) iterations
of the project. They produced the first versions of the parquet splits
and the 0.955 PR-AUC number that turned out to be inflated by three
leakage sources.

**Do not run them.** They import from a state of `src/` that no longer
exists (ref_aa / alt_aa extraction logic, gnomAD merge schema, etc.
have all been rewritten). Many cells reference files that have been
moved, renamed, or had their columns changed.

They live in this directory because:

1. `docs/CHANGELOG.md` references them as evidence of how we got from
   0.955 to the honest 0.838.
2. Future readers comparing v0.2 to v0.3 need the old narrative to see
   what changed.

The **current** project narrative lives in `notebooks/00_overview.ipynb`
through `notebooks/09_esm2_zero_shot.ipynb` — those are the reproducible,
up-to-date story.
