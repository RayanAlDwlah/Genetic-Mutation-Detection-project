"""External validation utilities for the XGBoost baseline.

The public surface is intentionally small:

- `variant_mapper.to_canonical_key` — turn heterogeneous variant records into
  the `chr:pos:ref:alt` key used throughout the pipeline.
- `denovo_loader.load_denovo_db` — parse the denovo-db TSV into a labeled
  variant table ready for featurization.
- `featurize.featurize_external` — attach dbNSFP/gnomAD features to an
  external variant table using the cached intermediate parquets.
- `evaluate.evaluate_external` — run the calibrated model, compute metrics
  with bootstrap CIs, and emit per-source CSV artifacts.

Everything is leakage-safe by construction: featurization uses the same
cached lookups produced during training, and no external labels or
predictions are ever written back into the training set.
"""
