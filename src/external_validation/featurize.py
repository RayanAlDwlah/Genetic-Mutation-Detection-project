"""Attach pipeline features to an external variant table.

Strategy
--------
The training model was fit on features extracted from dbNSFP 5.3.1a. To score
an external variant we need the *same* feature vector. This module joins the
external table to the cached extraction parquet
(`data/intermediate/dbnsfp_selected_features.parquet`) on `variant_key`.

Rows that don't appear in the cache cannot be scored without re-running
`src/dbnsfp_extraction.py` on a dbNSFP slice covering those coordinates.
They are returned in `unmapped` and logged — **never silently filled**. This
keeps the external-coverage metric honest.

The gnomAD allele-frequency features are *not* required by the post-lockdown
training matrix (see `src/training.py::select_feature_columns` — AF columns
are excluded from the final feature set), so we do not need a gnomAD join
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class FeaturizationResult:
    featurized: pd.DataFrame  # ready-to-score rows
    unmapped: pd.DataFrame  # rows without dbNSFP coverage
    coverage: float  # featurized / total

    def summary(self) -> str:
        tot = len(self.featurized) + len(self.unmapped)
        return (
            f"Featurization coverage: {len(self.featurized):,} / {tot:,} "
            f"({self.coverage:.1%}). "
            f"Unmapped (no dbNSFP entry): {len(self.unmapped):,}."
        )


def featurize_external(
    ext: pd.DataFrame,
    *,
    dbnsfp_cache: Path,
) -> FeaturizationResult:
    """Left-join external variants onto the cached dbNSFP feature frame.

    `ext` must contain `variant_key` plus a `label` column and whatever
    metadata the caller wants preserved (gene, study, phenotype, …).
    """
    if "variant_key" not in ext.columns:
        raise ValueError("external table must have a `variant_key` column")
    if "label" not in ext.columns:
        raise ValueError("external table must have a `label` column")

    cache = pd.read_parquet(dbnsfp_cache)
    # Drop any columns already present in `ext` (chr/pos/ref/alt etc.) to
    # avoid `_x` / `_y` pollution after the merge.
    overlap = set(cache.columns) & set(ext.columns)
    overlap.discard("variant_key")
    cache = cache.drop(columns=list(overlap))

    merged = ext.merge(cache, on="variant_key", how="left", indicator=True)
    ok = merged["_merge"] == "both"
    featurized = merged.loc[ok].drop(columns="_merge").reset_index(drop=True)
    unmapped = merged.loc[~ok, ext.columns.tolist()].reset_index(drop=True)

    coverage = len(featurized) / max(len(merged), 1)
    return FeaturizationResult(featurized=featurized, unmapped=unmapped, coverage=coverage)
