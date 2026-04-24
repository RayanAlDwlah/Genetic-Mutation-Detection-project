# Added for Phase 2.1 (S13): unit tests for build_phase21_train aggregation logic.
"""Unit tests for scripts/build_phase21_train.py.

Asserts the per-variant min(esm2_llr) aggregation, NaN propagation, and
the is_imputed_esm2_llr flag.
"""
from __future__ import annotations

import pandas as pd

from scripts.build_phase21_train import aggregate_min, merge_split


def test_aggregate_min_picks_lowest_per_variant():
    scores = pd.DataFrame(
        {
            "variant_key": ["A", "A", "B", "C"],
            "esm2_llr": [-1.0, -3.0, 0.5, 2.0],
        }
    )
    out = aggregate_min(scores).set_index("variant_key")["esm2_llr"]
    assert out["A"] == -3.0
    assert out["B"] == 0.5
    assert out["C"] == 2.0


def test_aggregate_min_drops_nan_rows():
    scores = pd.DataFrame(
        {
            "variant_key": ["A", "A", "B"],
            "esm2_llr": [None, -2.0, None],
        }
    )
    out = aggregate_min(scores)
    keys = set(out["variant_key"])
    assert keys == {"A"}, keys


def test_merge_split_adds_imputed_flag():
    split_df = pd.DataFrame({"variant_key": ["A", "B", "C"], "label": [1, 0, 1]})
    agg = pd.DataFrame({"variant_key": ["A", "C"], "esm2_llr": [-1.5, 2.0]})
    merged = merge_split(split_df, agg)
    flag = merged.set_index("variant_key")["is_imputed_esm2_llr"]
    assert flag["A"] == 0
    assert flag["B"] == 1
    assert flag["C"] == 0
    llr = merged.set_index("variant_key")["esm2_llr"]
    assert llr["A"] == -1.5
    assert pd.isna(llr["B"])
    assert llr["C"] == 2.0
