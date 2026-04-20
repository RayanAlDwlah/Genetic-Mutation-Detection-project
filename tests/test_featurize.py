"""Unit tests for `src.external_validation.featurize.featurize_external`.

The featurizer left-joins external variants onto the cached dbNSFP feature
parquet by `variant_key`. Rows without cache coverage must be reported as
`unmapped` — never silently filled with defaults. These tests lock in that
contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.external_validation.featurize import FeaturizationResult, featurize_external


@pytest.fixture
def tiny_cache(tmp_path: Path) -> Path:
    """A 3-row cached-features parquet with one column of made-up scores."""
    df = pd.DataFrame(
        {
            "variant_key": ["1:1:A:G", "1:2:A:G", "1:3:A:G"],
            "phyloP100way_vertebrate": [5.1, 3.0, -0.5],
            "BLOSUM62_score": [-1, 2, 0],
        }
    )
    p = tmp_path / "cache.parquet"
    df.to_parquet(p, index=False)
    return p


def test_full_coverage(tiny_cache: Path) -> None:
    ext = pd.DataFrame(
        {
            "variant_key": ["1:1:A:G", "1:2:A:G"],
            "label": [1, 0],
            "gene": ["BRCA1", "TP53"],
        }
    )
    r = featurize_external(ext, dbnsfp_cache=tiny_cache)
    assert isinstance(r, FeaturizationResult)
    assert r.coverage == pytest.approx(1.0)
    assert len(r.featurized) == 2
    assert len(r.unmapped) == 0
    # dbNSFP columns were attached
    assert {"phyloP100way_vertebrate", "BLOSUM62_score"}.issubset(r.featurized.columns)


def test_partial_coverage_reports_unmapped(tiny_cache: Path) -> None:
    ext = pd.DataFrame(
        {
            "variant_key": ["1:1:A:G", "99:999:A:G"],
            "label": [1, 0],
        }
    )
    r = featurize_external(ext, dbnsfp_cache=tiny_cache)
    assert r.coverage == pytest.approx(0.5)
    assert len(r.featurized) == 1
    assert len(r.unmapped) == 1
    assert r.unmapped.iloc[0]["variant_key"] == "99:999:A:G"


def test_missing_variant_key_column_raises(tiny_cache: Path) -> None:
    ext = pd.DataFrame({"label": [1], "gene": ["BRCA1"]})
    with pytest.raises(ValueError, match="variant_key"):
        featurize_external(ext, dbnsfp_cache=tiny_cache)


def test_missing_label_column_raises(tiny_cache: Path) -> None:
    ext = pd.DataFrame({"variant_key": ["1:1:A:G"]})
    with pytest.raises(ValueError, match="label"):
        featurize_external(ext, dbnsfp_cache=tiny_cache)


def test_overlapping_columns_preserve_external_value(tiny_cache: Path) -> None:
    """If the external table already carries a column that also appears in
    the cache, the external value wins. The featurizer drops the conflicting
    column from the cache before merging to avoid `_x`/`_y` pollution."""
    ext = pd.DataFrame(
        {
            "variant_key": ["1:1:A:G"],
            "label": [1],
            "BLOSUM62_score": [99.0],  # external value, should be preserved
        }
    )
    r = featurize_external(ext, dbnsfp_cache=tiny_cache)
    # Cache had -1 for this key, but external's 99.0 wins.
    assert r.featurized["BLOSUM62_score"].iloc[0] == pytest.approx(99.0)
