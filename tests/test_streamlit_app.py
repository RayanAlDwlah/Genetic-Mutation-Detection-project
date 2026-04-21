"""Round-trip tests for the Streamlit demo's scoring path.

We don't spin up Streamlit itself — we import the `score_variant`
function directly and assert that:

1. An invalid variant string returns an `error` key.
2. A non-SNV / multi-nucleotide ref|alt is rejected as not-missense.
3. A committed-split variant round-trips and produces a sensible
   probability + SHAP table.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def score_fn():
    """Import lazily so `streamlit` is only loaded when this test runs."""
    import scripts.streamlit_app as app

    return app.score_variant


class TestVariantKeyParsing:
    def test_malformed_key_rejected(self, score_fn) -> None:
        result = score_fn("not-a-variant-key")
        assert "error" in result
        assert "chr:pos:ref:alt" in result["error"]

    def test_multi_nucleotide_rejected(self, score_fn) -> None:
        result = score_fn("17:41244936:AT:G")
        assert "error" in result
        assert "missense" in result["error"].lower() or "valid" in result["error"].lower()


class TestCachedVariantRoundTrip:
    """Pick a random pathogenic test variant and verify it scores."""

    @pytest.fixture
    def cached_pathogenic_key(self, repo_root: Path) -> str:
        test = pd.read_parquet(repo_root / "data/splits/test.parquet")
        return test[test["label"] == 1].iloc[0]["variant_key"]

    def test_round_trip(self, score_fn, cached_pathogenic_key: str) -> None:
        out = score_fn(cached_pathogenic_key)
        assert "error" not in out, out.get("error")
        assert "p_raw" in out and "p_calibrated" in out
        assert 0.0 <= out["p_raw"] <= 1.0
        assert 0.0 <= out["p_calibrated"] <= 1.0

    def test_shap_df_shape(self, score_fn, cached_pathogenic_key: str) -> None:
        out = score_fn(cached_pathogenic_key)
        assert "shap_df" in out
        shap_df = out["shap_df"]
        assert {"feature", "shap_value", "input_value"}.issubset(shap_df.columns)
        assert len(shap_df) == 15  # top-15 by default
        # SHAP values sorted by descending abs — monotone non-increasing magnitude.
        abs_vals = shap_df["shap_value"].abs().to_numpy()
        assert (abs_vals[:-1] >= abs_vals[1:] - 1e-9).all()

    def test_source_note_present(self, score_fn, cached_pathogenic_key: str) -> None:
        out = score_fn(cached_pathogenic_key)
        assert "source_note" in out
        assert "split" in out["source_note"] or "cache" in out["source_note"]
