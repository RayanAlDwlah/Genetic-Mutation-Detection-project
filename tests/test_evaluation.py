"""Unit tests for `src.evaluation` — metrics, bootstrap CIs, reliability curve.

These tests use synthetic `(y_true, y_prob)` pairs with known properties so
we can anchor ROC/PR/F1/Brier to mathematically-defensible expected ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.evaluation import (
    bootstrap_metrics,
    compute_classification_metrics,
    reliability_curve,
    select_best_threshold,
)


class TestComputeClassificationMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["pr_auc"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["tn"] == 3 and m["tp"] == 3
        assert m["fn"] == 0 and m["fp"] == 0

    def test_random_signal_range(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        # All metrics must be in [0, 1]
        for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            assert 0.0 <= m[k] <= 1.0, f"{k}={m[k]} out of [0,1]"
        # Our mock signal is clearly non-trivial — ROC > 0.7 expected.
        assert m["roc_auc"] > 0.7
        # Brier must be in [0, 0.25] for balanced binary data and in [0, 1] always.
        assert 0.0 <= m["brier_loss"] <= 1.0

    def test_counts_add_up_to_support(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["support"]
        assert m["pathogenic_count"] + m["benign_count"] == m["support"]

    def test_single_class_graceful_degradation(self) -> None:
        """Single-class input: ROC-AUC must be NaN (undefined), but the
        rest of the metrics should still compute without crashing — this
        matches sklearn's graceful-nan behavior."""
        y_true = np.zeros(10, dtype=int)
        y_prob = np.full(10, 0.1)
        # Sklearn emits UndefinedMetricWarning; pyproject.toml has warnings
        # as errors, so we must wrap in warns().
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = compute_classification_metrics(y_true, y_prob)
        assert np.isnan(m["roc_auc"]), f"expected NaN ROC-AUC, got {m['roc_auc']}"
        # Support + counts remain sane.
        assert m["support"] == 10
        assert m["benign_count"] == 10 and m["pathogenic_count"] == 0

    def test_extreme_probabilities_are_clipped(self) -> None:
        """Exact 0 and 1 probabilities are clipped to avoid log(0)."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        # No infinity / NaN
        assert np.isfinite(m["log_loss"])
        assert np.isfinite(m["brier_loss"])


class TestBootstrapMetrics:
    def test_ci_contains_point_estimate(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        point = compute_classification_metrics(y_true, y_prob)
        boot = bootstrap_metrics(y_true, y_prob, n_boot=200, seed=42)
        # 95% CI should contain the full-sample point estimate for ROC/PR
        # almost always — we check the *median* of bootstrap ≈ point.
        for metric in ("roc_auc", "pr_auc"):
            b = boot[metric]
            assert b["ci_lo"] <= point[metric] <= b["ci_hi"], (
                f"{metric}: point {point[metric]:.4f} outside bootstrap CI "
                f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]"
            )
            assert b["ci_lo"] <= b["median"] <= b["ci_hi"]

    def test_ci_width_shrinks_with_more_samples(self, rng: np.random.Generator) -> None:
        """Wider dataset → tighter bootstrap CI. Sanity on variance scaling."""
        y_small = (rng.uniform(size=200) < 0.3).astype(int)
        p_small = np.where(y_small, rng.beta(5, 2, 200), rng.beta(2, 5, 200))
        y_big = (rng.uniform(size=2000) < 0.3).astype(int)
        p_big = np.where(y_big, rng.beta(5, 2, 2000), rng.beta(2, 5, 2000))
        small = bootstrap_metrics(y_small, p_small, n_boot=100, seed=42)["roc_auc"]
        big = bootstrap_metrics(y_big, p_big, n_boot=100, seed=42)["roc_auc"]
        small_width = small["ci_hi"] - small["ci_lo"]
        big_width = big["ci_hi"] - big["ci_lo"]
        assert (
            big_width < small_width
        ), f"bigger dataset produced wider CI: small={small_width:.4f}, big={big_width:.4f}"

    def test_deterministic_with_seed(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        a = bootstrap_metrics(y_true, y_prob, n_boot=50, seed=42)["roc_auc"]["mean"]
        b = bootstrap_metrics(y_true, y_prob, n_boot=50, seed=42)["roc_auc"]["mean"]
        assert a == b


class TestReliabilityCurve:
    def test_bin_counts_sum_to_support(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        df = reliability_curve(y_true, y_prob, n_bins=15)
        assert int(df["count"].sum()) == len(y_true)

    def test_ece_mce_in_attrs(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        df = reliability_curve(y_true, y_prob, n_bins=10)
        assert "ECE" in df.attrs and "MCE" in df.attrs
        assert 0.0 <= df.attrs["ECE"] <= 1.0
        assert 0.0 <= df.attrs["MCE"] <= 1.0
        assert df.attrs["MCE"] >= df.attrs["ECE"]  # MCE is max gap, ECE is mean

    def test_well_calibrated_classifier_has_low_ece(self, rng: np.random.Generator) -> None:
        """If p_true = p_predicted (perfectly calibrated), ECE ≈ 0."""
        n = 5000
        p = rng.uniform(0.05, 0.95, size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        df = reliability_curve(y, p, n_bins=15)
        assert (
            df.attrs["ECE"] < 0.05
        ), f"expected low ECE for calibrated data, got {df.attrs['ECE']:.4f}"


class TestSelectBestThreshold:
    def test_returns_threshold_in_search_range(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        t, curve = select_best_threshold(y_true, y_prob)
        assert 0.20 <= t <= 0.80
        assert isinstance(curve, pd.DataFrame)
        assert {"threshold", "f1"}.issubset(set(curve.columns))

    def test_f1_at_best_threshold_is_maximum(self, mock_binary_predictions) -> None:
        y_true, y_prob = mock_binary_predictions
        best_t, curve = select_best_threshold(y_true, y_prob)
        # The F1 at the chosen threshold must be ≥ every other F1 in the curve
        best_row = curve.loc[curve["threshold"].sub(best_t).abs().idxmin()]
        assert best_row["f1"] >= curve["f1"].max() - 1e-9
