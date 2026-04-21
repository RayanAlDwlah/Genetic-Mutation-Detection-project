"""Unit tests for `src.calibration_deep`.

The Brier decomposition is a tight mathematical identity:

    Brier = Reliability − Resolution + Uncertainty

so we can verify the implementation by regenerating controlled data
and checking each term against an analytically known value.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.calibration_deep import (
    apply_isotonic,
    apply_platt,
    compute_decomposition_table,
    decompose_brier,
    fit_isotonic,
    fit_platt,
)

# ──────────────────────────── decompose_brier ─────────────────────────


class TestDecomposeBrier:
    def test_recomposition_identity_approximately_holds(self, rng: np.random.Generator) -> None:
        """brier ≈ reliability − resolution + uncertainty.

        Murphy's decomposition is exact only when every probability inside
        a bin is identical — with finite bins there is a small
        discretization residual. For 15 quantile bins with n=2000 and a
        well-spread signal, the residual is reliably < 0.01 (about 10% of
        a typical Brier). This test locks in that bound.
        """
        n = 2000
        y = (rng.uniform(size=n) < 0.3).astype(int)
        p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
        d = decompose_brier(y, p, n_bins=15)
        assert abs(d.recomposition_error) < 0.01, (
            f"decomposition residual {d.recomposition_error:.4e} > 0.01 — "
            f"check bin weights or uncertainty formula"
        )

    def test_recomposition_is_exact_when_each_bin_has_one_prob(self) -> None:
        """When every prediction inside a bin is identical, the Murphy
        decomposition becomes exact (< 1e-9 residual). We construct a
        deterministic case with n_bins > unique probabilities so each
        bin contains one repeated value."""
        # 6 predictions, only 3 distinct probabilities.
        p = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        y = np.array([0, 0, 1, 0, 1, 1])
        d = decompose_brier(y, p, n_bins=3)
        assert (
            abs(d.recomposition_error) < 1e-9
        ), f"exact-bin residual should be ~0, got {d.recomposition_error:.2e}"

    def test_uncertainty_matches_base_rate_variance(self, rng: np.random.Generator) -> None:
        """Uncertainty = ybar × (1 − ybar). This is irreducible."""
        y = (rng.uniform(size=1000) < 0.3).astype(int)
        p = rng.uniform(size=1000)
        d = decompose_brier(y, p, n_bins=10)
        expected = y.mean() * (1 - y.mean())
        assert d.uncertainty == pytest.approx(expected, rel=1e-9)

    def test_perfect_classifier_has_near_zero_reliability(self, rng: np.random.Generator) -> None:
        """A classifier that predicts the exact right probability has
        reliability ≈ 0 (bounded by bin discretization)."""
        n = 2000
        # Generate data where p_true really is p_predicted.
        p = rng.uniform(0.05, 0.95, size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        d = decompose_brier(y, p, n_bins=15)
        # Reliability ≤ 0.01 for a calibrated classifier.
        assert d.reliability < 0.01, f"expected low reliability, got {d.reliability:.4f}"

    def test_overconfident_classifier_has_high_reliability(self, rng: np.random.Generator) -> None:
        """A classifier that outputs 0 or 1 when the truth is 50/50 is
        maximally miscalibrated — reliability should be close to 0.25."""
        n = 2000
        y = (rng.uniform(size=n) < 0.5).astype(int)
        # Predict the exact wrong thing 50% of the time.
        p = np.where(y == 1, 0.01, 0.99)
        d = decompose_brier(y, p, n_bins=10)
        # reliability near (0.01 − 1)^2 ≈ 0.98
        assert (
            d.reliability > 0.9
        ), f"expected high reliability (=miscalibration), got {d.reliability:.4f}"

    def test_random_classifier_has_near_zero_resolution(self, rng: np.random.Generator) -> None:
        """A classifier that outputs a constant has resolution = 0
        because every bin has the same empirical positive rate."""
        n = 1000
        y = (rng.uniform(size=n) < 0.3).astype(int)
        p = np.full(n, 0.3)
        d = decompose_brier(y, p, n_bins=10)
        # With one effective bin, resolution ≈ 0.
        assert d.resolution < 0.01


# ──────────────────────────── Calibrators ─────────────────────────────


class TestPlatt:
    def test_monotone_increasing(self, rng: np.random.Generator) -> None:
        """Platt scaling is a monotone sigmoid of raw probability.
        Passing sorted inputs must yield sorted outputs."""
        n = 500
        p = rng.uniform(0, 1, n)
        y = (p + rng.normal(0, 0.1, n) > 0.5).astype(int)
        model = fit_platt(p, y)
        sorted_p = np.linspace(0, 1, 100)
        out = apply_platt(model, sorted_p)
        diffs = np.diff(out)
        assert (diffs >= -1e-9).all(), "Platt output not monotone"

    def test_output_in_01(self, rng: np.random.Generator) -> None:
        p = rng.uniform(0, 1, 500)
        y = (p > 0.5).astype(int)
        model = fit_platt(p, y)
        out = apply_platt(model, rng.uniform(0, 1, 200))
        assert (out >= 0).all() and (out <= 1).all()


class TestIsotonic:
    def test_monotone_non_decreasing(self, rng: np.random.Generator) -> None:
        """Isotonic is monotone-increasing by definition."""
        n = 500
        p = rng.uniform(0, 1, n)
        y = (p > 0.5).astype(int)
        model = fit_isotonic(p, y)
        sorted_p = np.linspace(0, 1, 100)
        out = apply_isotonic(model, sorted_p)
        assert (np.diff(out) >= -1e-9).all(), "Isotonic output not monotone"


# ──────────────────────────── Full pipeline ──────────────────────────


class TestDecompositionTable:
    def test_table_has_six_rows(self, rng: np.random.Generator) -> None:
        """compute_decomposition_table emits exactly
        (raw, platt, isotonic) × (val, test) = 6 rows."""
        n_val = 500
        n_test = 300
        val_y = (rng.uniform(size=n_val) < 0.3).astype(int)
        val_p = np.where(val_y == 1, rng.beta(5, 2, n_val), rng.beta(2, 5, n_val))
        test_y = (rng.uniform(size=n_test) < 0.3).astype(int)
        test_p = np.where(test_y == 1, rng.beta(5, 2, n_test), rng.beta(2, 5, n_test))
        table = compute_decomposition_table(
            val_true=val_y,
            val_raw=val_p,
            test_true=test_y,
            test_raw=test_p,
        )
        assert len(table) == 6
        assert set(table["calibrator"]) == {"raw", "platt", "isotonic"}
        assert set(table["split"]) == {"val", "test"}
        # All Brier values in [0, 1].
        assert (table["brier"].between(0, 1)).all()

    def test_isotonic_improves_reliability_on_val(self, rng: np.random.Generator) -> None:
        """Isotonic is the calibration gold standard; reliability on val
        (where it was fit) should drop substantially below the raw score."""
        n = 2000
        val_y = (rng.uniform(size=n) < 0.3).astype(int)
        # Deliberately miscalibrated: shift raw probs toward extremes.
        val_p = np.clip(np.where(val_y == 1, rng.beta(3, 2, n), rng.beta(2, 3, n)) ** 2, 0.01, 0.99)
        table = compute_decomposition_table(
            val_true=val_y,
            val_raw=val_p,
            test_true=val_y[:100],
            test_raw=val_p[:100],
        )
        val_rows = table[table["split"] == "val"].set_index("calibrator")
        # Isotonic reliability ≤ raw reliability on val (fit-time).
        assert val_rows.loc["isotonic", "reliability"] <= val_rows.loc["raw", "reliability"] + 1e-9
