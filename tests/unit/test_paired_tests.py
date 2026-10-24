"""Unit tests for src.stats.paired_tests (DeLong, paired bootstrap, Holm)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from src.stats.paired_tests import (
    PairwiseRow,
    auc_greater_than_half,
    delong_test,
    holm_bonferroni,
    paired_bootstrap_prauc,
)


@pytest.fixture
def two_models():
    """Labels plus a strong and a weak scorer on the same 600 samples."""
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=600) < 0.3).astype(int)
    strong = y * 1.5 + rng.normal(size=600)
    weak = y * 0.3 + rng.normal(size=600)
    return y, strong, weak


def test_delong_aucs_match_sklearn(two_models) -> None:
    y, strong, weak = two_models
    out = delong_test(y, strong, weak)
    assert out["auc1"] == pytest.approx(roc_auc_score(y, strong), abs=1e-12)
    assert out["auc2"] == pytest.approx(roc_auc_score(y, weak), abs=1e-12)
    assert out["delta"] == pytest.approx(out["auc1"] - out["auc2"])


def test_delong_detects_clear_difference(two_models) -> None:
    y, strong, weak = two_models
    out = delong_test(y, strong, weak)
    assert out["delta"] > 0
    assert out["delta_ci_lo"] < out["delta"] < out["delta_ci_hi"]
    assert out["pvalue"] < 1e-3


def test_delong_identical_models_have_zero_variance(two_models) -> None:
    y, strong, _ = two_models
    out = delong_test(y, strong, strong)
    assert out["delta"] == 0.0
    assert out["delta_se"] == 0.0
    assert np.isnan(out["pvalue"])


def test_delong_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        delong_test([0, 1, 0], [0.1, 0.9, 0.2], [0.1, 0.9])


def test_paired_bootstrap_prauc_matches_point_estimates(two_models) -> None:
    y, strong, weak = two_models
    out = paired_bootstrap_prauc(y, strong, weak, n_boot=300, seed=1)
    assert out["prauc1"] == pytest.approx(average_precision_score(y, strong))
    assert out["prauc2"] == pytest.approx(average_precision_score(y, weak))
    assert out["delta"] == pytest.approx(out["prauc1"] - out["prauc2"])
    assert out["delta_ci_lo"] < out["delta"] < out["delta_ci_hi"]
    assert out["delta_ci_lo"] > 0
    assert out["pvalue"] < 0.01
    assert out["n_boot_valid"] == 300


def test_paired_bootstrap_prauc_is_seed_deterministic(two_models) -> None:
    y, strong, weak = two_models
    a = paired_bootstrap_prauc(y, strong, weak, n_boot=100, seed=7)
    b = paired_bootstrap_prauc(y, strong, weak, n_boot=100, seed=7)
    assert a == b


def test_auc_greater_than_half_signal_vs_noise() -> None:
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=300) < 0.4).astype(int)
    signal = auc_greater_than_half(y, y + rng.normal(size=300), n_boot=200, seed=0)
    noise = auc_greater_than_half(y, rng.normal(size=300), n_boot=200, seed=0)

    assert signal["auc"] > 0.7
    assert signal["pvalue_permutation"] < 0.01
    assert signal["auc_ci_lo"] > 0.5
    assert noise["pvalue_permutation"] > 0.01
    assert signal["n"] == 300 and signal["n_positive"] == int(y.sum())
    assert signal["n_permutations"] == 200


def test_holm_bonferroni_known_values() -> None:
    # Sorted p: 0.01*3 = 0.03, 0.03*2 = 0.06, max(0.06, 0.04*1) = 0.06.
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_bonferroni_caps_at_one() -> None:
    assert holm_bonferroni([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_pairwise_row_as_dict_round_trip() -> None:
    row = PairwiseRow("test", "A", "B", "roc_auc", 0.9, 0.8, 0.1, 0.05, 0.15, 0.001)
    d = row.as_dict()
    assert d["model_a"] == "A" and d["delta"] == 0.1
    assert np.isnan(d["pvalue_holm"])
