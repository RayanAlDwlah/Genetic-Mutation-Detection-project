# Added for Phase 2.1 (S13): unit tests for the paired-bootstrap helper in ablate_esm2.
"""Unit tests for scripts/ablate_esm2.py paired_bootstrap_diff."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.ablate_esm2 import paired_bootstrap_diff


def test_paired_bootstrap_zero_when_identical():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0, 1, size=n)
    out = paired_bootstrap_diff(y, p, p, metric_fn=roc_auc_score, n_boot=200, seed=1)
    assert abs(out["point_delta"]) < 1e-9
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_paired_bootstrap_positive_when_a_better():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    p_a = np.where(y == 1, rng.uniform(0.6, 0.95, size=n), rng.uniform(0.05, 0.4, size=n))
    p_b = np.clip(p_a + rng.normal(0, 0.25, size=n), 0.001, 0.999)
    out = paired_bootstrap_diff(y, p_a, p_b, metric_fn=roc_auc_score, n_boot=200, seed=1)
    # p_a is aligned with y, p_b is noisy => Δ > 0 with high confidence
    assert out["point_delta"] > 0
