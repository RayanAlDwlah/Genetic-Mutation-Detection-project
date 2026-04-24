# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-1 Step 3).
"""Unit tests for the paired bootstrap utility."""
from __future__ import annotations

import numpy as np

from scripts.paired_bootstrap_denovo import paired_bootstrap_delta


def test_paired_bootstrap_positive_delta():
    """When post is strictly better than pre, delta should be > 0 with low p-value."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    # p_post is aligned with y; p_pre is noisy
    p_post = np.where(
        y == 1,
        rng.uniform(0.6, 0.95, size=n),
        rng.uniform(0.05, 0.4, size=n),
    )
    p_pre = p_post + rng.normal(0, 0.25, size=n)
    p_pre = np.clip(p_pre, 0.001, 0.999)
    out = paired_bootstrap_delta(y, p_pre, p_post, metric="roc_auc", n_boot=200, seed=1)
    assert out["point_delta"] > 0
    assert out["ci_low"] > 0 or out["p_value_one_sided"] < 0.2


def test_paired_bootstrap_zero_delta():
    """Identical pre and post: delta should be ~0 and CI should span 0."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0, 1, size=n)
    out = paired_bootstrap_delta(y, p, p, metric="roc_auc", n_boot=200, seed=1)
    assert abs(out["point_delta"]) < 1e-9
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_paired_bootstrap_degenerate_resample_handled():
    """A tiny all-positive slice must not raise."""
    y = np.array([1, 1, 1, 0, 1])
    p_pre = np.array([0.4, 0.5, 0.6, 0.1, 0.7])
    p_post = np.array([0.7, 0.8, 0.9, 0.2, 0.85])
    # Should not crash; may skip some replicates
    out = paired_bootstrap_delta(y, p_pre, p_post, metric="roc_auc", n_boot=50, seed=0)
    assert out["n_boot_used"] + out["n_boot_skipped"] == 50
