# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-2 Step 3).
"""Unit tests for intersection-subset comparison helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.intersection_baselines import bootstrap_metric, parse_score, parse_score_min


def test_bootstrap_returns_point_and_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.uniform(0, 1, size=500)
    point, lo, hi = bootstrap_metric(y, p, roc_auc_score, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert 0.3 <= point <= 0.7  # random labels -> AUROC near 0.5


def test_parse_score_max_handles_dbnsfp_packing():
    # dbNSFP packs multi-isoform scores as ";" separated values
    assert parse_score("0.9;0.7;0.85") == 0.9
    assert parse_score(".") is None
    assert parse_score("0.42") == 0.42
    assert parse_score(".;0.5;.") == 0.5


def test_parse_score_min_for_sift():
    # SIFT lower = more pathogenic; we take the min across isoforms
    assert parse_score_min("0.05;0.30") == 0.05
    assert parse_score_min(".") is None
    assert parse_score_min("0.7") == 0.7
