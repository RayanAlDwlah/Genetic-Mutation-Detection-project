# Added for Phase 2.1 (S13): regression guard for the Phase-2.1 test PR-AUC.
"""Catches silent drift of the Phase-2.1 model headline number."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
BOOT = REPO / "results/metrics/phase21/xgboost_bootstrap_ci.csv"


@pytest.mark.skipif(not BOOT.exists(), reason="Phase-2.1 evaluation not run yet")
def test_phase21_test_pr_auc_in_band():
    df = pd.read_csv(BOOT)
    row = df[df["metric_set"] == "test_calibrated"]
    assert not row.empty, "test_calibrated row missing"
    pr_mean = float(row["pr_auc__mean"].iloc[0])
    assert 0.82 <= pr_mean <= 0.88, f"Phase-2.1 test PR-AUC drift: {pr_mean:.4f}"


@pytest.mark.skipif(not BOOT.exists(), reason="Phase-2.1 evaluation not run yet")
def test_phase21_test_roc_auc_in_band():
    df = pd.read_csv(BOOT)
    row = df[df["metric_set"] == "test_calibrated"]
    assert not row.empty
    roc_mean = float(row["roc_auc__mean"].iloc[0])
    assert 0.93 <= roc_mean <= 0.96, f"Phase-2.1 test ROC-AUC drift: {roc_mean:.4f}"
