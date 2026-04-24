# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-1 Step 5).
"""Regression guard for the pre-constraint denovo-db scoring.

Catches silent drift of the pre-constraint baseline that underpins the
paired-bootstrap test in scripts/paired_bootstrap_denovo.py. Pins the
holdout ROC-AUC to the Table 5.5 pre-constraint number (0.487) with a
tolerance of 0.02 on either side.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PRE_PRED = REPO / "results/metrics/denovo_predictions_pre_constraint.parquet"


@pytest.mark.skipif(not PRE_PRED.exists(),
                    reason="pre-constraint predictions not generated yet")
def test_pre_constraint_holdout_rocauc():
    df = pd.read_parquet(PRE_PRED)
    holdout = df[df["slice"] == "holdout"]
    assert len(holdout) == 201, f"expected 201 holdout rows, got {len(holdout)}"
    auc = roc_auc_score(holdout["y_true"], holdout["p_pred"])
    assert 0.467 <= auc <= 0.507, (
        f"Pre-constraint holdout ROC-AUC drift: {auc:.4f} "
        f"(target 0.487 +/- 0.02)"
    )
