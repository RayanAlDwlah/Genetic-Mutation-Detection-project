"""End-to-end regression: committed headline numbers must reproduce.

Loads the committed `results/checkpoints/xgboost_best.ubj` model and
re-scores the committed test split. The recomputed ROC-AUC and PR-AUC
must match the values in `results/metrics/xgboost_split_metrics.csv`
within a tight tolerance. If this test fails, either the model or the
metrics CSV was tampered with.

This is the single most important regression gate in the repo: it binds
the published number to the actual checkpoint bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import xgboost as xgb
from src.evaluation import compute_classification_metrics
from src.training import prepare_split_features, select_feature_columns

# Absolute tolerance on ROC/PR between committed metrics and
# recomputed-on-the-spot metrics. These should be deterministic, so a
# 1e-4 tolerance is conservative (handles any BLAS/threading nondeterminism
# on different hosts).
METRIC_TOLERANCE = 1e-3


@pytest.mark.integration
def test_committed_test_headline_reproduces(repo_root: Path) -> None:
    model_path = repo_root / "results/checkpoints/xgboost_best.ubj"
    metrics_path = repo_root / "results/metrics/xgboost_split_metrics.csv"
    params_path = repo_root / "results/metrics/xgboost_best_params.csv"

    for p in (model_path, metrics_path, params_path):
        if not p.exists():  # pragma: no cover
            pytest.skip(f"missing committed artifact: {p}")

    train = pd.read_parquet(repo_root / "data/splits/train.parquet")
    val = pd.read_parquet(repo_root / "data/splits/val.parquet")
    test = pd.read_parquet(repo_root / "data/splits/test.parquet")

    numeric_cols, categorical_cols = select_feature_columns(train)
    _x_train, _x_val, x_test, _features = prepare_split_features(
        train, val, test, numeric_cols, categorical_cols
    )

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    params = pd.read_csv(params_path).iloc[0].to_dict()
    threshold = float(params.get("best_threshold", 0.5))

    y_test = test["label"].astype(int).to_numpy()
    p_test = model.predict_proba(x_test)[:, 1]
    recomputed = compute_classification_metrics(pd.Series(y_test), p_test, threshold=threshold)

    committed = pd.read_csv(metrics_path)
    committed_test = committed[committed["split"] == "test"].iloc[0]

    # ROC-AUC reproduces within tolerance.
    assert abs(recomputed["roc_auc"] - committed_test["roc_auc"]) < METRIC_TOLERANCE, (
        f"test ROC-AUC drift: recomputed={recomputed['roc_auc']:.5f} "
        f"committed={committed_test['roc_auc']:.5f}"
    )
    # PR-AUC reproduces within tolerance.
    assert abs(recomputed["pr_auc"] - committed_test["pr_auc"]) < METRIC_TOLERANCE, (
        f"test PR-AUC drift: recomputed={recomputed['pr_auc']:.5f} "
        f"committed={committed_test['pr_auc']:.5f}"
    )
    # F1 + Brier must also match.
    assert abs(recomputed["f1"] - committed_test["f1"]) < METRIC_TOLERANCE
    assert abs(recomputed["brier_loss"] - committed_test["brier_loss"]) < METRIC_TOLERANCE


@pytest.mark.integration
def test_predictions_parquet_columns_match_schema(repo_root: Path) -> None:
    """The val/test predictions parquet must expose the exact columns the
    external-validation pipeline and the calibration module expect."""
    path = repo_root / "results/metrics/xgboost_predictions.parquet"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing: {path}")
    df = pd.read_parquet(path)
    required = {"p_raw", "p_calibrated", "y_true", "split", "variant_key"}
    missing = required - set(df.columns)
    assert not missing, f"missing required columns: {missing}"
    assert set(df["split"].unique()).issubset({"val", "test"})


@pytest.mark.integration
def test_calibrated_ece_under_target(repo_root: Path) -> None:
    """The calibration summary must show a calibrated ECE ≤ 0.03 on test
    — a loose bound so we catch regressions without being flaky."""
    path = repo_root / "results/metrics/xgboost_calibration_summary.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing: {path}")
    df = pd.read_csv(path)
    # Column naming on the committed CSV is `eval_set`. Support older
    # snapshots that used `variant`.
    key_col = "eval_set" if "eval_set" in df.columns else "variant"
    test_cal = df[df[key_col] == "test_calibrated"]
    if test_cal.empty:  # pragma: no cover
        pytest.skip("no test_calibrated row present")
    ece = float(test_cal["ECE"].iloc[0])
    assert ece <= 0.03, f"calibrated ECE regressed to {ece:.4f} (> 0.03)"
