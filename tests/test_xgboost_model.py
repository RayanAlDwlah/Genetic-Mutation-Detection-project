"""Unit tests for `src.models.xgboost_model`.

Covers the pure helpers (`build_xgboost_model`, `_baseline_params`,
`_sample_params`) and a tiny-data integration test for `tune_xgboost`
that runs in <30 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.xgboost_model import (
    XGBTuningConfig,
    _baseline_params,
    _sample_params,
    build_xgboost_model,
    tune_xgboost,
)


class TestBaselineAndSampleParams:
    def test_baseline_keys(self) -> None:
        p = _baseline_params(scale_pos_weight=3.0)
        expected = {
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
        }
        assert expected.issubset(p)
        assert p["scale_pos_weight"] == pytest.approx(3.0)

    def test_sample_params_within_ranges(self) -> None:
        rng = np.random.default_rng(42)
        p = _sample_params(rng, scale_pos_weight=2.5)
        assert 3 <= p["max_depth"] < 8
        assert 0.025 <= p["learning_rate"] <= 0.12
        assert 0.65 <= p["colsample_bytree"] <= 1.0
        assert p["scale_pos_weight"] == pytest.approx(2.5)
        assert p["reg_alpha"] > 0
        assert p["reg_lambda"] > 0


class TestBuildXgboostModel:
    def test_sensible_defaults(self) -> None:
        m = build_xgboost_model(
            {"max_depth": 4}, seed=42, n_estimators=100, early_stopping_rounds=10
        )
        assert m.max_depth == 4
        assert m.n_estimators == 100


class TestTuneXgboost:
    """Tiny-data integration test: must complete in <30 s."""

    @pytest.mark.slow
    def test_returns_model_params_and_history(self, rng: np.random.Generator) -> None:
        # Synthetic tabular problem: 3 informative features, 200 rows.
        n_train = 200
        n_val = 100
        X_train = pd.DataFrame(
            {
                "f0": rng.normal(size=n_train),
                "f1": rng.normal(size=n_train),
                "f2": rng.normal(size=n_train),
            }
        )
        y_train = pd.Series((X_train["f0"] + 0.5 * X_train["f1"] > 0.0).astype(int))
        X_val = pd.DataFrame(
            {
                "f0": rng.normal(size=n_val),
                "f1": rng.normal(size=n_val),
                "f2": rng.normal(size=n_val),
            }
        )
        y_val = pd.Series((X_val["f0"] + 0.5 * X_val["f1"] > 0.0).astype(int))

        config = XGBTuningConfig(
            n_trials=2,
            seed=42,
            n_estimators=50,
            early_stopping_rounds=10,
        )
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / max(pos, 1)

        model, params, history = tune_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            scale_pos_weight=spw,
        )

        # Model recovers the signal — val ROC > 0.8 on this easy problem.
        p_val = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score(y_val, p_val) > 0.8

        # History has the expected shape.
        assert isinstance(history, pd.DataFrame)
        assert {"val_roc_auc", "val_pr_auc", "score"}.issubset(history.columns)
        assert len(history) <= config.n_trials

        # Best params are a subset of the searched-over keys.
        assert "max_depth" in params
        assert "learning_rate" in params
