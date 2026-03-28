"""XGBoost utilities for mutation classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True)
class XGBTuningConfig:
    """Hyperparameter search configuration for XGBoost baseline."""

    n_trials: int = 14
    seed: int = 42
    n_estimators: int = 2500
    early_stopping_rounds: int = 80


def build_xgboost_model(
    params: dict[str, Any],
    *,
    seed: int,
    n_estimators: int,
    early_stopping_rounds: int,
) -> XGBClassifier:
    """Create an XGBoost classifier using strong tabular defaults."""

    base_params: dict[str, Any] = {
        "objective": "binary:logistic",
        # XGBoost monitors all three metrics during training. Early stopping is
        # triggered by the last metric in the list — here "logloss" — because
        # calibrated probabilities matter as much as ranking quality.
        "eval_metric": ["auc", "aucpr", "logloss"],
        # "hist" builds approximate histograms for fast split-finding (GPU/CPU).
        "tree_method": "hist",
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": seed,
        "n_jobs": -1,
    }
    base_params.update(params)
    return XGBClassifier(**base_params)


def _baseline_params(scale_pos_weight: float) -> dict[str, Any]:
    """Conservative baseline parameter set used as trial 0.

    Values are chosen from published XGBoost recommendations for tabular
    biomedical data: shallow trees (max_depth=5) to limit overfitting,
    moderate regularization (reg_lambda=2), and light subsampling.
    """
    return {
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "gamma": 0.2,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
        "scale_pos_weight": scale_pos_weight,
    }


def _sample_params(rng: np.random.Generator, scale_pos_weight: float) -> dict[str, Any]:
    """Sample one hyperparameter set from tuned ranges."""

    return {
        "max_depth": int(rng.integers(3, 8)),
        "learning_rate": float(rng.uniform(0.025, 0.12)),
        "min_child_weight": int(rng.integers(1, 9)),
        "subsample": float(rng.uniform(0.72, 1.0)),
        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
        "gamma": float(rng.uniform(0.0, 2.0)),
        "reg_alpha": float(10 ** rng.uniform(-4, 0.5)),
        "reg_lambda": float(10 ** rng.uniform(-0.3, 1.0)),
        "max_delta_step": int(rng.integers(0, 4)),
        "scale_pos_weight": float(rng.uniform(max(0.7, scale_pos_weight * 0.8), scale_pos_weight * 1.2 + 1e-8)),
    }


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    config: XGBTuningConfig,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, dict[str, Any], pd.DataFrame]:
    """Run randomized tuning and return best model, params, and history."""

    rng = np.random.default_rng(config.seed)

    best_model: XGBClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    trial_rows: list[dict[str, Any]] = []

    for trial_idx in range(config.n_trials):
        params = (
            _baseline_params(scale_pos_weight)
            if trial_idx == 0
            else _sample_params(rng, scale_pos_weight)
        )

        model = build_xgboost_model(
            params,
            seed=config.seed + trial_idx,
            n_estimators=config.n_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = float(roc_auc_score(y_val, val_prob))
        pr_auc = float(average_precision_score(y_val, val_prob))

        # Weighted objective: AUROC dominates, AUPRC refines ranking quality.
        score = 0.65 * roc_auc + 0.35 * pr_auc

        trial_info: dict[str, Any] = {
            "trial": trial_idx,
            "score": score,
            "val_roc_auc": roc_auc,
            "val_pr_auc": pr_auc,
            "best_iteration": int(getattr(model, "best_iteration", config.n_estimators - 1)),
            **params,
        }
        trial_rows.append(trial_info)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("XGBoost tuning failed to produce a model")

    history = pd.DataFrame(trial_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_model, best_params, history
