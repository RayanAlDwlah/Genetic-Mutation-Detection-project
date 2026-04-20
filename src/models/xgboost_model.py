"""XGBoost utilities for mutation classification.

TUNING POLICY (April 2026 rewrite):
  The previous random-search loop with a composite 0.65*ROC+0.35*PR objective
  was replaced with Optuna's TPE sampler and a pure PR-AUC objective. PR-AUC
  is more informative under class imbalance (≈24% positives in our missense-
  filtered, paralog-aware splits) and avoids the arbitrary 0.65/0.35 weights.
  MedianPruner terminates clearly underperforming trials early, letting us
  afford 40+ trials in the same wall-clock budget as the old 14-trial search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback


@dataclass(frozen=True)
class XGBTuningConfig:
    """Hyperparameter search configuration for XGBoost baseline."""

    n_trials: int = 40
    seed: int = 42
    n_estimators: int = 2500
    early_stopping_rounds: int = 80
    # Optuna MedianPruner parameters — prune a trial if its intermediate
    # AUCPR falls below the running median after `n_warmup_steps` boosting
    # rounds. Pruning saves ~30–40% wall-clock on TPE runs.
    pruner_n_startup_trials: int = 5
    pruner_n_warmup_steps: int = 200


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
        # XGBoost monitors these metrics during training. Early stopping is triggered
        # by the last metric — "aucpr" — which matches the composite selection
        # objective (0.65*AUC + 0.35*PRAUC) used in tune_xgboost().
        "eval_metric": ["auc", "aucpr"],
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
        # scale_pos_weight is fixed from class distribution — not a tunable parameter.
        "scale_pos_weight": scale_pos_weight,
    }


class _OptunaPruningCallback(TrainingCallback):
    """XGBoost callback that reports aucpr to Optuna for pruning decisions."""

    def __init__(self, trial: optuna.Trial, report_every: int = 50) -> None:
        self._trial = trial
        self._report_every = max(1, report_every)

    def after_iteration(self, model, epoch, evals_log):  # type: ignore[override]
        # evals_log is a nested dict: {dataset: {metric: [values]}}
        if epoch % self._report_every != 0:
            return False
        # Take aucpr on the first eval set (validation).
        for dataset_name, metrics in evals_log.items():
            aucpr_history = metrics.get("aucpr")
            if not aucpr_history:
                continue
            current = float(aucpr_history[-1])
            self._trial.report(current, step=epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned()
            break
        return False


def _suggest_params(trial: optuna.Trial, scale_pos_weight: float) -> dict[str, Any]:
    """Optuna parameter space tuned for missense classification on ~140K rows."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.55, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 15.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 6),
        "scale_pos_weight": scale_pos_weight,
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
    """Tune XGBoost with Optuna TPE + MedianPruner, maximizing val PR-AUC.

    Returns the best refit model, its parameters, and a trial history DataFrame
    sorted by validation PR-AUC descending. Pruned trials appear with state
    "PRUNED" and score NaN.
    """

    trial_rows: list[dict[str, Any]] = []
    best_model_box: dict[str, XGBClassifier] = {}

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, scale_pos_weight)
        trial_seed = int(np.random.default_rng(config.seed + trial.number * 997).integers(0, 2**31))
        model = build_xgboost_model(
            params,
            seed=trial_seed,
            n_estimators=config.n_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        pruning_cb = _OptunaPruningCallback(trial, report_every=50)
        model.set_params(callbacks=[pruning_cb])
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = float(roc_auc_score(y_val, val_prob))
        pr_auc = float(average_precision_score(y_val, val_prob))
        trial.set_user_attr("val_roc_auc", roc_auc)
        trial.set_user_attr("val_pr_auc", pr_auc)
        trial.set_user_attr(
            "best_iteration",
            int(getattr(model, "best_iteration", config.n_estimators - 1)),
        )
        # Stash model for the best trial so we don't retrain at the end.
        best_so_far = best_model_box.get("score", -np.inf)
        if pr_auc > best_so_far:
            best_model_box["score"] = pr_auc
            best_model_box["model"] = model
            best_model_box["params"] = params
        return pr_auc

    sampler = TPESampler(seed=config.seed, multivariate=True, n_startup_trials=8)
    pruner = MedianPruner(
        n_startup_trials=config.pruner_n_startup_trials,
        n_warmup_steps=config.pruner_n_warmup_steps,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"xgb_tune_seed{config.seed}",
    )
    study.optimize(objective, n_trials=config.n_trials, gc_after_trial=True)

    # Build history dataframe.
    for t in study.trials:
        row: dict[str, Any] = {
            "trial": t.number,
            "state": t.state.name,
            "val_pr_auc": t.value if t.value is not None else float("nan"),
            "val_roc_auc": t.user_attrs.get("val_roc_auc", float("nan")),
            "best_iteration": t.user_attrs.get("best_iteration", -1),
        }
        row.update(t.params)
        trial_rows.append(row)

    history = (
        pd.DataFrame(trial_rows)
        .sort_values("val_pr_auc", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    # Expose a legacy `score` column for downstream compatibility
    history["score"] = history["val_pr_auc"]

    if "model" not in best_model_box:
        raise RuntimeError("Optuna tuning finished with no completed trials.")
    best_model = best_model_box["model"]
    best_params = best_model_box["params"]
    return best_model, best_params, history
