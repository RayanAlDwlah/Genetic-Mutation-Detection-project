#!/usr/bin/env python3
"""Train tuned XGBoost baseline and export evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.evaluation import compute_classification_metrics, select_best_threshold
from src.models.xgboost_model import XGBTuningConfig, tune_xgboost
from src.output import echo
from src.utils import require_file, resolve_path


DEFAULT_TRAIN_PATH = "data/splits/train.parquet"
DEFAULT_VAL_PATH = "data/splits/val.parquet"
DEFAULT_TEST_PATH = "data/splits/test.parquet"
DEFAULT_MODEL_OUT = "results/checkpoints/xgboost_best.ubj"
DEFAULT_METRICS_OUT = "results/metrics/xgboost_split_metrics.csv"
DEFAULT_TUNING_OUT = "results/metrics/xgboost_tuning_history.csv"
DEFAULT_THRESHOLD_CURVE_OUT = "results/metrics/xgboost_val_threshold_curve.csv"
DEFAULT_FEATURES_OUT = "results/metrics/xgboost_feature_columns.csv"
DEFAULT_PARAMS_OUT = "results/metrics/xgboost_best_params.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned XGBoost mutation classifier")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Train parquet path")
    parser.add_argument("--val", default=DEFAULT_VAL_PATH, help="Validation parquet path")
    parser.add_argument("--test", default=DEFAULT_TEST_PATH, help="Test parquet path")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Output model path (.ubj)")
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT, help="Output metrics CSV")
    parser.add_argument("--tuning-out", default=DEFAULT_TUNING_OUT, help="Output tuning history CSV")
    parser.add_argument(
        "--threshold-curve-out",
        default=DEFAULT_THRESHOLD_CURVE_OUT,
        help="Output validation threshold curve CSV",
    )
    parser.add_argument("--features-out", default=DEFAULT_FEATURES_OUT, help="Output selected features CSV")
    parser.add_argument("--params-out", default=DEFAULT_PARAMS_OUT, help="Output best params CSV")
    parser.add_argument("--trials", type=int, default=14, help="Hyperparameter search trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select numeric and categorical columns while excluding leakage/IDs.

    Important:
    - We exclude `gene` intentionally because split is gene-level and gene identity
      is not intended as a direct model input feature.
    """

    excluded = {
        "label",
        "variant_key",
        "ClinicalSignificance",  # near-label leakage
        "PhenotypeIDS",  # high-cardinality raw IDs text
        "gene",  # split grouping key
    }

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        dtype = series.dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        elif (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found for model training")

    return numeric_cols, categorical_cols


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder with compatibility across sklearn versions."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            min_frequency=25,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def prepare_split_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    """Apply train-fitted preprocessing and categorical encoding."""

    selected_cols = [*numeric_cols, *categorical_cols]
    train_x = train_df[selected_cols].copy()
    val_x = val_df[selected_cols].copy()
    test_x = test_df[selected_cols].copy()

    if numeric_cols:
        for x in (train_x, val_x, test_x):
            x[numeric_cols] = x[numeric_cols].replace([np.inf, -np.inf], np.nan)

        medians = train_x[numeric_cols].median(numeric_only=True)
        train_x[numeric_cols] = train_x[numeric_cols].fillna(medians)
        val_x[numeric_cols] = val_x[numeric_cols].fillna(medians)
        test_x[numeric_cols] = test_x[numeric_cols].fillna(medians)

        bool_numeric = [c for c in numeric_cols if pd.api.types.is_bool_dtype(train_x[c])]
        for c in bool_numeric:
            train_x[c] = train_x[c].astype(np.int8)
            val_x[c] = val_x[c].astype(np.int8)
            test_x[c] = test_x[c].astype(np.int8)

        for x in (train_x, val_x, test_x):
            float_cols = [c for c in numeric_cols if str(x[c].dtype) == "float64"]
            if float_cols:
                x[float_cols] = x[float_cols].astype(np.float32)

    if categorical_cols:
        for x in (train_x, val_x, test_x):
            for c in categorical_cols:
                x[c] = x[c].fillna("MISSING").astype(str)

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _make_one_hot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    x_train_enc = preprocessor.fit_transform(train_x)
    x_val_enc = preprocessor.transform(val_x)
    x_test_enc = preprocessor.transform(test_x)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(x_train_enc.shape[1])]

    return x_train_enc, x_val_enc, x_test_enc, feature_names


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    train_path = resolve_path(repo_root, args.train)
    val_path = resolve_path(repo_root, args.val)
    test_path = resolve_path(repo_root, args.test)

    model_out = resolve_path(repo_root, args.model_out)
    metrics_out = resolve_path(repo_root, args.metrics_out)
    tuning_out = resolve_path(repo_root, args.tuning_out)
    threshold_curve_out = resolve_path(repo_root, args.threshold_curve_out)
    features_out = resolve_path(repo_root, args.features_out)
    params_out = resolve_path(repo_root, args.params_out)

    for p in (train_path, val_path, test_path):
        require_file(p)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    if "label" not in train_df.columns:
        raise ValueError("Training data must contain 'label' column")

    numeric_cols, categorical_cols = select_feature_columns(train_df)

    x_train, x_val, x_test, encoded_feature_names = prepare_split_features(
        train_df,
        val_df,
        test_df,
        numeric_cols,
        categorical_cols,
    )

    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    tuning_config = XGBTuningConfig(
        n_trials=max(2, int(args.trials)),
        seed=int(args.seed),
        n_estimators=2500,
        early_stopping_rounds=80,
    )

    best_model, best_params, history = tune_xgboost(
        x_train,
        y_train,
        x_val,
        y_val,
        config=tuning_config,
        scale_pos_weight=scale_pos_weight,
    )

    val_prob = best_model.predict_proba(x_val)[:, 1]
    best_threshold, threshold_curve = select_best_threshold(y_val, val_prob)

    split_rows: list[dict[str, float | int | str]] = []
    for split_name, x, y in (
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ):
        prob = best_model.predict_proba(x)[:, 1]
        row = compute_classification_metrics(y, prob, threshold=best_threshold)
        row["split"] = split_name
        split_rows.append(row)

    metrics_df = pd.DataFrame(split_rows)
    cols = ["split"] + [c for c in metrics_df.columns if c != "split"]
    metrics_df = metrics_df[cols]

    for p in (model_out, metrics_out, tuning_out, threshold_curve_out, features_out, params_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    best_model.save_model(str(model_out))
    history.to_csv(tuning_out, index=False)
    threshold_curve.to_csv(threshold_curve_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)
    pd.DataFrame({"encoded_feature": encoded_feature_names}).to_csv(features_out, index=False)
    pd.DataFrame(
        [
            {
                **best_params,
                "best_threshold": best_threshold,
                "selected_trial_score": float(history.iloc[0]["score"]),
                "selected_trial_roc_auc": float(history.iloc[0]["val_roc_auc"]),
                "selected_trial_pr_auc": float(history.iloc[0]["val_pr_auc"]),
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
                "n_encoded_features": len(encoded_feature_names),
                "categorical_columns": "|".join(categorical_cols),
            }
        ]
    ).to_csv(params_out, index=False)

    echo("Training complete")


if __name__ == "__main__":
    main()
