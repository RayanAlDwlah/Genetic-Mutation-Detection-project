#!/usr/bin/env python3
"""SHAP interpretability + error analysis for the XGBoost baseline.

Outputs
-------
* ``results/metrics/shap_values_test.parquet`` — TreeSHAP values per
  (variant, feature) for a stratified 2 000-variant test sample.
* ``results/figures/shap_summary.png`` — classic beeswarm (top 20 features).
* ``results/figures/shap_bar.png`` — mean-|SHAP| bar chart.
* ``results/figures/shap_dependence_top3.png`` — 3-panel dependence plots
  for the top-3 features.
* ``results/metrics/confident_errors.csv`` — variants where
  ``|p_calibrated − y_true| > 0.5`` (the model was confidently wrong).
  Includes gene, ClinVar review_stars, true vs predicted class, and the
  three SHAP features that pushed the prediction toward the wrong side.

The 2 000-row sample is stratified by label so we don't bias the
mean-|SHAP| ordering toward the benign majority class.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from src.training import prepare_split_features, select_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "results/checkpoints/xgboost_best.ubj"
PRED_PATH = REPO / "results/metrics/xgboost_predictions.parquet"

OUT_SHAP = REPO / "results/metrics/shap_values_test.parquet"
OUT_CONFIDENT_ERR = REPO / "results/metrics/confident_errors.csv"
OUT_SUMMARY = REPO / "results/figures/shap_summary.png"
OUT_BAR = REPO / "results/figures/shap_bar.png"
OUT_DEPENDENCE = REPO / "results/figures/shap_dependence_top3.png"

N_SAMPLE = 2_000
RNG_SEED = 42


def _load_sample(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Return (train, val, test_sample, X_test_sample, feature_names).

    `prepare_split_features` is reused so the feature matrix matches the
    committed model exactly — no risk of column re-ordering or
    encoding drift.
    """
    train = pd.read_parquet(REPO / "data/splits/train.parquet")
    val = pd.read_parquet(REPO / "data/splits/val.parquet")
    test = pd.read_parquet(REPO / "data/splits/test.parquet")

    # Stratified sampling by label.
    pos = test[test["label"] == 1]
    neg = test[test["label"] == 0]
    n_pos = min(len(pos), N_SAMPLE // 2)
    n_neg = min(len(neg), N_SAMPLE - n_pos)
    sample_idx = pd.concat(
        [pos.sample(n_pos, random_state=RNG_SEED), neg.sample(n_neg, random_state=RNG_SEED)]
    ).index
    test_sample = test.loc[sample_idx].reset_index(drop=True)

    numeric_cols, categorical_cols = select_feature_columns(train)
    _x_train, _x_val, x_test_all, feature_names = prepare_split_features(
        train, val, test, numeric_cols, categorical_cols
    )

    # Map sample indices back to rows in the encoded test matrix.
    original_idx = test.index.get_indexer(sample_idx)
    x_sample = x_test_all[original_idx]
    if hasattr(x_sample, "toarray"):
        x_sample = x_sample.toarray()
    return train, val, test_sample, np.asarray(x_sample, dtype=np.float32), feature_names


def _render_summary_plot(
    shap_vals: np.ndarray, x_sample: np.ndarray, feature_names: list[str]
) -> None:
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_vals,
        features=x_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    fig.tight_layout()
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SUMMARY, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_bar_plot(shap_vals: np.ndarray, feature_names: list[str]) -> None:
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    labels = [feature_names[i] for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], values[::-1], color="#2c3e50")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Top-20 features by mean |SHAP|", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_BAR, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_dependence_plots(
    shap_vals: np.ndarray, x_sample: np.ndarray, feature_names: list[str]
) -> None:
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top3 = np.argsort(mean_abs)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, idx in zip(axes, top3, strict=True):
        name = feature_names[idx]
        x = x_sample[:, idx]
        s = shap_vals[:, idx]
        ax.scatter(x, s, s=8, alpha=0.5, color="#2c3e50")
        ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value (log-odds contribution)")
        ax.set_title(name, fontweight="bold")
        ax.grid(alpha=0.25)
    fig.suptitle("SHAP dependence — top 3 features", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DEPENDENCE, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _identify_confident_errors(
    test_sample: pd.DataFrame,
    shap_vals: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Confident errors = |p_calibrated − y_true| > 0.5. For each row we
    list the 3 features with the largest-magnitude SHAP contribution
    so the analyst can see *why* the model was wrong."""
    preds = pd.read_parquet(PRED_PATH)
    preds = preds[preds["split"] == "test"][["variant_key", "p_raw", "p_calibrated", "y_true"]]

    merged = test_sample.merge(preds, on="variant_key", how="left")
    # Row index in shap_vals aligns with test_sample.
    abs_shap = np.abs(shap_vals)
    top3_idx = np.argsort(-abs_shap, axis=1)[:, :3]

    errs: list[dict[str, object]] = []
    for i, row in merged.iterrows():
        p = row.get("p_calibrated")
        y = row.get("y_true")
        if pd.isna(p) or pd.isna(y):
            continue
        if abs(p - y) <= 0.5:
            continue
        top3 = [
            {"feature": feature_names[j], "shap_value": float(shap_vals[i, j])} for j in top3_idx[i]
        ]
        errs.append(
            {
                "variant_key": row["variant_key"],
                "gene": row.get("gene"),
                "review_stars": row.get("review_stars"),
                "y_true": int(y),
                "p_raw": float(row["p_raw"]),
                "p_calibrated": float(p),
                "error_type": "false_positive" if y == 0 else "false_negative",
                "top1_feature": top3[0]["feature"],
                "top1_shap": top3[0]["shap_value"],
                "top2_feature": top3[1]["feature"],
                "top2_shap": top3[1]["shap_value"],
                "top3_feature": top3[2]["feature"],
                "top3_shap": top3[2]["shap_value"],
            }
        )
    return pd.DataFrame(errs)


def main() -> None:
    import shap

    rng = np.random.default_rng(RNG_SEED)
    print("[shap] loading data + model…")
    _train, _val, test_sample, x_sample, feature_names = _load_sample(rng)
    print(f"[shap] test sample: {len(test_sample):,} rows, {x_sample.shape[1]} encoded features")

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print(f"[shap] running TreeSHAP on {len(test_sample):,} rows…")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_sample)

    # shap_values returns (n_samples, n_features) for binary classification.
    print(f"[shap] SHAP values shape: {shap_vals.shape}")

    # Persist values for downstream analysis.
    OUT_SHAP.parent.mkdir(parents=True, exist_ok=True)
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.insert(0, "variant_key", test_sample["variant_key"].to_numpy())
    shap_df.to_parquet(OUT_SHAP, index=False)
    print(f"[shap] wrote {OUT_SHAP.relative_to(REPO)}")

    # Figures.
    print("[shap] rendering figures…")
    _render_summary_plot(shap_vals, x_sample, feature_names)
    _render_bar_plot(shap_vals, feature_names)
    _render_dependence_plots(shap_vals, x_sample, feature_names)
    print(f"[shap] wrote 3 figures to {OUT_SUMMARY.parent.relative_to(REPO)}/")

    # Error analysis.
    print("[shap] analyzing confident errors…")
    errs = _identify_confident_errors(test_sample, shap_vals, feature_names)
    errs.to_csv(OUT_CONFIDENT_ERR, index=False)
    print(
        f"[shap] wrote {OUT_CONFIDENT_ERR.relative_to(REPO)}: "
        f"{len(errs):,} confident errors "
        f"({(errs['error_type'] == 'false_positive').sum()} FP, "
        f"{(errs['error_type'] == 'false_negative').sum()} FN)"
    )

    # Top-10 features summary.
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    print("\nTop-10 features by mean(|SHAP|):")
    for r, i in enumerate(order, 1):
        print(f"  {r:2}. {feature_names[i]:42s}  {mean_abs[i]:.4f}")


if __name__ == "__main__":
    main()
