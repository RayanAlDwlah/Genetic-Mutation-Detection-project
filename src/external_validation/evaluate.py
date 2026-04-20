"""Evaluate the calibrated XGBoost baseline on an external variant table.

The evaluation protocol mirrors `src/evaluate_baseline.py`:

1. Load the trained model and the validation-fit isotonic calibrator.
2. Build the same `ColumnTransformer` used at training (one-hot `ref_aa`,
   one-hot `alt_aa`, passthrough numerics) from the persisted
   `xgboost_feature_columns.csv` manifest.
3. Score raw + calibrated probabilities.
4. Compute `roc_auc`, `pr_auc`, `brier`, per-decile calibration,
   plus 1,000-replicate bootstrap 95% CIs.
5. Report a **family-holdout slice**: only variants whose gene family is
   *not* in the training-time family set. This is the honest external
   generalization number.

Outputs
-------
`results/metrics/external_<source>_metrics.csv`      — point + CI
`results/metrics/external_<source>_predictions.parquet` — per-variant probs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from src.data_splitting import assign_gene_family
from src.evaluation import bootstrap_metrics, compute_classification_metrics


@dataclass
class ExternalEvalConfig:
    model_path: Path
    calibrator_path: Path | None
    feature_columns_csv: Path
    train_split_path: Path
    n_boot: int = 1000
    seed: int = 42


def _load_feature_transformer(
    feature_columns_csv: Path,
    train_split_path: Path,
):
    """Rebuild the training-time ColumnTransformer from the feature manifest.

    We cannot pickle the transformer because training doesn't persist it, so
    we reconstruct deterministically: numeric columns are the `num__…`
    entries and the OHE categorical columns are inferred from the `cat__…`
    entries. Fit on the original training split so categorical levels match.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    manifest = pd.read_csv(feature_columns_csv)["encoded_feature"].tolist()
    num_cols = [c.removeprefix("num__") for c in manifest if c.startswith("num__")]
    cat_prefixes = sorted(
        {c.removeprefix("cat__").rsplit("_", 1)[0] for c in manifest if c.startswith("cat__")}
    )

    train = pd.read_parquet(train_split_path)
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_prefixes),
        ],
        remainder="drop",
    )
    transformer.fit(train[num_cols + cat_prefixes])
    return transformer, num_cols, cat_prefixes


def _load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _load_calibrator(path: Path | None) -> IsotonicRegression | None:
    if path is None or not Path(path).exists():
        return None
    return joblib.load(path)


def _mark_family_holdout(ext: pd.DataFrame, train_split_path: Path) -> pd.Series:
    """Return a bool Series: True if the row's gene family does NOT appear
    in the training data (i.e. truly held-out at the paralog level)."""
    train = pd.read_parquet(train_split_path)
    train_fams = {assign_gene_family(g) for g in train["gene"].unique()}
    return ~ext["gene"].astype(str).map(assign_gene_family).isin(train_fams)


def evaluate_external(
    featurized: pd.DataFrame,
    *,
    config: ExternalEvalConfig,
    source_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Score `featurized` and emit CSV/parquet artifacts under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    transformer, num_cols, cat_cols = _load_feature_transformer(
        config.feature_columns_csv, config.train_split_path
    )
    model = _load_model(config.model_path)
    calibrator = _load_calibrator(config.calibrator_path)

    # Only keep rows that have all required raw columns.
    needed = set(num_cols + cat_cols)
    missing = needed - set(featurized.columns)
    if missing:
        raise ValueError(
            f"featurized table missing {len(missing)} training feature(s): "
            f"{sorted(missing)[:5]}…"
        )

    X = transformer.transform(featurized[num_cols + cat_cols])
    y = featurized["label"].astype(int).to_numpy()
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = calibrator.transform(p_raw) if calibrator is not None else p_raw

    # Held-out family slice.
    is_holdout = _mark_family_holdout(featurized, config.train_split_path).to_numpy()

    # Per-variant predictions.
    preds = featurized[["variant_key", "gene", "label"]].copy()
    preds["p_raw"] = p_raw
    preds["p_calibrated"] = p_cal
    preds["family_holdout"] = is_holdout
    preds.to_parquet(out_dir / f"external_{source_name}_predictions.parquet", index=False)

    # Evaluate full + holdout slices.
    rows = []
    for label, mask in [("full", np.ones_like(is_holdout)), ("family_holdout_only", is_holdout)]:
        if mask.sum() < 20 or y[mask].min() == y[mask].max():
            rows.append(
                {
                    "slice": label,
                    "n": int(mask.sum()),
                    "note": "skipped: too few rows or single-class",
                }
            )
            continue
        metrics = compute_classification_metrics(pd.Series(y[mask]), p_cal[mask], threshold=0.5)
        boot = bootstrap_metrics(
            y[mask],
            p_cal[mask],
            threshold=0.5,
            n_boot=config.n_boot,
            seed=config.seed,
        )
        rows.append(
            {
                "slice": label,
                "n": int(mask.sum()),
                "n_pos": int(y[mask].sum()),
                "roc_auc": metrics["roc_auc"],
                "roc_auc_ci_lo": boot["roc_auc"]["ci_lo"],
                "roc_auc_ci_hi": boot["roc_auc"]["ci_hi"],
                "pr_auc": metrics["pr_auc"],
                "pr_auc_ci_lo": boot["pr_auc"]["ci_lo"],
                "pr_auc_ci_hi": boot["pr_auc"]["ci_hi"],
                "brier": metrics["brier_loss"],
                "f1": metrics["f1"],
            }
        )
    out = pd.DataFrame(rows)
    out.insert(0, "source", source_name)
    out_csv = out_dir / f"external_{source_name}_metrics.csv"
    out.to_csv(out_csv, index=False)
    return {"metrics_csv": str(out_csv), "rows": rows}
