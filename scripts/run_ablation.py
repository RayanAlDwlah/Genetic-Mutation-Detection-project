"""Feature-group ablation (P1-1 of thesis review).

For each of the six feature groups used in Figure 3.7, retrain XGBoost with
the group *removed* and report the resulting ROC-AUC / PR-AUC / ECE on the
paralog-disjoint test split, each against the full-feature baseline. Uses
the Optuna-tuned hyperparameters from ``results/metrics/xgboost_best_params.csv``
(no per-ablation re-tuning; a single dedicated re-tune per ablation would
take hours and is out of scope for a deficiency table).

Groups (row names in the resulting table)::

  conservation      phyloP*, phastCons*, GERP++*, SiPhy    (6 columns)
  physicochemical   Grantham, BLOSUM62, hydrophobicity,     (~14 columns)
                    volume, polarity, charge, mol.\ weight,
                    pI, *_change
  aa_identity       ref_aa_{A..Y}, alt_aa_{A..Y}            (20 one-hot cols)
  allele_frequency  AF_popmax, AN, AC, log_AF               (4)
  gene_constraint   pLI, LOEUF, mis_z, oe_mis_upper, lof_z  (5)
  imputation_flags  is_imputed_*                            (~5)

Output::

  results/metrics/ablation.csv   columns: group, n_dropped, test_roc,
                                 test_pr, test_ece, d_roc, d_pr, d_ece
  results/figures/ablation.png   bar chart of delta ROC / PR per group

Usage::

  python -m scripts.run_ablation --seed 42 --n-boot 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]


FEATURE_GROUPS: dict[str, list[str]] = {
    "conservation": [
        "phyloP100way_vertebrate",
        "phyloP30way_mammalian",
        "phastCons100way_vertebrate",
        "phastCons30way_mammalian",
        "GERP++_RS",
        "GERP++_NR",
    ],
    "physicochemical": [
        "Grantham_distance",
        "BLOSUM62_score",
        "hydrophobicity_ref",
        "hydrophobicity_alt",
        "molecular_weight_ref",
        "molecular_weight_alt",
        "pI_alt",
        "volume_ref",
        "volume_alt",
        "polarity_ref",
        "polarity_alt",
        "charge_ref",
        "charge_alt",
        "polarity_change",
        "volume_change",
        "charge_change",
    ],
    "aa_identity": ["ref_aa", "alt_aa"],
    "allele_frequency": ["AF_popmax", "AN", "AC", "log_AF"],
    "gene_constraint": ["pLI", "oe_lof_upper", "mis_z", "oe_mis_upper", "lof_z"],
    "imputation_flags": [
        "is_imputed_phyloP100way_vertebrate",
        "is_imputed_phastCons100way_vertebrate",
        "is_imputed_GERP++_RS",
        "is_imputed_Grantham_distance",
        "is_imputed_BLOSUM62_score",
    ],
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=REPO / "data/splits/train.parquet")
    ap.add_argument("--val",   type=Path, default=REPO / "data/splits/val.parquet")
    ap.add_argument("--test",  type=Path, default=REPO / "data/splits/test.parquet")
    ap.add_argument(
        "--params",
        type=Path,
        default=REPO / "results/metrics/xgboost_best_params.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "results/metrics/ablation.csv",
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=REPO / "results/figures/ablation.png",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    return ap.parse_args()


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        e += (mask.sum() / len(y)) * abs(p[mask].mean() - y[mask].mean())
    return float(e)


def _paired_bootstrap_delta(y, p_full, p_abl, metric, n_boot, seed):
    rng = np.random.default_rng(seed)
    delta = metric(y, p_full) - metric(y, p_abl)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(y):
            deltas[b] = np.nan
            continue
        deltas[b] = metric(yb, p_full[idx]) - metric(yb, p_abl[idx])
    valid = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def _train_and_eval(x_tr, y_tr, x_val, y_val, x_te, y_te, params, args):
    import xgboost as xgb  # noqa: PLC0415

    # Build categorical DMatrix: we pass ref_aa / alt_aa as pandas 'category'.
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ("ref_aa", "alt_aa"):
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out

    x_tr, x_val, x_te = _prepare(x_tr), _prepare(x_val), _prepare(x_te)
    d_tr = xgb.DMatrix(x_tr, label=y_tr, enable_categorical=True)
    d_val = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    d_te = xgb.DMatrix(x_te, enable_categorical=True)

    xgb_params = {
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "gamma": float(params["gamma"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "max_delta_step": int(params["max_delta_step"]),
        "scale_pos_weight": float(params["scale_pos_weight"]),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "seed": args.seed,
    }
    booster = xgb.train(
        xgb_params,
        d_tr,
        num_boost_round=args.n_estimators,
        evals=[(d_val, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    p_val_raw = booster.predict(d_val)
    p_te_raw = booster.predict(d_te)

    # Calibrate with isotonic fit on val
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_te = iso.transform(p_te_raw)
    return p_te


def main() -> None:
    args = _parse_args()
    tr = pd.read_parquet(args.train)
    val = pd.read_parquet(args.val)
    te = pd.read_parquet(args.test)
    params = pd.read_csv(args.params).iloc[0].to_dict()

    banned = {"variant_key", "label", "review_stars", "chr", "ref", "alt",
              "gene", "pos", "ClinicalSignificance", "PhenotypeIDS",
              "is_common", "is_imputed_gnomad_constraint"}

    feature_cols = [c for c in tr.columns if c not in banned]
    print(f"full feature set: {len(feature_cols)} cols")

    y_tr = tr["label"].to_numpy(dtype=int)
    y_val = val["label"].to_numpy(dtype=int)
    y_te = te["label"].to_numpy(dtype=int)

    rows = []
    # 1. Full baseline
    p_full = _train_and_eval(
        tr[feature_cols], y_tr,
        val[feature_cols], y_val,
        te[feature_cols], y_te,
        params, args,
    )
    roc_full = float(roc_auc_score(y_te, p_full))
    pr_full = float(average_precision_score(y_te, p_full))
    ece_full = _ece(y_te, p_full)
    print(f"[full]           ROC={roc_full:.4f}  PR={pr_full:.4f}  ECE={ece_full:.4f}")
    rows.append(
        {"group": "FULL_BASELINE", "n_dropped": 0,
         "test_roc": roc_full, "test_pr": pr_full, "test_ece": ece_full,
         "d_roc": 0.0, "d_roc_lo": 0.0, "d_roc_hi": 0.0,
         "d_pr": 0.0,  "d_pr_lo":  0.0, "d_pr_hi":  0.0,
         "d_ece": 0.0}
    )

    # 2. Ablations
    for group, cols in FEATURE_GROUPS.items():
        keep = [c for c in feature_cols if c not in cols]
        n_dropped = len(feature_cols) - len(keep)
        p_abl = _train_and_eval(
            tr[keep], y_tr,
            val[keep], y_val,
            te[keep], y_te,
            params, args,
        )
        roc_abl = float(roc_auc_score(y_te, p_abl))
        pr_abl = float(average_precision_score(y_te, p_abl))
        ece_abl = _ece(y_te, p_abl)

        d_roc, dr_lo, dr_hi = _paired_bootstrap_delta(
            y_te, p_full, p_abl, roc_auc_score, args.n_boot, args.seed,
        )
        d_pr, dp_lo, dp_hi = _paired_bootstrap_delta(
            y_te, p_full, p_abl, average_precision_score, args.n_boot, args.seed,
        )
        print(
            f"[{group:18s}] dropped={n_dropped:2d}  ROC={roc_abl:.4f}  "
            f"PR={pr_abl:.4f}  d_ROC={d_roc:+.4f}"
        )
        rows.append({
            "group": group,
            "n_dropped": n_dropped,
            "test_roc": roc_abl,
            "test_pr": pr_abl,
            "test_ece": ece_abl,
            "d_roc": d_roc, "d_roc_lo": dr_lo, "d_roc_hi": dr_hi,
            "d_pr": d_pr,   "d_pr_lo":  dp_lo, "d_pr_hi":  dp_hi,
            "d_ece": ece_abl - ece_full,
        })

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    # Figure: bar chart of delta metrics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abl = out[out["group"] != "FULL_BASELINE"].sort_values("d_roc")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].barh(abl["group"], abl["d_roc"], xerr=[abl["d_roc"] - abl["d_roc_lo"],
                                                  abl["d_roc_hi"] - abl["d_roc"]],
               color="#045531", alpha=0.85)
    ax[0].axvline(0, color="gray", lw=0.7)
    ax[0].set_xlabel("$\\Delta$ ROC-AUC (full $-$ ablated)")
    ax[0].set_title("ROC-AUC ablation (paralog-disjoint test)")

    ax[1].barh(abl["group"], abl["d_pr"], xerr=[abl["d_pr"] - abl["d_pr_lo"],
                                                  abl["d_pr_hi"] - abl["d_pr"]],
               color="#C9A959", alpha=0.85)
    ax[1].axvline(0, color="gray", lw=0.7)
    ax[1].set_xlabel("$\\Delta$ PR-AUC (full $-$ ablated)")
    ax[1].set_title("PR-AUC ablation")

    fig.suptitle("Feature-group ablation on the paralog-disjoint test split")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
