"""Learning curve (P1-2 of thesis review).

Retrain XGBoost with the committed Optuna-tuned hyperparameters at a
subsampled fraction ``f`` of the training split for every
``f ∈ {0.10, 0.25, 0.50, 0.75, 1.00}``. Report val and test ROC-AUC
for each operating point. Answers the "does more data still help?"
question that a reviewer will ask of any single-operating-point number.

Subsampling is done at the **gene-family** level so the paralog-
disjoint invariant is preserved:

  * Compute the gene-family of every training row.
  * Draw a fraction of the distinct families uniformly.
  * Retain all rows whose family is in the drawn set.

This guarantees that each subsample is internally family-disjoint from
val/test exactly the same way the full training set is.

Output::

  results/metrics/learning_curve.csv         — one row per (fraction) with
                                               metrics on val and test
  results/figures/learning_curve.png         — ROC vs log(n_train)

Usage::

  python -m scripts.run_learning_curve --seed 42
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]


# Same assignment used in src/data_splitting.py (inline here so this script
# doesn't have to re-import the private regex constants).
_FAMILY_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (re.compile(r"^(KRT[A-Z]*)\d+"), "KRT"),
    (re.compile(r"^(CDH)\d+"), "CDH"),
    (re.compile(r"^(HLA)-"), "HLA"),
    (re.compile(r"^(ZNF)\d+"), "ZNF"),
    (re.compile(r"^(OR[0-9]+[A-Z]+)\d*"), None),
    (re.compile(r"^(RP[LS])\d+"), "RP"),
    (re.compile(r"^(MT-)"), "MT"),
    (re.compile(r"^(SLC)\d+"), "SLC"),
    (re.compile(r"^(TMEM)\d+"), "TMEM"),
    (re.compile(r"^(COL)\d+"), "COL"),
    (re.compile(r"^(USH)\d+"), "USH"),
    (re.compile(r"^(DNAH)\d+"), "DNAH"),
    (re.compile(r"^(KCN[A-Z]+)\d+"), None),
    (re.compile(r"^(MYH)\d+"), "MYH"),
    (re.compile(r"^(MUC)\d+"), "MUC"),
    (re.compile(r"^(ABCA)\d+"), "ABCA"),
]
_TRAILING_DIGITS = re.compile(r"\d+$")


def _assign_family(gene: str) -> str:
    if gene is None:
        return ""
    g = str(gene).upper()
    for pattern, fam in _FAMILY_PATTERNS:
        m = pattern.match(g)
        if m:
            return fam if fam is not None else m.group(1)
    stripped = _TRAILING_DIGITS.sub("", g)
    return stripped or g


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
        default=REPO / "results/metrics/learning_curve.csv",
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=REPO / "results/figures/learning_curve.png",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    return ap.parse_args()


def _subsample_family_aware(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return train.copy()
    fams = train["gene"].astype(str).map(_assign_family)
    unique = fams.unique()
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(unique) * fraction)))
    keep = set(rng.choice(unique, size=n_keep, replace=False))
    mask = fams.isin(keep).to_numpy()
    return train[mask].copy()


def _train_and_eval(x_tr, y_tr, x_val, y_val, x_te, y_te, params, args):
    import xgboost as xgb  # noqa: PLC0415

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
        xgb_params, d_tr,
        num_boost_round=args.n_estimators,
        evals=[(d_val, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    p_val_raw = booster.predict(d_val)
    p_te_raw = booster.predict(d_te)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_val = iso.transform(p_val_raw)
    p_te = iso.transform(p_te_raw)
    return p_val, p_te


def main() -> None:
    args = _parse_args()
    tr_full = pd.read_parquet(args.train)
    val = pd.read_parquet(args.val)
    te = pd.read_parquet(args.test)
    params = pd.read_csv(args.params).iloc[0].to_dict()

    banned = {"variant_key", "label", "review_stars", "chr", "ref", "alt",
              "gene", "pos", "ClinicalSignificance", "PhenotypeIDS",
              "is_common", "is_imputed_gnomad_constraint"}
    feature_cols = [c for c in tr_full.columns if c not in banned]

    y_val = val["label"].to_numpy(dtype=int)
    y_te = te["label"].to_numpy(dtype=int)

    rows = []
    for f in args.fractions:
        tr = _subsample_family_aware(tr_full, f, args.seed)
        y_tr = tr["label"].to_numpy(dtype=int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            print(f"[skip f={f}] degenerate single-class subsample")
            continue
        p_val, p_te = _train_and_eval(
            tr[feature_cols], y_tr,
            val[feature_cols], y_val,
            te[feature_cols], y_te,
            params, args,
        )
        row = {
            "fraction": float(f),
            "n_train": int(len(tr)),
            "n_train_pos": int(y_tr.sum()),
            "val_roc": float(roc_auc_score(y_val, p_val)),
            "val_pr":  float(average_precision_score(y_val, p_val)),
            "test_roc": float(roc_auc_score(y_te, p_te)),
            "test_pr":  float(average_precision_score(y_te, p_te)),
        }
        print(
            f"[f={f:.2f}] n_train={row['n_train']:,} "
            f"val_roc={row['val_roc']:.4f}  test_roc={row['test_roc']:.4f}  "
            f"test_pr={row['test_pr']:.4f}"
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(out["n_train"], out["test_roc"], "o-", lw=2, color="#045531", label="test ROC-AUC")
    ax.plot(out["n_train"], out["val_roc"], "s--", lw=1.5, color="#C9A959", label="val ROC-AUC")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training variants ($\\log$ scale)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning curve — paralog-aware subsampling")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0.80, 1.0)

    # Annotate each point with n and fraction.
    for _, r in out.iterrows():
        ax.annotate(
            f"f={r['fraction']:.2f}\n n={int(r['n_train']):,}",
            (r["n_train"], r["test_roc"]),
            textcoords="offset points", xytext=(5, -14),
            fontsize=7, color="#045531",
        )

    # Approximate slope at 100% (log-linear fit on last two points).
    if len(out) >= 2:
        tail = out.tail(2)
        dy = tail.iloc[-1]["test_roc"] - tail.iloc[0]["test_roc"]
        dx = np.log10(tail.iloc[-1]["n_train"]) - np.log10(tail.iloc[0]["n_train"])
        slope = dy / dx if dx != 0 else float("nan")
        ax.text(
            0.03, 0.95,
            f"slope @ tail: $\\Delta$ROC / $\\Delta\\log_{{10}} n$ $=$ {slope:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 3, "edgecolor": "#045531"},
        )

    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
