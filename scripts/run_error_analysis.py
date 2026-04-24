"""Error analysis stratified by paralog family and amino-acid class
(P1-3 of thesis review).

Produces three new figures + one CSV:

  1. ``results/figures/error_by_family.png`` — bar chart of FN rate, FP
     rate, and ROC-AUC for each of the top-10 paralog families by n in
     the test split.
  2. ``results/figures/error_by_aa_class.png`` — 6x6 heat-map of FN
     rate by (ref_aa_class, alt_aa_class). Six classes: polar,
     non-polar, positive, negative, aromatic, special.
  3. ``results/figures/confident_errors_by_phyloP.png`` — stacked bar
     chart of confident-error counts per phyloP100way quintile
     (tests the thesis hypothesis that confident FNs cluster at low
     conservation).
  4. ``results/metrics/error_analysis.csv`` — tidy long-format table of
     every (stratum, metric) used in the figures.

Usage::

  python -m scripts.run_error_analysis --seed 42
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]


# --- AA classification (Wikipedia: amino-acid properties, standard 20) ---
AA_CLASS = {
    "A": "non-polar", "V": "non-polar", "L": "non-polar",
    "I": "non-polar", "M": "non-polar", "P": "non-polar",
    "S": "polar", "T": "polar", "N": "polar",
    "Q": "polar", "C": "polar",
    "K": "positive", "R": "positive", "H": "positive",
    "D": "negative", "E": "negative",
    "F": "aromatic", "Y": "aromatic", "W": "aromatic",
    "G": "special",
}
AA_CLASSES = ["polar", "non-polar", "positive", "negative", "aromatic", "special"]


# Same family grouping as scripts/run_learning_curve.py
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
        default=REPO / "results/metrics/error_analysis.csv",
    )
    ap.add_argument(
        "--out-family",
        type=Path,
        default=REPO / "results/figures/error_by_family.png",
    )
    ap.add_argument(
        "--out-aa",
        type=Path,
        default=REPO / "results/figures/error_by_aa_class.png",
    )
    ap.add_argument(
        "--out-phylop",
        type=Path,
        default=REPO / "results/figures/confident_errors_by_phyloP.png",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.57)
    return ap.parse_args()


def _train_predict(tr, val, te, feature_cols, params, args):
    import xgboost as xgb  # noqa: PLC0415

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ("ref_aa", "alt_aa"):
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out

    x_tr = _prepare(tr[feature_cols]); y_tr = tr["label"].to_numpy(dtype=int)
    x_val = _prepare(val[feature_cols]); y_val = val["label"].to_numpy(dtype=int)
    x_te = _prepare(te[feature_cols])

    d_tr = xgb.DMatrix(x_tr, label=y_tr, enable_categorical=True)
    d_val = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    d_te = xgb.DMatrix(x_te, enable_categorical=True)

    xp = {
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
        xp, d_tr, num_boost_round=args.n_estimators,
        evals=[(d_val, "val")], early_stopping_rounds=30, verbose_eval=False,
    )
    p_val = booster.predict(d_val)
    p_te = booster.predict(d_te)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso.transform(p_te)


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

    print("training + predicting on test ...")
    p_te = _train_predict(tr, val, te, feature_cols, params, args)

    te = te.copy()
    te["p_calibrated"] = p_te
    te["y_pred"] = (p_te >= args.threshold).astype(int)
    te["family"] = te["gene"].astype(str).map(_assign_family)
    te["ref_class"] = te["ref_aa"].map(AA_CLASS)
    te["alt_class"] = te["alt_aa"].map(AA_CLASS)
    te["is_fn"] = ((te["label"] == 1) & (te["y_pred"] == 0)).astype(int)
    te["is_fp"] = ((te["label"] == 0) & (te["y_pred"] == 1)).astype(int)
    te["confident_err"] = (abs(te["p_calibrated"] - te["label"]) > 0.5).astype(int)

    # ---------- 1. Top-10 families ----------
    top_fams = te["family"].value_counts().head(10).index.tolist()
    fam_rows = []
    for fam in top_fams:
        sub = te[te["family"] == fam]
        try:
            roc = roc_auc_score(sub["label"], sub["p_calibrated"])
        except ValueError:
            roc = float("nan")
        fam_rows.append(
            {
                "stratum_type": "family",
                "stratum": fam,
                "n": int(len(sub)),
                "n_pos": int(sub["label"].sum()),
                "roc": float(roc),
                "fn_rate": float(sub["is_fn"].sum() / max(sub["label"].sum(), 1)),
                "fp_rate": float(sub["is_fp"].sum() / max((sub["label"] == 0).sum(), 1)),
            }
        )
    fam_df = pd.DataFrame(fam_rows)

    # ---------- 2. 6x6 AA class heat-map of FN rate ----------
    pos = te[te["label"] == 1]
    fn_heat = pd.DataFrame(
        index=AA_CLASSES, columns=AA_CLASSES, dtype=float,
    ).fillna(0.0)
    count_heat = pd.DataFrame(
        index=AA_CLASSES, columns=AA_CLASSES, dtype=int,
    ).fillna(0)
    for rc in AA_CLASSES:
        for ac in AA_CLASSES:
            sub = pos[(pos["ref_class"] == rc) & (pos["alt_class"] == ac)]
            count_heat.loc[rc, ac] = int(len(sub))
            fn_heat.loc[rc, ac] = float(sub["is_fn"].mean()) if len(sub) > 0 else np.nan

    aa_rows = []
    for rc in AA_CLASSES:
        for ac in AA_CLASSES:
            aa_rows.append(
                {
                    "stratum_type": "aa_class",
                    "stratum": f"{rc}->{ac}",
                    "n": int(count_heat.loc[rc, ac]),
                    "n_pos": int(count_heat.loc[rc, ac]),
                    "roc": float("nan"),
                    "fn_rate": float(fn_heat.loc[rc, ac]) if not np.isnan(fn_heat.loc[rc, ac]) else float("nan"),
                    "fp_rate": float("nan"),
                }
            )

    # ---------- 3. Confident errors by phyloP quintile ----------
    quint_edges = te["phyloP100way_vertebrate"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
    te["phyloP_q"] = pd.cut(
        te["phyloP100way_vertebrate"], bins=quint_edges, include_lowest=True,
        labels=[f"Q{i+1}" for i in range(5)],
    )
    conf = te[te["confident_err"] == 1]
    q_rows = []
    quint_summary = []
    for q in [f"Q{i+1}" for i in range(5)]:
        qsub = te[te["phyloP_q"] == q]
        qconf = conf[conf["phyloP_q"] == q]
        fn_count = int((qconf["label"] == 1).sum())
        fp_count = int((qconf["label"] == 0).sum())
        q_rows.append({
            "stratum_type": "phyloP_quintile",
            "stratum": q,
            "n": int(len(qsub)),
            "n_pos": int(qsub["label"].sum()),
            "roc": float("nan"),
            "fn_rate": fn_count / max(qsub["label"].sum(), 1),
            "fp_rate": fp_count / max((qsub["label"] == 0).sum(), 1),
        })
        quint_summary.append({"quintile": q, "FN": fn_count, "FP": fp_count})

    # ---------- Save CSV ----------
    out = pd.DataFrame(fam_rows + aa_rows + q_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    # ---------- Figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1 — top-10 families, triple-bar
    fam_df = fam_df.sort_values("n", ascending=True).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    y = np.arange(len(fam_df))
    ax1.barh(y - 0.25, fam_df["fn_rate"], height=0.23, color="#C9463D", label="FN rate", alpha=0.85)
    ax1.barh(y,         fam_df["fp_rate"], height=0.23, color="#E4A700", label="FP rate", alpha=0.85)
    ax1.barh(y + 0.25,  fam_df["roc"],     height=0.23, color="#045531", label="ROC-AUC", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(fam_df["stratum"].tolist())
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("rate / AUC")
    ax1.set_title("Error stratification by paralog family (top-10 by n in test)")
    ax1.legend(loc="lower right")
    for i, r in fam_df.iterrows():
        ax1.text(1.01, i, f"n={int(r['n']):,}", va="center", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(args.out_family, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_family}")

    # Fig 2 — 6x6 heat-map
    fig, ax2 = plt.subplots(figsize=(6.5, 5.5))
    im = ax2.imshow(fn_heat.to_numpy(dtype=float), cmap="Reds", vmin=0, vmax=max(0.3, fn_heat.to_numpy(dtype=float).max()))
    ax2.set_xticks(range(len(AA_CLASSES)))
    ax2.set_yticks(range(len(AA_CLASSES)))
    ax2.set_xticklabels(AA_CLASSES, rotation=35, ha="right")
    ax2.set_yticklabels(AA_CLASSES)
    ax2.set_xlabel("alt AA class")
    ax2.set_ylabel("ref AA class")
    ax2.set_title("False-negative rate by (ref -> alt) amino-acid class (pathogenic only)")
    for i, rc in enumerate(AA_CLASSES):
        for j, ac in enumerate(AA_CLASSES):
            v = fn_heat.loc[rc, ac]
            n = count_heat.loc[rc, ac]
            if n == 0:
                ax2.text(j, i, "--", ha="center", va="center", fontsize=8, color="#999")
            else:
                ax2.text(j, i, f"{v:.2f}\nn={n}", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.15 else "black")
    fig.colorbar(im, ax=ax2, label="FN rate")
    fig.tight_layout()
    fig.savefig(args.out_aa, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_aa}")

    # Fig 3 — confident errors by phyloP quintile
    qsum = pd.DataFrame(quint_summary)
    fig, ax3 = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(qsum))
    ax3.bar(x, qsum["FN"], color="#C9463D", label="Confident FN")
    ax3.bar(x, qsum["FP"], bottom=qsum["FN"], color="#E4A700", label="Confident FP")
    ax3.set_xticks(x)
    ax3.set_xticklabels(qsum["quintile"])
    ax3.set_xlabel("phyloP100way_vertebrate quintile (Q1 = least conserved, Q5 = most)")
    ax3.set_ylabel("count of confident errors")
    ax3.set_title(
        "Confident errors (|p - y| > 0.5) stratified by conservation quintile"
    )
    for i, r in enumerate(qsum.itertuples()):
        ax3.text(i, r.FN + r.FP + 0.5, f"{r.FN+r.FP}", ha="center", va="bottom", fontsize=9)
    ax3.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_phylop, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_phylop}")


if __name__ == "__main__":
    main()
