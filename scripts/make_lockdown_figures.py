#!/usr/bin/env python3
"""Generate the three Phase-1-Lockdown figures referenced from README.

Outputs
-------
results/figures/leakage_journey.png       — the 5-stage PR-AUC decline
results/figures/reliability_calibration.png — pre/post isotonic reliability
results/figures/pr_roc_curves.png         — test-set ROC + PR with bootstrap band

Regenerate:  python scripts/make_lockdown_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
MET = REPO / "results" / "metrics"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig_leakage_journey() -> None:
    df = pd.read_csv(MET / "leakage_fix_journey.csv")
    stages = df["stage"].tolist()
    roc = df["test_roc_auc"].tolist()
    pr = df["test_pr_auc"].tolist()
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, roc, marker="o", lw=2.2, color="#1f77b4", label="ROC-AUC")
    ax.plot(x, pr, marker="s", lw=2.2, color="#d62728", label="PR-AUC")
    for xi, v in zip(x, roc):
        ax.annotate(
            f"{v:.3f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color="#1f77b4",
            fontsize=9,
        )
    for xi, v in zip(x, pr):
        ax.annotate(
            f"{v:.3f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color="#d62728",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "1. Pre-audit\n(0.955)",
            "2. Missense\nfilter",
            "3. Feature\nhygiene",
            "4. Paralog-aware\nsplit",
            "5. Optuna\ntuning",
        ],
        fontsize=9,
    )
    ax.set_ylabel("Test-set AUC")
    ax.set_title(
        "The Leakage Hunt — five audit stages, one honest baseline", fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0.78, 0.98)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    ax.annotate(
        "−13.6 PR-AUC pts from\nfixing non-missense\ncontamination",
        xy=(1, 0.819),
        xytext=(1.4, 0.86),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        fontsize=8.5,
        color="gray",
    )
    fig.tight_layout()
    out = FIG / "leakage_journey.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO)}")


def fig_reliability() -> None:
    df = pd.read_csv(MET / "xgboost_reliability_curve.csv")
    summary = pd.read_csv(MET / "xgboost_calibration_summary.csv").set_index("eval_set")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, which in zip(axes, ["val", "test"]):
        raw = df[df["eval_set"] == f"{which}_raw"]
        cal = df[df["eval_set"] == f"{which}_calibrated"]
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        ax.plot(
            raw["mean_predicted_prob"],
            raw["fraction_positive"],
            marker="o",
            lw=1.6,
            color="#d62728",
            label=f"raw  (ECE={summary.loc[f'{which}_raw','ECE']:.3f})",
        )
        ax.plot(
            cal["mean_predicted_prob"],
            cal["fraction_positive"],
            marker="s",
            lw=1.6,
            color="#2ca02c",
            label=f"isotonic (ECE={summary.loc[f'{which}_calibrated','ECE']:.3f})",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title(f"{which.upper()} — reliability", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Probability calibration — isotonic brings ECE from 0.07 → 0.015",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG / "reliability_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO)}")


def fig_pr_roc() -> None:
    preds_path = MET / "xgboost_predictions.parquet"
    if not preds_path.exists():
        print(f"  (skipped: {preds_path.name} missing)")
        return
    preds = pd.read_parquet(preds_path)
    test = preds[preds["split"] == "test"]
    y = test["y_true"].to_numpy()
    p = test["p_calibrated"].to_numpy() if "p_calibrated" in test else test["p_raw"].to_numpy()

    ci = (
        pd.read_csv(MET / "xgboost_bootstrap_ci.csv").set_index("metric_set").loc["test_calibrated"]
    )

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(
        fpr,
        tpr,
        lw=2,
        color="#1f77b4",
        label=f"ROC-AUC = {ci['roc_auc__mean']:.3f}  "
        f"[{ci['roc_auc__ci_lo']:.3f}, {ci['roc_auc__ci_hi']:.3f}]",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — test set (calibrated)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(
        rec,
        prec,
        lw=2,
        color="#d62728",
        label=f"PR-AUC = {ci['pr_auc__mean']:.3f}  "
        f"[{ci['pr_auc__ci_lo']:.3f}, {ci['pr_auc__ci_hi']:.3f}]",
    )
    base_rate = float(y.mean())
    ax.axhline(base_rate, color="gray", ls="--", lw=1, label=f"baseline (π={base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — test set (calibrated)", fontweight="bold")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Held-out test performance with 95% bootstrap CIs (n=1000)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG / "pr_roc_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO)}")


def main() -> None:
    print("Generating Phase-1-Lockdown figures:")
    fig_leakage_journey()
    fig_reliability()
    fig_pr_roc()
    print("Done.")


if __name__ == "__main__":
    main()
