"""Calibration transfer to denovo-db (P1-4 of thesis review).

The isotonic calibrator in Section 5.6.4 was fit on validation
probabilities (a ClinVar-derived sample). This script tests whether
that calibrator transfers to the external \textsc{denovo-db} cohort
by computing Brier decomposition, ECE, and a reliability curve on
both the raw XGBoost probabilities and the isotonic-calibrated ones,
on both the full denovo-db sample and the family-holdout slice.

Output::

  results/metrics/external_calibration.csv
  results/figures/external_reliability.png

Usage::

  python -m scripts.run_external_calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPO = Path(__file__).resolve().parents[1]


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        total += (n / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _brier_decompose(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ybar = float(y.mean())
    rel = res = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        w = n / len(y)
        pbar = float(p[mask].mean())
        ybar_b = float(y[mask].mean())
        rel += w * (pbar - ybar_b) ** 2
        res += w * (ybar_b - ybar) ** 2
    return {
        "brier": float(brier_score_loss(y, p)),
        "reliability": rel,
        "resolution": res,
        "uncertainty": ybar * (1.0 - ybar),
    }


def _reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b, "n": 0, "mean_pred": np.nan, "mean_true": np.nan})
            continue
        rows.append({
            "bin": b,
            "n": n,
            "mean_pred": float(p[mask].mean()),
            "mean_true": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "results/metrics/external_denovo_db_predictions.parquet",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "results/metrics/external_calibration.csv",
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=REPO / "results/figures/external_reliability.png",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(args.predictions)
    assert {"label", "p_raw", "p_calibrated", "family_holdout"}.issubset(df.columns)

    slices = {
        "full": np.ones(len(df), dtype=bool),
        "family_holdout": df["family_holdout"].to_numpy(),
    }

    rows: list[dict] = []
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    for slice_name, mask in slices.items():
        y = df.loc[mask, "label"].to_numpy(dtype=int)
        for variant, col in [("raw", "p_raw"), ("isotonic", "p_calibrated")]:
            p = df.loc[mask, col].to_numpy(dtype=float)
            b = _brier_decompose(y, p)
            row = {
                "slice": slice_name,
                "variant": variant,
                "n": int(len(y)),
                "n_pos": int(y.sum()),
                "ece": _ece(y, p),
                **b,
            }
            rows.append(row)
            curves[(slice_name, variant)] = _reliability_curve(y, p)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")
    print(out.to_string(index=False))

    # Figure: 2x1 reliability curves (full + family_holdout)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)
    for ax, slice_name in zip(axes, ["full", "family_holdout"]):
        for variant, col, color, marker in [
            ("raw", "p_raw", "#C9463D", "o"),
            ("isotonic", "p_calibrated", "#045531", "s"),
        ]:
            c = curves[(slice_name, variant)]
            c = c[c["n"] > 0]
            ax.plot(
                c["mean_pred"], c["mean_true"],
                marker=marker, lw=1.8, color=color, alpha=0.9,
                label=f"{variant} (ECE={out[(out['slice'] == slice_name) & (out['variant'] == variant)]['ece'].iloc[0]:.3f})",
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="ideal")
        n = int(slices[slice_name].sum())
        ax.set_title(f"denovo-db {slice_name} (n={n})")
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("empirical pathogenic fraction (bin)")
    fig.suptitle("Calibration transfer to denovo-db")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
