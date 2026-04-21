"""Calibration deep-dive: Brier decomposition + Platt vs Isotonic.

Concepts
--------
A classifier's Brier score can be decomposed into three additive terms::

    Brier = Reliability − Resolution + Uncertainty

where each term is computed by grouping predictions into K bins:

* **Reliability** — weighted mean-squared gap between the predicted
  probability in each bin and the empirical positive rate in that bin.
  Lower is better; 0 means "predicted probabilities match the truth
  exactly in every bin".

* **Resolution** — weighted variance of empirical positive rates across
  bins, relative to the overall positive rate. Higher is better; 0 means
  "every bin looks the same (the classifier doesn't discriminate)".

* **Uncertainty** — irreducible data entropy :math:`\\bar y (1 - \\bar y)`.
  Constant for a given dataset.

The decomposition makes it explicit whether a bad Brier score comes
from **miscalibration** (fixable with Platt / Isotonic) or from
**poor resolution** (fixable only with a better model).

This module also fits three calibrators on validation probabilities,
scores them on test, and dumps a 3-panel reliability diagram so
reviewers can *see* the improvement, not just read an ECE number.

Outputs
-------
* ``results/metrics/brier_decomposition.csv`` — one row per
  (calibrator, split) with Brier / Reliability / Resolution /
  Uncertainty / ECE / MCE.
* ``results/figures/calibration_triptych.png`` — three-panel reliability
  diagram (raw / Platt / Isotonic).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation import reliability_curve

# ──────────────────────────── Brier decomposition ─────────────────────


@dataclass(frozen=True)
class BrierDecomposition:
    """Additive Brier decomposition on K-bin reliability groups."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    mce: float
    n: int

    @property
    def recomposition_error(self) -> float:
        """Reconstruction residual ``brier − (reliability − resolution + uncertainty)``.

        Should be ~1e-10 if the math is right; anything above 1e-6 is a
        numerical warning that bin edges or weights are off.
        """
        return self.brier - (self.reliability - self.resolution + self.uncertainty)


def decompose_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> BrierDecomposition:
    """Compute the Murphy (1973) additive Brier decomposition.

    ``strategy="quantile"`` puts equal-sample counts in each bin, which
    gives stable estimates in the tails. Use ``"uniform"`` for
    evenly-spaced bins (the textbook default; noisier at the extremes).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)
    n = len(y)
    ybar = float(y.mean())

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bin each prediction; digitize with `edges[1:-1]` so the first bin
    # starts at edges[0] and the last bin ends at edges[-1].
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        pbar_b = float(p[mask].mean())
        ybar_b = float(y[mask].mean())
        w = nb / n
        gap = abs(pbar_b - ybar_b)
        reliability += w * (pbar_b - ybar_b) ** 2
        resolution += w * (ybar_b - ybar) ** 2
        ece += w * gap
        mce = max(mce, gap)

    uncertainty = ybar * (1.0 - ybar)
    brier = float(brier_score_loss(y, p))

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        ece=ece,
        mce=mce,
        n=n,
    )


# ──────────────────────────── Calibrators ────────────────────────────


def fit_platt(val_prob: np.ndarray, val_true: np.ndarray) -> LogisticRegression:
    """Platt scaling: fit a 1-feature logistic regression on raw probs.

    We wrap the probability in a 2-D array because sklearn expects
    ``(n_samples, n_features)``. The logits are recovered via
    ``model.predict_proba`` at inference time.
    """
    x = np.asarray(val_prob, dtype=float).reshape(-1, 1)
    y = np.asarray(val_true, dtype=int)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return lr


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(prob, dtype=float).reshape(-1, 1))[:, 1]


def fit_isotonic(val_prob: np.ndarray, val_true: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(val_prob, dtype=float), np.asarray(val_true, dtype=int))
    return iso


def apply_isotonic(model: IsotonicRegression, prob: np.ndarray) -> np.ndarray:
    return np.asarray(model.transform(np.asarray(prob, dtype=float)))


# ──────────────────────────── Report drivers ─────────────────────────


def compute_decomposition_table(
    *,
    val_true: np.ndarray,
    val_raw: np.ndarray,
    test_true: np.ndarray,
    test_raw: np.ndarray,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Fit each calibrator on val, score on val+test, emit one row per
    (calibrator, split) with the full decomposition + log-loss."""
    platt = fit_platt(val_raw, val_true)
    iso = fit_isotonic(val_raw, val_true)

    val_platt = apply_platt(platt, val_raw)
    test_platt = apply_platt(platt, test_raw)
    val_iso = apply_isotonic(iso, val_raw)
    test_iso = apply_isotonic(iso, test_raw)

    def _row(name: str, split: str, y: np.ndarray, p: np.ndarray) -> dict[str, float | str | int]:
        d = decompose_brier(y, p, n_bins=n_bins)
        return {
            "calibrator": name,
            "split": split,
            "n": d.n,
            "brier": d.brier,
            "reliability": d.reliability,
            "resolution": d.resolution,
            "uncertainty": d.uncertainty,
            "ece": d.ece,
            "mce": d.mce,
            "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
            "recomposition_residual": d.recomposition_error,
        }

    rows = [
        _row("raw", "val", val_true, val_raw),
        _row("raw", "test", test_true, test_raw),
        _row("platt", "val", val_true, val_platt),
        _row("platt", "test", test_true, test_platt),
        _row("isotonic", "val", val_true, val_iso),
        _row("isotonic", "test", test_true, test_iso),
    ]
    return pd.DataFrame(rows)


def render_triptych(
    *,
    y_test: np.ndarray,
    p_test_raw: np.ndarray,
    p_test_platt: np.ndarray,
    p_test_iso: np.ndarray,
    out_path: Path,
    n_bins: int = 15,
) -> None:
    """3-panel reliability diagram on test: raw / Platt / Isotonic."""
    import matplotlib.pyplot as plt

    panels = [
        ("Raw", p_test_raw),
        ("Platt scaling", p_test_platt),
        ("Isotonic regression", p_test_iso),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (title, p) in zip(axes, panels, strict=True):
        curve = reliability_curve(y_test, p, n_bins=n_bins, strategy="quantile")
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, alpha=0.7, label="perfect")
        ax.plot(
            curve["mean_predicted_prob"],
            curve["fraction_positive"],
            "o-",
            color="#2c3e50",
            label=f"n={len(p):,}",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_title(f"{title}\nECE = {curve.attrs['ECE']:.3f}", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Fraction positive")
    fig.suptitle("Calibration triptych — test split (paralog-disjoint)", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── CLI driver ─────────────────────────────


def main() -> None:
    """Reproduce the decomposition + triptych from the committed
    predictions parquet. Usage:

        python -m src.calibration_deep
    """
    repo_root = Path(__file__).resolve().parents[1]
    preds_path = repo_root / "results/metrics/xgboost_predictions.parquet"
    if not preds_path.exists():  # pragma: no cover
        raise SystemExit(f"missing: {preds_path}")

    df = pd.read_parquet(preds_path)
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    out_csv = repo_root / "results/metrics/brier_decomposition.csv"
    out_fig = repo_root / "results/figures/calibration_triptych.png"

    table = compute_decomposition_table(
        val_true=val["y_true"].to_numpy(int),
        val_raw=val["p_raw"].to_numpy(float),
        test_true=test["y_true"].to_numpy(int),
        test_raw=test["p_raw"].to_numpy(float),
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    # Triptych needs test-side calibrated probs; fit on val again.
    platt = fit_platt(val["p_raw"].to_numpy(float), val["y_true"].to_numpy(int))
    iso = fit_isotonic(val["p_raw"].to_numpy(float), val["y_true"].to_numpy(int))
    render_triptych(
        y_test=test["y_true"].to_numpy(int),
        p_test_raw=test["p_raw"].to_numpy(float),
        p_test_platt=apply_platt(platt, test["p_raw"].to_numpy(float)),
        p_test_iso=apply_isotonic(iso, test["p_raw"].to_numpy(float)),
        out_path=out_fig,
    )

    print(f"wrote {out_csv.relative_to(repo_root)}")
    print(f"wrote {out_fig.relative_to(repo_root)}")
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
