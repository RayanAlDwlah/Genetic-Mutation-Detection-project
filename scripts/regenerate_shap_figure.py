#!/usr/bin/env python3
"""
Regenerate the SHAP comparison figure for slide 45.

Handles two cases:
  --mode=full    : Phase-1 SHAP parquet exists, produces side-by-side comparison
  --mode=single  : Phase-1 missing, produces clean single-panel Phase-2.1 figure

Usage:
    python 03_regenerate_shap_figure.py --mode=full
    python 03_regenerate_shap_figure.py --mode=single

Outputs to:
    figures/shap_phase1_vs_phase2.png   (full mode)
    figures/shap_phase2_only.png        (single mode)

Author: prepared for Rayan AlShahrani's defense, Apr 2026
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
# Phase-1 SHAP data is not preserved in this repo — only Phase-2.1 ranking CSV is
SHAP_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "report" / "academic" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Default file paths — point to actual data shipped in this repo
PHASE1_PARQUET = SHAP_DIR / "shap_ranking_phase1.csv"  # not present; --mode=full will fail
PHASE2_PARQUET = SHAP_DIR / "shap_ranking_phase21.csv"

# KKU palette — match your slide theme
KKU_GREEN = "#1B5E3F"
KKU_GOLD = "#C9A961"
NEUTRAL = "#444444"

# Figure aesthetics
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_shap_summary(data_path: Path, top_n: int = 15) -> pd.DataFrame:
    """
    Load a SHAP summary parquet or CSV and return the top-N features by mean |SHAP|.

    Expected columns:
        - 'feature': feature name
        - 'mean_abs_shap': mean absolute SHAP value
    """
    if not data_path.exists():
        raise FileNotFoundError(f"SHAP file not found: {data_path}")

    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    # Tolerate alternate column names
    if "mean_abs_shap" not in df.columns:
        for alt in ("mean_abs_shap_value", "abs_shap", "shap_mean_abs"):
            if alt in df.columns:
                df = df.rename(columns={alt: "mean_abs_shap"})
                break

    if "feature" not in df.columns:
        for alt in ("feature_name", "name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "feature"})
                break

    # Strip the num__ / cat__ prefixes for cleaner labels (Phase-2.1 has them)
    df["feature"] = df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)

    # Sort and take top N
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting — single panel
# -----------------------------------------------------------------------------

def plot_single_panel(phase2_df: pd.DataFrame, output_path: Path) -> None:
    """Single-panel horizontal bar chart of Phase-2.1 SHAP values."""

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

    # Reverse order so highest-impact feature is at top
    df = phase2_df.iloc[::-1].reset_index(drop=True)

    # Highlight ESM-2 features in gold, others in green
    colors = [
        KKU_GOLD if "esm2" in feat.lower() else KKU_GREEN
        for feat in df["feature"]
    ]

    bars = ax.barh(
        df["feature"],
        df["mean_abs_shap"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Add value labels at end of each bar
    for bar, val in zip(bars, df["mean_abs_shap"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
            color=NEUTRAL,
        )

    ax.set_xlabel("mean |SHAP value|", fontsize=11)
    ax.set_title(
        "Phase-2.1 (with ESM-2): Top-15 features by mean |SHAP|",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )

    # Legend explaining the color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=KKU_GOLD, edgecolor="white", label="ESM-2 features"),
        Patch(facecolor=KKU_GREEN, edgecolor="white", label="Phase-1 features"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Wrote {output_path}")


# -----------------------------------------------------------------------------
# Plotting — two panel
# -----------------------------------------------------------------------------

def plot_two_panel(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Side-by-side comparison of Phase-1 vs Phase-2.1 SHAP."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150, sharex=False)

    # Determine a shared x-axis upper bound for visual comparability
    xmax = max(
        phase1_df["mean_abs_shap"].max(),
        phase2_df["mean_abs_shap"].max(),
    ) * 1.15

    for ax, df, title, has_esm2 in [
        (axes[0], phase1_df, "Phase-1 (no ESM-2)", False),
        (axes[1], phase2_df, "Phase-2.1 (with ESM-2)", True),
    ]:
        df_plot = df.iloc[::-1].reset_index(drop=True)

        if has_esm2:
            colors = [
                KKU_GOLD if "esm2" in feat.lower() else KKU_GREEN
                for feat in df_plot["feature"]
            ]
        else:
            colors = [KKU_GREEN] * len(df_plot)

        ax.barh(
            df_plot["feature"],
            df_plot["mean_abs_shap"],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_xlim(0, xmax)
        ax.set_xlabel("mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Top-15 features by mean |SHAP| — Phase-1 vs Phase-2.1",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Legend on the right panel only
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=KKU_GOLD, edgecolor="white", label="ESM-2 features"),
        Patch(facecolor=KKU_GREEN, edgecolor="white", label="Phase-1 features"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Wrote {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["full", "single"],
        required=True,
        help="full = side-by-side (requires Phase-1 parquet); single = Phase-2.1 only",
    )
    parser.add_argument(
        "--phase1-path",
        type=Path,
        default=PHASE1_PARQUET,
        help=f"Phase-1 SHAP file, csv or parquet (default: {PHASE1_PARQUET})",
    )
    parser.add_argument(
        "--phase2-path",
        type=Path,
        default=PHASE2_PARQUET,
        help=f"Phase-2.1 SHAP file, csv or parquet (default: {PHASE2_PARQUET})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top features to plot (default: 15)",
    )
    args = parser.parse_args()

    # Always need Phase-2 data
    try:
        phase2_df = load_shap_summary(args.phase2_path, top_n=args.top_n)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "[HINT] Make sure your Phase-2.1 SHAP computation has been run.\n"
            "       Expected file at: " + str(args.phase2_path),
            file=sys.stderr,
        )
        return 1

    if args.mode == "full":
        try:
            phase1_df = load_shap_summary(args.phase1_path, top_n=args.top_n)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            print(
                "[HINT] Phase-1 SHAP parquet missing. Either:\n"
                "  (a) Run: python -m src.eval.shap_phase1\n"
                "      to regenerate Phase-1 SHAP, then re-run this script.\n"
                "  (b) Use --mode=single to skip the Phase-1 panel.",
                file=sys.stderr,
            )
            return 1

        output_path = FIGURES_DIR / "shap_phase1_vs_phase2.png"
        plot_two_panel(phase1_df, phase2_df, output_path)

    else:  # single
        output_path = FIGURES_DIR / "shap_phase2_only.png"
        plot_single_panel(phase2_df, output_path)

    print("[DONE] Update slide 45 in defense.tex to point at:")
    print(f"       {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
