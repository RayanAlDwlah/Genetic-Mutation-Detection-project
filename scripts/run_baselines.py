#!/usr/bin/env python3
"""Score published missense-effect predictors on our paralog-disjoint
test split + denovo-db, and emit a single comparison CSV + forest plot.

Baselines (Stage 1 scope):
  - SIFT             (VEP REST, lower = damaging)
  - PolyPhen-2       (VEP REST, higher = damaging)
  - AlphaMissense    (pre-scored TSV, higher = damaging)

Skipped (documented, deferred to later expansion):
  - REVEL            (requires separate 8 GB download; add in Stage 1.5
                       once infrastructure is proven)
  - CADD             (web-API based; different access pattern)
  - EVE              (limited gene coverage; better as a gene-specific
                       comparison, not a single-number overall baseline)

Outputs:
  results/metrics/baselines_comparison.csv          — one row per (baseline, slice)
  results/metrics/baselines_coverage.csv            — fraction of variants scored
  results/figures/baselines_forest_plot.png         — ROC/PR comparison plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from src.baselines.alphamissense import build_lookup as build_am_lookup
from src.baselines.evaluate import BaselineMetadata, evaluate_baseline
from src.baselines.sift_polyphen import fetch_sift_polyphen

REPO = Path(__file__).resolve().parents[1]

TEST_PATH = REPO / "data/splits/test.parquet"
TRAIN_PATH = REPO / "data/splits/train.parquet"
DENOVO_PREDS_PATH = REPO / "results/metrics/external_denovo_db_predictions.parquet"
DENOVO_RAW_PATH = REPO / "data/raw/external/denovo_db/denovo-db.non-ssc-samples.variants.tsv.gz"

AM_TSV = REPO / "data/raw/baselines/alphamissense/AlphaMissense_hg38.tsv.gz"
AM_CACHE = REPO / "data/intermediate/baselines/alphamissense_lookup.parquet"
SP_CACHE = REPO / "data/intermediate/baselines/sift_polyphen_lookup.parquet"

OUT_CSV = REPO / "results/metrics/baselines_comparison.csv"
OUT_COVERAGE = REPO / "results/metrics/baselines_coverage.csv"
OUT_FIG = REPO / "results/figures/baselines_forest_plot.png"


# ──────────────────────────────────────────────────────────────────────
# Baseline descriptors
# ──────────────────────────────────────────────────────────────────────

METADATA = {
    "sift": BaselineMetadata(
        name="sift",
        display_name="SIFT",
        year=2003,
        training_data="evolutionary conservation (unsupervised)",
        higher_is_pathogenic=False,  # SIFT: lower = damaging
        training_contamination_warning="",
    ),
    "polyphen2": BaselineMetadata(
        name="polyphen2",
        display_name="PolyPhen-2",
        year=2010,
        training_data="HumDiv (supervised on curated disease / benign labels)",
        higher_is_pathogenic=True,
        training_contamination_warning=(
            "PolyPhen-2 was trained on HumDiv/HumVar; some of those labels "
            "overlap with ClinVar entries in our test set — numbers may be "
            "inflated relative to a truly held-out benchmark."
        ),
    ),
    "alphamissense": BaselineMetadata(
        name="alphamissense",
        display_name="AlphaMissense",
        year=2023,
        training_data="human proteome-wide (weakly supervised on population / constraint)",
        higher_is_pathogenic=True,
        training_contamination_warning=(
            "AlphaMissense was calibrated against ClinVar at release time; "
            "any ClinVar-derived test set overlaps to some degree with the "
            "calibration data. Their own paper explicitly marks this as a "
            "caveat when interpreting ClinVar performance."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Loader helpers
# ──────────────────────────────────────────────────────────────────────


def _load_denovo_with_scores() -> pd.DataFrame:
    """Load denovo-db variants + their label, keeping `variant_key` / `gene`.

    Prefers the existing predictions parquet (which has `variant_key, gene,
    y_true as label` already joined) so we don't re-featurize from raw TSV.
    Fallback: re-run the loader against the raw TSV.
    """
    if DENOVO_PREDS_PATH.exists():
        df = pd.read_parquet(DENOVO_PREDS_PATH)
        # We need chr/pos/ref/alt for AM lookup.
        parts = df["variant_key"].str.split(":", expand=True)
        df["chr"] = parts[0]
        df["pos"] = parts[1].astype(int)
        df["ref"] = parts[2]
        df["alt"] = parts[3]
        df = df.rename(columns={"label": "label"})
        return df

    from src.external_validation.denovo_loader import load_denovo_db

    return load_denovo_db(DENOVO_RAW_PATH)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _forest_plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Include our own model by appending rows from committed metrics.
    ours = pd.read_csv(REPO / "results/metrics/xgboost_split_metrics.csv")
    our_test = ours[ours["split"] == "test"].iloc[0]
    ours_row = pd.DataFrame(
        [
            {
                "baseline": "ours",
                "baseline_display_name": "XGBoost (ours)",
                "slice": "clinvar_test",
                "roc_auc": our_test["roc_auc"],
                "roc_auc_ci_lo": our_test["roc_auc"],
                "roc_auc_ci_hi": our_test["roc_auc"],
                "pr_auc": our_test["pr_auc"],
                "pr_auc_ci_lo": our_test["pr_auc"],
                "pr_auc_ci_hi": our_test["pr_auc"],
                "n": our_test["support"],
            }
        ]
    )
    plot_df = pd.concat([df, ours_row], ignore_index=True)
    plot_df = plot_df[plot_df["slice"] == "clinvar_test"].dropna(subset=["pr_auc"])
    plot_df = plot_df.sort_values("pr_auc")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    y = range(len(plot_df))
    for ax, metric in [(ax_roc, "roc_auc"), (ax_pr, "pr_auc")]:
        ax.errorbar(
            plot_df[metric],
            y,
            xerr=[
                plot_df[metric] - plot_df[f"{metric}_ci_lo"],
                plot_df[f"{metric}_ci_hi"] - plot_df[metric],
            ],
            fmt="o",
            color="#2c3e50",
            ecolor="#7f8c8d",
            capsize=4,
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(plot_df["baseline_display_name"].tolist())
        ax.set_xlabel(metric.upper().replace("_", "-"))
        ax.axvline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.grid(axis="x", alpha=0.25)
    ax_roc.set_title(
        "Baseline comparison — ClinVar test (paralog-disjoint)", fontweight="bold", loc="left"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sift_polyphen", "alphamissense", "all"], default="all")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-vep", action="store_true", help="Skip SIFT/PolyPhen (VEP REST takes ~3 min)"
    )
    ap.add_argument(
        "--skip-am",
        action="store_true",
        help="Skip AlphaMissense (file scan takes ~5 min first time)",
    )
    args = ap.parse_args()

    test = pd.read_parquet(TEST_PATH)
    denovo = _load_denovo_with_scores()

    print(f"[baseline] test split: {len(test):,} variants")
    print(f"[baseline] denovo-db: {len(denovo):,} variants")

    all_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, float | int | str]] = []

    # ──── SIFT + PolyPhen ────
    if args.only in ("all", "sift_polyphen") and not args.skip_vep:
        print("\n=== SIFT + PolyPhen-2 ===")

        # Scores for test split
        test_sp = fetch_sift_polyphen(
            test[["chr", "pos", "ref", "alt", "variant_key"]],
            cache_path=SP_CACHE.with_suffix(".test.parquet"),
        )
        # Scores for denovo-db
        denovo_sp = fetch_sift_polyphen(
            denovo[["chr", "pos", "ref", "alt", "variant_key"]],
            cache_path=SP_CACHE.with_suffix(".denovo.parquet"),
        )

        # Merge on variant_key for aligned scoring.
        test_sift = test.merge(test_sp.scores, on="variant_key", how="left")
        denovo_sift = denovo.merge(denovo_sp.scores, on="variant_key", how="left")

        # SIFT — lower is damaging, so we flip direction in evaluate.
        sift_df = evaluate_baseline(
            meta=METADATA["sift"],
            test_df=test_sift,
            test_score=test_sift["sift_score"],
            denovo_df=denovo_sift,
            denovo_score=denovo_sift["sift_score"],
            train_split_path=TRAIN_PATH,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_rows.append(sift_df)

        # PolyPhen-2 — higher is damaging.
        poly_df = evaluate_baseline(
            meta=METADATA["polyphen2"],
            test_df=test_sift,
            test_score=test_sift["polyphen_score"],
            denovo_df=denovo_sift,
            denovo_score=denovo_sift["polyphen_score"],
            train_split_path=TRAIN_PATH,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_rows.append(poly_df)

        coverage_rows.append(
            {
                "baseline": "sift",
                "test_coverage": test_sp.coverage,
                "denovo_coverage": denovo_sp.coverage,
            }
        )
        coverage_rows.append(
            {
                "baseline": "polyphen2",
                "test_coverage": test_sp.coverage,
                "denovo_coverage": denovo_sp.coverage,
            }
        )

    # ──── AlphaMissense ────
    if args.only in ("all", "alphamissense") and not args.skip_am:
        print("\n=== AlphaMissense ===")
        if not AM_TSV.exists():
            print(f"  [skip] {AM_TSV} not found — run download first.")
        else:
            all_keys = pd.concat(
                [test[["variant_key"]], denovo[["variant_key"]]],
                ignore_index=True,
            )
            am = build_am_lookup(
                tsv_gz_path=AM_TSV,
                query_df=all_keys.drop_duplicates("variant_key"),
                cache_path=AM_CACHE,
            )
            test_am = test.merge(am.scores, on="variant_key", how="left")
            denovo_am = denovo.merge(am.scores, on="variant_key", how="left")

            am_df = evaluate_baseline(
                meta=METADATA["alphamissense"],
                test_df=test_am,
                test_score=test_am["am_pathogenicity"],
                denovo_df=denovo_am,
                denovo_score=denovo_am["am_pathogenicity"],
                train_split_path=TRAIN_PATH,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            all_rows.append(am_df)
            coverage_rows.append(
                {
                    "baseline": "alphamissense",
                    "test_coverage": float(test_am["am_pathogenicity"].notna().mean()),
                    "denovo_coverage": float(denovo_am["am_pathogenicity"].notna().mean()),
                }
            )

    if not all_rows:
        print("[baseline] no baselines ran; nothing to write.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT_COVERAGE, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(REPO)}")
    print(f"wrote {OUT_COVERAGE.relative_to(REPO)}")

    # Forest plot
    try:
        _forest_plot(out, OUT_FIG)
        print(f"wrote {OUT_FIG.relative_to(REPO)}")
    except (ImportError, FileNotFoundError) as e:
        print(f"[warn] forest plot skipped: {e}")

    # Print table
    print("\nBaselines vs ours (ClinVar test slice):")
    t = out[out["slice"] == "clinvar_test"][
        [
            "baseline_display_name",
            "roc_auc",
            "roc_auc_ci_lo",
            "roc_auc_ci_hi",
            "pr_auc",
            "pr_auc_ci_lo",
            "pr_auc_ci_hi",
            "n",
            "coverage",
        ]
    ]
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
