"""Pairwise model-vs-model significance tests (P0-4 of thesis review).

Runs the paired tests for every comparison the thesis makes and writes a
single machine-readable CSV that each results table cites.

Comparisons currently runnable from the persisted artefacts:
  * denovo-db family-holdout: XGBoost vs ESM-2 LLR
  * denovo-db family-holdout: XGBoost vs rank-fusion (tuned)
  * denovo-db family-holdout: ESM-2 vs rank-fusion (tuned)

Comparisons that need data that is not in the repository
(logged as ``not-computed`` rather than silently skipped):
  * XGBoost vs SIFT / PolyPhen-2 / AlphaMissense on the clinvar test split
    (requires persisted per-variant XGBoost test predictions — not in repo).
  * XGBoost-val vs XGBoost-test (different row sets, not a paired test).

Output
------
``results/metrics/pairwise_pvalues.csv`` — one row per comparison with
columns matching :class:`src.stats.paired_tests.PairwiseRow`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.stats.paired_tests import (
    PairwiseRow,
    delong_test,
    holm_bonferroni,
    paired_bootstrap_prauc,
)

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--denovo-preds",
        type=Path,
        default=REPO / "results/metrics/external_denovo_db_predictions.parquet",
    )
    ap.add_argument(
        "--esm2-denovo",
        type=Path,
        default=REPO / "results/metrics/esm2_denovo_db_scores.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/metrics/pairwise_pvalues.csv",
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _denovo_family_holdout_table(args: argparse.Namespace) -> pd.DataFrame:
    """XGBoost vs ESM-2 vs fusion, paired, on the denovo-db family-holdout slice."""
    xgb_df = pd.read_parquet(args.denovo_preds)
    esm = pd.read_parquet(args.esm2_denovo)

    ho = xgb_df[xgb_df["family_holdout"]].copy()
    merged = ho.merge(
        esm[["variant_key", "esm2_llr"]], on="variant_key", how="inner"
    )
    # Drop any rows missing ESM-2 (some variants skip due to window issues).
    merged = merged.dropna(subset=["esm2_llr"])
    print(f"[denovo family-holdout] paired set: {len(merged):,} variants "
          f"(of {len(ho):,} in the slice)")

    y = merged["label"].to_numpy(dtype=int)
    p_xgb = merged["p_calibrated"].to_numpy(dtype=float)
    # The denovo-db analysis in scripts/analyze_esm2_denovo.py ranks the RAW
    # LLR (higher LLR -> higher rank) and uses an unsigned mean fusion. Despite
    # the theoretical expectation that more-negative LLR means more pathogenic,
    # the empirical ROC of the raw LLR on this slice is 0.552 (higher LLR
    # happens to correlate positively with pathogenic labels on n=201). We
    # match the thesis formula here so the reported test speaks to the
    # thesis's actually-published numbers; the underlying sign anomaly is
    # documented as P0-2 in the thesis review (see tests/integration/
    # test_esm2_sanity.py).
    s_esm_raw = merged["esm2_llr"].to_numpy(dtype=float)
    rank_xgb = pd.Series(p_xgb).rank(pct=True).to_numpy()
    rank_esm = pd.Series(s_esm_raw).rank(pct=True).to_numpy()
    # Uniform rank-mean fusion, matching the denovo-db analysis script.
    p_fusion = 0.5 * rank_xgb + 0.5 * rank_esm
    # Direction-corrected "what fusion should be": XGB + (-LLR) ranks.
    p_esm_signed = -s_esm_raw

    rows: list[PairwiseRow] = []
    for a, b, p1, p2 in [
        ("XGBoost", "ESM-2 LLR (raw, thesis direction)", p_xgb, s_esm_raw),
        ("XGBoost", "rank-fusion (uniform, raw-LLR)", p_xgb, p_fusion),
        ("ESM-2 LLR (raw)", "rank-fusion (uniform, raw-LLR)", s_esm_raw, p_fusion),
    ]:
        d = delong_test(y, p1, p2)
        pr = paired_bootstrap_prauc(y, p1, p2, n_boot=args.n_boot, seed=args.seed)
        rows.append(
            PairwiseRow(
                slice="denovo-db family-holdout (n=paired)",
                model_a=a,
                model_b=b,
                metric="ROC-AUC",
                estimate_a=d["auc1"],
                estimate_b=d["auc2"],
                delta=d["delta"],
                ci_lo=d["delta_ci_lo"],
                ci_hi=d["delta_ci_hi"],
                pvalue=d["pvalue"],
            )
        )
        rows.append(
            PairwiseRow(
                slice="denovo-db family-holdout (n=paired)",
                model_a=a,
                model_b=b,
                metric="PR-AUC",
                estimate_a=pr["prauc1"],
                estimate_b=pr["prauc2"],
                delta=pr["delta"],
                ci_lo=pr["delta_ci_lo"],
                ci_hi=pr["delta_ci_hi"],
                pvalue=pr["pvalue"],
            )
        )
    return pd.DataFrame([r.as_dict() for r in rows])


def _unrunnable_table() -> pd.DataFrame:
    """Comparisons we cannot run here — recorded rather than hidden."""
    reason = (
        "per-variant XGBoost test predictions not persisted; "
        "run scripts/run_baselines_matched.py once xgboost_predictions.parquet "
        "is materialised to close D7"
    )
    rows = []
    for base in ["SIFT", "PolyPhen-2", "AlphaMissense"]:
        for metric in ["ROC-AUC", "PR-AUC"]:
            rows.append(
                {
                    "slice": "clinvar_test (own-coverage subsets differ)",
                    "model_a": "XGBoost",
                    "model_b": base,
                    "metric": metric,
                    "estimate_a": float("nan"),
                    "estimate_b": float("nan"),
                    "delta": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "pvalue": float("nan"),
                    "pvalue_holm": float("nan"),
                    "status": f"not-computed: {reason}",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    runnable = _denovo_family_holdout_table(args)

    # Holm-Bonferroni within each metric family (ROC-AUC and PR-AUC separately).
    for metric in runnable["metric"].unique():
        mask = runnable["metric"] == metric
        adjusted = holm_bonferroni(runnable.loc[mask, "pvalue"].tolist())
        runnable.loc[mask, "pvalue_holm"] = adjusted

    runnable["status"] = "computed"
    out = pd.concat([runnable, _unrunnable_table()], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
