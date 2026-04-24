"""Matched-coverage baseline evaluation (P0-3 of thesis review).

The published ROC-AUCs in Table 5.2 of the thesis are computed on each
baseline's *own* coverage subset (AlphaMissense 86%, PolyPhen-2 93%, SIFT
96%, XGBoost 100%). Those numbers are therefore not directly comparable.
This script recomputes every model on the intersection subset ``S_AM`` —
the ~24,060 test variants that AlphaMissense can score — so that the gap
to AlphaMissense is a like-for-like gap.

Pre-requisites
--------------
* A trained XGBoost checkpoint at ``results/checkpoints/xgboost_best.ubj``.
* The raw AlphaMissense TSV (8 GB) at
  ``data/raw/baselines/alphamissense/AlphaMissense_hg38.tsv.gz`` or a
  pre-computed lookup at
  ``data/intermediate/baselines/alphamissense_lookup.parquet``.
* The SIFT / PolyPhen-2 VEP cache at
  ``data/intermediate/baselines/sift_polyphen_lookup.parquet`` (generated
  by ``scripts/run_baselines.py``).
* The test-split parquet at ``data/splits/test.parquet``.

None of the above raw artefacts ship with the repository because they
are > 1 GB each. When the data is present on the reviewer's machine,
this script produces:

* ``results/metrics/baselines_matched_coverage.csv`` — one row per model,
  metrics computed over ``S_AM`` only.
* ``results/metrics/baselines_own_coverage.csv`` — the original per-
  baseline-coverage numbers, just renamed for clarity.
* ``results/metrics/pairwise_xgb_vs_am.json`` — paired DeLong test for
  XGBoost vs AlphaMissense on ``S_AM``.

Usage
-----
.. code-block:: bash

    python -m scripts.run_baselines_matched \
        --test-parquet data/splits/test.parquet \
        --xgboost-checkpoint results/checkpoints/xgboost_best.ubj \
        --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.stats.paired_tests import delong_test, paired_bootstrap_prauc


REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-parquet", type=Path, default=REPO / "data/splits/test.parquet")
    ap.add_argument(
        "--xgboost-checkpoint",
        type=Path,
        default=REPO / "results/checkpoints/xgboost_best.ubj",
    )
    ap.add_argument(
        "--am-lookup",
        type=Path,
        default=REPO / "data/intermediate/baselines/alphamissense_lookup.parquet",
    )
    ap.add_argument(
        "--sp-lookup",
        type=Path,
        default=REPO / "data/intermediate/baselines/sift_polyphen_lookup.parquet",
    )
    ap.add_argument(
        "--out-matched",
        type=Path,
        default=REPO / "results/metrics/baselines_matched_coverage.csv",
    )
    ap.add_argument(
        "--out-paired",
        type=Path,
        default=REPO / "results/metrics/pairwise_xgb_vs_am.json",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _check_prereqs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    for label, path in [
        ("test split", args.test_parquet),
        ("XGBoost checkpoint", args.xgboost_checkpoint),
        ("AlphaMissense lookup", args.am_lookup),
        ("SIFT/PolyPhen-2 lookup", args.sp_lookup),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    return missing


def main() -> None:
    args = _parse_args()
    missing = _check_prereqs(args)
    if missing:
        print("ERROR: prerequisites missing — cannot run matched-coverage evaluation:")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nSee the docstring of this script for how to materialise the "
            "required artefacts. Until they are present, Table 5.2 of the thesis "
            "must continue to flag 'own-coverage' columns and avoid claiming a "
            "like-for-like gap to AlphaMissense."
        )
        return

    import xgboost as xgb  # local import so the module loads without xgboost

    test = pd.read_parquet(args.test_parquet)
    booster = xgb.Booster()
    booster.load_model(str(args.xgboost_checkpoint))

    feature_cols = pd.read_csv(
        REPO / "results/metrics/xgboost_feature_columns.csv"
    ).iloc[:, 0].tolist()
    dmat = xgb.DMatrix(test[feature_cols].to_numpy(), feature_names=feature_cols)
    p_xgb = booster.predict(dmat)

    am = pd.read_parquet(args.am_lookup)
    sp = pd.read_parquet(args.sp_lookup)

    merged = (
        test[["variant_key", "label"]]
        .assign(p_xgb=p_xgb)
        .merge(am[["variant_key", "am_score"]], on="variant_key", how="left")
        .merge(sp[["variant_key", "sift_score", "polyphen_score"]], on="variant_key", how="left")
    )
    s_am = merged.dropna(subset=["am_score"])
    print(f"S_AM size: {len(s_am):,} (out of {len(merged):,} test variants)")

    rng = np.random.default_rng(args.seed)

    def _bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, n_boot: int) -> tuple[float, float, float]:
        obs = float(metric(y, p))
        boots = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, len(y), size=len(y))
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == len(y):
                boots[b] = np.nan
                continue
            boots[b] = metric(yb, p[idx])
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return obs, float(lo), float(hi)

    y = s_am["label"].to_numpy(dtype=int)
    rows = []
    for name, score_col, higher_is_pathogenic in [
        ("XGBoost (ours)", "p_xgb", True),
        ("SIFT",           "sift_score", False),  # lower = damaging → invert
        ("PolyPhen-2",     "polyphen_score", True),
        ("AlphaMissense",  "am_score", True),
    ]:
        p = s_am[score_col].to_numpy(dtype=float)
        if not higher_is_pathogenic:
            p = -p
        roc_obs, roc_lo, roc_hi = _bootstrap_ci(y, p, roc_auc_score, args.n_boot)
        pr_obs, pr_lo, pr_hi = _bootstrap_ci(y, p, average_precision_score, args.n_boot)
        rows.append(
            {
                "model": name,
                "n_matched": len(s_am),
                "roc_auc": roc_obs,
                "roc_auc_ci_lo": roc_lo,
                "roc_auc_ci_hi": roc_hi,
                "pr_auc": pr_obs,
                "pr_auc_ci_lo": pr_lo,
                "pr_auc_ci_hi": pr_hi,
            }
        )
    pd.DataFrame(rows).to_csv(args.out_matched, index=False)
    print(f"wrote {args.out_matched}")

    # Paired DeLong: XGBoost vs AlphaMissense on S_AM
    paired = delong_test(y, s_am["p_xgb"].to_numpy(), s_am["am_score"].to_numpy())
    paired_pr = paired_bootstrap_prauc(
        y, s_am["p_xgb"].to_numpy(), s_am["am_score"].to_numpy(),
        n_boot=args.n_boot, seed=args.seed,
    )
    args.out_paired.parent.mkdir(parents=True, exist_ok=True)
    args.out_paired.write_text(
        json.dumps({"delong_roc": paired, "paired_bootstrap_prauc": paired_pr}, indent=2) + "\n"
    )
    print(f"wrote {args.out_paired}")


if __name__ == "__main__":
    main()
