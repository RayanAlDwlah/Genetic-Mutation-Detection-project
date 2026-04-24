"""denovo-db external-validation significance tests (P0-1 of thesis review).

Computes the statistical tests that back the denovo-db generalisation claim:

  1. ``H0: AUC_post <= 0.5`` on the family-holdout slice (permutation test).
  2. Bootstrap 95% CI for ROC-AUC on the family-holdout slice (matches the
     thesis table), so we can verify whether the CI covers chance.
  3. A paired-delta placeholder: because the thesis "pre-constraint" numbers
     come from a *different trained model* (one without gnomAD-constraint
     features) whose predictions were not persisted, a strict paired test is
     not possible from the available artifacts. We therefore emit a
     ``paired_delta_status = 'not-computed: pre-constraint predictions missing'``
     so the thesis text can truthfully report this limitation.

Usage
-----
.. code-block:: bash

    python -m scripts.denovo_significance \
        --predictions results/metrics/external_denovo_db_predictions.parquet \
        --out         results/metrics/denovo_significance.json

Output
------
``results/metrics/denovo_significance.json``::

    {
      "n": 201,
      "n_positive": 161,
      "auc_post": 0.573,
      "auc_post_ci_lo": 0.47,
      "auc_post_ci_hi": 0.67,
      "auc_post_gt_half_pvalue": 0.083,
      "paired_delta_status": "not-computed: ...",
      ...
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.stats.paired_tests import auc_greater_than_half


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/metrics/external_denovo_db_predictions.parquet"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/metrics/denovo_significance.json"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-permutations", type=int, default=10_000)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(args.predictions)

    required = {"label", "p_calibrated", "family_holdout"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {missing}")

    report: dict[str, object] = {
        "seed": args.seed,
        "n_permutations": args.n_permutations,
        "predictions_file": str(args.predictions),
    }

    # Slice 1 — the 642-variant full denovo-db evaluation.
    full = df
    full_res = auc_greater_than_half(
        full["label"].to_numpy(),
        full["p_calibrated"].to_numpy(),
        n_boot=args.n_permutations,
        seed=args.seed,
    )
    report["full"] = {
        "n": int(full_res["n"]),
        "n_positive": int(full_res["n_positive"]),
        "auc_post": float(full_res["auc"]),
        "auc_post_ci_lo": float(full_res["auc_ci_lo"]),
        "auc_post_ci_hi": float(full_res["auc_ci_hi"]),
        "auc_post_gt_half_pvalue": float(full_res["pvalue_permutation"]),
    }

    # Slice 2 — the 201-variant family-holdout slice (the key claim).
    holdout = df[df["family_holdout"]]
    ho_res = auc_greater_than_half(
        holdout["label"].to_numpy(),
        holdout["p_calibrated"].to_numpy(),
        n_boot=args.n_permutations,
        seed=args.seed,
    )
    report["family_holdout"] = {
        "n": int(ho_res["n"]),
        "n_positive": int(ho_res["n_positive"]),
        "auc_post": float(ho_res["auc"]),
        "auc_post_ci_lo": float(ho_res["auc_ci_lo"]),
        "auc_post_ci_hi": float(ho_res["auc_ci_hi"]),
        "auc_post_gt_half_pvalue": float(ho_res["pvalue_permutation"]),
    }

    # Honest statement about the paired (pre vs post constraint) test.
    report["paired_delta"] = {
        "status": (
            "not-computed: pre-constraint predictions were not persisted "
            "(they come from a different trained model without gnomAD "
            "constraint features); a strict paired DeLong/bootstrap test "
            "requires both prediction vectors on the same variants. The "
            "thesis reports the pre- vs post-constraint numbers but softens "
            "the claim to 'point-estimate improvement whose confidence "
            "interval still includes chance on the 201-variant holdout'."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
