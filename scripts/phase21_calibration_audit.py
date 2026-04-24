#!/usr/bin/env python3
# Added for Phase 2.1 (S11): Phase-1 vs Phase-2.1 calibration comparison.
"""Compare ECE / MCE / Brier between Phase-1 and Phase-2.1 calibrators.

Reads:
  - Phase-1 calibration summary at results/metrics/xgboost_calibration_summary.csv
  - Phase-2.1 calibration summary at results/metrics/phase21/xgboost_calibration_summary.csv

Both already contain ECE + MCE on test_calibrated. Brier comes from the
respective bootstrap_ci CSVs. Writes a single comparison table; flags
whether Phase-2.1 calibration drifted by more than 0.01 ECE (the
fallback-to-Platt threshold from the plan).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
P1_CAL = REPO / "results/metrics/xgboost_calibration_summary.csv"
P21_CAL = REPO / "results/metrics/phase21/xgboost_calibration_summary.csv"
P1_BOOT = REPO / "results/metrics/xgboost_bootstrap_ci.csv"
P21_BOOT = REPO / "results/metrics/phase21/xgboost_bootstrap_ci.csv"
OUT = REPO / "results/metrics/phase21_calibration_comparison.csv"


def main() -> int:
    p1 = pd.read_csv(P1_CAL)
    p21 = pd.read_csv(P21_CAL)
    p1_boot = pd.read_csv(P1_BOOT)
    p21_boot = pd.read_csv(P21_BOOT)

    rows = []
    for phase, cal_df, boot_df in (("phase1", p1, p1_boot), ("phase21", p21, p21_boot)):
        for eval_set in ("val_calibrated", "test_calibrated", "val_raw", "test_raw"):
            cal_row = cal_df[cal_df["eval_set"] == eval_set]
            boot_row = boot_df[boot_df["metric_set"] == eval_set]
            if cal_row.empty or boot_row.empty:
                continue
            rows.append({
                "phase": phase,
                "eval_set": eval_set,
                "calibrator_method": "isotonic",
                "ece": float(cal_row["ECE"].iloc[0]),
                "mce": float(cal_row["MCE"].iloc[0]),
                "brier_loss": float(boot_row["brier_loss__mean"].iloc[0]),
                "roc_auc": float(boot_row["roc_auc__mean"].iloc[0]),
                "pr_auc": float(boot_row["pr_auc__mean"].iloc[0]),
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[calib_audit] wrote {OUT}")
    print(df.to_string(index=False))

    # Fallback decision
    p1_ece = df[(df.phase == "phase1") & (df.eval_set == "test_calibrated")]["ece"].iloc[0]
    p21_ece = df[(df.phase == "phase21") & (df.eval_set == "test_calibrated")]["ece"].iloc[0]
    delta = p21_ece - p1_ece
    print(f"\nΔECE (phase21 − phase1) = {delta:+.4f}")
    if delta > 0.01:
        print("[calib_audit] WARN: Phase-2.1 ECE worse than Phase-1 by > 0.01; consider Platt fallback.")
    else:
        print(f"[calib_audit] OK: |ΔECE| = {abs(delta):.4f}, isotonic kept as headline calibrator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
