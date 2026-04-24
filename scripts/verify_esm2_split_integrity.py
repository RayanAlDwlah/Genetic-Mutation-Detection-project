#!/usr/bin/env python3
# Added for Phase 2.1 (S2.5): hard guard against ESM-2/HEAD split drift.
"""Verify that the ESM-2 score parquets are consistent with HEAD splits.

Checks (all must pass):
    1. scored_train_keys subset of train_keys
    2. scored_val_keys   subset of val_keys
    3. scored_test_keys  subset of test_keys
    4. scored_train_keys cap test_keys == empty
    5. scored_train_keys cap val_keys  == empty
    6. scored_val_keys   cap test_keys == empty

Coverage per split is reported but not gated. Phase 2.1 must NOT proceed
past this script with any failed assertion --- a violation would mean a
training-set ESM-2 score leaked into val or test, inflating the
Phase-2.1 numbers undetectably.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def keys_from(path: Path) -> set[str]:
    return set(pd.read_parquet(path)["variant_key"].astype(str))


def main() -> int:
    train_keys = keys_from(REPO / "data/splits/train.parquet")
    val_keys = keys_from(REPO / "data/splits/val.parquet")
    test_keys = keys_from(REPO / "data/splits/test.parquet")
    scored_train = keys_from(REPO / "data/intermediate/esm2/scores_train.parquet")
    scored_val = keys_from(REPO / "data/intermediate/esm2/scores_val.parquet")
    scored_test = keys_from(REPO / "data/intermediate/esm2/scores_test.parquet")

    failures: list[str] = []

    # Subset checks
    if not scored_train.issubset(train_keys):
        leak = scored_train - train_keys
        failures.append(f"scored_train has {len(leak)} keys NOT in train (sample: {list(leak)[:3]})")
    if not scored_val.issubset(val_keys):
        leak = scored_val - val_keys
        failures.append(f"scored_val has {len(leak)} keys NOT in val (sample: {list(leak)[:3]})")
    if not scored_test.issubset(test_keys):
        leak = scored_test - test_keys
        failures.append(f"scored_test has {len(leak)} keys NOT in test (sample: {list(leak)[:3]})")

    # Pairwise cross-split intersection checks
    leak_train_test = scored_train & test_keys
    if leak_train_test:
        failures.append(
            f"CROSS-SPLIT LEAK: {len(leak_train_test)} variant_keys appear in scored_train AND in test "
            f"(sample: {list(leak_train_test)[:3]})"
        )
    leak_train_val = scored_train & val_keys
    if leak_train_val:
        failures.append(
            f"CROSS-SPLIT LEAK: {len(leak_train_val)} variant_keys appear in scored_train AND in val "
            f"(sample: {list(leak_train_val)[:3]})"
        )
    leak_val_test = scored_val & test_keys
    if leak_val_test:
        failures.append(
            f"CROSS-SPLIT LEAK: {len(leak_val_test)} variant_keys appear in scored_val AND in test "
            f"(sample: {list(leak_val_test)[:3]})"
        )

    # Coverage report (not gated)
    cov = {
        "train": len(scored_train) / max(len(train_keys), 1),
        "val": len(scored_val) / max(len(val_keys), 1),
        "test": len(scored_test) / max(len(test_keys), 1),
    }
    print(
        "[verify_esm2_split_integrity] coverage  "
        f"train={cov['train']:.4f}  val={cov['val']:.4f}  test={cov['test']:.4f}"
    )
    print(
        f"[verify_esm2_split_integrity]    sizes  "
        f"train={len(train_keys)} (scored {len(scored_train)})  "
        f"val={len(val_keys)} (scored {len(scored_val)})  "
        f"test={len(test_keys)} (scored {len(scored_test)})"
    )

    if failures:
        print("\n[verify_esm2_split_integrity] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("[verify_esm2_split_integrity] PASS  (all 6 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
