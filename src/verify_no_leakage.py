#!/usr/bin/env python3
"""Automated leakage gate for the XGBoost baseline.

Run as:  python -m src.verify_no_leakage

Exits 0 if all checks pass, 1 otherwise. Intended as a pre-commit hook and
a regression gate — if anyone ever reintroduces a banned feature or breaks
the paralog-aware split, this will catch it before the numbers ship.

Checks
------
1. Banned columns never appear in `xgboost_feature_columns.csv`
     {is_common, chr, ref, alt}
2. Training matrix has no non-missense rows (ref_aa AND alt_aa both non-null).
3. Train and test gene families are disjoint (paralog-aware split integrity).
4. No gene appears in both train and test.
5. Label distribution delta between train/val/test ≤ 8pp (sanity on stratification).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

BANNED_FEATURES = {"is_common", "chr", "ref", "alt"}


def _load(path: str) -> pd.DataFrame:
    p = REPO_ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Required artifact missing: {p}")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def check_feature_hygiene() -> list[str]:
    errors: list[str] = []
    try:
        cols = _load("results/metrics/xgboost_feature_columns.csv")
    except FileNotFoundError as e:
        return [f"[feature_hygiene] {e}"]
    feature_names = set(cols.iloc[:, 0].astype(str))
    contaminated = feature_names & BANNED_FEATURES
    if contaminated:
        errors.append(
            f"[feature_hygiene] Banned features present in training matrix: {sorted(contaminated)}"
        )
    # Also catch one-hot variants (e.g. chr_1, chr_X)
    onehot_banned = [c for c in feature_names if c.startswith(("chr_", "is_common_"))]
    if onehot_banned:
        errors.append(
            f"[feature_hygiene] Banned one-hot columns present: {onehot_banned[:5]}"
            f"{' ...' if len(onehot_banned) > 5 else ''}"
        )
    return errors


def check_missense_filter() -> list[str]:
    errors: list[str] = []
    try:
        train = _load("data/splits/train.parquet")
    except FileNotFoundError as e:
        return [f"[missense_filter] {e}"]
    if "ref_aa" not in train.columns or "alt_aa" not in train.columns:
        return ["[missense_filter] ref_aa / alt_aa columns missing from train split"]
    bad = train["ref_aa"].isna() | train["alt_aa"].isna()
    n_bad = int(bad.sum())
    if n_bad > 0:
        errors.append(
            f"[missense_filter] {n_bad:,} training rows have null ref_aa or alt_aa "
            f"(non-missense contamination). Expected 0."
        )
    return errors


def check_split_disjoint() -> list[str]:
    errors: list[str] = []
    try:
        train = _load("data/splits/train.parquet")
        test = _load("data/splits/test.parquet")
    except FileNotFoundError as e:
        return [f"[split_disjoint] {e}"]

    # Gene-level overlap
    shared_genes = set(train["gene"]) & set(test["gene"])
    if shared_genes:
        errors.append(
            f"[split_disjoint] {len(shared_genes)} genes appear in BOTH train and test: "
            f"{sorted(shared_genes)[:5]}..."
        )

    # Family-level overlap (paralog-aware). Recompute family from src code.
    try:
        from src.data_splitting import assign_gene_family
    except Exception as e:  # pragma: no cover
        errors.append(f"[split_disjoint] cannot import assign_gene_family: {e}")
        return errors

    train_fams = set(train["gene"].astype(str).map(assign_gene_family))
    test_fams = set(test["gene"].astype(str).map(assign_gene_family))
    shared_fams = train_fams & test_fams
    if shared_fams:
        errors.append(
            f"[split_disjoint] {len(shared_fams)} gene FAMILIES appear in both train "
            f"and test (paralog leakage): {sorted(shared_fams)[:5]}..."
        )
    return errors


def check_label_balance() -> list[str]:
    errors: list[str] = []
    try:
        train = _load("data/splits/train.parquet")
        val = _load("data/splits/val.parquet")
        test = _load("data/splits/test.parquet")
    except FileNotFoundError as e:
        return [f"[label_balance] {e}"]
    r_tr = train["label"].mean()
    r_va = val["label"].mean()
    r_te = test["label"].mean()
    worst = max(abs(r_tr - r_va), abs(r_tr - r_te), abs(r_va - r_te))
    if worst > 0.08:
        errors.append(
            f"[label_balance] Largest label-ratio gap between splits is {worst:.3f} "
            f"(>0.08). train={r_tr:.3f} val={r_va:.3f} test={r_te:.3f}."
        )
    return errors


def check_phase21_features_present() -> list[str]:
    """If Phase-2.1 was trained, its feature manifest must include
    num__esm2_llr and num__is_imputed_esm2_llr (catches an accidental drop)."""
    errors: list[str] = []
    p21_features = REPO_ROOT / "results/metrics/xgboost_phase21_feature_columns.csv"
    if not p21_features.exists():
        return errors  # Phase 2.1 not run; skip.
    cols = pd.read_csv(p21_features)
    feature_names = set(cols.iloc[:, 0].astype(str))
    required = {"num__esm2_llr", "num__is_imputed_esm2_llr"}
    missing = required - feature_names
    if missing:
        errors.append(
            f"[phase21_features] Phase-2.1 manifest missing required ESM-2 features: {sorted(missing)}"
        )
    return errors


def check_esm2_split_integrity() -> list[str]:
    """Assert that ESM-2 score parquets are consistent with HEAD splits.

    Skips silently when the parquets aren't available (Phase-2.1 not run).
    """
    errors: list[str] = []
    paths = {
        "train": REPO_ROOT / "data/splits/train.parquet",
        "val": REPO_ROOT / "data/splits/val.parquet",
        "test": REPO_ROOT / "data/splits/test.parquet",
        "scored_train": REPO_ROOT / "data/intermediate/esm2/scores_train.parquet",
        "scored_val": REPO_ROOT / "data/intermediate/esm2/scores_val.parquet",
        "scored_test": REPO_ROOT / "data/intermediate/esm2/scores_test.parquet",
    }
    if not all(p.exists() for p in paths.values()):
        return errors

    keys = {k: set(pd.read_parquet(p)["variant_key"].astype(str)) for k, p in paths.items()}
    if not keys["scored_train"].issubset(keys["train"]):
        leak = len(keys["scored_train"] - keys["train"])
        errors.append(f"[esm2_split_integrity] {leak} scored_train keys NOT in train")
    if not keys["scored_val"].issubset(keys["val"]):
        leak = len(keys["scored_val"] - keys["val"])
        errors.append(f"[esm2_split_integrity] {leak} scored_val keys NOT in val")
    if not keys["scored_test"].issubset(keys["test"]):
        leak = len(keys["scored_test"] - keys["test"])
        errors.append(f"[esm2_split_integrity] {leak} scored_test keys NOT in test")
    cross_tt = len(keys["scored_train"] & keys["test"])
    if cross_tt:
        errors.append(f"[esm2_split_integrity] {cross_tt} variant_keys in BOTH scored_train and test")
    cross_tv = len(keys["scored_train"] & keys["val"])
    if cross_tv:
        errors.append(f"[esm2_split_integrity] {cross_tv} variant_keys in BOTH scored_train and val")
    cross_vt = len(keys["scored_val"] & keys["test"])
    if cross_vt:
        errors.append(f"[esm2_split_integrity] {cross_vt} variant_keys in BOTH scored_val and test")
    return errors


CHECKS = [
    ("Feature hygiene", check_feature_hygiene),
    ("Missense filter", check_missense_filter),
    ("Split disjointness (gene + family)", check_split_disjoint),
    ("Label balance across splits", check_label_balance),
    ("Phase-2.1 ESM-2 features present", check_phase21_features_present),
    ("ESM-2 split integrity", check_esm2_split_integrity),
]


def main() -> int:
    all_errors: list[str] = []
    print("=" * 60)
    print("LEAKAGE GATE  —  src/verify_no_leakage.py")
    print("=" * 60)
    for name, fn in CHECKS:
        errs = fn()
        status = "PASS" if not errs else "FAIL"
        print(f"  [{status}] {name}")
        for e in errs:
            print(f"         └─ {e}")
        all_errors.extend(errs)
    print("=" * 60)
    if all_errors:
        print(f"RESULT: {len(all_errors)} leakage issue(s) detected. Fix before shipping.")
        return 1
    print("RESULT: all checks passed. Baseline is leak-free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
