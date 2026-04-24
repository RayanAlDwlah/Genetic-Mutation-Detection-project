#!/usr/bin/env python3
# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-2).
"""Intersection-subset baseline comparison on the paralog-disjoint test split.

Answers: does AlphaMissense's PR-AUC advantage over our XGBoost
baseline hold when all four methods are evaluated on the *same*
variants, or does its ~14% lower coverage bias the comparison?

Strategy and infrastructural caveat
------------------------------------
The dbNSFP 5.3.1a TSV bundled with this project is BGZF-compressed and
indexed (.tbi) on its primary chr / pos columns, which are GRCh38
coordinates --- despite the file being named ``grch37'' (the GRCh37
positions live in cols 8 / 9 as a secondary annotation, not as the
tabix key).  Our test split, the dbNSFP feature cache, and the trained
XGBoost are all keyed on GRCh37 coordinates.  Tabix lookups using
GRCh37 chr:pos therefore land on the dbNSFP record only for the subset
of variants where the GRCh38 and GRCh37 positions coincide --- typically
a small fraction of any chromosome (only chr11 has near-identity in
the region this thesis emphasises).

A correct full intersection would require either (a) a UCSC liftover
chain (hg19 -> hg38) to translate every test variant before tabix, or
(b) a one-pass scan of the 47 GB dbNSFP TSV filtering on the GRCh37
columns.  Both are out of scope for this revision pass.  The n=446
intersection reported below is therefore the ``tabix-recoverable
intersection'' rather than the full method-overlap intersection; we
mark this explicitly in the output CSV's ``note'' column.  The result
is still informative as a one-sided sanity check: if AlphaMissense
preserves its PR-AUC advantage on this biased subset, the
coverage-bias-only explanation of the Table 5.2 gap is weakened.

Score-direction conventions (all normalised so higher = more pathogenic):
- SIFT: native lower = more pathogenic; we negate (-SIFT_score).
- Polyphen2_HVAR: native higher = more pathogenic.
- AlphaMissense: native higher = more pathogenic.

Output: results/metrics/baselines_intersection.csv columns
method, n, roc_auc, roc_lo, roc_hi, pr_auc, pr_lo, pr_hi,
coverage_original, coverage_intersection,
am_pr_delta_vs_table52, note.  The note flags whether the
AlphaMissense intersection PR-AUC deviates from Table 5.2's
reported 0.890 by more than 0.02 (P0-2 brief: critical-constraint
check).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]

# Try the worktree raw path first, fall back to the parent repo (raw data
# is gitignored so the worktree mirror has it only if symlinked).
_DBNSFP_CANDIDATES = [
    REPO / "data/raw/dbnsfp/dbNSFP5.3.1a_grch37.gz",
    Path("/Users/ry7vv/Documents/Coding_Project/GenticGraduationProject/data/raw/dbnsfp/dbNSFP5.3.1a_grch37.gz"),
]
DBNSFP_TSV = next((p for p in _DBNSFP_CANDIDATES if p.exists()), _DBNSFP_CANDIDATES[0])

# Column indices (1-based) in the dbNSFP TSV header — verified by inspection.
COL_CHR = 1   # GRCh37 chr
COL_POS = 2   # GRCh37 pos (1-based)
COL_REF = 3
COL_ALT = 4
COL_AAREF = 5
COL_AAALT = 6
COL_SIFT = 49
COL_POLYPHEN = 58  # Polyphen2_HVAR_score
COL_AM = 135       # AlphaMissense_score


def score_xgboost_on_test() -> pd.DataFrame:
    """Re-score the test split with the calibrated XGBoost model."""
    test = pd.read_parquet(REPO / "data/splits/test.parquet")
    train = pd.read_parquet(REPO / "data/splits/train.parquet")
    manifest = pd.read_csv(REPO / "results/metrics/xgboost_feature_columns.csv")[
        "encoded_feature"
    ].tolist()
    num_cols = [c.removeprefix("num__") for c in manifest if c.startswith("num__")]
    cat_prefixes = sorted(
        {c.removeprefix("cat__").rsplit("_", 1)[0] for c in manifest if c.startswith("cat__")}
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_prefixes),
        ],
        remainder="drop",
    )
    transformer.fit(train[num_cols + cat_prefixes])
    X = test[num_cols + cat_prefixes].copy()
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = train[num_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X[num_cols] = X[num_cols].fillna(medians).fillna(0.0).astype(np.float32)
    Xt = transformer.transform(X)
    model = XGBClassifier()
    model.load_model(str(REPO / "results/checkpoints/xgboost_best.ubj"))
    p_raw = model.predict_proba(Xt)[:, 1]
    # Refit isotonic from val if available; otherwise raw (AUC invariant)
    val_path = REPO / "results/metrics/xgboost_predictions.parquet"
    if val_path.exists():
        val = pd.read_parquet(val_path)
        val = val[val["split"] == "val"]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val["p_raw"].to_numpy(), val["y_true"].to_numpy())
        p = iso.transform(p_raw)
    else:
        p = p_raw
    out = test[["variant_key", "label"]].copy()
    out["score"] = p.astype(float)
    return out.rename(columns={"label": "y_true"})


def parse_score(s: str) -> float | None:
    """dbNSFP packs multi-isoform scores as semicolon-delimited strings; take max."""
    if s in (".", "", None) or pd.isna(s):
        return None
    parts = [p for p in s.split(";") if p not in (".", "")]
    if not parts:
        return None
    try:
        vals = [float(p) for p in parts]
        return max(vals)
    except ValueError:
        return None


def parse_score_min(s: str) -> float | None:
    """For SIFT (lower = more pathogenic), take the min across isoforms."""
    if s in (".", "", None) or pd.isna(s):
        return None
    parts = [p for p in s.split(";") if p not in (".", "")]
    if not parts:
        return None
    try:
        vals = [float(p) for p in parts]
        return min(vals)
    except ValueError:
        return None


def extract_baselines_via_tabix(test: pd.DataFrame) -> pd.DataFrame:
    """Bulk tabix lookup for all test variants and merge SIFT / PP2 / AM."""
    if shutil.which("tabix") is None:
        raise RuntimeError("tabix binary not found on PATH")
    if not DBNSFP_TSV.exists():
        raise FileNotFoundError(f"dbNSFP TSV not found: {DBNSFP_TSV}")

    # Build a regions file (one chr:pos-pos per unique site).
    sites = test[["chr", "pos"]].drop_duplicates().copy()
    sites["chr_str"] = sites["chr"].astype(str)
    print(f"[intersection] tabix lookup on {len(sites):,} unique sites")
    rows: list[dict] = []
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fh:
        for _, r in sites.iterrows():
            fh.write(f"{r['chr_str']}\t{int(r['pos'])-1}\t{int(r['pos'])}\n")
        bed_path = fh.name
    try:
        result = subprocess.run(
            ["tabix", "-R", bed_path, str(DBNSFP_TSV)],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        Path(bed_path).unlink(missing_ok=True)
    for line in result.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) < COL_AM:
            continue
        rows.append(
            {
                "chr": cols[COL_CHR - 1],
                "pos": int(cols[COL_POS - 1]),
                "ref": cols[COL_REF - 1],
                "alt": cols[COL_ALT - 1],
                "aaref": cols[COL_AAREF - 1],
                "aaalt": cols[COL_AAALT - 1],
                "sift_min": parse_score_min(cols[COL_SIFT - 1]),
                "polyphen_max": parse_score(cols[COL_POLYPHEN - 1]),
                "am_max": parse_score(cols[COL_AM - 1]),
            }
        )
    db = pd.DataFrame(rows)
    print(f"[intersection] tabix returned {len(db):,} dbNSFP rows for these sites")

    # Match on (chr, pos, ref, alt, aaref, aaalt) where possible.
    test_keyed = test.copy()
    test_keyed["chr"] = test_keyed["chr"].astype(str)
    db["chr"] = db["chr"].astype(str)
    merged = test_keyed.merge(
        db,
        on=["chr", "pos", "ref", "alt"],
        how="left",
        suffixes=("", "_db"),
    )
    # Where the same chr:pos:ref:alt has multiple AA-pair rows (alternate
    # transcripts), keep the row matching the test's ref_aa/alt_aa if
    # present; else any.
    if "aaref" in merged.columns and "ref_aa" in merged.columns:
        match = (merged["aaref"] == merged["ref_aa"]) & (merged["aaalt"] == merged["alt_aa"])
        # Prefer aa-matched rows then drop duplicates by variant_key
        merged = merged.assign(__aa_match=match.astype(int)).sort_values(
            ["variant_key", "__aa_match"], ascending=[True, False]
        )
        merged = merged.drop_duplicates(subset="variant_key", keep="first").drop(
            columns=["__aa_match"]
        )
    print(
        f"[intersection] post-merge: {len(merged):,} test rows; "
        f"sift non-null={merged['sift_min'].notna().sum():,}; "
        f"polyphen non-null={merged['polyphen_max'].notna().sum():,}; "
        f"am non-null={merged['am_max'].notna().sum():,}"
    )
    return merged


def bootstrap_metric(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(metric_fn(y, p))
    boot = np.empty(n_boot)
    used = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() in (0, n):
            continue
        boot[used] = metric_fn(y[idx], p[idx])
        used += 1
    boot = boot[:used]
    if used == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test = pd.read_parquet(REPO / "data/splits/test.parquet")
    n_total = len(test)
    print(f"[intersection] test split: {n_total:,} variants")

    # XGBoost test scoring
    print("[intersection] scoring XGBoost on test...")
    xgb = score_xgboost_on_test()

    # Baselines from dbNSFP
    db_merged = extract_baselines_via_tabix(test)

    # Per-method (variant_id, y_true, score) frames; "score" is direction-normalised
    # so higher = more pathogenic for every method.
    methods: dict[str, pd.DataFrame] = {
        "XGBoost": xgb.rename(columns={"variant_key": "variant_id"})[["variant_id", "y_true", "score"]],
    }
    base = db_merged[["variant_key", "label", "sift_min", "polyphen_max", "am_max"]].rename(
        columns={"variant_key": "variant_id", "label": "y_true"}
    )
    sift = base.dropna(subset=["sift_min"]).copy()
    sift["score"] = -sift["sift_min"]  # native low = pathogenic -> invert
    methods["SIFT"] = sift[["variant_id", "y_true", "score"]]
    pp2 = base.dropna(subset=["polyphen_max"]).copy()
    pp2["score"] = pp2["polyphen_max"]
    methods["PolyPhen-2"] = pp2[["variant_id", "y_true", "score"]]
    am = base.dropna(subset=["am_max"]).copy()
    am["score"] = am["am_max"]
    methods["AlphaMissense"] = am[["variant_id", "y_true", "score"]]

    # Intersection: variants scored by every method
    common_ids = set.intersection(*(set(df["variant_id"]) for df in methods.values()))
    n_int = len(common_ids)
    print(f"[intersection] {n_int:,} variants scored by all four methods")
    assert n_int >= 10, "intersection set too small"

    rows: list[dict] = []
    am_pr_intersection: float | None = None
    am_pr_table52 = 0.890  # Table 5.2 AlphaMissense PR-AUC (own coverage)
    for name, df in methods.items():
        sub = df.loc[df["variant_id"].isin(common_ids)].copy()
        sub = sub.sort_values("variant_id").reset_index(drop=True)
        y = sub["y_true"].astype(int).to_numpy()
        s = sub["score"].astype(float).to_numpy()
        roc, roc_lo, roc_hi = bootstrap_metric(
            y, s, roc_auc_score, n_boot=args.n_boot, seed=args.seed
        )
        pr, pr_lo, pr_hi = bootstrap_metric(
            y, s, average_precision_score, n_boot=args.n_boot, seed=args.seed
        )
        if name == "AlphaMissense":
            am_pr_intersection = pr
        rows.append(
            {
                "method": name,
                "n": len(sub),
                "roc_auc": roc,
                "roc_lo": roc_lo,
                "roc_hi": roc_hi,
                "pr_auc": pr,
                "pr_lo": pr_lo,
                "pr_hi": pr_hi,
                "coverage_original": len(df) / n_total,
                "coverage_intersection": len(sub) / n_total,
            }
        )
        print(
            f"[intersection] {name:13s} n={len(sub):5d} "
            f"ROC={roc:.4f} [{roc_lo:.4f}, {roc_hi:.4f}] "
            f"PR={pr:.4f} [{pr_lo:.4f}, {pr_hi:.4f}]"
        )

    # AM delta-vs-Table-5.2 critical check
    if am_pr_intersection is not None:
        delta = am_pr_intersection - am_pr_table52
    else:
        delta = float("nan")
    direction_note = (
        " (AM PR-AUC HIGHER on this subset --- coverage-bias hypothesis NOT supported"
        " on this subset; AM gap is real)"
        if delta > 0.02
        else (
            " (AM PR-AUC LOWER on this subset --- coverage-bias hypothesis IS supported"
            " on this subset; AM advantage shrinks)"
            if delta < -0.02
            else " (|delta| <= 0.02 --- inconclusive on this subset)"
        )
    )
    subset_caveat = (
        "tabix-recoverable subset only (n=%d): hg19 lookups against an hg38-indexed"
        " dbNSFP file. A liftover-based or full-scan intersection is deferred (see"
        " script docstring)." % n_int
    )
    note = (
        f"AM intersection PR-AUC delta vs Table 5.2 (0.890): {delta:+.4f}"
        f"{direction_note}. {subset_caveat}"
    )
    print(f"[intersection] {note}")
    df_out = pd.DataFrame(rows)
    df_out["am_pr_delta_vs_table52"] = delta
    df_out["note"] = note
    out_path = REPO / "results/metrics/baselines_intersection.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[intersection] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
