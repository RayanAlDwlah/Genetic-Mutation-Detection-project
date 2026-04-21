"""Evaluate published missense-effect predictors on our exact test set.

The module is score-agnostic: any baseline that can produce a
continuous score per `variant_key` can be plugged in. We compute
ROC-AUC and PR-AUC with 1,000-replicate bootstrap 95% CIs for:

- the full paralog-disjoint test split (~28k variants, the headline
  comparison audience expects);
- the denovo-db external slice (~642 variants, where the question
  "does the baseline actually generalize?" matters most);
- the denovo-db `family_holdout_only` slice (unseen gene families).

Reviewer caveats are reported *in the output CSV itself* as a
`training_contamination_warning` column — baselines trained on
ClinVar supersets (AlphaMissense, EVE) have inflated numbers on
ClinVar-derived test sets. We never hide this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_splitting import assign_gene_family
from src.evaluation import bootstrap_metrics


@dataclass(frozen=True)
class BaselineMetadata:
    """How a baseline is described in the comparison table and plots."""

    name: str  # short ID, e.g. "alphamissense"
    display_name: str  # human-readable, e.g. "AlphaMissense"
    year: int
    training_data: str  # what corpus the baseline was trained on
    higher_is_pathogenic: bool = True  # some baselines use "benign = high"
    training_contamination_warning: str = ""  # printed with every metric row


def _score_on_slice(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Compute ROC/PR + bootstrap CIs on a single slice."""
    ok = ~np.isnan(score)
    if ok.sum() < 20 or y[ok].min() == y[ok].max():
        return {
            "n": int(ok.sum()),
            "n_pos": int(y[ok].sum()),
            "roc_auc": np.nan,
            "roc_auc_ci_lo": np.nan,
            "roc_auc_ci_hi": np.nan,
            "pr_auc": np.nan,
            "pr_auc_ci_lo": np.nan,
            "pr_auc_ci_hi": np.nan,
            "coverage": float(ok.mean()) if len(ok) else 0.0,
            "note": "too few or single-class",
        }
    y_, s_ = y[ok], score[ok]
    # ROC/PR are order-invariant, so we use sklearn directly on the raw
    # scores without probability-clipping. (`compute_classification_metrics`
    # clips to [1e-12, 1-1e-12] for Brier/log-loss, which would destroy
    # negative-range baseline scores.) We still rely on our project's
    # `bootstrap_metrics` helper for CIs — it calls sklearn the same way.
    roc = float(roc_auc_score(y_, s_))
    pr = float(average_precision_score(y_, s_))
    # For bootstrap CIs we shift+scale the score into [0,1] so the
    # clipping inside bootstrap_metrics is a no-op and ROC/PR are
    # unaffected (monotone transform).
    lo_s, hi_s = float(s_.min()), float(s_.max())
    s_norm = (s_ - lo_s) / (hi_s - lo_s) if hi_s > lo_s else np.zeros_like(s_)
    boot = bootstrap_metrics(y_, s_norm, threshold=0.5, n_boot=n_boot, seed=seed)
    return {
        "n": int(ok.sum()),
        "n_pos": int(y_.sum()),
        "roc_auc": roc,
        "roc_auc_ci_lo": boot["roc_auc"]["ci_lo"],
        "roc_auc_ci_hi": boot["roc_auc"]["ci_hi"],
        "pr_auc": pr,
        "pr_auc_ci_lo": boot["pr_auc"]["ci_lo"],
        "pr_auc_ci_hi": boot["pr_auc"]["ci_hi"],
        "coverage": float(ok.mean()),
        "note": "",
    }


def evaluate_baseline(
    *,
    meta: BaselineMetadata,
    test_df: pd.DataFrame,
    test_score: pd.Series,
    denovo_df: pd.DataFrame | None = None,
    denovo_score: pd.Series | None = None,
    train_split_path: Path | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of metric rows, one per (slice, baseline).

    Parameters
    ----------
    meta : BaselineMetadata
        Describes the baseline; persisted into every output row.
    test_df : pd.DataFrame
        Our paralog-disjoint test split. Must contain `label` and
        `variant_key`. Must be indexed or joined by `variant_key` so
        `test_score.index` aligns row-wise.
    test_score : pd.Series
        Baseline score per test variant. NaN = no coverage. Must be
        *higher is pathogenic*; if `meta.higher_is_pathogenic` is False
        we flip the sign internally.
    denovo_df, denovo_score : optional
        Same convention for the denovo-db slice. If denovo_df is
        provided and has a `gene` column, we compute the
        `family_holdout_only` slice using
        `src.data_splitting.assign_gene_family`.
    """
    rows: list[dict] = []

    # Test split.
    y_test = test_df["label"].astype(int).to_numpy()
    s_test = np.asarray(test_score, dtype=float)
    if not meta.higher_is_pathogenic:
        s_test = -s_test
    row = _score_on_slice(y_test, s_test, n_boot=n_boot, seed=seed)
    row["slice"] = "clinvar_test"
    rows.append(row)

    # Denovo-db full + family-holdout.
    if denovo_df is not None and denovo_score is not None:
        y_d = denovo_df["label"].astype(int).to_numpy()
        s_d = np.asarray(denovo_score, dtype=float)
        if not meta.higher_is_pathogenic:
            s_d = -s_d

        row = _score_on_slice(y_d, s_d, n_boot=n_boot, seed=seed)
        row["slice"] = "denovo_db_full"
        rows.append(row)

        # family_holdout_only slice
        if train_split_path is not None and "gene" in denovo_df.columns:
            train = pd.read_parquet(train_split_path)
            train_fams = {assign_gene_family(g) for g in train["gene"].unique()}
            holdout = ~denovo_df["gene"].astype(str).map(assign_gene_family).isin(train_fams)
            mask = holdout.to_numpy()
            if mask.sum() >= 20:
                row = _score_on_slice(y_d[mask], s_d[mask], n_boot=n_boot, seed=seed)
                row["slice"] = "denovo_db_family_holdout"
                rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "baseline", meta.name)
    df.insert(1, "baseline_display_name", meta.display_name)
    df["year"] = meta.year
    df["training_data"] = meta.training_data
    df["training_contamination_warning"] = meta.training_contamination_warning
    return df
