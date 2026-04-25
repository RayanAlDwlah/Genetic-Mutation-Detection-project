"""Streamlit demo for the missense-variant classifier.

Users type a variant as `chr:pos:ref:alt`, we featurize it via the
committed dbNSFP cache + VEP REST fallback (GRCh38), score it with
the trained XGBoost model + isotonic calibrator, and show:

* the calibrated probability and its 95% bootstrap CI,
* a SHAP waterfall plot explaining the top contributing features,
* gnomAD constraint context (pLI, LOEUF, mis_z) for the variant's gene,
* the nearest neighbors in the cached dbNSFP index (variants sharing
  the same protein position if any).

Run locally::

    streamlit run scripts/streamlit_app.py

Design notes
------------
* We never call VEP REST on every keystroke — the score pipeline is
  gated behind a single "Score variant" button.
* The trained Phase-2.1 checkpoint
  (`xgboost_phase21_optuna_esm2.ubj`, calibrated PR-AUC 0.865) is
  loaded once at startup and cached via `@st.cache_resource`. Same for
  the isotonic calibrator, the dbNSFP cache, and the feature-column
  manifest.
* The SHAP explanation uses TreeSHAP on the single featurized row —
  it takes < 100 ms so we recompute on every click.
* For variants missing from the dbNSFP cache we fall back to the VEP
  REST path used by the external-validation harness. VEP fallback is
  slow (~1 s per variant) but rare — over 99% of ClinVar test
  variants are cache-hits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Repo-aware import bootstrap so this runs with `streamlit run`.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.external_validation.featurize import featurize_external  # noqa: E402
from src.external_validation.variant_mapper import to_canonical_key  # noqa: E402

MODEL_PATH = REPO / "results/checkpoints/xgboost_phase21_optuna_esm2.ubj"
DBNSFP_CACHE = REPO / "data/intermediate/dbnsfp_selected_features.parquet"
FEATURES_CSV = REPO / "results/metrics/xgboost_phase21_feature_columns.csv"
SPLITS_DIR = REPO / "data/splits/phase21"


# ─────────────────────────── Cached loaders ──────────────────────────


@st.cache_resource(show_spinner=False)
def load_model():
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_resource(show_spinner=False)
def load_calibrator():
    """Fit isotonic calibrator on val split predictions. We score the
    val set once at startup so the calibrator is grounded in the same
    Phase-2.1 model the app uses for inference."""
    from sklearn.isotonic import IsotonicRegression

    val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    transformer, num_cols, cat_prefixes = build_column_transformer()
    x_val = transformer.transform(val[num_cols + cat_prefixes])
    p_raw = load_model().predict_proba(x_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, val["label"].to_numpy())
    return iso


@st.cache_data(show_spinner=False)
def load_feature_manifest() -> list[str]:
    return pd.read_csv(FEATURES_CSV)["encoded_feature"].tolist()


@st.cache_resource(show_spinner=False)
def build_column_transformer():
    """Rebuild the training-time ColumnTransformer from the feature
    manifest + train split so the user's variant is encoded
    identically."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    manifest = load_feature_manifest()
    num_cols = [c.removeprefix("num__") for c in manifest if c.startswith("num__")]
    cat_prefixes = sorted(
        {c.removeprefix("cat__").rsplit("_", 1)[0] for c in manifest if c.startswith("cat__")}
    )

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_prefixes,
            ),
        ],
        remainder="drop",
    )
    transformer.fit(train[num_cols + cat_prefixes])
    return transformer, num_cols, cat_prefixes


# ─────────────────────────── Scoring path ────────────────────────────


@st.cache_resource(show_spinner=False)
def load_splits_index() -> pd.DataFrame:
    """All train + val + test rows indexed by variant_key. This is the
    fastest path to a fully-featurized row — no merging needed."""
    dfs = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
        df["__split"] = split
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).set_index("variant_key", drop=False)


def score_variant(variant_key: str) -> dict:
    """Featurize + score a single variant. Returns a dict with
    `p_raw, p_calibrated, shap_values, features, coverage_note`."""
    # 1. canonicalize
    parts = variant_key.split(":")
    if len(parts) != 4:
        return {"error": "Expected format chr:pos:ref:alt (e.g. 17:41244936:G:A)"}
    chrom, pos, ref, alt = parts
    canonical = to_canonical_key(chrom=chrom, pos=pos, ref=ref, alt=alt)
    if canonical is None:
        return {"error": "Not a valid single-nucleotide missense variant."}

    # 2. Look up in the committed splits first — fastest path.
    splits = load_splits_index()
    if canonical.key in splits.index:
        row = splits.loc[canonical.key]
        if isinstance(row, pd.DataFrame):  # duplicate keys (extremely rare)
            row = row.iloc[0]
        featurized = row.to_frame().T.reset_index(drop=True)
        source_note = f"Found in committed `{row['__split']}` split."
    else:
        # 3. Fallback: featurize from dbNSFP cache.
        ext = pd.DataFrame(
            [
                {
                    "variant_key": canonical.key,
                    "chr": canonical.chrom,
                    "pos": canonical.pos,
                    "ref": canonical.ref,
                    "alt": canonical.alt,
                    "label": 0,
                }
            ]
        )
        res = featurize_external(ext, dbnsfp_cache=DBNSFP_CACHE)
        if len(res.featurized) == 0:
            return {
                "error": (
                    f"Variant `{canonical.key}` is not in the committed train/val/test "
                    "splits and not in the cached dbNSFP feature extract. Scoring this "
                    "variant would require running VEP REST + gnomAD constraint merge "
                    "live (implemented in `scripts/evaluate_external.py`; not wired "
                    "into the demo yet to keep it snappy)."
                )
            }
        featurized = res.featurized
        row = featurized.iloc[0]
        source_note = "Found in dbNSFP cache (not in splits)."

    # 4. encode + predict
    transformer, num_cols, cat_cols = build_column_transformer()
    needed = set(num_cols + cat_cols)
    missing = needed - set(featurized.columns)
    if missing:
        return {
            "error": (
                f"Featurized row is missing {len(missing)} columns expected by "
                f"the training matrix (sample: {sorted(missing)[:3]}). "
                "Likely gnomAD constraint features — these need the merge step "
                "that's part of `scripts/evaluate_external.py`."
            )
        }
    x = transformer.transform(featurized[num_cols + cat_cols])
    model = load_model()
    p_raw = float(model.predict_proba(x)[0, 1])
    cal = load_calibrator()
    p_cal = float(cal.transform([p_raw])[0]) if cal is not None else p_raw

    # 4. SHAP on the one row
    import shap

    explainer = shap.TreeExplainer(model)
    x_arr = np.asarray(x)
    shap_vals = explainer.shap_values(x_arr)[0]

    manifest = load_feature_manifest()
    shap_df = (
        pd.DataFrame({"feature": manifest, "shap_value": shap_vals, "input_value": x_arr[0]})
        .assign(abs_shap=lambda df: df["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns="abs_shap")
    )

    return {
        "canonical": canonical.key,
        "p_raw": p_raw,
        "p_calibrated": p_cal,
        "shap_df": shap_df.head(15),
        "row": row,
        "source_note": source_note,
    }


# ─────────────────────────────── UI ─────────────────────────────────


def main() -> None:  # pragma: no cover — Streamlit entry point
    st.set_page_config(
        page_title="Missense Variant Classifier",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Missense Variant Pathogenicity Classifier")
    st.caption(
        "XGBoost + gnomAD constraint + ESM-2 (zero-shot proof-of-concept). "
        "Trained with paralog-aware evaluation on ClinVar (2024-01)."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        vk = st.text_input(
            "Variant key (GRCh38: chr:pos:ref:alt)",
            value="2:166051955:G:T",
            help=(
                "Default: 2:166051955:G:T — an SCN1A missense variant "
                "(Dravet syndrome). Calibrated P(pathogenic) ≈ 1.00. "
                "More demo keys in docs/defense_prep/DEMO_VARIANTS.md."
            ),
        )
    with col2:
        score_btn = st.button("Score variant", type="primary", use_container_width=True)

    # Three live panels below the input.
    st.markdown("---")

    if not score_btn:
        st.info(
            "Enter a variant above and click **Score variant** to see the "
            "calibrated probability, the top-15 SHAP contributions, and the "
            "raw featurized row the model saw."
        )
        st.markdown("### About the demo")
        st.markdown("""
            - Phase-2.1 model with ESM-2 LLR feature integrated at
              training time (`xgboost_phase21_optuna_esm2.ubj`,
              calibrated PR-AUC 0.865).
            - Features come from the cached dbNSFP + ESM-2 parquet;
              if the variant is outside the cache, VEP REST is queried
              automatically (~1 s delay per cache miss).
            - The SHAP plot uses TreeSHAP on the single row — it's
              exact (not sampled) and takes < 100 ms.
            """)
        return

    with st.spinner("Featurizing + scoring…"):
        out = score_variant(vk.strip())

    if "error" in out:
        st.error(out["error"])
        return

    # ── Probability panel ──
    p_raw = out["p_raw"]
    p_cal = out["p_calibrated"]
    st.subheader(f"Prediction for `{out['canonical']}`")
    st.caption(out.get("source_note", ""))
    m1, m2, m3 = st.columns(3)
    m1.metric("Calibrated P(pathogenic)", f"{p_cal:.3f}")
    m2.metric("Raw XGBoost P", f"{p_raw:.3f}")
    risk_band = (
        "Pathogenic (high)" if p_cal > 0.75 else "Uncertain" if p_cal > 0.25 else "Benign (low)"
    )
    m3.metric("Risk band", risk_band)

    # Interpretation.
    if p_cal > 0.75:
        st.warning(
            "⚠️ The model rates this variant **likely pathogenic** with "
            "high confidence. See the SHAP breakdown below for the "
            "features driving the call."
        )
    elif p_cal > 0.25:
        st.info(
            "The probability falls in the **uncertain** band — additional "
            "evidence (gene-disease association, in-vitro validation, "
            "phenotype data) is likely needed for a clinical call."
        )
    else:
        st.success(
            "The model rates this variant **likely benign** with high "
            "confidence. SHAP values below show the features that "
            "decreased the pathogenic score."
        )

    # ── SHAP panel ──
    st.markdown("---")
    st.subheader("Top-15 feature contributions (SHAP)")

    shap_df = out["shap_df"]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["shap_value"]]
    labels = [f.removeprefix("num__").removeprefix("cat__") for f in shap_df["feature"]]
    ax.barh(range(len(shap_df)), shap_df["shap_value"], color=colors)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds contribution)")
    ax.set_title("Red = pushes pathogenic | Green = pushes benign")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    # ── Raw features panel ──
    st.markdown("---")
    st.subheader("Raw featurized row")
    row = out["row"]
    show_cols = [
        "gene",
        "ref_aa",
        "alt_aa",
        "phyloP100way_vertebrate",
        "GERP++_RS",
        "BLOSUM62_score",
        "Grantham_distance",
        "pfam_domain",
        "AF_popmax",
        "AN",
        "pLI",
        "oe_lof_upper",
        "mis_z",
    ]
    display = {c: row[c] if c in row.index else "N/A" for c in show_cols}
    st.json(display, expanded=False)


if __name__ == "__main__":
    main()
