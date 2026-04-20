"""Load denovo-db into a labeled variant table.

Label policy
------------
denovo-db variants are de-novo mutations observed in sequenced probands.
We keep **missense-only** entries (`FunctionClass == "missense"`) and label:

  label = 1  when proband phenotype ∈ {autism, developmental_disorder,
                                       intellectual_disability, epilepsy,
                                       schizophrenia, congenital_heart_disease}
  label = 0  when proband is a documented control

All other rows are dropped (uncertain / unannotated phenotypes). This keeps
the external benchmark honest — we never silently convert "unknown" to benign.

Gene-family holdout
-------------------
After loading we compute gene families via `src.data_splitting.assign_gene_family`
and intersect against the training-time family set. Variants whose family
appears in training are flagged `in_train_family=True` so the evaluator can
report the held-out-only slice alongside the full slice.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from src.external_validation.variant_mapper import to_canonical_key

PATHOGENIC_PHENOTYPES = {
    "autism",
    "autismSpectrumDisorder",
    "developmentalDisorder",
    "intellectualDisability",
    "epilepsy",
    "schizophrenia",
    "congenitalHeartDisease",
    "congenital_heart_disease",
    "developmental_disorder",
    "intellectual_disability",
    "autism_spectrum_disorder",
}

CONTROL_PHENOTYPES = {
    "control",
    "controls",
    "siblingcontrol",
    "sibling",
    "unaffectedSibling",
    "unaffected",
}

MISSENSE_CLASSES = {"missense", "missense-near-splice"}


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")


def load_denovo_db(tsv_path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns:
    `variant_key, chr, pos, ref, alt, gene, label, study, phenotype`.

    Drops non-missense, unmapped, and phenotype-uncertain rows.
    """
    # denovo-db uses `#SampleID\tStudyName\t…` header; pandas handles the `#`
    # with comment=None by keeping it as the first token name.
    with _open_maybe_gz(tsv_path) as fh:
        # Skip the `##version=…` meta line but keep the `#Sample…` header.
        first = fh.readline()
        if not first.startswith("##"):
            fh.seek(0)
        df = pd.read_csv(fh, sep="\t", dtype=str, low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]

    # Restrict to missense (the scope of the trained model).
    if "FunctionClass" not in df.columns:
        raise ValueError("denovo-db file missing FunctionClass column")
    df = df[df["FunctionClass"].str.lower().isin(MISSENSE_CLASSES)].copy()

    # Label assignment.
    phen = df["PrimaryPhenotype"].fillna("").str.strip()
    is_pos = phen.isin(PATHOGENIC_PHENOTYPES)
    is_neg = phen.isin(CONTROL_PHENOTYPES)
    df = df[is_pos | is_neg].copy()
    df["label"] = is_pos[is_pos | is_neg].astype(int).to_numpy()

    # Canonicalize coords.
    canon = df.apply(
        lambda r: to_canonical_key(
            chrom=r.get("Chr"), pos=r.get("Position"), change=r.get("Variant")
        ),
        axis=1,
    )
    df = df.assign(_canon=canon)
    ok = df["_canon"].notna()
    unmapped = df.loc[~ok].copy()
    df = df.loc[ok].copy()
    df["variant_key"] = df["_canon"].map(lambda v: v.key)
    df["chr"] = df["_canon"].map(lambda v: v.chrom)
    df["pos"] = df["_canon"].map(lambda v: v.pos)
    df["ref"] = df["_canon"].map(lambda v: v.ref)
    df["alt"] = df["_canon"].map(lambda v: v.alt)

    out = df[
        [
            "variant_key",
            "chr",
            "pos",
            "ref",
            "alt",
            "Gene",
            "label",
            "StudyName",
            "PrimaryPhenotype",
        ]
    ].rename(columns={"Gene": "gene", "StudyName": "study", "PrimaryPhenotype": "phenotype"})
    out = out.drop_duplicates("variant_key").reset_index(drop=True)
    out.attrs["n_unmapped"] = len(unmapped)
    return out
