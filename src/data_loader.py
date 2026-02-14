"""Data loading utilities for mutation detection."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable

import pandas as pd

CLINVAR_COLUMNS = [
    "GeneSymbol",
    "ClinicalSignificance",
    "Chromosome",
    "Start",
    "ReferenceAllele",
    "AlternateAllele",
    "ReviewStatus",
]

DBNSFP_KEY_COLUMNS = ["chr", "pos(1-based)", "ref", "alt"]
DBNSFP_FEATURE_COLUMNS = [
    "PhyloP",
    "phastCons",
    "GERP++",
    "BLOSUM62",
    "Grantham",
    "Polyphen2_HDIV_score",
    "SIFT_score",
]


def _assert_columns(df: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source_name}: {missing}")


def load_csv(path: str | Path, sep: str = ",") -> pd.DataFrame:
    """Load a delimited file into a DataFrame."""
    return pd.read_csv(path, sep=sep, low_memory=False)


def _map_clinvar_label(value: str) -> int | None:
    text = str(value).lower()

    if "uncertain significance" in text or "vus" in text:
        return None

    has_pathogenic = "pathogenic" in text
    has_benign = "benign" in text

    if has_pathogenic and not has_benign:
        return 1
    if has_benign and not has_pathogenic:
        return 0

    return None


def load_clinvar(path: str | Path, drop_vus: bool = True) -> pd.DataFrame:
    """Load ClinVar variant summary and create binary labels."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    _assert_columns(df, CLINVAR_COLUMNS, "ClinVar")

    df = df[CLINVAR_COLUMNS].copy()
    df["label"] = df["ClinicalSignificance"].map(_map_clinvar_label)

    if drop_vus:
        df = df[df["label"].notna()].copy()

    df["label"] = df["label"].astype("Int64")
    return df


def _parse_info_value(info_field: str, key: str) -> str | None:
    token = f"{key}="
    for item in info_field.split(";"):
        if item.startswith(token):
            return item[len(token) :]
    return None


def load_gnomad_vcf(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load gnomAD VCF (or VCF.GZ) into a tabular variant frequency DataFrame."""
    path = Path(path)
    open_fn = gzip.open if path.suffix == ".gz" else open

    rows: list[dict[str, object]] = []
    with open_fn(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            chrom, pos, _vid, ref, alt, _qual, _flt, info = fields[:8]
            alts = alt.split(",")

            af_values = (_parse_info_value(info, "AF") or "").split(",")
            ac_values = (_parse_info_value(info, "AC") or "").split(",")

            for i, alt_allele in enumerate(alts):
                af = af_values[i] if i < len(af_values) and af_values[i] else None
                ac = ac_values[i] if i < len(ac_values) and ac_values[i] else None

                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": int(pos),
                        "Ref": ref,
                        "Alt": alt_allele,
                        "AF": float(af) if af is not None else None,
                        "AC": float(ac) if ac is not None else None,
                    }
                )

                if max_rows is not None and len(rows) >= max_rows:
                    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def apply_benign_proxy(df: pd.DataFrame, af_threshold: float = 0.01) -> pd.DataFrame:
    """Add benign proxy flag from gnomAD AF threshold."""
    if "AF" not in df.columns:
        raise ValueError("gnomAD DataFrame must include 'AF' column")

    out = df.copy()
    out["benign_proxy"] = (out["AF"] > af_threshold).astype(int)
    return out


def load_dbnsfp(path: str | Path, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load dbNSFP table with selected feature columns."""
    features = list(feature_columns) if feature_columns is not None else DBNSFP_FEATURE_COLUMNS
    usecols = DBNSFP_KEY_COLUMNS + features

    df = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    _assert_columns(df, usecols, "dbNSFP")
    return df


def _extract_gene_name(header: str) -> str | None:
    marker = " GN="
    if marker not in header:
        return None
    return header.split(marker, 1)[1].split(" ", 1)[0]


def _extract_protein_id(header: str) -> str:
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            return parts[1]
    return header.split(" ", 1)[0]


def load_uniprot_fasta(path: str | Path) -> pd.DataFrame:
    """Parse UniProt FASTA into a sequence table."""
    records: list[dict[str, object]] = []

    protein_id: str | None = None
    gene_name: str | None = None
    description: str = ""
    seq_parts: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if protein_id is not None:
                    sequence = "".join(seq_parts)
                    records.append(
                        {
                            "ProteinID": protein_id,
                            "GeneName": gene_name,
                            "Sequence": sequence,
                            "Length": len(sequence),
                            "Description": description,
                        }
                    )

                header = line[1:]
                protein_id = _extract_protein_id(header)
                gene_name = _extract_gene_name(header)
                description = header
                seq_parts = []
            else:
                seq_parts.append(line)

    if protein_id is not None:
        sequence = "".join(seq_parts)
        records.append(
            {
                "ProteinID": protein_id,
                "GeneName": gene_name,
                "Sequence": sequence,
                "Length": len(sequence),
                "Description": description,
            }
        )

    return pd.DataFrame(records)
