#!/usr/bin/env python3
"""Extract allele frequencies from gnomAD for ClinVar merging.

This script supports three approaches:
1) VCF streaming extraction (cyvcf2 or pysam backend) for very large gnomAD files.
2) TSV/CSV pre-extracted table normalization.
3) A Colab-friendly fallback function using gnomAD GraphQL API for specific variants.

Example:
    python src/gnomad_extraction.py \
        --input data/raw/gnomad/gnomad.exomes.r2.1.1.sites.vcf.bgz \
        --clinvar-variants data/intermediate/clinvar_labeled_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - runtime dependency guard
    pa = None
    pq = None


POPULATIONS = ["AFR", "AMR", "ASJ", "EAS", "FIN", "NFE", "SAS"]
EPSILON = 1e-8
DEFAULT_OUTPUT = "data/intermediate/gnomad_af_clean.parquet"
DEFAULT_METADATA = "data/intermediate/gnomad_metadata.json"


def resolve_path(repo_root: Path, path_str: str) -> Path:
    """Resolve absolute/relative path against repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_chromosome(value: Any) -> str | None:
    """Normalize chromosome labels to 1-22/X/Y style without 'chr' prefix."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.lower().startswith("chr"):
        text = text[3:]

    text = text.upper()
    if text == "23":
        return "X"
    if text == "24":
        return "Y"
    if text.isdigit():
        return str(int(text))
    if text in {"X", "Y"}:
        return text
    return text


def variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Build standardized variant key used for cross-dataset merges."""
    return f"{chrom}:{pos}:{ref}:{alt}"


def to_float(value: Any) -> float | None:
    """Convert value to float safely; return None on invalid/missing values."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def to_list(value: Any) -> list[Any]:
    """Normalize INFO values to a Python list for multi-allelic handling."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    if isinstance(value, str):
        value = value.strip()
        if not value or value == ".":
            return []
        return value.split(",")
    return [value]


def get_alt_value(info_getter: Callable[[str], Any], key: str, alt_index: int) -> float | None:
    """Read an alt-specific INFO field value."""
    values = to_list(info_getter(key))
    if not values:
        return None
    if alt_index < len(values):
        return to_float(values[alt_index])
    if len(values) == 1:
        return to_float(values[0])
    return None


def get_scalar_value(info_getter: Callable[[str], Any], key: str) -> float | None:
    """Read a scalar INFO field value."""
    values = to_list(info_getter(key))
    if not values:
        return None
    return to_float(values[0])


def compute_af_popmax(info_getter: Callable[[str], Any], alt_index: int) -> float | None:
    """Compute AF_popmax from INFO fields or population-specific AF/AC/AN."""
    direct = get_alt_value(info_getter, "AF_popmax", alt_index)
    if direct is None:
        direct = get_alt_value(info_getter, "popmax_AF", alt_index)
    if direct is not None:
        return direct

    pop_af_values: list[float] = []
    for pop in POPULATIONS:
        af = None
        for af_key in (f"AF_{pop}", f"AF_{pop.lower()}"):
            af = get_alt_value(info_getter, af_key, alt_index)
            if af is not None:
                break

        if af is None:
            ac = None
            for ac_key in (f"AC_{pop}", f"AC_{pop.lower()}"):
                ac = get_alt_value(info_getter, ac_key, alt_index)
                if ac is not None:
                    break

            an = None
            for an_key in (f"AN_{pop}", f"AN_{pop.lower()}"):
                an = get_scalar_value(info_getter, an_key)
                if an is not None:
                    break

            if ac is not None and an is not None and an > 0:
                af = ac / an

        if af is not None:
            pop_af_values.append(af)

    if pop_af_values:
        return float(max(pop_af_values))
    return None


def normalize_input_format(input_path: Path, input_format: str | None) -> str:
    """Infer input format if not explicitly provided."""
    if input_format and input_format != "auto":
        return input_format.lower()

    name = input_path.name.lower()
    if name.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")):
        return "vcf"
    if name.endswith((".tsv", ".tsv.gz", ".csv", ".csv.gz", ".txt", ".txt.gz", ".parquet")):
        return "tsv"
    raise ValueError(
        "Could not auto-detect input format. Use --input-format vcf|tsv explicitly."
    )


def normalize_name(name: str) -> str:
    """Canonicalize column names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_clinvar_variant_set(path: Path) -> set[str]:
    """Load ClinVar variant_key set for memory-efficient filtering."""
    df = pd.read_parquet(path, columns=["variant_key"])
    keys = set(df["variant_key"].dropna().astype(str))
    print(f"Loaded ClinVar variant keys: {len(keys):,}")
    return keys


class StreamingParquetSink:
    """Incrementally write rows to parquet while tracking metadata statistics."""

    ordered_columns = ["variant_key", "AF", "AF_popmax", "AN", "AC", "log_AF", "is_common"]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.writer = None

        self.total_variants = 0
        self.common_variants_count = 0
        self.rare_variants_count = 0
        self.af_sum = 0.0
        self.af_count = 0
        self.af_values = array("d")

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col in ["variant_key", "AF", "AF_popmax", "AN", "AC"]:
            if col not in out.columns:
                out[col] = np.nan

        out["variant_key"] = out["variant_key"].astype(str)
        out["AF"] = pd.to_numeric(out["AF"], errors="coerce")
        out["AF_popmax"] = pd.to_numeric(out["AF_popmax"], errors="coerce")
        out["AN"] = pd.to_numeric(out["AN"], errors="coerce")
        out["AC"] = pd.to_numeric(out["AC"], errors="coerce")

        out["log_AF"] = np.log10(out["AF"].fillna(0.0) + EPSILON)
        out["is_common"] = out["AF"] > 0.01

        out = out[self.ordered_columns]
        out = out.dropna(subset=["variant_key"])
        out = out[out["variant_key"].str.len() > 0]
        return out

    def _update_stats(self, df: pd.DataFrame) -> None:
        self.total_variants += len(df)

        af_non_null = df["AF"].dropna()
        if not af_non_null.empty:
            self.af_sum += float(af_non_null.sum())
            self.af_count += int(len(af_non_null))
            self.af_values.extend(af_non_null.astype(float).tolist())

            self.common_variants_count += int((af_non_null > 0.01).sum())
            self.rare_variants_count += int((af_non_null <= 0.01).sum())

    def write_dataframe(self, df: pd.DataFrame) -> None:
        """Write prepared dataframe chunk and update running statistics."""
        if df.empty:
            return

        prepared = self._prepare_df(df)
        if prepared.empty:
            return

        self._update_stats(prepared)

        if pa is None or pq is None:
            raise ImportError("pyarrow is required to write parquet output")

        table = pa.Table.from_pandas(prepared, preserve_index=False)

        if self.writer is None:
            if self.output_path.exists():
                self.output_path.unlink()
            self.writer = pq.ParquetWriter(str(self.output_path), table.schema, compression="snappy")

        self.writer.write_table(table)

    def finalize(self) -> dict[str, Any]:
        """Ensure output exists and return summary stats for metadata JSON."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        else:
            empty_df = pd.DataFrame(columns=self.ordered_columns)
            empty_df.to_parquet(self.output_path, index=False)

        mean_af = (self.af_sum / self.af_count) if self.af_count else 0.0
        median_af = float(np.median(np.array(self.af_values))) if self.af_count else 0.0

        return {
            "total_variants": int(self.total_variants),
            "common_variants_count": int(self.common_variants_count),
            "rare_variants_count": int(self.rare_variants_count),
            "mean_AF": float(mean_af),
            "median_AF": float(median_af),
        }


def build_row(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    af: float | None,
    af_popmax: float | None,
    an: float | None,
    ac: float | None,
) -> dict[str, Any]:
    """Create one normalized output record."""
    return {
        "variant_key": variant_key(chrom, pos, ref, alt),
        "AF": af,
        "AF_popmax": af_popmax,
        "AN": an,
        "AC": ac,
    }


def extract_from_vcf(
    input_path: Path,
    sink: StreamingParquetSink,
    clinvar_keys: set[str] | None,
    progress_every: int = 100_000,
) -> None:
    """Stream variants from VCF using cyvcf2, then fallback to pysam."""
    try:
        from cyvcf2 import VCF  # type: ignore

        backend = "cyvcf2"
        reader = VCF(str(input_path))

        processed_records = 0
        rows_buffer: list[dict[str, Any]] = []

        for record in reader:
            processed_records += 1

            chrom = normalize_chromosome(record.CHROM)
            if chrom is None:
                continue

            pos = int(record.POS)
            ref = str(record.REF).upper()
            alts = [str(alt).upper() for alt in (record.ALT or []) if alt]

            info_getter = record.INFO.get
            an = get_scalar_value(info_getter, "AN")

            for alt_idx, alt in enumerate(alts):
                key = variant_key(chrom, pos, ref, alt)
                if clinvar_keys is not None and key not in clinvar_keys:
                    continue

                af = get_alt_value(info_getter, "AF", alt_idx)
                ac = get_alt_value(info_getter, "AC", alt_idx)
                af_popmax = compute_af_popmax(info_getter, alt_idx)
                rows_buffer.append(build_row(chrom, pos, ref, alt, af, af_popmax, an, ac))

            if len(rows_buffer) >= 100_000:
                sink.write_dataframe(pd.DataFrame(rows_buffer))
                rows_buffer = []

            if processed_records % progress_every == 0:
                print(
                    f"Processed {processed_records:,} VCF records "
                    f"(backend={backend})"
                )

        if rows_buffer:
            sink.write_dataframe(pd.DataFrame(rows_buffer))

        print(f"Finished VCF parsing with {backend}: {processed_records:,} records")
        return

    except ImportError:
        pass

    try:
        import pysam  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "VCF parsing requires cyvcf2 or pysam. Install one of them first."
        ) from exc

    backend = "pysam"
    reader = pysam.VariantFile(str(input_path))

    processed_records = 0
    rows_buffer: list[dict[str, Any]] = []

    for record in reader:
        processed_records += 1

        chrom = normalize_chromosome(record.chrom)
        if chrom is None:
            continue

        pos = int(record.pos)
        ref = str(record.ref).upper()
        alts = [str(alt).upper() for alt in (record.alts or []) if alt]

        info_getter = record.info.get
        an = get_scalar_value(info_getter, "AN")

        for alt_idx, alt in enumerate(alts):
            key = variant_key(chrom, pos, ref, alt)
            if clinvar_keys is not None and key not in clinvar_keys:
                continue

            af = get_alt_value(info_getter, "AF", alt_idx)
            ac = get_alt_value(info_getter, "AC", alt_idx)
            af_popmax = compute_af_popmax(info_getter, alt_idx)
            rows_buffer.append(build_row(chrom, pos, ref, alt, af, af_popmax, an, ac))

        if len(rows_buffer) >= 100_000:
            sink.write_dataframe(pd.DataFrame(rows_buffer))
            rows_buffer = []

        if processed_records % progress_every == 0:
            print(f"Processed {processed_records:,} VCF records (backend={backend})")

    if rows_buffer:
        sink.write_dataframe(pd.DataFrame(rows_buffer))

    print(f"Finished VCF parsing with {backend}: {processed_records:,} records")


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Find first matching column using normalized names."""
    normalized_map = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None


def standardize_table_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize pre-extracted TSV/CSV chunk to output schema."""
    out = pd.DataFrame(index=df.index)

    vk_col = find_column(df.columns, ["variant_key", "variant key"])
    chr_col = find_column(df.columns, ["chr", "chrom", "chromosome", "#chr"])
    pos_col = find_column(df.columns, ["pos", "position", "pos(1-based)", "start"])
    ref_col = find_column(df.columns, ["ref", "referenceallele", "reference", "referenceallelevcf"])
    alt_col = find_column(df.columns, ["alt", "alternateallele", "alternate", "alternateallelevcf"])

    if vk_col is None and (chr_col is None or pos_col is None or ref_col is None or alt_col is None):
        raise ValueError(
            "TSV/CSV input must include either variant_key or CHROM/POS/REF/ALT columns."
        )

    if vk_col is not None:
        out["variant_key"] = df[vk_col].astype(str)
    else:
        chrom = df[chr_col].map(normalize_chromosome)
        pos = pd.to_numeric(df[pos_col], errors="coerce")
        ref = df[ref_col].fillna("").astype(str).str.strip().str.upper()
        alt = df[alt_col].fillna("").astype(str).str.strip().str.upper()

        out["variant_key"] = [
            variant_key(c, int(p), r, a) if c and not pd.isna(p) and r and a else ""
            for c, p, r, a in zip(chrom, pos, ref, alt)
        ]

    af_col = find_column(df.columns, ["AF", "allele_frequency", "global_af", "gnomad_af"])
    af_popmax_col = find_column(df.columns, ["AF_popmax", "popmax_af", "af_popmax"])
    an_col = find_column(df.columns, ["AN", "allele_number", "sample_size"])
    ac_col = find_column(df.columns, ["AC", "allele_count"])

    out["AF"] = pd.to_numeric(df[af_col], errors="coerce") if af_col else np.nan
    out["AN"] = pd.to_numeric(df[an_col], errors="coerce") if an_col else np.nan
    out["AC"] = pd.to_numeric(df[ac_col], errors="coerce") if ac_col else np.nan

    if af_popmax_col:
        out["AF_popmax"] = pd.to_numeric(df[af_popmax_col], errors="coerce")
    else:
        pop_cols: list[str] = []
        for pop in POPULATIONS:
            candidate = find_column(df.columns, [f"AF_{pop}", f"AF_{pop.lower()}", f"af{pop.lower()}"])
            if candidate:
                pop_cols.append(candidate)

        if pop_cols:
            out["AF_popmax"] = df[pop_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        else:
            out["AF_popmax"] = out["AF"]

    return out


def extract_from_table(
    input_path: Path,
    sink: StreamingParquetSink,
    clinvar_keys: set[str] | None,
) -> None:
    """Read pre-extracted TSV/CSV/parquet and standardize to gnomAD AF schema."""
    name = input_path.name.lower()

    if name.endswith(".parquet"):
        chunk = pd.read_parquet(input_path)
        standardized = standardize_table_chunk(chunk)
        if clinvar_keys is not None:
            standardized = standardized[standardized["variant_key"].isin(clinvar_keys)]
        sink.write_dataframe(standardized)
        print(f"Processed parquet table rows: {len(chunk):,}")
        return

    sep = "\t" if ".tsv" in name or name.endswith(".txt") else ","
    total_rows = 0
    for idx, chunk in enumerate(
        pd.read_csv(input_path, sep=sep, low_memory=False, chunksize=200_000),
        start=1,
    ):
        standardized = standardize_table_chunk(chunk)
        if clinvar_keys is not None:
            standardized = standardized[standardized["variant_key"].isin(clinvar_keys)]

        sink.write_dataframe(standardized)
        total_rows += len(chunk)
        print(f"Processed table chunk {idx:,}: input_rows={len(chunk):,}, total_input={total_rows:,}")


def fetch_gnomad_af(variant_keys: list[str]) -> pd.DataFrame:
    """Fetch AF values for explicit variants using gnomAD GraphQL API.

    This method is slower than local VCF parsing but useful in Colab when full
    gnomAD downloads are impractical.

    Returns a DataFrame with columns:
    variant_key, AF, AF_popmax, AN, AC, log_AF, is_common
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("requests is required for API-based extraction") from exc

    endpoint = "https://gnomad.broadinstitute.org/api"
    query = """
    query VariantQuery($variantId: String!, $dataset: DatasetId!, $referenceGenome: ReferenceGenomeId!) {
      variant(variantId: $variantId, dataset: $dataset, referenceGenome: $referenceGenome) {
        exome {
          ac
          an
          af
          populations { id ac an af }
        }
        genome {
          ac
          an
          af
          populations { id ac an af }
        }
      }
    }
    """

    rows: list[dict[str, Any]] = []
    pop_ids = {p.lower() for p in POPULATIONS}

    for idx, key in enumerate(variant_keys, start=1):
        variant_id = key if key.startswith("chr") else f"chr{key}"
        payload = {
            "query": query,
            "variables": {
                "variantId": variant_id,
                "dataset": "gnomad_r2_1",
                "referenceGenome": "GRCh37",
            },
        }

        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", {}).get("variant", {})
        block = data.get("exome") or data.get("genome") or {}

        af = to_float(block.get("af"))
        an = to_float(block.get("an"))
        ac = to_float(block.get("ac"))

        pop_af_values = []
        for pop in block.get("populations", []) or []:
            pop_id = str(pop.get("id", "")).lower()
            if pop_id in pop_ids:
                pop_af = to_float(pop.get("af"))
                if pop_af is not None:
                    pop_af_values.append(pop_af)

        af_popmax = max(pop_af_values) if pop_af_values else af

        parts = key.split(":", 3)
        if len(parts) != 4:
            continue
        chrom, pos_text, ref, alt = parts
        pos = int(pos_text)
        rows.append(build_row(chrom, pos, ref, alt, af=af, af_popmax=af_popmax, an=an, ac=ac))

        if idx % 100 == 0:
            print(f"API progress: {idx:,}/{len(variant_keys):,}")

    out = pd.DataFrame(rows)
    out["AF"] = pd.to_numeric(out["AF"], errors="coerce")
    out["AF_popmax"] = pd.to_numeric(out["AF_popmax"], errors="coerce")
    out["AN"] = pd.to_numeric(out["AN"], errors="coerce")
    out["AC"] = pd.to_numeric(out["AC"], errors="coerce")
    out["log_AF"] = np.log10(out["AF"].fillna(0.0) + EPSILON)
    out["is_common"] = out["AF"] > 0.01
    return out[["variant_key", "AF", "AF_popmax", "AN", "AC", "log_AF", "is_common"]]


def save_metadata(metadata_path: Path, summary: dict[str, Any]) -> None:
    """Save gnomAD extraction metadata JSON."""
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "gnomAD",
        "version": "r2.1.1",
        "total_variants": summary["total_variants"],
        "common_variants_count": summary["common_variants_count"],
        "rare_variants_count": summary["rare_variants_count"],
        "mean_AF": summary["mean_AF"],
        "median_AF": summary["median_AF"],
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract gnomAD allele frequencies")
    parser.add_argument("--input", required=True, help="Path to gnomAD VCF or pre-extracted TSV/CSV")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "vcf", "tsv"],
        help="Input format (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output parquet path (default: data/intermediate/gnomad_af_clean.parquet)",
    )
    parser.add_argument(
        "--clinvar-variants",
        default=None,
        help="Optional path to clinvar_labeled_clean.parquet for variant filtering",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    input_path = resolve_path(repo_root, args.input)
    output_path = resolve_path(repo_root, args.output)
    metadata_path = resolve_path(repo_root, DEFAULT_METADATA)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_format = normalize_input_format(input_path, args.input_format)
    print(f"Input: {input_path}")
    print(f"Detected format: {input_format}")
    print(f"Output: {output_path}")

    clinvar_keys = None
    if args.clinvar_variants:
        clinvar_path = resolve_path(repo_root, args.clinvar_variants)
        if not clinvar_path.exists():
            raise FileNotFoundError(f"ClinVar variants file not found: {clinvar_path}")
        clinvar_keys = load_clinvar_variant_set(clinvar_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sink = StreamingParquetSink(output_path)

    if input_format == "vcf":
        extract_from_vcf(input_path=input_path, sink=sink, clinvar_keys=clinvar_keys)
    else:
        extract_from_table(input_path=input_path, sink=sink, clinvar_keys=clinvar_keys)

    summary = sink.finalize()
    save_metadata(metadata_path, summary)

    print("Extraction complete:")
    print(f"  total_variants={summary['total_variants']:,}")
    print(f"  common_variants_count={summary['common_variants_count']:,}")
    print(f"  rare_variants_count={summary['rare_variants_count']:,}")
    print(f"  mean_AF={summary['mean_AF']:.6g}")
    print(f"  median_AF={summary['median_AF']:.6g}")
    print(f"Saved parquet: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
