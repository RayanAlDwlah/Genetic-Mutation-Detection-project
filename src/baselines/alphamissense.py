"""Extract AlphaMissense scores for a given set of (chr, pos, ref, alt) variants.

AlphaMissense publishes a per-missense-variant TSV for GRCh37 and GRCh38
that is ~600 MB gzipped. Every row is
`CHROM \\t POS \\t REF \\t ALT \\t genome \\t uniprot_id \\t transcript_id
 \\t protein_variant \\t am_pathogenicity \\t am_class`.

Instead of tabix-indexing (no published index), we stream-scan the file
once, building an in-memory lookup for variants we care about
(~28k test + ~644 denovo-db = ~29k positions). A single scan over the
full file takes ~3-5 min on a laptop and the result fits in RAM
(< 20 MB parquet). We cache it to
`data/intermediate/baselines/alphamissense_lookup.parquet` so re-runs
skip the scan.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Column names in the published AlphaMissense TSV.
_AM_COLS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]


@dataclass(frozen=True)
class AlphaMissenseLookup:
    scores: pd.DataFrame  # columns: variant_key, am_pathogenicity, am_class
    coverage: float  # fraction of query keys that got a score


def _canonical_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Normalize to the `chr:pos:ref:alt` convention used elsewhere."""
    c = str(chrom).removeprefix("chr").upper()
    if c in {"M", "MT"}:
        c = "MT"
    return f"{c}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def extract_scores_for_keys(
    tsv_gz_path: Path,
    keys: set[str],
    *,
    progress: bool = True,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream-scan the AlphaMissense TSV and emit rows whose variant_key
    is in `keys`.

    Returns a DataFrame with columns
    `variant_key, am_pathogenicity, am_class`. Missing query keys are
    simply absent from the output — caller should left-join and
    record coverage.
    """
    if progress:
        print(f"[AM] scanning {tsv_gz_path} for {len(keys):,} query variants…")
    hits: list[dict[str, object]] = []
    found_keys: set[str] = set()
    rows_scanned = 0

    # AlphaMissense TSVs have a leading `#CHROM` header comment line.
    with gzip.open(tsv_gz_path, "rt") as fh:
        # Skip the meta/header comment block — any line starting with `#`.
        while True:
            pos_before = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.startswith("#"):
                # Seek back so pandas can read the data row too.
                fh.seek(pos_before)
                break

        reader = pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=_AM_COLS,
            dtype={
                "chrom": str,
                "pos": "int64",
                "ref": str,
                "alt": str,
                "am_pathogenicity": "float32",
                "am_class": str,
                "genome": str,
                "uniprot_id": str,
                "transcript_id": str,
                "protein_variant": str,
            },
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk_i, chunk in enumerate(reader, 1):
            rows_scanned += len(chunk)
            # Build canonical keys for the chunk and filter.
            vk = (
                chunk["chrom"].str.removeprefix("chr").str.upper()
                + ":"
                + chunk["pos"].astype(str)
                + ":"
                + chunk["ref"].str.upper()
                + ":"
                + chunk["alt"].str.upper()
            )
            mask = vk.isin(keys)
            if mask.any():
                sel = chunk.loc[mask, ["am_pathogenicity", "am_class"]].copy()
                sel.insert(0, "variant_key", vk[mask].to_numpy())
                hits.append(sel)
                found_keys.update(vk[mask].tolist())
            if progress and chunk_i % 10 == 0:
                print(
                    f"  [AM] chunk {chunk_i} — {rows_scanned:,} rows scanned, "
                    f"{len(found_keys):,}/{len(keys):,} query keys hit"
                )
            # Early exit: if we've already seen every queried key, stop scanning.
            if len(found_keys) == len(keys):
                if progress:
                    print(
                        f"  [AM] all {len(keys):,} keys hit after "
                        f"{rows_scanned:,} rows — stopping early"
                    )
                break

    if not hits:
        return pd.DataFrame(columns=["variant_key", "am_pathogenicity", "am_class"])
    out = pd.concat(hits, ignore_index=True).drop_duplicates("variant_key")
    if progress:
        print(
            f"[AM] final hits: {len(out):,} / {len(keys):,} = "
            f"{len(out) / max(len(keys), 1):.1%}"
        )
    return out


def build_lookup(
    *,
    tsv_gz_path: Path,
    query_df: pd.DataFrame,
    cache_path: Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> AlphaMissenseLookup:
    """Produce a `variant_key -> am_pathogenicity` lookup for the variants
    in `query_df` (must have `chr, pos, ref, alt` or `variant_key` column).

    If `cache_path` exists and `force=False`, the cache is loaded and
    the scan is skipped — so the slow first-scan cost is paid once.
    """
    if cache_path is not None and cache_path.exists() and not force:
        scores = pd.read_parquet(cache_path)
        coverage = len(scores) / max(len(query_df), 1)
        if progress:
            print(f"[AM] cache hit: {cache_path} ({len(scores):,} rows)")
        return AlphaMissenseLookup(scores=scores, coverage=coverage)

    if "variant_key" in query_df.columns:
        keys = set(query_df["variant_key"].astype(str))
    else:
        keys = {
            _canonical_key(r["chr"], r["pos"], r["ref"], r["alt"]) for _, r in query_df.iterrows()
        }

    scores = extract_scores_for_keys(tsv_gz_path, keys, progress=progress)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(cache_path, index=False)
        if progress:
            print(f"[AM] cache saved: {cache_path}")

    return AlphaMissenseLookup(
        scores=scores,
        coverage=len(scores) / max(len(query_df), 1),
    )
