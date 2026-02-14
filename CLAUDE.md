# AI Agent Instructions

## Project Goal
Build robust mutation classification pipelines using four coordinated data sources:
1. ClinVar (labels)
2. gnomAD (population frequency)
3. dbNSFP (precomputed variant features)
4. UniProt (protein sequence for ESM-2)

## Repository Map
- `data/raw/`: untouched source files.
- `data/processed/`: cleaned and merged tables.
- `data/splits/`: train/val/test split outputs.
- `notebooks/`: staged workflow notebooks (01-06).
- `src/`: reusable Python modules.
- `configs/config.yaml`: data and training configuration.
- `results/figures/`, `results/checkpoints/`, `results/metrics/`: experiment outputs.

## Data Source Contract
### ClinVar (priority 1)
- Core supervised labels.
- Required file: `variant_summary.txt`.
- Required fields: `GeneSymbol`, `ClinicalSignificance`, `Chromosome`, `Start`, `ReferenceAllele`, `AlternateAllele`, `ReviewStatus`.
- Label mapping: `Pathogenic` -> `1`, `Benign` -> `0`, `VUS` dropped by default.

### gnomAD (priority 2)
- Frequency feature (`AF`, `AC`, population AF columns).
- Use as benign proxy when `AF > 0.01` where appropriate.
- Join on genomic variant key: chromosome + position + ref + alt.

### dbNSFP (priority 3)
- Pull precomputed features instead of recalculating manually.
- Focus columns: `PhyloP`, `phastCons`, `GERP++`, `BLOSUM62`, `Grantham`, `PolyPhen2`, `SIFT`.

### UniProt (priority 4, ESM-2 workflow)
- Use human proteome FASTA to provide protein sequences.
- Needed for sequence-based embedding extraction with ESM-2.

## Working Rules
1. Keep all source data immutable in `data/raw/`.
2. Write all merged/training tables to `data/processed/`.
3. Prefer deterministic preprocessing and document filters in code.
4. Keep notebooks for exploration; production logic must live in `src/`.
5. Save metrics and artifacts with timestamps in `results/`.

## Coding Conventions
- Python 3.10+.
- Type hints and concise docstrings.
- No hardcoded absolute paths.
- Keep model interfaces consistent (`fit`, `predict`, optional `predict_proba`).

## Execution Priorities
1. Build stable ClinVar-labeled baseline first.
2. Add gnomAD and dbNSFP features to improve tabular models.
3. Add UniProt + ESM-2 workflow as advanced track.
4. Optimize only after a reproducible baseline is established.
