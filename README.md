# Mutation Detection

## Overview
This project detects and classifies human genetic variants using both classical ML and deep learning. The pipeline covers data ingestion, preprocessing, feature engineering, training, and evaluation for clinically relevant mutation classification.

## Installation
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in `notebooks/` or production code in `src/`.

## Data
Project directories:
- `data/raw/`: original downloaded files.
- `data/processed/`: cleaned and merged modeling tables.
- `data/splits/`: train/validation/test split artifacts.

Primary sources (priority order):

| Source | Role in project | Main file | Key columns | Priority |
| --- | --- | --- | --- | --- |
| ClinVar | Supervised labels for pathogenicity | `variant_summary.txt` | `GeneSymbol`, `ClinicalSignificance`, `Chromosome`, `Start`, `ReferenceAllele`, `AlternateAllele`, `ReviewStatus` | 1 (required) |
| gnomAD | Population allele frequency feature + benign proxy | Exomes sites-only VCF | `Chromosome`, `Position`, `Ref`, `Alt`, `AF`, `AC`, population AFs | 2 |
| dbNSFP | Precomputed functional and conservation features | `dbNSFP4.x` tables | `PhyloP`, `phastCons`, `GERP++`, `BLOSUM62`, `Grantham`, `PolyPhen2`, `SIFT` | 3 |
| UniProt | Protein sequences for ESM-2 inputs | Human proteome FASTA | `Protein ID`, `Gene Name`, `Sequence`, `Length`, annotation fields | 4 (ESM-2 track) |

Label policy:
- `Pathogenic` -> `1`
- `Benign` -> `0`
- `VUS` is excluded by default from supervised training unless explicitly enabled.

## Models
- `src/models/xgboost_model.py`: tabular baseline using engineered features.
- `src/models/cnn_model.py`: sequence-focused deep learning baseline.
- `src/models/esm2_model.py`: transformer embedding workflow using UniProt sequences.

Training and evaluation entry points:
- `src/training.py`
- `src/evaluation.py`

## Team
Genetic Graduation Project Team.

## License
To be finalized by the team. Current use is academic/research.
