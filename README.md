# Mutation Detection

## Overview
This project focuses on detecting and classifying genetic mutations using both classical machine learning and deep learning approaches. The workflow covers data collection, preprocessing, feature engineering, model training, and evaluation for clinically relevant variant prediction tasks.

## Installation
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start working with notebooks or scripts in `src/`.

## Data
- `data/raw/`: original datasets (e.g., ClinVar, gnomAD, and other external sources).
- `data/processed/`: cleaned and transformed datasets ready for modeling.
- `data/splits/`: train/validation/test split files.

Keep raw and large derived data out of git, and document data versions when you update datasets.

## Models
The project includes multiple modeling tracks:
- `xgboost_model.py`: gradient-boosted tree baseline.
- `cnn_model.py`: convolutional neural network for sequence-based learning.
- `esm2_model.py`: transformer-based modeling using ESM2 representations.

Training and evaluation pipelines are orchestrated through `src/training.py` and `src/evaluation.py`.

## Team
Genetic Graduation Project Team.

Suggested roles:
- Data engineering and preprocessing
- Modeling and experimentation
- Evaluation and reporting

## License
License is to be defined by the team. For now, this repository is intended for academic/research use.
