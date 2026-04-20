"""Shared pytest fixtures for the missense-classifier test suite.

Anything a test needs from the real repo (splits parquet, metrics CSV,
etc.) should be wrapped in a fixture here — never hard-coded paths in
individual tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic numpy RNG (seed=42) for mock-data generation."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def train_split(repo_root: Path) -> pd.DataFrame:
    """Committed training split parquet. Required for integration tests."""
    path = repo_root / "data/splits/train.parquet"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing committed training split at {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def val_split(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/splits/val.parquet"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing committed val split at {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def test_split(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/splits/test.parquet"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing committed test split at {path}")
    return pd.read_parquet(path)


@pytest.fixture
def mock_binary_predictions(rng: np.random.Generator):
    """(y_true, y_prob) pair with known class balance and a signal.

    Returns labels (1,000 rows, 30% positive) and probabilities that
    concentrate higher for positives so the classifier is clearly
    non-trivial — useful for verifying that metrics cross sanity thresholds
    (ROC > 0.7, PR > 0.5, etc.) without depending on a trained model.
    """
    n = 1000
    y_true = (rng.uniform(size=n) < 0.30).astype(int)
    # Positives drawn from Beta(5,2), negatives from Beta(2,5)
    y_prob = np.where(
        y_true == 1,
        rng.beta(5, 2, size=n),
        rng.beta(2, 5, size=n),
    )
    return y_true, y_prob
