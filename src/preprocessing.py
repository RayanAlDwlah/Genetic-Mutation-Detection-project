"""Preprocessing helpers for mutation detection datasets."""

import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning steps before feature engineering."""
    return df.drop_duplicates().reset_index(drop=True)
