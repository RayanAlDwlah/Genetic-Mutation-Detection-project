"""XGBoost model wrapper."""

from xgboost import XGBClassifier


def build_xgboost_model(random_state: int = 42) -> XGBClassifier:
    """Create a default XGBoost classifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        eval_metric="logloss",
    )
