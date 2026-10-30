"""XGBoost classifier for the Tier 1 experiment."""
from pathlib import Path

import xgboost as xgb

from fraudgraph.config import SEED


def train_xgb(X, y, X_val, y_val, n_estimators: int = 2000) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        early_stopping_rounds=100, tree_method="hist", random_state=SEED,
    )
    m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    return m


def save_model(m: xgb.XGBClassifier, path) -> None:
    m.save_model(str(path))


def load_model(path) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m
