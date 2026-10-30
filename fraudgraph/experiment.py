"""Shared split/feature recipe so Tier 1 (XGBoost) and Tier 2 (GNN) use identical rows."""
from dataclasses import dataclass

import pandas as pd

from fraudgraph.splits import Splits, label_known, make_splits, split_masks
from fraudgraph.tabular import tabular_features


@dataclass
class Prepared:
    splits: Splits
    train: pd.Series
    val: pd.Series
    test: pd.Series
    train_fit: pd.Series   # train rows whose label is known at train_end_day
    tab_cols: list[str]    # tabular columns, derived from the TRAIN frame
    tab: pd.DataFrame      # tabular features for every row, reindexed to tab_cols


def prepare(df: pd.DataFrame) -> Prepared:
    s = make_splits(df)
    train_m, val_m, test_m = split_masks(df, s)
    train_fit = train_m & label_known(df, s.train_end_day)
    tab_cols = list(tabular_features(df[train_m]).columns)
    tab = tabular_features(df, columns=tab_cols)
    return Prepared(s, train_m, val_m, test_m, train_fit, tab_cols, tab)
