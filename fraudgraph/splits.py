from dataclasses import dataclass

import pandas as pd

from fraudgraph.config import LABEL_DELAY_DAYS


@dataclass
class Splits:
    train_end_day: int
    val_end_day: int


def make_splits(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> Splits:
    """Integer day boundaries from quantiles of per-row `day`; each day lands in one split."""
    train_end = int(df["day"].quantile(train_frac, interpolation="lower"))
    val_end = int(df["day"].quantile(train_frac + val_frac, interpolation="lower"))
    return Splits(train_end_day=train_end, val_end_day=val_end)


def split_masks(df: pd.DataFrame, s: Splits) -> tuple[pd.Series, pd.Series, pd.Series]:
    day = df["day"]
    train = day <= s.train_end_day
    val = (day > s.train_end_day) & (day <= s.val_end_day)
    test = day > s.val_end_day
    return train, val, test


def label_known(df: pd.DataFrame, as_of_day: int, delay: int = LABEL_DELAY_DAYS) -> pd.Series:
    return df["day"] <= as_of_day - delay
