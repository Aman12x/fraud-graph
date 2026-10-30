"""Point-in-time entity features: prior activity and label-known fraud counts per entity.

For row i with entity value v (strictly earlier = smaller TransactionDT):
  prior_n     = same-v rows with TransactionDT < DT_i
  prior_fraud = same-v rows with isFraud == 1 and day <= day_i - delay
  fraud_rate  = (prior_fraud + RATE_PRIOR_FRAUD) / (prior_labeled + RATE_PRIOR_TOTAL)
where prior_labeled = same-v rows with day <= day_i - delay. Null v -> 0, 0, 1/20.
"""
import numpy as np
import pandas as pd

from fraudgraph.config import LABEL_DELAY_DAYS
from fraudgraph.entities import ENTITY_TYPES

RATE_PRIOR_FRAUD = 1
RATE_PRIOR_TOTAL = 20
FEATURE_SUFFIXES = ("prior_n", "prior_fraud", "fraud_rate")


def feature_columns() -> list[str]:
    return [f"{t}_{s}" for t in ENTITY_TYPES for s in FEATURE_SUFFIXES]


def _entity_features(v: pd.Series, dt: pd.Series, day: pd.Series, y: pd.Series, delay: int):
    n = len(v)
    prior_n = np.zeros(n, dtype=np.int64)
    prior_fraud = np.zeros(n, dtype=np.int64)
    prior_labeled = np.zeros(n, dtype=np.int64)
    valid = v.notna().to_numpy()
    if valid.any():
        pos = np.flatnonzero(valid)
        sub = pd.DataFrame({"v": v.to_numpy()[pos], "dt": dt.to_numpy()[pos],
                            "day": day.to_numpy(dtype=np.int64)[pos],
                            "y": y.to_numpy(dtype=np.int64)[pos], "pos": pos})
        # strict prior count: rank "min" counts rows with a smaller DT, ties excluded
        prior_n[pos] = (sub.groupby("v")["dt"].rank(method="min").to_numpy() - 1).astype(np.int64)
        # per (v, day) totals, cumulated within v: rows of v with day <= key day
        agg = sub.groupby(["v", "day"], sort=True).agg(cnt=("y", "size"), fr=("y", "sum"))
        agg = agg.groupby(level="v").cumsum().reset_index()
        agg = agg.sort_values("day", kind="mergesort")
        left = sub[["v", "pos"]].assign(q=sub["day"] - delay).sort_values("q", kind="mergesort")
        m = pd.merge_asof(left, agg, left_on="q", right_on="day", by="v", direction="backward")
        prior_labeled[m["pos"].to_numpy()] = m["cnt"].fillna(0).to_numpy(dtype=np.int64)
        prior_fraud[m["pos"].to_numpy()] = m["fr"].fillna(0).to_numpy(dtype=np.int64)
    rate = (prior_fraud + RATE_PRIOR_FRAUD) / (prior_labeled + RATE_PRIOR_TOTAL)
    return prior_n, prior_fraud, rate.astype(np.float64)


def graph_features(df: pd.DataFrame, keys: pd.DataFrame, delay: int = LABEL_DELAY_DAYS) -> pd.DataFrame:
    """Per-entity point-in-time features, indexed like `df`. Labels with day > day_i - delay are never read."""
    keys = keys.reindex(df.index)
    out = {}
    for t in ENTITY_TYPES:
        n, f, r = _entity_features(keys[t], df["TransactionDT"], df["day"], df["isFraud"], delay)
        out[f"{t}_prior_n"] = n
        out[f"{t}_prior_fraud"] = f
        out[f"{t}_fraud_rate"] = r
    return pd.DataFrame(out, index=df.index)[feature_columns()]
