"""Fit-free tabular features: each row's output depends only on that row."""
import zlib

import numpy as np
import pandas as pd

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]
_EXCLUDED = {"TransactionID", "isFraud", "TransactionDT", "day", "TransactionAmt", *CATEGORICAL}


def _hash_code(v) -> int:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return -1
    return zlib.crc32(str(v).encode()) % 2**31


def tabular_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amt = df["TransactionAmt"].astype(float)
    out["TransactionAmt"] = amt
    out["log_amt"] = np.log1p(amt)
    out["hour"] = (df["TransactionDT"] // 3600) % 24
    numeric = [c for c in df.columns
               if c not in _EXCLUDED and pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    for c in sorted(numeric):
        out[c] = df[c]
    for c in CATEGORICAL:
        if c in df.columns:
            out[c] = [_hash_code(v) for v in df[c].tolist()]
        else:
            out[c] = -1
    if columns is not None:
        out = out.reindex(columns=columns)
    return out
