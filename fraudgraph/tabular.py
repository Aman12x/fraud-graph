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
    """Fit-free features for each row of df.

    ONLINE SCORING MUST PASS `columns=` (the training column list). Without it the
    numeric feature set depends on which columns the frame has (and an all-None
    numeric column arrives as object dtype and is dropped). With it, the output is
    reindexed to `columns` and requested numeric columns are coerced to float, so one
    row's values equal that row's values in the full-frame output.
    """
    # Whole-block operations only: building ~400 columns one insert (or one Series lookup)
    # at a time cost ~40 ms per 1-row call and a PerformanceWarning per insert.
    amt = df["TransactionAmt"].astype(float)
    head = pd.DataFrame({"TransactionAmt": amt, "log_amt": np.log1p(amt),
                         "hour": (df["TransactionDT"] // 3600) % 24}, index=df.index)
    numeric = sorted(c for c, d in df.dtypes.items()
                     if c not in _EXCLUDED and pd.api.types.is_numeric_dtype(d)
                     and not pd.api.types.is_bool_dtype(d))
    blocks = [head, df[numeric].astype(float)]
    blocks.append(pd.DataFrame(
        {c: [_hash_code(v) for v in df[c].tolist()] if c in df.columns else -1 for c in CATEGORICAL},
        index=df.index))
    if columns is not None:
        have = {*head.columns, *numeric, *CATEGORICAL}
        extra = [c for c in dict.fromkeys(columns)
                 if c not in have and c in df.columns and c not in _EXCLUDED]
        if extra:
            blocks.append(df[extra].apply(pd.to_numeric, errors="coerce").astype(float))
    out = pd.concat(blocks, axis=1)
    if columns is not None:
        out = out.reindex(columns=columns)
    return out
