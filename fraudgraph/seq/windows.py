"""Per-card windows of prior transactions for the sequence model.

Leakage rule: a window holds only same-card rows with TransactionDT STRICTLY less than the
current row's, so a same-timestamp twin is never visible.
"""
import numpy as np
import pandas as pd

from fraudgraph.config import SEQ_LEN


def card_windows(df: pd.DataFrame, keys: pd.DataFrame, X: np.ndarray,
                 k: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Return (hist, mask) aligned to df rows by position.

    hist: (N, k, F+1) float32 = prior features + log1p(seconds since that txn), most recent
    last, zeros where padded. mask: (N, k) bool, True where padded. Null card -> all padded.
    `X` is already standardized and aligned to df rows by position.
    """
    n = len(df)
    X = np.asarray(X, dtype=np.float32)
    hist = np.zeros((n, k, X.shape[1] + 1), dtype=np.float32)
    mask = np.ones((n, k), dtype=bool)
    dt = df["TransactionDT"].to_numpy(dtype=np.int64)
    card = pd.Series(keys["card"].to_numpy(dtype=object))
    ar = np.arange(k)
    for rows in card.groupby(card, sort=False).indices.values():
        order = rows[np.argsort(dt[rows], kind="stable")]  # positions, time ascending
        t = dt[order]
        # strictly-earlier count per row: side="left" excludes equal timestamps
        prior = np.searchsorted(t, t, side="left")
        idx = prior[:, None] - k + ar[None, :]  # (m, k) offsets into `order`
        valid = idx >= 0
        src = order[np.where(valid, idx, 0)]
        gap = np.log1p(np.maximum(t[:, None] - dt[src], 0)).astype(np.float32)
        feats = np.concatenate([X[src], gap[..., None]], axis=2)
        hist[order] = np.where(valid[..., None], feats, 0.0)
        mask[order] = ~valid
    return hist, mask
