"""Per-card windows of prior transactions for the sequence model.

Leakage rule: a window holds only same-card rows with TransactionDT STRICTLY less than the
current row's, so a same-timestamp twin is never visible.
"""
import numpy as np
import pandas as pd

from fraudgraph.config import SEQ_LEN


def card_window_index(df: pd.DataFrame, keys: pd.DataFrame,
                      k: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, gap) aligned to df rows by position, without materialising features.

    idx: (N, k) int64 positional row index of each prior same-card txn, most recent last,
    -1 where padded (null card -> all -1). gap: (N, k) float32 log1p(seconds since that txn),
    0 where padded.
    """
    n = len(df)
    idx_out = np.full((n, k), -1, dtype=np.int64)
    gap_out = np.zeros((n, k), dtype=np.float32)
    dt = df["TransactionDT"].to_numpy(dtype=np.int64)
    card = pd.Series(keys["card"].to_numpy(dtype=object))
    ar = np.arange(k)
    for rows in card.groupby(card, sort=False).indices.values():
        order = rows[np.argsort(dt[rows], kind="stable")]  # positions, time ascending
        t = dt[order]
        # strictly-earlier count per row: side="left" excludes equal timestamps
        prior = np.searchsorted(t, t, side="left")
        off = prior[:, None] - k + ar[None, :]  # (m, k) offsets into `order`
        valid = off >= 0
        src = order[np.where(valid, off, 0)]
        gap = np.log1p(np.maximum(t[:, None] - dt[src], 0)).astype(np.float32)
        idx_out[order] = np.where(valid, src, -1)
        gap_out[order] = np.where(valid, gap, 0.0)
    return idx_out, gap_out


def card_windows(df: pd.DataFrame, keys: pd.DataFrame, X: np.ndarray,
                 k: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Return (hist, mask) aligned to df rows by position.

    hist: (N, k, F+1) float32 = prior features + log1p(seconds since that txn), most recent
    last, zeros where padded. mask: (N, k) bool, True where padded. Null card -> all padded.
    `X` is already standardized and aligned to df rows by position.
    """
    X = np.asarray(X, dtype=np.float32)
    idx, gap = card_window_index(df, keys, k)
    mask = idx < 0
    feats = np.concatenate([X[np.where(mask, 0, idx)], gap[..., None]], axis=2)
    hist = np.where(mask[..., None], np.float32(0.0), feats).astype(np.float32)
    return hist, mask
