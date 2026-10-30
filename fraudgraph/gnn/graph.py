"""Heterogeneous transaction-entity graph and temporal neighbour sampling.

Leakage contract: labels live only in data["txn"].y, never in any node's x; a seed
transaction's sampled subgraph holds only transactions with time <= the seed's time
(entity nodes have time -1, so they are always admissible).
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from fraudgraph.entities import ENTITY_TYPES

ENTITY_TIME = -1


def standardize(X: np.ndarray, fit_rows=None) -> np.ndarray:
    """(X - mean) / std with stats over X[fit_rows] (all rows if None), ignoring NaN.

    Zero-variance or all-NaN columns get std 1; NaN becomes 0 after standardizing.
    """
    X = np.asarray(X, dtype=float)
    fit = X if fit_rows is None else X[fit_rows]
    ok = ~np.isnan(fit)
    n = ok.sum(axis=0)
    filled = np.where(ok, fit, 0.0)
    mean = np.divide(filled.sum(axis=0), n, out=np.zeros(X.shape[1]), where=n > 0)
    var = np.divide((np.where(ok, fit - mean, 0.0) ** 2).sum(axis=0), n,
                    out=np.zeros(X.shape[1]), where=n > 0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    out = (X - mean) / std
    out[np.isnan(out)] = 0.0
    return out


def build_hetero(df: pd.DataFrame, keys: pd.DataFrame, X: np.ndarray, *,
                 fit_rows) -> HeteroData:
    """Transactions (row order of df) linked to one node per distinct non-null entity key.

    X is the RAW numeric feature matrix aligned with df (do not pre-standardize it); it is
    standardized exactly once here, with stats over `fit_rows` (required, e.g. the train
    mask, so val/test statistics never reach the features).
    """
    if len(X) != len(df) or len(keys) != len(df):
        raise ValueError("df, keys and X must have the same number of rows")
    data = HeteroData()
    data["txn"].x = torch.as_tensor(standardize(X, fit_rows), dtype=torch.float)
    data["txn"].y = torch.as_tensor(df["isFraud"].to_numpy(), dtype=torch.long)
    data["txn"].time = torch.as_tensor(df["TransactionDT"].to_numpy(), dtype=torch.long)
    data["txn"].amt = torch.as_tensor(df["TransactionAmt"].to_numpy(), dtype=torch.float)
    for t in ENTITY_TYPES:
        col = keys[t].to_numpy(dtype=object)
        present = np.flatnonzero(pd.notna(col))
        codes, uniques = pd.factorize(col[present])
        n = len(uniques)
        data[t].x = torch.ones(n, 1)
        data[t].time = torch.full((n,), ENTITY_TIME, dtype=torch.long)
        edge = torch.as_tensor(np.stack([present, codes]), dtype=torch.long)
        data["txn", f"has_{t}", t].edge_index = edge
        data[t, f"rev_has_{t}", "txn"].edge_index = edge.flip(0)
    return data


def make_loader(data: HeteroData, seed_idx, batch_size: int = 1024,
                num_neighbors=(15, 10), shuffle: bool = False) -> NeighborLoader:
    """Per-seed (disjoint) temporal subgraphs: only transactions with time <= seed time.

    seed_idx is an index array or a boolean mask over txn nodes.
    """
    seeds = torch.as_tensor(np.asarray(seed_idx))
    if seeds.dtype == torch.bool:
        seeds = seeds.nonzero().flatten()
    return NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=("txn", seeds.long()),
        batch_size=batch_size,
        shuffle=shuffle,
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
    )
