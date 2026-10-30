"""Tier 4: card-sequence Transformer, same rows and metrics as Tiers 1 and 2."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fraudgraph.results as results
from fraudgraph.config import SEED, SEQ_LEN
from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys
from fraudgraph.experiment import prepare
from fraudgraph.gnn.graph import standardize
from fraudgraph.metrics import evaluate, pr_auc
from fraudgraph.seq.model import SeqTransformer
from fraudgraph.seq.windows import card_window_index

ROOT = Path(__file__).resolve().parent.parent


def _gather(X, idx, gap, rows):
    """Build (hist, mask, current) tensors for `rows` from the standardized X."""
    ix, gp = idx[rows], gap[rows]
    mask = ix < 0
    feats = X[np.where(mask, 0, ix)]
    hist = np.concatenate([feats, gp[..., None]], axis=2)
    hist[mask] = 0.0
    return (torch.from_numpy(hist), torch.from_numpy(mask),
            torch.from_numpy(X[rows]))


@torch.no_grad()
def predict(model, X, idx, gap, rows, batch_size=2048) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(*_gather(X, idx, gap, rows[i:i + batch_size]))).numpy()
           for i in range(0, len(rows), batch_size)]
    return np.concatenate(out) if out else np.zeros(0)


def run(df: pd.DataFrame, epochs: int = 20, lr: float = 1e-3, patience: int = 3,
        batch_size: int = 512, k: int = SEQ_LEN) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prep = prepare(df)
    fit = prep.train_fit.to_numpy()
    X = standardize(prep.tab.to_numpy(dtype=float), fit_rows=fit).astype(np.float32)
    idx, gap = card_window_index(df, entity_keys(df), k)

    y = df["isFraud"].to_numpy()
    train_rows = np.flatnonzero(fit)
    val_rows = np.flatnonzero(prep.val.to_numpy())
    test_rows = np.flatnonzero(prep.test.to_numpy())
    pos = int(y[train_rows].sum())
    neg = len(train_rows) - pos
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(neg / pos if pos else 1.0, dtype=torch.float32))

    model = SeqTransformer(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    y_val = y[val_rows]

    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(train_rows)
        for i in range(0, len(order), batch_size):
            rows = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(*_gather(X, idx, gap, rows)),
                           torch.from_numpy(y[rows].astype(np.float32)))
            loss.backward()
            opt.step()
        score = pr_auc(y_val, predict(model, X, idx, gap, val_rows))
        if best_state is None or score > best:   # first epoch always kept, never untrained
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    ev = {}
    for part, rows in (("val", val_rows), ("test", test_rows)):
        ev[part] = evaluate(df["isFraud"].iloc[rows], predict(model, X, idx, gap, rows),
                            df["TransactionAmt"].iloc[rows])
    payload = {"model": "seq_transformer", "features": prep.tab_cols,
               "n_train": int(fit.sum()), **ev}
    results.write_result("seq_transformer", payload)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    p = run(load_raw(a.raw_dir), epochs=a.epochs)
    print("seq_transformer test pr_auc", round(p["test"]["pr_auc"], 4))
