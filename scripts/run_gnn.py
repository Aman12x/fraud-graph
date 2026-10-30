"""Tier 2: heterogeneous GraphSAGE, same rows and metrics as Tier 1."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fraudgraph.results as results
from fraudgraph.config import SEED
from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys
from fraudgraph.experiment import prepare
from fraudgraph.gnn.graph import build_hetero, make_loader
from fraudgraph.gnn.model import HeteroSAGE
from fraudgraph.metrics import evaluate, pr_auc

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict(model, loader) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(b)).numpy() for b in loader])


def run(df: pd.DataFrame, epochs: int = 20, lr: float = 1e-3, patience: int = 3,
        batch_size: int = 1024) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prep = prepare(df)
    data = build_hetero(df, entity_keys(df), prep.tab.to_numpy(dtype=float),
                        fit_rows=prep.train_fit.to_numpy())
    train_l = make_loader(data, prep.train_fit.to_numpy(), batch_size, shuffle=True)
    val_l = make_loader(data, prep.val.to_numpy(), batch_size)
    test_l = make_loader(data, prep.test.to_numpy(), batch_size)

    y = df["isFraud"].to_numpy()
    pos = int(y[prep.train_fit.to_numpy()].sum())
    neg = int(prep.train_fit.sum()) - pos
    pos_weight = torch.tensor(neg / pos if pos else 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HeteroSAGE(data.metadata())
    with torch.no_grad():                      # materialise lazy layers
        model(next(iter(train_l)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y_val = y[prep.val.to_numpy()]

    best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for b in train_l:
            opt.zero_grad()
            logits = model(b)
            loss = loss_fn(logits, b["txn"].y[: b["txn"].batch_size].float())
            loss.backward()
            opt.step()
        score = pr_auc(y_val, predict(model, val_l))
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    ev, thr = {}, None
    for part, mask, loader in (("val", prep.val, val_l), ("test", prep.test, test_l)):
        m = mask.to_numpy()            # val first: its cost-optimal threshold is applied to test
        ev[part] = evaluate(df.loc[m, "isFraud"], predict(model, loader),
                            df.loc[m, "TransactionAmt"], threshold=thr)
        thr = ev["val"]["best_cost"]["threshold"]
    payload = {"model": "gnn_sage", "features": prep.tab_cols,
               "n_train": int(prep.train_fit.sum()), **ev}
    results.write_result("gnn_sage", payload)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    p = run(load_raw(a.raw_dir), epochs=a.epochs)
    print("gnn_sage test pr_auc", round(p["test"]["pr_auc"], 4))
