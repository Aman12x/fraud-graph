"""Weekly drift replay through the online path, and k6 payload dump.

Every pre-test row (train + val) is replayed through EntityStore in time order
(features -> score -> observe -> ingest_label) so live state matches offline. Val scores from
that pass are the PSI reference. Then each test week (day // 7) is scored the same way.
Features are built row by row through the store; scoring is batched, one predict_proba per week.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import fraudgraph.results as results
from fraudgraph.data import load_raw
from fraudgraph.drift import psi
from fraudgraph.metrics import pr_auc
from fraudgraph.models.xgb import load_model
from fraudgraph.serve.app import feature_frame, tabular_part
from fraudgraph.serve.store import EntityStore
from fraudgraph.splits import make_splits, split_masks

ROOT = Path(__file__).resolve().parent.parent


def txn_body(rec: dict) -> dict:
    """A /score request body: no label/day, NaN dropped, numpy scalars to Python."""
    out = {}
    for k, v in rec.items():
        if k in ("isFraud", "day"):
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out


def _replay_rows(df, store, model, cols, tab_cols, score_mask, chunk):
    """Replay rows through the store in time order; score the masked rows one chunk at a time.

    Per row, in order: store.features (state strictly before the row), observe, ingest_label,
    exactly as one /score + /label call would. Predictions never touch the store, so a chunk's
    masked rows are collected and scored with one predict_proba when the chunk key changes.
    `chunk` must be non-decreasing (week = day // 7 over time-sorted rows), so one chunk is
    held in memory at a time.
    """
    scores: list[np.ndarray] = []
    bodies: list[dict] = []
    graph_rows: list[dict] = []

    def flush():
        if bodies:
            x = feature_frame(bodies, graph_rows, tab_cols, cols)
            scores.append(model.predict_proba(x)[:, 1])
            bodies.clear()
            graph_rows.clear()

    prev = None
    for rec, want, k in zip(df.to_dict("records"), score_mask, chunk):
        if prev is not None and k < prev:
            raise ValueError("chunk keys must be non-decreasing (rows in time order)")
        if k != prev:
            flush()
            prev = k
        body = txn_body(rec)
        g = store.features(body)
        if want:
            bodies.append(body)
            graph_rows.append(g)
        store.observe(body)
        store.ingest_label(body["TransactionID"], bool(rec["isFraud"]))
    flush()
    return np.concatenate(scores) if scores else np.empty(0)


def run(df: pd.DataFrame, model, feature_columns: list[str]) -> list[dict]:
    train_m, val_m, test_m = split_masks(df, make_splits(df))
    tab_cols = tabular_part(feature_columns)
    store = EntityStore()
    pre = df[train_m | val_m]
    ref = _replay_rows(pre, store, model, feature_columns, tab_cols, val_m[pre.index].to_numpy(),
                       (pre["day"] // 7).to_numpy())
    test = df[test_m]
    week = (test["day"] // 7).to_numpy()
    cur = _replay_rows(test, store, model, feature_columns, tab_cols, np.ones(len(test), bool), week)
    y = test["isFraud"].to_numpy()
    out = []
    for w in np.unique(week):
        m = week == w
        both = 0 < y[m].sum() < m.sum()
        out.append({"week": int(w), "n": int(m.sum()), "psi": psi(ref, cur[m]),
                    "pr_auc": float(pr_auc(y[m], cur[m])) if both else None})
    return out


def dump_payloads(df: pd.DataFrame, n: int, path: Path) -> int:
    _, _, test_m = split_masks(df, make_splits(df))
    rows = [txn_body(r) for r in df[test_m].head(n).to_dict("records")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "xgb_graph.ubj")
    ap.add_argument("--dump-payloads", type=int, default=0, metavar="N")
    a = ap.parse_args()
    df = load_raw(a.raw_dir)
    if a.dump_payloads:
        print("wrote", dump_payloads(df, a.dump_payloads, ROOT / "bench" / "payloads.json"), "payloads")
    else:
        cols = json.loads(a.model.with_name(a.model.stem + ".features.json").read_text())
        weeks = run(df, load_model(a.model), cols)
        results.write_result("drift", {"weeks": weeks})
        print(len(weeks), "weeks written")
