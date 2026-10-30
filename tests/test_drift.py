import json
import math

import numpy as np
import pandas as pd

from fraudgraph.data import load_raw
from fraudgraph.drift import psi
from fraudgraph.entities import entity_keys
from fraudgraph.graph_features import graph_features
from fraudgraph.models.xgb import train_xgb
from fraudgraph.splits import make_splits, split_masks
from fraudgraph.tabular import tabular_features


def test_psi_zero_on_identical():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) == 0.0


def test_psi_positive_on_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=1000), rng.normal(2.0, 1.0, size=1000)) > 0.25


def test_psi_handles_empty_bins():
    v = psi(np.linspace(0, 1, 100), np.full(50, 5.0))  # every cur value in the top bin
    assert math.isfinite(v) and v > 0


def test_replay_on_fixture(raw_dir, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "replay", Path(__file__).resolve().parent.parent / "scripts" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dump_payloads, run = mod.dump_payloads, mod.run
    df = load_raw(raw_dir)
    X = pd.concat([tabular_features(df), graph_features(df, entity_keys(df))], axis=1)
    model = train_xgb(X, df["isFraud"], X, df["isFraud"], n_estimators=20)
    out = run(df, model, list(X.columns))
    _, _, test_m = split_masks(df, make_splits(df))
    weeks = sorted((df.loc[test_m, "day"] // 7).unique())
    assert [e["week"] for e in out] == [int(w) for w in weeks]
    assert sum(e["n"] for e in out) == int(test_m.sum())
    assert all(math.isfinite(e["psi"]) for e in out)
    assert all(e["pr_auc"] is None or 0 <= e["pr_auc"] <= 1 for e in out)
    n = dump_payloads(df, 5000, tmp_path / "payloads.json")
    rows = json.loads((tmp_path / "payloads.json").read_text())
    assert n == len(rows) == int(test_m.sum())
    assert len({r["TransactionID"] for r in rows}) == n
    assert all(not any(isinstance(v, float) and math.isnan(v) for v in r.values()) for r in rows)


def test_replay_batched_scores_equal_per_row_online_scores(raw_dir):
    import importlib.util
    from pathlib import Path

    from fraudgraph.serve.app import feature_vector, tabular_part
    from fraudgraph.serve.store import EntityStore
    spec = importlib.util.spec_from_file_location(
        "replay", Path(__file__).resolve().parent.parent / "scripts" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = load_raw(raw_dir)
    X = pd.concat([tabular_features(df), graph_features(df, entity_keys(df))], axis=1)
    model = train_xgb(X, df["isFraud"], X, df["isFraud"], n_estimators=20)
    cols = list(X.columns)
    tab = tabular_part(cols)
    want = np.asarray(df.index % 3 != 0)
    batched = mod._replay_rows(df, EntityStore(), model, cols, tab, want, (df["day"] // 7).to_numpy())
    store, per_row = EntityStore(), []
    for rec, w in zip(df.to_dict("records"), want):
        body = mod.txn_body(rec)
        x = feature_vector(store, body, tab, cols)
        if w:
            per_row.append(float(model.predict_proba(x)[0, 1]))
        store.observe(body)
        store.ingest_label(body["TransactionID"], bool(rec["isFraud"]))
    assert len(batched) == want.sum()
    np.testing.assert_allclose(batched, per_row, rtol=0, atol=1e-7)
