import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import fraudgraph.results as results
from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys
from fraudgraph.gnn.graph import build_hetero, make_loader
from fraudgraph.gnn.model import HeteroSAGE
from fraudgraph.tabular import tabular_features


def _setup(raw_dir):
    df = load_raw(raw_dir)
    data = build_hetero(df, entity_keys(df), tabular_features(df).to_numpy(dtype=float),
                        fit_rows=np.ones(len(df), dtype=bool))
    return df, data


def test_output_shape(raw_dir):
    df, data = _setup(raw_dir)
    loader = make_loader(data, np.arange(len(df)), batch_size=32)
    model = HeteroSAGE(data.metadata())
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == (batch["txn"].batch_size,)


def test_overfits_tiny_graph(raw_dir):
    torch.manual_seed(42)
    df, data = _setup(raw_dir)
    loader = make_loader(data, np.arange(len(df)), batch_size=len(df))
    batch = next(iter(loader))
    model = HeteroSAGE(data.metadata(), dropout=0.0)
    y = batch["txn"].y[: batch["txn"].batch_size].float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        model(batch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(batch), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_run_gnn_on_fixture(raw_dir, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_gnn", Path(__file__).resolve().parent.parent / "scripts" / "run_gnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    out = mod.run(load_raw(raw_dir), epochs=2)
    body = json.loads((tmp_path / "results" / "gnn_sage.json").read_text())
    for k in ("model", "features", "n_train", "val", "test"):
        assert k in body
    assert body["model"] == "gnn_sage" and "pr_auc" in body["test"]
    assert out["n_train"] == body["n_train"] > 0
