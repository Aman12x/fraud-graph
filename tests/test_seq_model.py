import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import fraudgraph.results as results
from fraudgraph.data import load_raw
from fraudgraph.seq.model import SeqTransformer

F, K = 5, 8


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(n, K, F + 1, generator=g)
    current = torch.randn(n, F, generator=g)
    mask = torch.zeros(n, K, dtype=torch.bool)
    mask[:, :3] = True  # first three positions padded
    return hist, mask, current


def test_padding_invariance():
    torch.manual_seed(0)
    m = SeqTransformer(F).eval()
    hist, mask, cur = _batch()
    base = m(hist, mask, cur)
    h2 = hist.clone()
    h2[mask] = torch.randn_like(h2[mask]) * 100
    assert torch.allclose(base, m(h2, mask, cur), atol=1e-5)


def test_all_padded_row_finite():
    torch.manual_seed(0)
    m = SeqTransformer(F).eval()
    hist, mask, cur = _batch()
    mask[0] = True
    out = m(hist, mask, cur)
    assert out.shape == (6,) and torch.isfinite(out).all()


def test_eval_is_deterministic():
    torch.manual_seed(0)
    m = SeqTransformer(F, dropout=0.5).eval()
    hist, mask, cur = _batch()
    assert torch.equal(m(hist, mask, cur), m(hist, mask, cur))


def test_overfits_tiny():
    torch.manual_seed(0)
    m = SeqTransformer(F, dropout=0.0)
    hist, mask, cur = _batch(32, seed=1)
    y = (cur[:, 0] > 0).float()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn(m(hist, mask, cur), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_run_seq_on_fixture(raw_dir, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_seq", Path(__file__).resolve().parent.parent / "scripts" / "run_seq.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    out = mod.run(load_raw(raw_dir), epochs=2)
    body = json.loads((tmp_path / "results" / "seq_transformer.json").read_text())
    for k in ("model", "features", "n_train", "val", "test"):
        assert k in body
    assert body["model"] == "seq_transformer" and "pr_auc" in body["test"]
    assert out["n_train"] == body["n_train"] > 0
