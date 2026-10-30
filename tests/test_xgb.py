import json

import numpy as np
import pandas as pd

from fraudgraph.data import load_raw
from fraudgraph.models.xgb import load_model, save_model, train_xgb


def _separable(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    X = pd.DataFrame({"a": rng.normal(size=n) + 4.0 * y, "b": rng.normal(size=n)})
    return X, pd.Series(y.astype(int))


def test_train_xgb_learns_signal():
    from fraudgraph.metrics import pr_auc
    X, y = _separable(2000, 0)
    Xv, yv = _separable(800, 1)
    m = train_xgb(X, y, Xv, yv, n_estimators=100)
    assert pr_auc(yv, m.predict_proba(Xv)[:, 1]) > 0.9


def test_save_load_roundtrip(tmp_path):
    X, y = _separable(500, 0)
    Xv, yv = _separable(200, 1)
    m = train_xgb(X, y, Xv, yv, n_estimators=20)
    p = tmp_path / "m.ubj"
    save_model(m, p)
    m2 = load_model(p)
    assert np.array_equal(m.predict_proba(Xv), m2.predict_proba(Xv))


def test_run_tier1_on_fixture(raw_dir, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import fraudgraph.results as results
    spec = importlib.util.spec_from_file_location(
        "run_tier1", Path(__file__).resolve().parent.parent / "scripts" / "run_tier1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    models_dir = tmp_path / "models"
    out = mod.run(load_raw(raw_dir), models_dir=models_dir, n_estimators=10)
    assert set(out) == {"xgb_base", "xgb_graph"}
    for name in out:
        body = json.loads((tmp_path / "results" / f"{name}.json").read_text())
        for k in ("model", "features", "n_train", "val", "test"):
            assert k in body
        assert "pr_auc" in body["test"]
        assert body["test"]["cost_at_val_threshold"]["threshold"] == body["val"]["best_cost"]["threshold"]
        assert "cost_at_val_threshold" not in body["val"]
        feats = json.loads((models_dir / f"{name}.features.json").read_text())
        assert feats == body["features"]
        assert (models_dir / f"{name}.ubj").exists()
    assert any(f.endswith("_fraud_rate") for f in out["xgb_graph"]["features"])
    assert not any(f.endswith("_fraud_rate") for f in out["xgb_base"]["features"])
