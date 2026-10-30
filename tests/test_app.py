import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys
from fraudgraph.graph_features import feature_columns, graph_features
from fraudgraph.models.xgb import save_model, train_xgb
from fraudgraph.serve.app import create_app
from fraudgraph.serve.store import EntityStore
from fraudgraph.tabular import tabular_features


@pytest.fixture
def setup(raw_dir):
    df = load_raw(raw_dir)
    tab = tabular_features(df)
    X = pd.concat([tab, graph_features(df, entity_keys(df))], axis=1)
    model = train_xgb(X, df["isFraud"], X, df["isFraud"], n_estimators=20)
    return df, X, model


def _client(setup):
    df, X, model = setup
    return TestClient(create_app(model, EntityStore(), list(X.columns)))


def _payload(rec):
    out = {}
    for k, v in rec.items():
        if k in ("isFraud", "day"):
            continue
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        out[k] = v
    return out


BASE = {"TransactionID": 1, "TransactionDT": 100, "TransactionAmt": 12.5}


def test_score_returns_probability(setup):
    r = _client(setup).post("/score", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["TransactionID"] == 1 and body["model"] == "xgb_graph"
    assert body["latency_ms"] >= 0


def test_malformed_payload_422(setup):
    c = _client(setup)
    assert c.post("/score", json={"TransactionID": 1, "TransactionDT": 100}).status_code == 422
    assert c.post("/score", json={**BASE, "TransactionAmt": "abc"}).status_code == 422


def test_metrics_exposes_histogram(setup):
    c = _client(setup)
    c.post("/score", json=BASE)
    text = c.get("/metrics").text
    assert "score_latency_seconds_bucket" in text
    assert "scores_total 1.0" in text
    assert c.get("/health").json() == {"ok": True}


def test_two_apps_do_not_share_metrics(setup):
    a, b = _client(setup), _client(setup)
    a.post("/score", json=BASE)
    assert "scores_total 0.0" in b.get("/metrics").text


def test_score_matches_offline(setup):
    df, X, model = setup
    offline = model.predict_proba(X)[:, 1]
    c = _client(setup)
    online = []
    for rec in df.to_dict("records"):
        r = c.post("/score", json=_payload(rec))
        assert r.status_code == 200, r.text
        online.append(r.json()["score"])
        lab = c.post("/label", json={"TransactionID": int(rec["TransactionID"]),
                                     "isFraud": bool(rec["isFraud"])})
        assert lab.status_code == 200, lab.text
    np.testing.assert_allclose(online, offline, rtol=0, atol=1e-6)
    assert X["card_prior_fraud"].sum() > 0  # labels really fed the graph features


def test_duplicate_transaction_409_and_state_untouched(setup):
    c = _client(setup)
    assert c.post("/score", json=BASE).status_code == 200
    r = c.post("/score", json={**BASE, "TransactionDT": 200})
    assert r.status_code == 409 and "already" in r.json()["detail"]


def test_out_of_order_dt_409(setup):
    c = _client(setup)
    c.post("/score", json=BASE)
    r = c.post("/score", json={**BASE, "TransactionID": 2, "TransactionDT": 99})
    assert r.status_code == 409 and "earlier" in r.json()["detail"]
    assert c.post("/score", json={**BASE, "TransactionID": 3, "TransactionDT": 100}).status_code == 200


def test_label_errors(setup):
    c = _client(setup)
    assert c.post("/label", json={"TransactionID": 9, "isFraud": True}).status_code == 404
    c.post("/score", json=BASE)
    assert c.post("/label", json={"TransactionID": 1, "isFraud": True}).status_code == 200
    assert c.post("/label", json={"TransactionID": 1, "isFraud": False}).status_code == 409
    assert c.post("/label", json={"TransactionID": 1}).status_code == 422


def test_module_import_needs_no_env(monkeypatch, setup, tmp_path):
    import fraudgraph.serve.app as m
    monkeypatch.delenv("FRAUDGRAPH_MODEL", raising=False)
    assert hasattr(m, "create_app")
    df, X, model = setup
    save_model(model, tmp_path / "xgb_graph.ubj")
    (tmp_path / "xgb_graph.features.json").write_text(json.dumps(list(X.columns)))
    monkeypatch.setenv("FRAUDGRAPH_MODEL", str(tmp_path / "xgb_graph.ubj"))
    monkeypatch.setattr(m, "_app", None)
    c = TestClient(m.app)
    assert c.post("/score", json=BASE).status_code == 200
