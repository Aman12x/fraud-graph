"""FastAPI scoring service: xgb_graph model + live EntityStore graph features.

POST /score   score a txn from state strictly before it, then observe it
POST /label   deliver a (delayed) label for a scored txn
GET  /metrics Prometheus exposition   GET /health  liveness

The feature vector is built exactly as offline: tabular_features(1-row frame, columns=<the
tabular part of features.json>) + store.features(txn), reindexed to the full features.json
order. The module-level `app` is built lazily from env FRAUDGRAPH_MODEL.
"""
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from fraudgraph.graph_features import feature_columns
from fraudgraph.models.xgb import load_model
from fraudgraph.serve.store import EntityStore
from fraudgraph.tabular import tabular_features

MODEL_NAME = "xgb_graph"


class Txn(BaseModel):
    """Raw transaction. Extra numeric columns pass through to tabular_features."""
    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str | None = None
    card4: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    card1: int | None = None
    card2: int | None = None
    card3: int | None = None
    card5: int | None = None
    addr1: float | None = None
    addr2: float | None = None


class Label(BaseModel):
    TransactionID: int
    isFraud: bool


def create_app(model, store: EntityStore, feature_columns_: list[str]) -> FastAPI:
    graph_cols = set(feature_columns())
    tab_cols = [c for c in feature_columns_ if c not in graph_cols]
    lock = threading.Lock()  # store state and the DT ordering are not safe to interleave
    registry = CollectorRegistry()
    latency = Histogram("score_latency_seconds", "Time to build features and score one txn",
                        registry=registry)
    scores = Counter("scores_total", "Transactions scored", registry=registry)
    app = FastAPI(title="fraud-graph scoring")

    def vector(txn: dict) -> pd.DataFrame:
        tab = tabular_features(pd.DataFrame([txn]), columns=tab_cols)
        graph = pd.DataFrame([store.features(txn)])
        return pd.concat([tab, graph], axis=1).reindex(columns=feature_columns_)

    @app.post("/score")
    def score(t: Txn):
        start = time.perf_counter()
        txn = t.model_dump()
        with lock:
            if store.has_observed(t.TransactionID):
                raise HTTPException(409, f"TransactionID {t.TransactionID} was already scored")
            if store.last_dt is not None and t.TransactionDT < store.last_dt:
                raise HTTPException(
                    409, f"TransactionDT {t.TransactionDT} is earlier than the last scored "
                         f"TransactionDT {store.last_dt}; scoring must be in non-decreasing time order")
            p = float(model.predict_proba(vector(txn))[0, 1])
            store.observe(txn)
        elapsed = time.perf_counter() - start
        latency.observe(elapsed)
        scores.inc()
        return {"TransactionID": t.TransactionID, "score": p, "model": MODEL_NAME,
                "latency_ms": elapsed * 1000.0}

    @app.post("/label")
    def label(body: Label):
        with lock:
            try:
                store.ingest_label(body.TransactionID, body.isFraud)
            except KeyError as e:
                raise HTTPException(404, str(e.args[0])) from None
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
        return {"TransactionID": body.TransactionID, "ok": True}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def build_app_from_env() -> FastAPI:
    path = Path(os.environ["FRAUDGRAPH_MODEL"])
    cols = json.loads(path.with_name(path.stem + ".features.json").read_text())
    return create_app(load_model(path), EntityStore(), cols)


_app: FastAPI | None = None


def __getattr__(name: str):
    """Module-level `app`, built on first access so importing needs no env var."""
    global _app
    if name == "app":
        if _app is None:
            _app = build_app_from_env()
        return _app
    raise AttributeError(name)
