"""Tier 1: XGBoost on tabular features vs tabular + point-in-time graph features."""
import argparse
import json
from pathlib import Path

import pandas as pd

import fraudgraph.results as results
from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys
from fraudgraph.experiment import prepare
from fraudgraph.graph_features import graph_features
from fraudgraph.metrics import evaluate
from fraudgraph.models.xgb import save_model, train_xgb

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def run(df: pd.DataFrame, models_dir: Path | None = None, n_estimators: int = 2000) -> dict[str, dict]:
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare(df)
    val_m, test_m, train_fit, tab = prep.val, prep.test, prep.train_fit, prep.tab
    graph = graph_features(df, entity_keys(df))  # point-in-time by construction
    full = pd.concat([tab, graph], axis=1)
    frames = {"xgb_base": tab, "xgb_graph": full}

    out = {}
    for name, X in frames.items():
        cols = list(X.columns)
        m = train_xgb(X[train_fit], df.loc[train_fit, "isFraud"],
                      X[val_m], df.loc[val_m, "isFraud"], n_estimators=n_estimators)
        ev = {}
        for part, mask in (("val", val_m), ("test", test_m)):
            p = m.predict_proba(X[mask])[:, 1]
            ev[part] = evaluate(df.loc[mask, "isFraud"], p, df.loc[mask, "TransactionAmt"])
        payload = {"model": name, "features": cols, "n_train": int(train_fit.sum()), **ev}
        results.write_result(name, payload)
        save_model(m, models_dir / f"{name}.ubj")
        (models_dir / f"{name}.features.json").write_text(json.dumps(cols))
        out[name] = payload
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    a = ap.parse_args()
    for name, p in run(load_raw(a.raw_dir)).items():
        print(name, "test pr_auc", round(p["test"]["pr_auc"], 4))
