import numpy as np
import pandas as pd
import pytest

from fraudgraph.config import LABEL_DELAY_DAYS, SEED
from fraudgraph.data import load_raw
from fraudgraph.entities import entity_keys, render_component
from fraudgraph.graph_features import (RATE_PRIOR_FRAUD, RATE_PRIOR_TOTAL, feature_columns,
                                       graph_features)
from fraudgraph.serve.store import EntityStore

D = 86400


def _txn(tid, dt, card1=1111, **kw):
    base = dict(TransactionID=tid, TransactionDT=dt, card1=card1, card2=100.0, card3=150.0,
                card4="visa", card5=226.0, card6="debit", addr1=300.0, addr2=87.0,
                P_emaildomain="gmail.com", DeviceInfo="dev0")
    base.update(kw)
    return base


def _replay(df, label_order="immediate"):
    """Replay df in time order; isFraud reaches the store only through ingest_label."""
    store = EntityStore()
    rows, held, ingested = [], [], []
    rng = np.random.default_rng(SEED)
    for rec in df.to_dict("records"):
        y = rec.pop("isFraud")
        rec.pop("day")
        day = int(rec["TransactionDT"]) // D
        if label_order == "shuffled":
            due = [h for h in held if h[2] <= day - LABEL_DELAY_DAYS]
            held = [h for h in held if h[2] > day - LABEL_DELAY_DAYS]
            for k in rng.permutation(len(due)):
                store.ingest_label(*due[k][:2])
                ingested.append(due[k][0])
        rows.append(store.features(rec))
        store.observe(rec)
        if label_order == "shuffled":
            held.append((rec["TransactionID"], bool(y), day))
        else:
            store.ingest_label(rec["TransactionID"], bool(y))
            ingested.append(rec["TransactionID"])
    return pd.DataFrame(rows, index=df.index), ingested


@pytest.fixture
def offline(raw_dir):
    df = load_raw(raw_dir)
    off = graph_features(df, entity_keys(df))
    # non-vacuous: labels and repeat activity actually move the offline features
    assert off["card_prior_fraud"].sum() > 0
    assert (off["card_fraud_rate"] != RATE_PRIOR_FRAUD / RATE_PRIOR_TOTAL).any()
    assert off["device_prior_n"].sum() > 0
    return df, off


def _assert_parity(on, off):
    assert list(on.columns) == feature_columns()
    assert on.index.equals(off.index)
    np.testing.assert_allclose(on.to_numpy(dtype=float), off.to_numpy(dtype=float),
                               rtol=0, atol=1e-9)


def test_offline_online_parity(offline):
    df, off = offline
    on, _ = _replay(df)
    _assert_parity(on, off)


def test_offline_online_parity_shuffled_label_arrival(offline):
    df, off = offline
    on, ingested = _replay(df, "shuffled")
    seen_order = [i for i in df["TransactionID"] if i in set(ingested)]
    assert ingested != seen_order  # labels really arrived out of row order
    _assert_parity(on, off)


def test_unseen_entity_scores():
    f = EntityStore().features(_txn(1, 10 * D, card1=999999))
    assert list(f) == feature_columns()
    for t in ("card", "email", "device", "addr"):
        assert f[f"{t}_prior_n"] == 0 and f[f"{t}_prior_fraud"] == 0
        assert f[f"{t}_fraud_rate"] == pytest.approx(RATE_PRIOR_FRAUD / RATE_PRIOR_TOTAL)


def test_missing_and_null_fields_score_as_null_entities():
    s = EntityStore()
    f = s.features({"TransactionID": 1, "TransactionDT": 0, "card1": None})
    assert all(f[f"{t}_prior_n"] == 0 for t in ("card", "email", "device", "addr"))
    s.observe({"TransactionID": 1, "TransactionDT": 0, "card1": None})
    f = s.features({"TransactionID": 2, "TransactionDT": 5, "card1": None, "addr1": np.nan})
    assert f["card_prior_n"] == 0 and f["addr_prior_n"] == 0  # null keys never accumulate


def test_same_timestamp_txn_not_counted():
    s = EntityStore()
    s.observe(_txn(1, 100))
    assert s.features(_txn(2, 100))["card_prior_n"] == 0
    s.observe(_txn(2, 100))
    assert s.features(_txn(3, 101))["card_prior_n"] == 2


def test_label_delay_boundary():
    s = EntityStore()
    s.features(_txn(1, 5 * D))
    s.observe(_txn(1, 5 * D))
    s.ingest_label(1, True)
    before = s.features(_txn(2, (5 + LABEL_DELAY_DAYS - 1) * D + D - 1))
    assert before["card_prior_fraud"] == 0 and before["card_prior_n"] == 1
    after = s.features(_txn(3, (5 + LABEL_DELAY_DAYS) * D))
    assert after["card_prior_fraud"] == 1
    assert after["card_fraud_rate"] == pytest.approx((1 + RATE_PRIOR_FRAUD) / (1 + RATE_PRIOR_TOTAL))


def test_unknown_txn_label_raises():
    with pytest.raises(KeyError, match="never observed"):
        EntityStore().ingest_label(42, True)


def test_duplicate_label_raises():
    s = EntityStore()
    s.observe(_txn(1, 0))
    s.ingest_label(1, False)
    with pytest.raises(ValueError, match="already labeled"):
        s.ingest_label(1, True)


def test_scoring_backwards_in_time_raises():
    s = EntityStore()
    s.features(_txn(1, 100))
    s.features(_txn(2, 100))  # equal DT is fine
    with pytest.raises(ValueError, match="earlier"):
        s.features(_txn(3, 99))


def test_render_component_numpy_floats():
    assert render_component(np.float32(100.0)) == "100"
    assert render_component(np.float64(100.0)) == "100"
    assert render_component(np.float32("nan")) == ""
    assert render_component(np.float32(1.5)) == "1.5"


def test_duplicate_observe_raises():
    s = EntityStore()
    s.observe(_txn(1, 0))
    with pytest.raises(ValueError, match="already observed"):
        s.observe(_txn(1, 5))


def test_features_cost_does_not_grow_with_pending_same_dt():
    import time
    from statistics import median

    def cost(n):
        s = EntityStore()
        vals = {"card": "c", "email": "e", "device": "d", "addr": "a"}
        s.features(_txn(0, 1000))
        s._pending.extend((1000, vals) for _ in range(n))  # n same-DT observed txns
        probe = _txn(10**9, 1000)
        reps = []
        for _ in range(7):
            t0 = time.perf_counter()
            s.features(probe)
            reps.append(time.perf_counter() - t0)
        return median(reps)

    assert cost(200_000) / cost(1_000) < 3
