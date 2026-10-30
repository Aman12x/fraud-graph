import numpy as np
import pandas as pd
import pytest

from fraudgraph.config import LABEL_DELAY_DAYS
from fraudgraph.data import load_raw
from fraudgraph.entities import ENTITY_TYPES, entity_keys
from fraudgraph.graph_features import RATE_PRIOR_FRAUD, RATE_PRIOR_TOTAL, graph_features

D = 86400


def _frame(rows):
    """rows: (card1, email, TransactionDT, isFraud)."""
    df = pd.DataFrame(rows, columns=["card1", "P_emaildomain", "TransactionDT", "isFraud"])
    for c in ("card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "DeviceInfo"):
        df[c] = np.nan
    df["day"] = df["TransactionDT"] // D
    return df


def _gf(df, delay=LABEL_DELAY_DAYS):
    return graph_features(df, entity_keys(df), delay=delay)


def test_columns_and_dtypes():
    df = _frame([(1, "a", 0, 0), (1, "a", 10, 1)])
    f = _gf(df)
    expected = [f"{t}_{s}" for t in ENTITY_TYPES for s in ("prior_n", "prior_fraud", "fraud_rate")]
    assert list(f.columns) == expected
    for t in ENTITY_TYPES:
        assert f[f"{t}_prior_n"].dtype == np.int64
        assert f[f"{t}_prior_fraud"].dtype == np.int64
        assert f[f"{t}_fraud_rate"].dtype == np.float64
    assert (RATE_PRIOR_FRAUD, RATE_PRIOR_TOTAL) == (1, 20)


def test_null_entity_values_create_no_link():
    df = _frame([(1, None, 0, 1), (2, None, 100 * D, 0)])
    f = _gf(df)
    assert f["email_prior_n"].tolist() == [0, 0]
    assert f["email_prior_fraud"].tolist() == [0, 0]
    assert f["email_fraud_rate"].tolist() == [1 / 20, 1 / 20]
    # device is null everywhere in this frame
    assert (f["device_prior_n"] == 0).all() and (f["device_fraud_rate"] == 1 / 20).all()


def test_same_timestamp_not_counted():
    df = _frame([(7, "a", 5 * D + 3, 0), (7, "a", 5 * D + 3, 0)])
    f = _gf(df)
    assert f["card_prior_n"].tolist() == [0, 0]
    # a third, later row sees both
    df = _frame([(7, "a", 5 * D + 3, 0), (7, "a", 5 * D + 3, 0), (7, "a", 5 * D + 4, 0)])
    assert _gf(df)["card_prior_n"].tolist() == [0, 0, 2]


def test_future_label_never_read():
    d, delay = 3, LABEL_DELAY_DAYS
    df = _frame([
        (9, "a", d * D + 10, 1),
        (9, "a", (d + delay - 1) * D + 86399, 0),  # last second of day d + delay - 1
        (9, "a", (d + delay) * D, 0),              # first second of day d + delay
        (9, "a", (d + delay + 5) * D, 0),
    ])
    f = _gf(df)
    assert f["card_prior_fraud"].tolist() == [0, 0, 1, 1]
    assert f["card_prior_n"].tolist() == [0, 1, 2, 3]
    # rows 2 and 3 see only the fraud row as label-known (row 1's day d+delay-1 > d+5)
    assert f["card_fraud_rate"].tolist() == pytest.approx(
        [1 / 20, 1 / 20, 2 / 21, 2 / 21])


def test_leakage_shuffle_invariance(raw_dir):
    df = load_raw(raw_dir)
    keys = entity_keys(df)
    base = graph_features(df, keys)
    rng = np.random.default_rng(7)
    checked = 0
    for as_of in (60, 108, 150):
        window = df["day"] > as_of - LABEL_DELAY_DAYS
        perm = df.copy()
        perm.loc[window, "isFraud"] = rng.permutation(df.loc[window, "isFraud"].to_numpy())
        assert (perm["isFraud"] != df["isFraud"]).any(), "permutation must change some labels"
        flipped = df.copy()
        flipped.loc[window, "isFraud"] = 1 - df.loc[window, "isFraud"]
        at = df["day"] <= as_of
        for mutated in (perm, flipped):
            got = graph_features(mutated, keys)
            pd.testing.assert_frame_equal(got.loc[at], base.loc[at])
        # the flip is visible to later rows, so the features do read labels at all
        assert not graph_features(flipped, keys).loc[~at].equals(base.loc[~at])
        # rows at as_of have label-known history, so invariance is not trivially 0 == 0
        assert (base.loc[df["day"] == as_of, "card_fraud_rate"] != 1 / 20).any() or \
            (base.loc[df["day"] == as_of, "email_fraud_rate"] != 1 / 20).any()
        checked += int((df["day"] == as_of).sum())
    assert checked > 0


def _bruteforce(df, keys, delay):
    """Nested-loop reference, straight from the definitions."""
    out = {}
    dt = df["TransactionDT"].to_numpy()
    day = df["day"].to_numpy()
    y = df["isFraud"].to_numpy()
    for t in ENTITY_TYPES:
        v = keys[t].to_numpy()
        n_col, f_col, r_col = [], [], []
        for i in range(len(df)):
            if pd.isna(v[i]):
                n_col.append(0); f_col.append(0); r_col.append(1 / 20)
                continue
            n = fr = lab = 0
            for j in range(len(df)):
                if pd.isna(v[j]) or v[j] != v[i]:
                    continue
                if dt[j] < dt[i]:
                    n += 1
                if day[j] <= day[i] - delay:
                    lab += 1
                    if y[j] == 1:
                        fr += 1
            n_col.append(n); f_col.append(fr); r_col.append((fr + 1) / (lab + 20))
        out[f"{t}_prior_n"] = n_col
        out[f"{t}_prior_fraud"] = f_col
        out[f"{t}_fraud_rate"] = r_col
    return pd.DataFrame(out, index=df.index)


@pytest.mark.parametrize("delay", [LABEL_DELAY_DAYS, 7])
@pytest.mark.parametrize("shuffled", [False, True])
def test_matches_bruteforce(raw_dir, delay, shuffled):
    df = load_raw(raw_dir)
    if shuffled:
        df = df.sample(frac=1, random_state=3)
        df.index = df.index * 10 + 5  # non-RangeIndex, unsorted
    keys = entity_keys(df)
    got = graph_features(df, keys, delay=delay)
    ref = _bruteforce(df, keys, delay)
    assert got.index.equals(df.index)
    for c in ref.columns:
        if c.endswith("fraud_rate"):
            np.testing.assert_allclose(got[c].to_numpy(), ref[c].to_numpy(), rtol=0, atol=1e-12)
        else:
            assert got[c].tolist() == ref[c].tolist(), c
    # the fixture must actually exercise links and known labels
    for t in ENTITY_TYPES:
        assert (got[f"{t}_prior_n"] > 0).any(), t
    for t in ("card", "email", "addr", "device"):
        assert (got[f"{t}_prior_fraud"] > 0).any(), t
