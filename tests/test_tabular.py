import numpy as np

from fraudgraph.data import load_raw
from fraudgraph.tabular import CATEGORICAL, tabular_features


def test_columns_stable(raw_dir):
    df = load_raw(raw_dir)
    a = tabular_features(df.iloc[:100])
    b = tabular_features(df.iloc[100:])
    assert list(a.columns) == list(b.columns)
    assert list(a.columns[:3]) == ["TransactionAmt", "log_amt", "hour"]
    assert not {"TransactionID", "isFraud", "TransactionDT", "day", "DeviceInfo"} & set(a.columns)
    assert set(CATEGORICAL) <= set(a.columns)


def test_values(raw_dir):
    df = load_raw(raw_dir)
    f = tabular_features(df)
    assert np.allclose(f.log_amt, np.log1p(df.TransactionAmt))
    assert (f.hour == (df.TransactionDT // 3600) % 24).all()
    assert (f.loc[df.P_emaildomain.isna(), "P_emaildomain"] == -1).all()


def test_categorical_encoding_slice_independent(raw_dir):
    df = load_raw(raw_dir)
    full = tabular_features(df)
    row = df[df.card4.notna()].iloc[[5]]
    alone = tabular_features(row)
    for c in CATEGORICAL:
        assert alone[c].iloc[0] == full.loc[row.index[0], c]


def test_columns_reindex(raw_dir):
    df = load_raw(raw_dir)
    cols = list(tabular_features(df).columns)
    row = df.iloc[[0]].drop(columns=["card2", "addr1"])
    out = tabular_features(row, columns=cols)
    assert list(out.columns) == cols
    assert np.isnan(out["card2"].iloc[0]) and np.isnan(out["addr1"].iloc[0])
    extra = tabular_features(df.iloc[[0]], columns=["log_amt", "nonexistent"])
    assert list(extra.columns) == ["log_amt", "nonexistent"] and np.isnan(extra["nonexistent"].iloc[0])


def test_single_row_with_columns_equals_full_frame(raw_dir):
    from pandas.testing import assert_frame_equal

    df = load_raw(raw_dir)
    full = tabular_features(df)
    cols = list(full.columns)
    nan_rows = df.index[df.card2.isna() | df.addr1.isna()][:2].tolist()
    assert nan_rows, "fixture must contain NaN numerics"
    for idx in [df.index[0], df.index[50], *nan_rows]:
        one = df.loc[[idx]]
        # an all-None numeric value arrives as object dtype in a 1-row slice
        one = one.astype({c: object for c in ("card2", "addr1")})
        assert_frame_equal(tabular_features(one, columns=cols), full.loc[[idx]])
