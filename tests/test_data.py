import pandas as pd

from fraudgraph.data import load_raw, profile


def test_left_join_keeps_identityless_rows(raw_dir):
    n_tx = len(pd.read_csv(raw_dir / "train_transaction.csv"))
    df = load_raw(raw_dir)
    assert len(df) == n_tx
    assert df.DeviceInfo.isna().sum() > 0
    assert df.DeviceInfo.notna().sum() > 0


def test_sorted_by_time(raw_dir):
    df = load_raw(raw_dir)
    assert df.TransactionDT.is_monotonic_increasing


def test_day_column(raw_dir):
    df = load_raw(raw_dir)
    assert (df.day == df.TransactionDT // 86400).all()


def test_identity_coverage_is_join_coverage(raw_dir):
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    assert ident.DeviceType.isna().any(), "fixture needs an identity row with null DeviceType"
    df = load_raw(raw_dir)
    want = tx.TransactionID.isin(ident.TransactionID).mean()
    assert 0.45 < want < 0.55  # identity for half the transactions
    assert profile(df)["identity_coverage"] == want
    assert profile(df)["identity_coverage"] > df.DeviceType.notna().mean()
    assert list(df.columns) == [*tx.columns, *[c for c in ident.columns if c != "TransactionID"], "day"]


def test_profile_refuses_frame_without_join_info(raw_dir):
    import pytest
    df = load_raw(raw_dir)
    df.attrs.clear()
    with pytest.raises(ValueError, match="load_raw"):
        profile(df)


def test_profile_refuses_a_slice_of_the_loaded_frame(raw_dir):
    # pandas carries attrs onto slices, so the full-frame coverage must not be reported for a subset
    import pytest
    with pytest.raises(ValueError, match="load_raw"):
        profile(load_raw(raw_dir).iloc[:50])


def test_profile_keys(raw_dir):
    p = profile(load_raw(raw_dir))
    assert set(p) == {"rows", "fraud_rate", "identity_coverage", "day_min", "day_max"}
    assert 0 <= p["identity_coverage"] <= 1


def test_fixture_shape_for_later_tasks(raw_dir):
    df = load_raw(raw_dir)
    assert df.P_emaildomain.isna().any()
    assert df.groupby("card1").size().max() >= 21
    assert df.duplicated(["card1", "TransactionDT"], keep=False).any()
    a = df[df.card1 == 1111]
    fraud_day = a[a.isFraud == 1].day.min()
    assert (a.day >= fraud_day + 31).any()
    assert df.day.max() - df.day.min() >= 170
