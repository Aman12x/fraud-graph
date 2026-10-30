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
