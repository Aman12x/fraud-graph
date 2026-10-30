import numpy as np
import pandas as pd

from fraudgraph.entities import entity_keys
from fraudgraph.seq.windows import card_windows


def _frame(rows, index=None):
    """rows: (card1, TransactionDT)."""
    df = pd.DataFrame(rows, columns=["card1", "TransactionDT"])
    for c in ("card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
              "P_emaildomain", "DeviceInfo"):
        df[c] = np.nan
    if index is not None:
        df.index = index
    return df


def _run(rows, k=16, index=None):
    df = _frame(rows, index)
    X = np.arange(len(df), dtype=float).reshape(-1, 1) + 1.0  # feature = position + 1
    return card_windows(df, entity_keys(df), X, k=k)


def test_window_strictly_prior():
    # rows 1 and 2 share a timestamp: neither sees the other
    hist, mask = _run([(1, 10), (1, 20), (1, 20), (1, 30)])
    assert hist.shape == (4, 16, 2) and mask.shape == (4, 16)
    for i in (1, 2):
        assert (~mask[i]).sum() == 1
        assert hist[i, -1, 0] == 1.0  # only row 0
    assert sorted(hist[3, ~mask[3], 0]) == [1.0, 2.0, 3.0]
    assert mask[0].all()


def test_null_card_all_padded():
    hist, mask = _run([(None, 10), (None, 20), (1, 30)])
    assert mask[0].all() and mask[1].all()
    assert not hist[:2].any()
    assert mask[2].all()  # no earlier row of card 1


def test_order_most_recent_last():
    # input deliberately out of time order
    hist, mask = _run([(1, 30), (1, 10), (1, 20), (1, 40)])
    row = hist[3]
    assert list(row[~mask[3], 0]) == [2.0, 3.0, 1.0]  # t=10, 20, 30
    assert mask[3][:13].all() and not mask[3][13:].any()
    # extra channel: log1p(seconds since), shrinking toward the most recent
    np.testing.assert_allclose(row[13:, 1], np.log1p([30, 20, 10]))
    assert not hist[3, :13].any()


def test_k_truncation():
    rows = [(1, 100 + i) for i in range(21)]
    hist, mask = _run(rows, k=16)
    assert not mask[20].any()
    assert list(hist[20, :, 0]) == [float(j + 1) for j in range(4, 20)]  # latest 16 of 20 priors
    assert (~mask[5]).sum() == 5


def test_non_range_index_and_dtype():
    hist, mask = _run([(1, 10), (1, 20)], index=[50, 7])
    assert hist.dtype == np.float32
    assert (~mask[1]).sum() == 1 and mask[0].all()
