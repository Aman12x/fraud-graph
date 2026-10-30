import pandas as pd

from fraudgraph.config import LABEL_DELAY_DAYS
from fraudgraph.data import load_raw
from fraudgraph.splits import Splits, label_known, make_splits, split_masks


def _masks(raw_dir):
    df = load_raw(raw_dir)
    s = make_splits(df)
    return df, s, split_masks(df, s)


def test_splits_disjoint_and_cover(raw_dir):
    df, s, (tr, va, te) = _masks(raw_dir)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert tr.any() and va.any() and te.any()
    assert s.train_end_day < s.val_end_day


def test_no_day_straddles(raw_dir):
    df, s, masks = _masks(raw_dir)
    for m in masks:
        days_in = set(df.loc[m, "day"])
        for other in masks:
            if other is not m:
                assert not (days_in & set(df.loc[other, "day"]))


def test_label_known_respects_delay():
    df = pd.DataFrame({"day": [10, 11, 70, 71]})
    known = label_known(df, as_of_day=40, delay=30)
    assert known.tolist() == [True, False, False, False]
    df = pd.DataFrame({"day": [100 - LABEL_DELAY_DAYS, 100 - LABEL_DELAY_DAYS + 1]})
    assert label_known(df, 100).tolist() == [True, False]


def test_training_targets_exclude_unlabeled_tail():
    # fixture spans only ~45 days (< delay after the train split), so use a wider synthetic range
    df = pd.DataFrame({"day": list(range(180)) * 3})
    s = make_splits(df)
    tr, _, _ = split_masks(df, s)
    usable = tr & label_known(df, s.train_end_day)
    cutoff = s.train_end_day - LABEL_DELAY_DAYS
    assert df.loc[usable, "day"].max() == cutoff
    tail = tr & (df["day"] > cutoff)
    assert df.loc[tail, "day"].nunique() == LABEL_DELAY_DAYS
    assert not (usable & tail).any()
