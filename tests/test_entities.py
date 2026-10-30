import numpy as np
import pandas as pd

from fraudgraph.data import load_raw
from fraudgraph.entities import ENTITY_TYPES, entity_keys


def _row(**kw):
    base = dict(card1=1111, card2=100.0, card3=150.0, card4="visa", card5=226.0, card6="debit",
                addr1=300.0, addr2=87.0, P_emaildomain="gmail.com", DeviceInfo="dev0")
    base.update(kw)
    return base


def test_entity_types():
    assert ENTITY_TYPES == ("card", "email", "device", "addr")


def test_key_strings_pinned():
    df = pd.DataFrame([
        _row(),
        _row(card2=np.nan),
        _row(card1=np.nan),
        _row(addr1=np.nan),
        _row(addr1=np.nan, addr2=np.nan, P_emaildomain=None, DeviceInfo=np.nan),
    ], index=[10, 11, 12, 13, 14])
    k = entity_keys(df)
    assert list(k.columns) == list(ENTITY_TYPES)
    assert k.index.tolist() == [10, 11, 12, 13, 14]
    assert k.loc[10, "card"] == "1111|100|150|visa|226|debit"
    assert k.loc[11, "card"] == "1111||150|visa|226|debit"
    assert pd.isna(k.loc[12, "card"])
    assert k.loc[10, "addr"] == "300|87"
    assert k.loc[13, "addr"] == "|87"
    assert pd.isna(k.loc[14, "addr"])
    assert k.loc[10, "email"] == "gmail.com" and pd.isna(k.loc[14, "email"])
    assert k.loc[10, "device"] == "dev0" and pd.isna(k.loc[14, "device"])


def test_int_and_float_card1_render_identically():
    a = entity_keys(pd.DataFrame([_row(card1=1111)]))
    b = entity_keys(pd.DataFrame([_row(card1=1111.0)]))
    assert a.loc[0, "card"] == b.loc[0, "card"]


def test_no_nan_strings_on_fixture(raw_dir):
    k = entity_keys(load_raw(raw_dir))
    for t in ENTITY_TYPES:
        vals = k[t].dropna()
        assert not vals.str.contains("nan", case=False).any()
        assert not vals.str.contains("None").any()
    assert k.isna().any().all()  # the fixture has nulls in every entity type
