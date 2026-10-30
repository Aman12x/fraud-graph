"""Entity keys linking transactions: card, email domain, device, billing address.

Key rendering is a contract shared with the online scorer (Task 10): use `render_component`
for every component so offline and online keys are byte-identical.
"""
import math

import numpy as np
import pandas as pd

ENTITY_TYPES = ("card", "email", "device", "addr")
CARD_COLS = ("card1", "card2", "card3", "card4", "card5", "card6")
ADDR_COLS = ("addr1", "addr2")
SEP = "|"


def render_component(v) -> str:
    """One key component as text: null -> "", integral float -> int ("100.0" -> "100").

    numpy floats of any width count as floats, so np.float32(100.0) -> "100" and NaN -> "".
    """
    if v is None or v is pd.NA:
        return ""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        if float(v).is_integer():
            return str(int(v))
    return str(v)


def _joined(df: pd.DataFrame, cols, present: pd.Series) -> pd.Series:
    parts = [df[c].map(render_component) if c in df else pd.Series("", index=df.index)
             for c in cols]
    key = parts[0].str.cat(parts[1:], sep=SEP)
    return key.where(present, None).astype(object)


def _passthrough(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(None, index=df.index, dtype=object)
    s = df[col]
    return s.map(render_component).where(s.notna(), None).astype(object)


def entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """One column per ENTITY_TYPES entry, indexed like `df`; null keys are None, never "nan"."""
    card_present = df["card1"].notna()
    addr_present = df[list(ADDR_COLS)].notna().any(axis=1)
    return pd.DataFrame({
        "card": _joined(df, CARD_COLS, card_present),
        "email": _passthrough(df, "P_emaildomain"),
        "device": _passthrough(df, "DeviceInfo"),
        "addr": _joined(df, ADDR_COLS, addr_present),
    }, index=df.index)
