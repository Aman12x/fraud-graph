from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    # share of transactions with an identity record (a membership test, no helper column)
    df.attrs["identity_coverage"] = float(tx["TransactionID"].isin(ident["TransactionID"]).mean())
    df.attrs["identity_rows"] = len(df)  # attrs survive slicing; profile checks it got this frame
    return df


def profile(df: pd.DataFrame) -> dict:
    """Summary of a load_raw frame. identity_coverage = share of transactions whose
    TransactionID appears in the identity table (not any one identity field's fill rate)."""
    if "identity_coverage" not in df.attrs or df.attrs.get("identity_rows") != len(df):
        raise ValueError("profile needs the whole frame returned by load_raw "
                         "(identity join info in df.attrs; a slice would report the full-frame value)")
    return {
        "rows": int(len(df)),
        "fraud_rate": float(df["isFraud"].mean()),
        "identity_coverage": df.attrs["identity_coverage"],
        "day_min": int(df["day"].min()),
        "day_max": int(df["day"].max()),
    }
