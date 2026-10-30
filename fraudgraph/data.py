from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    return df


def profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "fraud_rate": float(df["isFraud"].mean()),
        "identity_coverage": float(df["DeviceType"].notna().mean()),
        "day_min": int(df["day"].min()),
        "day_max": int(df["day"].max()),
    }
