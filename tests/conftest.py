import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    tid = 2987000
    # card A: 25 rows over 40 days; early fraud at day 0 (>=31 days before the day-39 row)
    for i in range(25):
        dt = int(i * 1.6 * 86400) + 100 + i
        rows.append(dict(card1=1111, card2=100.0, card3=150.0, card4="visa", card5=226.0,
                         card6="debit", addr1=300.0, addr2=87.0, dt=dt, fraud=int(i == 0)))
    # card B: 20 rows over 20 days, equal-DT pair at the end
    for i in range(20):
        rows.append(dict(card1=2222, card2=200.0, card3=150.0, card4="mastercard", card5=117.0,
                         card6="credit", addr1=310.0, addr2=87.0,
                         dt=int(i * 86400) + 5000, fraud=int(i in (3, 11))))
    # equal-TransactionDT pair sharing card B
    for _ in range(2):
        rows.append(dict(card1=2222, card2=200.0, card3=150.0, card4="mastercard", card5=117.0,
                         card6="credit", addr1=310.0, addr2=87.0, dt=22 * 86400 + 777, fraud=0))
    # singles from other cards
    for i in range(153):
        rows.append(dict(card1=3000 + i, card2=float(rng.integers(100, 600)), card3=150.0,
                         card4="visa", card5=166.0, card6="debit",
                         addr1=float(rng.integers(100, 500)), addr2=87.0,
                         dt=int(rng.integers(0, 45 * 86400)), fraud=int(rng.random() < 0.05)))
    rng.shuffle(rows)
    emails = ["gmail.com", "yahoo.com", "hotmail.com", None]
    recs = []
    for k, r in enumerate(rows):
        recs.append(dict(
            TransactionID=tid + k, isFraud=r["fraud"], TransactionDT=r["dt"],
            TransactionAmt=round(float(rng.uniform(5, 500)), 2),
            ProductCD=str(rng.choice(list("WCRHS"))),
            card1=r["card1"], card2=r["card2"], card3=r["card3"], card4=r["card4"],
            card5=r["card5"], card6=r["card6"], addr1=r["addr1"], addr2=r["addr2"],
            P_emaildomain=emails[k % 4],
        ))
    tx = pd.DataFrame(recs)
    ids = tx.TransactionID.iloc[::2]
    ident = pd.DataFrame({
        "TransactionID": ids.values,
        "DeviceType": ["mobile", "desktop"] * (len(ids) // 2) + (["mobile"] if len(ids) % 2 else []),
        "DeviceInfo": [f"dev{i}" for i in range(len(ids))],
    })
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    ident.to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path
