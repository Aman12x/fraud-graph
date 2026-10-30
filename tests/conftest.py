import torch  # noqa: F401  torch's OpenMP runtime must load before xgboost's, else torch training deadlocks
import numpy as np
import pandas as pd
import pytest

SPAN_DAYS = 180


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    tid = 2987000
    # card A: 25 rows over ~173 days; frauds at day 0 (>=31 days before later rows) and mid-span
    for i in range(25):
        dt = int(i * 7.2 * 86400) + 100 + i
        rows.append(dict(card1=1111, card2=100.0, card3=150.0, card4="visa", card5=226.0,
                         card6="debit", addr1=300.0, addr2=87.0, dt=dt, fraud=int(i in (0, 12))))
    # card B: 20 rows over ~171 days (card2 null: exercises a null component in the card key)
    for i in range(20):
        rows.append(dict(card1=2222, card2=None, card3=150.0, card4="mastercard", card5=117.0,
                         card6="credit", addr1=310.0, addr2=87.0,
                         dt=int(i * 9 * 86400) + 5000, fraud=int(i in (3, 11))))
    # equal-TransactionDT pair sharing card B
    for _ in range(2):
        rows.append(dict(card1=2222, card2=None, card3=150.0, card4="mastercard", card5=117.0,
                         card6="credit", addr1=310.0, addr2=87.0, dt=100 * 86400 + 777, fraud=0))
    # generic rows: a pool of 40 cards with fixed attributes, spread over the whole span
    pool = [dict(card1=3000 + j, card2=float(rng.integers(100, 600)), card3=150.0,
                 card4=str(rng.choice(["visa", "mastercard"])), card5=166.0, card6="debit")
            for j in range(40)]
    for _ in range(173):
        c = pool[int(rng.integers(0, len(pool)))]
        rows.append(dict(**c, addr1=float(rng.integers(100, 130)), addr2=87.0,
                         dt=int(rng.integers(0, SPAN_DAYS * 86400)),
                         fraud=int(rng.random() < 0.08)))
    # a few fraud rows spread across the span, plus rows with no card1 / no address
    for d in (10, 50, 90, 130, 170):
        c = pool[d % len(pool)]
        rows.append(dict(**c, addr1=120.0, addr2=87.0, dt=d * 86400 + 4321, fraud=1))
    for d in (20, 140):
        rows.append(dict(card1=None, card2=None, card3=None, card4=None, card5=None, card6=None,
                         addr1=None, addr2=None, dt=d * 86400 + 99, fraud=0))
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
        # a small pool so device links repeat; every 7th identity row has no DeviceInfo
        "DeviceInfo": [None if i % 7 == 6 else f"dev{i % 12}" for i in range(len(ids))],
    })
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    ident.to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path
