import numpy as np
import torch

from fraudgraph.data import load_raw
from fraudgraph.entities import ENTITY_TYPES, entity_keys
from fraudgraph.gnn.graph import build_hetero, make_loader, standardize
from fraudgraph.tabular import tabular_features


def _build(raw_dir, flip_labels=False):
    df = load_raw(raw_dir)
    if flip_labels:
        df = df.assign(isFraud=1 - df["isFraud"])
    tab = tabular_features(df)
    return df, tab, build_hetero(df, entity_keys(df), tab.to_numpy(dtype=float))


def test_no_label_in_features(raw_dir):
    df, tab, data = _build(raw_dir)
    assert "isFraud" not in tab.columns
    assert data["txn"].x.shape == (len(df), tab.shape[1])
    assert torch.equal(data["txn"].y, torch.as_tensor(df["isFraud"].to_numpy(), dtype=torch.long))
    # x must not depend on the label: flipping every label leaves x bit-identical
    _, _, flipped = _build(raw_dir, flip_labels=True)
    assert torch.equal(data["txn"].x, flipped["txn"].x)
    assert not torch.equal(data["txn"].y, flipped["txn"].y)
    for t in ENTITY_TYPES:
        assert torch.equal(data[t].x, torch.ones(data[t].num_nodes, 1))
    assert not torch.isnan(data["txn"].x).any()


def test_null_values_no_entity_node(raw_dir):
    df, _, data = _build(raw_dir)
    keys = entity_keys(df)
    for t in ENTITY_TYPES:
        col = keys[t]
        assert col.isna().any(), f"fixture should have a null {t} key"
        assert data[t].num_nodes == col.dropna().nunique()
        src, dst = data["txn", f"has_{t}", t].edge_index
        # exactly one edge per non-null row, none from null rows
        assert sorted(src.tolist()) == np.flatnonzero(col.notna().to_numpy()).tolist()
        assert int(dst.max()) < data[t].num_nodes
        rev = data[t, f"rev_has_{t}", "txn"].edge_index
        assert torch.equal(rev, data["txn", f"has_{t}", t].edge_index.flip(0))
        assert torch.equal(data[t].time, torch.full((data[t].num_nodes,), -1, dtype=torch.long))
    assert torch.equal(data["txn"].time,
                       torch.as_tensor(df["TransactionDT"].to_numpy(), dtype=torch.long))


def test_temporal_sampling_no_future(raw_dir):
    df, _, data = _build(raw_dir)
    seeds = torch.arange(len(df))
    loader = make_loader(data, seeds, batch_size=32, shuffle=True)
    sampled_neighbours = 0
    for batch in loader:
        txn = batch["txn"]
        seed_time = txn.time[:txn.batch_size]
        assert (txn.time <= seed_time[txn.batch]).all()
        sampled_neighbours += txn.num_nodes - txn.batch_size
    # the check above is vacuous unless neighbour transactions were actually sampled
    assert sampled_neighbours > 0


def test_standardize_fits_on_given_rows():
    X = np.array([[1.0, 5.0], [3.0, np.nan], [100.0, 5.0]])
    out = standardize(X, np.array([True, True, False]))
    assert np.allclose(out[:2, 0], [-1.0, 1.0])
    assert out[2, 0] == (100.0 - 2.0) / 1.0
    assert out[1, 1] == 0.0          # NaN -> 0 after standardizing
    assert out[0, 1] == 0.0          # zero-variance column does not divide by zero
    assert not np.isnan(out).any()
