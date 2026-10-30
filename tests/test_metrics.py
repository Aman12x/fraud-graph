import numpy as np

from fraudgraph.metrics import best_cost, evaluate, expected_cost, pr_auc, recall_at_fpr

Y = np.array([0, 0, 0, 0, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.5])
AMT = np.array([10.0, 10, 10, 10, 100, 40])


def test_recall_at_fpr_hand_case():
    # 4 negatives: fpr 0.25 allows one false positive at most
    r = recall_at_fpr(Y, S, fpr=0.25)
    # t=0.6 flags {0.9,0.6}: fpr .25, recall .5 ; t=0.5 flags 3 -> still fpr .25, recall 1 ; t=0.3 -> fpr .5
    assert r == {"threshold": 0.5, "recall": 1.0, "fpr": 0.25}
    r0 = recall_at_fpr(Y, S, fpr=0.0)
    assert r0["threshold"] == 0.9 and r0["recall"] == 0.5 and r0["fpr"] == 0.0
    assert isinstance(r0["recall"], float)


def test_expected_cost_hand_case():
    # threshold 0.5 flags 3 rows (5 each) and misses nothing
    assert expected_cost(Y, S, AMT, 0.5) == 15.0
    # threshold 0.7 flags 1 row (5) and misses the 40 fraud
    assert expected_cost(Y, S, AMT, 0.7) == 45.0


def test_best_cost_not_above_default():
    b = best_cost(Y, S, AMT)
    assert b["cost"] <= expected_cost(Y, S, AMT, 0.5)
    assert b == {"threshold": 0.5, "cost": 15.0}


def test_evaluate_shape():
    e = evaluate(Y, S, AMT)
    assert set(e) == {"pr_auc", "recall_at_fpr", "best_cost", "cost_at_0.5"}
    assert e["pr_auc"] == pr_auc(Y, S)
    assert e["cost_at_0.5"] == 15.0


def _best_cost_reference(y, s, amt):
    # the original O(n^2) implementation, kept as the equivalence oracle
    s = np.asarray(s, dtype=float)
    cands = np.concatenate([np.unique(s), [np.inf]])
    costs = [expected_cost(y, s, amt, t) for t in cands]
    i = int(np.argmin(costs))
    return {"threshold": float(cands[i]), "cost": float(costs[i])}


def test_best_cost_equals_reference_on_random_data():
    rng = np.random.default_rng(7)
    for trial in range(200):
        n = int(rng.integers(1, 60))
        y = rng.random(n) < rng.uniform(0, 0.6)
        s = np.round(rng.random(n), int(rng.integers(1, 3)))   # many tied scores
        amt = rng.integers(0, 40, n).astype(float)              # integer amounts: exact sums, real ties
        assert best_cost(y, s, amt) == _best_cost_reference(y, s, amt), trial


def test_best_cost_all_negative_flags_nothing():
    assert best_cost([0, 0], [0.2, 0.7], [10.0, 10.0]) == {"threshold": float("inf"), "cost": 0.0}


def test_evaluate_at_given_threshold():
    e = evaluate(Y, S, AMT, threshold=0.7)
    assert e["cost_at_val_threshold"] == {"threshold": 0.7, "cost": 45.0}
    assert "cost_at_val_threshold" not in evaluate(Y, S, AMT)
