"""Fraud evaluation metrics: ranking quality and review-cost trade-off."""
import numpy as np
from sklearn.metrics import average_precision_score

from fraudgraph.config import REVIEW_COST, TARGET_FPR


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def recall_at_fpr(y, s, fpr: float = TARGET_FPR) -> dict:
    """Highest threshold (flag score >= t) whose FPR <= fpr."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    neg, pos = (~y).sum(), y.sum()
    best = None
    for t in np.unique(s)[::-1]:
        flagged = s >= t
        f = float((flagged & ~y).sum() / neg) if neg else 0.0
        if f <= fpr:
            best = (float(t), float((flagged & y).sum() / pos) if pos else 0.0, f)
        else:
            break
    if best is None:  # nothing can be flagged within the budget
        return {"threshold": float(np.inf), "recall": 0.0, "fpr": 0.0}
    return {"threshold": best[0], "recall": best[1], "fpr": best[2]}


def expected_cost(y, s, amt, threshold: float) -> float:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(s, dtype=float) >= threshold
    amt = np.asarray(amt, dtype=float)
    return float(REVIEW_COST * flagged.sum() + amt[y & ~flagged].sum())


def best_cost(y, s, amt) -> dict:
    s = np.asarray(s, dtype=float)
    cands = np.concatenate([np.unique(s), [np.inf]])  # inf = flag nothing
    costs = [expected_cost(y, s, amt, t) for t in cands]
    i = int(np.argmin(costs))
    return {"threshold": float(cands[i]), "cost": float(costs[i])}


def evaluate(y, s, amt) -> dict:
    return {
        "pr_auc": pr_auc(y, s),
        "recall_at_fpr": recall_at_fpr(y, s),
        "best_cost": best_cost(y, s, amt),
        "cost_at_0.5": expected_cost(y, s, amt, 0.5),
    }
