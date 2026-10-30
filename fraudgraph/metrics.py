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
    """Cheapest threshold over every unique score plus inf (flag nothing), on these labels.

    An oracle when called on the evaluation set itself: report it beside, never instead of,
    the cost at a threshold chosen on validation. O(n log n): one sort, then cumulative sums
    of row counts and fraud amounts below each candidate. Ties go to the lowest threshold.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    amt = np.asarray(amt, dtype=float)
    uniq, inv = np.unique(s, return_inverse=True)
    below_n = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(uniq)))])
    below_fraud = np.concatenate([[0.0], np.cumsum(np.bincount(inv, weights=amt * y,
                                                               minlength=len(uniq)))])
    # candidate j < len(uniq) flags s >= uniq[j]; the last candidate (inf) flags nothing
    costs = REVIEW_COST * (len(s) - below_n) + below_fraud
    i = int(np.argmin(costs))
    thr = float(uniq[i]) if i < len(uniq) else float(np.inf)
    return {"threshold": thr, "cost": expected_cost(y, s, amt, thr)}


def evaluate(y, s, amt, threshold: float | None = None) -> dict:
    """Metrics on one split. Pass `threshold` (chosen on validation) to add the honest cost:
    `cost_at_val_threshold`. `best_cost` picks its threshold on these same labels."""
    out = {
        "pr_auc": pr_auc(y, s),
        "recall_at_fpr": recall_at_fpr(y, s),
        "best_cost": best_cost(y, s, amt),
        "cost_at_0.5": expected_cost(y, s, amt, 0.5),
    }
    if threshold is not None:
        out["cost_at_val_threshold"] = {"threshold": float(threshold),
                                        "cost": expected_cost(y, s, amt, threshold)}
    return out
