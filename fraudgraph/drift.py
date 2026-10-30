"""Population stability index between a reference and a current score distribution."""
import numpy as np

EPS = 1e-6


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI over bins cut at the reference's quantiles; empty bins get epsilon mass."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.where(r == 0, EPS, r), np.where(c == 0, EPS, c)
    return float(np.sum((c - r) * np.log(c / r)))
