"""Online entity feature store whose output equals fraudgraph.graph_features row for row.

Scoring loop for each live txn, in non-decreasing TransactionDT order:
    feats = store.features(txn)            # state strictly before txn["TransactionDT"]
    store.observe(txn)                     # after scoring
    ...later: store.ingest_label(txn_id, is_fraud)

Rules, identical to graph_features:
  prior_n     = observed same-value txns with a strictly smaller TransactionDT.
  prior_fraud = labeled same-value fraud txns with label_day <= day_i - LABEL_DELAY_DAYS.
  fraud_rate  = (prior_fraud + RATE_PRIOR_FRAUD) / (prior_labeled + RATE_PRIOR_TOTAL).
  Null entity value -> 0, 0, RATE_PRIOR_FRAUD / RATE_PRIOR_TOTAL.
day_i is always TransactionDT // 86400; txn["day"] and txn["isFraud"] are never read.

Same-timestamp rule: observe() only buffers. A buffered txn moves into the prior_n counters
when features() is called for a txn with a strictly larger TransactionDT, so two txns at the
same DT never see each other. features() refuses a DT earlier than the last one scored, since
the buffer has already been flushed past it and the answer would leak later activity.

Labels: observe() records each txn's OWN day (TransactionDT // 86400) and ingest_label uses
that recorded day, not the day the label arrived; visibility is by whole days, exactly the
offline rule. Observing a TransactionID twice raises ValueError. Label
arrival order does not matter as long as each label is ingested before any txn that should
see it. Per (type, value) the store keeps sorted label-day lists (bisect), so features() is
O(log n) per entity and never scans the event history.
"""
from bisect import bisect_right, insort
from collections import defaultdict, deque

import pandas as pd

from fraudgraph.config import LABEL_DELAY_DAYS
from fraudgraph.entities import ADDR_COLS, CARD_COLS, ENTITY_TYPES, entity_keys
from fraudgraph.graph_features import RATE_PRIOR_FRAUD, RATE_PRIOR_TOTAL, feature_columns

SECONDS_PER_DAY = 86400
_KEY_COLS = (*CARD_COLS, *ADDR_COLS, "P_emaildomain", "DeviceInfo")


def txn_entity_values(txn: dict) -> dict[str, str | None]:
    """Entity values of one txn via the offline entity_keys (one rendering rule)."""
    row = {c: txn.get(c) for c in _KEY_COLS}
    keys = entity_keys(pd.DataFrame([row]))
    return {t: (None if pd.isna(keys.at[0, t]) else keys.at[0, t]) for t in ENTITY_TYPES}


def _day(txn: dict) -> int:
    return int(txn["TransactionDT"]) // SECONDS_PER_DAY


class EntityStore:
    def __init__(self) -> None:
        self._prior_n: dict[tuple[str, str], int] = defaultdict(int)
        self._labeled_days: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._fraud_days: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._pending: deque[tuple[int, dict]] = deque()  # (TransactionDT, entity values)
        self._values_by_txn: dict[int, dict[str, str | None]] = {}
        self._day_by_txn: dict[int, int] = {}
        self._labeled: set[int] = set()
        self._last_dt: int | None = None

    def _flush_before(self, dt: int) -> None:
        # _pending is in DT order (features() refuses an earlier DT), so pop from the front.
        while self._pending and self._pending[0][0] < dt:
            _, vals = self._pending.popleft()
            for t, v in vals.items():
                if v is not None:
                    self._prior_n[(t, v)] += 1

    def features(self, txn: dict) -> dict[str, float]:
        dt = int(txn["TransactionDT"])
        if self._last_dt is not None and dt < self._last_dt:
            raise ValueError(f"txn at DT {dt} is earlier than the last scored DT {self._last_dt}")
        self._last_dt = dt
        self._flush_before(dt)
        cutoff = _day(txn) - LABEL_DELAY_DAYS
        out: dict[str, float] = {}
        for t, v in txn_entity_values(txn).items():
            n = fraud = labeled = 0
            if v is not None:
                k = (t, v)
                n = self._prior_n.get(k, 0)
                fraud = bisect_right(self._fraud_days.get(k, ()), cutoff)
                labeled = bisect_right(self._labeled_days.get(k, ()), cutoff)
            out[f"{t}_prior_n"] = float(n)
            out[f"{t}_prior_fraud"] = float(fraud)
            out[f"{t}_fraud_rate"] = (fraud + RATE_PRIOR_FRAUD) / (labeled + RATE_PRIOR_TOTAL)
        return {c: out[c] for c in feature_columns()}

    def observe(self, txn: dict) -> None:
        tid = int(txn["TransactionID"])
        if tid in self._values_by_txn:
            raise ValueError(f"TransactionID {tid} was already observed")
        vals = txn_entity_values(txn)
        self._values_by_txn[tid] = vals
        self._day_by_txn[tid] = _day(txn)
        self._pending.append((int(txn["TransactionDT"]), vals))

    def has_observed(self, txn_id: int) -> bool:
        return int(txn_id) in self._values_by_txn

    @property
    def last_dt(self) -> int | None:
        return self._last_dt

    def ingest_label(self, txn_id: int, is_fraud: bool) -> None:
        """Record txn_id's label. Visible to day_i >= (txn's own day) + delay."""
        txn_id = int(txn_id)
        if txn_id not in self._values_by_txn:
            raise KeyError(f"label for TransactionID {txn_id}, which was never observed")
        if txn_id in self._labeled:
            raise ValueError(f"TransactionID {txn_id} is already labeled")
        self._labeled.add(txn_id)
        day = self._day_by_txn[txn_id]
        for t, v in self._values_by_txn[txn_id].items():
            if v is None:
                continue
            insort(self._labeled_days[(t, v)], day)
            if is_fraud:
                insort(self._fraud_days[(t, v)], day)
