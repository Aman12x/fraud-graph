# fraud-graph — design

Scoped 2026-10-29 (Stripe Radar ML resume gap). Dataset decision: IEEE-CIS (default; Elliptic rejected because it has no
card/device/email identity fields). Tier 4 included at his instruction.

## What it is

A payment-fraud model that links card transactions through the cards, devices, email domains and addresses they share,
and scores each transaction from its own fields plus its graph neighbourhood and its card's recent history. Trained and
evaluated the way a production fraud system lives: labels arrive late, patterns drift, and blocking a good customer costs
something.

## Data

- Kaggle competition `ieee-fraud-detection` (Vesta e-commerce transactions): `train_transaction.csv` +
  `train_identity.csv`, joined on `TransactionID`, label `isFraud`, time `TransactionDT` (seconds from an unknown
  reference).
- Competition rules: the data is NEVER committed. The repo ships `scripts/download.sh` (Kaggle CLI; he accepts the
  rules once in the browser) and reproduces everything from it.
- Row counts, fraud prevalence, identity-join coverage and day span are MEASURED on download into
  `results/data_profile.json`. No number about the data appears anywhere until that file exists.

## Tiers

1. **Delayed-label baseline + graph features.** Time-ordered train / val / test by day. A label is usable at time t only
   if its transaction is at least `label_delay_days` (default 30) older than t — that is the chargeback delay. XGBoost on
   transaction fields, then XGBoost + point-in-time graph features (per entity: prior transaction count, prior
   known-fraud count, smoothed known-fraud rate). A leakage test fails the build if any feature reads a future label or a
   same-timestamp row. Metrics: PR-AUC, recall at 1% false-positive rate (legitimate-user impact), and expected cost with
   a $5 review cost per flagged transaction plus the missed fraud amount (the framing from Credit-Card-Fraud-Detection).
2. **Heterogeneous GraphSAGE** (PyTorch Geometric) on the transaction–entity graph, temporal neighbour sampling so a
   transaction only sees neighbours at or before its own time; labels are never node features. Compared head to head
   with Tier 1 on the same splits. A null result is reported as a null result.
3. **Serving + monitoring.** FastAPI `/score` backed by an in-memory online feature store that computes the SAME graph
   features as the offline code (parity test), Prometheus `/metrics`, a week-by-week replay of the test period with
   score PSI drift against the validation reference, and a k6 load test exporting p50/p95 latency.
4. **Card-sequence transformer.** A small Transformer encoder over each card's last K prior transactions plus the
   current one, compared on the same splits.

## Rules

- Every number in the README is rendered from `results/*.json` by `scripts/report.py`; nothing is typed by hand.
- Seed 42 everywhere. Tiers 1 and 3 run on CPU; Tiers 2 and 4 run on CPU or one GPU session.
- Resume use: a numberless technical-product bullet once the repo is public and curls 200; content.md's "Never: GNN"
  line is lifted only then.

## Out of scope

Distributed training, Kubernetes, a real feature-store product (Feast etc.), streaming ingestion (pulse-stream covers
Kafka), LLMs.
