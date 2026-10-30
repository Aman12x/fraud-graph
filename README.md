# fraud-graph

A delayed-label payment-fraud benchmark on the IEEE-CIS fraud dataset. It asks one question: does relational
structure (shared cards, emails, devices and addresses) help catch fraud once you respect the fact that labels
arrive late? Four models are compared on the same temporal split: gradient-boosted trees on transaction
features alone, the same trees with point-in-time graph features, a heterogeneous GraphSAGE network, and a
transformer over each card's recent history. Models are scored on PR-AUC, recall at a fixed false-positive
rate, and expected review cost. A FastAPI service scores transactions online, with a drift replay and a k6
latency benchmark.

## Results

Every number below is rendered from `results/*.json` by `scripts/report.py`. Nothing here is typed by hand.

The headline cost applies the threshold that minimised expected cost on the validation weeks to the test weeks. The oracle column picks its threshold on the test labels themselves, so it is the floor for that model's scores, reachable only with hindsight; it is shown only for comparison.

<!-- results:start -->
Data profile: not run.

| model | test PR-AUC | recall @ 1% FPR | expected cost (val-chosen threshold) | oracle cost (threshold tuned on test; not achievable) |
|---|---|---|---|---|
| xgb_base | not run | not run | not run | not run |
| xgb_graph | not run | not run | not run | not run |
| gnn_sage | not run | not run | not run | not run |
| seq_transformer | not run | not run | not run | not run |

Drift: not run.

Scoring latency: not run.
<!-- results:end -->

## Get the data

1. Accept the competition rules on the Kaggle page for `ieee-fraud-detection` (downloads fail until you do).
2. Put your API token at `~/.kaggle/kaggle.json` (`chmod 600` it).
3. Run `uv sync`. The `kaggle` CLI is a pinned dev dependency, so nothing is installed globally.
4. Run `scripts/download.sh` from anywhere. It calls `uv run kaggle`, then unzips every downloaded zip into
   `data/raw/`, which is never committed.

## Run

Install with `uv sync`. Each script writes a JSON file into `results/`.

- Data profile, after download: `uv run python scripts/data_profile.py` writes `results/data_profile.json`.

- Tier 1, XGBoost with and without graph features: `uv run python scripts/run_tier1.py`
- Tier 2, heterogeneous GNN: `uv run python scripts/run_gnn.py`
- Tier 4, card-history transformer: `uv run python scripts/run_seq.py`
- Tier 3, serving and monitoring: `uv run python scripts/replay.py` runs the weekly drift replay through
  the online store. It needs `models/xgb_graph.ubj` and its `.features.json` from `run_tier1.py` first.
  `uv run python scripts/replay.py --dump-payloads N` writes the k6 payloads. Start the service with
  `FRAUDGRAPH_MODEL=models/xgb_graph.ubj uv run uvicorn fraudgraph.serve.app:app`; it exposes Prometheus
  metrics at `/metrics`. Then run `k6 run --summary-export results/latency.json bench/score.k6.js`.
- Report: `uv run python scripts/report.py` refreshes the block above.
- Tests: `uv run pytest -q` (uses a synthetic fixture, no data needed).

## Leakage guarantees

- Label delay: a label is usable at day t only if its transaction happened at least 30 days earlier. The
  delay is a constant in `fraudgraph/config.py`. Training rows are also restricted to labels known at the
  end of the training period.
- Strict prior: a feature for a transaction only counts events with a strictly earlier timestamp. Labels
  are never model inputs, only feeding neighbour fraud counts, and only once they are old enough.
- Temporal neighbour sampling: the GNN loader samples only neighbours from the past, so a transaction
  never sees its own future through the graph.
- Online/offline parity: a test replays transactions through the online store and checks the features
  match the offline computation exactly.

## Known hazard (macOS)

Importing xgboost before torch can deadlock torch training through a clash of OpenMP runtimes. Import
torch first. The test suite's `conftest.py` does this.
