# fraud-graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A delayed-label, time-split payment-fraud benchmark on IEEE-CIS comparing XGBoost, XGBoost + point-in-time graph features, a heterogeneous GraphSAGE and a card-sequence Transformer, served behind FastAPI with drift monitoring and a latency benchmark.

**Architecture:** One Python package `fraudgraph` with one module per concern (load → split → entities/graph features → models → online store → API). Offline and online graph features share one counting rule so a parity test can pin train/serve equality. Every model writes `results/<model>.json`; the README table is generated from those files.

**Tech Stack:** Python 3.11, uv, pandas + pyarrow, xgboost, scikit-learn, torch, torch_geometric, FastAPI + uvicorn, prometheus_client, pytest, k6.

**Spec:** `docs/superpowers/specs/2026-10-29-fraud-graph-design.md`

## Global Constraints

- Python 3.11 (`uv venv --python 3.11`); dependencies pinned in `pyproject.toml`, lockfile `uv.lock` committed.
- `data/` is gitignored and never committed (Kaggle competition rules).
- `SEED = 42` (in `fraudgraph/config.py`) for numpy, torch, xgboost and every split.
- `LABEL_DELAY_DAYS = 30`, `REVIEW_COST = 5.0`, `TARGET_FPR = 0.01`, `SEQ_LEN = 16` live in `fraudgraph/config.py` only.
- A transaction's day is `TransactionDT // 86400`.
- Every number in README.md is rendered from `results/*.json` by `scripts/report.py`; no hand-typed figures.
- Tests run on a synthetic fixture (`tests/conftest.py`), never on the Kaggle files; `pytest -q` must pass with `data/` absent.

## Review Focus

1. **Null entity values** (missing email domain, device, address): must yield NO entity, never one giant shared "nan" node that links every identity-less transaction. → Task 3 `test_null_entity_values_create_no_link`.
2. **Same-timestamp transactions**: two rows with equal `TransactionDT` must not count each other as "prior". → Task 3 `test_same_timestamp_not_counted`.
3. **Transactions with no identity row** (left join): device features must be null-safe, the row must not be dropped. → Task 1 `test_left_join_keeps_identityless_rows`.
4. **Unseen entity at serve time** (a brand-new card): `/score` returns a score with zero-count features, no KeyError. → Task 10 `test_unseen_entity_scores`.
5. **Malformed `/score` payload** (missing `TransactionAmt`, string where a number belongs): 422, never 500. → Task 11 `test_malformed_payload_422`.

---

## File Structure

```
pyproject.toml, uv.lock, .gitignore, README.md
scripts/download.sh          Kaggle CLI download into data/raw/
scripts/run_tier1.py         baseline + graph-feature XGBoost → results/
scripts/run_gnn.py           Tier 2 → results/gnn_sage.json
scripts/run_seq.py           Tier 4 → results/seq_transformer.json
scripts/replay.py            Tier 3 drift replay → results/drift.json
scripts/report.py            results/*.json → README table
bench/score.k6.js            k6 load test → results/latency.json
fraudgraph/config.py         constants above
fraudgraph/data.py           load + join + profile
fraudgraph/splits.py         day splits + label-known mask
fraudgraph/entities.py       entity keys
fraudgraph/graph_features.py point-in-time counts (offline)
fraudgraph/tabular.py        transaction feature matrix
fraudgraph/metrics.py        PR-AUC, recall@FPR, expected cost
fraudgraph/models/xgb.py     XGBoost train/predict
fraudgraph/gnn/graph.py      HeteroData builder
fraudgraph/gnn/model.py      hetero GraphSAGE
fraudgraph/seq/windows.py    per-card history windows
fraudgraph/seq/model.py      sequence Transformer
fraudgraph/serve/store.py    online feature store
fraudgraph/serve/app.py      FastAPI app
fraudgraph/drift.py          PSI
fraudgraph/results.py        write/read results JSON
tests/conftest.py            synthetic IEEE-CIS-shaped fixture
tests/test_*.py              one file per module
```

---

### Task 1: Scaffold, loader, data profile

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`data/`, `.venv/`, `*.pt`, `*.ubj`), `scripts/download.sh`, `fraudgraph/config.py`, `fraudgraph/data.py`, `fraudgraph/results.py`, `tests/conftest.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `load_raw(raw_dir: Path) -> pd.DataFrame` (transaction LEFT JOIN identity on `TransactionID`, sorted by `TransactionDT` then `TransactionID`, adds `day: int`); `profile(df) -> dict` with keys `rows, fraud_rate, identity_coverage, day_min, day_max`; `write_result(name: str, payload: dict) -> Path` (writes `results/<name>.json`, adds `git_sha` and `created_at`); `read_results() -> dict[str, dict]`.
- `tests/conftest.py` fixture `raw_dir(tmp_path)` writes a ~200-row synthetic `train_transaction.csv` / `train_identity.csv` with columns `TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD, card1..card6, addr1, addr2, P_emaildomain` and `TransactionID, DeviceType, DeviceInfo`; identity rows for half the transactions; at least one pair of equal `TransactionDT`; some null `P_emaildomain`.

- [ ] **Step 1: Write failing tests** — `test_left_join_keeps_identityless_rows` (row count == transaction count; identity-less rows have null `DeviceInfo`), `test_sorted_by_time` (`df.TransactionDT.is_monotonic_increasing`), `test_day_column` (`df.day == df.TransactionDT // 86400`), `test_profile_keys` (exact key set above; `0 <= identity_coverage <= 1`).
- [ ] **Step 2: Run** `uv run pytest tests/test_data.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `load_raw`, `profile`, `write_result`, `read_results`. `download.sh` runs `kaggle competitions download -c ieee-fraud-detection -p data/raw && unzip -o data/raw/*.zip -d data/raw`.
- [ ] **Step 4: Run** tests → PASS.
- [ ] **Step 5: Manual (needs Kaggle rules accepted):** `bash scripts/download.sh && uv run python -c "from fraudgraph.data import *; from fraudgraph.results import write_result; write_result('data_profile', profile(load_raw(Path('data/raw'))))"` → `results/data_profile.json` exists. Verify the real column names match the fixture; if any differ, fix the fixture and loader before Task 2.
- [ ] **Step 6: Commit** `feat: scaffold, IEEE-CIS loader and data profile`.

### Task 2: Time splits with label delay

**Files:** Create `fraudgraph/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `load_raw` output (`day` column).
- Produces: `@dataclass Splits(train_end_day: int, val_end_day: int)`; `make_splits(df, train_frac=0.6, val_frac=0.2) -> Splits` (boundaries at day quantiles, so no day straddles two splits); `split_masks(df, s) -> tuple[pd.Series, pd.Series, pd.Series]` (train: `day <= train_end_day`, val: `train_end_day < day <= val_end_day`, test: rest); `label_known(df, as_of_day: int, delay: int = LABEL_DELAY_DAYS) -> pd.Series` (`day <= as_of_day - delay`).

- [ ] **Step 1: Failing tests** — `test_splits_disjoint_and_cover` (masks pairwise disjoint, union all rows), `test_no_day_straddles`, `test_label_known_respects_delay` (row at `as_of_day - delay` True, at `as_of_day - delay + 1` False), `test_training_targets_exclude_unlabeled_tail` (train ∧ label_known(train_end_day) excludes the last `delay` days of train).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: day-based time splits with label-delay mask`.

### Task 3: Entity keys and point-in-time graph features

**Files:** Create `fraudgraph/entities.py`, `fraudgraph/graph_features.py`, `tests/test_entities.py`, `tests/test_graph_features.py`

**Interfaces:**
- Produces: `ENTITY_TYPES = ("card", "email", "device", "addr")`; `entity_keys(df) -> pd.DataFrame` with one column per type: `card` = `card1..card6` joined with `|` (null if `card1` null), `email` = `P_emaildomain`, `device` = `DeviceInfo`, `addr` = `addr1|addr2` (null if both null). Null stays null — never the string "nan".
- Produces: `graph_features(df, keys, delay: int = LABEL_DELAY_DAYS) -> pd.DataFrame` indexed like `df`, columns `{t}_prior_n, {t}_prior_fraud, {t}_fraud_rate` for each type. Definitions for transaction i with entity value v:
  - `prior_n` = count of rows with the same v and `TransactionDT < DT_i` (strict).
  - `prior_fraud` = count of rows with the same v, `isFraud == 1` and `day <= day_i - delay`.
  - `fraud_rate` = `(prior_fraud + 1) / (prior_labeled + 20)` where `prior_labeled` = same-v rows with `day <= day_i - delay` (Laplace-style smoothing toward 5%).
  - Null v → all three are 0 / 0 / `1/20`.
- Approach: per entity type, sort by `(v, TransactionDT)`; `prior_n` via groupby cumcount minus same-timestamp ties (use `groupby([v, DT]).cumcount()` correction or rank `method="min"`); label-known counts via `pd.merge_asof` of each row's `day - delay` against per-v cumulative labeled/fraud counts keyed by day. Vectorized; no Python loop over rows.

- [ ] **Step 1: Failing tests**
  - `test_null_entity_values_create_no_link`: two rows with null email and different cards → each `email_prior_n == 0`.
  - `test_same_timestamp_not_counted`: two rows, same card, same `TransactionDT` → both `card_prior_n == 0`.
  - `test_future_label_never_read`: a fraud row at day d only affects `card_prior_fraud` of same-card rows at day ≥ d + delay; at d + delay − 1 it is 0.
  - `test_leakage_shuffle_invariance`: permuting `isFraud` of every row with `day > as_of - delay` leaves features for rows at `as_of` unchanged.
  - `test_matches_bruteforce`: on the fixture, every column equals a nested-loop reference implementation written in the test.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: entity keys and point-in-time graph features with leakage tests`.

### Task 4: Tabular features and metrics

**Files:** Create `fraudgraph/tabular.py`, `fraudgraph/metrics.py`, `tests/test_tabular.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `tabular_features(df) -> pd.DataFrame`: `TransactionAmt`, `log_amt = log1p(amt)`, `hour = (DT // 3600) % 24`, every other numeric transaction column except `TransactionID, isFraud, TransactionDT, day`, and categorical columns (`ProductCD, card4, card6, P_emaildomain, DeviceType`) as pandas `category` codes with null → -1. Same columns in the same order for any input (fit-free).
- Produces: `pr_auc(y, s) -> float` (sklearn `average_precision_score`); `recall_at_fpr(y, s, fpr=TARGET_FPR) -> dict(threshold, recall, fpr)` (highest threshold whose FPR ≤ target); `expected_cost(y, s, amt, threshold) -> float` = `REVIEW_COST * flagged + sum(amt[fraud & ~flagged])`; `best_cost(y, s, amt) -> dict(threshold, cost)` scanning unique score thresholds; `evaluate(y, s, amt) -> dict` combining all four.

- [ ] **Step 1: Failing tests** — `test_columns_stable` (two disjoint slices give identical column lists), `test_recall_at_fpr_hand_case` (6 hand-built scores, known answer), `test_expected_cost_hand_case`, `test_best_cost_not_above_default` (best ≤ cost at 0.5).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat: tabular features and fraud evaluation metrics`.

### Task 5: Tier 1 experiment (XGBoost vs XGBoost + graph features)

**Files:** Create `fraudgraph/models/xgb.py`, `scripts/run_tier1.py`, `tests/test_xgb.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `train_xgb(X, y, X_val, y_val) -> xgb.XGBClassifier` (`n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist", random_state=SEED`); `save_model(m, path)` / `load_model(path)` using `.ubj`.
- `run_tier1.py`: train rows = train ∧ `label_known(df, train_end_day)`; validation = val split (early stopping); report on test. Writes `results/xgb_base.json` and `results/xgb_graph.json`, each `{model, features: [...], n_train, val: evaluate(...), test: evaluate(...)}`; saves `models/xgb_graph.ubj` for Task 11.

- [ ] **Step 1: Failing test** `test_train_xgb_learns_signal` (synthetic separable data → val PR-AUC > 0.9) and `test_save_load_roundtrip` (identical predictions).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Manual:** `uv run python scripts/run_tier1.py` → both JSON files exist with `test.pr_auc` keys.
- [ ] **Step 6: Commit** `feat: Tier 1 XGBoost baseline vs point-in-time graph features`.

### Task 6: Heterogeneous graph with temporal sampling

**Files:** Create `fraudgraph/gnn/graph.py`, `tests/test_gnn_graph.py`

**Interfaces:**
- Consumes: `entity_keys`, `tabular_features`.
- Produces: `build_hetero(df, keys, X: np.ndarray) -> HeteroData`: node type `txn` (`x` = standardized X, `y`, `time` = `TransactionDT`, `amt`), one node type per non-null value of each entity type with `x = ones(n,1)` and `time = -1`; edges `("txn", f"has_{t}", t)` and reverse `(t, f"rev_has_{t}", "txn")`. Labels are stored only in `data["txn"].y`, never in `x`.
- Produces: `make_loader(data, seed_idx, batch_size=1024, num_neighbors=[15, 10], shuffle=False) -> NeighborLoader` with `time_attr="time"`, `temporal_strategy="last"` and `disjoint=True`, so each seed gets its own subgraph containing only transactions with `time <= seed time`.

- [ ] **Step 1: Failing tests** — `test_no_label_in_features` (`x` columns exclude `isFraud`), `test_null_values_no_entity_node`, `test_temporal_sampling_no_future` (for every batch: `batch["txn"].time <= seed_time[batch["txn"].batch]` elementwise, where `seed_time = batch["txn"].time[:batch["txn"].batch_size]`).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat: heterogeneous transaction-entity graph with temporal neighbour sampling`.

### Task 7: Tier 2 GraphSAGE

**Files:** Create `fraudgraph/gnn/model.py`, `scripts/run_gnn.py`, `tests/test_gnn_model.py`

**Interfaces:**
- Consumes: Task 6 `build_hetero`, `make_loader`; Task 4 `evaluate`; Task 1 `write_result`.
- Produces: `HeteroSAGE(metadata, hidden=64, num_layers=2, dropout=0.2)` = `torch_geometric.nn.to_hetero` over a homogeneous `SAGEConv` stack + linear head on `txn`; `forward(batch) -> Tensor[logits for txn seeds]`. Loss `BCEWithLogitsLoss(pos_weight = negatives/positives in train targets)`, Adam lr 1e-3, up to 20 epochs, early stop on val PR-AUC (patience 3), seed 42.
- `run_gnn.py` uses the SAME train/val/test rows and label-known mask as Task 5 and writes `results/gnn_sage.json` in the Task 5 schema.

- [ ] **Step 1: Failing test** `test_overfits_tiny_graph` (fixture, 50 steps → train loss drops by half) and `test_output_shape`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Manual:** `uv run python scripts/run_gnn.py` → `results/gnn_sage.json`.
- [ ] **Step 6: Commit** `feat: Tier 2 heterogeneous GraphSAGE fraud model`.

### Task 8: Card history windows

**Files:** Create `fraudgraph/seq/windows.py`, `tests/test_windows.py`

**Interfaces:**
- Consumes: `entity_keys` (`card`), `tabular_features` (standardized array).
- Produces: `card_windows(df, keys, X: np.ndarray, k: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]` → `hist` shape `(N, k, F+1)` (prior transactions' features plus `log1p(seconds since that txn)`), `mask` shape `(N, k)` True where padded. Only rows with the same card and `TransactionDT` strictly less than the current row; most recent last. Null card → fully padded.

- [ ] **Step 1: Failing tests** — `test_window_strictly_prior` (same-timestamp twin excluded), `test_null_card_all_padded`, `test_order_most_recent_last`, `test_k_truncation` (card with 20 prior rows keeps the latest 16).
- [ ] **Step 2–4:** run → FAIL, implement (vectorized per card group with numpy stride indexing), run → PASS.
- [ ] **Step 5: Commit** `feat: per-card prior-transaction windows`.

### Task 9: Tier 4 sequence Transformer

**Files:** Create `fraudgraph/seq/model.py`, `scripts/run_seq.py`, `tests/test_seq_model.py`

**Interfaces:**
- Consumes: Task 8 `card_windows`; Task 4; Task 1.
- Produces: `SeqTransformer(n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1)`: linear projection + learned position embedding, `nn.TransformerEncoder(batch_first=True)` with `src_key_padding_mask=mask`, the CURRENT transaction's features projected as an extra final token (never masked), logit read from that token. `forward(hist, mask, current) -> Tensor[N]`. Same loss/optimizer/early-stop recipe as Task 7.
- `run_seq.py` writes `results/seq_transformer.json` in the Task 5 schema, same rows as Task 5.

- [ ] **Step 1: Failing tests** — `test_padding_invariance` (changing padded positions' values does not change the output), `test_all_padded_row_finite` (null-card row gives a finite logit), `test_overfits_tiny`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Manual:** `uv run python scripts/run_seq.py` → `results/seq_transformer.json`.
- [ ] **Step 6: Commit** `feat: Tier 4 card-sequence Transformer`.

### Task 10: Online feature store with offline parity

**Files:** Create `fraudgraph/serve/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: Task 3 definitions (same formulas; import the smoothing constants from `graph_features`).
- Produces: `class EntityStore` with `features(txn: dict) -> dict[str, float]` (the Task 3 columns, computed from state strictly before `txn["TransactionDT"]`), `observe(txn: dict) -> None` (adds to counts after scoring), `ingest_label(txn_id: int, is_fraud: bool, day: int) -> None` (a label becomes visible to rows with `day >= label_day + LABEL_DELAY_DAYS`). State: per (type, value) count map plus a per (type, value) list of `(day, is_fraud)` labeled events.
- Same-timestamp rule: `observe` of a txn at DT does not affect `features` of another txn at the same DT (buffer observes until DT advances).

- [ ] **Step 1: Failing tests** — `test_offline_online_parity` (replay the fixture in time order: for each row `features()` → `observe()` → `ingest_label()`; result equals `graph_features(df, keys)` row for row within 1e-9), `test_unseen_entity_scores` (fresh store, new card → zero counts, rate `1/20`, no exception).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat: online entity feature store pinned to offline features`.

### Task 11: FastAPI scoring service

**Files:** Create `fraudgraph/serve/app.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `EntityStore`, `tabular_features`, `entity_keys`, `load_model` (`models/xgb_graph.ubj`, path from env `FRAUDGRAPH_MODEL`).
- Produces: `create_app(model, store) -> FastAPI`. `POST /score` body = Pydantic `Txn` (`TransactionID: int, TransactionDT: int, TransactionAmt: float` required; the other raw columns optional) → `{"TransactionID", "score": float, "model": "xgb_graph", "latency_ms": float}`; scoring then `store.observe`. `POST /label` → `store.ingest_label`. `GET /metrics` = prometheus_client exposition with `score_latency_seconds` (Histogram) and `scores_total` (Counter). `GET /health` → `{"ok": true}`. Module-level `app = create_app(load_model(os.environ["FRAUDGRAPH_MODEL"]), EntityStore())`, built lazily so importing the module in tests does not need the env var.

- [ ] **Step 1: Failing tests** (FastAPI `TestClient`, a tiny XGBoost fitted in the test) — `test_score_returns_probability` (0 ≤ score ≤ 1), `test_malformed_payload_422` (missing `TransactionAmt`; `"TransactionAmt": "abc"`), `test_metrics_exposes_histogram`, `test_score_matches_offline` (same row scored via API and via offline features + model → equal within 1e-6).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat: FastAPI scoring service with Prometheus metrics`.

### Task 12: Drift replay and latency benchmark

**Files:** Create `fraudgraph/drift.py`, `scripts/replay.py`, `bench/score.k6.js`, `tests/test_drift.py`

**Interfaces:**
- Produces: `psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float` (bins = ref quantiles, epsilon 1e-6 on empty bins).
- `replay.py`: scores the validation split offline as the reference distribution, then replays the test split week by week (`day // 7`) through `EntityStore` + model in time order, writing `results/drift.json` = `[{week, n, psi, pr_auc}]`.
- `bench/score.k6.js`: ramps 1 → 50 virtual users over 60 s against `POST /score` with test-split rows from `bench/payloads.json` (generated by `replay.py --dump-payloads 5000`); run as `k6 run --summary-export results/latency.json bench/score.k6.js`.

- [ ] **Step 1: Failing tests** — `test_psi_zero_on_identical`, `test_psi_positive_on_shift`, `test_psi_handles_empty_bins` (finite).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Manual:** `uv run python scripts/replay.py --dump-payloads 5000`, start `uv run uvicorn fraudgraph.serve.app:app`, run k6 → `results/drift.json` and `results/latency.json` exist.
- [ ] **Step 6: Commit** `feat: weekly drift replay and k6 latency benchmark`.

### Task 13: Generated report and README

**Files:** Create `scripts/report.py`, `README.md`, `tests/test_report.py`

**Interfaces:**
- Consumes: `read_results()`.
- Produces: `render(results: dict) -> str` — markdown: data profile line, a model table (`model | test PR-AUC | recall @ 1% FPR | best expected cost`), drift table, latency line (p50/p95 from `results/latency.json` `metrics.http_req_duration`). `report.py` replaces the block between `<!-- results:start -->` and `<!-- results:end -->` in README.md. A missing results file renders "not run", never a placeholder number.

- [ ] **Step 1: Failing tests** — `test_render_uses_json_values` (fixture results dict → its numbers appear verbatim), `test_missing_file_says_not_run`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5:** README prose: what it is, how to download, how to run each tier, the leakage rules (delay, strict prior, temporal sampling, parity test). Run `uv run python scripts/report.py`.
- [ ] **Step 6: Commit** `docs: generated results report and README`.
