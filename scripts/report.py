"""Render results/*.json into the generated block of README.md. No figure is ever hand-typed."""
from pathlib import Path

import fraudgraph.results as results

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
MODELS = ("xgb_base", "xgb_graph", "gnn_sage", "seq_transformer")
NOT_RUN = "not run"


def _rate(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _cost(v) -> str:
    return "n/a" if v is None else f"${v:,.0f}"


def _profile(r: dict) -> str:
    p = r.get("data_profile")
    if p is None:
        return f"Data profile: {NOT_RUN}."
    return (f"Data profile: {p['rows']:,} rows, fraud rate {p['fraud_rate']:.4f}, "
            f"identity coverage {p['identity_coverage']:.3f}, days {p['day_min']} to {p['day_max']}.")


def _models(r: dict) -> list[str]:
    lines = ["| model | test PR-AUC | recall @ 1% FPR | expected cost (val-chosen threshold) "
             "| oracle cost (threshold tuned on test; not achievable) |", "|---|---|---|---|---|"]
    for name in MODELS:
        m = r.get(name)
        if m is None:
            lines.append(f"| {name} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |")
            continue
        t = m["test"]
        val_thr = (t.get("cost_at_val_threshold") or {}).get("cost")
        lines.append(f"| {name} | {_rate(t.get('pr_auc'))} | {_rate(t['recall_at_fpr']['recall'])} "
                     f"| {_cost(val_thr)} | {_cost(t['best_cost']['cost'])} |")
    return lines


def _drift(r: dict) -> list[str]:
    d = r.get("drift")
    if d is None:
        return [f"Drift: {NOT_RUN}."]
    lines = ["| week | n | PSI | PR-AUC |", "|---|---|---|---|"]
    for w in d["weeks"]:
        lines.append(f"| {w['week']} | {w['n']} | {_rate(w.get('psi'))} | {_rate(w.get('pr_auc'))} |")
    return lines


def _latency(r: dict) -> str:
    lat = r.get("latency")
    if lat is None:
        return f"Scoring latency: {NOT_RUN}."
    m = lat["metrics"]
    dur = m["http_req_duration"]
    r = m["http_reqs"].get("rate")
    rate = f" ({r:.1f} req/s)" if r is not None else ""
    return (f"Scoring latency (k6, POST /score): p50 {dur['med']:.1f} ms, p95 {dur['p(95)']:.1f} ms "
            f"over {int(m['http_reqs']['count'])} requests{rate}. Caveats: one uvicorn worker "
            "serialises requests through the feature-store lock, so under up to 50 concurrent users "
            "these latencies are mostly queueing, not per-request work; and the benchmark uses a "
            "constant TransactionDT, so every request sees zero prior-neighbour counts and the "
            "latency excludes real neighbour-count work.")


def render(r: dict) -> str:
    parts = [_profile(r), "", *_models(r), "", *_drift(r), "", _latency(r)]
    return "\n".join(parts)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}")


if __name__ == "__main__":
    update_readme(ROOT / "README.md", render(results.read_results()))
