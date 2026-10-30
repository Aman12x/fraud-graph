import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parent.parent / "scripts" / "report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _ev(pr, rec, cost, val_cost=None):
    ev = {"pr_auc": pr, "recall_at_fpr": {"threshold": 0.5, "recall": rec, "fpr": 0.01},
          "best_cost": {"threshold": 0.4, "cost": cost}, "cost_at_0.5": 1.0}
    if val_cost is not None:
        ev["cost_at_val_threshold"] = {"threshold": 0.3, "cost": val_cost}
    return ev


def test_render_uses_json_values():
    r = {
        "data_profile": {"rows": 590540, "fraud_rate": 0.0349, "identity_coverage": 0.244,
                         "day_min": 1, "day_max": 182},
        "xgb_base": {"test": _ev(0.4567, 0.3219, 12345.6, val_cost=23456.4)},
        "xgb_graph": {"test": _ev(None, 0.5, 999.0)},
        "drift": {"weeks": [{"week": 3, "n": 777, "psi": 0.1234, "pr_auc": None}]},
        "latency": {"metrics": {"http_req_duration": {"med": 4.25, "p(95)": 9.75},
                                "http_reqs": {"count": 4321, "rate": 71.25}}},
    }
    out = report.render(r)
    for s in ("590,540", "0.0349", "0.457", "0.322", "$12,346", "n/a", "$999", "777", "0.123",
              "4.2", "9.8", "4321", "constant TransactionDT", "71.2 req/s", "mostly queueing"):
        assert s in out, s
    assert "| gnn_sage | not run |" in out
    assert "expected cost (val-chosen threshold)" in out and "oracle cost" in out
    # headline column is the val-chosen cost, the oracle sits beside it
    assert "| 0.457 | 0.322 | $23,456 | $12,346 |" in out
    # an older result file without the val-threshold cost renders n/a there, not the oracle
    assert "| n/a | 0.500 | n/a | $999 |" in out


def test_missing_file_says_not_run(tmp_path, monkeypatch):
    import fraudgraph.results as res
    monkeypatch.setattr(res, "RESULTS_DIR", tmp_path / "results")
    out = report.render(res.read_results())
    assert out.count("not run") == 4 * 4 + 3
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- results:start -->\nold\n<!-- results:end -->\nb\n")
    report.update_readme(readme, out)
    t = readme.read_text()
    assert "\nold\n" not in t and t.startswith("a\n") and t.endswith("\nb\n")
