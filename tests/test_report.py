import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parent.parent / "scripts" / "report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _ev(pr, rec, cost):
    return {"pr_auc": pr, "recall_at_fpr": {"threshold": 0.5, "recall": rec, "fpr": 0.01},
            "best_cost": {"threshold": 0.4, "cost": cost}, "cost_at_0.5": 1.0}


def test_render_uses_json_values():
    r = {
        "data_profile": {"rows": 590540, "fraud_rate": 0.0349, "identity_coverage": 0.244,
                         "day_min": 1, "day_max": 182},
        "xgb_base": {"test": _ev(0.4567, 0.3219, 12345.6)},
        "xgb_graph": {"test": _ev(None, 0.5, 999.0)},
        "drift": {"weeks": [{"week": 3, "n": 777, "psi": 0.1234, "pr_auc": None}]},
        "latency": {"metrics": {"http_req_duration": {"med": 4.25, "p(95)": 9.75},
                                "http_reqs": {"count": 4321}}},
    }
    out = report.render(r)
    for s in ("590,540", "0.0349", "0.457", "0.322", "$12,346", "n/a", "$999", "777", "0.123",
              "4.2", "9.8", "4321", "constant TransactionDT"):
        assert s in out, s
    assert "| gnn_sage | not run |" in out


def test_missing_file_says_not_run(tmp_path, monkeypatch):
    import fraudgraph.results as res
    monkeypatch.setattr(res, "RESULTS_DIR", tmp_path / "results")
    out = report.render(res.read_results())
    assert out.count("not run") >= 4 * 3 + 3
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- results:start -->\nold\n<!-- results:end -->\nb\n")
    report.update_readme(readme, out)
    t = readme.read_text()
    assert "old" not in t and t.startswith("a\n") and t.endswith("\nb\n")
