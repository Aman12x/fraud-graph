from fraudgraph import results


def test_write_and_read_results(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    p = results.write_result("x", {"a": 1})
    assert p.name == "x.json"
    got = results.read_results()["x"]
    assert got["a"] == 1 and "git_sha" in got and "created_at" in got
