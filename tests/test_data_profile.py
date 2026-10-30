import importlib.util
from pathlib import Path

from fraudgraph.data import load_raw

SPEC = importlib.util.spec_from_file_location("data_profile_script", Path(__file__).resolve().parent.parent / "scripts" / "data_profile.py")
prof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prof)


def test_run_writes_data_profile(raw_dir, tmp_path, monkeypatch):
    import fraudgraph.results as res
    monkeypatch.setattr(res, "RESULTS_DIR", tmp_path / "results")
    out = prof.run(load_raw(raw_dir))
    stored = res.read_results()["data_profile"]
    assert stored["rows"] == out["rows"] == 227
    assert {"fraud_rate", "identity_coverage", "day_min", "day_max", "git_sha"} <= stored.keys()
