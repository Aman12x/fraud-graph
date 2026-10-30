"""Write results/data_profile.json from the raw Kaggle files."""
import argparse
from pathlib import Path

import pandas as pd

import fraudgraph.results as results
from fraudgraph.data import load_raw, profile

ROOT = Path(__file__).resolve().parent.parent


def run(df: pd.DataFrame) -> dict:
    results.write_result("data_profile", profile(df))
    return profile(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    run(load_raw(ap.parse_args().raw_dir))
