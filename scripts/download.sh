#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw && unzip -o data/raw/*.zip -d data/raw
