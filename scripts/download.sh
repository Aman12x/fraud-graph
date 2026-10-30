#!/usr/bin/env bash
# Fetch the IEEE-CIS competition files into data/raw/ (never committed).
# Needs: competition rules accepted on Kaggle, and an API token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
shopt -s nullglob
zips=(data/raw/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "no zip in data/raw after download" >&2
  exit 1
fi
for z in "${zips[@]}"; do
  unzip -o "$z" -d data/raw
done
