#!/usr/bin/env bash
# Assemble ./space_build with only what the Hugging Face Space needs (code, small model files, front end). Run from the repo root.
set -euo pipefail
OUT=space_build; rm -rf "$OUT"; mkdir -p "$OUT/configs" "$OUT/outputs/experiments/phase1/checkpoints" "$OUT/outputs/experiments/severity" "$OUT/testing_data/app"
rsync -a --exclude '__pycache__' --exclude '*.pyc' src/ "$OUT/src/"
cp configs/phase1.yaml "$OUT/configs/"
cp outputs/experiments/phase1/checkpoints/best.pt "$OUT/outputs/experiments/phase1/checkpoints/"
cp outputs/experiments/phase1/metrics.json "$OUT/outputs/experiments/phase1/"
cp outputs/experiments/severity/severity_model.joblib outputs/experiments/severity/severity_results.json "$OUT/outputs/experiments/severity/"
cp testing_data/app/index.html testing_data/app/server.py testing_data/app/store.py testing_data/app/brain3d.js testing_data/app/brain.json "$OUT/testing_data/app/"
cp testing_data/answer_key_and_results.csv "$OUT/testing_data/"; mkdir -p "$OUT/final testing time series data"; cp "final testing time series data/comparison_results.json" "$OUT/final testing time series data/"
cp deploy/Dockerfile "$OUT/Dockerfile"; cp deploy/requirements-space.txt "$OUT/requirements.txt"; cp deploy/space_README.md "$OUT/README.md"
du -sh "$OUT"; echo "Built $OUT/ - copy its contents into your Hugging Face Space repo (see deploy/README.md)."
