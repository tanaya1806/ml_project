---
type: Experiment
title: Phase 1 - Baseline Temporal Anomaly Detection
description: GIN+GRU trained on normal (control) dynamics; unsupervised anomaly score evaluated on the held-out test set.
resource: /outputs/experiments/phase1
sources:
  - resource: /configs/phase1.yaml
  - resource: /docs/phase1_report.md
executor: { resource: /src/training/train_phase1.py }
status: stable
generated: { by: process:train_phase1+detect_baseline, at: 2026-09-14T00:00:00Z }
verified: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Setup

328 train-control subjects x 20 seed ROIs, 5 epochs, Adam lr=1e-3, MPS device. Threshold
calibrated on 70 val-control subjects. Evaluated on all 131 test subjects (70 control +
61 ASD), untouched until evaluation.

# Results

| Metric | Value |
|---|---|
| Precision | 0.500 |
| Recall | 0.033 |
| F1 | 0.062 |
| AUROC | 0.453 |
| AUPRC | 0.467 |

Full numbers: `/outputs/experiments/phase1/metrics.json`,
`/outputs/experiments/phase1/training_report.json`.

# Tests

43 passed, 3 skipped (raw-data tests; `data/raw/` not shipped).
