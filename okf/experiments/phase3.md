---
type: Experiment
title: Phase 3 - Few-Shot Final Comparison
description: Few-shot-only vs. few-shot+structure-aware-generated vs. few-shot+random-generated, at 1/5/10 real ASD shots, 5 trials each.
resource: /outputs/experiments/phase3
sources:
  - resource: /configs/phase3.yaml
  - resource: /docs/phase3_report.md
executor: { resource: /src/evaluate/phase3_final_comparison.py }
status: stable
generated: { by: process:phase3_final_comparison, at: 2026-09-14T00:00:00Z }
verified: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Setup

Frozen Phase 1 GIN+GRU; logistic-regression head on a 20-dim per-seed score vector.
Negatives = all 328 train-control subjects. Positives = k real ASD support subjects
(1/5/10-shot, 5 independent draws), optionally expanded with 30 generated anomalies
(structure-aware or random, high severity). Evaluated on the same untouched test set
for every condition/trial.

# Results (mean over 5 trials)

| Shot | Condition | F1 | AUROC |
|---|---|---|---|
| 1 | few-shot only | 0.006 | 0.516 |
| 1 | +structure-aware | 0.387 | 0.497 |
| 1 | +random | 0.403 | 0.506 |
| 5 | few-shot only | 0.149 | 0.485 |
| 5 | +structure-aware | 0.390 | 0.486 |
| 5 | +random | 0.399 | 0.502 |
| 10 | few-shot only | 0.311 | 0.524 |
| 10 | +structure-aware | 0.402 | 0.501 |
| 10 | +random | 0.431 | 0.515 |

Full numbers, per-trial results, and deltas: `/outputs/experiments/phase3/final_comparison.json`,
`/outputs/experiments/phase3/raw_trial_results.json`.

# Honest finding

Expanding the training set with generated data (either method) substantially improves
F1/Recall at low shot counts, but AUROC/AUPRC stay near chance across all conditions,
and structure-aware does not show a clear edge over the random baseline in this
classifier — unlike Phase 2's raw detectability comparison. See
[/okf/limitations.md](../limitations.md) for discussion. Not adjusted to show a
preferred result.
