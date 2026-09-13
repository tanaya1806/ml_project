---
type: Model
title: Anomaly Detector (baseline + few-shot-adapted)
description: Frozen GIN+GRU anomaly score, plus (Phase 3) a logistic-regression head on per-seed score vectors adaptable with few real/generated labeled examples.
resource: /src/evaluate/detect_baseline.py
tags: [detector, few-shot, evaluation]
runtime: python (scikit-learn, pytorch)
computation: /src/evaluate/phase3_final_comparison.py
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Two forms

**Phase 1 (unsupervised baseline):** subject-level score = max `A_t` over seeds/windows;
threshold = 95th percentile of val-control scores. No labeled anomalous examples used.
Test AUROC 0.453 (near chance) — reported honestly as the "before" state.

**Phase 3 (few-shot-adapted):** frozen GIN+GRU stays fixed; a logistic-regression head
is fit on a per-seed max-score feature VECTOR (one dimension per seed ROI, not a single
scalar — a single scalar makes AUROC/AUPRC invariant to training data because any
monotonic 1-D classifier preserves test-set ranking regardless of what trained it; this
was found and fixed during Phase 3 development). The head is trained from real ASD
support subjects, optionally expanded with generated anomalies (see
[fewshot_protocol.md](fewshot_protocol.md), [/docs/phase3_report.md](../docs/phase3_report.md)).

# Evaluation

Precision/Recall/F1/AUROC/AUPRC, computed once per (shot count x condition x trial) on
the same untouched test set (`/data/processed/metadata/phase1_subject_pools.json`,
`test_all`).
