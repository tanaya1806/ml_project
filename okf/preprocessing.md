---
type: Pipeline
title: M1 - Temporal Functional Graph Construction
description: Sliding-window Pearson FC and deterministic top-20% sparsification into signed, symmetric temporal graphs.
resource: /src/data
tags: [M1, preprocessing, functional-connectivity]
parameters:
  - { name: window_size, type: int, required: true }
  - { name: stride, type: int, required: true }
  - { name: sparsity_ratio, type: float, required: true }
computation: /configs/baseline.yaml
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Method

For each subject: sliding windows (size=30 TR, stride=10 TR) over ROI time series,
Pearson correlation per window (`compute_pearson_fc`,
[/src/data/sliding_window.py](../src/data/sliding_window.py)), diagonal removed, clipped
to [-1, 1], symmetrized. Each window's 360x360 FC matrix is sparsified to the top 20% of
edges by absolute magnitude (`sparsify_fc_matrix`,
[/src/data/graph_construction.py](../src/data/graph_construction.py)), keeping the
**signed** Pearson r as edge weight and both directed copies (u,v)/(v,u) for symmetry.
Result: 25,848 directed edges per window.

# Output

One `.npz` per subject in `/data/processed/graphs` (1.4 GB total): `edge_indices`
`(K, 2, 25848)`, `edge_weights` `(K, 25848)`, plus subject metadata (site, label,
window boundaries, threshold values). `K` = number of windows, variable per subject.

# Validation

21 tests in `/tests/test_pipeline.py`: FC shape/symmetry/diagonal handling, sparsification
correctness (edge count, threshold, symmetry), variable-length subject handling.
