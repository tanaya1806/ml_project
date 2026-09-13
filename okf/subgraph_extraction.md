---
type: Pipeline
title: M2 - Stable Temporal Subgraph Tracking
description: k-hop candidate subgraph extraction with a fixed (stable) ROI set tracked across all temporal windows.
resource: /src/graphs
tags: [M2, subgraph-extraction]
parameters:
  - { name: k, type: int, required: true }
  - { name: ref_window_idx, type: int, required: true }
computation: /configs/baseline.yaml
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Method

For seed ROI `v`, the k-hop neighborhood `V_v^ref` is computed once from a reference
window (`t=0`) and held fixed across all subsequent windows
(`track_temporal_subgraph_stable`,
[/src/graphs/temporal_tracking.py](../src/graphs/temporal_tracking.py)); edge presence
and signed weight within that fixed ROI set still evolve per window. Chosen over
dynamic per-window k-hop membership because the M2 pilot measured mean consecutive
Jaccard similarity of 0.378 for dynamic membership vs. 1.0 for stable tracking
(`/data/processed/metadata/subgraph_pilot_report.json`).

`k=1`, all 360 ROIs are valid seeds (`seed_strategy: all_rois`); Phase 1+ downstream
work uses a fixed 20-ROI deterministic subsample for compute-budget reasons (see
[gru_temporal_model.md](gru_temporal_model.md)) without changing this module's capability.

# Validation

6 tests in `/tests/test_subgraphs.py`: k-hop bounds, induced-subgraph correctness, edge
weight correspondence to the parent graph, stable-vs-dynamic tracking, configurable k.
