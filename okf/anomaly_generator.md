---
type: Model
title: Phase 2 - Structure-Aware Anomaly Generator
description: Gradient-guided edge-weight perturbation of real normal subgraph transitions, grounded in the trained GIN+GRU's own residual objective. Primary project contribution.
resource: /src/generation/structure_aware_generator.py
tags: [Phase2, anomaly-generation, USP]
runtime: python (pytorch, MPS/CPU)
parameters:
  - { name: severity, type: float, required: true }
  - { name: method, type: string, required: true }
  - { name: edge_perturb_max_frac, type: float, required: false }
  - { name: step_size_max, type: float, required: false }
computation: /src/generation/generate_phase2_dataset.py
executor: { resource: /configs/phase2.yaml }
attester: { resource: /src/evaluate/validate_generator.py }
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Method

For a real transition `(G_t, G_{t+1})` from a real, **normal** (train-control) subject:
compute `d(||GIN(G_{t+1};w) - z_hat_{t+1}||^2)/dw` for every edge weight `w`; select the
top-m most sensitive undirected edge pairs (m scales with severity `alpha in [0,1]`);
take 5 small recomputed gradient-ascent steps on just those edges (PGD-style), applied
symmetrically, clipped to `[-1, 1]`. Only edge weights change — node/edge set,
symmetry, and single-transition locality are preserved by construction. The
`method="random"` baseline uses the identical pipeline with random edge selection and
random sign, for the required structure-aware-vs-random comparison.

# Why PGD-style, not one FGSM-style step

A single large step was empirically NOT monotonic in severity (GIN's ReLU
nonlinearities break the local-gradient approximation at large step sizes) — found by
running the generator on real data, documented in
[/docs/phase2_report.md](../docs/phase2_report.md).

# Validated Properties

0 constraint violations across 7,200 generated records; severity strictly monotonic for
structure-aware (not for random); mean pairwise Jaccard diversity of touched-edge sets
~0.999; structure-aware/high-severity detected ~9x more often than random at the same
edit budget under the frozen Phase 1 detector (0.458 vs 0.048).
