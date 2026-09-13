---
type: Model
title: Phase 1 - GIN+GRU Normal Temporal Model
description: GRU next-embedding predictor trained jointly with the GIN encoder on control-subject subgraph trajectories only.
resource: /src/models/gru_temporal.py
tags: [Phase1, GRU, temporal-model]
runtime: python (pytorch, MPS/CPU)
parameters:
  - { name: hidden_dim, type: int, required: true }
  - { name: epochs, type: int, required: true }
  - { name: lambda_var, type: float, required: true }
computation: /src/training/train_phase1.py
executor: { resource: /configs/phase1.yaml }
status: stable
generated: { by: process:train_phase1, at: 2026-09-14T00:00:00Z }
verified: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Method

`z_t = GIN(G_t[S_v])`; `h_t = GRU(z_t, h_{t-1})`; `z_hat_{t+1} = Linear(h_t)`. GIN is
trained end-to-end via the GRU's own next-embedding prediction loss (no separate
pretext task), with a VICReg-style variance hinge term to prevent representation
collapse (both networks mapping every subgraph to one constant vector, which would
trivially minimize the raw prediction loss). Trained on 328 train-control subjects x 20
deterministic seed ROIs only, so learned dynamics reflect *normal* brain-network
behavior. See [/docs/phase1_report.md](../docs/phase1_report.md) for the full derivation.

# Checkpoint

`/outputs/experiments/phase1/checkpoints/best.pt` (epoch 4, val prediction loss 0.1133).
Post-training embedding std check: mean 1.159, min 1.028 (no collapse).

# Anomaly Score

`A_t = ||z_{t+1} - z_hat_{t+1}||_2`. Two calibrated thresholds (both from val-control
only, never test/ASD): `tau_transition` (per-transition, localization) and
`tau_subject` (on the subject-level max-over-seeds/windows statistic, classification —
needed to avoid a multiple-comparisons false-positive inflation found during evaluation).
