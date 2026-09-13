---
type: Model
title: M3 - GIN Spatial Encoder
description: Weighted Graph Isomorphism Network producing 128-d subgraph embeddings from signed edge weights.
resource: /src/models/gin_encoder.py
tags: [M3, GIN, spatial-representation]
parameters:
  - { name: in_channels, type: int, required: true }
  - { name: hidden_channels, type: int, required: true }
  - { name: embedding_dim, type: int, required: true }
  - { name: num_layers, type: int, required: true }
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Architecture

2-layer weighted GIN (`WeightedGINConv`), hidden=64, input node feature = constant 1.0
(structure-only), ReLU after each layer, global mean pool, linear projection to 128-d.
Signed Pearson edge weights are used directly as message-passing scalars:
`h_v^(k) = MLP_k((1+eps) h_v^(k-1) + sum_u w_uv h_u^(k-1))`.

# Training

M3 alone only validates the architecture's forward pass (24 tests, random init, no
training). Phase 1 (`generated: { by: process:train_phase1, at: ... }` in
[gru_temporal_model.md](gru_temporal_model.md)) trains this encoder jointly with the
GRU — M3's own tests do not claim a biologically meaningful trained representation.

# Validation

9 tests in `/tests/test_gin_encoder.py`: output dimension, batching, variable node/edge
counts, numerical stability, shared-parameter consistency, permutation invariance, edge
weights affecting output.
