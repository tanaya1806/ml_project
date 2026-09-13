---
type: Documentation
title: Limitations
description: Honest scope and validity caveats across all three phases.
tags: [limitations, honesty]
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Weak label proxy

ABIDE has no ground-truth per-transition anomaly labels. Subject diagnosis (ASD vs.
control) is used as a weak proxy for "shows an unusual temporal connectivity change."
This affects every detection metric reported in this project.

# Baseline detector is near chance

Phase 1's unsupervised detector scores AUROC 0.453 on the held-out test set — no
better than chance. This is reported as found, not tuned away.

# Structure-aware advantage does not fully transfer to the Phase 3 classifier

Phase 2's raw per-transition threshold shows structure-aware anomalies detected ~9x
more often than random ones. Phase 3's few-shot logistic-regression comparison does
**not** show a clear structure-aware-vs-random edge in AUROC/AUPRC (both near chance,
overlapping error bars across shot counts), though both meaningfully improve F1/Recall
over few-shot-only at low shot counts. A plausible explanation (not confirmed): each
generated example only informs one of 20 feature dimensions, diluting the
structure-aware signal for a linear classifier trained on few tens of examples spread
across those dimensions. Reported honestly rather than adjusted to show a preferred
result.

# Compute-budget subsampling

20 of 360 ROIs are used as seeds throughout Phase 1-3 (a fixed, deterministic,
stratified subsample) — a compute-budget decision, not a change to M2's own
360-seed capability. Documented in `/configs/phase1.yaml`.

# Generator scope

The generator edits edge *weights* only (no added/removed edges or ROIs) — anomalies
here are connectivity-strength anomalies, not topology-changing ones.

# No claim of biological mechanism

This project reports which functional subgraphs show unusual temporal connectivity
changes; it does not claim to establish a causal biological mechanism for ASD.
