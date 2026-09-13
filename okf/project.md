---
type: Project
title: GRAPH_ANOMALY
description: Structure-aware few-shot temporal subgraph anomaly generation and detection in functional brain networks (ABIDE I).
tags: [neuroimaging, graph-neural-networks, anomaly-generation, few-shot, ABIDE]
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Objective

Learn the structural and temporal organization of functional brain subnetworks, use
that learned structure to generate meaningful structure-aware temporal anomalies
(rather than random perturbations), and evaluate whether those generated anomalies can
be detected and whether they help few-shot ASD anomaly detection.

# Research Question

Instead of "is this person's brain graph abnormal?", ask: does any brain substructure
undergo an unusual connectivity change over time compared with normal subjects? And:
does expanding a small set of real abnormal examples with structure-aware generated
anomalies improve detection of real, subtle ASD-associated connectivity changes?

# Main Contribution

The primary contribution is **structure-aware generation of meaningful temporal
brain-subgraph anomalies from learned normal graph structure** (see
[anomaly_generator.md](anomaly_generator.md)); anomaly detection is the *validation*
mechanism for whether generated anomalies are meaningful, not the end goal.

Novelty framing (per project convention): this project investigates a structure-aware,
few-shot temporal subgraph anomaly generation framework — it does not claim to be the
first such method absent a dedicated literature review.

# Pipeline

```
ABIDE rs-fMRI -> ROI time series -> sliding-window FC -> temporal graph sequence
  -> stable k-hop candidate subgraphs -> GIN spatial embedding -> GRU temporal model
  -> structure-aware anomaly generation -> anomaly detector -> few-shot comparison
```

# Branches / Milestones

- `main`: M1 (temporal graphs) / M2 (stable subgraph tracking) / M3 (GIN encoder).
- `phase-1`: normal temporal representation + baseline detection.
- `phase-2`: structure-aware anomaly generation (this project's USP).
- `phase-3`: few-shot adaptation + final comparison.

# Hardware

Apple Silicon (MPS-first, CPU fallback) — see [src/utils/device.py](../src/utils/device.py).
No CUDA assumptions anywhere in the codebase.
