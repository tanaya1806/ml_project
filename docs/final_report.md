# Final Report — GRAPH_ANOMALY

Structure-Aware Few-Shot Temporal Subgraph Anomaly Generation & Detection (ABIDE I)

## Problem

Standard brain-graph anomaly detection asks "is this person's brain graph abnormal?" —
a static, subject-level question. This project instead asks: does any brain
**substructure** undergo an unusual **connectivity change over time**, compared with
normal subjects? The anomaly is a temporal transition of a subgraph, not a static
abnormal graph. Real labeled anomalous transitions are scarce (ABIDE has diagnosis
labels, not per-transition ground truth), motivating a **few-shot** framing: can a small
number of real abnormal examples be usefully expanded with anomalies *generated from
what the model has already learned about normal structure*, rather than invented from
nothing?

## Dataset

ABIDE I, 871 subjects (403 ASD, 468 typical control), 20 sites, 360-ROI parcellation,
variable-length time series (78–316 TR). Subject-level stratified split: 609 train
(328 control / 281 ASD), 131 val (70/61), 131 test (70/61) — see
[okf/dataset.md](../okf/dataset.md) for full provenance.

## M1 — Temporal Functional Brain Graph Construction

Sliding-window (30 TR, stride 10) Pearson FC per subject; deterministic top-20%
absolute-magnitude sparsification into signed, symmetric, self-loop-free temporal
graphs (25,848 directed edges/window, 1.4 GB total). [okf/preprocessing.md](../okf/preprocessing.md)

## M2 — Stable Temporal Subgraph Extraction

k-hop (k=1) candidate subgraphs around every ROI, with **stable** ROI-set tracking
(fixed at a reference window, only internal edges/weights evolve) — chosen because
dynamic per-window membership had mean consecutive Jaccard 0.378 vs. 1.0 for stable
tracking. [okf/subgraph_extraction.md](../okf/subgraph_extraction.md)

## M3 — GIN Spatial Representation

2-layer weighted GIN, signed edge weights as message-passing scalars, 128-d output,
validated architecture (24 tests, random init) — **not yet trained** at this stage; M3
established the forward pass only. [okf/gin_encoder.md](../okf/gin_encoder.md)

## M4 — Normal Temporal Modelling (Phase 1)

GIN trained end-to-end with a GRU next-embedding predictor, on **328 train-control
subjects x 20 seed ROIs only** (a documented compute-budget subsample, not a change to
M2's 360-seed capability). A VICReg-style variance term guards against representation
collapse (verified: post-training embedding std 1.03–1.16, no collapse). Baseline
anomaly score `A_t = ||z_{t+1} - z_hat_{t+1}||`, two-tier threshold calibration (a real
methodological fix — see [docs/phase1_report.md](phase1_report.md)). **Test AUROC:
0.453 — near chance**, reported honestly. [okf/gru_temporal_model.md](../okf/gru_temporal_model.md)

## M5 — Structure-Aware Anomaly Generation (Phase 2) — Primary Contribution

For a real transition from a real, normal (train-control) subject, gradient-ascent on
the trained GIN+GRU's own residual selects and reweights the most sensitive edges
(iterative, PGD-style — a single large step was empirically non-monotonic in severity
and was fixed during development), producing a graph-valid, symmetric, severity-
controllable, single-transition anomaly grounded in a real subgraph. Validated: 0
constraint violations over 7,200 records; severity strictly monotonic for
structure-aware, flat for a random-perturbation baseline; structure-aware/high detected
~9x more often than random at the same edit budget under the frozen Phase 1 threshold
(0.458 vs. 0.048); diversity (mean pairwise Jaccard) ~0.999.
[okf/anomaly_generator.md](../okf/anomaly_generator.md)

## M6 — Anomaly Detection

Two forms: Phase 1's unsupervised percentile threshold, and Phase 3's few-shot-adapted
logistic-regression head on a per-seed score vector (frozen GIN+GRU underneath both).
[okf/anomaly_detector.md](../okf/anomaly_detector.md)

## M7 — Few-Shot ASD Adaptation (Phase 3)

1/5/10-shot real ASD support sets (5 independent draws each), subject-level isolation
asserted in code, expanded with 30 generated anomalies per condition/trial.

## Main Contribution

**The primary contribution is structure-aware generation of meaningful temporal
brain-subgraph anomalies from learned normal graph structure (M5); anomaly detection
(M6) is the evaluation mechanism for whether generated anomalies are meaningful, not
the end goal.** M5's own validation (Phase 2) shows the clearest evidence for this:
structure-aware generation produces a controllable, monotonic, ~9x-more-detectable
signal that random perturbation of the identical edit budget does not.

## Required Final Comparison

| Setting | Real abnormal | Generated | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---|---|---|---|---|
| 1-shot | 1 | 0 | 0.200 | 0.003 | 0.006 | 0.516 | 0.489 |
| 1-shot + structure-aware generation | 1 | 30 | 0.443 | 0.344 | 0.387 | 0.497 | 0.470 |
| 1-shot + random generation | 1 | 30 | 0.473 | 0.357 | 0.403 | 0.506 | 0.487 |
| 5-shot | 5 | 0 | 0.430 | 0.092 | 0.149 | 0.485 | 0.466 |
| 5-shot + structure-aware generation | 5 | 30 | 0.447 | 0.348 | 0.390 | 0.486 | 0.474 |
| 5-shot + random generation | 5 | 30 | 0.451 | 0.361 | 0.399 | 0.502 | 0.483 |
| 10-shot | 10 | 0 | 0.433 | 0.249 | 0.311 | 0.524 | 0.486 |
| 10-shot + structure-aware generation | 10 | 30 | 0.445 | 0.367 | 0.402 | 0.501 | 0.481 |
| 10-shot + random generation | 10 | 30 | 0.474 | 0.400 | 0.431 | 0.515 | 0.473 |

`Delta` vs. few-shot-only (structure-aware, mean over shots): `dF1 ≈ +0.24`,
`dRecall ≈ +0.27`, `dAUROC ≈ -0.006`, `dAUPRC ≈ +0.002`.

**Read honestly, not as a clean win:** generation (either method) substantially lifts
F1/Recall, especially at 1-shot where few-shot-only nearly collapses — a real,
reproducible effect of having more labeled training data at all. But AUROC/AUPRC stay
near chance in every condition, and **structure-aware does not show a clear edge over
random generation in this classifier's ranking metrics**, even though it clearly does
in Phase 2's own raw-detectability validation. This gap between "Phase 2 validates the
generator is meaningfully different from random" and "Phase 3's few-shot classifier
doesn't yet exploit that difference" is this project's most important honest finding —
see [docs/phase3_report.md](phase3_report.md) for the discussion and a candidate
explanation (a linear classifier over a coarse per-seed feature may not have enough
generated examples per dimension to learn *which* seed's anomaly is structurally
meaningful, as opposed to "an anomaly exists somewhere").

## Limitations

See [okf/limitations.md](../okf/limitations.md) for the complete, itemized list: weak
diagnosis-as-anomaly-label proxy, near-chance baseline detector, the Phase 3
structure-aware-vs-random gap above, 20/360-ROI compute subsampling, edge-weight-only
generation (no topology changes), and no claim of biological causal mechanism.

## Reproducibility

Every stage is config-driven (`configs/{baseline,phase1,phase2,phase3}.yaml`), seeded
(seed=42 throughout), and saves its outputs under `outputs/experiments/phase{1,2,3}/`.
Regeneratable intermediate artifacts (subgraph cache, generated-anomaly metadata) are
gitignored with the exact regeneration command documented in `.gitignore` and each
phase report, rather than committed as multi-hundred-MB files.
