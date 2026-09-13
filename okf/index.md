---
okf_version: "0.2"
type: Index
title: GRAPH_ANOMALY Knowledge Bundle
description: Machine-readable OKF representation of the GRAPH_ANOMALY project (structure-aware few-shot temporal subgraph anomaly generation and detection on ABIDE I).
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# GRAPH_ANOMALY

This bundle follows the [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md):
a directory of markdown files with YAML frontmatter, no schema registry, no required
tooling. Every concept below is one file; `resource` frontmatter fields point at the
actual source files/configs/outputs in this repository (bundle-relative paths), so this
bundle documents the project rather than duplicating it.

## Concepts

* [Project](project.md) - objective, research question, main contribution
* [Dataset](dataset.md) - ABIDE I, provenance, train/val/test splits
* [Preprocessing pipeline](preprocessing.md) - M1: sliding-window FC, sparse temporal graphs
* [Subgraph extraction](subgraph_extraction.md) - M2: stable k-hop temporal subgraph tracking
* [GIN encoder](gin_encoder.md) - M3: spatial representation model
* [GRU temporal model](gru_temporal_model.md) - Phase 1: normal-dynamics model
* [Anomaly generator](anomaly_generator.md) - Phase 2: structure-aware generation (primary contribution)
* [Anomaly detector](anomaly_detector.md) - Phase 1/3: scoring + few-shot-adaptable classifier
* [Few-shot protocol](fewshot_protocol.md) - support-set sampling, isolation, recurrence
* [Data provenance](data_provenance.md) - real vs. generated, train/val/test distinctions
* [Limitations](limitations.md) - honest scope/validity caveats

## Experiments

* [Phase 1: baseline detection](experiments/phase1.md)
* [Phase 2: generator validation](experiments/phase2.md)
* [Phase 3: final comparison](experiments/phase3.md)

See [log.md](log.md) for the bundle's update history.
