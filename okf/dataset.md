---
type: Dataset
title: ABIDE I (processed)
description: 871-subject ABIDE I resting-state fMRI ROI time series, 360-ROI parcellation, 20 sites, subject-level train/val/test splits.
resource: /data/processed
tags: [ABIDE, fMRI, dataset]
sources:
  - resource: /data/raw/abide/Phenotypic_V1_0b_preprocessed1.csv
    title: ABIDE I phenotypic file (not shipped in this repo; data/raw/ is gitignored)
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Summary

| | |
|---|---|
| Subjects | 871 |
| ASD | 403 |
| Typical controls | 468 |
| Sites | 20 |
| ROIs | 360 |
| Timepoints per subject | 78-316 (variable length) |

# Splits

Subject-level, stratified by diagnosis, seed=42 (`/src/data/dataset_splits.py`,
manifest at `/data/processed/metadata/subject_splits.json`):

| Split | Count | Controls | ASD |
|---|---|---|---|
| train | 609 | 328 | 281 |
| val | 131 | 70 | 61 |
| test | 131 | 70 | 61 |

No subject appears in more than one split (asserted in `create_subject_splits`).

# Provenance

Raw time series (`data/raw/Time_serie_per_site/*.npz`) are NOT committed to this
repository (`.gitignore`); only the processed, derived artifacts are tracked:
temporal graphs (`/data/processed/graphs`, 1.4 GB) and metadata
(`/data/processed/metadata`). See [data_provenance.md](data_provenance.md) for the full
real-vs-generated and train/val/test provenance chain.
