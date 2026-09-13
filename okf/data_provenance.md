---
type: Provenance
title: Data Provenance - Real vs. Generated, Train/Val/Test
description: Explicit accounting of which data is real ABIDE, which is synthetically generated, and which split it may touch.
tags: [provenance, reproducibility, no-leakage]
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Real data

All subgraphs, embeddings, and anomaly scores derive from real ABIDE I subjects
(`/data/processed/graphs`) unless explicitly labeled `is_real: false` (Phase 2
generator output). Real ASD "abnormal" examples are ordinary ABIDE subjects labeled
`DX_GROUP=ASD`, never separately validated as anomalous transitions by clinicians —
subject diagnosis is used as a weak proxy label (documented limitation, see
[limitations.md](limitations.md)).

# Generated data

Every record produced by `/src/generation/structure_aware_generator.py` carries
`"is_real": false`, `"method"` (`structure_aware` or `random`), `"severity"`,
provenance (`subject_id`, `seed_roi`, `window_transition`), and both pre/post edge
weights for every touched edge — never presented as a real ABIDE subject. Generated
anomalies are grounded in real, normal (train-control) base graphs, never in random or
val/test graphs.

# Train / Val / Test provenance

| Split | Used for | Never used for |
|---|---|---|
| train-control (328) | GIN+GRU training (Phase 1), negative examples (Phase 3), generation base graphs (Phase 2/3) | threshold calibration, test evaluation |
| train-ASD pool (281) | few-shot support sampling (Phase 1E/3) | GIN+GRU training, generation base graphs, test evaluation |
| val-control (70) | threshold calibration (`tau_transition`, `tau_subject`) | training, test evaluation |
| test (131: 70 control + 61 ASD) | **final evaluation only** (Phase 1/3) | training, threshold calibration, generation, support-set sampling |

Isolation is asserted in code, not just documented: every few-shot/generation call path
runs `assert_subject_isolation(candidate_ids, val_control + test_all)` before use
(`/src/fewshot/support.py`), and `/tests/test_fewshot.py` covers both the raise and
pass cases.
