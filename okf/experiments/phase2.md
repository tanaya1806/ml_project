---
type: Experiment
title: Phase 2 - Structure-Aware Generator Validation
description: 7,200 generated anomaly records (structure-aware + random baseline, 3 severities) validated for constraint compliance, diversity, severity control, and detectability.
resource: /outputs/experiments/phase2
sources:
  - resource: /configs/phase2.yaml
  - resource: /docs/phase2_report.md
executor: { resource: /src/generation/generate_phase2_dataset.py }
attester: { resource: /src/evaluate/validate_generator.py }
status: stable
generated: { by: process:generate_phase2_dataset+validate_generator, at: 2026-09-14T00:00:00Z }
verified: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Setup

60 deterministic train-control subjects x 20 seed ROIs x 1 transition each x 3
severities (low=0.25, medium=0.55, high=0.9) x 2 methods (structure_aware, random).

# Results

| Method | low | medium | high | Monotonic? |
|---|---|---|---|---|
| structure-aware | 4.02 | 5.82 | 7.63 | Yes |
| random | 2.91 | 2.88 | 2.84 | No |

Detectability (fraction exceeding Phase 1's `tau_transition`): structure-aware/high =
0.458 vs. random/high = 0.048 vs. real normal = 0.050 vs. real ASD = 0.049.
0 constraint violations across 7,200 records. Full numbers:
`/outputs/experiments/phase2/generator_validation.json`.

# Tests

10 new tests in `/tests/test_generator.py` (63 passed / 3 skipped overall); one caught
a real bug (zero severity still touched one edge due to an unconditional floor).
