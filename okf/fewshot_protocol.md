---
type: Protocol
title: Few-Shot ASD Support Protocol
description: 1/5/10-shot support-set sampling from train-ASD subjects only, with explicit subject-level isolation and reproducibility.
resource: /src/fewshot/support.py
tags: [few-shot, protocol, no-leakage]
parameters:
  - { name: shot, type: int, required: true }
  - { name: rng_seed, type: int, required: true }
status: stable
generated: { by: claude/sonnet-5, at: 2026-09-14T00:00:00Z }
---

# Method

`sample_support_set(pool, shot, seed)` draws deterministically, without replacement,
from `train_asd_pool` (281 subjects) — never from validation or test subjects.
`assert_subject_isolation` is called before every use of a support set (Phase 1E and
Phase 3) and raises if any support/generation-base subject overlaps val/test IDs.

Phase 1E reports subgraph recurrence `C(S) = (# support subjects flagged anomalous in
S) / (# support subjects)` across 5 independent draws per shot count (mean +/- std).
Phase 3 uses the same sampling to build the real-positive-example pool for the
few-shot-adapted classifier, run for 5 independent trials per shot count.

# Validation

7 tests in `/tests/test_fewshot.py`: shot count, determinism, distinct-seeds differ,
oversized-shot error, isolation raises on overlap, isolation passes when disjoint,
recurrence computation correctness.
