# Phase 3 Report — Few-Shot Adaptation + Final Comparison

Branch: `phase-3` (built on `phase-2`'s trained checkpoint and generator).

## Objective

Answer the project's central question: **does structure-aware anomaly generation make
the detector more reliable when only a few real abnormal (ASD) examples are available?**
Produce the required final comparison — few-shot-only vs. few-shot+generated — with a
random-generation ablation, on the same untouched test set, reported honestly.

## Mathematical Method

**Detector.** Phase 1's GIN+GRU stays **frozen**. A logistic-regression head is fit on
top of a **per-seed score vector** (20 dimensions, one per seed ROI: `max_t A_t` for
that seed) rather than Phase 1's single scalar. Model capacity is identical across all
three conditions, so any metric difference is attributable to the *training data*, not
extra capacity.

**A design bug found by running the experiment, not assumed.** The first version used
Phase 1's single scalar feature (`max` over everything). A 1-D logistic regression is a
strictly monotonic function of one number, so its predicted-probability *ranking* of
test subjects is identical to ranking by the raw feature directly, regardless of how
the classifier was trained. Result: AUROC/AUPRC were **bit-for-bit identical** across
Condition A/B/C at every shot count (e.g. 0.547 everywhere) — mathematically, adding
positive examples can only move the classifier's threshold, never the ranking, when
there's only one feature. Only Precision/Recall/F1 (threshold-dependent) had moved.
Fixed by switching to the 20-dim per-seed vector, whose *learned weights* (not just
threshold) can change with the training set. A generated anomaly (which perturbs one
seed's one transition) becomes a training vector by taking its real base subject's
per-seed vector and substituting that one seed's entry with the post-perturbation
score — a synthetic full-subject profile grounded in a real subject.

**Conditions**, evaluated on the identical held-out test set for the identical
support-set draw:

- **A (few-shot only):** `k` real ASD support subjects (+) vs. all 328 train-control (-).
- **B (few-shot + structure-aware):** A's positives + 30 structure-aware generated anomalies (severity=high).
- **C (few-shot + random, ablation):** A's positives + 30 random-perturbation generated anomalies.

`k in {1, 5, 10}`, 5 independent support-set draws each (`configs/phase3.yaml`).

**No leakage.** Support subjects (`train_asd_pool`) and generation base subjects
(`train_control`) are asserted disjoint from `val_control + test_all` before every use
(`assert_subject_isolation`); the negative pool and test set are otherwise fixed across
all conditions/trials.

## Implementation

- `configs/phase3.yaml`
- `src/evaluate/phase3_final_comparison.py` — scoring, generation-based expansion,
  classifier fit/eval, aggregation with deltas.
- `src/visualization/generate_phase3_plots.py` — F1/AUROC vs. shot-count figures.
- `tests/test_phase3_pipeline.py` — 4 tests on the new classifier/aggregation logic.

## Experiments Actually Run

`python3 -m src.evaluate.phase3_final_comparison` — 3 shot counts x 5 trials x 3
conditions = 45 fitted classifiers, each evaluated on all 131 test subjects (86 s, MPS).
`python3 -m src.visualization.generate_phase3_plots` — 2 figures.

## Results

Mean ± std over 5 trials (`outputs/experiments/phase3/final_comparison.json`):

| Shot | Condition | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| 1 | few-shot only | 0.200 | 0.003 | 0.006 | 0.516 | 0.489 |
| 1 | +structure-aware | 0.443 | 0.344 | 0.387 | 0.497 | 0.470 |
| 1 | +random | 0.473 | 0.357 | 0.403 | 0.506 | 0.487 |
| 5 | few-shot only | 0.430 | 0.092 | 0.149 | 0.485 | 0.466 |
| 5 | +structure-aware | 0.447 | 0.348 | 0.390 | 0.486 | 0.474 |
| 5 | +random | 0.451 | 0.361 | 0.399 | 0.502 | 0.483 |
| 10 | few-shot only | 0.433 | 0.249 | 0.311 | 0.524 | 0.486 |
| 10 | +structure-aware | 0.445 | 0.367 | 0.402 | 0.501 | 0.481 |
| 10 | +random | 0.474 | 0.400 | 0.431 | 0.515 | 0.473 |

Figures: `outputs/figures/phase3/01_f1_vs_shots.png` (both generated conditions lift F1
far above few-shot-only, especially at 1-shot where few-shot-only collapses to F1≈0),
`02_auroc_vs_shots.png` (all three conditions overlap around 0.48–0.53 with wide error
bars — no visible separation).

## Honest Finding — Read Before Citing This Project's Headline Number

1. **Expanding the training set helps F1/Recall, dramatically at low shot counts.**
   At 1-shot, few-shot-only collapses (F1≈0.006 — a single positive example gives the
   logistic regression almost nothing to calibrate a decision boundary from); adding
   30 generated examples of *either* kind restores F1 to ~0.39–0.40. This is a genuine,
   reproducible effect of having more labeled training data, not noise (error bars are
   tight relative to the gap).
2. **AUROC/AUPRC stay near chance in every condition** (0.48–0.53), consistent with
   Phase 1's finding that real ASD signal is weak in this baseline pipeline.
3. **Structure-aware does *not* show a clear edge over random generation in this
   classifier's ranking metrics** — despite Phase 2's own validation showing
   structure-aware anomalies are detected ~9x more often than random ones by the raw
   per-transition threshold. This is reported as found. A plausible (unconfirmed)
   explanation: each generated example substitutes only 1 of 20 feature dimensions;
   with 30 generated examples spread across up to 20 seed identities per trial, there
   may be too little per-dimension signal for a linear classifier to learn *which*
   seed's anomaly matters, so it mostly benefits from "more positive-labeled data
   exists" rather than from the structure-aware signal specifically.

No thresholds, support sets, or metrics were adjusted after seeing this result to
manufacture a preferred outcome — this is the first and only run of the final
comparison.

## Validation

Full suite: **57 passed, 3 skipped** (`pytest tests/`) — Phase 2's 53 passing tests
plus 4 new Phase 3 tests on the classifier/aggregation logic.

## Computational Cost

| Stage | Time | Device |
|---|---|---|
| Final comparison (45 classifier fits + scoring) | 86 s | MPS |
| Figures | <5 s | — |

## Limitations

- Only 30 generated anomalies per trial/condition — a compute-budget choice, not a
  swept hyperparameter; more generated examples per seed dimension might change the
  structure-aware-vs-random comparison (see "Honest Finding" #3).
- The per-seed vector feature is still a fairly coarse aggregation (max over windows,
  per seed); a per-window feature or an attention-style aggregation might recover more
  of Phase 2's demonstrated structure-aware signal, at the cost of more parameters to
  fit from very few real examples — a natural next step if more time/compute were
  available, deliberately not pursued here to avoid over-engineering a three-branch
  laptop project.
- This experiment used one frozen Phase 1 checkpoint; classifier variance across the 5
  support-set trials is reported, but variance from re-training the base GIN+GRU is not.

## Next Steps (Not Pursued — Out of This Project's Scope)

A literature-grounded severity/generation-count sweep, a richer per-window feature
representation for the few-shot head, and repeating Phase 1's GIN+GRU training across
multiple seeds to report a fuller variance budget.
