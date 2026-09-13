# Phase 1 Report — Normal Temporal Representation + Baseline Detection

Branch: `phase-1` (built on top of the completed M1/M2/M3 work on `main`).

## Objective

Build the complete baseline temporal anomaly pipeline described in the master brief:

```
M1 graphs -> M2 stable subgraphs -> M3 GIN -> temporal embeddings -> GRU
  -> next-state prediction -> anomaly score -> few-shot baseline evaluation
```

This is the **"before" state**: an unsupervised detector trained only on the normal
(control) temporal dynamics of stable brain subgraphs, with no structure-aware
generation involved yet. Phase 2 must improve on whatever this establishes.

## Mathematical Method

**Seed ROI subsample.** M2's candidate generator supports all 360 ROIs as seeds. Training
and evaluating on all 360 x all subjects x all windows is not tractable on a laptop (the
M2 pilot measured ~71s/subject to extract 360 seeds across all windows). We use a fixed,
deterministic, stratified subsample of **20 seed ROIs** (`configs/phase1.yaml`,
`select_seed_rois`, seed=42), cached once to disk per subject
(`src/data/subgraph_cache.py`) instead of re-extracted every epoch. This is a compute
budget decision, not a change to M2's capability — measured cache-build cost for all 871
subjects x 20 seeds was **112 s** (not 71 s/subject; the 360->20 seed reduction plus
reusing the already-sparsified parent graph dominates the difference from the pilot's
per-graph profiling overhead).

**Spatial embedding.** For seed `v`, window `t`: `z_t = GIN(G_t[S_v^ref])`, using the
unmodified M3 `GINEncoder` (2 layers, hidden=64, output=128), the *same* weights across
all subjects/seeds/windows.

**Temporal model (Phase 1C).** A single-layer GRU (hidden=128) reads the embedding
sequence and predicts the next step:

```
h_t = GRU(z_t, h_{t-1})
z_hat_{t+1} = Linear(h_t)
```

**Training strategy (Phase 1A).** The GIN is *not* pretrained with a separate pretext
task. It is trained jointly with the GRU using the next-embedding prediction loss itself
as the only objective — the simplest strategy that still moves GIN away from its random
initialization, avoiding a second self-supervised task (contrastive/GAN/etc.) per the
brief's "avoid unnecessary complexity" instruction.

*Collapse risk.* Because the prediction target `z_{t+1}` comes from the same GIN being
trained, the pure prediction loss has a trivial global optimum (GIN collapses every
subgraph to one constant vector; GRU learns identity). We add one VICReg-style variance
hinge term per mini-batch (16 pooled `(subject, seed)` sequences):

```
L = MSE(z_hat, z_target) + lambda * mean_d( relu(1 - std_d(z)) ),  lambda = 0.5
```

Post-training check: mean per-dimension embedding std = **1.159**, min = **1.028** across
a 200-sequence validation sample — no collapse.

**Anomaly score (Phase 1D).** `A_t = || z_{t+1} - z_hat_{t+1} ||_2`.

**Two-tier thresholding — a methodological fix found during evaluation.** Calibrating a
single per-transition `tau = percentile_95(A_t)` on pooled validation-control transitions,
then flagging a subject anomalous if *any* of its ~280 transitions (20 seeds x ~14 windows)
exceeded `tau`, is a multiple-comparisons error: under the null distribution itself, ~5% of
~280 transitions are *expected* to exceed a per-transition 95th-percentile cutoff, so
nearly every subject trips it regardless of label. First run: 131/131 test subjects flagged,
AUROC 0.453. Fix: calibrate a **second** threshold, `tau_subject`, directly on the
val-control distribution of the *subject-level* statistic (max `A_t` over seeds/windows)
that is actually used for classification. `tau_transition` is kept only for localization
(which window/seed) and few-shot recurrence flags; `tau_subject` is used for the
classification metrics below.

**Few-shot support (Phase 1E).** 1/5/10-shot support sets sampled without replacement from
the **train-ASD pool only** (`src/fewshot/support.py`), 5 independent draws per shot count,
with an explicit isolation assertion against validation/test subject IDs.
`C(S) = (# support subjects with >=1 flagged transition in seed S) / (# support subjects)`.

## Implementation

New files (main/M1-M3 files untouched except two portability fixes, see below):

- `configs/phase1.yaml` — all Phase 1 hyperparameters and rationale comments.
- `src/data/subgraph_cache.py`, `src/data/build_phase1_cache.py` — seed selection + cache.
- `src/models/embedding_sequence.py` — shared-GIN batched embedding sequence construction.
- `src/models/gru_temporal.py` — the GRU.
- `src/training/train_phase1.py` — joint GIN+GRU training loop with collapse guard.
- `src/anomaly/scoring.py` — `A_t`, threshold calibration, flagging.
- `src/fewshot/support.py` — support-set sampling, isolation check, recurrence.
- `src/evaluate/detect_baseline.py` — full evaluation + localization + few-shot pipeline.
- `src/visualization/generate_phase1_plots.py` — training curves + score distributions.
- `src/utils/device.py` — MPS-first device selection, no hard-coded CUDA.
- Tests: `test_subgraph_cache.py`, `test_embedding_sequence.py`, `test_gru_temporal.py`,
  `test_anomaly_scoring.py`, `test_fewshot.py`.

Preparatory fix (first commit on this branch): `tests/test_pipeline.py`,
`tests/test_subgraphs.py`, `src/graphs/pilot_experiment.py`, `src/models/pilot_gin.py` had
a hardcoded Windows dev path (`D:\Graph anomaly detection\...`); repointed at
`configs/baseline.yaml` with `pytest.skip` when raw data isn't present locally. Also
stopped tracking `__pycache__/` (already gitignored, but committed in the initial commit).

## Experiments Actually Run

1. `python3 -m src.data.build_phase1_cache` — cached all 871 subjects x 20 seeds (112 s).
2. `python3 -m src.training.train_phase1` — 5 epochs, 328 train-control subjects x 20 seeds
   = 6,560 (subject, seed) sequences/epoch, batch size 16, Adam lr=1e-3, MPS device.
3. `python3 -m src.evaluate.detect_baseline` — threshold calibration on 70 val-control
   subjects, evaluation on all 131 held-out test subjects (70 control + 61 ASD, untouched
   until this step), few-shot recurrence over the 281-subject train-ASD pool.
4. `python3 -m src.visualization.generate_phase1_plots` — training curves + score
   distributions.

## Results

Training (`outputs/experiments/phase1/training_report.json`):

| Epoch | train pred loss | val pred loss | var penalty |
|---|---|---|---|
| 1 | 0.1337 | 0.1600 | 0.307 |
| 2 | 0.1357 | 0.1365 | 0.089 |
| 3 | 0.1237 | 0.1407 | 0.075 |
| 4 | 0.1204 | **0.1133** (best) | 0.070 |
| 5 | 0.1162 | 0.1206 | 0.068 |

Total training time: 299 s on MPS (Apple M-series GPU).

Baseline detection on the held-out test set (`outputs/experiments/phase1/metrics.json`),
`tau_subject = 19.81` (p95 of 70 val-control subject-max scores):

| Metric | Value |
|---|---|
| Precision | 0.500 |
| Recall | 0.033 |
| F1 | 0.062 |
| AUROC | 0.453 |
| AUPRC | 0.467 |
| TP / FP / TN / FN | 2 / 2 / 68 / 59 |
| Subjects flagged anomalous | 4 / 131 |
| Anomalous transitions localized | 1,890 |

Few-shot subgraph recurrence among ASD support subjects (mean +/- std over 5 draws):

| Shot | Mean recurrence |
|---|---|
| 1-shot | 0.550 +/- 0.207 |
| 5-shot | 0.512 +/- 0.105 |
| 10-shot | 0.459 +/- 0.040 |

Figures: `outputs/figures/phase1/01_training_curves.png`,
`outputs/figures/phase1/02_anomaly_score_distributions.png` (the val-control, test-control,
and test-ASD score distributions visually overlap almost completely).

## Validation

- Full existing M1/M2/M3 test suite still passes after the portability fixes: **21
  passed, 3 skipped** (skips are raw-data tests; `data/raw/` is gitignored and never
  shipped, not a regression).
- New Phase 1 tests: **22 passed** covering seed-selection determinism, cache
  build/roundtrip, shared-GIN embedding-sequence correctness (batched output matches
  per-window forward calls), GRU causality (no future leakage), anomaly-score math,
  threshold calibration, and few-shot isolation/reproducibility.
- Full suite: **43 passed, 3 skipped** (`pytest tests/`).
- Post-training embedding-std check confirms the GIN did not collapse (min std 1.028).

## Computational Cost

| Stage | Time | Device |
|---|---|---|
| Seed-subgraph cache build (871 subjects x 20 seeds) | 112 s | CPU |
| Training (5 epochs) | 299 s | MPS |
| Evaluation + few-shot | 27 s | MPS |
| Cache disk footprint | 1.5 GB | — |

All comfortably within a single MacBook Air M-series session; no CUDA assumptions
anywhere (`src/utils/device.py` is MPS-first with a CPU fallback).

## Limitations

- **AUROC 0.453 — near chance.** The baseline detector does *not* meaningfully
  discriminate ASD from control at the subject level. This is reported honestly rather
  than tuned away, per the brief's explicit instruction not to force a preferred result.
  Plausible causes: only 5 training epochs on a simple architecture; a single
  loudest-transition (`max`) subject-level statistic is high-variance; ABIDE has no
  ground-truth per-transition anomaly labels, so subject diagnosis is only a weak,
  noisy proxy for "shows an unusual temporal connectivity change." This is the explicit
  motivation for Phase 2/3: a stronger, structure-aware signal (generated anomalies used
  to shape the detector) is what should move this number.
- Only 20 of 360 ROIs are used as seeds, and only 1 reference window (`t=0`) anchors the
  stable ROI set, both compute-budget decisions inherited from M2's own tracking design.
- Few-shot recurrence (Phase 1E) is reported as a localization/diagnostic statistic, not
  yet used to adapt the detector itself — that comparison is Phase 3's job.
- Subject-level diagnosis as an anomaly-transition proxy is a real limitation of what
  can be validated on ABIDE without per-transition ground truth; this is stated as a
  scientific caveat, not hidden.

## Next Phase

Phase 2 must not restart from scratch: it reuses this checkpoint
(`outputs/experiments/phase1/checkpoints/best.pt`), the same seed-subgraph cache, and the
same subject splits. It adds the structure-aware anomaly generator that perturbs real
normal subgraph trajectories using what the trained GIN+GRU learned (embedding-space
geometry, prediction residual directions), producing graph-level anomalies with
controllable severity — and validates them against this Phase 1 baseline detector plus a
random-perturbation baseline.
