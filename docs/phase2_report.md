# Phase 2 Report — Structure-Aware Anomaly Generation

Branch: `phase-2` (built on top of `phase-1`'s trained GIN+GRU checkpoint).

## Objective

Implement the project's primary research contribution: generate meaningful temporal
brain-subgraph anomalies grounded in what Phase 1 learned about normal structure and
dynamics, rather than injecting arbitrary noise — and validate that these anomalies are
(a) constraint-valid, (b) diverse, (c) severity-controllable, (d) grounded in real
subgraphs, (e) temporally localized, and (f) detectably different from normal, while a
random-perturbation baseline is *not*.

## Research Question

Given a learned normal temporal brain-subgraph trajectory, how can we deliberately
create a plausible but abnormal connectivity transition?

## Mathematical Method

**What we inspected first (per the brief's instruction not to invent the generator
arbitrarily).** The Phase 1 checkpoint gives, for any real transition, a small residual
`A_real = ||GIN(G_{t+1}) - GRU-predicted z_hat_{t+1}||` (mean ≈ 0.11 in embedding-loss
terms; the anomaly-score version has median ≈ 5, p95 ≈ 7.4 — see
`outputs/experiments/phase1/metrics.json`). GIN is differentiable end to end with
respect to its edge-weight inputs. That gives a direct, model-grounded answer to "which
dimensions can normal behavior be deviated along": the edges whose weight change most
increases `A` are exactly the edges the model itself considers structurally load-bearing
for that transition being normal.

**Generator.** For a real transition `(G_t[S_v], G_{t+1}[S_v])` from a real,
**normal** (train-control) subject:

1. `z_hat_{t+1} = GRU(z_1, ..., z_t)` (frozen Phase 1 weights, causal — uses only the
   real history up to `t`, never the graph being perturbed).
2. `g = d(||GIN(G_{t+1}; w) - z_hat_{t+1}||^2) / dw` for every edge weight `w` of the
   real `G_{t+1}`.
3. Group edges into undirected pairs `(u,v)`/`(v,u)` (both directions carry the same
   physical FC value by M1's construction); combined sensitivity = sum of the two
   directed gradients. Select the top-`m` pairs by `|combined sensitivity|`, where
   `m = max(1, round(alpha * 0.15 * num_undirected_edges))`.
4. Take `K=5` small gradient-ascent steps (recomputing the gradient each step) on just
   those `m` pairs, total budget `alpha * 0.6`, clipped to `[-1, 1]`, applied identically
   to both directed copies (preserves `w_ij = w_ji`).

**Why iterative steps, not one big step (a real finding, not a design guess).** The
first implementation took one gradient step of size `alpha * step_size_max` directly.
Testing severity `{0.25, 0.55, 0.9}` on a held-out example showed anomaly score
`5.28 -> 8.03 -> 5.03` — **not monotonic**: a single large step overshoots the region
where the gradient's linear direction is valid (GIN's ReLU nonlinearities mean large
jumps can cross into a different local regime — the same overshoot FGSM is known to
exhibit at large epsilon). Switching to `K=5` recomputed small steps (PGD-style) fixed
this; re-tested on 5 subjects x 8 severities, structure-aware became strictly monotonic
increasing (`5.76 -> 6.04 -> 6.84 -> 7.54 -> 8.17 -> 8.67 -> 9.38 -> 9.76`) while the
random baseline stayed flat/noisy (`5.76 -> 5.75 -> 5.79 -> 5.67 -> 5.77 -> 5.40 -> 6.17
-> 6.03`), confirmed again at scale (see Results).

**Random baseline (section 32C).** Identical pipeline, except the `m` edges are chosen
uniformly at random and pushed by a random sign — no gradient involved. This is the
comparison the brief asks for: *is structure-aware generation actually better than
random generation?*

**Constraints satisfied by construction, not by a post-hoc filter:** node/edge set is
never changed (only weights of existing edges move), so ROI identities, graph structure,
and "no self-loops" are automatically preserved; symmetric edits keep `w_ij = w_ji`;
`clip(·, -1, 1)` and finite-gradient arithmetic rule out range violations and NaN/Inf.
Only one window (`t+1`) is edited per record — the rest of the subject's real sequence
is untouched (temporal plausibility, section 24).

## Implementation

- `configs/phase2.yaml` — generator/severity/dataset config.
- `src/generation/structure_aware_generator.py` — `generate_anomaly(...)`, both methods.
- `src/generation/generate_phase2_dataset.py` — builds the generated-anomaly set.
- `src/evaluate/validate_generator.py` — validity/diversity/severity/detectability report + figures.
- `tests/test_generator.py` — 10 tests (range, no self-loops, symmetry, determinism,
  differs-from-normal, zero-severity-no-op, stays-connected-to-source, severity
  monotonicity, invalid-input errors).

## Experiments Actually Run

1. `python3 -m src.generation.generate_phase2_dataset` — 60 deterministic train-control
   subjects x 20 seed ROIs x 1 random valid transition each x 3 severities
   (`low=0.25, medium=0.55, high=0.9`) x 2 methods = **7,200 generated-anomaly records**
   (79 s). Metadata only (provenance, scores, touched-edge list) is persisted, not full
   graph tensors — everything is deterministic given `(subject_id, seed, t, severity,
   method, rng_seed)`, so Phase 3 regenerates on demand instead of us storing a second
   multi-GB dataset (same rationale as Phase 1's subgraph cache).
2. `python3 -m src.evaluate.validate_generator` — validity/diversity/severity/
   detectability checks + 3 figures (16 s).

## Results

**Validity:** 0 constraint violations across 7,200 records.

**Severity control** (`outputs/experiments/phase2/generator_validation.json`):

| Method | low (α=0.25) | medium (α=0.55) | high (α=0.9) | Monotonic? |
|---|---|---|---|---|
| structure-aware | 4.02 ± 2.45 | 5.82 ± 2.69 | 7.63 ± 3.17 | **Yes** |
| random | 2.91 ± 2.35 | 2.88 ± 2.35 | 2.84 ± 2.31 | No |

**Detectability** — fraction of transitions exceeding the Phase 1 `tau_transition`
(7.435, calibrated on val-control, never on generated or ASD data):

| Source | Detection rate |
|---|---|
| real normal (val control) | 0.050 |
| real ASD (test) | 0.049 |
| generated, random, high severity | 0.048 |
| generated, structure-aware, low severity | 0.078 |
| generated, structure-aware, medium severity | 0.230 |
| generated, structure-aware, high severity | **0.458** |

Two things stand out, both reported as found (not adjusted to look better):

- Structure-aware generation at high severity is detected **~9x** more often than
  random perturbation of the *same edit budget* (0.458 vs 0.048) — the central Phase 2
  claim, and it did not need any threshold tuning to show up.
- Real ASD test transitions are detected at essentially the same rate as real normal
  transitions (0.049 vs 0.050) — consistent with Phase 1's near-chance AUROC. Real
  clinical anomalies are far subtler than what this baseline detector (or, it turns out,
  even severe synthetic perturbation) easily separates from normal at this threshold.
  This is exactly the gap Phase 3 investigates: does training with generated anomalies
  help the detector recognize this kind of subtle deviation, not just gross ones?

**Diversity:** mean pairwise Jaccard distance between touched-edge sets ≈ 0.999–1.000 in
every method/severity group — generated anomalies are essentially never near-duplicates
of each other (expected: different base subjects/seeds/transitions have different local
structure and different top-gradient edges).

Figures: `outputs/figures/phase2/01_severity_vs_score.png` (clear separating lines,
structure-aware rising, random flat, `tau_transition` marked),
`02_score_distributions.png` (real normal and real ASD distributions visually
overlap almost entirely; generated structure-aware/high is visibly right-shifted past
`tau_transition`), `03_edge_weight_changes.png` (edit-magnitude histograms for both
methods).

## Validation

- Full suite: **63 passed, 3 skipped** (`pytest tests/`) — Phase 1's 43/3 plus 10 new
  generator tests, including one that caught a real bug: `test_zero_severity_changes_
  nothing` failed on first run because `num_touch = max(1, ...)` forced at least one
  edge to be touched even at `severity=0`; fixed to floor at 0 when `severity == 0`.
- Severity monotonicity and the overshoot-at-large-step issue were each caught by
  actually running the generator on real data before writing the "final" version —
  not assumed from the design alone.

## Computational Cost

| Stage | Time | Device |
|---|---|---|
| Generate 7,200 anomaly records | 79 s | MPS |
| Validation report + figures | 16 s | MPS |

## Limitations

- The generator edits edge *weights* only; it does not add or remove ROIs/edges from
  the subgraph (a deliberate constraint per section 21, but it means "structural"
  anomalies here are connectivity-strength anomalies, not topology-changing ones).
- `K=5` ascent steps and the `0.15`/`0.6` severity-scaling constants are reasonable,
  documented choices, not the result of a hyperparameter search — a natural Phase 3+
  extension if time permits.
- Real ASD transitions are *not* well separated from normal by this baseline detector,
  so the "detectability" comparison currently contrasts structure-aware vs. random
  synthetic anomalies convincingly, but does not yet show synthetic anomalies resembling
  real ASD ones — Phase 3's few-shot comparison is the test of whether that gap matters
  for the thing we actually care about (detecting real ASD-associated transitions).

## Next Phase

Phase 3 uses this generator directly (not just its saved outputs) to expand the small
real few-shot ASD support sets from Phase 1E, and runs the required final comparison:
few-shot-only vs. few-shot+generated-anomalies vs. few-shot+random-perturbation, on the
same untouched test set, reporting Precision/Recall/F1/AUROC/AUPRC and their deltas.
