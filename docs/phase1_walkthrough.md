# Phase 1 Walkthrough — What It Does, How It Works, How to Check It Yourself

This is a plain-language companion to [`docs/phase1_report.md`](phase1_report.md) (the
dense, technical version). Read this one first if you want to understand what Phase 1
actually built before diving into equations.

> Note on where you are: your repo has three branches — `phase-1`, `phase-2`, `phase-3`
> — each building on the last. Phase 2 and Phase 3 are **already implemented and
> committed**, not future work; this file focuses on Phase 1 as requested, and points
> to the Phase 2 files at the end so you know what to look at next.

---

## 1. What Phase 1 Actually Does

Milestones 1–3 (already on `main` before this work started) turned raw brain scans
into small, trackable subgraphs and gave you an untrained neural network (GIN) that
*could* turn a subgraph into a 128-number summary. Phase 1's job was to answer three
questions M1–M3 left open:

1. **Can that GIN actually learn anything useful, or is it just random weights?**
   → Train it.
2. **What does "normal" brain-network behavior over time look like, mathematically?**
   → Learn it with a GRU, using only non-autistic (control) subjects.
3. **Given that learned notion of "normal," can you tell when a subject's brain
   subgraph is doing something unusual?**
   → Turn the GRU's prediction error into an anomaly score, calibrate a threshold, and
   test it on subjects the model never saw during training.

Phase 1 is deliberately a **baseline**: a simple, honest, unsupervised detector. It is
not expected to be great (and it isn't — see Section 4). Its job is to give Phase 2/3
something concrete to improve on.

## 2. How It Was Technically Achieved

### 2.1 The pipeline, end to end

```
real brain graphs (M1)
   -> stable subgraphs around 20 chosen ROIs (M2, cached once to disk)
   -> GIN turns each window's subgraph into a 128-number vector  z_t
   -> GRU reads z_1...z_t and predicts what z_{t+1} should be     z_hat_{t+1}
   -> anomaly score A_t = distance between the real z_{t+1} and the prediction
   -> threshold, calibrated only on healthy validation subjects
   -> flag test subjects whose score crosses that threshold
```

### 2.2 Why only 20 ROIs, not all 360?

M2 supports all 360 ROIs as candidate seeds, but training a model on all of them,
for every subject, every epoch, would take about half a day on a laptop. So Phase 1
picks a fixed, reproducible sample of 20 ROIs spread across the brain
(`src/data/subgraph_cache.py::select_seed_rois`, seed=42) and pre-computes their
subgraphs once (`src/data/build_phase1_cache.py`) instead of recomputing them every
training step. This is a speed decision, not a change to what M2 can do.

### 2.3 Training the GIN and GRU together

The GIN was *not* pre-trained separately. It was trained **at the same time** as the
GRU, using one simple rule: "get better at predicting the next window's embedding."
That's the whole training signal — no extra self-supervised trick, no contrastive
learning, nothing exotic.

One catch, and it's worth understanding because it's the main technical risk in Phase
1: if the only goal is "make predictions match," the laziest possible solution is for
the GIN to map *every* subgraph to the exact same number, and for the GRU to just copy
that number forward. Loss becomes ~0, and the model has learned nothing. To stop this,
training also includes a small penalty that punishes the model whenever its output
numbers stop varying across different subgraphs (`lambda_var` term in
`src/training/train_phase1.py`). After training, we checked this directly — the
model's outputs still vary a healthy amount (not collapsed) — see Section 3.

Training used only the 328 **control** (non-ASD) subjects in the training split. This
matters: "normal" is defined purely from healthy subjects, so anything the model later
flags as unusual is unusual *relative to healthy brains*, not relative to some
arbitrary average that includes ASD subjects.

### 2.4 Turning prediction error into a yes/no answer

The raw anomaly score is just a distance: `A_t = ||real z_{t+1} - predicted z_{t+1}||`.
The tricky part is deciding how big a distance counts as "anomalous."

We calibrate that cutoff using only the 70 **validation control** subjects — subjects
the model didn't train on, but who are still healthy, so we know what their scores
*should* look like if nothing unusual is happening. The cutoff is the 95th percentile
of their scores: by definition, only about 5% of a healthy subject's own transitions
should exceed it.

There's a subtlety here that we actually got wrong on the first attempt and then fixed
(documented in `docs/phase1_report.md`): each subject has about 280 transitions (20
ROIs × ~14 time windows). If you calibrate the cutoff on individual transitions and
then flag a subject as anomalous whenever *any single one* of its 280 transitions
crosses that cutoff, you will flag almost every subject — healthy or not — just by
chance, because out of 280 rolls of the dice, about 5%×280 ≈ 14 are expected to cross a
95th-percentile line even in a perfectly healthy person. The fix was to calibrate a
*second*, separate cutoff directly on the "worst transition per subject" statistic,
not on individual transitions. This is why the code and report both mention two
thresholds (`tau_transition` and `tau_subject`) instead of one.

## 3. How To Validate That Phase 1 Is Actually Working

You don't have to take the report's word for it. Here's exactly what to run and what
you should see. All of these are safe, read/compute-only commands.

### 3.1 Run the automated tests

```bash
python3 -m pytest tests/ -v
```

Expect **57 passed, 3 skipped** (the 3 skips are tests that need raw ABIDE scans,
which aren't shipped in this repo — that's expected, not a failure). Tests relevant to
Phase 1 specifically: `test_gru_temporal.py`, `test_anomaly_scoring.py`,
`test_fewshot.py`, `test_subgraph_cache.py`, `test_embedding_sequence.py`.

### 3.2 Check the model didn't collapse

```bash
python3 -c "import json; d=json.load(open('outputs/experiments/phase1/training_report.json')); print('embedding_std_min:', d['embedding_std_min'])"
```

You should see a number close to **1.0** (we measured 1.028). If this were near 0, it
would mean the GIN gave up and started outputting the same vector for every subgraph
— the collapse failure mode described in Section 2.3. It isn't near 0, so the model is
producing genuinely different embeddings for different subgraphs.

### 3.3 Look at the training curve

```bash
open outputs/figures/phase1/01_training_curves.png
```

(On a system without `open`, just view the PNG file directly.) Both the train and
validation prediction-loss lines should trend downward across the 5 epochs — evidence
the model is actually learning to predict, not just running in place.

### 3.4 Check the final detection numbers

```bash
python3 -c "import json; print(json.dumps(json.load(open('outputs/experiments/phase1/metrics.json')), indent=2))"
```

You'll see `AUROC ≈ 0.45`. **This is supposed to look unimpressive** — Phase 1 is an
honest, weak, unsupervised baseline (see Section 4). What you're actually checking
here is not "is the number high" but "does the pipeline run end-to-end and produce a
believable, non-broken number" (e.g. not NaN, not exactly 1.0 or 0.0, not obviously a
bug).

### 3.5 Look at the score-distribution figure

```bash
open outputs/figures/phase1/02_anomaly_score_distributions.png
```

You should see the val-control, test-control, and test-ASD curves almost entirely
overlapping. That overlap *is* the AUROC≈0.45 result, drawn as a picture — a quick way
to sanity-check that the number in the JSON file matches what the actual score
distributions look like, rather than being some downstream computation bug.

### 3.6 Reproduce everything from scratch (optional, ~7 minutes)

If you want to confirm none of this depends on a stale file, delete the derived cache
and regenerate everything:

```bash
rm -rf data/processed/subgraph_cache
python3 -m src.data.build_phase1_cache        # ~2 min: rebuilds the seed-subgraph cache
python3 -m src.training.train_phase1          # ~5 min: retrains GIN+GRU from scratch
python3 -m src.evaluate.detect_baseline        # ~30s: recalibrates thresholds, re-evaluates
python3 -m src.visualization.generate_phase1_plots
```

Because everything is seeded (`random_seed: 42` throughout
`configs/phase1.yaml`), you should get numbers very close to the ones already
committed (small differences are possible from non-determinism in GPU/MPS floating
point ops, but the overall picture — AUROC near chance, no collapse — should match).

## 4. Why AUROC ≈ 0.45 Is the Right Result, Not a Bug

It's worth internalizing this before moving to Phase 2, because Phase 2/3 are built
specifically to react to it:

- ABIDE has no ground truth for "this exact transition, at this exact time window, is
  abnormal." All we have is each subject's overall diagnosis (ASD or control), used as
  a weak stand-in.
- Phase 1's detector is fully unsupervised — it never sees a single labeled ASD
  example during training. It only knows what *healthy* dynamics look like, and flags
  deviation from that.
- Given both of those constraints, a near-chance result is a believable, honest
  outcome for a first-pass baseline, not a red flag. If Phase 1 had reported AUROC 0.9
  with this little labeled signal, *that* would be the thing to be suspicious of.

## 5. What Phase 2 Does Next (Already Built — See the `phase-2` Branch)

Phase 1 gives you a model that knows what "normal" looks like, but no way to manufacture
more abnormal examples for training or testing purposes — real ASD examples are scarce
and, per Section 4, hard to detect with this baseline in the first place.

Phase 2's job: **generate realistic anomalies on purpose**, grounded in real healthy
subgraphs, rather than by injecting random noise. Concretely:

1. Take a real transition from a real, healthy subject.
2. Ask the *trained* Phase 1 model: "which specific connections, if changed, would
   most increase your prediction error?" (a gradient computation through the GIN —
   the same kind of computation used in adversarial ML, but used here constructively).
3. Nudge exactly those connections, by an amount you control (a "severity" knob from
   0 to 1), in small repeated steps rather than one big jump (a big jump turned out to
   *not* reliably increase the anomaly score — documented as a real finding in
   `docs/phase2_report.md`).
4. Check that the result is still a valid, physically sensible brain graph (same ROIs,
   symmetric connections, weights in range) and that turning up the severity knob
   reliably makes the anomaly more detectable.

Phase 2 also builds a "random perturbation" version of the same recipe as a control
group, specifically to answer: is this smarter than just shaking the graph randomly?
(Short answer, already measured: yes, by about 9x, on the raw detection threshold —
see `docs/phase2_report.md` and the figures in `outputs/figures/phase2/`.)

If you want to inspect Phase 2 the same way you just inspected Phase 1:

```bash
git log --oneline --all              # see all three phase commits
cat docs/phase2_report.md            # full technical writeup
open outputs/figures/phase2/01_severity_vs_score.png
open outputs/figures/phase2/02_score_distributions.png
```

Phase 3 then asks the follow-up question — does adding these generated anomalies to a
tiny real training set actually help a detector? — and reports an honest (mixed) answer
in `docs/phase3_report.md` and `docs/final_report.md`.
