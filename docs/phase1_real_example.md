# Phase 1, Told Through One Real Control Subject and One Real ASD Subject

This is a companion to [`phase1_walkthrough.md`](phase1_walkthrough.md) and
[`phase1_report.md`](phase1_report.md), written for explaining Phase 1 to someone (like
a professor) who wants the real-world story next to the math, not instead of it.

Every number below is real, pulled directly from this project's own data and trained
model — not a made-up illustration. The two subjects are:

- **UM_1_007** — a **control** (non-autistic) subject, University of Michigan site
- **UM_1_062** — an **ASD** subject, same site, same scanner, same scan length

Same site and same scan length on purpose: it removes "different hospital, different
scanner, different recording length" as an explanation for anything we see, so any
difference we do see is a better candidate for being about the two subjects themselves.

---

## 1. What is a "subject" here, physically?

A subject is a real person who lay still in an MRI scanner for a few minutes while
**doing nothing in particular** — no task, no button-pressing. This is called
"resting-state fMRI." The scanner doesn't take a photo of brain structure (that's a
different kind of scan); it repeatedly measures blood oxygen levels across the brain,
roughly once every 1–2 seconds, because active brain tissue pulls in slightly more
oxygenated blood. So what you get out of the scanner, for one person, is:

> "How active was each of 360 predefined brain regions, at each moment in time, for
> the ~2–5 minutes they were lying in the scanner?"

That's it. No labels, no diagnosis baked into the numbers themselves — a matrix of
**time × 360 brain regions**, where region 1 might have 78 timepoints and region 1 of
a different subject might have 316, because scans varied in length across sites. This
raw file is not in this repository (`data/raw/` is intentionally not shipped — see the
earlier discussion in this conversation), but it's what M1's `ABIDELoader` reads before
any of the processing below happens.

## 2. So what actually differs between a control brain and an ASD brain here?

This is the part that's easy to get wrong intuitively. **Nothing looks visibly
different.** There's no lesion, no missing region, no structural damage on the scan. If
you looked at region 1's raw activity curve for UM_1_007 (control) next to UM_1_062
(ASD), you would not be able to tell which is which just by eyeballing it — both would
look like normal, healthy, noisy brain-activity wiggles.

The hypothesized difference is in **coordination between regions over time** — how well
different brain regions stay "in sync" or "out of sync" with each other, and whether
that pattern of synchrony is stable or drifts around. A rough analogy: imagine two
orchestras playing the same piece of music. Every individual musician (region) is
playing their part correctly — nothing sounds broken if you listen to any one
instrument alone. But in one orchestra, the sections stay locked to the same tempo and
handoffs happen cleanly; in the other, the coordination between sections drifts subtly
— a violin section quietly speeds up in one passage, a brass section lags in another.
You'd only notice by listening to *how sections relate to each other over time*, not by
listening to any single instrument. That's what functional connectivity is trying to
measure, and it's why this whole project is built around *pairs of regions over time*,
not single regions in isolation.

This also tells you something important about why Phase 1 turned out to be a hard,
honest baseline (see Section 6): if the underlying signal is this subtle even in
principle, a first-pass unsupervised model isn't going to easily find it.

## 3. From raw brain activity to "who's talking to whom": functional connectivity

M1 takes each subject's raw time × 360 matrix and slides a ~20–60 second window (30
timepoints, technically "TRs") across it, computing, for every window, the Pearson
correlation between every pair of the 360 regions. That gives one **360×360 matrix per
window**: entry `(i, j)` says "in this ~20-second stretch, did region i and region j
move up and down together (positive), move in opposite directions (negative), or show
no consistent relationship (near zero)?"

Here's an actual snippet, from the very first window of each subject, restricted to
region 1 (our example subject's "seed" region — more on that in Section 4):

| Connection | UM_1_007 (control) | UM_1_062 (ASD) |
|---|---|---|
| Region 1 ↔ Region 12 | −0.562 | *(not among region 1's top connections)* |
| Region 1 ↔ Region 9 | *(not among region 1's top connections)* | +0.751 |
| Region 1 ↔ Region 33 | *(not among region 1's top connections)* | +0.499 |

("Region" numbers are anonymous indices 0–359 from the ABIDE parcellation — this
repository doesn't carry a name lookup table, e.g. "region 1 = left amygdala," so we
report indices honestly rather than inventing anatomical labels that aren't backed by
data in this project.)

Only the strongest 20% of all 64,620 possible region-pairs are kept per window (everything
else is discarded as noise) — that's M1's sparsification step. Notice that the two
subjects don't even keep the *same* connections as "strong" around region 1 — already a
hint that their region-1 neighborhoods look structurally different, before any model
gets involved.

## 4. From a whole-brain matrix to one neighborhood: the subgraph

360 regions all interacting is too much to reason about at once, and most of it is
irrelevant to any one specific question. M2 picks a **seed region** and pulls out its
local neighborhood — every region directly connected to it (in this walkthrough, seed
= region 1) — and keeps that same set of regions fixed across every time window for
that subject, so you're always looking at "region 1's neighborhood," even as the
*connections within it* strengthen, weaken, or flip sign over time.

For our two subjects, region 1's neighborhood turned out to be:

| | UM_1_007 (control) | UM_1_062 (ASD) |
|---|---|---|
| Neighborhood size | 39 regions | 38 regions |
| Strong connections inside it (window 1) | 280 | 235 |

Similar size, meaningfully different connection count — the control subject's region-1
neighborhood is a bit more densely, strongly connected in this first window than the
ASD subject's.

## 5. From a neighborhood to one list of 128 numbers: the GIN

A raw list of "39 regions and 280 weighted connections" isn't something a simple model
can compare across subjects or across time directly — the exact same neighborhood, as
a set of numbers, is a different-sized object every time. The GIN's job is to compress
"this neighborhood, with these specific connection strengths, right now" into one
**fixed-size list of 128 numbers** — call it `z`. It does this by having each region
repeatedly "listen" to its directly connected neighbors (weighted by connection
strength) and update its own summary, twice over (2 layers), then averaging everyone's
summary into one 128-number fingerprint for the whole neighborhood.

Concretely, the first 5 (of 128) numbers, for the very first two time windows:

| | UM_1_007 (control), z at window 0 | UM_1_007, z at window 1 |
|---|---|---|
| | `[2.036, 1.657, -1.904, 1.922, -1.748]` | `[2.489, 2.096, -2.294, 2.323, -2.137]` |

| | UM_1_062 (ASD), z at window 0 | UM_1_062, z at window 1 |
|---|---|---|
| | `[1.139, 0.882, -1.005, 1.101, -0.820]` | `[1.529, 1.231, -1.425, 1.449, -1.222]` |

These numbers don't mean anything on their own by inspection — nobody can look at
"1.657" and say what it represents biologically. That's expected and fine: the GIN
wasn't told what any of these numbers should mean. It only had to get good at one
thing, described next.

## 6. Learning "normal": the GRU, and how an anomaly is actually decided

The GRU watches a subject's sequence of `z` values, window after window, and — using
only what it's seen **so far** — predicts what the *next* `z` should look like. It was
trained only on control subjects, so "what should come next" specifically means "what
would a healthy brain's neighborhood typically do next."

Continuing the same subjects: given only `z` at window 0, here is what the trained GRU
predicted for window 1, next to what the region's neighborhood actually did:

| | Predicted `z_hat` at window 1 | Actual `z` at window 1 | Distance between them (`A_0`) |
|---|---|---|---|
| UM_1_007 (control) | `[2.271, 1.965, -2.061, 2.088, -1.864]` | `[2.489, 2.096, -2.294, 2.323, -2.137]` | **2.456** |
| UM_1_062 (ASD) | `[1.491, 1.258, -1.408, 1.394, -1.210]` | `[1.529, 1.231, -1.425, 1.449, -1.222]` | **0.507** |

**This distance — "how wrong was the model's guess about what happens next" — is the
entire anomaly score.** Large distance = this transition looked less like what a
healthy brain normally does. Nothing more mysterious than that: it's a prediction-error
number, decided per region-neighborhood, per time window, per subject.

Doing this across **all 20 tracked seed regions and every time window** for a subject,
and taking the single worst (largest) one, gives one overall number for that subject:

| Subject | Worst-transition anomaly score (whole subject, all 20 seed regions) |
|---|---|
| UM_1_007 (control) | **14.45** |
| UM_1_062 (ASD) | **11.73** |

A subject is finally flagged "anomalous" if this number crosses a cutoff
(`tau_subject ≈ 19.81`), which was calibrated *only* on a separate set of 70 healthy
subjects the model never trained on — specifically so the cutoff represents "how
surprising is this, for a healthy person" and not something tuned to catch ASD subjects
directly.

## 7. What this example honestly shows

Neither subject crosses the cutoff (19.81) here — **both get scored "not anomalous"**
by Phase 1's detector, and, notably, the *control* subject's own worst score (14.45) is
actually higher than the ASD subject's (11.73). This is not a cherry-picked embarrassing
example — it's a fair, unremarkable pair pulled by matching site and scan length, and
it's a faithful, concrete picture of exactly what Phase 1's headline number means:

> **Test AUROC ≈ 0.45 (chance level).** Given a real control subject and a real ASD
> subject, this baseline detector's prediction-error score is about as likely to rank
> the control subject as "more anomalous" as it is to rank the ASD subject that way.

That is the honest state of the baseline, and it's *why* the project exists past Phase
1: if simply learning "predict the next window's neighborhood summary" already
separated the two groups cleanly, there'd be nothing left to build. Instead, Phase 1's
real function is to hand Phase 2 something concrete to react to: a trained model that
knows what a healthy brain's dynamics look like, even if it doesn't yet reliably know
what ASD dynamics look like.

## 8. Where Phase 2 picks this up

Phase 2 takes exactly this same trained GIN+GRU and asks a constructive question of it:
"for this real, healthy subject's real neighborhood, which specific connections, if I
deliberately changed them, would make *you* (the model) most surprised?" It then nudges
exactly those connections by a controllable amount, producing a synthetic-but-realistic
anomalous version of a real healthy subject's brain-region neighborhood — the graph
equivalent of "make this specific orchestra section drift out of sync, on purpose, by
a controllable amount, and check that a listener actually notices more drift as you
turn the knob up." See [`phase2_report.md`](phase2_report.md) for the numbers on that.
