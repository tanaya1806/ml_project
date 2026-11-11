# Output_to_be_shown — Real Visual Evidence of the GRAPH_ANOMALY Pipeline

**Every figure in this folder is generated from the actual repository: the actual
ABIDE-derived processed data, the actual trained Phase 1 checkpoint
(`outputs/experiments/phase1/checkpoints/best.pt`), the actual Phase 2 generator
code, and the actual committed experiment results.** No metric was invented,
adjusted, or "improved." Where a figure is not built from real data (one figure,
clearly marked), it is explicitly labeled **ILLUSTRATIVE SCHEMATIC — NOT
EXPERIMENTAL DATA** in the image itself. Every other figure carries a green
**REAL DATA · REAL COMPUTATION** badge.

## How this was built, and how to reproduce it

All generation code lives in `_scripts/` (a `style.py` shared style module,
`common_model.py` that loads the real checkpoint once, and one numbered script
per section below). Run them in order from the repository root:

```bash
python3 Output_to_be_shown/_scripts/01_original_data.py
python3 Output_to_be_shown/_scripts/02_graph_and_gin.py
python3 Output_to_be_shown/_scripts/03_gru_phase1.py
python3 Output_to_be_shown/_scripts/04_phase1_results.py
python3 Output_to_be_shown/_scripts/05_phase2_generation.py
python3 Output_to_be_shown/_scripts/06_phase2_results.py
python3 Output_to_be_shown/_scripts/07_anomaly_examples.py
```

Each script prints a `[verify]` block to the console showing the real numbers it
computed, and several **assert against the project's own committed result files**
(`outputs/experiments/phase1/metrics.json`, `outputs/experiments/phase2/generated_anomalies.json`,
`outputs/experiments/phase2/generator_validation.json`) so the script fails loudly
if a recomputation ever stops matching the real, already-published results. Each
folder also has a `_verification.json` with the exact numbers used in its figures.

**Two independent, bit-for-bit verifications were performed while building this:**
1. `04_phase1_results.py` recomputes AUROC/AUPRC/the confusion matrix for all 131
   real test subjects from scratch (a fresh forward pass through the real GIN+GRU)
   and asserts the result is identical to the committed `metrics.json` — confirmed:
   AUROC 0.4526932084309133 both times, TP/TN/FP/FN = 2/68/2/59 both times.
2. `05_phase2_generation.py` re-runs the real 5-step generator on a real record and
   asserts its final anomaly score matches the corresponding row already persisted
   in `generated_anomalies.json`, for both the structure-aware and random methods —
   confirmed to within float32 tolerance (8.133944 vs. 8.133946418762237, etc.).

## A necessary honesty note: what "real" means here for the earliest stage

Raw ABIDE ROI time series (`data/raw/Time_serie_per_site/site_*_time_series.npz`,
the path `configs/baseline.yaml` expects) **are not present in this repository
instance** — consistent with this project's own test suite, which already skips
its raw-data tests for the same reason. `src/data/graph_construction.py` computes
the dense 360×360 Pearson matrix and sparsifies it in one step, without ever
persisting the dense intermediate (a deliberate storage-efficiency choice: the
sparse form is ~1.4GB across all subjects/windows; the dense form would be roughly
two orders of magnitude larger). So:

- **The sparsified functional-connectivity matrix is 100% real** — every nonzero
  cell is an actual Pearson r computed by the actual pipeline, reconstructed from
  `data/processed/graphs/*.npz`, which stores exactly the `(edge_index, edge_weight)`
  pairs that survived top-20% sparsification.
- **The one figure that would require the raw time series themselves**
  (`01_original_data/temporal_window_example.png`) uses clearly-labeled synthetic
  example signals to explain the sliding-window mechanism, and is the only
  illustrative figure in this entire folder.
- Every other figure — subgraphs, embeddings, GRU predictions, anomaly scores,
  Phase 1 results, Phase 2 generation, Phase 2 results — is computed from real,
  already-processed project data and/or a real forward pass through the real
  trained model.

## Subjects used (for continuity across figures)

| Subject | Role | Why chosen |
|---|---|---|
| `UM_1_007` | real ABIDE I **test-control** subject | Phase 1 figures (control side) |
| `UM_1_062` | real ABIDE I **test-ASD** subject, same site as above | Phase 1 figures (ASD side) — matched site for a fair comparison |
| `KKI_002`, and 2 more real subjects | real ABIDE I **train-control** subjects, actually among the 60 base subjects Phase 2's real generator sampled | Phase 2 figures |

Seed ROI 284 (33-node neighborhood) and seed ROI 1 are used where a compact,
readable subgraph was needed — both are real Phase 1 seed ROIs
(`configs/phase1.yaml` → `select_seed_rois()`), just chosen for having fewer nodes
than some of the other 18 seeds, for legibility.

---

## 01_original_data/ — What the ABIDE-derived data actually looks like

| File | Purpose | Source | Status |
|---|---|---|---|
| `pearson_correlation_matrix.png` | The real 360×360 sparsified functional-connectivity matrix for one real subject/window | `data/processed/graphs/UM_1_007.npz`, window 0 | **REAL** |
| `pearson_correlation_matrix_labeled.png` | Same real matrix, with a reading-guide panel (rows/columns/cell meaning/color/diagonal) | same | **REAL** |
| `temporal_window_example.png` | Concept diagram: ROI time series → 30-TR window → correlation | synthetic example signals (raw time series not present in this repo instance) | **ILLUSTRATIVE** |

**Interpretation:** each nonzero cell is a real signed Pearson correlation that
survived the pipeline's own top-20%-by-|r| sparsification (12,924 undirected edges
kept out of 64,620 possible — exactly 20.0%). White cells were not "no signal,"
they simply weren't among the strongest 20% for that window.

## 02_graph_construction/ — The real subgraph that Phase 1/2 actually operate on

| File | Purpose | Source | Status |
|---|---|---|---|
| `selected_subgraph.png` | Real cached stable subgraph: seed ROI 284, subject UM_1_007, window 0 (33 nodes, 231 real undirected edges) | `data/processed/subgraph_cache/UM_1_007.npz` | **REAL** |

**Interpretation:** this is a genuine 1-hop, stable-tracked neighborhood as
produced by `src/graphs/temporal_tracking.py` — the same ROI set is used at every
one of this subject's 27 time windows, per Phase 1's "stable tracking" design
(only the internal edge weights evolve across windows).

## 03_gin/ — What actually enters the GIN, and what it actually produces

| File | Purpose | Source | Status |
|---|---|---|---|
| `gin_input_subgraph.png` | The real subgraph above, annotated with exactly what GIN receives (constant node feature 1.0, signed edge weights) | same subgraph | **REAL** |
| `gin_message_passing.png` | Real weighted message-passing into the seed ROI — arrow labels are the real edge weights (layer-1 messages equal the edge weight exactly, since node features start at 1.0); only the 8 strongest of 32 real neighbors are drawn, for legibility | same subgraph | **REAL** |
| `gin_embedding_128d.png` | The real 128-D embedding vector produced by a real forward pass through the trained GIN | `outputs/experiments/phase1/checkpoints/best.pt` | **REAL** |
| `embedding_comparison.png` | Two real embeddings, same seed ROI and window, two real subjects (UM_1_007 control vs. UM_1_062 ASD) | same checkpoint | **REAL** |

**Interpretation:** the 128 dimensions are a *joint* learned representation of
subgraph structure — individual dimensions are not claimed to correspond to
individual biological variables. A real, reproducible property of this trained
checkpoint is visible in the embedding plots: each dimension tends to keep a
consistent sign across different subjects/subgraphs, with the *magnitude*
carrying most of the structural signal — reported here as observed, not
smoothed over.

## 04_gru_phase1/ — What the real GRU actually predicts, and the real anomaly score

| File | Purpose | Source | Status |
|---|---|---|---|
| `temporal_embedding_sequence.png` | The real z₁...z_T sequence (27 windows × 128 dims) that feeds the GRU, subject UM_1_007 | real forward pass | **REAL** |
| `gru_actual_vs_predicted.png` | Real actual-next-embedding vs. real GRU-predicted-embedding vs. their real difference, plus the real computed A_t | real forward pass | **REAL** |
| `anomaly_score_over_time.png` | Real per-transition anomaly score A_t across all 26 transitions, for both UM_1_007 (control) and UM_1_062 (ASD), with both real thresholds (τ_transition, τ_subject) drawn and kept visually distinct | real forward pass + `outputs/experiments/phase1/metrics.json` | **REAL** |

**Interpretation:** both real subjects shown cross τ_transition at least once —
this is expected and documented: with ~26 independent transitions per subject at
a 95th-percentile per-transition cutoff, occasional crossings happen by chance
for control and ASD subjects alike. That is precisely why the actual subject-level
decision uses τ_subject (calibrated on each subject's *own maximum* A_t), not
τ_transition — see the two-threshold discussion in `docs/phase1_report.md`.

## 05_phase1_results/ — Real, aggregate Phase 1 detection results

| File | Purpose | Source | Status |
|---|---|---|---|
| `phase1_metrics.png` | All real Phase 1 metrics in one panel (Precision, Recall, F1, AUROC, AUPRC, both τ values, confusion-matrix counts, subjects flagged) | `outputs/experiments/phase1/metrics.json` | **REAL** |
| `phase1_confusion_matrix.png` | Real confusion matrix, recomputed from a fresh forward pass over all 131 real test subjects, verified to exactly match the committed metrics | recomputed + verified | **REAL** |
| `phase1_roc_curve.png` | Real ROC curve from real per-subject scores vs. true diagnosis labels (AUROC 0.453, at chance) | recomputed + verified | **REAL** |
| `phase1_pr_curve.png` | Real Precision-Recall curve (AUPRC 0.467, base rate 0.466) | recomputed + verified | **REAL** |
| `phase1_training_curves.png` | Copied from the project's own existing figure — real 5-epoch training/validation loss | `outputs/figures/phase1/01_training_curves.png` | **REAL** (pre-existing) |
| `anomaly_score_distribution.png` | Copied from the project's own existing figure — real subject-level score distributions (val-control / test-control / test-ASD) with τ_subject marked | `outputs/figures/phase1/02_anomaly_score_distributions.png` | **REAL** (pre-existing) |

**Interpretation:** AUROC 0.453 is below chance (0.500). This is reported as
measured. Phase 1's role in this project is to establish the normal-dynamics
baseline that Phase 2 is built on and measured against — not to be presented as
a strong standalone classifier.

## 06_phase2_generation/ — How Phase 2 actually modifies a real graph

| File | Purpose | Source | Status |
|---|---|---|---|
| `normal_subgraph_before.png` | The real train-control subgraph before any perturbation (subject KKI_002, seed 1, transition 1→2) | `data/processed/subgraph_cache/KKI_002.npz` | **REAL** |
| `edge_sensitivity.png` | Real gradient-based sensitivity `\|∂A_t/∂w_ij\|` for every one of the graph's 548 real edges, sorted, with the real top-15%-of-severity selection highlighted | real `torch.autograd.grad` through the trained GIN | **REAL** |
| `selected_top15_edges.png` | The same real selection, drawn on the real graph | same | **REAL** |
| `anomaly_generation_steps.png` | The real 5-step gradient-ascent trajectory (PGD-style), re-run step by step with the exact real inputs, verified to reach the exact real final score already persisted in `generated_anomalies.json` | instrumented re-run of `structure_aware_generator.generate_anomaly()`'s real algorithm | **REAL** |
| `anomalous_subgraph_after.png` | Real before/after graphs, same nodes and edges, only weights changed | same | **REAL** |
| `before_after_weight_changes.png` | Real before/after/Δ table for the modified edges | same | **REAL** |
| `anomaly_examples.png` | 6 real generated-anomaly records (3 real subjects × 2 methods), pulled directly from the persisted dataset with their exact subject/seed/severity/transition/edges-changed/scores | `outputs/experiments/phase2/generated_anomalies.json` | **REAL** |

**Interpretation:** the generator only ever edits the *weights* of edges that
already exist — it never adds or removes a node or an edge. That is why every
"before" and "after" panel above shows the identical node layout and the
identical edge set; only color/thickness (i.e., the real weight) changes on the
edges that were selected.

## 07_phase2_results/ — Real, aggregate Phase 2 validation results

| File | Purpose | Source | Status |
|---|---|---|---|
| `severity_vs_anomaly_score.png` | Real mean±std anomaly score vs. severity, both methods, across all 7,200 generated records | `outputs/experiments/phase2/generator_validation.json` | **REAL** |
| `structure_aware_vs_random.png` | Same real numbers, as a grouped bar comparison | same | **REAL** |
| `detection_rate_comparison.png` | Real threshold-crossing (detection) rate: real normal, real ASD, and all 3 severities of both generation methods — including the real ~9.6× ratio at high severity | same | **REAL** |
| `phase2_metrics.png` | Full real summary panel: record count, constraint violations, monotonicity, detection rates, diversity | same | **REAL** |

**Terminology note (deliberate):** Phase 2 does not report a classification
accuracy. It reports an **anomaly score** and a **threshold-crossing (detection)
rate** — the 45.8% vs. 4.8% figures are *detection rates*, not accuracies, and are
labeled as such throughout this folder.

**Interpretation, stated plainly:** structure-aware generation is strictly
monotonic in severity and ~9.6× more detectable than a random perturbation of the
identical edit budget at high severity — the central, validated Phase 2 claim.
Separately, and just as real: actual ASD test transitions are detected at
essentially the same rate as real normal transitions (4.9% vs. 5.0%) — real
clinical signal remains far subtler than even a severe synthetic perturbation at
this threshold. Both facts are shown, neither is hidden.

## 08_presentation_selected/ — The 11 figures that tell the whole story

A curated subset of the real figures above, renumbered for a linear walkthrough:
raw data → graph → subgraph → GIN embedding → GRU prediction → Phase 1 result →
Phase 2 before → edge sensitivity → before/after → generated examples → Phase 2
comparison → Phase 2 summary. Every image here is a copy of a real figure
documented above (see the tables for its original source and interpretation) —
nothing new was generated for this subfolder.

---

## What this folder deliberately does NOT contain

- No fabricated, adjusted, or "improved" metrics of any kind.
- No Phase 3 results — Phase 3 (few-shot evaluation) has real results elsewhere in
  this repository (`outputs/experiments/phase3/`), but this folder's brief was
  Phase 1 and Phase 2 only, so none are shown here.
- No claim of clinical diagnosis, biological causality, or that a generated
  anomaly is a real ASD biomarker. Phase 2's generated anomalies are a
  methodological tool for a few-shot detection experiment, not a clinical finding.
- No "accuracy" language applied to Phase 2, which does not compute one.

## Full number-by-number verification table

| Number shown | Value | Verified against | Figure(s) |
|---|---|---|---|
| Undirected edges kept, UM_1_007 window 0 | 12,924 (exactly 20.0% of 64,620) | recomputed from `data/processed/graphs/UM_1_007.npz` | `01_original_data/pearson_correlation_matrix.png` |
| Phase 1 checkpoint epoch / val loss | epoch 4 / 0.1133 | `outputs/experiments/phase1/checkpoints/best.pt` (loaded directly) | all `03_gin`, `04_gru_phase1` figures |
| Phase 1 AUROC | 0.4526932084309133 | recomputed == `outputs/experiments/phase1/metrics.json` (bit-for-bit) | `05_phase1_results/*` |
| Phase 1 AUPRC | 0.4669719215900707 | recomputed == `metrics.json` (bit-for-bit) | `05_phase1_results/*` |
| Phase 1 confusion matrix | TP=2, TN=68, FP=2, FN=59 | recomputed == `metrics.json` (exact) | `05_phase1_results/phase1_confusion_matrix.png` |
| τ_transition / τ_subject | 7.4350 / 19.8132 | `outputs/experiments/phase1/metrics.json` | `04_gru_phase1`, `05_phase1_results` |
| Phase 2 example score_after, structure-aware | 8.133944 (rerun) vs. 8.133946418762237 (persisted) | matches to float32 tolerance | `06_phase2_generation/anomaly_generation_steps.png` |
| Phase 2 example score_after, random | 2.871539 (rerun) vs. 2.87153959274292 (persisted) | matches to float32 tolerance | `06_phase2_generation/*` |
| Generated records / violations | 7,200 / 0 | `outputs/experiments/phase2/generator_validation.json` | `07_phase2_results/phase2_metrics.png` |
| Detection rate, structure-aware/high | 45.8% | same file (`detectability`) | `07_phase2_results/detection_rate_comparison.png` |
| Detection rate, random/high | 4.8% | same file | same |
| Ratio | 9.6× | computed directly from the two lines above | same |
| Diversity (Jaccard distance) | 0.999–1.000 | same file (`diversity`) | `07_phase2_results/phase2_metrics.png` |

Every `_verification.json` file inside each numbered folder carries the complete
set of real numbers used in that folder's figures, for independent spot-checking.
