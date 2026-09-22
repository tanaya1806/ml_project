# Codebase Guide — What Each File Does

A quick map of the repository. Grouped by folder, roughly in pipeline order. See
`docs/final_report.md` for the research narrative; this file is just "what's where."

## `configs/` — settings for each stage

| File | Purpose |
|---|---|
| `baseline.yaml` | M1/M2 settings: dataset paths, window size/stride, sparsification ratio, split ratios, subgraph k-hop/seed strategy. |
| `phase1.yaml` | Phase 1 settings: seed-ROI count, GIN/GRU dimensions, training hyperparameters, threshold percentile, few-shot shot counts. |
| `phase2.yaml` | Phase 2 settings: edge-perturbation budget, severity levels, generation dataset size. |
| `phase3.yaml` | Phase 3 settings: shot counts, trials per shot, number of generated anomalies per trial, classifier hyperparameters. |

## `src/data/` — raw data → temporal graphs (M1)

| File | Purpose |
|---|---|
| `loader.py` | `ABIDELoader` — reads the per-site NPZ files into `SubjectRecord`s (time series, site, diagnosis label). |
| `sliding_window.py` | Turns one subject's ROI time series into a sequence of windowed Pearson functional-connectivity matrices. |
| `graph_construction.py` | Sparsifies each FC matrix to the top-20%-by-|weight| edges, keeping signed correlation — produces `TemporalGraph`s. |
| `dataset_splits.py` | Builds the stratified, subject-level train/val/test split (never window-level). |
| `preprocess_pipeline.py` | End-to-end M1 script: loads all subjects, builds splits, writes `data/processed/graphs/*.npz`. |
| `validate_fc.py`, `verify_dataset.py` | Sanity-check scripts for the processed dataset (symmetry, bounds, counts). |
| `subgraph_cache.py` | Picks the 20 reproducible seed ROIs and caches each subject's stable k-hop subgraph sequence to disk (Phase 1). |
| `build_phase1_cache.py` | Driver script that runs `subgraph_cache.py` over every subject once. |

## `src/graphs/` — candidate subgraphs (M2)

| File | Purpose |
|---|---|
| `subgraph_extraction.py` | Core k-hop neighborhood + induced-subgraph extraction around one seed ROI in one window. |
| `temporal_tracking.py` | Tracks a subgraph across windows — `stable` (fixed ROI set, used everywhere downstream) and `dynamic` (M2 comparison only) strategies. |
| `pilot_experiment.py` | One-off M2 benchmark script (runtime/memory/storage estimates); not used by later phases. |

## `src/models/` — the trained backbone (M3 + Phase 1)

| File | Purpose |
|---|---|
| `gin_encoder.py` | `GINEncoder` — the weighted GIN that turns one subgraph into a 128-D embedding; `subgraph_to_pyg()` converts a subgraph to a PyG `Data` object. |
| `gru_temporal.py` | `TemporalGRU` — predicts the next embedding in a sequence from the embeddings seen so far. |
| `embedding_sequence.py` | Runs the GIN over every window of a cached subgraph in one batched call, producing the `z_1...z_T` sequence. |
| `pilot_gin.py` | One-off M3 benchmark script (throughput/NaN checks); not used by later phases. |

## `src/training/` — Phase 1 training loop

| File | Purpose |
|---|---|
| `train_phase1.py` | Trains GIN + GRU jointly on train-control subjects, with the collapse-prevention variance penalty; saves the checkpoint. |

## `src/anomaly/` — the anomaly score itself

| File | Purpose |
|---|---|
| `scoring.py` | `transition_scores()` (A_t = ‖actual − predicted‖) and `calibrate_threshold()` (percentile-based threshold from a set of scores). |

## `src/evaluate/` — turning scores into detection results

| File | Purpose |
|---|---|
| `detect_baseline.py` | Phase 1 evaluation: calibrates `tau_transition`/`tau_subject` on validation controls, scores the held-out test set, runs the few-shot recurrence check. |
| `validate_generator.py` | Phase 2 validation: checks generated anomalies for validity/diversity/severity-monotonicity and measures detectability against Phase 1's frozen threshold. |
| `phase3_final_comparison.py` | Phase 3: the three-condition (few-shot-only / +structure-aware / +random) few-shot classifier comparison, evaluated on the same held-out test set. |

## `src/generation/` — Phase 2, the structure-aware generator

| File | Purpose |
|---|---|
| `structure_aware_generator.py` | `generate_anomaly()` — the actual generator: gradient-guided iterative edge perturbation (`method="structure_aware"`) and the random-perturbation ablation (`method="random"`). |
| `generate_phase2_dataset.py` | Driver script that runs the generator over many (subject, seed, severity, method) combinations and saves the metadata. |

## `src/fewshot/` — support-set sampling shared by Phase 1E and Phase 3

| File | Purpose |
|---|---|
| `support.py` | `sample_support_set()` (deterministic k-shot draw from a subject pool), `assert_subject_isolation()` (no val/test leakage), `compute_recurrence()`. |

## `src/utils/`

| File | Purpose |
|---|---|
| `device.py` | Picks MPS (Apple Silicon) if available, else CPU. Never hard-codes CUDA. |

## `src/visualization/`

| File | Purpose |
|---|---|
| `generate_plots.py`, `generate_subgraph_plots.py` | M1/M2 sanity-check figures. |
| `generate_phase1_plots.py` | Training-curve and anomaly-score-distribution figures for Phase 1. |
| `generate_phase3_plots.py` | F1/AUROC-vs-shot-count comparison figures for Phase 3. |

## `tests/`

One file per module above, named to match (`test_pipeline.py` ↔ M1 loader/graph code,
`test_subgraphs.py` ↔ M2, `test_gin_encoder.py`/`test_gru_temporal.py` ↔ models,
`test_subgraph_cache.py`/`test_embedding_sequence.py`/`test_anomaly_scoring.py`/
`test_fewshot.py` ↔ Phase 1, `test_generator.py` ↔ Phase 2, `test_phase3_pipeline.py` ↔
Phase 3's classifier/aggregation logic).

## `docs/`

| File | Purpose |
|---|---|
| `phase1_report.md`, `phase2_report.md`, `phase3_report.md` | Technical write-up per phase: method, experiments actually run, results, limitations. |
| `final_report.md` | The project-wide narrative and the headline comparison table. |
| `phase1_walkthrough.md`, `phase1_real_example.md` | Plainer-language companions to the Phase 1 report, the second grounded in two real subjects' actual numbers. |
| `phase1_flowchart.docx` / `.puml` | Detailed, editable Phase 1 methodology flowchart (Word shapes + PlantUML source). |
| `project_system_architecture.docx` | The one-page, whole-project, left-to-right architecture diagram. |
| `codebase_guide.md` | This file. |

## `okf/`

A machine-readable [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle describing the same project (dataset, each model, each phase's experiment,
data provenance, limitations) — see `okf/index.md` for the entry point.

## Where things actually run from

There's no single `main.py`; each stage is its own module run with `python -m`:

```bash
python -m src.data.preprocess_pipeline          # M1: raw -> temporal graphs
python -m src.data.build_phase1_cache           # cache the 20-seed subgraphs
python -m src.training.train_phase1             # train GIN+GRU
python -m src.evaluate.detect_baseline           # Phase 1 metrics
python -m src.generation.generate_phase2_dataset  # Phase 2 generated-anomaly set
python -m src.evaluate.validate_generator         # Phase 2 validation
python -m src.evaluate.phase3_final_comparison    # Phase 3 final comparison
pytest tests/                                     # everything above, tested
```
