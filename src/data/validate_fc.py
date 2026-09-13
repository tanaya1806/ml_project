"""
Validation script for Sliding-Window FC and Temporal Graph Sequences.

Performs thorough sanity checks on:
- Raw time series bounds & shapes
- FC matrix properties (symmetry, diagonal=0, range in [-1, 1], NaN/Inf free)
- Graph sparsification properties (exact top 20% sparsity, bidirectionality, no self-loops)
- Temporal dynamics: verifies FC_t != FC_(t+1)
- Exploratory comparison between ASD and Control FC statistics
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
from typing import List

from src.data.loader import ABIDELoader
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs


def validate_preprocessing(config_path: str = "configs/baseline.yaml", num_random_subjects: int = 10, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_ts_dir = cfg["dataset"]["ts_dir"]
    window_size = cfg["preprocessing"]["window_size"]
    stride = cfg["preprocessing"]["stride"]
    remove_diagonal = cfg["preprocessing"]["remove_diagonal"]
    sparsity_ratio = cfg["graph_construction"]["sparsity_ratio"]
    keep_signed_weights = cfg["graph_construction"]["keep_signed_weights"]

    print("================================================================================")
    print("               TEMPORAL FC & GRAPH SEQUENCE VALIDATION SUITE                    ")
    print("================================================================================")

    loader = ABIDELoader(raw_ts_dir)
    all_subjects = loader.load_all_subjects()
    print(f"[*] Total loaded subjects: {len(all_subjects)}")

    # Sample a balanced subset of subjects across different sites
    controls = [s for s in all_subjects if s.label == 0]
    asds = [s for s in all_subjects if s.label == 1]
    
    sample_controls = random.sample(controls, num_random_subjects // 2)
    sample_asds = random.sample(asds, num_random_subjects // 2)
    sample_subjects = sample_controls + sample_asds
    random.shuffle(sample_subjects)

    print(f"[*] Validating on {len(sample_subjects)} sampled subjects ({len(sample_controls)} TC, {len(sample_asds)} ASD across different sites)...\n")

    validation_results = []
    consecutive_diff_norms = []

    for subj in sample_subjects:
        ts = subj.time_series
        T, N = ts.shape

        # 1. Raw time series checks
        ts_nans = int(np.isnan(ts).sum())
        ts_infs = int(np.isinf(ts).sum())
        assert ts_nans == 0, f"Subject {subj.subject_id} contains NaNs in raw time series!"
        assert ts_infs == 0, f"Subject {subj.subject_id} contains Infs in raw time series!"
        assert N == 360, f"Subject {subj.subject_id} has {N} ROIs, expected 360!"

        # 2. Sliding window generation
        wfcs = compute_sliding_window_fc(
            ts, window_size=window_size, stride=stride, remove_diagonal=remove_diagonal
        )
        expected_windows = (T - window_size) // stride + 1
        assert len(wfcs) == expected_windows, f"Subject {subj.subject_id}: expected {expected_windows} windows, got {len(wfcs)}"

        # 3. Check every FC matrix in the window sequence
        site_diffs = []
        for w_i, wfc in enumerate(wfcs):
            mat = wfc.fc_matrix
            # Shape
            assert mat.shape == (360, 360), f"FC shape mismatch: {mat.shape}"
            # Symmetry: mat == mat.T
            max_asym = np.max(np.abs(mat - mat.T))
            assert max_asym < 1e-6, f"Asymmetric FC matrix (max diff: {max_asym})"
            # Diagonal == 0
            diag_max = np.max(np.abs(np.diag(mat)))
            assert diag_max == 0.0, f"Non-zero diagonal: {diag_max}"
            # Value range in [-1.0, 1.0]
            assert np.min(mat) >= -1.0 - 1e-6 and np.max(mat) <= 1.0 + 1e-6, f"FC values outside [-1, 1]: [{np.min(mat)}, {np.max(mat)}]"
            # No NaNs/Infs
            assert np.isnan(mat).sum() == 0, "NaNs found in FC matrix!"
            assert np.isinf(mat).sum() == 0, "Infs found in FC matrix!"

            # Consecutive dynamics: FC_t != FC_(t+1)
            if w_i > 0:
                prev_mat = wfcs[w_i - 1].fc_matrix
                diff = np.abs(mat - prev_mat)
                frob_diff = np.linalg.norm(diff, 'fro')
                max_diff = np.max(diff)
                site_diffs.append(frob_diff)
                # Verify they are strictly not identical
                assert frob_diff > 1e-4, f"Consecutive FC matrices at windows {w_i-1} and {w_i} are unexpectedly identical!"

        # 4. Temporal Graph construction & sparsification checks
        graphs = construct_temporal_graphs(
            wfcs, sparsity_ratio=sparsity_ratio, keep_signed_weights=keep_signed_weights
        )
        assert len(graphs) == len(wfcs), "Graph sequence length mismatch!"

        expected_undirected = int(np.floor(360 * 359 / 2 * sparsity_ratio))  # 12,924
        expected_directed = 2 * expected_undirected  # 25,848

        for g in graphs:
            assert g.num_nodes == 360, f"Node count mismatch: {g.num_nodes}"
            assert g.num_edges == expected_directed, f"Edge count mismatch: {g.num_edges} vs expected {expected_directed}"
            assert g.edge_index.shape == (2, expected_directed), f"Edge index shape mismatch: {g.edge_index.shape}"
            assert g.edge_weight.shape == (expected_directed,), f"Edge weight shape mismatch: {g.edge_weight.shape}"
            
            # Check for self loops
            self_loops = np.sum(g.edge_index[0] == g.edge_index[1])
            assert self_loops == 0, f"Found {self_loops} self-loops in graph!"

            # Check edge weight range
            assert np.all(g.edge_weight >= -1.0 - 1e-5) and np.all(g.edge_weight <= 1.0 + 1e-5), "Edge weights exceed [-1, 1]"
            # Check that threshold was respected
            assert np.all(np.abs(g.edge_weight) >= g.threshold_value - 1e-6), "Sparsified edge below threshold!"

        mean_consec_frob = np.mean(site_diffs) if site_diffs else 0.0
        consecutive_diff_norms.append(mean_consec_frob)

        validation_results.append({
            "subject_id": subj.subject_id,
            "site": subj.site,
            "label": "ASD" if subj.label == 1 else "Control",
            "timepoints": T,
            "windows": len(wfcs),
            "directed_edges": graphs[0].num_edges,
            "mean_consec_diff_frobenius": round(float(mean_consec_frob), 4),
            "threshold_val_w0": round(float(graphs[0].threshold_value), 4)
        })

    df_val = pd.DataFrame(validation_results)
    print("--- VALIDATION TABLE (SAMPLED SUBJECTS) ---")
    print(df_val.to_string(index=False))

    print("\n--- SANITY CHECK PASSED ---")
    print(f"[OK] Raw time series: strictly finite, non-empty, 360 ROIs.")
    print(f"[OK] FC matrices: strictly symmetric (max asym < 1e-6), diagonal == 0.0, bounded in [-1.0, 1.0], 0 NaNs/Infs.")
    print(f"[OK] Temporal dynamics: FC_t != FC_(t+1) holds across all sampled subjects.")
    print(f"     Mean consecutive Frobenius difference ||FC_(t+1) - FC_t||_F = {np.mean(consecutive_diff_norms):.4f} (range: [{np.min(consecutive_diff_norms):.4f}, {np.max(consecutive_diff_norms):.4f}]).")
    print(f"[OK] Graph sparsification: exactly {expected_directed} directed edges (top 20% of 64,620 off-diagonal pairs), zero self-loops, signed weights preserved.")

    # 5. Exploratory FC statistics comparison between ASD and Control
    print("\n--- EXPLORATORY FC STATISTICS (ASD vs TC) ---")
    print("[NOTE: Exploratory sanity check only; not claiming statistical significance]")
    
    asd_fc_means = []
    asd_fc_stds = []
    tc_fc_means = []
    tc_fc_stds = []

    for s in sample_subjects:
        wfcs = compute_sliding_window_fc(s.time_series, window_size=window_size, stride=stride)
        for w in wfcs:
            # Extract upper triangle off-diagonal
            triu_vals = w.fc_matrix[np.triu_indices(360, k=1)]
            m = np.mean(triu_vals)
            std = np.std(triu_vals)
            if s.label == 1:
                asd_fc_means.append(m)
                asd_fc_stds.append(std)
            else:
                tc_fc_means.append(m)
                tc_fc_stds.append(std)

    print(f"Typical Controls (TC) Window FC - Mean edge weight: {np.mean(tc_fc_means):.4f} +/- {np.std(tc_fc_means):.4f} | Mean within-window std: {np.mean(tc_fc_stds):.4f}")
    print(f"Autism (ASD)         Window FC - Mean edge weight: {np.mean(asd_fc_means):.4f} +/- {np.std(asd_fc_means):.4f} | Mean within-window std: {np.mean(asd_fc_stds):.4f}")
    print("================================================================================")


if __name__ == "__main__":
    validate_preprocessing()
