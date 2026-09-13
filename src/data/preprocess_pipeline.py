"""
End-to-End Preprocessing Pipeline for ABIDE I Temporal Graphs.

Executes:
1. Subject loading across all 20 sites
2. Subject-level metadata extraction and splitting (train/val/test)
3. Sliding-window Functional Connectivity (FC) computation
4. Deterministic graph sparsification (top 20% absolute magnitude)
5. Efficient per-subject compressed storage (.npz) in data/processed/graphs/
"""

import os
import yaml
import time
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from src.data.loader import ABIDELoader
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs
from src.data.dataset_splits import create_subject_splits


def run_preprocessing_pipeline(config_path: str = "configs/baseline.yaml", max_subjects: Optional[int] = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_ts_dir = cfg["dataset"]["ts_dir"]
    graphs_dir = cfg["dataset"]["graphs_dir"]
    metadata_dir = cfg["dataset"]["metadata_dir"]
    
    window_size = cfg["preprocessing"]["window_size"]
    stride = cfg["preprocessing"]["stride"]
    remove_diagonal = cfg["preprocessing"]["remove_diagonal"]
    sparsity_ratio = cfg["graph_construction"]["sparsity_ratio"]
    keep_signed_weights = cfg["graph_construction"]["keep_signed_weights"]

    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print("================================================================================")
    print("                 ABIDE I TEMPORAL GRAPH PREPROCESSING PIPELINE                 ")
    print("================================================================================")
    print(f"[*] Configuration: {config_path}")
    print(f"[*] Window Size: {window_size} TRs | Stride: {stride} TRs")
    print(f"[*] Sparsity Ratio: {sparsity_ratio*100:.0f}% (Absolute magnitude cutoff)")
    print(f"[*] Output Graphs Directory: {graphs_dir}")

    # 1. Load summary and create splits
    loader = ABIDELoader(raw_ts_dir)
    print(f"[*] Scanning {len(loader.available_sites)} sites...")
    summary_df = loader.get_summary_df()
    
    summary_csv_path = os.path.join(metadata_dir, "subject_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[*] Saved subject summary to: {summary_csv_path}")

    # Create subject-level splits
    splits_json_path = os.path.join(metadata_dir, "subject_splits.json")
    create_subject_splits(
        subject_summary_df=summary_df,
        train_ratio=cfg["splitting"]["train_ratio"],
        val_ratio=cfg["splitting"]["val_ratio"],
        test_ratio=cfg["splitting"]["test_ratio"],
        random_seed=cfg["splitting"]["random_seed"],
        output_path=splits_json_path
    )

    # 2. Process subjects into temporal graphs
    print("\n[*] Processing subjects and generating temporal graph sequences...")
    start_time = time.time()
    total_processed = 0
    total_windows_generated = 0
    total_bytes_written = 0

    all_subjects = loader.load_all_subjects()
    if max_subjects is not None:
        all_subjects = all_subjects[:max_subjects]

    for subj in tqdm(all_subjects, desc="Generating Graph Sequences"):
        # Step 1: Sliding-window FC
        window_fcs = compute_sliding_window_fc(
            time_series=subj.time_series,
            window_size=window_size,
            stride=stride,
            remove_diagonal=remove_diagonal
        )
        
        # Step 2: Sparsify into temporal graphs
        graphs = construct_temporal_graphs(
            window_fcs=window_fcs,
            sparsity_ratio=sparsity_ratio,
            keep_signed_weights=keep_signed_weights
        )

        # Step 3: Pack into compact arrays
        edge_indices = np.stack([g.edge_index for g in graphs], axis=0)  # Shape: (K, 2, num_edges)
        edge_weights = np.stack([g.edge_weight for g in graphs], axis=0)  # Shape: (K, num_edges)
        window_starts = np.array([g.start_t for g in graphs], dtype=np.int32)
        window_ends = np.array([g.end_t for g in graphs], dtype=np.int32)
        thresholds = np.array([g.threshold_value for g in graphs], dtype=np.float32)

        out_fname = f"{subj.subject_id}.npz"
        out_fpath = os.path.join(graphs_dir, out_fname)

        np.savez_compressed(
            out_fpath,
            subject_id=subj.subject_id,
            site=subj.site,
            label=subj.label,
            num_windows=len(graphs),
            num_nodes=360,
            num_edges=graphs[0].num_edges if len(graphs) > 0 else 0,
            window_starts=window_starts,
            window_ends=window_ends,
            thresholds=thresholds,
            edge_indices=edge_indices,
            edge_weights=edge_weights
        )

        total_processed += 1
        total_windows_generated += len(graphs)
        total_bytes_written += os.path.getsize(out_fpath)

    elapsed = time.time() - start_time
    print("\n--- PREPROCESSING COMPLETED ---")
    print(f"[*] Processed subjects: {total_processed} / {len(all_subjects)}")
    print(f"[*] Total temporal graph windows generated: {total_windows_generated}")
    print(f"[*] Total disk storage consumed: {total_bytes_written / (1024**2):.2f} MB ({total_bytes_written / (1024**3):.3f} GB)")
    print(f"[*] Average file size per subject: {total_bytes_written / total_processed / 1024:.1f} KB")
    print(f"[*] Total processing time: {elapsed:.2f} seconds ({elapsed / total_processed:.3f} s/subject)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ABIDE temporal graph preprocessing pipeline.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects for testing")
    args = parser.parse_args()
    run_preprocessing_pipeline(args.config, args.max_subjects)
