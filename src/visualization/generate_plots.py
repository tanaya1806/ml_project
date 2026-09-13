"""
Validation Visualizations for ABIDE Temporal Graph Sequences.

Generates 7 core sanity-check figures:
1. One subject's ROI time-series snippet
2. One window FC heatmap
3. Consecutive window FC heatmaps (Window t vs Window t+1)
4. FC difference heatmap: FC_(t+1) - FC_t
5. Distribution of edge weights (dense vs sparsified)
6. Distribution of number of temporal windows per subject across sites
7. Subject counts by site and diagnosis
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import ABIDELoader
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs


def generate_all_visualizations(config_path: str = "configs/baseline.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_ts_dir = cfg["dataset"]["ts_dir"]
    fig_dir = cfg["dataset"]["figures_dir"]
    window_size = cfg["preprocessing"]["window_size"]
    stride = cfg["preprocessing"]["stride"]
    sparsity_ratio = cfg["graph_construction"]["sparsity_ratio"]

    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    print("================================================================================")
    print("                 GENERATING VALIDATION VISUALIZATIONS                           ")
    print("================================================================================")

    loader = ABIDELoader(raw_ts_dir)
    summary_df = loader.get_summary_df()
    summary_df["num_windows"] = (summary_df["timepoints"] - window_size) // stride + 1

    # Select representative subject (e.g. from NYU or PITT with uniform 176 or 196 timepoints)
    nyu_subjs = loader.load_site("NYU")
    sample_subj = nyu_subjs[0]  # Control subject
    ts = sample_subj.time_series
    wfcs = compute_sliding_window_fc(ts, window_size=window_size, stride=stride)
    graphs = construct_temporal_graphs(wfcs, sparsity_ratio=sparsity_ratio)

    # -------------------------------------------------------------------------
    # 1. One subject's ROI time-series snippet
    # -------------------------------------------------------------------------
    print("[1/7] Generating: 01_roi_time_series.png")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    # Plot first 10 ROIs across all timepoints
    rois_to_plot = 10
    time_axis = np.arange(ts.shape[0])
    for roi_idx in range(rois_to_plot):
        ax.plot(time_axis, ts[:, roi_idx] + roi_idx * 3.0, label=f"ROI {roi_idx+1}", alpha=0.85, lw=1.2)
    # Shade first two windows
    ax.axvspan(0, window_size, color="tab:blue", alpha=0.15, label="Window 0 (t=0..30)")
    ax.axvspan(stride, stride + window_size, color="tab:orange", alpha=0.15, label="Window 1 (t=10..40)")
    ax.set_title(f"Subject {sample_subj.subject_id} ({sample_subj.site}) - First 10 ROIs rs-fMRI Time Series", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timepoints (TRs)", fontsize=11)
    ax.set_ylabel("Normalized BOLD (Stacked)", fontsize=11)
    ax.set_xlim(0, ts.shape[0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_roi_time_series.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. One FC heatmap
    # -------------------------------------------------------------------------
    print("[2/7] Generating: 02_single_window_fc.png")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    w0_mat = wfcs[0].fc_matrix
    im = ax.imshow(w0_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation (r)", fontsize=11)
    ax.set_title(f"Functional Connectivity: Window 0 (t={wfcs[0].start_t}..{wfcs[0].end_t})\n360 ROIs | Subject {sample_subj.subject_id}", fontsize=12, fontweight="bold")
    ax.set_xlabel("ROI Index", fontsize=11)
    ax.set_ylabel("ROI Index", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_single_window_fc.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Consecutive FC heatmaps (Window t vs Window t+1)
    # -------------------------------------------------------------------------
    print("[3/7] Generating: 03_consecutive_fc_heatmaps.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)
    w0_fc = wfcs[0].fc_matrix
    w1_fc = wfcs[1].fc_matrix

    im1 = ax1.imshow(w0_fc, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax1.set_title(f"Window 0 (t={wfcs[0].start_t}..{wfcs[0].end_t})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("ROI Index")
    ax1.set_ylabel("ROI Index")

    im2 = ax2.imshow(w1_fc, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax2.set_title(f"Window 1 (t={wfcs[1].start_t}..{wfcs[1].end_t})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("ROI Index")
    ax2.set_ylabel("ROI Index")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label("Pearson Correlation (r)", fontsize=11)
    plt.suptitle(f"Consecutive Dynamic FC Windows (Stride = {stride} TRs)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "03_consecutive_fc_heatmaps.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. FC difference heatmap: FC_(t+1) - FC_t
    # -------------------------------------------------------------------------
    print("[4/7] Generating: 04_fc_difference_heatmap.png")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    fc_diff = w1_fc - w0_fc
    max_abs_diff = max(float(np.abs(np.min(fc_diff))), float(np.max(fc_diff)))
    im = ax.imshow(fc_diff, cmap="bwr", vmin=-max_abs_diff, vmax=max_abs_diff, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FC Change: FC(t+1) - FC(t)", fontsize=11)
    frob_norm = np.linalg.norm(fc_diff, 'fro')
    ax.set_title(f"Temporal Dynamics: FC(1) - FC(0)\n||Delta FC||_F = {frob_norm:.2f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("ROI Index", fontsize=11)
    ax.set_ylabel("ROI Index", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_fc_difference_heatmap.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. Distribution of edge weights
    # -------------------------------------------------------------------------
    print("[5/7] Generating: 05_edge_weight_distributions.png")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    dense_weights = w0_fc[np.triu_indices(360, k=1)]
    sparse_weights = graphs[0].edge_weight
    # sparse_weights contains both (u, v) and (v, u), take unique positive half for plotting
    thresh = graphs[0].threshold_value

    sns.histplot(dense_weights, bins=80, color="gray", alpha=0.45, stat="density", label="Full Dense FC (64,620 pairs)", ax=ax)
    sns.histplot(sparse_weights, bins=80, color="crimson", alpha=0.6, stat="density", label=f"Sparsified Top 20% (|r| >= {thresh:.3f})", ax=ax)
    ax.axvline(thresh, color="black", linestyle="--", lw=1.5, label=f"+Threshold: {thresh:.3f}")
    ax.axvline(-thresh, color="black", linestyle="--", lw=1.5, label=f"-Threshold: -{thresh:.3f}")
    ax.set_title("Edge Weight Distribution: Dense vs Top 20% Sparsified", fontsize=13, fontweight="bold")
    ax.set_xlabel("Signed Pearson Correlation (r)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(-1.0, 1.0)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "05_edge_weight_distributions.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Distribution of number of windows per subject
    # -------------------------------------------------------------------------
    print("[6/7] Generating: 06_windows_per_subject_distribution.png")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    sns.countplot(data=summary_df, x="num_windows", hue="diagnosis", palette={"Control": "tab:blue", "ASD": "tab:orange"}, ax=ax)
    ax.set_title(f"Distribution of Temporal Windows per Subject (window_size={window_size}, stride={stride})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Windows", fontsize=11)
    ax.set_ylabel("Subject Count", fontsize=11)
    ax.legend(title="Diagnosis", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_windows_per_subject_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 7. Subject counts by site and diagnosis
    # -------------------------------------------------------------------------
    print("[7/7] Generating: 07_subject_counts_by_site.png")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    site_order = summary_df["site"].value_counts().index
    sns.countplot(data=summary_df, x="site", hue="diagnosis", order=site_order, palette={"Control": "tab:blue", "ASD": "tab:orange"}, ax=ax)
    ax.set_title("ABIDE I Preprocessed Subject Counts by Site and Diagnosis (Total N=871: 468 TC, 403 ASD)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scanning Site", fontsize=11)
    ax.set_ylabel("Number of Subjects", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Diagnosis", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_subject_counts_by_site.png"))
    plt.close(fig)

    print(f"\n[*] All 7 validation figures saved to: {fig_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
