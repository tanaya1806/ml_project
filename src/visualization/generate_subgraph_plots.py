"""
Milestone 2 Visualizations: Candidate k-Hop Subgraphs and Temporal Tracking.

Generates 6 validation figures:
1. Example parent graph with seed ROI highlighted
2. Example k-hop induced subgraph
3. Same seed subgraph across 3 consecutive windows (Window 0, 1, 2)
4. Distribution of subgraph node counts (from pilot cohort)
5. Distribution of subgraph edge counts (from pilot cohort)
6. Distribution of consecutive-window ROI-set Jaccard similarity (Dynamic vs Stable)
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from src.graphs.subgraph_extraction import (
    extract_khop_neighborhood,
    extract_induced_subgraph,
)
from src.graphs.temporal_tracking import (
    track_temporal_subgraph_dynamic,
    track_temporal_subgraph_stable,
    compute_subgraph_trajectory_metrics,
)


def generate_all_subgraph_plots(
    graph_dir: str = r"D:\Graph anomaly detection\data\processed\graphs",
    fig_dir: str = r"D:\Graph anomaly detection\outputs\figures",
    pilot_report_path: str = r"D:\Graph anomaly detection\data\processed\metadata\subgraph_pilot_report.json"
):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    print("================================================================================")
    print("           GENERATING MILESTONE 2 SUBGRAPH VALIDATION VISUALIZATIONS            ")
    print("================================================================================")

    # Load one representative subject: NYU or CALTECH
    files = sorted(glob.glob(os.path.join(graph_dir, "*.npz")))
    rep_file = files[0]
    data = np.load(rep_file)
    subj_id = str(data["subject_id"])
    edge_indices = data["edge_indices"]  # (K, 2, 25848)
    edge_weights = data["edge_weights"]  # (K, 25848)

    seed_roi = 42
    k = 1

    # -------------------------------------------------------------------------
    # 1. Example parent graph with seed ROI highlighted
    # -------------------------------------------------------------------------
    print("[1/6] Generating: 08_parent_graph_seed_highlight.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    w0_edge_index = edge_indices[0]
    w0_rois = extract_khop_neighborhood(w0_edge_index, seed_roi=seed_roi, k=k)
    
    # Circular layout for all 360 nodes
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(360)}

    # Subsample 2% of background edges for visual clarity
    bg_mask = np.random.rand(w0_edge_index.shape[1]) < 0.02
    bg_src = w0_edge_index[0, bg_mask]
    bg_dst = w0_edge_index[1, bg_mask]
    for u, v in zip(bg_src, bg_dst):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="lightgray", alpha=0.3, lw=0.5, zorder=1)

    # Plot incident edges to seed ROI
    seed_edges_mask = (w0_edge_index[0] == seed_roi)
    for v in w0_edge_index[1, seed_edges_mask]:
        ax.plot([pos[seed_roi][0], pos[v][0]], [pos[seed_roi][1], pos[v][1]], color="crimson", alpha=0.7, lw=1.2, zorder=2)

    # Draw nodes
    other_nodes = [i for i in range(360) if i != seed_roi and i not in w0_rois]
    ax.scatter([pos[i][0] for i in other_nodes], [pos[i][1] for i in other_nodes], color="gainsboro", s=15, label="Other ROIs", zorder=3)
    ax.scatter([pos[i][0] for i in w0_rois if i != seed_roi], [pos[i][1] for i in w0_rois if i != seed_roi], color="dodgerblue", s=40, label=f"1-Hop Neighbors ({len(w0_rois)-1})", zorder=4)
    ax.scatter([pos[seed_roi][0]], [pos[seed_roi][1]], color="crimson", s=140, marker="*", edgecolor="black", label=f"Seed ROI {seed_roi}", zorder=5)

    ax.set_title(f"Parent Brain Network (360 ROIs, Top 20% FC)\nSeed ROI {seed_roi} and its 1-Hop Neighborhood (Subject {subj_id})", fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_parent_graph_seed_highlight.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Example k-hop induced subgraph
    # -------------------------------------------------------------------------
    print("[2/6] Generating: 09_khop_induced_subgraph.png")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    sub = extract_induced_subgraph(
        roi_nodes=w0_rois,
        edge_index=w0_edge_index,
        edge_weight=edge_weights[0],
        seed_roi=seed_roi,
        window_idx=0,
        subject_id=subj_id,
        k_hop=k
    )

    G = nx.Graph()
    for i, roi in enumerate(sub.roi_indices):
        G.add_node(i, global_id=roi, is_seed=(roi == seed_roi))

    for e in range(sub.num_edges):
        u, v = int(sub.edge_index[0, e]), int(sub.edge_index[1, e])
        w = float(sub.edge_weight[e])
        if u < v:
            G.add_edge(u, v, weight=w)

    pos_sub = nx.spring_layout(G, seed=42, k=0.35)
    edge_weights_list = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = ["red" if w > 0 else "blue" for w in edge_weights_list]
    edge_widths = [0.5 + 2.5 * abs(w) for w in edge_weights_list]

    nx.draw_networkx_edges(G, pos_sub, alpha=0.35, edge_color=edge_colors, width=edge_widths, ax=ax)
    node_colors = ["crimson" if G.nodes[n]["is_seed"] else "skyblue" for n in G.nodes()]
    node_sizes = [220 if G.nodes[n]["is_seed"] else 45 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos_sub, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.5, ax=ax)
    
    # Label seed
    seed_loc = sub.seed_local_idx
    ax.text(pos_sub[seed_loc][0], pos_sub[seed_loc][1] + 0.08, f"Seed {seed_roi}", fontsize=11, fontweight="bold", color="black", ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7))

    ax.set_title(f"Induced 1-Hop Subgraph S_{seed_roi}^0 (Window 0)\n|V| = {sub.num_nodes} ROIs | |E| = {sub.num_edges//2} Undirected Edges", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_khop_induced_subgraph.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Same seed subgraph across 3 consecutive windows
    # -------------------------------------------------------------------------
    print("[3/6] Generating: 10_consecutive_subgraphs_tracking.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

    seq_stable = track_temporal_subgraph_stable(edge_indices, edge_weights, seed_roi=seed_roi, subject_id=subj_id, k=k, ref_window_idx=0)
    
    # Use fixed spring layout based on window 0 for direct visual comparability
    for t_idx, ax in enumerate(axes):
        st = seq_stable.subgraphs[t_idx]
        Gt = nx.Graph()
        for i, roi in enumerate(st.roi_indices):
            Gt.add_node(i, is_seed=(roi == seed_roi))
        for e in range(st.num_edges):
            u, v = int(st.edge_index[0, e]), int(st.edge_index[1, e])
            w = float(st.edge_weight[e])
            if u < v:
                Gt.add_edge(u, v, weight=w)

        e_weights = [Gt[u][v]["weight"] for u, v in Gt.edges()]
        e_colors = ["crimson" if w > 0 else "royalblue" for w in e_weights]
        e_widths = [0.4 + 2.0 * abs(w) for w in e_weights]

        nx.draw_networkx_edges(Gt, pos_sub, alpha=0.35, edge_color=e_colors, width=e_widths, ax=ax)
        nx.draw_networkx_nodes(Gt, pos_sub, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.5, ax=ax)
        ax.set_title(f"Window {t_idx} (t={t_idx*10}..{t_idx*10+30})\n|V|={st.num_nodes}, |E|={st.num_edges//2}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Temporal Subgraph Tracking: Seed ROI {seed_roi} Across Consecutive Windows\n(Stable ROI Identity: Same {len(sub.roi_indices)} ROIs, Evolving Topology)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "10_consecutive_subgraphs_tracking.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4 & 5. Node and Edge Count Distributions from Pilot
    # -------------------------------------------------------------------------
    print("[4/6] Generating: 11_subgraph_node_count_distribution.png")
    print("[5/6] Generating: 12_subgraph_edge_count_distribution.png")
    
    # Load pilot data if available, or sample from current file
    node_counts = []
    edge_counts = []
    jaccards_dyn = []

    for seed in range(360):
        seq_d = track_temporal_subgraph_dynamic(edge_indices, edge_weights, seed_roi=seed, subject_id=subj_id, k=k)
        met = compute_subgraph_trajectory_metrics(seq_d)
        node_counts.extend(met["nodes_per_window"])
        edge_counts.extend([e // 2 for e in met["edges_per_window"]])  # Undirected edges
        jaccards_dyn.extend(met["jaccard_consecutive"])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.histplot(node_counts, bins=40, color="teal", kde=True, ax=ax)
    ax.axvline(np.mean(node_counts), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(node_counts):.1f}")
    ax.set_title(f"Distribution of Subgraph Node Counts (|V|) Across 360 Seeds (k={k})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Nodes in 1-Hop Subgraph", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "11_subgraph_node_count_distribution.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.histplot(edge_counts, bins=40, color="indigo", kde=True, ax=ax)
    ax.axvline(np.mean(edge_counts), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(edge_counts):.1f}")
    ax.set_title(f"Distribution of Subgraph Undirected Edge Counts (|E|) Across 360 Seeds (k={k})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Undirected Edges in 1-Hop Subgraph", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "12_subgraph_edge_count_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Consecutive-Window ROI-Set Jaccard Similarity Distribution
    # -------------------------------------------------------------------------
    print("[6/6] Generating: 13_consecutive_jaccard_similarity.png")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.histplot(jaccards_dyn, bins=35, color="coral", kde=True, stat="density", label="Strategy A (Dynamic Membership)", ax=ax)
    ax.axvline(np.mean(jaccards_dyn), color="darkred", linestyle="--", lw=2, label=f"Mean Dynamic Jaccard: {np.mean(jaccards_dyn):.3f}")
    ax.axvline(1.0, color="forestgreen", linestyle="-", lw=3, label="Strategy B (Stable ROI Set: Jaccard = 1.0)")
    ax.set_title("Consecutive-Window ROI-Set Jaccard Similarity J(V_t, V_t+1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Jaccard Similarity |V_t ∩ V_{t+1}| / |V_t ∪ V_{t+1}|", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0.0, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "13_consecutive_jaccard_similarity.png"))
    plt.close(fig)

    print(f"\n[*] All 6 Milestone 2 figures successfully saved to: {fig_dir}")


if __name__ == "__main__":
    generate_all_subgraph_plots()
