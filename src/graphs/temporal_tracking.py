"""
Temporal Subgraph Tracking across Dynamic Functional Brain Network Sequences.

Supports two complementary temporal tracking formulations:
1. Strategy A: Dynamic k-hop membership (V_v^t changes dynamically at each window t)
2. Strategy B: Stable ROI set (V_v^ref defined from initial reference window; fixed ROI set across all windows)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from .subgraph_extraction import (
    CandidateSubgraph,
    extract_khop_neighborhood,
    extract_induced_subgraph,
)


@dataclass
class TemporalSubgraphSequence:
    subject_id: str
    seed_roi: int
    k_hop: int
    tracking_strategy: str  # 'dynamic' or 'stable'
    subgraphs: List[CandidateSubgraph]
    num_windows: int


def track_temporal_subgraph_dynamic(
    edge_indices: np.ndarray,
    edge_weights: np.ndarray,
    seed_roi: int,
    subject_id: str = "unknown",
    k: int = 1,
    num_total_nodes: int = 360
) -> TemporalSubgraphSequence:
    """
    Strategy A: Dynamic k-hop membership.
    At each window t, extracts the k-hop neighborhood from graph G_t.
    ROI membership evolves over time as functional connections form or dissolve.
    """
    K = edge_indices.shape[0]
    subgraphs = []

    for t in range(K):
        t_edge_index = edge_indices[t]
        t_edge_weight = edge_weights[t]

        # Extract k-hop neighborhood at current window t
        roi_nodes = extract_khop_neighborhood(
            edge_index=t_edge_index,
            seed_roi=seed_roi,
            k=k,
            num_nodes=num_total_nodes
        )
        sub = extract_induced_subgraph(
            roi_nodes=roi_nodes,
            edge_index=t_edge_index,
            edge_weight=t_edge_weight,
            seed_roi=seed_roi,
            window_idx=t,
            subject_id=subject_id,
            k_hop=k,
            num_total_nodes=num_total_nodes
        )
        subgraphs.append(sub)

    return TemporalSubgraphSequence(
        subject_id=subject_id,
        seed_roi=seed_roi,
        k_hop=k,
        tracking_strategy="dynamic",
        subgraphs=subgraphs,
        num_windows=K
    )


def track_temporal_subgraph_stable(
    edge_indices: np.ndarray,
    edge_weights: np.ndarray,
    seed_roi: int,
    subject_id: str = "unknown",
    k: int = 1,
    ref_window_idx: int = 0,
    num_total_nodes: int = 360
) -> TemporalSubgraphSequence:
    """
    Strategy B: Stable ROI set.
    Defines reference ROI set V_v^ref from reference window (default: initial window t=0).
    Keeps the exact same ROI identities fixed across all temporal windows t = 0..K-1.
    Internal edge presence and weights evolve dynamically according to graph G_t.
    """
    K = edge_indices.shape[0]
    if ref_window_idx >= K:
        raise ValueError(f"Reference window index {ref_window_idx} exceeds total windows {K}")

    # Define reference ROI set from window ref_window_idx (no future leakage: t=0)
    ref_edge_index = edge_indices[ref_window_idx]
    stable_roi_nodes = extract_khop_neighborhood(
        edge_index=ref_edge_index,
        seed_roi=seed_roi,
        k=k,
        num_nodes=num_total_nodes
    )

    subgraphs = []
    for t in range(K):
        t_edge_index = edge_indices[t]
        t_edge_weight = edge_weights[t]

        # Extract induced subgraph using the FIXED stable_roi_nodes
        sub = extract_induced_subgraph(
            roi_nodes=stable_roi_nodes,
            edge_index=t_edge_index,
            edge_weight=t_edge_weight,
            seed_roi=seed_roi,
            window_idx=t,
            subject_id=subject_id,
            k_hop=k,
            num_total_nodes=num_total_nodes
        )
        subgraphs.append(sub)

    return TemporalSubgraphSequence(
        subject_id=subject_id,
        seed_roi=seed_roi,
        k_hop=k,
        tracking_strategy="stable",
        subgraphs=subgraphs,
        num_windows=K
    )


def compute_subgraph_trajectory_metrics(sequence: TemporalSubgraphSequence) -> Dict[str, Any]:
    """
    Computes trajectory stability metrics across consecutive windows:
    - Node counts per window
    - Edge counts per window
    - Jaccard similarity between consecutive ROI sets: |V_t n V_{t+1}| / |V_t u V_{t+1}|
    - Edge weight changes between consecutive windows
    """
    subs = sequence.subgraphs
    K = len(subs)

    nodes_per_window = [s.num_nodes for s in subs]
    edges_per_window = [s.num_edges for s in subs]

    jaccard_consecutive = []
    edge_diff_consecutive = []

    for t in range(K - 1):
        s_curr = subs[t]
        s_next = subs[t + 1]

        # Jaccard similarity of ROI sets
        set_curr = set(s_curr.roi_indices)
        set_next = set(s_next.roi_indices)
        intersection = len(set_curr.intersection(set_next))
        union = len(set_curr.union(set_next))
        jaccard = intersection / union if union > 0 else 1.0
        jaccard_consecutive.append(jaccard)

        # Edge difference (matching edges present in both)
        # Create edge dictionary for current
        edges_curr = {
            (s_curr.roi_indices[s_curr.edge_index[0, e]], s_curr.roi_indices[s_curr.edge_index[1, e]]): s_curr.edge_weight[e]
            for e in range(s_curr.num_edges)
        }
        edges_next = {
            (s_next.roi_indices[s_next.edge_index[0, e]], s_next.roi_indices[s_next.edge_index[1, e]]): s_next.edge_weight[e]
            for e in range(s_next.num_edges)
        }
        all_edge_pairs = set(edges_curr.keys()).union(set(edges_next.keys()))
        diff_sq = sum((edges_curr.get(pair, 0.0) - edges_next.get(pair, 0.0)) ** 2 for pair in all_edge_pairs)
        edge_diff_consecutive.append(float(np.sqrt(diff_sq)))

    return {
        "subject_id": sequence.subject_id,
        "seed_roi": sequence.seed_roi,
        "tracking_strategy": sequence.tracking_strategy,
        "num_windows": K,
        "nodes_per_window": nodes_per_window,
        "edges_per_window": edges_per_window,
        "mean_nodes": float(np.mean(nodes_per_window)),
        "min_nodes": int(np.min(nodes_per_window)),
        "max_nodes": int(np.max(nodes_per_window)),
        "mean_edges": float(np.mean(edges_per_window)),
        "min_edges": int(np.min(edges_per_window)),
        "max_edges": int(np.max(edges_per_window)),
        "jaccard_consecutive": jaccard_consecutive,
        "mean_jaccard": float(np.mean(jaccard_consecutive)) if jaccard_consecutive else 1.0,
        "edge_diff_consecutive": edge_diff_consecutive,
        "mean_edge_diff": float(np.mean(edge_diff_consecutive)) if edge_diff_consecutive else 0.0
    }
