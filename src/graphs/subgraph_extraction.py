"""
Candidate k-Hop Subgraph Extraction for Brain Networks.

Extracts localized, induced k-hop neighborhoods around candidate seed ROIs
from temporal graph windows, preserving signed edge weights and canonical local reindexing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import numpy as np


@dataclass
class CandidateSubgraph:
    subject_id: str
    seed_roi: int            # Global ROI ID in [0, 359]
    window_idx: int          # Temporal window index t
    k_hop: int               # Hop radius (e.g. 1 or 2)
    roi_indices: np.ndarray  # Sorted global ROI IDs, shape: (num_nodes,), dtype: int16
    seed_local_idx: int      # Local index of seed_roi within roi_indices
    num_nodes: int           # |V_sub|
    num_edges: int           # |E_sub| (directed edges)
    edge_index: np.ndarray   # Local reindexed edges in [0, num_nodes-1], shape: (2, num_edges), dtype: int16
    edge_weight: np.ndarray  # Signed Pearson FC edge weights, shape: (num_edges,), dtype: float32
    global_edge_index: Optional[np.ndarray] = None  # Original global node IDs, shape: (2, num_edges), dtype: int16


def extract_khop_neighborhood(
    edge_index: np.ndarray,
    seed_roi: int,
    k: int = 1,
    num_nodes: int = 360
) -> np.ndarray:
    """
    Finds the set of unique ROI IDs within k hops of seed_roi in the given graph.
    The seed_roi itself is always included.
    
    Args:
        edge_index: np.ndarray of shape (2, E), global directed edge indices
        seed_roi: Global seed ROI integer in [0, num_nodes - 1]
        k: Hop distance (default: 1)
        num_nodes: Total nodes in parent graph (default: 360)
        
    Returns:
        Sorted np.ndarray of unique ROI IDs in the k-hop neighborhood, dtype np.int16
    """
    if not (0 <= seed_roi < num_nodes):
        raise ValueError(f"Seed ROI {seed_roi} is out of bounds [0, {num_nodes - 1}]")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Build adjacency list representation for fast traversal
    src, dst = edge_index[0], edge_index[1]
    
    # Using a fast BFS
    visited = {seed_roi}
    frontier = {seed_roi}

    # Pre-group neighbors by source node using array grouping or dict of sets
    # Since num_nodes is 360, this is very compact
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(int(v))

    for _ in range(k):
        next_frontier = set()
        for u in frontier:
            for neighbor in adj[u]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return np.array(sorted(visited), dtype=np.int16)


def extract_induced_subgraph(
    roi_nodes: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    seed_roi: int,
    window_idx: int,
    subject_id: str = "unknown",
    k_hop: int = 1,
    store_global_edges: bool = False,
    num_total_nodes: int = 360
) -> CandidateSubgraph:
    """
    Extracts the induced subgraph over the specified ROI node set from the parent graph.
    Reindexes node IDs locally to [0, |V_sub| - 1] for GNN compatibility.
    
    Args:
        roi_nodes: 1D np.ndarray of global node IDs forming the subgraph
        edge_index: Parent graph edge index (2, E)
        edge_weight: Parent graph edge weights (E,)
        seed_roi: Seed ROI integer
        window_idx: Window index
        subject_id: Identifier string
        k_hop: Hop distance parameter
        store_global_edges: Whether to include original global edge_index
        num_total_nodes: Total nodes in parent graph (360)
        
    Returns:
        CandidateSubgraph dataclass instance
    """
    # Ensure sorted unique node list
    unique_rois = np.unique(roi_nodes).astype(np.int16)
    num_sub_nodes = len(unique_rois)

    # Fast membership mask over total nodes
    in_subgraph = np.zeros(num_total_nodes, dtype=bool)
    in_subgraph[unique_rois] = True

    # Find edges where BOTH source and target are inside the subgraph
    src = edge_index[0]
    dst = edge_index[1]
    edge_mask = in_subgraph[src] & in_subgraph[dst]

    sub_src_global = src[edge_mask]
    sub_dst_global = dst[edge_mask]
    sub_weights = edge_weight[edge_mask]

    # Map global node IDs to local indices [0, num_sub_nodes - 1]
    local_map = np.full(num_total_nodes, -1, dtype=np.int16)
    local_map[unique_rois] = np.arange(num_sub_nodes, dtype=np.int16)

    sub_src_local = local_map[sub_src_global]
    sub_dst_local = local_map[sub_dst_global]
    local_edge_index = np.vstack([sub_src_local, sub_dst_local])

    # Find local index of the seed node
    seed_pos = np.where(unique_rois == seed_roi)[0]
    if len(seed_pos) == 0:
        raise ValueError(f"Seed ROI {seed_roi} is not present in the extracted ROI node set!")
    seed_local_idx = int(seed_pos[0])

    global_edges = np.vstack([sub_src_global, sub_dst_global]) if store_global_edges else None

    return CandidateSubgraph(
        subject_id=subject_id,
        seed_roi=seed_roi,
        window_idx=window_idx,
        k_hop=k_hop,
        roi_indices=unique_rois,
        seed_local_idx=seed_local_idx,
        num_nodes=num_sub_nodes,
        num_edges=len(sub_weights),
        edge_index=local_edge_index,
        edge_weight=sub_weights.astype(np.float32),
        global_edge_index=global_edges
    )


def extract_all_seed_subgraphs_for_window(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    window_idx: int,
    subject_id: str,
    k: int = 1,
    seed_rois: Optional[List[int]] = None,
    num_total_nodes: int = 360
) -> List[CandidateSubgraph]:
    """
    Extracts k-hop candidate subgraphs for all specified seed ROIs in a single window.
    If seed_rois is None, uses all ROIs [0, ..., num_total_nodes - 1].
    """
    if seed_rois is None:
        seed_rois = list(range(num_total_nodes))

    subgraphs = []
    for seed in seed_rois:
        roi_nodes = extract_khop_neighborhood(
            edge_index=edge_index,
            seed_roi=seed,
            k=k,
            num_nodes=num_total_nodes
        )
        sub = extract_induced_subgraph(
            roi_nodes=roi_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            seed_roi=seed,
            window_idx=window_idx,
            subject_id=subject_id,
            k_hop=k,
            num_total_nodes=num_total_nodes
        )
        subgraphs.append(sub)

    return subgraphs
