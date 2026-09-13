"""
Temporal Graph Construction & Deterministic Sparsification.

Converts windowed Functional Connectivity (FC) matrices into sparse,
weighted, undirected graph sequences:
G_t = (V, E_t), |V| = 360 ROIs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TemporalGraph:
    window_idx: int
    start_t: int
    end_t: int
    num_nodes: int
    num_edges: int  # Total directed edges (2 * undirected edges)
    edge_index: np.ndarray  # Shape: (2, num_edges), dtype: np.int16
    edge_weight: np.ndarray  # Shape: (num_edges,), dtype: np.float32 (signed Pearson r)
    threshold_value: float


def sparsify_fc_matrix(
    fc_matrix: np.ndarray,
    sparsity_ratio: float = 0.20,
    keep_signed_weights: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sparsifies a 360x360 symmetric FC matrix by selecting the top-k strongest
    connections based on absolute correlation magnitude, while preserving the
    original signed correlation values as edge weights.
    
    Args:
        fc_matrix: np.ndarray of shape (N, N), assumed symmetric with 0.0 diagonal
        sparsity_ratio: Fraction of top connections to keep (default: 0.20 for top 20%)
        keep_signed_weights: If True, edge_weight is the signed correlation r;
                             if False, absolute value |r| is used.
                             
    Returns:
        edge_index: np.ndarray of shape (2, 2 * k), dtype np.int16
        edge_weight: np.ndarray of shape (2 * k,), dtype np.float32
        threshold_value: Absolute correlation cutoff value
    """
    N = fc_matrix.shape[0]
    # Get upper triangular indices excluding diagonal
    triu_i, triu_j = np.triu_indices(N, k=1)
    
    # Extract edge weights for all candidate off-diagonal pairs
    triu_weights = fc_matrix[triu_i, triu_j]
    abs_weights = np.abs(triu_weights)
    
    total_pairs = len(triu_weights)  # 360 * 359 / 2 = 64,620
    k = int(np.floor(total_pairs * sparsity_ratio))
    
    if k == 0:
        empty_index = np.zeros((2, 0), dtype=np.int16)
        empty_weight = np.zeros((0,), dtype=np.float32)
        return empty_index, empty_weight, 0.0

    # Deterministic top-k selection using argpartition + sort
    # argpartition finds the k largest elements in O(M) time
    partition_idx = np.argpartition(abs_weights, -k)[-k:]
    # Sort the top-k indices descending by absolute magnitude for exact determinism
    top_k_sorted = partition_idx[np.argsort(-abs_weights[partition_idx])]
    
    threshold_val = float(abs_weights[top_k_sorted[-1]])
    
    selected_i = triu_i[top_k_sorted]
    selected_j = triu_j[top_k_sorted]
    
    if keep_signed_weights:
        selected_w = triu_weights[top_k_sorted]
    else:
        selected_w = abs_weights[top_k_sorted]
        
    # Create symmetric undirected graph representation: include (u, v) and (v, u)
    src = np.concatenate([selected_i, selected_j]).astype(np.int16)
    dst = np.concatenate([selected_j, selected_i]).astype(np.int16)
    weights = np.concatenate([selected_w, selected_w]).astype(np.float32)
    
    # Sort edges primarily by src, secondarily by dst for canonical graph indexing
    sort_order = np.lexsort((dst, src))
    edge_index = np.vstack([src[sort_order], dst[sort_order]])
    edge_weight = weights[sort_order]
    
    return edge_index, edge_weight, threshold_val


def construct_temporal_graphs(
    window_fcs: list,
    sparsity_ratio: float = 0.20,
    keep_signed_weights: bool = True
) -> List[TemporalGraph]:
    """
    Constructs a sequence of sparse TemporalGraph objects from a sequence of WindowFC matrices.
    """
    graph_sequence = []
    for w in window_fcs:
        edge_index, edge_weight, thresh = sparsify_fc_matrix(
            w.fc_matrix,
            sparsity_ratio=sparsity_ratio,
            keep_signed_weights=keep_signed_weights
        )
        graph = TemporalGraph(
            window_idx=w.window_idx,
            start_t=w.start_t,
            end_t=w.end_t,
            num_nodes=w.fc_matrix.shape[0],
            num_edges=edge_index.shape[1],
            edge_index=edge_index,
            edge_weight=edge_weight,
            threshold_value=thresh
        )
        graph_sequence.append(graph)
    return graph_sequence
