"""
Phase 1B: Temporal Sequence Construction.

For one subject and one seed ROI, turns the cached stable-subgraph sequence
G_1[S_v], ..., G_T[S_v] into an embedding sequence z_1, ..., z_T (z_t in R^128)
by running the *same* GIN parameters (M3) over every window, batched in a
single forward pass. The same GIN is shared across subjects, seeds, and windows
-- no per-ROI models.
"""

from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.models.gin_encoder import GINEncoder


def _window_to_data(num_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray) -> Data:
    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    ei = torch.tensor(np.asarray(edge_index, dtype=np.int64), dtype=torch.long)
    ew = torch.tensor(np.asarray(edge_weight, dtype=np.float32), dtype=torch.float32)
    return Data(x=x, edge_index=ei, edge_weight=ew)


def seed_embedding_sequence(
    gin: GINEncoder,
    cache: Dict,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns z_seq of shape [T, embedding_dim] for the given subject cache and seed ROI.
    """
    roi_indices = cache[f"roi_indices_{seed}"]
    num_nodes = len(roi_indices)
    edge_index_seq = cache[f"edge_index_{seed}"]
    edge_weight_seq = cache[f"edge_weight_{seed}"]
    T = len(edge_index_seq)

    data_list = [
        _window_to_data(num_nodes, edge_index_seq[t], edge_weight_seq[t])
        for t in range(T)
    ]
    batch = Batch.from_data_list(data_list).to(device)
    z_seq = gin(batch.x, batch.edge_index, batch.edge_weight, batch.batch)  # [T, D]
    return z_seq
