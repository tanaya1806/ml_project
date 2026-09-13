"""Unit tests for Phase 1B temporal embedding-sequence construction."""

import numpy as np
import torch

from src.models.gin_encoder import GINEncoder, subgraph_to_pyg
from src.graphs.subgraph_extraction import CandidateSubgraph
from src.models.embedding_sequence import seed_embedding_sequence


def _make_synthetic_cache(num_nodes=6, num_windows=4, seed=42):
    rng = np.random.default_rng(0)
    roi_indices = np.arange(num_nodes, dtype=np.int16)
    edge_index_seq = np.empty(num_windows, dtype=object)
    edge_weight_seq = np.empty(num_windows, dtype=object)
    for t in range(num_windows):
        num_edges = 8
        ei = rng.integers(0, num_nodes, size=(2, num_edges)).astype(np.int16)
        ew = rng.uniform(-1, 1, size=(num_edges,)).astype(np.float32)
        edge_index_seq[t] = ei
        edge_weight_seq[t] = ew
    return {
        "seed_rois": np.array([seed], dtype=np.int16),
        f"roi_indices_{seed}": roi_indices,
        f"edge_index_{seed}": edge_index_seq,
        f"edge_weight_{seed}": edge_weight_seq,
    }


def test_embedding_sequence_shape():
    torch.manual_seed(0)
    gin = GINEncoder(in_channels=1, hidden_channels=16, embedding_dim=32, num_layers=2)
    cache = _make_synthetic_cache(num_nodes=6, num_windows=5, seed=7)
    device = torch.device("cpu")

    z_seq = seed_embedding_sequence(gin, cache, seed=7, device=device)
    assert z_seq.shape == (5, 32)
    assert not torch.isnan(z_seq).any()


def test_embedding_sequence_matches_single_graph_forward():
    """Batched-per-window embeddings must equal calling the GIN on each window alone."""
    torch.manual_seed(0)
    gin = GINEncoder(in_channels=1, hidden_channels=16, embedding_dim=32, num_layers=2)
    gin.eval()
    cache = _make_synthetic_cache(num_nodes=6, num_windows=3, seed=7)
    device = torch.device("cpu")

    z_seq = seed_embedding_sequence(gin, cache, seed=7, device=device)

    roi_indices = cache["roi_indices_7"]
    num_nodes = len(roi_indices)
    for t in range(3):
        sub = CandidateSubgraph(
            subject_id="x", seed_roi=7, window_idx=t, k_hop=1,
            roi_indices=roi_indices, seed_local_idx=0, num_nodes=num_nodes,
            num_edges=cache["edge_index_7"][t].shape[1],
            edge_index=cache["edge_index_7"][t], edge_weight=cache["edge_weight_7"][t],
        )
        data = subgraph_to_pyg(sub)
        with torch.no_grad():
            z_single = gin(data.x, data.edge_index, data.edge_weight)
        assert torch.allclose(z_seq[t], z_single[0], atol=1e-5)
