"""
Unit Tests for GIN Encoder (Milestone 3).
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

from src.graphs.subgraph_extraction import CandidateSubgraph
from src.models.gin_encoder import GINEncoder, subgraph_to_pyg


def create_dummy_candidate_subgraph(num_nodes=10, num_edges=15, seed=42):
    # Ensure directed edges in range [0, num_nodes-1]
    if num_edges > 0:
        edge_index = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int16)
        edge_weight = np.random.uniform(-1, 1, size=(num_edges,)).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int16)
        edge_weight = np.zeros((0,), dtype=np.float32)
        
    roi_indices = np.arange(num_nodes, dtype=np.int16)
    
    return CandidateSubgraph(
        subject_id="sub-001",
        seed_roi=seed,
        window_idx=0,
        k_hop=1,
        roi_indices=roi_indices,
        seed_local_idx=0,
        num_nodes=num_nodes,
        num_edges=num_edges,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

@pytest.fixture
def model():
    # Use deterministic initialization if needed, or simply standard
    torch.manual_seed(42)
    return GINEncoder(in_channels=1, hidden_channels=64, embedding_dim=128, num_layers=2)

# Test 1 - Output dimension for one valid subgraph
def test_output_dimension(model):
    subgraph = create_dummy_candidate_subgraph()
    data = subgraph_to_pyg(subgraph)
    
    out = model(data.x, data.edge_index, data.edge_weight)
    assert out.shape == (1, 128)

# Test 2 - Batch processing
def test_batch_processing(model):
    subgraphs = [create_dummy_candidate_subgraph() for _ in range(5)]
    data_list = [subgraph_to_pyg(sg) for sg in subgraphs]
    batch = Batch.from_data_list(data_list)
    
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    assert out.shape == (5, 128)

# Test 3 - Variable node counts
def test_variable_node_counts(model):
    s1 = create_dummy_candidate_subgraph(num_nodes=5, num_edges=10)
    s2 = create_dummy_candidate_subgraph(num_nodes=20, num_edges=40)
    
    batch = Batch.from_data_list([subgraph_to_pyg(s1), subgraph_to_pyg(s2)])
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    assert out.shape == (2, 128)

# Test 4 - Variable edge counts
def test_variable_edge_counts(model):
    s1 = create_dummy_candidate_subgraph(num_nodes=10, num_edges=0)  # zero edges
    s2 = create_dummy_candidate_subgraph(num_nodes=10, num_edges=30)
    
    batch = Batch.from_data_list([subgraph_to_pyg(s1), subgraph_to_pyg(s2)])
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    assert out.shape == (2, 128)

# Test 5 - Numerical stability
def test_numerical_stability(model):
    s1 = create_dummy_candidate_subgraph(num_nodes=50, num_edges=200)
    data = subgraph_to_pyg(s1)
    
    out = model(data.x, data.edge_index, data.edge_weight)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

# Test 6 - Shared parameters
def test_shared_parameters(model):
    model.eval()
    s1 = create_dummy_candidate_subgraph(seed=10)
    s2 = create_dummy_candidate_subgraph(seed=20)
    
    # Process independently
    out1 = model(subgraph_to_pyg(s1).x, subgraph_to_pyg(s1).edge_index, subgraph_to_pyg(s1).edge_weight)
    out2 = model(subgraph_to_pyg(s2).x, subgraph_to_pyg(s2).edge_index, subgraph_to_pyg(s2).edge_weight)
    
    # Process as batch
    batch = Batch.from_data_list([subgraph_to_pyg(s1), subgraph_to_pyg(s2)])
    out_batch = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    
    # The outputs should be exactly the same (or very close due to float precision)
    assert torch.allclose(out1, out_batch[0:1], atol=1e-5)
    assert torch.allclose(out2, out_batch[1:2], atol=1e-5)

# Test 7 - Seed ROI is not a node feature
def test_seed_roi_not_feature():
    s = create_dummy_candidate_subgraph(seed=42)
    data = subgraph_to_pyg(s)
    
    # Check that data.x has only constant features and 42 is not embedded in the data.x values
    assert data.x.shape[1] == 1
    assert torch.all(data.x == 1.0)
    assert data.seed_roi == 42
    
# Test 8 - Edge weights affect output
def test_edge_weights_affect_output(model):
    model.eval() # Ensure no dropout or batchnorm issues if any exist
    s = create_dummy_candidate_subgraph(num_nodes=5, num_edges=5)
    data = subgraph_to_pyg(s)
    
    # Graph 1 with positive weight 0.8
    data.edge_weight[:] = 0.8
    out1 = model(data.x, data.edge_index, data.edge_weight)
    
    # Graph 2 with positive weight 0.1
    data.edge_weight[:] = 0.1
    out2 = model(data.x, data.edge_index, data.edge_weight)
    
    # Graph 3 with negative weight -0.8
    data.edge_weight[:] = -0.8
    out3 = model(data.x, data.edge_index, data.edge_weight)
    
    # Ensure they are not identical
    assert not torch.allclose(out1, out2, atol=1e-5)
    assert not torch.allclose(out1, out3, atol=1e-5)

# Test 9 - Permutation invariance
def test_permutation_invariance(model):
    model.eval()
    
    # Base graph
    num_nodes = 8
    num_edges = 12
    edge_index = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int16)
    edge_weight = np.random.uniform(-1, 1, size=(num_edges,)).astype(np.float32)
    
    x = torch.ones((num_nodes, 1))
    e_idx = torch.tensor(edge_index, dtype=torch.long)
    e_wt = torch.tensor(edge_weight, dtype=torch.float32)
    
    out_original = model(x, e_idx, e_wt)
    
    # Permute nodes
    perm = torch.randperm(num_nodes)
    
    # Map old node -> new node (pi(v) = perm[v])
    new_edge_index = perm[e_idx]
    
    # Permuted graph representation
    out_permuted = model(x, new_edge_index, e_wt)
    
    assert torch.allclose(out_original, out_permuted, atol=1e-4)
