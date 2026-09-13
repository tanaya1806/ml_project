"""
Unit Tests for Candidate k-Hop Subgraph Extraction and Temporal Tracking.
"""

import os
import glob
import yaml
import pytest
import numpy as np

from src.graphs.subgraph_extraction import (
    extract_khop_neighborhood,
    extract_induced_subgraph,
    extract_all_seed_subgraphs_for_window,
    CandidateSubgraph,
)
from src.graphs.temporal_tracking import (
    track_temporal_subgraph_dynamic,
    track_temporal_subgraph_stable,
    TemporalSubgraphSequence,
    compute_subgraph_trajectory_metrics,
)

_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))
GRAPH_DIR = _CFG["dataset"]["graphs_dir"]


@pytest.fixture(scope="module")
def sample_graph_data():
    """Load one sample subject's processed temporal graph sequence."""
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.npz")))
    if not files:
        pytest.skip(f"No processed graph files found in {GRAPH_DIR}")
    data = np.load(files[0])
    return {
        "subject_id": str(data["subject_id"]),
        "num_windows": int(data["num_windows"]),
        "edge_indices": data["edge_indices"],  # (K, 2, 25848)
        "edge_weights": data["edge_weights"],  # (K, 25848)
    }


# 1. Test k-hop neighborhood extraction bounds and presence
def test_khop_neighborhood_extraction(sample_graph_data):
    edge_index = sample_graph_data["edge_indices"][0]
    
    # Test valid seeds
    for seed in [0, 42, 180, 359]:
        rois = extract_khop_neighborhood(edge_index, seed_roi=seed, k=1, num_nodes=360)
        assert len(rois) > 1, f"Seed {seed} has no neighbors!"
        assert seed in rois, f"Seed {seed} not in its own neighborhood!"
        assert np.all(rois >= 0) and np.all(rois < 360), "ROI index out of bounds!"
        assert len(rois) == len(set(rois)), "Duplicate ROI IDs in neighborhood!"

    # Out of bounds seed should raise ValueError
    with pytest.raises(ValueError):
        extract_khop_neighborhood(edge_index, seed_roi=-1, k=1, num_nodes=360)
    with pytest.raises(ValueError):
        extract_khop_neighborhood(edge_index, seed_roi=360, k=1, num_nodes=360)


# 2. Test induced subgraph properties and local reindexing
def test_induced_subgraph_properties(sample_graph_data):
    edge_index = sample_graph_data["edge_indices"][0]
    edge_weight = sample_graph_data["edge_weights"][0]
    seed = 42

    rois = extract_khop_neighborhood(edge_index, seed_roi=seed, k=1, num_nodes=360)
    sub = extract_induced_subgraph(
        roi_nodes=rois,
        edge_index=edge_index,
        edge_weight=edge_weight,
        seed_roi=seed,
        window_idx=0,
        subject_id=sample_graph_data["subject_id"],
        k_hop=1
    )

    assert isinstance(sub, CandidateSubgraph)
    assert sub.num_nodes == len(rois)
    assert sub.seed_roi == seed
    assert sub.roi_indices[sub.seed_local_idx] == seed
    assert sub.edge_index.shape[0] == 2
    assert sub.edge_index.shape[1] == sub.num_edges
    assert len(sub.edge_weight) == sub.num_edges

    # Local edge indices strictly within [0, num_nodes - 1]
    if sub.num_edges > 0:
        assert np.all(sub.edge_index >= 0)
        assert np.all(sub.edge_index < sub.num_nodes)
        # Edge weights correspond directly to parent graph weights
        assert np.all(sub.edge_weight >= -1.0) and np.all(sub.edge_weight <= 1.0)


# 3. Test edge weight correspondence with parent graph
def test_edge_weight_correspondence(sample_graph_data):
    edge_index = sample_graph_data["edge_indices"][0]
    edge_weight = sample_graph_data["edge_weights"][0]
    seed = 100

    # Build dictionary of parent graph edges
    parent_edges = {}
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        parent_edges[(u, v)] = float(edge_weight[e])

    rois = extract_khop_neighborhood(edge_index, seed_roi=seed, k=1)
    sub = extract_induced_subgraph(rois, edge_index, edge_weight, seed_roi=seed, window_idx=0)

    # Check each edge in subgraph matches parent graph edge
    for e in range(sub.num_edges):
        loc_u, loc_v = int(sub.edge_index[0, e]), int(sub.edge_index[1, e])
        glob_u = int(sub.roi_indices[loc_u])
        glob_v = int(sub.roi_indices[loc_v])
        sub_w = float(sub.edge_weight[e])

        assert (glob_u, glob_v) in parent_edges, f"Subgraph edge ({glob_u}, {glob_v}) not found in parent graph!"
        assert np.isclose(parent_edges[(glob_u, glob_v)], sub_w, atol=1e-6), "Edge weight mismatch between subgraph and parent graph!"


# 4. Test dynamic temporal tracking (Strategy A)
def test_temporal_tracking_dynamic(sample_graph_data):
    seq = track_temporal_subgraph_dynamic(
        edge_indices=sample_graph_data["edge_indices"],
        edge_weights=sample_graph_data["edge_weights"],
        seed_roi=42,
        subject_id=sample_graph_data["subject_id"],
        k=1
    )

    assert isinstance(seq, TemporalSubgraphSequence)
    assert seq.tracking_strategy == "dynamic"
    assert seq.num_windows == sample_graph_data["num_windows"]
    assert len(seq.subgraphs) == sample_graph_data["num_windows"]

    # Temporally ordered and correct subject_id
    for t, sub in enumerate(seq.subgraphs):
        assert sub.window_idx == t
        assert sub.seed_roi == 42
        assert sub.subject_id == sample_graph_data["subject_id"]
        assert 42 in sub.roi_indices

    # Compute metrics
    metrics = compute_subgraph_trajectory_metrics(seq)
    assert metrics["num_windows"] == sample_graph_data["num_windows"]
    assert len(metrics["jaccard_consecutive"]) == sample_graph_data["num_windows"] - 1
    for jacc in metrics["jaccard_consecutive"]:
        assert 0.0 <= jacc <= 1.0


# 5. Test stable temporal tracking (Strategy B)
def test_temporal_tracking_stable(sample_graph_data):
    seq = track_temporal_subgraph_stable(
        edge_indices=sample_graph_data["edge_indices"],
        edge_weights=sample_graph_data["edge_weights"],
        seed_roi=42,
        subject_id=sample_graph_data["subject_id"],
        k=1,
        ref_window_idx=0
    )

    assert isinstance(seq, TemporalSubgraphSequence)
    assert seq.tracking_strategy == "stable"
    assert seq.num_windows == sample_graph_data["num_windows"]

    # In stable tracking, the ROI set must be strictly identical across all windows
    ref_rois = seq.subgraphs[0].roi_indices
    for sub in seq.subgraphs:
        assert np.array_equal(sub.roi_indices, ref_rois)
        assert sub.num_nodes == len(ref_rois)

    metrics = compute_subgraph_trajectory_metrics(seq)
    # Consecutive Jaccard must be 1.0 across all windows for stable strategy
    for jacc in metrics["jaccard_consecutive"]:
        assert np.isclose(jacc, 1.0)


# 6. Test configurable k parameter (k=2)
def test_configurable_k_hop(sample_graph_data):
    edge_index = sample_graph_data["edge_indices"][0]
    seed = 42

    rois_k1 = extract_khop_neighborhood(edge_index, seed_roi=seed, k=1)
    rois_k2 = extract_khop_neighborhood(edge_index, seed_roi=seed, k=2)

    assert set(rois_k1).issubset(set(rois_k2))
    assert len(rois_k2) >= len(rois_k1)
