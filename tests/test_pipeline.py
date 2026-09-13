"""
Unit Tests for ABIDE I Data Loader, Sliding-Window FC, and Temporal Graph Construction.
"""

import os
import pytest
import numpy as np

from src.data.loader import ABIDELoader, SubjectRecord
from src.data.sliding_window import compute_pearson_fc, compute_sliding_window_fc, WindowFC
from src.data.graph_construction import sparsify_fc_matrix, construct_temporal_graphs, TemporalGraph


DATA_DIR = r"D:\Graph anomaly detection\data\raw\Time_serie_per_site"


@pytest.fixture
def loader():
    return ABIDELoader(DATA_DIR)


# 1. Test NPZ loading
def test_npz_loading(loader):
    assert len(loader.available_sites) == 20
    assert "CALTECH" in loader.available_sites
    assert "NYU" in loader.available_sites

    caltech_subjs = loader.load_site("CALTECH")
    assert len(caltech_subjs) == 15
    for s in caltech_subjs:
        assert isinstance(s, SubjectRecord)
        assert s.num_rois == 360
        assert s.timepoints == 146
        assert s.time_series.shape == (146, 360)
        assert s.label in (0, 1)


# 2. Test label alignment
def test_label_alignment(loader):
    # Test on multiple sites that labels are correctly mapped
    for site in ["CALTECH", "OHSU", "LEUVEN_1"]:
        subjs = loader.load_site(site)
        labels = [s.label for s in subjs]
        # In NPZ files, normal (0) are first, followed by autism (1)
        zero_indices = [i for i, l in enumerate(labels) if l == 0]
        one_indices = [i for i, l in enumerate(labels) if l == 1]
        assert len(zero_indices) > 0 and len(one_indices) > 0
        assert max(zero_indices) < min(one_indices), f"Site {site} labels not strictly separated!"


# 3. Test variable-length handling
def test_variable_length_handling(loader):
    # CMU has variable lengths: 202 to 316
    cmu_subjs = loader.load_site("CMU")
    assert len(cmu_subjs) == 11
    lengths = [s.timepoints for s in cmu_subjs]
    assert min(lengths) >= 202
    assert max(lengths) <= 316
    assert len(set(lengths)) > 1, "CMU expected to have variable time series lengths"
    
    # Check that sliding window works for each variable-length subject
    for s in cmu_subjs:
        wfcs = compute_sliding_window_fc(s.time_series, window_size=30, stride=10)
        expected_w = (s.timepoints - 30) // 10 + 1
        assert len(wfcs) == expected_w


# 4. Test sliding-window generation
def test_sliding_window_generation():
    np.random.seed(42)
    # Synthetic subject: T=100, N=360
    synthetic_ts = np.random.randn(100, 360).astype(np.float32)
    window_size = 30
    stride = 10
    wfcs = compute_sliding_window_fc(synthetic_ts, window_size=window_size, stride=stride)

    expected_num_windows = (100 - 30) // 10 + 1  # 8 windows
    assert len(wfcs) == expected_num_windows

    for i, w in enumerate(wfcs):
        assert w.window_idx == i
        assert w.start_t == i * stride
        assert w.end_t == i * stride + window_size
        assert w.fc_matrix.shape == (360, 360)


# 5. Test FC shape
def test_fc_shape():
    np.random.seed(42)
    window_data = np.random.randn(30, 360).astype(np.float32)
    fc = compute_pearson_fc(window_data)
    assert fc.shape == (360, 360)
    assert fc.dtype == np.float32


# 6. Test FC symmetry
def test_fc_symmetry():
    np.random.seed(42)
    window_data = np.random.randn(30, 360).astype(np.float32)
    fc = compute_pearson_fc(window_data)
    # Strict numerical symmetry
    assert np.allclose(fc, fc.T, atol=1e-7)


# 7. Test diagonal handling
def test_diagonal_handling():
    np.random.seed(42)
    window_data = np.random.randn(30, 360).astype(np.float32)
    fc_no_diag = compute_pearson_fc(window_data, remove_diagonal=True)
    assert np.all(np.diag(fc_no_diag) == 0.0)

    fc_with_diag = compute_pearson_fc(window_data, remove_diagonal=False)
    assert np.allclose(np.diag(fc_with_diag), 1.0, atol=1e-5)


# 8. Test thresholding & sparsification
def test_thresholding_and_sparsification():
    np.random.seed(42)
    window_data = np.random.randn(30, 360).astype(np.float32)
    fc = compute_pearson_fc(window_data, remove_diagonal=True)

    sparsity_ratio = 0.20
    edge_index, edge_weight, thresh = sparsify_fc_matrix(fc, sparsity_ratio=sparsity_ratio, keep_signed_weights=True)

    total_off_diag = 360 * 359 // 2  # 64,620
    expected_undirected = int(np.floor(total_off_diag * sparsity_ratio))  # 12,924
    expected_directed = 2 * expected_undirected  # 25,848

    assert edge_index.shape == (2, expected_directed)
    assert edge_weight.shape == (expected_directed,)
    assert edge_index.dtype == np.int16
    assert edge_weight.dtype == np.float32

    # Verify no self-loops
    assert np.all(edge_index[0] != edge_index[1])

    # Verify thresholding: every kept edge must have |r| >= thresh
    assert np.all(np.abs(edge_weight) >= thresh - 1e-6)

    # Verify edge symmetry: if (u, v, w) in edges, then (v, u, w) must also be in edges
    edge_dict = {}
    for idx in range(expected_directed):
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        w = float(edge_weight[idx])
        edge_dict[(u, v)] = w

    for (u, v), w in edge_dict.items():
        assert (v, u) in edge_dict, f"Reverse edge ({v}, {u}) missing!"
        assert np.isclose(edge_dict[(v, u)], w, atol=1e-5), f"Edge weight mismatch for reverse edge ({v}, {u})!"


# 9. Test graph construction
def test_graph_construction():
    np.random.seed(42)
    synthetic_ts = np.random.randn(80, 360).astype(np.float32)
    wfcs = compute_sliding_window_fc(synthetic_ts, window_size=30, stride=10)
    graphs = construct_temporal_graphs(wfcs, sparsity_ratio=0.20)

    assert len(graphs) == len(wfcs)
    for g in graphs:
        assert isinstance(g, TemporalGraph)
        assert g.num_nodes == 360
        assert g.num_edges == 25848
        assert g.edge_index.shape == (2, 25848)
        assert g.edge_weight.shape == (25848,)
        assert g.threshold_value > 0.0
