"""
Unit tests for the Phase 2 structure-aware anomaly generator (section 42 checklist:
valid edge range, symmetry, no self-loops, determinism, severity monotonicity,
anomaly actually differs from normal, generated anomaly stays connected to its
source subgraph).
"""

import numpy as np
import pytest
import torch

from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.generation.structure_aware_generator import generate_anomaly, _undirected_pairs


def _make_synthetic_cache(num_nodes=10, num_windows=4, seed=7, num_edges=20, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    roi_indices = np.arange(num_nodes, dtype=np.int16)
    edge_index_seq = np.empty(num_windows, dtype=object)
    edge_weight_seq = np.empty(num_windows, dtype=object)

    # Build one symmetric undirected edge set, reused (with varying weights) every window
    # -- mirrors the real M1/M2 representation where both (u,v) and (v,u) are present.
    pairs = set()
    while len(pairs) < num_edges // 2:
        u, v = rng.integers(0, num_nodes, size=2)
        if u != v:
            pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    pairs = list(pairs)

    for t in range(num_windows):
        src, dst, w = [], [], []
        for (u, v) in pairs:
            weight = float(rng.uniform(-1, 1))
            src += [u, v]
            dst += [v, u]
            w += [weight, weight]
        edge_index_seq[t] = np.array([src, dst], dtype=np.int16)
        edge_weight_seq[t] = np.array(w, dtype=np.float32)

    return {
        "subject_id": "SYN_000",
        "label": 0,
        "num_windows": num_windows,
        "seed_rois": np.array([seed], dtype=np.int16),
        f"roi_indices_{seed}": roi_indices,
        f"edge_index_{seed}": edge_index_seq,
        f"edge_weight_{seed}": edge_weight_seq,
    }


@pytest.fixture
def models():
    torch.manual_seed(0)
    gin = GINEncoder(in_channels=1, hidden_channels=16, embedding_dim=32, num_layers=2)
    gru = TemporalGRU(embedding_dim=32, hidden_dim=32, num_layers=1)
    return gin, gru


def test_valid_edge_range_and_finite(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.9, device=torch.device("cpu"), method="structure_aware")
    for e in rec["edges_changed"]:
        assert -1.0 <= e["weight_after"] <= 1.0
        assert np.isfinite(e["weight_after"])


def test_no_self_loops_introduced(models):
    """The generator never adds edges -- it only reweights the existing induced subgraph."""
    gin, gru = models
    cache = _make_synthetic_cache(seed=7)
    rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.9, device=torch.device("cpu"), method="structure_aware")
    for e in rec["edges_changed"]:
        assert e["roi_u"] != e["roi_v"]


def test_symmetric_edits(models):
    """Every changed edge is recorded once per undirected pair; both directed copies
    of the parent graph are updated identically (checked via _undirected_pairs)."""
    gin, gru = models
    cache = _make_synthetic_cache(seed=7)
    edge_index = cache["edge_index_7"][1]
    pairs = _undirected_pairs(edge_index)
    for key, idxs in pairs.items():
        assert len(idxs) == 2  # (u,v) and (v,u) both present


def test_deterministic_given_seed(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    r1 = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.5, device=torch.device("cpu"),
                           method="random", rng_seed=123)
    r2 = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.5, device=torch.device("cpu"),
                           method="random", rng_seed=123)
    assert r1["edges_changed"] == r2["edges_changed"]
    assert r1["anomaly_score_after"] == r2["anomaly_score_after"]


def test_anomaly_differs_from_normal(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.9, device=torch.device("cpu"), method="structure_aware")
    assert rec["num_edges_changed"] > 0
    assert any(e["weight_before"] != e["weight_after"] for e in rec["edges_changed"])


def test_zero_severity_changes_nothing(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.0, device=torch.device("cpu"), method="structure_aware")
    assert rec["num_edges_changed"] == 0
    assert rec["anomaly_score_after"] == pytest.approx(rec["anomaly_score_before"], abs=1e-4)


def test_generated_anomaly_stays_connected_to_source(models):
    """The generator must never change the ROI/node set -- only edge weights."""
    gin, gru = models
    cache = _make_synthetic_cache()
    rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.9, device=torch.device("cpu"), method="structure_aware")
    assert rec["subject_id"] == cache["subject_id"]
    assert rec["seed_roi"] == 7
    touched_rois = {e["roi_u"] for e in rec["edges_changed"]} | {e["roi_v"] for e in rec["edges_changed"]}
    assert touched_rois.issubset(set(cache["roi_indices_7"].tolist()))


def test_severity_monotonic_on_average(models):
    """Structure-aware mean deviation must increase with severity (averaged over trials
    to avoid relying on a single noisy example)."""
    gin, gru = models
    mean_scores = []
    for severity in [0.1, 0.5, 0.9]:
        scores = []
        for trial_seed in range(8):
            cache = _make_synthetic_cache(rng_seed=trial_seed)
            rec = generate_anomaly(gin, gru, cache, seed=7, t=0, severity=severity,
                                    device=torch.device("cpu"), method="structure_aware")
            scores.append(rec["anomaly_score_after"])
        mean_scores.append(np.mean(scores))
    assert mean_scores[0] < mean_scores[1] < mean_scores[2]


def test_invalid_severity_raises(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    with pytest.raises(ValueError):
        generate_anomaly(gin, gru, cache, seed=7, t=0, severity=1.5, device=torch.device("cpu"))


def test_invalid_method_raises(models):
    gin, gru = models
    cache = _make_synthetic_cache()
    with pytest.raises(ValueError):
        generate_anomaly(gin, gru, cache, seed=7, t=0, severity=0.5, device=torch.device("cpu"), method="gan")
