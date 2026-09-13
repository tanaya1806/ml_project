"""Unit tests for the Phase 1 seed-subgraph cache."""

import glob
import os

import numpy as np
import pytest
import yaml

from src.data.subgraph_cache import (
    select_seed_rois,
    build_subject_cache,
    build_cache_for_subjects,
    load_subject_cache,
    cache_path_for,
)

_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")))
GRAPH_DIR = _CFG["dataset"]["graphs_dir"]


def test_select_seed_rois_deterministic_and_bounded():
    seeds_a = select_seed_rois(num_seeds=20, selection_seed=42, num_total_rois=360)
    seeds_b = select_seed_rois(num_seeds=20, selection_seed=42, num_total_rois=360)
    assert np.array_equal(seeds_a, seeds_b)
    assert len(seeds_a) == len(set(seeds_a.tolist()))
    assert np.all(seeds_a >= 0) and np.all(seeds_a < 360)


def test_select_seed_rois_different_seed_differs():
    seeds_a = select_seed_rois(num_seeds=20, selection_seed=42)
    seeds_b = select_seed_rois(num_seeds=20, selection_seed=7)
    assert not np.array_equal(seeds_a, seeds_b)


@pytest.fixture(scope="module")
def sample_graph_file():
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.npz")))
    if not files:
        pytest.skip(f"No processed graph files found in {GRAPH_DIR}")
    return files[0]


def test_build_and_roundtrip_cache(tmp_path, sample_graph_file):
    seed_rois = select_seed_rois(num_seeds=3, selection_seed=42)
    cache = build_subject_cache(sample_graph_file, seed_rois)

    out_path = os.path.join(tmp_path, f"{cache['subject_id']}.npz")
    np.savez_compressed(out_path, **cache)
    loaded = load_subject_cache(cache["subject_id"], str(tmp_path))

    assert loaded["subject_id"] == cache["subject_id"]
    assert loaded["num_windows"] == cache["num_windows"]
    assert np.array_equal(loaded["seed_rois"], np.asarray(seed_rois))

    for seed in seed_rois:
        seed = int(seed)
        roi_indices = loaded[f"roi_indices_{seed}"]
        edge_index_seq = loaded[f"edge_index_{seed}"]
        edge_weight_seq = loaded[f"edge_weight_{seed}"]
        assert seed in roi_indices
        assert len(edge_index_seq) == cache["num_windows"]
        # ROI membership must be identical (stable tracking) across all windows
        for t in range(len(edge_index_seq)):
            assert edge_index_seq[t].shape[0] == 2
            assert edge_index_seq[t].shape[1] == edge_weight_seq[t].shape[0]
            if edge_index_seq[t].shape[1] > 0:
                assert np.all(edge_index_seq[t] >= 0)
                assert np.all(edge_index_seq[t] < len(roi_indices))


def test_build_cache_for_subjects_skips_existing(tmp_path, sample_graph_file):
    subject_id = os.path.basename(sample_graph_file).replace(".npz", "")
    graphs_dir = os.path.dirname(sample_graph_file)
    seed_rois = select_seed_rois(num_seeds=2, selection_seed=42)

    build_cache_for_subjects([subject_id], graphs_dir, str(tmp_path), seed_rois, progress=False)
    cache_path = cache_path_for(subject_id, str(tmp_path))
    assert os.path.exists(cache_path)

    mtime_before = os.path.getmtime(cache_path)
    build_cache_for_subjects([subject_id], graphs_dir, str(tmp_path), seed_rois, progress=False, force=False)
    assert os.path.getmtime(cache_path) == mtime_before  # not rebuilt
