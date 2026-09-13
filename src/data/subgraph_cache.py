"""
Phase 1: Stable Temporal Subgraph Cache.

Reuses M2's stable tracking (src.graphs.temporal_tracking) to extract, once, the
per-seed temporal subgraph sequence for a fixed subsample of seed ROIs, and caches
the result to a compact per-subject .npz file.

Rationale (see configs/phase1.yaml): recomputing k-hop extraction from the parent
graph (1.4 GB) on every training epoch is wasteful (M2 pilot: ~71s/subject for all
360 seeds). GIN embeddings can't be cached instead because the GIN weights are being
trained (Phase 1A). So we cache the intermediate representation that is stable across
epochs: subgraph structure (edge_index/edge_weight per seed per window).
"""

import os
from typing import Dict, List, Sequence

import numpy as np

from src.graphs.temporal_tracking import track_temporal_subgraph_stable


def select_seed_rois(num_seeds: int = 20, selection_seed: int = 42, num_total_rois: int = 360) -> np.ndarray:
    """Deterministic, reproducible stratified subsample of seed ROIs spanning the atlas."""
    rng = np.random.default_rng(selection_seed)
    step = num_total_rois / num_seeds
    # Stratified: one draw per equal-width bin, so seeds spread across the full ROI range
    # rather than clustering under pure uniform sampling.
    bins = (np.arange(num_seeds) * step).astype(int)
    offsets = rng.integers(0, max(int(step), 1), size=num_seeds)
    seeds = np.clip(bins + offsets, 0, num_total_rois - 1)
    return np.unique(seeds).astype(np.int16)


def cache_path_for(subject_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{subject_id}.npz")


def build_subject_cache(
    graph_npz_path: str,
    seed_rois: Sequence[int],
    k: int = 1,
    ref_window_idx: int = 0,
) -> Dict:
    """Extracts stable temporal subgraphs for all seed_rois from one subject's parent graph."""
    data = np.load(graph_npz_path)
    subject_id = str(data["subject_id"])
    site = str(data["site"])
    label = int(data["label"])
    num_windows = int(data["num_windows"])
    edge_indices = data["edge_indices"]
    edge_weights = data["edge_weights"]

    out = {
        "subject_id": subject_id,
        "site": site,
        "label": label,
        "num_windows": num_windows,
        "seed_rois": np.asarray(seed_rois, dtype=np.int16),
    }

    for seed in seed_rois:
        seq = track_temporal_subgraph_stable(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            seed_roi=int(seed),
            subject_id=subject_id,
            k=k,
            ref_window_idx=ref_window_idx,
        )
        out[f"roi_indices_{seed}"] = seq.subgraphs[0].roi_indices

        edge_index_arr = np.empty(len(seq.subgraphs), dtype=object)
        edge_weight_arr = np.empty(len(seq.subgraphs), dtype=object)
        for i, s in enumerate(seq.subgraphs):
            edge_index_arr[i] = s.edge_index
            edge_weight_arr[i] = s.edge_weight
        out[f"edge_index_{seed}"] = edge_index_arr
        out[f"edge_weight_{seed}"] = edge_weight_arr

    return out


def build_cache_for_subjects(
    subject_ids: Sequence[str],
    graphs_dir: str,
    cache_dir: str,
    seed_rois: Sequence[int],
    k: int = 1,
    force: bool = False,
    progress: bool = True,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    iterator = subject_ids
    if progress:
        from tqdm import tqdm
        iterator = tqdm(subject_ids, desc="Caching seed subgraphs")

    for subject_id in iterator:
        out_path = cache_path_for(subject_id, cache_dir)
        if os.path.exists(out_path) and not force:
            continue
        graph_path = os.path.join(graphs_dir, f"{subject_id}.npz")
        cache = build_subject_cache(graph_path, seed_rois, k=k)
        np.savez_compressed(out_path, **cache)


def load_subject_cache(subject_id: str, cache_dir: str) -> Dict:
    path = cache_path_for(subject_id, cache_dir)
    data = np.load(path, allow_pickle=True)
    out = {
        "subject_id": str(data["subject_id"]),
        "site": str(data["site"]),
        "label": int(data["label"]),
        "num_windows": int(data["num_windows"]),
        "seed_rois": data["seed_rois"],
    }
    for seed in out["seed_rois"]:
        out[f"roi_indices_{seed}"] = data[f"roi_indices_{seed}"]
        out[f"edge_index_{seed}"] = data[f"edge_index_{seed}"]
        out[f"edge_weight_{seed}"] = data[f"edge_weight_{seed}"]
    return out
