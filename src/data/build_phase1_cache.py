"""
Builds the Phase 1 seed-subgraph cache (see src/data/subgraph_cache.py) for all 871
subjects, restricted to a fixed 20-ROI seed subsample (configs/phase1.yaml).

The M2 pilot measured ~71s/subject to extract all 360 seeds across all windows for
one subject (~17h for the full dataset) -- too slow to redo every training epoch.
Restricted to 20 seeds, extraction measured ~0.05s/subject (~1 min for all 871
subjects), so we cache everyone once and let downstream code (training, threshold
calibration, evaluation, few-shot) select subjects by ID from the subject splits.
"""

import json
import os
import yaml
import numpy as np
import pandas as pd

from src.data.subgraph_cache import select_seed_rois, build_cache_for_subjects


def load_configs():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    return base_cfg, p1_cfg


def get_subject_pools(base_cfg, p1_cfg):
    splits = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "subject_splits.json")))
    summary = pd.read_csv(os.path.join(base_cfg["dataset"]["metadata_dir"], "subject_summary.csv"))
    label_map = dict(zip(summary["subject_id"], summary["label"]))

    train_ids = splits["splits"]["train"]
    val_ids = splits["splits"]["val"]
    test_ids = splits["splits"]["test"]

    return {
        "train_control": sorted([s for s in train_ids if label_map[s] == 0]),
        "val_control": sorted([s for s in val_ids if label_map[s] == 0]),
        "train_asd_pool": sorted([s for s in train_ids if label_map[s] == 1]),
        "test_all": sorted(test_ids),
        "all_subjects": sorted(label_map.keys()),
    }


def main():
    base_cfg, p1_cfg = load_configs()
    pools = get_subject_pools(base_cfg, p1_cfg)

    seed_rois = select_seed_rois(
        num_seeds=p1_cfg["seeds"]["num_seeds"],
        selection_seed=p1_cfg["seeds"]["selection_seed"],
    )
    print(f"[*] Seed ROIs ({len(seed_rois)}): {seed_rois.tolist()}")
    print(f"[*] Subjects to cache: {len(pools['all_subjects'])} "
          f"(train_control={len(pools['train_control'])}, val_control={len(pools['val_control'])}, "
          f"train_asd_pool={len(pools['train_asd_pool'])}, test_all={len(pools['test_all'])})")

    cache_dir = p1_cfg["cache"]["dir"]
    graphs_dir = base_cfg["dataset"]["graphs_dir"]
    build_cache_for_subjects(pools["all_subjects"], graphs_dir, cache_dir, seed_rois)

    manifest_path = os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")
    with open(manifest_path, "w") as f:
        json.dump({"seed_rois": seed_rois.tolist(), **pools}, f, indent=2)
    print(f"[*] Saved subject-pool manifest to {manifest_path}")


if __name__ == "__main__":
    main()
