"""
Builds the Phase 2 generated-anomaly dataset: for a deterministic subsample of real
TRAIN-CONTROL subjects (never val/test), generates one anomalous transition per
(subject, seed ROI) at each severity level, with both the structure-aware generator
and the random-perturbation baseline (section 32C).

Saves per-record provenance + scores (not the full edge tensors, to keep this a
lightweight, regeneratable metadata file per configs/phase2.yaml's rationale:
everything here is deterministic given (subject_id, seed, t, severity, method,
rng_seed), so Phase 3 can regenerate the exact same graphs on demand instead of us
persisting gigabytes of anomalous graph tensors).
"""

import json
import os
from typing import List

import numpy as np
import torch
import yaml

from src.data.subgraph_cache import load_subject_cache
from src.evaluate.detect_baseline import load_models
from src.generation.structure_aware_generator import generate_anomaly
from src.utils.device import get_device


def load_configs():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    p2_cfg = yaml.safe_load(open("configs/phase2.yaml"))
    return base_cfg, p1_cfg, p2_cfg


def pick_base_subjects(pools: dict, p2_cfg: dict) -> List[str]:
    rng = np.random.default_rng(p2_cfg["dataset"]["random_seed"])
    pool = pools["train_control"]
    n = min(p2_cfg["dataset"]["num_base_subjects"], len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    return sorted([pool[i] for i in idx])


def build_dataset(p1_cfg, p2_cfg, pools, gin, gru, device) -> List[dict]:
    cache_dir = p1_cfg["cache"]["dir"]
    seed_rois = pools["seed_rois"]
    severities = p2_cfg["severities"]
    methods = p2_cfg["dataset"]["methods"]
    rng = np.random.default_rng(p2_cfg["dataset"]["random_seed"])

    base_subjects = pick_base_subjects(pools, p2_cfg)
    print(f"[*] Base (real, normal, train-control) subjects: {len(base_subjects)}")

    records = []
    for sid in base_subjects:
        cache = load_subject_cache(sid, cache_dir)
        if cache["num_windows"] < 2:
            continue
        for seed in seed_rois:
            t = int(rng.integers(0, cache["num_windows"] - 1))
            for sev_name, sev_val in severities.items():
                for method in methods:
                    rec = generate_anomaly(
                        gin, gru, cache, int(seed), t, sev_val, device,
                        method=method,
                        edge_perturb_max_frac=p2_cfg["generator"]["edge_perturb_max_frac"],
                        step_size_max=p2_cfg["generator"]["step_size_max"],
                        clip_range=tuple(p2_cfg["generator"]["clip_range"]),
                        rng_seed=p2_cfg["dataset"]["random_seed"],
                    )
                    rec["severity_name"] = sev_name
                    # Drop bulky arrays; keep lightweight provenance + summary.
                    rec.pop("edge_index", None)
                    rec.pop("edge_weight_anomalous", None)
                    rec.pop("roi_indices", None)
                    records.append(rec)

    return records


def main():
    base_cfg, p1_cfg, p2_cfg = load_configs()
    device = get_device(p2_cfg["device"])
    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))

    gin, gru, ckpt = load_models(p1_cfg, device)
    print(f"[*] Loaded Phase 1 checkpoint (epoch {ckpt['epoch']}, val_pred_loss={ckpt['val_pred_loss']:.4f})")

    records = build_dataset(p1_cfg, p2_cfg, pools, gin, gru, device)
    print(f"[*] Generated {len(records)} anomaly records "
          f"({len(p2_cfg['dataset']['methods'])} methods x {len(p2_cfg['severities'])} severities)")

    output_dir = p2_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_anomalies.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Saved generated-anomaly metadata to {out_path}")


if __name__ == "__main__":
    main()
