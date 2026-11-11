"""
Shared, real model-loading + inference helpers used by every script in this folder.
Loads the ACTUAL trained Phase 1 checkpoint (outputs/experiments/phase1/checkpoints/best.pt)
and runs the ACTUAL GIN / GRU forward pass (src/models/*) on ACTUAL cached subgraphs
(data/processed/subgraph_cache/*.npz) -- no reimplementation, the real project modules.
"""
import os
import sys
import yaml
import numpy as np
import torch

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
sys.path.insert(0, ROOT)

from src.data.subgraph_cache import load_subject_cache
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.models.embedding_sequence import seed_embedding_sequence
from src.anomaly.scoring import transition_scores
from src.utils.device import get_device

CACHE_DIR = os.path.join(ROOT, "data", "processed", "subgraph_cache")


def load_configs():
    base_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/baseline.yaml")))
    p1_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/phase1.yaml")))
    p2_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/phase2.yaml")))
    return base_cfg, p1_cfg, p2_cfg


_MODELS = {}


def get_models():
    if "gin" in _MODELS:
        return _MODELS["gin"], _MODELS["gru"], _MODELS["ckpt"], _MODELS["device"], _MODELS["p1_cfg"]
    base_cfg, p1_cfg, p2_cfg = load_configs()
    device = get_device(p1_cfg["device"])
    gin = GINEncoder(**p1_cfg["gin"]).to(device)
    gru = TemporalGRU(
        embedding_dim=p1_cfg["gin"]["embedding_dim"],
        hidden_dim=p1_cfg["gru"]["hidden_dim"],
        num_layers=p1_cfg["gru"]["num_layers"],
    ).to(device)
    ckpt_path = os.path.join(ROOT, p1_cfg["output_dir"], "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    gin.load_state_dict(ckpt["gin"])
    gru.load_state_dict(ckpt["gru"])
    gin.eval()
    gru.eval()
    print(f"[common_model] loaded REAL checkpoint {ckpt_path} "
          f"(epoch {ckpt['epoch']}, val_pred_loss={ckpt['val_pred_loss']:.4f}) on device={device}")
    _MODELS.update(gin=gin, gru=gru, ckpt=ckpt, device=device, p1_cfg=p1_cfg)
    return gin, gru, ckpt, device, p1_cfg


def cache_for(subject_id):
    return load_subject_cache(subject_id, CACHE_DIR)


def embedding_sequence(subject_id, seed):
    """Real forward pass: cached subgraph sequence -> [T, 128] real GIN embeddings."""
    gin, gru, ckpt, device, p1_cfg = get_models()
    cache = cache_for(subject_id)
    with torch.no_grad():
        z_seq = seed_embedding_sequence(gin, cache, int(seed), device)
    return z_seq, cache


def gru_predict(z_seq):
    gin, gru, ckpt, device, p1_cfg = get_models()
    with torch.no_grad():
        z_hat_seq = gru(z_seq.unsqueeze(0)).squeeze(0)
    return z_hat_seq


def anomaly_scores(z_seq, z_hat_seq):
    return transition_scores(z_seq, z_hat_seq)


def pools():
    import json
    base_cfg, p1_cfg, p2_cfg = load_configs()
    return json.load(open(os.path.join(ROOT, base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
