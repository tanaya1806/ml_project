"""
Phase 1A/1B/1C: Joint GIN + GRU training on the normal (control-only) temporal
dynamics of stable brain subgraphs.

Training strategy (Phase 1A decision)
--------------------------------------
The GIN (M3) is NOT pretrained with a separate pretext task. Instead it is trained
end-to-end, jointly with the GRU, using the same next-embedding prediction loss
that Phase 1D uses for anomaly scoring:

    z_t       = GIN(G_t[S_v])                       (shared weights, all subjects/seeds/windows)
    z_hat_t   = GRU(z_1, ..., z_t)
    L_pred    = mean_t || z_hat_t - z_{t+1} ||^2

This is the simplest defensible strategy that still forces GIN away from its random
initialization: it avoids a second self-supervised pretext task (contrastive/GAN/
diffusion/etc.), which the master brief explicitly asks to avoid absent a concrete
need.

Collapse guard
---------------
Because the prediction target z_{t+1} is produced by the *same* GIN being trained,
the pure prediction loss has a trivial global optimum: GIN maps every subgraph to
the same constant vector, and GRU learns the identity/copy function -> loss ~ 0
with zero biological signal learned. To rule this degenerate solution out we add a
single VICReg-style variance hinge term computed over each mini-batch of pooled
embeddings:

    L_var = mean_d( relu(1 - std_d(z)) )
    L     = L_pred + lambda_var * L_var

which penalizes any embedding dimension whose batch standard deviation collapses
below 1. This is one extra term, not a second network or adversarial objective.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.subgraph_cache import load_subject_cache
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.models.embedding_sequence import seed_embedding_sequence
from src.utils.device import get_device


def load_configs() -> Tuple[dict, dict]:
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    return base_cfg, p1_cfg


def _preload_caches(subject_ids: List[str], cache_dir: str) -> Dict[str, dict]:
    return {sid: load_subject_cache(sid, cache_dir) for sid in subject_ids}


def _run_batch(gin, gru, batch_examples, subject_caches, device, lambda_var):
    pred_losses = []
    all_z = []
    for sid, seed in batch_examples:
        cache = subject_caches[sid]
        z_seq = seed_embedding_sequence(gin, cache, int(seed), device)
        if z_seq.shape[0] < 2:
            continue
        z_hat_seq = gru(z_seq.unsqueeze(0)).squeeze(0)
        pred_losses.append(F.mse_loss(z_hat_seq[:-1], z_seq[1:]))
        all_z.append(z_seq)

    if not pred_losses:
        return None

    pred_loss = torch.stack(pred_losses).mean()
    z_cat = torch.cat(all_z, dim=0)
    std_z = torch.sqrt(z_cat.var(dim=0, unbiased=False) + 1e-4)
    var_penalty = torch.relu(1.0 - std_z).mean()
    loss = pred_loss + lambda_var * var_penalty
    return loss, pred_loss.item(), var_penalty.item(), z_cat.detach()


def evaluate_split(gin, gru, examples, subject_caches, device):
    gin.eval()
    gru.eval()
    losses = []
    with torch.no_grad():
        for sid, seed in examples:
            cache = subject_caches[sid]
            z_seq = seed_embedding_sequence(gin, cache, int(seed), device)
            if z_seq.shape[0] < 2:
                continue
            z_hat_seq = gru(z_seq.unsqueeze(0)).squeeze(0)
            losses.append(F.mse_loss(z_hat_seq[:-1], z_seq[1:]).item())
    return float(np.mean(losses)) if losses else float("nan")


def train():
    base_cfg, p1_cfg = load_configs()
    device = get_device(p1_cfg["device"])
    print(f"[*] Device: {device}")

    torch.manual_seed(p1_cfg["training"]["random_seed"])
    np.random.seed(p1_cfg["training"]["random_seed"])

    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
    seed_rois = pools["seed_rois"]
    train_control = pools["train_control"]
    val_control = pools["val_control"]
    cache_dir = p1_cfg["cache"]["dir"]

    print(f"[*] Preloading caches: {len(train_control)} train-control + {len(val_control)} val-control subjects")
    train_caches = _preload_caches(train_control, cache_dir)
    val_caches = _preload_caches(val_control, cache_dir)

    gin = GINEncoder(**p1_cfg["gin"]).to(device)
    gru = TemporalGRU(
        embedding_dim=p1_cfg["gin"]["embedding_dim"],
        hidden_dim=p1_cfg["gru"]["hidden_dim"],
        num_layers=p1_cfg["gru"]["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(
        list(gin.parameters()) + list(gru.parameters()),
        lr=p1_cfg["training"]["lr"],
        weight_decay=p1_cfg["training"]["weight_decay"],
    )

    train_examples = [(sid, seed) for sid in train_control for seed in seed_rois]
    val_examples = [(sid, seed) for sid in val_control for seed in seed_rois]

    batch_size = p1_cfg["training"].get("batch_size", 16)
    lambda_var = p1_cfg["training"].get("lambda_var", 0.5)
    epochs = p1_cfg["training"]["epochs"]
    rng = np.random.default_rng(p1_cfg["training"]["random_seed"])

    output_dir = p1_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(epochs):
        gin.train()
        gru.train()
        order = rng.permutation(len(train_examples))
        shuffled = [train_examples[i] for i in order]

        epoch_pred_losses, epoch_var_penalties = [], []
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            result = _run_batch(gin, gru, batch, train_caches, device, lambda_var)
            if result is None:
                continue
            loss, pred_l, var_p, _ = result
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_pred_losses.append(pred_l)
            epoch_var_penalties.append(var_p)

        val_loss = evaluate_split(gin, gru, val_examples, val_caches, device)
        mean_train_pred = float(np.mean(epoch_pred_losses))
        mean_train_var = float(np.mean(epoch_var_penalties))
        elapsed = time.time() - t_start
        print(f"[epoch {epoch+1}/{epochs}] train_pred_loss={mean_train_pred:.4f} "
              f"var_penalty={mean_train_var:.4f} val_pred_loss={val_loss:.4f} "
              f"elapsed={elapsed:.1f}s")
        history.append({
            "epoch": epoch + 1,
            "train_pred_loss": mean_train_pred,
            "train_var_penalty": mean_train_var,
            "val_pred_loss": val_loss,
            "elapsed_seconds": elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"gin": gin.state_dict(), "gru": gru.state_dict(), "epoch": epoch + 1, "val_pred_loss": val_loss},
                os.path.join(ckpt_dir, "best.pt"),
            )

    torch.save(
        {"gin": gin.state_dict(), "gru": gru.state_dict(), "epoch": epochs, "val_pred_loss": val_loss},
        os.path.join(ckpt_dir, "last.pt"),
    )

    # Post-training collapse check: per-dimension embedding std over a sample of val sequences.
    gin.eval()
    with torch.no_grad():
        sample_z = []
        for sid, seed in val_examples[:200]:
            z_seq = seed_embedding_sequence(gin, val_caches[sid], int(seed), device)
            sample_z.append(z_seq)
        z_all = torch.cat(sample_z, dim=0)
        embedding_std = z_all.std(dim=0)

    training_report = {
        "config": {"baseline": base_cfg, "phase1": p1_cfg},
        "device": str(device),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "history": history,
        "best_val_pred_loss": best_val_loss,
        "embedding_std_mean": float(embedding_std.mean().item()),
        "embedding_std_min": float(embedding_std.min().item()),
        "total_training_seconds": time.time() - t_start,
    }
    with open(os.path.join(output_dir, "training_report.json"), "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"[*] Training report saved to {os.path.join(output_dir, 'training_report.json')}")
    print(f"[*] Post-training embedding std: mean={embedding_std.mean().item():.3f} "
          f"min={embedding_std.min().item():.3f} (collapse if near 0)")

    return gin, gru, training_report


if __name__ == "__main__":
    train()
