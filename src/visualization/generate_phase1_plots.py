"""
Phase 1 Validation Visualizations.

1. Training curves: train/val next-embedding prediction loss per epoch.
2. Anomaly score distribution: val-control (calibration) vs test-control vs test-ASD
   subject-level scores, with tau_subject marked.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

from src.data.subgraph_cache import load_subject_cache
from src.evaluate.detect_baseline import load_models, subject_scores
from src.utils.device import get_device


def plot_training_curves(training_report: dict, out_path: str):
    history = training_report["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_pred_loss"] for h in history]
    val_loss = [h["val_pred_loss"] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", label="train (control)")
    plt.plot(epochs, val_loss, marker="o", label="val (control)")
    plt.xlabel("Epoch")
    plt.ylabel("Next-embedding prediction loss (MSE)")
    plt.title("Phase 1: GIN+GRU normal-dynamics training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[*] Saved {out_path}")


def plot_score_distributions(base_cfg, p1_cfg, metrics: dict, out_path: str):
    device = get_device(p1_cfg["device"])
    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
    seed_rois = pools["seed_rois"]
    cache_dir = p1_cfg["cache"]["dir"]

    gin, gru, _ = load_models(p1_cfg, device)

    def subject_max_scores(subject_ids):
        vals = []
        for sid in subject_ids:
            cache = load_subject_cache(sid, cache_dir)
            scores = subject_scores(gin, gru, sid, cache, seed_rois, device)
            m = max((a_t.max() for a_t in scores.values() if len(a_t) > 0), default=0.0)
            vals.append(float(m))
        return vals

    val_control_scores = subject_max_scores(pools["val_control"])
    test_control_scores = subject_max_scores([s for s in pools["test_all"]
                                               if load_subject_cache(s, cache_dir)["label"] == 0])
    test_asd_scores = subject_max_scores([s for s in pools["test_all"]
                                          if load_subject_cache(s, cache_dir)["label"] == 1])

    plt.figure(figsize=(7, 4.5))
    sns.kdeplot(val_control_scores, label="val control (calibration)", fill=True, alpha=0.3)
    sns.kdeplot(test_control_scores, label="test control", fill=True, alpha=0.3)
    sns.kdeplot(test_asd_scores, label="test ASD", fill=True, alpha=0.3)
    plt.axvline(metrics["tau_subject"], color="red", linestyle="--", label="tau_subject (p95)")
    plt.xlabel("Subject-level anomaly score (max A_t over seeds/windows)")
    plt.title("Phase 1: subject-level anomaly score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[*] Saved {out_path}")


def main():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig_dir = os.path.join(base_cfg["dataset"]["figures_dir"], "phase1")
    os.makedirs(fig_dir, exist_ok=True)

    training_report = json.load(open(os.path.join(p1_cfg["output_dir"], "training_report.json")))
    plot_training_curves(training_report, os.path.join(fig_dir, "01_training_curves.png"))

    metrics = json.load(open(os.path.join(p1_cfg["output_dir"], "metrics.json")))
    plot_score_distributions(base_cfg, p1_cfg, metrics, os.path.join(fig_dir, "02_anomaly_score_distributions.png"))


if __name__ == "__main__":
    main()
