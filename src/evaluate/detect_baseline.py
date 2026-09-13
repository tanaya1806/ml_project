"""
Phase 1D/1E: Baseline Temporal Anomaly Detection + Few-Shot Support Evaluation.

Pipeline:
  1. Load the GIN+GRU checkpoint trained in Phase 1C (control subjects only).
  2. Calibrate tau = percentile_95(A_t) using VALIDATION CONTROL subjects only.
  3. Score every TEST subject (both control and ASD, untouched until now) and report
     Precision/Recall/F1/AUROC/AUPRC against the subject's diagnosis label, used here
     as a weak proxy for "shows abnormal temporal connectivity" (ABIDE has no
     ground-truth per-transition anomaly labels).
  4. Localize anomalies to (subject, seed ROI, window transition).
  5. Draw 1/5/10-shot ASD support sets from the TRAIN ASD pool (never val/test) and
     report per-subgraph recurrence C(S) across independent support-set draws.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

from src.data.subgraph_cache import load_subject_cache
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.models.embedding_sequence import seed_embedding_sequence
from src.anomaly.scoring import transition_scores, calibrate_threshold
from src.fewshot.support import sample_support_set, assert_subject_isolation, compute_recurrence
from src.utils.device import get_device


def load_configs():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    return base_cfg, p1_cfg


def load_models(p1_cfg, device):
    gin = GINEncoder(**p1_cfg["gin"]).to(device)
    gru = TemporalGRU(
        embedding_dim=p1_cfg["gin"]["embedding_dim"],
        hidden_dim=p1_cfg["gru"]["hidden_dim"],
        num_layers=p1_cfg["gru"]["num_layers"],
    ).to(device)
    ckpt_path = os.path.join(p1_cfg["output_dir"], "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    gin.load_state_dict(ckpt["gin"])
    gru.load_state_dict(ckpt["gru"])
    gin.eval()
    gru.eval()
    return gin, gru, ckpt


def subject_scores(gin, gru, subject_id, cache, seed_rois, device) -> Dict[int, np.ndarray]:
    """Returns {seed_roi: A_t array of shape [T-1]} for one subject."""
    out = {}
    with torch.no_grad():
        for seed in seed_rois:
            z_seq = seed_embedding_sequence(gin, cache, int(seed), device)
            if z_seq.shape[0] < 2:
                out[int(seed)] = np.zeros(0)
                continue
            z_hat_seq = gru(z_seq.unsqueeze(0)).squeeze(0)
            a_t = transition_scores(z_seq, z_hat_seq).cpu().numpy()
            out[int(seed)] = a_t
    return out


def run_evaluation():
    base_cfg, p1_cfg = load_configs()
    device = get_device(p1_cfg["device"])
    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
    seed_rois = pools["seed_rois"]
    cache_dir = p1_cfg["cache"]["dir"]

    gin, gru, ckpt = load_models(p1_cfg, device)
    print(f"[*] Loaded checkpoint from epoch {ckpt['epoch']} (val_pred_loss={ckpt['val_pred_loss']:.4f})")

    # --- Step 1: calibrate thresholds on validation control subjects only ---
    #
    # Two thresholds are needed, not one, because the subject-level detection
    # decision aggregates hundreds of individual transitions (20 seeds x ~14
    # windows). Calibrating a single per-transition tau at the 95th percentile
    # and then flagging a subject "anomalous" if ANY of its ~280 transitions
    # exceeds tau is a multiple-comparisons error: under the null (normal)
    # distribution itself, ~5% of ~280 transitions are *expected* to exceed a
    # per-transition 95th-percentile cutoff, so nearly every subject -- normal
    # or not -- would trip it (verified empirically: this first attempt gave
    # 131/131 test subjects flagged, AUROC ~0.45, i.e. no discrimination).
    #
    # tau_transition: per-transition threshold, used only for localization
    #                 (which window/seed looks unusual) and few-shot recurrence.
    # tau_subject:    threshold on the SAME subject-level statistic
    #                 (max A_t over seeds/windows) used for classification,
    #                 calibrated on that statistic's own val-control distribution.
    val_control = pools["val_control"]
    val_normal_scores: List[float] = []
    val_subject_max: List[float] = []
    for sid in val_control:
        cache = load_subject_cache(sid, cache_dir)
        scores = subject_scores(gin, gru, sid, cache, seed_rois, device)
        subj_max = 0.0
        for a_t in scores.values():
            val_normal_scores.extend(a_t.tolist())
            if len(a_t) > 0:
                subj_max = max(subj_max, float(a_t.max()))
        val_subject_max.append(subj_max)

    tau_transition = calibrate_threshold(val_normal_scores, p1_cfg["anomaly"]["threshold_percentile"])
    tau_subject = calibrate_threshold(val_subject_max, p1_cfg["anomaly"]["threshold_percentile"])
    print(f"[*] tau_transition (percentile_{p1_cfg['anomaly']['threshold_percentile']} of "
          f"{len(val_normal_scores)} val-control transitions) = {tau_transition:.4f}")
    print(f"[*] tau_subject (percentile_{p1_cfg['anomaly']['threshold_percentile']} of "
          f"{len(val_subject_max)} val-control subject-max scores) = {tau_subject:.4f}")
    tau = tau_transition  # used below for localization / few-shot recurrence flags

    # --- Step 2: score every TEST subject (held out, untouched until now) ---
    test_all = pools["test_all"]
    subject_level_score: Dict[str, float] = {}
    subject_label: Dict[str, int] = {}
    subject_seed_flags: Dict[str, Dict[int, bool]] = {}
    localization: List[dict] = []

    for sid in test_all:
        cache = load_subject_cache(sid, cache_dir)
        subject_label[sid] = cache["label"]
        scores = subject_scores(gin, gru, sid, cache, seed_rois, device)

        max_score = 0.0
        seed_flags = {}
        for seed, a_t in scores.items():
            if len(a_t) == 0:
                seed_flags[seed] = False
                continue
            seed_max = float(a_t.max())
            max_score = max(max_score, seed_max)
            anomalous_windows = np.where(a_t > tau)[0]
            seed_flags[seed] = len(anomalous_windows) > 0
            for w in anomalous_windows:
                localization.append({
                    "subject_id": sid,
                    "label": cache["label"],
                    "seed_roi": seed,
                    "transition": f"{int(w)}->{int(w)+1}",
                    "anomaly_score": float(a_t[w]),
                })
        subject_level_score[sid] = max_score
        subject_seed_flags[sid] = seed_flags

    # --- Step 3: metrics against subject diagnosis (weak proxy label) ---
    y_true = np.array([subject_label[sid] for sid in test_all])
    y_score = np.array([subject_level_score[sid] for sid in test_all])
    y_pred = (y_score > tau_subject).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "tau_transition": tau_transition,
        "tau_subject": tau_subject,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "num_test_subjects": len(test_all),
        "num_subjects_flagged_anomalous": int(y_pred.sum()),
        "num_anomalous_transitions_detected": len(localization),
    }
    print("[*] Baseline detection metrics on TEST set:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # --- Step 4: few-shot support sets + recurrence (Phase 1E) ---
    train_asd_pool = pools["train_asd_pool"]
    assert_subject_isolation(train_asd_pool, val_control + test_all)

    # We need anomaly flags for the *support* subjects too (they come from train, not
    # yet scored above), computed with the SAME frozen model/threshold.
    support_subject_flags: Dict[str, Dict[int, bool]] = {}

    def get_or_compute_flags(sid: str) -> Dict[int, bool]:
        if sid in support_subject_flags:
            return support_subject_flags[sid]
        cache = load_subject_cache(sid, cache_dir)
        scores = subject_scores(gin, gru, sid, cache, seed_rois, device)
        flags = {seed: bool(len(a_t) > 0 and (a_t > tau).any()) for seed, a_t in scores.items()}
        support_subject_flags[sid] = flags
        return flags

    fewshot_results = {}
    for shot in p1_cfg["fewshot"]["shots"]:
        draws = []
        for trial in range(p1_cfg["fewshot"]["num_support_seeds"]):
            support_ids = sample_support_set(train_asd_pool, shot, seed=p1_cfg["fewshot"]["random_seed"] + trial)
            assert_subject_isolation(support_ids, val_control + test_all)
            for sid in support_ids:
                get_or_compute_flags(sid)
            recurrence = compute_recurrence(support_ids, support_subject_flags)
            draws.append({
                "trial": trial,
                "support_subjects": support_ids,
                "recurrence": recurrence,
                "mean_recurrence": float(np.mean(list(recurrence.values()))) if recurrence else 0.0,
            })
        fewshot_results[f"{shot}-shot"] = draws
        mean_over_trials = float(np.mean([d["mean_recurrence"] for d in draws]))
        std_over_trials = float(np.std([d["mean_recurrence"] for d in draws]))
        print(f"[*] {shot}-shot: mean subgraph recurrence = {mean_over_trials:.3f} +/- {std_over_trials:.3f} "
              f"(over {len(draws)} support-set draws)")

    # --- Save everything ---
    output_dir = p1_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "localization.json"), "w") as f:
        json.dump(localization, f, indent=2)
    with open(os.path.join(output_dir, "fewshot_results.json"), "w") as f:
        json.dump(fewshot_results, f, indent=2)
    print(f"[*] Saved metrics/localization/fewshot results to {output_dir}")

    return metrics, localization, fewshot_results


if __name__ == "__main__":
    run_evaluation()
