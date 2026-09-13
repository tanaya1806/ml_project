"""
Phase 3: Few-Shot Adaptation + Final Comparison.

Question: does structure-aware anomaly generation make the detector more reliable
when only a few real abnormal (ASD) examples are available?

Detector for this comparison. Phase 1's GIN+GRU stays FROZEN (it already encodes
normal structure/dynamics and is not retrained here). The only thing allowed to see
labeled examples is a logistic-regression head on a per-seed anomaly-score feature
vector (max A_t over windows, one entry per seed ROI) -- this keeps model capacity
fixed across all three conditions below, so any metric difference is attributable to
the TRAINING DATA (real vs real+generated), not to extra model capacity.

  Condition A (few-shot only):        k real ASD support subjects (+) vs train-control (-)
  Condition B (few-shot + generated): A's positives + N structure-aware generated anomalies
  Condition C (few-shot + random):    A's positives + N random-perturbation generated anomalies

Why a per-seed VECTOR feature, not Phase 1's single scalar (max over everything)
-- a design bug caught by actually running this experiment. A first version used
Phase 1's single subject-level scalar score as the only feature. A 1-D logistic
regression is a strictly monotonic function of that one number, so its AUROC/AUPRC
(rank-based metrics) were IDENTICAL across conditions A/B/C at every shot count --
adding real or generated positives only moved the classifier's threshold, never the
ranking of test subjects by their one fixed feature, so those metrics couldn't have
moved no matter what data was added. Rank-sensitive comparison needs a
multi-dimensional feature whose LEARNED WEIGHTS can change with the training set: we
use the vector of per-seed max scores (one dimension per seed ROI). A generated
anomaly (which only perturbs one seed's one transition) is turned into a full feature
vector by taking its real base subject's per-seed vector and substituting that one
seed's entry with the post-perturbation score -- "what would this real, otherwise-
normal subject's profile look like if this one seed's transition were structurally
anomalous," grounded in a real subject rather than invented from nothing.

All three conditions are evaluated on the SAME held-out, untouched test set for the
SAME support-set draw, so the comparison isolates the effect of the added data.

No leakage: support subjects and generation base subjects are drawn only from the
TRAIN split; isolation against val/test subject IDs is asserted explicitly.
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.data.subgraph_cache import load_subject_cache
from src.evaluate.detect_baseline import load_models, subject_scores
from src.generation.structure_aware_generator import generate_anomaly
from src.fewshot.support import sample_support_set, assert_subject_isolation
from src.utils.device import get_device


def load_configs():
    return (
        yaml.safe_load(open("configs/baseline.yaml")),
        yaml.safe_load(open("configs/phase1.yaml")),
        yaml.safe_load(open("configs/phase2.yaml")),
        yaml.safe_load(open("configs/phase3.yaml")),
    )


def subject_seed_vector(gin, gru, subject_id, seed_rois, cache_dir, device) -> np.ndarray:
    """Per-seed max A_t vector, shape (len(seed_rois),). This IS the feature vector."""
    cache = load_subject_cache(subject_id, cache_dir)
    scores = subject_scores(gin, gru, subject_id, cache, seed_rois, device)
    return np.array([float(scores[int(s)].max()) if len(scores[int(s)]) > 0 else 0.0 for s in seed_rois])


def compute_vectors_for_pool(gin, gru, subject_ids, seed_rois, cache_dir, device) -> Dict[str, np.ndarray]:
    return {sid: subject_seed_vector(gin, gru, sid, seed_rois, cache_dir, device) for sid in subject_ids}


def generate_expansion_vectors(
    gin, gru, base_vectors: Dict[str, np.ndarray], base_subjects: List[str], seed_rois, cache_dir, device,
    method: str, severity: float, n: int, rng_seed: int,
) -> List[np.ndarray]:
    """
    N synthetic feature vectors grounded in real (train-control) subjects: each is a
    real base subject's per-seed vector with ONE seed's entry replaced by the
    post-perturbation score from a generated anomaly at that seed/transition.
    """
    rng = np.random.default_rng(rng_seed)
    seed_index = {int(s): i for i, s in enumerate(seed_rois)}
    vectors = []
    attempts = 0
    while len(vectors) < n and attempts < n * 5:
        sid = base_subjects[rng.integers(0, len(base_subjects))]
        seed = int(seed_rois[rng.integers(0, len(seed_rois))])
        cache = load_subject_cache(sid, cache_dir)
        attempts += 1
        if cache["num_windows"] < 2:
            continue
        t = int(rng.integers(0, cache["num_windows"] - 1))
        rec = generate_anomaly(gin, gru, cache, seed, t, severity, device, method=method,
                                rng_seed=rng_seed + attempts)
        vec = base_vectors[sid].copy()
        vec[seed_index[seed]] = rec["anomaly_score_after"]
        vectors.append(vec)
    return vectors


def fit_and_evaluate(pos_vectors: List[np.ndarray], neg_vectors: List[np.ndarray],
                      test_vectors: Dict[str, np.ndarray], test_labels: Dict[str, int],
                      classifier_cfg: dict) -> dict:
    X = np.vstack(pos_vectors + neg_vectors)
    y = np.array([1] * len(pos_vectors) + [0] * len(neg_vectors))

    clf = LogisticRegression(C=classifier_cfg["C"], class_weight=classifier_cfg["class_weight"], max_iter=1000)
    clf.fit(X, y)

    subject_ids = list(test_vectors.keys())
    X_test = np.vstack([test_vectors[s] for s in subject_ids])
    y_true = np.array([test_labels[s] for s in subject_ids])
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "num_positive_train": len(pos_vectors),
        "num_negative_train": len(neg_vectors),
    }


def run_phase3():
    base_cfg, p1_cfg, p2_cfg, p3_cfg = load_configs()
    device = get_device(p3_cfg["device"])
    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
    seed_rois = pools["seed_rois"]
    cache_dir = p1_cfg["cache"]["dir"]

    gin, gru, ckpt = load_models(p1_cfg, device)
    print(f"[*] Loaded frozen Phase 1 checkpoint (epoch {ckpt['epoch']})")

    train_control = pools["train_control"]
    train_asd_pool = pools["train_asd_pool"]
    val_control = pools["val_control"]
    test_all = pools["test_all"]

    # Isolation: support/generation pools must never include val/test subjects.
    assert_subject_isolation(train_asd_pool, val_control + test_all)
    assert_subject_isolation(train_control, val_control + test_all)

    print("[*] Scoring negative pool (train control) and test set (held out, untouched)...")
    neg_vectors_by_id = compute_vectors_for_pool(gin, gru, train_control, seed_rois, cache_dir, device)
    neg_vectors = list(neg_vectors_by_id.values())

    test_vectors = compute_vectors_for_pool(gin, gru, test_all, seed_rois, cache_dir, device)
    test_labels = {sid: load_subject_cache(sid, cache_dir)["label"] for sid in test_all}

    print("[*] Scoring train-ASD support pool...")
    asd_pool_vectors = compute_vectors_for_pool(gin, gru, train_asd_pool, seed_rois, cache_dir, device)

    results = {}
    for shot in p3_cfg["shots"]:
        shot_results = {"few_shot_only": [], "few_shot_plus_structure_aware": [], "few_shot_plus_random": []}
        for trial in range(p3_cfg["num_trials"]):
            trial_seed = p3_cfg["random_seed"] + trial
            support_ids = sample_support_set(train_asd_pool, shot, seed=trial_seed)
            assert_subject_isolation(support_ids, val_control + test_all)
            real_pos_vectors = [asd_pool_vectors[sid] for sid in support_ids]

            # Condition A: few-shot only
            metrics_a = fit_and_evaluate(real_pos_vectors, neg_vectors, test_vectors, test_labels, p3_cfg["classifier"])
            shot_results["few_shot_only"].append(metrics_a)

            # Condition B: few-shot + structure-aware generated
            gen_sa = generate_expansion_vectors(
                gin, gru, neg_vectors_by_id, train_control, seed_rois, cache_dir, device,
                method="structure_aware", severity=p3_cfg["generation_severity"],
                n=p3_cfg["num_generated_per_trial"], rng_seed=trial_seed,
            )
            metrics_b = fit_and_evaluate(real_pos_vectors + gen_sa, neg_vectors, test_vectors, test_labels, p3_cfg["classifier"])
            metrics_b["num_generated"] = len(gen_sa)
            shot_results["few_shot_plus_structure_aware"].append(metrics_b)

            # Condition C: few-shot + random-perturbation generated (ablation baseline)
            gen_rand = generate_expansion_vectors(
                gin, gru, neg_vectors_by_id, train_control, seed_rois, cache_dir, device,
                method="random", severity=p3_cfg["generation_severity"],
                n=p3_cfg["num_generated_per_trial"], rng_seed=trial_seed + 1000,
            )
            metrics_c = fit_and_evaluate(real_pos_vectors + gen_rand, neg_vectors, test_vectors, test_labels, p3_cfg["classifier"])
            metrics_c["num_generated"] = len(gen_rand)
            shot_results["few_shot_plus_random"].append(metrics_c)

            print(f"[shot={shot} trial={trial}] "
                  f"A: F1={metrics_a['f1']:.3f} AUROC={metrics_a['auroc']:.3f} | "
                  f"B(+SA): F1={metrics_b['f1']:.3f} AUROC={metrics_b['auroc']:.3f} | "
                  f"C(+rand): F1={metrics_c['f1']:.3f} AUROC={metrics_c['auroc']:.3f}")

        results[f"{shot}-shot"] = shot_results

    summary = summarize(results)
    output_dir = p3_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "raw_trial_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "final_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Saved raw trial results and final comparison to {output_dir}")
    print_summary_table(summary)
    return results, summary


def summarize(results: dict) -> dict:
    summary = {}
    for shot_key, conditions in results.items():
        summary[shot_key] = {}
        for cond_name, trials in conditions.items():
            metric_names = ["precision", "recall", "f1", "auroc", "auprc"]
            agg = {}
            for m in metric_names:
                vals = [t[m] for t in trials if not np.isnan(t[m])]
                agg[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
                agg[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
            summary[shot_key][cond_name] = agg

        base = summary[shot_key]["few_shot_only"]
        for cond_name in ["few_shot_plus_structure_aware", "few_shot_plus_random"]:
            deltas = {}
            for m in ["f1", "auroc", "auprc", "recall"]:
                deltas[f"delta_{m}"] = summary[shot_key][cond_name][f"{m}_mean"] - base[f"{m}_mean"]
            summary[shot_key][cond_name]["deltas_vs_few_shot_only"] = deltas
    return summary


def print_summary_table(summary: dict):
    print("\n" + "=" * 100)
    print("PHASE 3 FINAL COMPARISON")
    print("=" * 100)
    for shot_key, conditions in summary.items():
        print(f"\n-- {shot_key} --")
        for cond_name, agg in conditions.items():
            line = (f"  {cond_name:32s} P={agg['precision_mean']:.3f} R={agg['recall_mean']:.3f} "
                    f"F1={agg['f1_mean']:.3f} AUROC={agg['auroc_mean']:.3f} AUPRC={agg['auprc_mean']:.3f}")
            if "deltas_vs_few_shot_only" in agg:
                d = agg["deltas_vs_few_shot_only"]
                line += f"  | dF1={d['delta_f1']:+.3f} dAUROC={d['delta_auroc']:+.3f} dAUPRC={d['delta_auprc']:+.3f}"
            print(line)


if __name__ == "__main__":
    run_phase3()
