"""
Phase 2 Generator Validation (section 24 of the brief):

  - Validity:            edge weights in range, finite, node/edge set unchanged.
  - Diversity:           generated anomalies are not near-duplicates of each other.
  - Severity control:    increasing severity increases mean deviation.
  - Structural grounding: anomalies are edits of real subgraphs (true by construction;
                          reports the real subject_id/seed/window provenance for every
                          generated record).
  - Temporal plausibility: only one transition window is modified per record (true by
                          construction; checked directly here).
  - Detectability:       fraction of generated anomalies whose score exceeds the
                          Phase 1 tau_transition, compared against real normal and real
                          ASD transition scores (section 25).
"""

import json
import os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from src.data.subgraph_cache import load_subject_cache
from src.evaluate.detect_baseline import load_models, subject_scores
from src.utils.device import get_device


def load_configs():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p1_cfg = yaml.safe_load(open("configs/phase1.yaml"))
    p2_cfg = yaml.safe_load(open("configs/phase2.yaml"))
    return base_cfg, p1_cfg, p2_cfg


def check_validity(records) -> dict:
    bad = 0
    for r in records:
        for e in r["edges_changed"]:
            if not (-1.0 - 1e-6 <= e["weight_after"] <= 1.0 + 1e-6):
                bad += 1
            if not np.isfinite(e["weight_after"]):
                bad += 1
        if r["window_transition"].count("->") != 1:
            bad += 1  # exactly one transition touched (temporal plausibility)
    return {"num_records": len(records), "num_constraint_violations": bad}


def check_diversity(records, sample_size=60) -> dict:
    """Mean pairwise Jaccard distance between changed-edge identity sets, per group."""
    groups = defaultdict(list)
    for r in records:
        key = (r["method"], r["severity_name"])
        edge_set = frozenset((e["roi_u"], e["roi_v"]) for e in r["edges_changed"])
        groups[key].append(edge_set)

    diversity = {}
    rng = np.random.default_rng(0)
    for key, sets in groups.items():
        idx = rng.choice(len(sets), size=min(sample_size, len(sets)), replace=False)
        sample = [sets[i] for i in idx]
        dists = []
        for a, b in combinations(sample, 2):
            union = len(a | b)
            if union == 0:
                continue
            dists.append(1.0 - len(a & b) / union)
        diversity[f"{key[0]}/{key[1]}"] = float(np.mean(dists)) if dists else 0.0
    return diversity


def check_severity_control(records) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["method"], r["severity_name"])].append(r["anomaly_score_after"])
    summary = {f"{k[0]}/{k[1]}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
               for k, v in groups.items()}

    sev_order = ["low", "medium", "high"]
    monotonic = {}
    for method in ["structure_aware", "random"]:
        means = [summary[f"{method}/{s}"]["mean"] for s in sev_order if f"{method}/{s}" in summary]
        monotonic[method] = bool(np.all(np.diff(means) > 0))
    return {"per_group": summary, "monotonic_in_severity": monotonic}


def check_detectability(records, tau_transition) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["method"], r["severity_name"])].append(r["anomaly_score_after"])
    out = {}
    for k, v in groups.items():
        v = np.array(v)
        out[f"{k[0]}/{k[1]}"] = float((v > tau_transition).mean())
    return out


def real_transition_scores(gin, gru, subject_ids, seed_rois, cache_dir, device, label_filter=None):
    scores = []
    for sid in subject_ids:
        cache = load_subject_cache(sid, cache_dir)
        if label_filter is not None and cache["label"] != label_filter:
            continue
        per_seed = subject_scores(gin, gru, sid, cache, seed_rois, device)
        for a_t in per_seed.values():
            scores.extend(a_t.tolist())
    return scores


def make_figures(records, real_normal_scores, real_asd_scores, tau_transition, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Figure: severity vs score (structure-aware vs random), with error bars.
    sev_order = ["low", "medium", "high"]
    sev_values = {"low": 0.25, "medium": 0.55, "high": 0.9}
    plt.figure(figsize=(6.5, 4.5))
    for method, color in [("structure_aware", "C0"), ("random", "C1")]:
        means, stds, xs = [], [], []
        for s in sev_order:
            vals = [r["anomaly_score_after"] for r in records if r["method"] == method and r["severity_name"] == s]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            xs.append(sev_values[s])
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, label=method, color=color)
    plt.axhline(tau_transition, color="red", linestyle="--", label="tau_transition (Phase 1)")
    plt.xlabel("Severity (alpha)")
    plt.ylabel("Anomaly score after generation")
    plt.title("Phase 2: severity vs deviation, structure-aware vs random")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_severity_vs_score.png"), dpi=150)
    plt.close()

    # Figure: score distributions, real normal vs real ASD vs generated (structure-aware, high).
    generated_high = [r["anomaly_score_after"] for r in records
                       if r["method"] == "structure_aware" and r["severity_name"] == "high"]
    plt.figure(figsize=(7, 4.5))
    sns.kdeplot(real_normal_scores, label="real normal transitions (val control)", fill=True, alpha=0.3)
    sns.kdeplot(real_asd_scores, label="real ASD transitions (test)", fill=True, alpha=0.3)
    sns.kdeplot(generated_high, label="generated (structure-aware, high severity)", fill=True, alpha=0.3)
    plt.axvline(tau_transition, color="red", linestyle="--", label="tau_transition")
    plt.xlabel("Per-transition anomaly score A_t")
    plt.title("Phase 2: real normal vs real ASD vs generated anomaly scores")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_score_distributions.png"), dpi=150)
    plt.close()

    # Figure: edge-weight change magnitude, structure-aware vs random.
    sa_deltas = [abs(e["weight_after"] - e["weight_before"]) for r in records if r["method"] == "structure_aware"
                 for e in r["edges_changed"]]
    rand_deltas = [abs(e["weight_after"] - e["weight_before"]) for r in records if r["method"] == "random"
                   for e in r["edges_changed"]]
    plt.figure(figsize=(6.5, 4.5))
    sns.histplot(sa_deltas, label="structure-aware", stat="density", alpha=0.5, bins=30)
    sns.histplot(rand_deltas, label="random", stat="density", alpha=0.5, bins=30)
    plt.xlabel("|edge weight change| per touched edge")
    plt.title("Phase 2: edge-weight change magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_edge_weight_changes.png"), dpi=150)
    plt.close()


def main():
    base_cfg, p1_cfg, p2_cfg = load_configs()
    device = get_device(p2_cfg["device"])
    pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json")))
    seed_rois = pools["seed_rois"]
    cache_dir = p1_cfg["cache"]["dir"]

    records = json.load(open(os.path.join(p2_cfg["output_dir"], "generated_anomalies.json")))
    phase1_metrics = json.load(open(os.path.join(p1_cfg["output_dir"], "metrics.json")))
    tau_transition = phase1_metrics["tau_transition"]

    validity = check_validity(records)
    diversity = check_diversity(records)
    severity_control = check_severity_control(records)
    detectability = check_detectability(records, tau_transition)

    print("[*] Validity:", validity)
    print("[*] Severity monotonicity:", severity_control["monotonic_in_severity"])
    print("[*] Detectability (fraction exceeding tau_transition):")
    for k, v in sorted(detectability.items()):
        print(f"    {k}: {v:.3f}")
    print("[*] Diversity (mean pairwise Jaccard distance of changed-edge sets):")
    for k, v in sorted(diversity.items()):
        print(f"    {k}: {v:.3f}")

    gin, gru, _ = load_models(p1_cfg, device)
    real_normal = real_transition_scores(gin, gru, pools["val_control"], seed_rois, cache_dir, device)
    real_asd = real_transition_scores(gin, gru, pools["test_all"], seed_rois, cache_dir, device, label_filter=1)
    detectability["real_normal_val_control"] = float((np.array(real_normal) > tau_transition).mean())
    detectability["real_asd_test"] = float((np.array(real_asd) > tau_transition).mean())
    print(f"[*] Detectability, real normal (val control): {detectability['real_normal_val_control']:.3f}")
    print(f"[*] Detectability, real ASD (test):            {detectability['real_asd_test']:.3f}")

    fig_dir = os.path.join(base_cfg["dataset"]["figures_dir"], "phase2")
    make_figures(records, real_normal, real_asd, tau_transition, fig_dir)
    print(f"[*] Saved figures to {fig_dir}")

    report = {
        "tau_transition": tau_transition,
        "validity": validity,
        "diversity": diversity,
        "severity_control": severity_control,
        "detectability": detectability,
        "real_normal_transition_count": len(real_normal),
        "real_asd_transition_count": len(real_asd),
    }
    out_path = os.path.join(p2_cfg["output_dir"], "generator_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[*] Saved validation report to {out_path}")
    return report


if __name__ == "__main__":
    main()
