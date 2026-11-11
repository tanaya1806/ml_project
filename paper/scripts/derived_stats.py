"""Derived statistics from the real result files, used verbatim in the paper text/tables."""
import json, os, sys, numpy as np
ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
sys.path.insert(0, f"{ROOT}/Output_to_be_shown/_scripts")
out = {}
# ---- Phase 3: chance-level F1 for a prevalence-matched random flagger at the same flag rate (derivable only when TP>0)
f3 = json.load(open(f"{ROOT}/outputs/experiments/phase3/final_comparison.json"))
raw = json.load(open(f"{ROOT}/outputs/experiments/phase3/raw_trial_results.json"))
P_TEST, N_TEST = 61, 131
pi = P_TEST / N_TEST
rows = {}
for shot, conds in raw.items():
    for cond, trials in conds.items():
        chance_f1, q, f1, auroc, dF1 = [], [], [], [], []
        ok = True
        for t in trials:
            tp = round(t["recall"] * P_TEST)
            f1.append(t["f1"]); auroc.append(t["auroc"])
            if tp > 0 and t["precision"] > 0:
                pred = tp / t["precision"]; qq = pred / N_TEST; q.append(qq)
                chance_f1.append(2 * pi * qq / (pi + qq))
            else:
                ok = False
        rows[f"{shot}/{cond}"] = {"mean_flag_rate": float(np.mean(q)) if ok else None,
            "mean_chance_f1_at_flag_rate": float(np.mean(chance_f1)) if ok else None,
            "mean_f1": float(np.mean(f1)), "mean_auroc": float(np.mean(auroc)), "all_trials_have_TP": ok}
out["phase3_chance_f1"] = rows
# paired per-trial differences vs few-shot-only
pd_ = {}
for shot in raw:
    a = np.array([t["f1"] for t in raw[shot]["few_shot_only"]])
    for c in ["few_shot_plus_structure_aware", "few_shot_plus_random"]:
        b = np.array([t["f1"] for t in raw[shot][c]])
        pd_[f"{shot}/{c}/dF1"] = {"mean": float((b - a).mean()), "min": float((b - a).min()), "max": float((b - a).max())}
        aa = np.array([t["auroc"] for t in raw[shot]["few_shot_only"]]); bb = np.array([t["auroc"] for t in raw[shot][c]])
        pd_[f"{shot}/{c}/dAUROC"] = {"mean": float((bb - aa).mean()), "min": float((bb - aa).min()), "max": float((bb - aa).max())}
    s = np.array([t["f1"] for t in raw[shot]["few_shot_plus_structure_aware"]]); r = np.array([t["f1"] for t in raw[shot]["few_shot_plus_random"]])
    pd_[f"{shot}/SA_minus_random/dF1"] = {"mean": float((s - r).mean()), "min": float((s - r).min()), "max": float((s - r).max())}
    s = np.array([t["auroc"] for t in raw[shot]["few_shot_plus_structure_aware"]]); r = np.array([t["auroc"] for t in raw[shot]["few_shot_plus_random"]])
    pd_[f"{shot}/SA_minus_random/dAUROC"] = {"mean": float((s - r).mean()), "min": float((s - r).min()), "max": float((s - r).max())}
out["phase3_paired"] = pd_
# ---- Phase 2: fraction of records whose score increased
recs = json.load(open(f"{ROOT}/outputs/experiments/phase2/generated_anomalies.json"))
g = {}
for r in recs:
    k = f"{r['method']}/{r['severity_name']}"
    g.setdefault(k, []).append(r["anomaly_score_after"] - r["anomaly_score_before"])
out["phase2_delta"] = {k: {"n": len(v), "frac_increased": float(np.mean(np.array(v) > 0)), "median_delta": float(np.median(v)),
                           "mean_delta": float(np.mean(v))} for k, v in g.items()}
out["phase2_n_edges_changed_mean_high"] = float(np.mean([r["num_edges_changed"] for r in recs if r["severity_name"] == "high" and r["method"] == "structure_aware"]))
out["phase2_n_base_subjects"] = len(set(r["subject_id"] for r in recs))
# ---- subgraph size / stability statistics over the cache
from common_model import cache_for, pools
P = pools(); seeds = [int(s) for s in P["seed_rois"]]
nodes, jac, ecount, nwin = [], [], [], []
rng = np.random.default_rng(0)
for sid in P["all_subjects"]:
    c = cache_for(sid); nwin.append(c["num_windows"])
    for s in seeds:
        nodes.append(len(c[f"roi_indices_{s}"]))
for sid in rng.choice(P["all_subjects"], 60, replace=False):
    c = cache_for(sid)
    for s in seeds:
        sets = []
        for ei in c[f"edge_index_{s}"]:
            ei = np.asarray(ei); sets.append({(min(a, b), max(a, b)) for a, b in zip(ei[0], ei[1])})
            ecount.append(len(sets[-1]))
        for a, b in zip(sets[:-1], sets[1:]):
            if a | b: jac.append(len(a & b) / len(a | b))
out["subgraph"] = {"n_subgraph_sequences": len(nodes), "nodes_mean": float(np.mean(nodes)), "nodes_median": float(np.median(nodes)),
   "nodes_min": int(min(nodes)), "nodes_max": int(max(nodes)), "nodes_p5_p95": [float(np.percentile(nodes, 5)), float(np.percentile(nodes, 95))],
   "undirected_edges_per_window_mean": float(np.mean(ecount)), "consecutive_edge_set_jaccard_stable_mean": float(np.mean(jac)),
   "windows_per_subject_min": int(min(nwin)), "windows_per_subject_max": int(max(nwin)), "windows_per_subject_mean": float(np.mean(nwin))}
json.dump(out, open(f"{ROOT}/paper/results/derived_stats.json", "w"), indent=2)
print(json.dumps(out, indent=2))
