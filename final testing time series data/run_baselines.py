"""Add the two scoring baselines from the paper (same frozen encoder; thresholds = 95th percentile of validation-control patient scores) to comparison_results.json.
  persistence: A_t = ||z_{t+1} - z_t||      static: ||(z_t - mu_seed)/sd_seed|| / sqrt(D)  (distance to the training-control embedding centroid)
Run from the repo root after run_comparison.py."""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from src.inference import engine as E
from src.models.embedding_sequence import seed_embedding_sequence
D = "final testing time series data"
base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p3_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seeds = [int(s) for s in pools["seed_rois"]]; cd_ = p1_cfg["cache"]["dir"]
gin, gru, _ = P.load_models(p1_cfg, device)
def embeds(cache):
    with torch.no_grad(): return {s: seed_embedding_sequence(gin, cache, s, device).cpu().numpy() for s in seeds}
acc = {s: [] for s in seeds}
for sid in pools["train_control"]:
    z = embeds(P.load_subject_cache(sid, cd_))
    for s in seeds: acc[s].append(z[s])
mu = {s: np.concatenate(acc[s]).mean(0) for s in seeds}; sd = {s: np.concatenate(acc[s]).std(0) + 1e-6 for s in seeds}
def scores(z):
    per = np.concatenate([np.linalg.norm(z[s][1:] - z[s][:-1], axis=1) for s in seeds]).max()
    st = max((np.linalg.norm((z[s] - mu[s]) / sd[s], axis=1) / np.sqrt(128)).max() for s in seeds)
    return float(per), float(st)
val = np.array([scores(embeds(P.load_subject_cache(sid, cd_))) for sid in pools["val_control"]]); tau = np.percentile(val, 95, axis=0)
print("thresholds (persistence, static):", tau.round(3))
key = pd.read_csv(f"{D}/answer_key_and_results.csv"); res = json.load(open(f"{D}/comparison_results.json")); truth = (key.true_group == "injected_anomaly").values
S = np.array([scores(embeds(E.cache_from_timeseries(E.as_timeseries(np.loadtxt(f"{D}/scans/{fn}", delimiter=",")), fn))) for fn in key.file])
for j, name in enumerate(("persistence_baseline", "static_baseline")):
    flag = S[:, j] > tau[j]
    res["arms"][name] = dict(correct=int((flag == truth).sum()), accuracy=float((flag == truth).mean()), tp=int((flag & truth).sum()), fn=int((~flag & truth).sum()),
                             fp=int((flag & ~truth).sum()), tn=int((~flag & ~truth).sum()), flagged=int(flag.sum()), per_trial_accuracy=[])
    for i, s in enumerate(res["scans"]): s[name] = dict(prob=round(float(S[i, j]), 3), flagged=bool(flag[i]), correct=bool(flag[i] == truth[i]))
json.dump(res, open(f"{D}/comparison_results.json", "w"), indent=1)
for a, v in res["arms"].items(): print(f"{a:22s} correct {v['correct']}/100 TP {v['tp']} FN {v['fn']} FP {v['fp']} TN {v['tn']}")
