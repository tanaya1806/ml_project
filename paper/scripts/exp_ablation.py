"""
Additional experiments run for the paper (not part of the original Phase 1-3 pipeline):
  (1) scoring-function ablation on the SAME frozen encoder, SAME subjects, SAME protocol:
        GRU     : A_t = ||z_{t+1} - zhat_{t+1}||            (the proposed temporal predictor; reproduces Phase 1)
        persist : A_t = ||z_{t+1} - z_t||                    (no learned dynamics)
        static  : S_t = ||(z_t - mu_seed)/sd_seed||/sqrt(D)  (distance to the train-control embedding centroid; no time)
      subject score = max over seeds and windows; tau_subject = P95 of val-control subject scores.
  (2) bootstrap CIs / permutation test for AUROC, and a scan-length confound check.
Everything is computed from the real checkpoint and the real cached subgraphs.
"""
import os, sys, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH/Output_to_be_shown/_scripts")
from common_model import get_models, cache_for, pools
from src.models.embedding_sequence import seed_embedding_sequence

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
gin, gru, ckpt, device, p1 = get_models()
P = pools(); seeds = [int(s) for s in P["seed_rois"]]
rng = np.random.default_rng(0)

def seq(sid):
    c = cache_for(sid); out = {}
    with torch.no_grad():
        for s in seeds:
            z = seed_embedding_sequence(gin, c, s, device)
            zh = gru(z.unsqueeze(0)).squeeze(0)
            out[s] = (z.cpu().numpy(), zh.cpu().numpy())
    return c, out

# --- train-control centroids for the static baseline (controls only, same as GRU training pool)
acc = {s: [] for s in seeds}
for sid in P["train_control"]:
    _, o = seq(sid)
    for s in seeds: acc[s].append(o[s][0])
mu = {s: np.concatenate(acc[s]).mean(0) for s in seeds}
sd = {s: np.concatenate(acc[s]).std(0) + 1e-6 for s in seeds}
D = 128

def scores(sid):
    c, o = seq(sid)
    r = {"gru": [], "persist": [], "static": []}
    for s in seeds:
        z, zh = o[s]
        if len(z) < 2: continue
        r["gru"].append(np.linalg.norm(z[1:] - zh[:-1], axis=1))
        r["persist"].append(np.linalg.norm(z[1:] - z[:-1], axis=1))
        r["static"].append(np.linalg.norm((z - mu[s]) / sd[s], axis=1) / np.sqrt(D))
    flat = {k: np.concatenate(v) for k, v in r.items()}
    return c["label"], c["num_windows"], c["site"], {k: float(v.max()) for k, v in flat.items()}, \
           {k: float(v.mean()) for k, v in flat.items()}

def run(ids):
    L, W, S, mx, mn = [], [], [], [], []
    for sid in ids:
        l, w, s, a, b = scores(sid); L.append(l); W.append(w); S.append(s); mx.append(a); mn.append(b)
    return np.array(L), np.array(W), np.array(S), mx, mn

Lv, Wv, Sv, mxv, mnv = run(P["val_control"])
Lt, Wt, St, mxt, mnt = run(P["test_all"])
res = {"n_val_control": len(Lv), "n_test": len(Lt), "n_test_asd": int(Lt.sum()), "n_test_control": int((Lt == 0).sum()),
       "methods": {}}
perm_rng = np.random.default_rng(1)
for m in ["gru", "persist", "static"]:
    yv = np.array([d[m] for d in mxv]); yt = np.array([d[m] for d in mxt])
    tau = float(np.percentile(yv, 95))
    pred = yt > tau
    tp = int((pred & (Lt == 1)).sum()); fp = int((pred & (Lt == 0)).sum()); fn = int((~pred & (Lt == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    auc = roc_auc_score(Lt, yt); ap = average_precision_score(Lt, yt)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(Lt), len(Lt))
        if Lt[i].min() == Lt[i].max(): continue
        boots.append(roc_auc_score(Lt[i], yt[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    null = [roc_auc_score(perm_rng.permutation(Lt), yt) for _ in range(5000)]
    p_two = float(np.mean(np.abs(np.array(null) - 0.5) >= abs(auc - 0.5)))
    ytm = np.array([d[m] for d in mnt])
    res["methods"][m] = {"tau_subject_P95_valcontrol": tau, "auroc": auc, "auroc_ci95": [float(lo), float(hi)],
        "perm_p_vs_chance": p_two, "auprc": ap, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
        "n_flagged": int(pred.sum()), "auroc_meanA_aggregation": roc_auc_score(Lt, ytm)}
    np.save(os.path.join(ROOT, f"paper/results/test_scores_{m}.npy"), yt)
    np.save(os.path.join(ROOT, f"paper/results/val_scores_{m}.npy"), yv)
np.save(os.path.join(ROOT, "paper/results/test_labels.npy"), Lt)
np.save(os.path.join(ROOT, "paper/results/val_labels.npy"), Lv)
# --- scan-length confound
res["confound"] = {
  "auroc_of_num_windows_alone_for_label": float(roc_auc_score(Lt, Wt)),
  "mean_windows_test_control": float(Wt[Lt == 0].mean()), "mean_windows_test_asd": float(Wt[Lt == 1].mean()),
  "spearman_gru_maxscore_vs_num_windows": float(__import__("scipy.stats", fromlist=["x"]).spearmanr([d["gru"] for d in mxt], Wt)[0]),
  "n_sites_test": int(len(set(St)))}
json.dump(res, open(os.path.join(ROOT, "paper/results/ablation.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
