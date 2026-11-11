"""
Null reference for the few-shot recurrence statistic C(S) (Phase 1E).
Phase 1E flags a (subject, seed) pair when ANY transition exceeds tau_transition. With ~13-26 transitions per
seed, the chance that a *normal* subject trips that flag is high (multiple comparisons). We therefore measure the
same flag on held-out controls (val-control + test-control, never used for training) and on the ASD pool
(train-ASD, never used for training), and test whether ASD differs from the control null.
Also: reproducibility of per-seed recurrence rankings across independent support draws (from fewshot_results.json).
"""
import os, sys, json
import numpy as np, torch
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH/Output_to_be_shown/_scripts")
from common_model import get_models, cache_for, pools
from src.models.embedding_sequence import seed_embedding_sequence
from src.anomaly.scoring import transition_scores
ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
gin, gru, ckpt, device, p1 = get_models(); P = pools(); seeds = [int(s) for s in P["seed_rois"]]
tau = json.load(open(f"{ROOT}/outputs/experiments/phase1/metrics.json"))["tau_transition"]

def flags(sid):
    c = cache_for(sid); f = []; exc = []; nt = 0
    with torch.no_grad():
        for s in seeds:
            z = seed_embedding_sequence(gin, c, s, device); zh = gru(z.unsqueeze(0)).squeeze(0)
            a = transition_scores(z, zh).cpu().numpy()
            f.append(bool((a > tau).any())); exc.append(int((a > tau).sum())); nt += len(a)
    return np.array(f), sum(exc) / max(nt, 1), c["num_windows"]

test_ctrl = [s for s in P["test_all"] if cache_for(s)["label"] == 0]
ctrl_ids = list(P["val_control"]) + test_ctrl
asd_ids = list(P["train_asd_pool"])
def group(ids):
    F = []; R = []; W = []
    for sid in ids:
        f, r, w = flags(sid); F.append(f); R.append(r); W.append(w)
    return np.array(F), np.array(R), np.array(W)
Fc, Rc, Wc = group(ctrl_ids); Fa, Ra, Wa = group(asd_ids)
fc = Fc.mean(1); fa = Fa.mean(1)
U = mannwhitneyu(fa, fc, alternative="two-sided")
rng = np.random.default_rng(0)
def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {"tau_transition": tau, "n_control_heldout": len(ctrl_ids), "n_asd_pool": len(asd_ids),
  "mean_fraction_of_seeds_flagged_control": float(fc.mean()), "ci95_control": ci(fc),
  "mean_fraction_of_seeds_flagged_asd": float(fa.mean()), "ci95_asd": ci(fa),
  "mannwhitney_p_asd_vs_control": float(U.pvalue),
  "per_transition_exceedance_rate_control": float(Rc.mean()), "per_transition_exceedance_rate_asd": float(Ra.mean()),
  "mean_windows_control": float(Wc.mean()), "mean_windows_asd": float(Wa.mean())}
# per-seed rates and a seed-wise test with Bonferroni
pc = Fc.mean(0); pa = Fa.mean(0)
from scipy.stats import fisher_exact
pv = []
for j in range(len(seeds)):
    t = [[int(Fa[:, j].sum()), int((~Fa[:, j]).sum())], [int(Fc[:, j].sum()), int((~Fc[:, j]).sum())]]
    pv.append(fisher_exact(t)[1])
out["per_seed"] = [{"seed": seeds[j], "rate_control": float(pc[j]), "rate_asd": float(pa[j]), "fisher_p": float(pv[j])} for j in range(len(seeds))]
out["n_seeds_significant_bonferroni_0.05"] = int(sum(p < 0.05 / len(seeds) for p in pv))
out["min_fisher_p"] = float(min(pv))
# reproducibility of recurrence rankings across independent support draws
fs = json.load(open(f"{ROOT}/outputs/experiments/phase1/fewshot_results.json"))
rep = {}
for k, draws in fs.items():
    mats = []
    for d in draws:
        rec = d["recurrence"]; mats.append([rec[str(s)] if str(s) in rec else rec.get(s, np.nan) for s in seeds])
    mats = np.array(mats, float); rhos = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            if np.nanstd(mats[i]) > 0 and np.nanstd(mats[j]) > 0: rhos.append(spearmanr(mats[i], mats[j])[0])
    rep[k] = {"mean_recurrence_mean": float(np.mean([d["mean_recurrence"] for d in draws])),
              "mean_recurrence_std": float(np.std([d["mean_recurrence"] for d in draws])),
              "mean_pairwise_spearman_between_draws": float(np.mean(rhos)) if rhos else None, "n_pairs": len(rhos)}
out["recurrence_reproducibility"] = rep
json.dump(out, open(f"{ROOT}/paper/results/recurrence_null.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=2))
