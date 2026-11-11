"""Detection of GENERATED anomalies on held-out TEST-control scans (not ASD diagnosis).
For each of the 70 test controls x 20 seed regions: one random transition; guided and random edits with the SAME budget (severity 0.9).
Negative = the real unedited transition score, positive = score after the edit. Detection threshold = transition threshold calibrated on
validation controls (7.435). These test scans were never used for training, calibration, support sets or generation.
Caveat (stated in the paper): the same model guides the edits and scores them. Writes paper/results/synth_benchmark.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from sklearn.metrics import roc_auc_score
base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p3_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seeds = pools["seed_rois"]; cd_ = p1_cfg["cache"]["dir"]
tau = json.load(open("outputs/experiments/phase1/metrics.json"))["tau_transition"]
gin, gru, _ = P.load_models(p1_cfg, device)
test_ctl = [s for s in pools["test_all"] if P.load_subject_cache(s, cd_)["label"] == 0]; print(len(test_ctl), "test controls")
out = {}
for method in ("structure_aware", "random"):
    before, after = [], []; rng = np.random.default_rng(7)
    for sid in test_ctl:
        c = P.load_subject_cache(sid, cd_)
        for seed in seeds:
            t = int(rng.integers(0, c["num_windows"] - 1)); r = P.generate_anomaly(gin, gru, c, int(seed), t, 0.9, device, method=method, rng_seed=int(rng.integers(1e9)))
            before.append(r["anomaly_score_before"]); after.append(r["anomaly_score_after"])
    b, a = np.array(before), np.array(after); y = np.r_[np.zeros(len(b)), np.ones(len(a))]; sc = np.r_[b, a]; pr = sc > tau
    tp, fp, tn, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / (tp + fn); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    out[method] = dict(n_pairs=len(b), accuracy=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1, auroc=float(roc_auc_score(y, sc)), mean_before=float(b.mean()), mean_after=float(a.mean()), tp=tp, fp=fp, tn=tn, fn=fn)
    print(method, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[method].items()})
out["tau_transition"] = tau; json.dump(out, open("paper/results/synth_benchmark.json", "w"), indent=1)
