"""Re-run the committed Phase 3 few-shot comparison (same seeds/code) but ALSO record accuracy and confusion counts per trial.
Writes paper/results/fewshot_with_accuracy.json only (committed outputs are not touched). Run from repo root."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

def fit_and_evaluate(pos, neg, test_vectors, test_labels, cfg):
    X = np.vstack(pos + neg); y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=1000).fit(X, y)
    ids = list(test_vectors.keys()); Xt = np.vstack([test_vectors[s] for s in ids]); yt = np.array([test_labels[s] for s in ids])
    sc = clf.predict_proba(Xt)[:, 1]; pr = clf.predict(Xt); tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
    return dict(precision=float(precision_score(yt, pr, zero_division=0)), recall=float(recall_score(yt, pr, zero_division=0)), f1=float(f1_score(yt, pr, zero_division=0)),
                auroc=float(roc_auc_score(yt, sc)), auprc=float(average_precision_score(yt, sc)), accuracy=float(accuracy_score(yt, pr)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p3_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seed_rois = pools["seed_rois"]; cache_dir = p1_cfg["cache"]["dir"]
gin, gru, ckpt = P.load_models(p1_cfg, device)
tc, pool, vc, test_all = pools["train_control"], pools["train_asd_pool"], pools["val_control"], pools["test_all"]
P.assert_subject_isolation(pool, vc + test_all); P.assert_subject_isolation(tc, vc + test_all)
negd = P.compute_vectors_for_pool(gin, gru, tc, seed_rois, cache_dir, device); neg = list(negd.values())
testv = P.compute_vectors_for_pool(gin, gru, test_all, seed_rois, cache_dir, device); testl = {s: P.load_subject_cache(s, cache_dir)["label"] for s in test_all}
asdv = P.compute_vectors_for_pool(gin, gru, pool, seed_rois, cache_dir, device)
res = {}
for shot in p3_cfg["shots"]:
    r = {"few_shot_only": [], "few_shot_plus_structure_aware": [], "few_shot_plus_random": []}
    for trial in range(p3_cfg["num_trials"]):
        ts = p3_cfg["random_seed"] + trial; sup = P.sample_support_set(pool, shot, seed=ts); P.assert_subject_isolation(sup, vc + test_all); rp = [asdv[s] for s in sup]
        r["few_shot_only"].append(fit_and_evaluate(rp, neg, testv, testl, p3_cfg["classifier"]))
        g = P.generate_expansion_vectors(gin, gru, negd, tc, seed_rois, cache_dir, device, method="structure_aware", severity=p3_cfg["generation_severity"], n=p3_cfg["num_generated_per_trial"], rng_seed=ts)
        r["few_shot_plus_structure_aware"].append(fit_and_evaluate(rp + g, neg, testv, testl, p3_cfg["classifier"]))
        g = P.generate_expansion_vectors(gin, gru, negd, tc, seed_rois, cache_dir, device, method="random", severity=p3_cfg["generation_severity"], n=p3_cfg["num_generated_per_trial"], rng_seed=ts + 1000)
        r["few_shot_plus_random"].append(fit_and_evaluate(rp + g, neg, testv, testl, p3_cfg["classifier"]))
    res[f"{shot}-shot"] = r
# agreement with the committed run
old = json.load(open("outputs/experiments/phase3/raw_trial_results.json")); mx = 0
for s in res:
    for c in res[s]:
        for a, b in zip(res[s][c], old[s][c]):
            for k in ("precision", "recall", "f1", "auroc", "auprc"): mx = max(mx, abs(a[k] - b[k]))
print("max abs difference vs committed per-trial metrics:", mx)
agg = {s: {c: {m: [float(np.mean([t[m] for t in res[s][c]])), float(np.std([t[m] for t in res[s][c]]))] for m in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")} for c in res[s]} for s in res}
json.dump({"trials": res, "agg_mean_std": agg, "max_diff_vs_committed": mx}, open("paper/results/fewshot_with_accuracy.json", "w"), indent=1)
for s in agg:
    for c in agg[s]: print(s, c, {m: round(v[0], 3) for m, v in agg[s][c].items()})
