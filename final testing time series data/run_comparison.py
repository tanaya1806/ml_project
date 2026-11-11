"""Measure three few-shot heads on the 100 synthetic scans (real numbers, nothing hand-set). Run from the repo root.
Heads = the project's Phase 3 design (frozen Phase 1 GIN+GRU; logistic regression on the 20 per-seed max scores; config chosen on validation):
  few_shot_only   10 real ASD scans (+) vs 328 training controls (-)
  random          the same + 100 randomly-edited anomalies (+)
  structure_aware the same + 100 structure-aware (gradient-guided) anomalies (+)
Each arm is trained 5 times (different support draws / generation seeds); the verdict uses the mean probability, threshold 0.5.
A scan counts as 'flagged' when the arm predicts positive; correct = flagged for injected-anomaly scans, not flagged for normal-like scans."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from src.inference import engine as E
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D = "final testing time series data"
key = pd.read_csv(f"{D}/answer_key_and_results.csv")
base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p3_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seeds = pools["seed_rois"]; cd_ = p1_cfg["cache"]["dir"]
assert [int(s) for s in seeds] == [int(s) for s in E.select_seed_rois(20, 42)], "seed order differs between training and the engine"
splits = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "subject_splits.json")))["splits"]
gin, gru, _ = P.load_models(p1_cfg, device)
tc, pool = pools["train_control"], pools["train_asd_pool"]
P.assert_subject_isolation(pool, splits["val"] + splits["test"]); P.assert_subject_isolation(tc, splits["val"] + splits["test"])
sidx = {int(s): i for i, s in enumerate(seeds)}
def mx(sid):
    c = P.load_subject_cache(sid, cd_); sc = P.subject_scores(gin, gru, sid, c, seeds, device)
    return np.array([sc[int(s)].max() if len(sc[int(s)]) else 0.0 for s in seeds])
Ftc = {s: mx(s) for s in tc}; Fpool = {s: mx(s) for s in pool}; print("training features ready", flush=True)
def gen(method, n, rseed):                                    # same recipe as paper/scripts/exp_fewshot_v2.py, default budget (0.15, 0.6), severity 0.9
    rng = np.random.default_rng(rseed); out = []; att = 0
    while len(out) < n and att < n * 5:
        sid = tc[rng.integers(0, len(tc))]; seed = int(seeds[rng.integers(0, len(seeds))]); cache = P.load_subject_cache(sid, cd_); att += 1
        if cache["num_windows"] < 2: continue
        t = int(rng.integers(0, cache["num_windows"] - 1))
        r = P.generate_anomaly(gin, gru, cache, seed, t, 0.9, device, method=method, edge_perturb_max_frac=0.15, step_size_max=0.6, rng_seed=rseed + att)
        v = Ftc[sid].copy(); v[sidx[seed]] = max(v[sidx[seed]], r["anomaly_score_after"]); out.append(v)   # 'max' update: the edited seed's score can only go up
    return out
# the 100 scans -> 20-D feature (max score per seed), via exactly what the web app runs
rows = []; X = []
for fn in key.file:
    r = E.analyze(np.loadtxt(f"{D}/scans/{fn}", delimiter=","), fn); X.append(r["seed_max"])
X = np.array(X); truth = (key.true_group == "injected_anomaly").values
arms = ("few_shot_only", "random", "structure_aware"); prob = {a: [] for a in arms}; negs = [Ftc[s] for s in tc]
for t in [42 + i for i in range(5)]:
    sup = P.sample_support_set(pool, 10, seed=t); pos = [Fpool[s] for s in sup]
    for a in arms:
        add = [] if a == "few_shot_only" else gen("structure_aware" if a == "structure_aware" else "random", 100, t + (1000 if a == "random" else 0))
        Xt = np.vstack(pos + add + negs); yt = np.array([1] * (len(pos) + len(add)) + [0] * len(negs))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(Xt, yt)
        prob[a].append(clf.predict_proba(X)[:, 1])
    print("trial", t, "done", flush=True)
out = dict(note="Synthetic anomaly-injection scans, not patient data. Labels: injected anomaly vs normal-like.", n=len(key), n_anomaly=int(truth.sum()), n_normal=int((~truth).sum()), arms={}, scans=[])
for a in arms:
    p = np.mean(prob[a], axis=0); flag = p > 0.5; per_trial = [float((((q > 0.5) == truth)).mean()) for q in prob[a]]
    out["arms"][a] = dict(correct=int((flag == truth).sum()), accuracy=float((flag == truth).mean()), tp=int((flag & truth).sum()), fn=int((~flag & truth).sum()),
                          fp=int((flag & ~truth).sum()), tn=int((~flag & ~truth).sum()), flagged=int(flag.sum()), per_trial_accuracy=per_trial)
det = (key.detector_output == "flagged").values
out["arms"]["screening_detector"] = dict(correct=int((det == truth).sum()), accuracy=float((det == truth).mean()), tp=int((det & truth).sum()), fn=int((~det & truth).sum()),
                                         fp=int((det & ~truth).sum()), tn=int((~det & ~truth).sum()), flagged=int(det.sum()), per_trial_accuracy=[])
for i, r in key.iterrows():
    s = dict(file=r.file, truth="injected anomaly" if truth[i] else "normal-like", strength=r.synthetic_strength if isinstance(r.synthetic_strength, str) else "")
    for a in arms:
        p = float(np.mean([q[i] for q in prob[a]])); s[a] = dict(prob=round(p, 3), flagged=bool(p > 0.5), correct=bool((p > 0.5) == truth[i]))
    s["screening_detector"] = dict(prob=float(r.subject_score), flagged=bool(det[i]), correct=bool(det[i] == truth[i])); out["scans"].append(s)
json.dump(out, open(f"{D}/comparison_results.json", "w"), indent=1)
for a, v in out["arms"].items(): print(f"{a:20s} correct {v['correct']}/100  acc {v['accuracy']:.2f}  TP {v['tp']} FN {v['fn']} FP {v['fp']} TN {v['tn']}  per-trial {np.round(v['per_trial_accuracy'],2).tolist()}")
