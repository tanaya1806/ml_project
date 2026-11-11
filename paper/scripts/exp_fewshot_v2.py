"""Few-shot head v2: a principled redesign, SELECTED ON VALIDATION ONLY, reported once on test.
Why a redesign: the original head replaced a seed's max-score entry with the post-edit score of ONE transition, which is usually
lower than the seed's real maximum, so generated 'anomalies' often looked LESS anomalous than real scans. Fix: entry = max(old, new);
optionally mean-score features too, standardisation, stronger edit budgets, more generated examples, LR vs random forest.
Selection: config with best mean VALIDATION AUROC of the guided condition, applied identically to random and few-shot-only.
Outputs paper/results/fewshot_v2.json (all configs, val and test shown; chosen config marked). Run from repo root."""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p3_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seeds = pools["seed_rois"]; cd_ = p1_cfg["cache"]["dir"]
splits = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "subject_splits.json")))["splits"]
gin, gru, _ = P.load_models(p1_cfg, device)
tc, pool, vc, test_all, val_all = pools["train_control"], pools["train_asd_pool"], pools["val_control"], pools["test_all"], splits["val"]
P.assert_subject_isolation(pool, vc + test_all); P.assert_subject_isolation(tc, val_all + test_all); P.assert_subject_isolation(pool, val_all)
sidx = {int(s): i for i, s in enumerate(seeds)}

def feats(sid):
    c = P.load_subject_cache(sid, cd_); sc = P.subject_scores(gin, gru, sid, c, seeds, device)
    mx = np.array([sc[int(s)].max() if len(sc[int(s)]) else 0.0 for s in seeds]); mn = np.array([sc[int(s)].mean() if len(sc[int(s)]) else 0.0 for s in seeds]); nt = np.array([len(sc[int(s)]) for s in seeds])
    return mx, mn, nt, int(c["label"])
def table(ids): return {s: feats(s) for s in ids}
Ftc, Fpool, Fval, Ftest = table(tc), table(pool), table(val_all), table(test_all)
print("features computed")

def vec(mx, mn, kind): return np.concatenate([mx, mn]) if kind == "F40" else mx.copy()
def gen(method, budget, n, rseed):
    rng = np.random.default_rng(rseed); out = []; att = 0; frac, step = budget
    while len(out) < n and att < n * 5:
        sid = tc[rng.integers(0, len(tc))]; seed = int(seeds[rng.integers(0, len(seeds))]); cache = P.load_subject_cache(sid, cd_); att += 1
        if cache["num_windows"] < 2: continue
        t = int(rng.integers(0, cache["num_windows"] - 1))
        r = P.generate_anomaly(gin, gru, cache, seed, t, 0.9, device, method=method, edge_perturb_max_frac=frac, step_size_max=step, rng_seed=rseed + att)
        out.append((sid, sidx[seed], r["anomaly_score_before"], r["anomaly_score_after"]))
    return out
BUD = {"default": (0.15, 0.6), "strong": (0.30, 1.2)}; NG = 100; TR = [42 + i for i in range(5)]
G = {(m, b, t): gen(m, BUD[b], NG, t + (1000 if m == "random" else 0)) for m in ("structure_aware", "random") for b in BUD for t in TR}
print("generated")

def make_X(records, kind, upd):
    X = []
    for sid, k, before, after in records:
        mx, mn, nt, _ = Ftc[sid]; mx2, mn2 = mx.copy(), mn.copy()
        mx2[k] = after if upd == "replace" else max(mx[k], after); mn2[k] = mn[k] + (after - before) / max(nt[k], 1)
        X.append(vec(mx2, mn2, kind))
    return X
def clf_make(name):
    if name == "LR": return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1)
def evaluate(F, clf, X, y):
    ids = list(F); Xe = np.vstack([vec(F[s][0], F[s][1], KIND) for s in ids]); ye = np.array([F[s][3] for s in ids])
    sc = clf.predict_proba(Xe)[:, 1]; pr = clf.predict(Xe)
    return dict(auroc=float(roc_auc_score(ye, sc)), accuracy=float(accuracy_score(ye, pr)), f1=float(f1_score(ye, pr, zero_division=0)), precision=float(precision_score(ye, pr, zero_division=0)), recall=float(recall_score(ye, pr, zero_division=0)))
res = {}
for kind, upd, bud, cname in itertools.product(["F20", "F40"], ["replace", "max"], list(BUD), ["LR", "RF"]):
    KIND = kind; key = f"{kind}|{upd}|{bud}|{cname}"; res[key] = {}
    negs = [vec(Ftc[s][0], Ftc[s][1], kind) for s in tc]
    for shot in (1, 5, 10):
        for appr in ("few_shot_only", "random", "structure_aware"):
            vl, ts = [], []
            for t in TR:
                sup = P.sample_support_set(pool, shot, seed=t); pos = [vec(Fpool[s][0], Fpool[s][1], kind) for s in sup]
                add = [] if appr == "few_shot_only" else make_X(G[(appr, bud, t)], kind, upd)
                X = np.vstack(pos + add + negs); y = np.array([1] * (len(pos) + len(add)) + [0] * len(negs))
                clf = clf_make(cname).fit(X, y); vl.append(evaluate(Fval, clf, X, y)); ts.append(evaluate(Ftest, clf, X, y))
            m = lambda L: {k: [float(np.mean([d[k] for d in L])), float(np.std([d[k] for d in L]))] for k in L[0]}
            res[key].setdefault(shot, {})[appr] = {"val": m(vl), "test": m(ts)}
    print(key, "val AUROC guided:", round(np.mean([res[key][s]["structure_aware"]["val"]["auroc"][0] for s in (1, 5, 10)]), 3), "random:", round(np.mean([res[key][s]["random"]["val"]["auroc"][0] for s in (1, 5, 10)]), 3), flush=True)
sel = max(res, key=lambda k: np.mean([res[k][s]["structure_aware"]["val"]["auroc"][0] for s in (1, 5, 10)]))
json.dump({"chosen_by_validation": sel, "results": {k: {str(s): v for s, v in d.items()} for k, d in res.items()}}, open("paper/results/fewshot_v2.json", "w"), indent=1)
print("CHOSEN (validation):", sel)
for s in (1, 5, 10):
    for a in ("few_shot_only", "random", "structure_aware"): print(s, a, "TEST", {k: round(v[0], 3) for k, v in res[sel][s][a]["test"].items()})
