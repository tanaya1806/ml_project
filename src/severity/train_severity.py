"""Exploratory severity estimation on ASD subjects with an ADOS-2 calibrated severity score (CSS).

Features: 40-D per-seed max / mean transition scores from the frozen Phase 1 model (no re-training of the detector).
Targets: CSS (regression) and analyst-defined bands low(<=5)/moderate(6-7)/high(>=8).
Fit on train-split ASD subjects with CSS; evaluate on val+test ASD with CSS, and by repeated stratified CV over all.
Everything is reported as found, including null results. Run from repo root.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from src.inference import engine as E
from src.data.subgraph_cache import load_subject_cache
from src.severity.linkage import build_link, severity_band

OUT = "outputs/experiments/severity"; os.makedirs(OUT, exist_ok=True)
link = build_link(); link.to_csv("data/processed/metadata/subject_link_phenotype.csv", index=False)
sp = json.load(open("data/processed/metadata/subject_splits.json"))["splits"]
split_of = {s: k for k, v in sp.items() for s in v}
feat_path = f"{OUT}/features.npz"
if os.path.exists(feat_path):
    z = np.load(feat_path, allow_pickle=True); ids, X = list(z["ids"]), z["X"]
else:
    ids, X = [], []
    for sid in link.subject_id:
        sc, _ = E.seed_scores_from_cache(load_subject_cache(sid, "data/processed/subgraph_cache"))
        ids.append(sid); X.append(E.features_from_scores(sc))
    X = np.array(X); np.savez(feat_path, ids=np.array(ids), X=X)
link = link.set_index("subject_id").loc[ids].reset_index()
link["split"] = [split_of[s] for s in link.subject_id]
link["S"] = X[:, :20].max(1)

a = link[(link.label == 1) & link.ados_css.notna()].copy()
a["band"] = a.ados_css.map(severity_band)
Xa = X[[ids.index(s) for s in a.subject_id]]
res = dict(n_asd_with_css=int(len(a)), band_counts=a.band.value_counts().to_dict(),
           split_counts=a.groupby("split").size().to_dict(), band_cutoffs="low<=5, moderate 6-7, high>=8 (analyst-defined)")
tr = (a.split == "train").values; te = ~tr
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
clf = mk().fit(Xa[tr], a.band[tr]); pred = clf.predict(Xa[te]); y = a.band[te].values
maj = a.band[tr].value_counts().idxmax()
res["holdout"] = dict(n_train=int(tr.sum()), n_eval=int(te.sum()), accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                      majority_class=maj, majority_accuracy=float((y == maj).mean()), chance_uniform=1 / 3)
rg = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(Xa[tr], a.ados_css[tr])
rho, p = spearmanr(rg.predict(Xa[te]), a.ados_css[te]); res["holdout"]["css_spearman_pred_vs_true"] = [float(rho), float(p)]
rho, p = spearmanr(a.S, a.ados_css); res["subject_score_vs_css_spearman_all_asd"] = [float(rho), float(p)]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42); accs, f1s, maccs = [], [], []
yb = a.band.values
for tri, tei in cv.split(Xa, yb):
    m = mk().fit(Xa[tri], yb[tri]); pr = m.predict(Xa[tei])
    accs.append(accuracy_score(yb[tei], pr)); f1s.append(f1_score(yb[tei], pr, average="macro"))
    maccs.append(float((yb[tei] == pd.Series(yb[tri]).value_counts().idxmax()).mean()))
rng = np.random.default_rng(0); perm = []
for _ in range(200):
    yp = rng.permutation(yb); pa = []
    for tri, tei in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1).split(Xa, yp):
        pa.append(f1_score(yp[tei], mk().fit(Xa[tri], yp[tri]).predict(Xa[tei]), average="macro"))
    perm.append(np.mean(pa))
res["cv_5fold_x10"] = dict(accuracy_mean=float(np.mean(accs)), accuracy_sd=float(np.std(accs)), macro_f1_mean=float(np.mean(f1s)),
                           majority_accuracy_mean=float(np.mean(maccs)), perm_p_macro_f1=float((np.sum(np.array(perm) >= np.mean(f1s)) + 1) / 201))
# deployable model = the train-split-only model above, so val/test samples shown in the app are out-of-sample
import joblib
final = clf; joblib.dump(dict(model=final, classes=list(final.classes_)), f"{OUT}/severity_model.joblib")
json.dump(res, open(f"{OUT}/severity_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
