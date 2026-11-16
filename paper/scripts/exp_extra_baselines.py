"""Extra baselines + full metric set, on (A) the real held-out test set (131 patients, ASD vs control) and (B) the 100 synthetic scans.
New baselines (all trained on training CONTROLS only, threshold = 95th percentile of validation-control scores, like the proposed detector):
  whole_brain_gin_gru : same GIN+GRU recipe (5 epochs, Adam 1e-3, VICReg hinge 0.5) on the FULL 360-node graph per window instead of seed subgraphs
  ocsvm / iforest / autoencoder : on 720-D node-strength features (mean and std over windows of each region's signed strength)
Existing models' real-test scores come from paper/results/*.npy, synthetic-set results from final testing time series data/comparison_results.json.
Run from the repo root. Writes paper/results/extra_baselines.json."""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
sys.path.insert(0, os.getcwd())
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef
from torch_geometric.data import Data, Batch
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs
D = "final testing time series data"; R = "paper/results"
torch.manual_seed(42); np.random.seed(42)
pools = json.load(open("data/processed/metadata/phase1_subject_pools.json"))
cfg = yaml.safe_load(open("configs/phase1.yaml"))
tc, vc, te = pools["train_control"], pools["val_control"], pools["test_all"]

def load(sid):
    d = np.load(f"data/processed/graphs/{sid}.npz", allow_pickle=True)
    return d["edge_indices"].astype(np.int64), d["edge_weights"].astype(np.float32), int(d["label"])
def from_ts(ts):
    fcs = compute_sliding_window_fc(np.asarray(ts, dtype=np.float32), window_size=30, stride=10, remove_diagonal=True)
    g = construct_temporal_graphs(fcs, sparsity_ratio=0.20, keep_signed_weights=True)
    return np.stack([x.edge_index for x in g]).astype(np.int64), np.stack([x.edge_weight for x in g]).astype(np.float32)
def feat(ei, ew):
    S = np.stack([np.bincount(ei[t, 0], weights=ew[t], minlength=360) for t in range(len(ei))])
    return np.concatenate([S.mean(0), S.std(0)])
t0 = time.time()
G = {s: load(s) for s in tc + vc + te}; print("graphs loaded", len(G), round(time.time() - t0), "s", flush=True)
key = __import__("pandas").read_csv(f"{D}/answer_key_and_results.csv")
SY = [from_ts(np.loadtxt(f"{D}/scans/{f}", delimiter=",")) for f in key.file]; print("synthetic graphs ready", flush=True)
y_te = np.array([G[s][2] for s in te]); y_sy = (key.true_group == "injected_anomaly").values.astype(int)

# ---------------- feature baselines ----------------
Xtc = np.array([feat(*G[s][:2]) for s in tc]); Xvc = np.array([feat(*G[s][:2]) for s in vc]); Xte = np.array([feat(*G[s][:2]) for s in te]); Xsy = np.array([feat(*g) for g in SY])
sc = StandardScaler().fit(Xtc); Z = lambda X: sc.transform(X)
scores = {"real": {}, "syn": {}}
def put(name, f):                                   # f: standardized features -> anomaly score (higher = more abnormal)
    v = f(Z(Xvc)); scores["real"][name] = (v, f(Z(Xte))); scores["syn"][name] = (v, f(Z(Xsy)))
oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Z(Xtc)); put("ocsvm", lambda X: -oc.decision_function(X))
iso = IsolationForest(n_estimators=300, random_state=0).fit(Z(Xtc)); put("iforest", lambda X: -iso.score_samples(X))
class AE(nn.Module):
    def __init__(s, d): super().__init__(); s.e = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 32)); s.d = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(s, x): return s.d(s.e(x))
ae = AE(Xtc.shape[1]); opt = torch.optim.Adam(ae.parameters(), 1e-3, weight_decay=1e-4); xt = torch.tensor(Z(Xtc), dtype=torch.float32)
for ep in range(300):
    opt.zero_grad(); loss = F.mse_loss(ae(xt), xt); loss.backward(); opt.step()
ae.eval(); put("autoencoder", lambda X: ((ae(torch.tensor(X, dtype=torch.float32)) - torch.tensor(X, dtype=torch.float32)) ** 2).mean(1).detach().numpy())
print("feature baselines done", round(time.time() - t0), "s", flush=True)

# ---------------- whole-brain GIN + GRU ----------------
gin = GINEncoder(**cfg["gin"]); gru = TemporalGRU(embedding_dim=cfg["gin"]["embedding_dim"], hidden_dim=cfg["gru"]["hidden_dim"], num_layers=cfg["gru"]["num_layers"])
def zseq(ei, ew):
    data = [Data(x=torch.ones((360, 1)), edge_index=torch.tensor(ei[t]), edge_weight=torch.tensor(ew[t])) for t in range(len(ei))]
    b = Batch.from_data_list(data); return gin(b.x, b.edge_index, b.edge_weight, b.batch)
opt = torch.optim.Adam(list(gin.parameters()) + list(gru.parameters()), lr=1e-3); rng = np.random.default_rng(42)
for ep in range(5):
    order = rng.permutation(len(tc)); tot = []
    for i in range(0, len(order), 16):
        losses, zs = [], []
        for j in order[i:i + 16]:
            ei, ew, _ = G[tc[j]]
            if len(ei) < 2: continue
            z = zseq(ei, ew); zh = gru(z.unsqueeze(0)).squeeze(0); losses.append(F.mse_loss(zh[:-1], z[1:])); zs.append(z)
        zc = torch.cat(zs); var = torch.relu(1.0 - torch.sqrt(zc.var(0, unbiased=False) + 1e-4)).mean()
        loss = torch.stack(losses).mean() + 0.5 * var; opt.zero_grad(); loss.backward(); opt.step(); tot.append(loss.item())
    print("whole-brain epoch", ep + 1, "loss", round(float(np.mean(tot)), 3), round(time.time() - t0), "s", flush=True)
gin.eval(); gru.eval()
def wb(ei, ew):
    with torch.no_grad():
        z = zseq(ei, ew); zh = gru(z.unsqueeze(0)).squeeze(0); return float(torch.linalg.norm(z[1:] - zh[:-1], dim=1).max())
v = np.array([wb(*G[s][:2]) for s in vc]); scores["real"]["whole_brain_gin_gru"] = (v, np.array([wb(*G[s][:2]) for s in te])); scores["syn"]["whole_brain_gin_gru"] = (v, np.array([wb(*g) for g in SY]))
print("whole-brain done", round(time.time() - t0), "s", flush=True)

# ---------------- metrics ----------------
def metrics(y, score, flag, n_boot=1000, seed=0):
    y = np.asarray(y); flag = np.asarray(flag).astype(int); tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    def one(yy, ss, ff):
        a = (yy == ff).mean(); tp_ = ((ff == 1) & (yy == 1)).sum(); tn_ = ((ff == 0) & (yy == 0)).sum(); fp_ = ((ff == 1) & (yy == 0)).sum(); fn_ = ((ff == 0) & (yy == 1)).sum()
        sens = tp_ / max(tp_ + fn_, 1); spec = tn_ / max(tn_ + fp_, 1); pr = tp_ / max(tp_ + fp_, 1)
        f1 = 2 * pr * sens / max(pr + sens, 1e-12); mcc = matthews_corrcoef(yy, ff) if len(set(ff)) > 1 else 0.0
        auc = roc_auc_score(yy, ss) if len(set(yy)) > 1 else float("nan")
        return dict(accuracy=a, sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2, precision=pr, f1=f1, mcc=mcc, auroc=auc)
    m = one(y, score, flag); m["auprc"] = float(average_precision_score(y, score)); m.update(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    rng = np.random.default_rng(seed); bs = {k: [] for k in ("accuracy", "f1", "mcc", "auroc")}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        o = one(y[i], np.asarray(score)[i], flag[i])
        for k in bs: bs[k].append(o[k])
    m["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
out = {"real": {}, "syn": {}, "curves": {}, "n": {"real_test": int(len(y_te)), "real_asd": int(y_te.sum()), "syn": int(len(y_sy)), "syn_anomaly": int(y_sy.sum())}}
# real test: existing scores
for nm, lab in (("gru", "temporal_subgraph_detector"), ("static", "static_baseline"), ("persist", "persistence_baseline")):
    v = np.load(f"{R}/val_scores_{nm}.npy"); t = np.load(f"{R}/test_scores_{nm}.npy"); scores["real"][lab] = (v, t)
assert np.array_equal(np.load(f"{R}/test_labels.npy"), y_te), "test label order differs"
for lab, (v, t) in scores["real"].items():
    tau = float(np.percentile(v, 95)); out["real"][lab] = metrics(y_te, t, t > tau); out["curves"].setdefault("real", {})[lab] = t.tolist()
# synthetic: score-based baselines
cmp_ = json.load(open(f"{D}/comparison_results.json"))
arm_map = {"screening_detector": "temporal_subgraph_detector", "static_baseline": "static_baseline", "persistence_baseline": "persistence_baseline", "few_shot_only": "few_shot_only", "random": "few_shot_random_anomalies", "structure_aware": "few_shot_structure_aware_anomalies"}
for a, lab in arm_map.items():
    sc_ = np.array([s[a]["prob"] for s in cmp_["scans"]]); fl = np.array([s[a]["flagged"] for s in cmp_["scans"]])
    out["syn"][lab] = metrics(y_sy, sc_, fl); out["curves"].setdefault("syn", {})[lab] = sc_.tolist(); out["curves"].setdefault("syn_flag", {})[lab] = fl.astype(int).tolist()
for lab in ("ocsvm", "iforest", "autoencoder", "whole_brain_gin_gru"):
    v, t = scores["syn"][lab]; tau = float(np.percentile(v, 95)); out["syn"][lab] = metrics(y_sy, t, t > tau); out["curves"]["syn"][lab] = t.tolist(); out["curves"]["syn_flag"][lab] = (t > tau).astype(int).tolist(); out["curves"].setdefault("val", {})[lab] = v.tolist()
for lab, nm in (("temporal_subgraph_detector", "gru"), ("static_baseline", "static"), ("persistence_baseline", "persist")): out["curves"].setdefault("val", {})[lab] = np.load(f"{R}/val_scores_{nm}.npy").tolist()
out["curves"]["y_real"] = y_te.tolist(); out["curves"]["y_syn"] = y_sy.tolist(); out["curves"]["val_gru_scores"] = np.load(f"{R}/val_scores_gru.npy").tolist()
out["curves"]["strength"] = [s["strength"] for s in cmp_["scans"]]
json.dump(out, open(f"{R}/extra_baselines.json", "w"))
for split in ("real", "syn"):
    print("\n==", split)
    for k, m in out[split].items(): print(f"{k:38s} acc {m['accuracy']:.3f} sens {m['sensitivity']:.2f} spec {m['specificity']:.2f} F1 {m['f1']:.2f} MCC {m['mcc']:.2f} AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f}")
