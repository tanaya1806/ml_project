"""Model-wise comparison: same data, splits, loss (MSE + variance hinge), optimiser, epochs and seed for every variant.
Variants: GIN+GRU (proposed, committed checkpoint), GCN+GRU, GraphConv+GRU, GIN+LSTM (GCN/GraphConv are weight-aware; GAT/GraphSAGE
cannot use a scalar signed edge weight and are therefore not part of the comparison). For each trained model:
  (a) ASD detection on the 131 test patients: AUROC and accuracy at the patient threshold (95th pct of validation controls)
  (b) detection of generated anomalies on the 70 test controls (1,400 edited transitions per method, severity 0.9)
Writes paper/results/model_compare.json. Run from repo root: python3 paper/scripts/exp_model_compare.py [variant ...]"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.getcwd())
import src.evaluate.phase3_final_comparison as P
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.models.embedding_sequence import seed_embedding_sequence
from src.training.train_phase1 import _preload_caches, _run_batch, evaluate_split
from src.anomaly.scoring import transition_scores
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from sklearn.metrics import roc_auc_score

class GraphEnc(nn.Module):
    """Same shape as GINEncoder (2 conv layers, hidden 64, mean pool, linear to 128) with a different convolution."""
    def __init__(self, kind, hidden=64, emb=128):
        super().__init__(); mk = (lambda i, o: GCNConv(i, o, normalize=False, add_self_loops=False)) if kind == "gcn" else (lambda i, o: GraphConv(i, o, aggr="add"))
        self.convs = nn.ModuleList([mk(1, hidden), mk(hidden, hidden)]); self.relu = nn.ReLU(); self.proj = nn.Linear(hidden, emb)
    def forward(self, x, edge_index, edge_weight, batch=None):
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        for c in self.convs: h = self.relu(c(h, edge_index, edge_weight))
        return self.proj(global_mean_pool(h, batch))
class TemporalLSTM(nn.Module):
    def __init__(self, emb=128, hid=128):
        super().__init__(); self.lstm = nn.LSTM(emb, hid, 1, batch_first=True); self.readout = nn.Linear(hid, emb)
    def forward(self, z): h, _ = self.lstm(z); return self.readout(h)

base_cfg, p1_cfg, p2_cfg, p3_cfg = P.load_configs(); device = P.get_device(p1_cfg["device"])
pools = json.load(open(os.path.join(base_cfg["dataset"]["metadata_dir"], "phase1_subject_pools.json"))); seeds = pools["seed_rois"]; cd_ = p1_cfg["cache"]["dir"]
tc, vc, test_all = pools["train_control"], pools["val_control"], pools["test_all"]
tr_c, va_c = _preload_caches(tc, cd_), _preload_caches(vc, cd_); te_c = _preload_caches(test_all, cd_)
labels = {s: int(te_c[s]["label"]) for s in test_all}; test_ctl = [s for s in test_all if labels[s] == 0]
def build(variant):
    torch.manual_seed(42); np.random.seed(42)
    if variant == "GIN + GRU (proposed)": return None, None
    enc = {"GCN + GRU": GraphEnc("gcn"), "GraphConv + GRU": GraphEnc("graphconv"), "GIN + LSTM": GINEncoder(**p1_cfg["gin"])}[variant].to(device)
    tem = (TemporalLSTM() if "LSTM" in variant else TemporalGRU(embedding_dim=128, hidden_dim=128, num_layers=1)).to(device); return enc, tem
def train(enc, tem):
    opt = torch.optim.Adam(list(enc.parameters()) + list(tem.parameters()), lr=1e-3); bs, lam, epochs = 16, 0.5, 5
    ex = [(s, sd) for s in tc for sd in seeds]; vex = [(s, sd) for s in vc for sd in seeds]; rng = np.random.default_rng(42); best, best_state = 1e9, None; t0 = time.time()
    for ep in range(epochs):
        enc.train(); tem.train(); order = rng.permutation(len(ex)); sh = [ex[i] for i in order]
        for i in range(0, len(sh), bs):
            r = _run_batch(enc, tem, sh[i:i + bs], tr_c, device, lam)
            if r is None: continue
            opt.zero_grad(); r[0].backward(); opt.step()
        v = evaluate_split(enc, tem, vex, va_c, device); print(f"  epoch {ep+1} val_pred_loss={v:.4f} t={time.time()-t0:.0f}s", flush=True)
        if v < best: best = v; best_state = ({k: x.clone() for k, x in enc.state_dict().items()}, {k: x.clone() for k, x in tem.state_dict().items()})
    enc.load_state_dict(best_state[0]); tem.load_state_dict(best_state[1]); enc.eval(); tem.eval(); return best, time.time() - t0
def scores(enc, tem, cache):
    out = []
    with torch.no_grad():
        for sd in seeds:
            z = seed_embedding_sequence(enc, cache, int(sd), device)
            if z.shape[0] < 2: out.append(np.zeros(0)); continue
            out.append(transition_scores(z, tem(z.unsqueeze(0)).squeeze(0)).cpu().numpy())
    return out
def evaluate(enc, tem):
    vsc = [scores(enc, tem, va_c[s]) for s in vc]; tau_tr = float(np.percentile(np.concatenate([a for sc in vsc for a in sc]), 95)); vS = [max(a.max() for a in sc if len(a)) for sc in vsc]; tau_sub = float(np.percentile(vS, 95))
    S = np.array([max(a.max() for a in scores(enc, tem, te_c[s]) if len(a)) for s in test_all]); y = np.array([labels[s] for s in test_all]); pr = S > tau_sub
    res = {"asd_auroc": float(roc_auc_score(y, S)), "asd_accuracy": float((pr == y).mean()), "tau_transition": tau_tr, "tau_patient": tau_sub}
    for method in ("structure_aware", "random"):
        rng = np.random.default_rng(7); b, a = [], []
        for sid in test_ctl:
            c = te_c[sid]
            for sd in seeds:
                t = int(rng.integers(0, c["num_windows"] - 1)); r = P.generate_anomaly(enc, tem, c, int(sd), t, 0.9, device, method=method, rng_seed=int(rng.integers(1e9)))
                b.append(r["anomaly_score_before"]); a.append(r["anomaly_score_after"])
        b, a = np.array(b), np.array(a); yy = np.r_[np.zeros(len(b)), np.ones(len(a))]; ss = np.r_[b, a]; p = ss > tau_tr
        tp, fp = int((p & (yy == 1)).sum()), int((p & (yy == 0)).sum()); fn = int((~p & (yy == 1)).sum()); pre = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        res[method] = {"accuracy": float((p == yy).mean()), "precision": pre, "recall": rec, "f1": 2 * pre * rec / max(pre + rec, 1e-9), "auroc": float(roc_auc_score(yy, ss)), "mean_before": float(b.mean()), "mean_after": float(a.mean())}
    return res
variants = sys.argv[1:] or ["GIN + GRU (proposed)", "GCN + GRU", "GraphConv + GRU", "GIN + LSTM"]
outp = "paper/results/model_compare.json"; all_ = json.load(open(outp)) if os.path.exists(outp) else {}
for v in variants:
    print("==", v, flush=True); enc, tem = build(v)
    if enc is None:
        enc = GINEncoder(**p1_cfg["gin"]).to(device); tem = TemporalGRU(embedding_dim=128, hidden_dim=128, num_layers=1).to(device); ck = torch.load("outputs/experiments/phase1/checkpoints/best.pt", map_location=device)
        enc.load_state_dict(ck["gin"]); tem.load_state_dict(ck["gru"]); enc.eval(); tem.eval(); info = {"val_pred_loss": float(ck["val_pred_loss"]), "train_seconds": 299.7, "params": sum(p.numel() for p in list(enc.parameters()) + list(tem.parameters()))}
    else:
        best, secs = train(enc, tem); info = {"val_pred_loss": best, "train_seconds": secs, "params": sum(p.numel() for p in list(enc.parameters()) + list(tem.parameters()))}
    r = evaluate(enc, tem); r.update(info); all_[v] = r; json.dump(all_, open(outp, "w"), indent=1)
    print(v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if not isinstance(x, dict)}, "| guided acc/auroc", round(r["structure_aware"]["accuracy"], 3), round(r["structure_aware"]["auroc"], 3), "| random", round(r["random"]["accuracy"], 3), round(r["random"]["auroc"], 3), flush=True)
