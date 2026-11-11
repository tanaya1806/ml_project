"""Run the trained Phase 1 detector on ONE raw ROI time series (T x 360).

Same real modules as training: sliding-window FC -> top-20% signed graph -> stable 1-hop subgraphs of the 20 seed ROIs
-> GIN -> GRU -> A_t = ||z_{t+1} - z_hat_{t+1}||. Thresholds come from the committed Phase 1 metrics.json
(calibrated on validation controls only). This is a research detector with chance-level test AUROC; it is not a diagnostic.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs
from src.data.subgraph_cache import build_cache_from_arrays, select_seed_rois
from src.models.embedding_sequence import seed_embedding_sequence
from src.anomaly.scoring import transition_scores
from src.models.gin_encoder import GINEncoder
from src.models.gru_temporal import TemporalGRU
from src.utils.device import get_device

N_ROI, WINDOW, STRIDE, SPARSITY = 360, 30, 10, 0.20
MIN_T = WINDOW + STRIDE  # >=2 windows so at least one transition exists
TRAIN_T_RANGE = (78, 316)


_M = {}


def get_models():
    """Loads the committed Phase 1 checkpoint (outputs/experiments/phase1/checkpoints/best.pt) once."""
    if not _M:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/phase1.yaml")))
        device = get_device(cfg["device"])
        gin = GINEncoder(**cfg["gin"]).to(device)
        gru = TemporalGRU(embedding_dim=cfg["gin"]["embedding_dim"], hidden_dim=cfg["gru"]["hidden_dim"],
                          num_layers=cfg["gru"]["num_layers"]).to(device)
        ck = torch.load(os.path.join(ROOT, cfg["output_dir"], "checkpoints", "best.pt"), map_location=device)
        gin.load_state_dict(ck["gin"]); gru.load_state_dict(ck["gru"]); gin.eval(); gru.eval()
        _M.update(g=(gin, gru, ck, device, cfg))
    return _M["g"]


def load_thresholds():
    m = json.load(open(os.path.join(ROOT, "outputs/experiments/phase1/metrics.json")))
    return m["tau_transition"], m["tau_subject"]


def as_timeseries(arr: np.ndarray) -> np.ndarray:
    """Accept (T,360) or (360,T); reject anything else."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D array (time x 360 ROIs); got {a.ndim}-D.")
    if a.shape[1] != N_ROI and a.shape[0] == N_ROI:
        a = a.T
    if a.shape[1] != N_ROI:
        raise ValueError(f"Expected 360 ROI columns; got shape {a.shape}.")
    if not np.isfinite(a).all():
        raise ValueError("Time series contains NaN/inf values.")
    if a.shape[0] < MIN_T:
        raise ValueError(f"Need at least {MIN_T} time points (got {a.shape[0]}).")
    return a.astype(np.float32)


def seed_scores_from_cache(cache: dict):
    """-> (list of per-seed A_t arrays, seed ROI list)."""
    gin, gru, _, device, _ = get_models()
    seeds = [int(s) for s in cache["seed_rois"]]
    out = []
    with torch.no_grad():
        for s in seeds:
            z = seed_embedding_sequence(gin, cache, s, device)
            zh = gru(z.unsqueeze(0)).squeeze(0)
            out.append(transition_scores(z, zh).cpu().numpy())
    return out, seeds


def cache_from_timeseries(ts: np.ndarray, subject_id: str = "upload") -> dict:
    fcs = compute_sliding_window_fc(ts, window_size=WINDOW, stride=STRIDE, remove_diagonal=True)
    graphs = construct_temporal_graphs(fcs, sparsity_ratio=SPARSITY, keep_signed_weights=True)
    ei = np.stack([g.edge_index for g in graphs]); ew = np.stack([g.edge_weight for g in graphs])
    return build_cache_from_arrays(subject_id, "upload", -1, ei, ew, select_seed_rois(20, 42))


def features_from_scores(scores) -> np.ndarray:
    """40-D subject feature: per-seed max then per-seed mean of A_t."""
    return np.array([a.max() for a in scores] + [a.mean() for a in scores], dtype=np.float64)


def explain_top_seed(cache: dict, seed: int, scores: np.ndarray, k: int = 10, history: int = 3) -> dict:
    """Which connections of the top-scoring seed's subgraph drove its highest anomaly score.

    Takes the transition t* with the largest A_t, back-propagates A_{t*} to the edge weights of the newest window, and ranks every
    connection by |gradient x weight change|, where the change is measured against the mean of the previous `history` windows
    (an edge absent in a window counts as weight 0). This is a first-order attribution of the model's score, not a clinical finding.
    """
    gin, gru, _, device, _ = get_models()
    roi = [int(r) for r in cache[f"roi_indices_{seed}"]]
    ei_seq, ew_seq = cache[f"edge_index_{seed}"], cache[f"edge_weight_{seed}"]
    t_star = int(np.argmax(scores)); new = t_star + 1          # A[t] compares the observed window t+1 with the GRU prediction made at t
    from torch_geometric.data import Batch
    from src.models.embedding_sequence import _window_to_data
    batch = Batch.from_data_list([_window_to_data(len(roi), ei_seq[t], ew_seq[t]) for t in range(len(ei_seq))]).to(device)
    ew = batch.edge_weight.clone().requires_grad_(True)
    z = gin(batch.x, batch.edge_index, ew, batch.batch); zh = gru(z.unsqueeze(0)).squeeze(0)
    transition_scores(z, zh)[t_star].backward()
    grad = ew.grad.detach().cpu().numpy(); counts = [int(np.asarray(e).shape[1]) for e in ei_seq]
    off = int(sum(counts[:new])); g_new = grad[off:off + counts[new]]

    def weights(t):
        d = {}
        for (a, b), w in zip(np.asarray(ei_seq[t]).T, np.asarray(ew_seq[t])):
            d.setdefault((min(a, b), max(a, b)), []).append(float(w))
        return {key: float(np.mean(v)) for key, v in d.items()}
    cur = weights(new); prev = [weights(t) for t in range(max(0, new - history), new)]
    g_pair = {}
    for (a, b), g in zip(np.asarray(ei_seq[new]).T, g_new):
        g_pair[(min(a, b), max(a, b))] = g_pair.get((min(a, b), max(a, b)), 0.0) + float(g)
    edges = []
    for key, w in cur.items():
        before = float(np.mean([p.get(key, 0.0) for p in prev])) if prev else 0.0
        delta = w - before
        edges.append(dict(a=roi[key[0]], b=roi[key[1]], weight=w, before=before, delta=delta, influence=abs(g_pair.get(key, 0.0) * delta)))
    edges.sort(key=lambda e: -e["influence"]); top = edges[:k]; mx = max([e["influence"] for e in top] + [1e-12])
    for e in top: e["influence"] = e["influence"] / mx
    return dict(seed=int(seed), seed_roi=int(seed), transition=t_star, n_windows=int(len(ei_seq)), nodes=roi, top_edges=top,
                background_edges=[dict(a=roi[a], b=roi[b], weight=w) for (a, b), w in cur.items()][:400])


def analyze(arr: np.ndarray, subject_id: str = "upload") -> dict:
    ts = as_timeseries(arr)
    tau_tr, tau_sub = load_thresholds()
    cache = cache_from_timeseries(ts, subject_id)
    scores, seeds = seed_scores_from_cache(cache)
    seed_max = np.array([a.max() for a in scores])
    S = float(seed_max.max())
    top = int(seed_max.argmax())
    n_flag = int(sum((a > tau_tr).sum() for a in scores))
    return dict(
        n_timepoints=int(ts.shape[0]), n_windows=int(len(scores[0]) + 1), subject_score=S,
        tau_subject=tau_sub, tau_transition=tau_tr, flagged=bool(S > tau_sub),
        n_flagged_transitions=n_flag, n_transitions=int(sum(len(a) for a in scores)),
        seeds=seeds, seed_max=seed_max.tolist(), top_seed=seeds[top], top_seed_scores=scores[top].tolist(),
        features=features_from_scores(scores).tolist(),
        explain=explain_top_seed(cache, seeds[top], scores[top]),
        outside_training_length=bool(not (TRAIN_T_RANGE[0] <= ts.shape[0] <= TRAIN_T_RANGE[1])),
    )
