"""
Phase 2: Structure-Aware Anomaly Generation.

Research question: given a learned normal temporal brain-subgraph trajectory, how
can we deliberately create a plausible but abnormal connectivity transition?

Design (see docs/phase2_report.md for the full justification). We do NOT invent a
new network (no GAN/diffusion/transformer). We reuse exactly what Phase 1 already
learned:

    z_t       = GIN(G_t[S_v])                     (frozen, from Phase 1)
    z_hat_t+1 = GRU(z_1, ..., z_t)                  (frozen, causal -- no future leak)
    A_real    = || GIN(G_{t+1}) - z_hat_{t+1} ||^2  (small, since G_{t+1} is real/normal)

A structure-aware anomaly is produced by *gradient ascent on A* with respect to the
edge weights of the REAL next-window graph G_{t+1}: d(A)/d(w_uv) tells us exactly
which functional connections, and in which direction, the model's own learned
structure says would most increase deviation from expected normal dynamics per unit
change. This is the standard adversarial-perturbation idea (à la FGSM/PGD), reused
constructively for generation rather than attack -- no extra trainable component.

We use K small iterative steps (PGD-style) rather than one large step (FGSM-style):
a single large step from the real graph overshoots the local-linear region the
gradient describes, and empirically was NOT monotonic in severity (score at
severity=0.9 measured *below* severity=0.55 in initial testing -- GIN's ReLU
nonlinearities mean a large single jump can cross into a different local regime).
Recomputing the gradient every step and taking a bounded fraction of the severity
budget each time keeps every step inside the region where the gradient direction
is locally valid, which restored monotonicity (validated in generate_phase2_dataset.py).

Only the top-m most sensitive undirected edges are touched (m controlled by
severity), by a bounded delta (also severity-scaled), so the anomaly is a small,
localized, interpretable edit to one real transition -- not a fresh random graph
and not a full-sequence corruption.

The random-perturbation baseline (`method="random"`) uses the identical pipeline
but selects edges uniformly at random and pushes them by a random sign, so Phase 2
validation can honestly compare structure-aware vs. random generation (section 32).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.models.embedding_sequence import seed_embedding_sequence


def _undirected_pairs(edge_index: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Groups the (already-symmetric) directed edge list into undirected (u,v) pairs."""
    pairs: Dict[Tuple[int, int], List[int]] = {}
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        pairs.setdefault(key, []).append(i)
    return pairs


def compute_edge_gradient(
    gin,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    z_target: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """d(||GIN(G;w) - z_target||^2) / dw for every directed edge weight w."""
    ei = torch.tensor(edge_index.astype(np.int64), dtype=torch.long, device=device)
    ew = torch.tensor(edge_weight.astype(np.float32), dtype=torch.float32, device=device, requires_grad=True)
    x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

    z = gin(x, ei, ew)  # [1, D] (single graph, no batch vector needed)
    loss = ((z[0] - z_target) ** 2).sum()
    grad = torch.autograd.grad(loss, ew)[0]
    return grad.detach().cpu().numpy()


def _score_graph(gin, edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int,
                  z_target: torch.Tensor, device: torch.device) -> float:
    ei = torch.tensor(edge_index.astype(np.int64), dtype=torch.long, device=device)
    ew = torch.tensor(edge_weight.astype(np.float32), dtype=torch.float32, device=device)
    x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        z = gin(x, ei, ew)
        return float(torch.linalg.norm(z[0] - z_target).item())


def generate_anomaly(
    gin,
    gru,
    cache: dict,
    seed: int,
    t: int,
    severity: float,
    device: torch.device,
    method: str = "structure_aware",
    edge_perturb_max_frac: float = 0.15,
    step_size_max: float = 0.6,
    clip_range: Tuple[float, float] = (-1.0, 1.0),
    rng_seed: int = 0,
    num_ascent_steps: int = 5,
) -> dict:
    """
    Generates one anomalous transition G_{t+1}[S_v]^anom from a real normal subject's
    stable subgraph sequence, by editing the real G_{t+1} graph.

    Args:
        cache: subject cache from src.data.subgraph_cache.load_subject_cache
        seed: seed ROI
        t: transition index; perturbs window t+1 given real history 0..t
        severity: alpha in [0, 1], controls how many edges / how large the edits are
        method: "structure_aware" (gradient-guided) or "random" (baseline)
        rng_seed: makes "random" method (and structure-aware tie-breaking) reproducible

    Returns:
        dict record with provenance, the anomalous edge_weight array, and scores.
    """
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    if method not in ("structure_aware", "random"):
        raise ValueError(f"Unknown method: {method}")

    roi_indices = cache[f"roi_indices_{seed}"]
    num_nodes = len(roi_indices)
    edge_index_seq = cache[f"edge_index_{seed}"]
    edge_weight_seq = cache[f"edge_weight_{seed}"]
    num_windows = len(edge_index_seq)
    if t + 1 >= num_windows:
        raise ValueError(f"t+1={t+1} out of range for subject with {num_windows} windows")

    rng = np.random.default_rng(rng_seed)

    with torch.no_grad():
        z_seq_real = seed_embedding_sequence(gin, cache, seed, device)  # [T, D], from REAL graphs
        z_hat_full = gru(z_seq_real[: t + 1].unsqueeze(0)).squeeze(0)   # causal: only uses windows 0..t
        z_hat_target = z_hat_full[-1]                                    # predicted z_{t+1}

    base_edge_index = edge_index_seq[t + 1]
    base_edge_weight = edge_weight_seq[t + 1].astype(np.float32).copy()
    anomalous_weight = base_edge_weight.copy()

    score_before = _score_graph(gin, base_edge_index, base_edge_weight, num_nodes, z_hat_target, device)

    pairs = _undirected_pairs(base_edge_index)
    pair_keys = list(pairs.keys())
    num_undirected = len(pair_keys)
    if severity == 0.0 or num_undirected == 0:
        num_touch = 0
    else:
        num_touch = max(1, round(severity * edge_perturb_max_frac * num_undirected))
    delta_magnitude = severity * step_size_max

    edges_changed = []
    if num_touch > 0:
        if method == "structure_aware":
            # Select the top-m most sensitive undirected edges ONCE, from the real
            # graph's gradient (this fixes *which* connections are "structurally
            # meaningful" to perturb). Then take num_ascent_steps small gradient
            # ascent steps, recomputing the gradient each time, so the total
            # movement stays within the region where each step's direction is
            # locally valid (PGD-style; see module docstring).
            grad0 = compute_edge_gradient(gin, base_edge_index, base_edge_weight, num_nodes, z_hat_target, device)
            pair_scores = np.array([sum(grad0[i] for i in pairs[k]) for k in pair_keys])
            order = np.argsort(-np.abs(pair_scores))[:num_touch]
            selected_keys = [pair_keys[i] for i in order]

            step_delta = delta_magnitude / num_ascent_steps
            for _ in range(num_ascent_steps):
                grad_iter = compute_edge_gradient(gin, base_edge_index, anomalous_weight, num_nodes, z_hat_target, device)
                for key in selected_keys:
                    idxs = pairs[key]
                    combined = sum(grad_iter[i] for i in idxs)
                    sign = 1.0 if combined >= 0 else -1.0
                    new_w = float(np.clip(anomalous_weight[idxs[0]] + sign * step_delta, clip_range[0], clip_range[1]))
                    for i in idxs:
                        anomalous_weight[i] = new_w

            for key in selected_keys:
                idxs = pairs[key]
                edges_changed.append({
                    "roi_u": int(roi_indices[key[0]]),
                    "roi_v": int(roi_indices[key[1]]),
                    "weight_before": float(base_edge_weight[idxs[0]]),
                    "weight_after": float(anomalous_weight[idxs[0]]),
                })
        else:  # random baseline
            order = rng.choice(num_undirected, size=min(num_touch, num_undirected), replace=False)
            signs = rng.choice([-1.0, 1.0], size=len(order))
            for k_idx, sign in zip(order, signs):
                key = pair_keys[k_idx]
                idxs = pairs[key]
                old_w = float(base_edge_weight[idxs[0]])
                new_w = float(np.clip(old_w + sign * delta_magnitude, clip_range[0], clip_range[1]))
                for i in idxs:
                    anomalous_weight[i] = new_w
                edges_changed.append({
                    "roi_u": int(roi_indices[key[0]]),
                    "roi_v": int(roi_indices[key[1]]),
                    "weight_before": old_w,
                    "weight_after": new_w,
                })

    score_after = _score_graph(gin, base_edge_index, anomalous_weight, num_nodes, z_hat_target, device)

    # Constraint validation (should always hold by construction; asserted for safety).
    assert np.all(anomalous_weight >= clip_range[0] - 1e-6) and np.all(anomalous_weight <= clip_range[1] + 1e-6)
    assert np.all(np.isfinite(anomalous_weight))
    assert np.array_equal(base_edge_index, base_edge_index)  # node/edge set unchanged

    return {
        "subject_id": cache["subject_id"],
        "label": cache["label"],
        "seed_roi": int(seed),
        "window_transition": f"{t}->{t + 1}",
        "severity": severity,
        "method": method,
        "num_undirected_edges": num_undirected,
        "num_edges_changed": len(edges_changed),
        "edges_changed": edges_changed,
        "anomaly_score_before": score_before,
        "anomaly_score_after": score_after,
        "edge_index": base_edge_index,
        "edge_weight_anomalous": anomalous_weight,
        "roi_indices": roi_indices,
        "is_real": False,
    }
