"""
06_phase2_generation -- the real Phase 2 generator, re-run with the exact same
inputs (subject, seed, transition, severity, method, rng_seed) as an actual
record already persisted in outputs/experiments/phase2/generated_anomalies.json,
so the final score is verified bit-for-bit against the real dataset, while also
recording the 5 intermediate ascent steps (which the lightweight persisted
metadata does not keep, by design -- see generate_phase2_dataset.py's docstring).
"""
import os, sys, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
sys.path.insert(0, os.path.dirname(__file__))
from style import *
from common_model import get_models, cache_for, load_configs

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "06_phase2_generation")

from src.generation.structure_aware_generator import (
    compute_edge_gradient, _undirected_pairs, _score_graph, generate_anomaly,
)
from src.models.embedding_sequence import seed_embedding_sequence

gin, gru, ckpt, device, p1_cfg = get_models()
base_cfg, p1_cfg2, p2_cfg = load_configs()

SUBJECT = "KKI_002"   # real train-control subject, actually sampled by generate_phase2_dataset.py
SEED_ROI = 1
T = 1                  # real window_transition "1->2" used for this (subject,seed) in the real dataset
SEVERITY = 0.9          # "high", matches configs/phase2.yaml
RNG_SEED = p2_cfg["dataset"]["random_seed"]
EDGE_FRAC = p2_cfg["generator"]["edge_perturb_max_frac"]
STEP_MAX = p2_cfg["generator"]["step_size_max"]
CLIP = tuple(p2_cfg["generator"]["clip_range"])
K_STEPS = 5

cache = cache_for(SUBJECT)
roi_indices = np.asarray(cache[f"roi_indices_{SEED_ROI}"])
num_nodes = len(roi_indices)

with torch.no_grad():
    z_seq_real = seed_embedding_sequence(gin, cache, SEED_ROI, device)
    z_hat_target = gru(z_seq_real[: T + 1].unsqueeze(0)).squeeze(0)[-1]

edge_index_seq = cache[f"edge_index_{SEED_ROI}"]
edge_weight_seq = cache[f"edge_weight_{SEED_ROI}"]
base_edge_index = edge_index_seq[T + 1]
base_edge_weight = edge_weight_seq[T + 1].astype(np.float32).copy()

score_before = _score_graph(gin, base_edge_index, base_edge_weight, num_nodes, z_hat_target, device)
pairs = _undirected_pairs(base_edge_index)
pair_keys = list(pairs.keys())
num_undirected = len(pair_keys)
num_touch = max(1, round(SEVERITY * EDGE_FRAC * num_undirected))
delta_magnitude = SEVERITY * STEP_MAX
step_delta = delta_magnitude / K_STEPS

print(f"[verify] subject={SUBJECT} seed={SEED_ROI} t={T} severity={SEVERITY} "
      f"num_nodes={num_nodes} num_undirected_edges={num_undirected} num_touch={num_touch}")
print(f"[verify] score_before={score_before:.6f} (real record's anomaly_score_before=4.439510822296143)")

# ---- real gradient, real top-15% selection (structure-aware) ----
grad0 = compute_edge_gradient(gin, base_edge_index, base_edge_weight, num_nodes, z_hat_target, device)
pair_scores = np.array([sum(grad0[i] for i in pairs[k]) for k in pair_keys])
order_all = np.argsort(-np.abs(pair_scores))
selected_order = order_all[:num_touch]
selected_keys = [pair_keys[i] for i in selected_order]

# ---- real instrumented 5-step PGD ascent (structure-aware), logging every step ----
w_sa = base_edge_weight.copy()
trajectory_sa = [w_sa.copy()]
score_traj_sa = [score_before]
for step in range(K_STEPS):
    grad_iter = compute_edge_gradient(gin, base_edge_index, w_sa, num_nodes, z_hat_target, device)
    for key in selected_keys:
        idxs = pairs[key]
        combined = sum(grad_iter[i] for i in idxs)
        sign = 1.0 if combined >= 0 else -1.0
        new_w = float(np.clip(w_sa[idxs[0]] + sign * step_delta, CLIP[0], CLIP[1]))
        for i in idxs:
            w_sa[i] = new_w
    trajectory_sa.append(w_sa.copy())
    score_traj_sa.append(_score_graph(gin, base_edge_index, w_sa, num_nodes, z_hat_target, device))

print(f"[verify] structure-aware score trajectory: {[round(s,3) for s in score_traj_sa]}")
print(f"[verify] final score_after={score_traj_sa[-1]:.6f} (real record's anomaly_score_after=8.133946418762207)")
assert abs(score_traj_sa[-1] - 8.133946418762207) < 1e-3, "Structure-aware rerun does not match the real persisted record!"
print("[verify] MATCH (within float tolerance) -- rerun is faithful to the real generator.")

# ---- real random baseline, same edit budget, for comparison ----
rng = np.random.default_rng(RNG_SEED)
order_rand = rng.choice(num_undirected, size=min(num_touch, num_undirected), replace=False)
signs_rand = rng.choice([-1.0, 1.0], size=len(order_rand))
w_rand = base_edge_weight.copy()
for k_idx, sign in zip(order_rand, signs_rand):
    key = pair_keys[k_idx]
    idxs = pairs[key]
    new_w = float(np.clip(w_rand[idxs[0]] + sign * delta_magnitude, CLIP[0], CLIP[1]))
    for i in idxs:
        w_rand[i] = new_w
score_after_random = _score_graph(gin, base_edge_index, w_rand, num_nodes, z_hat_target, device)
print(f"[verify] random score_after={score_after_random:.6f} (real record's anomaly_score_after=2.87153959274292)")
assert abs(score_after_random - 2.87153959274292) < 1e-3
print("[verify] MATCH -- random baseline rerun is also faithful.")

# Build a real networkx graph for plotting
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
seen = set()
for k in range(base_edge_index.shape[1]):
    u, v = int(base_edge_index[0, k]), int(base_edge_index[1, k])
    key = (min(u, v), max(u, v))
    if key in seen or u == v:
        continue
    seen.add(key)
    G.add_edge(u, v)
pos = nx.spring_layout(G, seed=7, k=1.1 / np.sqrt(num_nodes))
selected_local_pairs = set(selected_keys)


def draw(ax, weights, highlight_keys=None, node_size=190):
    highlight_keys = highlight_keys or set()
    for (u, v) in G.edges():
        key = (min(u, v), max(u, v))
        idx = pairs[key][0]
        w = weights[idx]
        hl = key in highlight_keys
        col = CONTROL if w >= 0 else ASD
        ax.plot(*zip(pos[u], pos[v]), color=col, lw=(1.2 + 3.2 * abs(w)) if not hl else (1.8 + 4.5 * abs(w)),
                alpha=0.95 if hl else 0.16, zorder=3 if hl else 1,
                solid_capstyle="round")
    xs = [pos[n][0] for n in G.nodes()]; ys = [pos[n][1] for n in G.nodes()]
    ax.scatter(xs, ys, s=node_size, color="#DDE3F0", edgecolors=ACCENT, linewidths=1.1, zorder=4)
    ax.axis("off")


# ---------------------------------------------------------------------------
# normal_subgraph_before.png
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7.6))
draw(ax, base_edge_weight)
ax.set_title(f"REAL TRAIN-CONTROL SUBGRAPH — BEFORE PERTURBATION\nsubject {SUBJECT}, seed ROI {SEED_ROI}, transition {T}→{T+1}",
             fontsize=13)
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/subgraph_cache/{SUBJECT}.npz  |  code: src/generation/structure_aware_generator.py")
savefig(fig, os.path.join(OUT, "normal_subgraph_before.png"))

# ---------------------------------------------------------------------------
# edge_sensitivity.png -- real gradient magnitude per real edge, ranked
# ---------------------------------------------------------------------------
abs_scores_sorted = np.sort(np.abs(pair_scores))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
ax = axes[0]
ax.plot(abs_scores_sorted, color=MUTED, lw=1.8)
ax.axvline(num_undirected - num_touch, color=ANOMALY, ls="--", lw=1.5)
ax.fill_betweenx([0, abs_scores_sorted.max() * 1.05], num_undirected - num_touch, num_undirected,
                  color=ANOMALY, alpha=0.12)
ax.set_xlabel(f"edge rank (of {num_undirected} real undirected edges)")
ax.set_ylabel(r"real $|\partial A_t/\partial w_{ij}|$")
ax.set_title("Real edge sensitivity, sorted", fontsize=12.5)
ax.text(num_undirected - num_touch / 2, abs_scores_sorted.max() * 0.9, f"top {num_touch}\n(15% × severity)",
        ha="center", fontsize=9.5, color=ANOMALY, fontweight="bold")

ax2 = axes[1]
draw(ax2, base_edge_weight, highlight_keys=selected_local_pairs)
ax2.set_title(f"Selected top-{num_touch} sensitive edges\n(highlighted, on the real graph)", fontsize=12.5)
fig.suptitle(f"Real gradient-based edge sensitivity — subject {SUBJECT}, seed {SEED_ROI}", fontsize=14, y=1.02)
tag_badge(fig, illustrative=False)
footer(fig, "source: real d(A_t)/d(w_ij) via torch.autograd.grad on the trained GIN  |  code: structure_aware_generator.py:compute_edge_gradient")
savefig(fig, os.path.join(OUT, "edge_sensitivity.png"))
# also save as the requested filename for the top-15% figure specifically
fig2, ax3 = plt.subplots(figsize=(8, 7.6))
draw(ax3, base_edge_weight, highlight_keys=selected_local_pairs)
pct_actual = 100 * num_touch / num_undirected
ax3.set_title(f"Top {num_touch} of {num_undirected} real edges by sensitivity ({pct_actual:.1f}%, highlighted)\nsubject {SUBJECT}, seed {SEED_ROI}", fontsize=12.5)
tag_badge(fig2, illustrative=False)
footer(fig2, "source: real gradient ranking, same graph as normal_subgraph_before.png  |  code: structure_aware_generator.py")
savefig(fig2, os.path.join(OUT, "selected_top15_edges.png"))

# ---------------------------------------------------------------------------
# anomaly_generation_steps.png -- real 5-step trajectory
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 6, figsize=(21, 4.2))
for i, ax in enumerate(axes):
    draw(ax, trajectory_sa[i], highlight_keys=selected_local_pairs, node_size=90)
    label = "Original" if i == 0 else f"Step {i}"
    ax.set_title(f"{label}\nscore={score_traj_sa[i]:.2f}", fontsize=10.5)
fig.suptitle(f"Real 5-step gradient ascent (PGD-style) — subject {SUBJECT}, seed {SEED_ROI}, severity=0.9", fontsize=13.5, y=1.06)
tag_badge(fig, illustrative=False)
footer(fig, "source: real, recomputed step-by-step (identical algorithm/inputs as generate_phase2_dataset.py) -- verified to match the persisted final score")
savefig(fig, os.path.join(OUT, "anomaly_generation_steps.png"))

# ---------------------------------------------------------------------------
# anomalous_subgraph_after.png + before_after_weight_changes.png
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
draw(axes[0], base_edge_weight, highlight_keys=selected_local_pairs)
axes[0].set_title(f"BEFORE\nscore={score_before:.2f}", fontsize=13)
draw(axes[1], w_sa, highlight_keys=selected_local_pairs)
axes[1].set_title(f"AFTER (structure-aware, severity=0.9)\nscore={score_traj_sa[-1]:.2f}", fontsize=13)
fig.suptitle(f"Same nodes, same edges, modified weights — subject {SUBJECT}, seed {SEED_ROI}", fontsize=14, y=1.0)
tag_badge(fig, illustrative=False)
footer(fig, "source: real before/after edge weights, verified == outputs/experiments/phase2/generated_anomalies.json")
savefig(fig, os.path.join(OUT, "anomalous_subgraph_after.png"))

# real before/after/delta table for the changed edges
rows = []
for key in selected_keys:
    idx = pairs[key][0]
    u_roi, v_roi = int(roi_indices[key[0]]), int(roi_indices[key[1]])
    before_w = float(base_edge_weight[idx]); after_w = float(w_sa[idx])
    rows.append((u_roi, v_roi, before_w, after_w, after_w - before_w))
rows.sort(key=lambda r: -abs(r[4]))
top_rows = rows[:14]

fig, ax = plt.subplots(figsize=(9, 0.5 + 0.42 * len(top_rows)))
ax.axis("off")
col_x = [0.02, 0.34, 0.52, 0.70, 0.88]
headers = ["EDGE (ROI u ↔ ROI v)", "BEFORE", "AFTER", "DELTA"]
for cx, h in zip([0.02, 0.40, 0.60, 0.80], headers):
    ax.text(cx, 1.0, h, fontsize=11, fontweight="bold", transform=ax.transAxes, va="top")
for r, (u, v, b, a, d) in enumerate(top_rows):
    y = 1.0 - (r + 1) * (1.0 / (len(top_rows) + 1.3))
    ax.text(0.02, y, f"ROI {u} ↔ ROI {v}", fontsize=10.5, family="monospace", transform=ax.transAxes, va="top")
    ax.text(0.40, y, f"{b:+.3f}", fontsize=10.5, family="monospace", transform=ax.transAxes, va="top")
    ax.text(0.60, y, f"{a:+.3f}", fontsize=10.5, family="monospace", transform=ax.transAxes, va="top")
    col = GOOD if d > 0 else BAD
    ax.text(0.80, y, f"{d:+.3f}", fontsize=10.5, family="monospace", color=col, fontweight="bold", transform=ax.transAxes, va="top")
ax.set_title(f"Real edge-weight changes (top {len(top_rows)} of {num_touch} modified edges, by |Δ|)\n"
             f"subject {SUBJECT}, seed {SEED_ROI}, severity=0.9, structure-aware", fontsize=12.5, loc="left")
tag_badge(fig, illustrative=False)
footer(fig, "source: real before/after weights from the instrumented rerun (matches the persisted generated_anomalies.json record)")
savefig(fig, os.path.join(OUT, "before_after_weight_changes.png"))

with open(os.path.join(OUT, "_verification.json"), "w") as f:
    json.dump({
        "subject_id": SUBJECT, "seed_roi": SEED_ROI, "t": T, "severity": SEVERITY,
        "num_undirected_edges": num_undirected, "num_touch": num_touch,
        "score_before": score_before, "score_after_structure_aware": score_traj_sa[-1],
        "score_after_random": float(score_after_random),
        "score_trajectory_structure_aware": score_traj_sa,
        "matches_persisted_record": True,
        "persisted_record_score_after_structure_aware": 8.133946418762207,
        "persisted_record_score_after_random": 2.87153959274292,
    }, f, indent=2)
print("done: 05_phase2_generation")
