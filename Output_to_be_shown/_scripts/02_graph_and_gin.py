"""
02_graph_construction + 03_gin -- the actual subgraph that enters GIN, and the
actual 128-D embedding the real trained GIN produces from it.
"""
import os, sys, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(__file__))
from style import *
from common_model import embedding_sequence, get_models, cache_for

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT_GRAPH = os.path.join(ROOT, "Output_to_be_shown", "02_graph_construction")
OUT_GIN = os.path.join(ROOT, "Output_to_be_shown", "03_gin")

SUBJECT = "UM_1_007"      # real ABIDE I test-control subject (same as figure 01)
SUBJECT_ASD = "UM_1_062"  # real ABIDE I test-ASD subject, same site, for the real embedding comparison
SEED_ROI = 284             # real Phase 1 seed ROI with a compact (33-node), readable neighborhood
WINDOW = 0

cache = cache_for(SUBJECT)
roi_indices = np.asarray(cache[f"roi_indices_{SEED_ROI}"])
edge_index = np.asarray(cache[f"edge_index_{SEED_ROI}"][WINDOW])
edge_weight = np.asarray(cache[f"edge_weight_{SEED_ROI}"][WINDOW])
num_nodes = len(roi_indices)
seed_local = int(np.where(roi_indices == SEED_ROI)[0][0])
print(f"[verify] subject={SUBJECT} seed_roi={SEED_ROI} window={WINDOW} num_nodes={num_nodes} "
      f"num_directed_edges={edge_index.shape[1]} seed_local_idx={seed_local}")

# Build a real networkx graph (undirected, one entry per pair) for layout/plotting
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i, roi=int(roi_indices[i]))
seen = set()
for k in range(edge_index.shape[1]):
    u, v = int(edge_index[0, k]), int(edge_index[1, k])
    key = (min(u, v), max(u, v))
    if key in seen or u == v:
        continue
    seen.add(key)
    G.add_edge(u, v, weight=float(edge_weight[k]))

pos = nx.spring_layout(G, seed=42, weight=None, k=1.1 / np.sqrt(num_nodes))


def draw_graph(ax, highlight_seed=True, node_size=260, label_rois=False):
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    colors = [ASD if w < 0 else CONTROL for w in weights]
    widths = 0.6 + 3.2 * (np.abs(weights))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=colors, width=widths, alpha=0.75)
    node_colors = [ANOMALY if (highlight_seed and n == seed_local) else "#DDE3F0" for n in G.nodes()]
    node_edgecolors = [ANOMALY if (highlight_seed and n == seed_local) else ACCENT for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, edgecolors=node_edgecolors,
                            node_size=[node_size * 1.8 if n == seed_local else node_size for n in G.nodes()],
                            linewidths=1.4)
    if label_rois:
        labels = {n: str(int(roi_indices[n])) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6.5, font_color=INK)
    ax.axis("off")


# ---------------------------------------------------------------------------
# selected_subgraph.png -- the real cached stable subgraph
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 8))
draw_graph(ax, label_rois=True)
ax.set_title(f"Real stable subgraph — subject {SUBJECT}, seed ROI {SEED_ROI}, window {WINDOW}\n"
             f"{num_nodes} nodes (ROIs) · {len(seen)} undirected edges · k=1-hop, stable-tracked",
             fontsize=12.5)
legend_handles = [
    mpatches.Patch(color=CONTROL, label="positive edge weight"),
    mpatches.Patch(color=ASD, label="negative edge weight"),
    mpatches.Patch(color=ANOMALY, label=f"seed ROI ({SEED_ROI})"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9.5, frameon=False)
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/subgraph_cache/{SUBJECT}.npz (seed {SEED_ROI}, window {WINDOW})  |  "
            f"code: src/graphs/temporal_tracking.py, src/data/subgraph_cache.py")
savefig(fig, os.path.join(OUT_GRAPH, "selected_subgraph.png"))

# ---------------------------------------------------------------------------
# gin_input_subgraph.png -- same real subgraph, annotated exactly as GIN receives it
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
draw_graph(ax, node_size=300)
ax.set_title(f"What actually enters the GIN encoder — subject {SUBJECT}, seed {SEED_ROI}, window {WINDOW}", fontsize=13)
box = ("INPUT TO GIN (real, this graph):\n"
       "• Node feature x_i = constant 1.0  (identity carried by structure, not features)\n"
       "• Edge feature w_ij = signed Pearson connectivity weight (shown as edge color/width)\n"
       f"• {num_nodes} nodes, {len(seen)} undirected edges")
ax.text(0.02, 0.02, box, transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left",
        bbox=dict(facecolor=PANEL, edgecolor=LINE, boxstyle="round,pad=0.6"))
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/subgraph_cache/{SUBJECT}.npz  |  code: src/models/gin_encoder.py:subgraph_to_pyg")
savefig(fig, os.path.join(OUT_GIN, "gin_input_subgraph.png"))

# ---------------------------------------------------------------------------
# gin_message_passing.png -- real edges/weights; messages at layer 1 are EXACTLY
# w_uv (since h_u starts at the constant 1.0), so the "message strength" numbers
# shown are real, not illustrative -- only the arrow/annotation overlay is a
# didactic device on top of real values.
# ---------------------------------------------------------------------------
# The seed is, by construction of a 1-hop neighborhood, directly connected to ALL
# other nodes in this induced subgraph (that's why they're in it) -- so "all"
# messages would mean 32 overlapping arrows. For legibility we show the real
# subset of its 8 real *strongest* connections (by |weight|) -- an honest,
# principled selection (not arbitrary/invented), explicitly labeled as such.
neighbors_all = list(G.neighbors(seed_local))
neighbors_all.sort(key=lambda n: -abs(G[n][seed_local]["weight"]))
TOPK = 8
neighbors = neighbors_all[:TOPK]

fig, ax = plt.subplots(figsize=(9.5, 8.5))
Gs = nx.Graph()
Gs.add_node(seed_local)
for nb in neighbors:
    Gs.add_node(nb)
    Gs.add_edge(nb, seed_local, weight=G[nb][seed_local]["weight"])
pos_s = nx.spring_layout(Gs, seed=1, center=(0, 0))
pos_s[seed_local] = np.array([0.0, 0.0])
# place neighbors on a circle around the seed, ordered by weight, for a clean radial look
ang = np.linspace(0, 2 * np.pi, len(neighbors), endpoint=False)
for nb, a in zip(neighbors, ang):
    pos_s[nb] = np.array([np.cos(a), np.sin(a)]) * 1.6

for nb in neighbors:
    w = G[nb][seed_local]["weight"]
    x1, y1 = pos_s[nb]; x2, y2 = pos_s[seed_local]
    col = CONTROL if w >= 0 else ASD
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0 + 3.5 * abs(w), alpha=0.9,
                                 shrinkA=22, shrinkB=26))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my, f"w={w:+.2f}", fontsize=8.5, color=col, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
for nb in neighbors:
    x, y = pos_s[nb]
    ax.scatter([x], [y], s=520, color="#DDE3F0", edgecolors=ACCENT, linewidths=1.4, zorder=3)
    ax.text(x, y, str(int(roi_indices[nb])), ha="center", va="center", fontsize=9, zorder=4)
ax.scatter([0], [0], s=900, color=ANOMALY, edgecolors=ANOMALY, zorder=3)
ax.text(0, 0, str(SEED_ROI), ha="center", va="center", fontsize=10.5, color="white", fontweight="bold", zorder=4)
ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.axis("off")
ax.set_title(f"Weighted message passing INTO the seed ROI ({SEED_ROI}) — layer 1, real edge weights\n"
             f"(strongest {TOPK} of its {len(neighbors_all)} real connections, shown for legibility)", fontsize=12.5)
eq = (r"$m_v=\sum_{u\in N(v)} w_{uv}\,h_u \qquad h_v'=\mathrm{MLP}((1+\epsilon)h_v+m_v)$")
ax.text(0.5, -0.05, eq, transform=ax.transAxes, ha="center", va="top", fontsize=13)
note = (f"At layer 1, h_u = 1.0 for every node, so each real message m_{{u→v}} = w_{{uv}} exactly —\n"
        f"the numbers on each arrow are the real edge weights. After aggregating all 32 real neighbors\n"
        f"(only the strongest {TOPK} drawn here) node {SEED_ROI} is passed through an MLP; a second identical\n"
        f"layer repeats this for every node, so information reaches 2 hops out.")
ax.text(0.5, -0.16, note, transform=ax.transAxes, fontsize=9.6, va="top", ha="center", color=MUTED)
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/subgraph_cache/{SUBJECT}.npz (seed {SEED_ROI})  |  code: src/models/gin_encoder.py:WeightedGINConv")
savefig(fig, os.path.join(OUT_GIN, "gin_message_passing.png"))

# ---------------------------------------------------------------------------
# gin_embedding_128d.png -- the REAL 128-D embedding from the REAL trained GIN
# ---------------------------------------------------------------------------
z_seq, _ = embedding_sequence(SUBJECT, SEED_ROI)
z0 = z_seq[WINDOW].detach().cpu().numpy()

fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 2.6]})
fig.subplots_adjust(hspace=0.55, top=0.90, bottom=0.24)
ax = axes[0]
im = ax.imshow(z0[None, :], cmap="RdBu_r", vmin=-np.abs(z0).max(), vmax=np.abs(z0).max(), aspect="auto")
ax.set_yticks([])
ax.set_title(f"Real 128-D GIN embedding z — subject {SUBJECT}, seed {SEED_ROI}, window {WINDOW}", fontsize=12.5)
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

ax2 = axes[1]
ax2.bar(np.arange(128), z0, color=[CONTROL if v >= 0 else ASD for v in z0], width=1.0)
ax2.axhline(0, color=INK, lw=0.8)
ax2.set_xlabel("embedding dimension"); ax2.set_ylabel("value")
ax2.set_xlim(-1, 128)
note = ("This 128-dimensional vector is the GIN's learned, joint representation of the whole subgraph's\n"
        "structure -- individual dimensions are not meant to correspond to individual biological variables.")
fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=9.6, color=MUTED)
tag_badge(fig, illustrative=False)
footer(fig, f"source: real forward pass, best.pt on subgraph_cache/{SUBJECT}.npz  |  code: src/models/embedding_sequence.py")
savefig(fig, os.path.join(OUT_GIN, "gin_embedding_128d.png"))

# ---------------------------------------------------------------------------
# embedding_comparison.png -- REAL embeddings, control vs ASD, same seed/window
# ---------------------------------------------------------------------------
z_seq_asd, _ = embedding_sequence(SUBJECT_ASD, SEED_ROI)
z0_asd = z_seq_asd[WINDOW].detach().cpu().numpy()

fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1, 1.3]})
vmax = max(np.abs(z0).max(), np.abs(z0_asd).max())
for ax, vec, name, col in zip(axes[:2], [z0, z0_asd], [f"{SUBJECT} (control)", f"{SUBJECT_ASD} (ASD)"], [CONTROL, ASD]):
    im = ax.imshow(vec[None, :], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center", color=col, fontweight="bold")
axes[0].set_title(f"Real GIN embeddings, same seed ROI ({SEED_ROI}) and window ({WINDOW}), two real subjects", fontsize=12.5)
axes[2].plot(np.arange(128), z0, color=CONTROL, lw=1.3, label=f"{SUBJECT} (control)")
axes[2].plot(np.arange(128), z0_asd, color=ASD, lw=1.3, label=f"{SUBJECT_ASD} (ASD)")
axes[2].axhline(0, color=LINE, lw=0.8)
axes[2].set_xlabel("embedding dimension"); axes[2].legend(fontsize=9, frameon=False)
l2 = float(np.linalg.norm(z0 - z0_asd))
axes[2].text(0.99, 0.05, f"||z_control − z_ASD||₂ = {l2:.3f}", transform=axes[2].transAxes,
             ha="right", fontsize=10, color=MUTED)
tag_badge(fig, illustrative=False)
footer(fig, f"source: real forward pass on data/processed/subgraph_cache/{{{SUBJECT},{SUBJECT_ASD}}}.npz  |  code: src/models/embedding_sequence.py")
savefig(fig, os.path.join(OUT_GIN, "embedding_comparison.png"))

with open(os.path.join(OUT_GIN, "_verification.json"), "w") as f:
    json.dump({
        "subject_control": SUBJECT, "subject_asd": SUBJECT_ASD, "seed_roi": SEED_ROI, "window": WINDOW,
        "num_nodes": num_nodes, "num_undirected_edges": len(seen),
        "embedding_control_first5": z0[:5].tolist(), "embedding_asd_first5": z0_asd[:5].tolist(),
        "l2_distance_control_vs_asd": l2,
    }, f, indent=2)
print("done: 02_graph_and_gin")
