"""
01_original_data -- the actual functional-connectivity matrix this project computes.

IMPORTANT, VERIFIED DATA-AVAILABILITY NOTE (see Output_to_be_shown/README.md):
Raw ABIDE ROI time series (data/raw/Time_serie_per_site/site_*_time_series.npz,
per configs/baseline.yaml) are NOT present in this repository instance -- M1's own
graph_construction.py computes the dense Pearson matrix and sparsifies it in one
step without ever persisting the dense intermediate (a documented storage-efficiency
choice: 1.4GB sparse vs. ~450GB dense for all subjects/windows). So there is no dense,
pre-sparsification 360x360 matrix stored anywhere to load.

What IS real and IS used below: data/processed/graphs/<subject>.npz stores, per
window, the REAL signed Pearson r values that survived the top-20% sparsification
-- i.e. every nonzero cell in the matrix built here is an actual computed
correlation from the actual pipeline. We reconstruct the full 360x360 array by
placing those real values back at their real (i,j) positions and leaving every
sparsified-out cell at 0. This is labeled explicitly as a SPARSE matrix, not
claimed to be the dense original.

The one figure that truly cannot be built from real data (raw ROI time series
themselves) is clearly marked ILLUSTRATIVE.
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from style import *

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "01_original_data")
SUBJECT = "UM_1_007"   # real ABIDE I test-control subject, University of Michigan site
WINDOW = 0

graph_path = os.path.join(ROOT, "data", "processed", "graphs", f"{SUBJECT}.npz")
d = np.load(graph_path)
subject_id = str(d["subject_id"]); site = str(d["site"]); label = int(d["label"])
num_windows = int(d["num_windows"]); num_nodes = int(d["num_nodes"])
edge_index = d["edge_indices"][WINDOW]     # [2, E] real ROI index pairs kept after top-20% sparsification
edge_weight = d["edge_weights"][WINDOW]    # [E] real signed Pearson r for those pairs
threshold = float(d["thresholds"][WINDOW])
w_start, w_end = int(d["window_starts"][WINDOW]), int(d["window_ends"][WINDOW])

assert num_nodes == 360, "unexpected ROI count"
M = np.zeros((num_nodes, num_nodes), dtype=np.float32)
M[edge_index[0], edge_index[1]] = edge_weight
n_nonzero_undirected = int((edge_index[0] < edge_index[1]).sum())

print(f"[verify] subject={subject_id} site={site} label={'ASD' if label else 'Control'} "
      f"window={WINDOW} ({w_start}-{w_end} TR) |E_undirected_kept|={n_nonzero_undirected} "
      f"|r|_threshold={threshold:.4f} matrix_shape={M.shape}")

# ---------------------------------------------------------------------------
# Figure 1: the real sparsified FC matrix
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Signed Pearson correlation r", fontsize=10.5)
ax.set_xlabel("ROI index (1–360)")
ax.set_ylabel("ROI index (1–360)")
ax.set_title(f"Functional-Connectivity Matrix — Subject {subject_id} ({site}), window {WINDOW}\n"
             f"(TRs {w_start}–{w_end}) — top-20% sparsified, {n_nonzero_undirected} undirected edges kept",
             fontsize=12.5)
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/graphs/{SUBJECT}.npz (window {WINDOW})  |  code: src/data/graph_construction.py:sparsify_fc_matrix")
savefig(fig, os.path.join(OUT, "pearson_correlation_matrix.png"))

# ---------------------------------------------------------------------------
# Figure 2: the same matrix, with an explanatory reading-guide panel
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 0.14, 1.3], wspace=0.55)
ax = fig.add_subplot(gs[0])
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
cax = fig.add_subplot(gs[1])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Signed Pearson r", fontsize=10)
ax.set_xlabel("ROI index j")
ax.set_ylabel("ROI index i")
ax.set_title(f"Subject {subject_id} · window {WINDOW}", fontsize=12)
# Zoom inset box on a small block to make cell-level reading obvious
zi0, zi1 = 40, 55
import matplotlib.patches as mpatches
ax.add_patch(mpatches.Rectangle((zi0, zi0), zi1 - zi0, zi1 - zi0, fill=False, edgecolor=INK, linewidth=1.3))

axr = fig.add_subplot(gs[2])
axr.axis("off")
axr.set_title("How to read this matrix", fontsize=13, loc="left", pad=14)
lines = [
    ("Rows / Columns", "Each of the 360 axis positions is one ABIDE ROI (brain region)."),
    ("Cell (i, j)", "The signed Pearson correlation between ROI i's and ROI j's\nBOLD time series inside this 30-TR window."),
    ("Color", "Red = positive correlation, Blue = negative (anti-)correlation.\nWhite / 0 = not among the top-20% strongest connections\n(sparsified out -- not “no signal”, just not retained as an edge)."),
    ("Diagonal", "Always 0 -- a region is never connected to itself\n(no self-loops, by construction)."),
    (f"Highlighted box", f"ROIs {zi0}–{zi1}, a small real block, zoomed conceptually\nto show individual cell values are real, not interpolated."),
]
y = 0.86
LINE_H = 0.038
for head, body in lines:
    axr.text(0, y, head, fontsize=11.5, fontweight="bold", va="top", transform=axr.transAxes)
    y -= LINE_H * 1.15
    axr.text(0, y, body, fontsize=9.8, va="top", color=MUTED, transform=axr.transAxes, linespacing=1.5)
    n_lines = body.count("\n") + 1
    y -= LINE_H * n_lines + 0.055
tag_badge(fig, illustrative=False)
footer(fig, f"source: data/processed/graphs/{SUBJECT}.npz (window {WINDOW})  |  code: src/data/graph_construction.py")
savefig(fig, os.path.join(OUT, "pearson_correlation_matrix_labeled.png"))

# ---------------------------------------------------------------------------
# Figure 3: ILLUSTRATIVE ONLY -- time series -> window -> correlation concept.
# Raw ROI time series are not present in this repo instance (data/raw/ was never
# shipped -- see note above and README.md), so this uses clearly-labeled
# synthetic example signals to explain the MECHANISM, not to claim a real result.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
T = 60
t = np.arange(T)
base = np.sin(t / 6.0)
roiA = base + rng.normal(0, 0.25, T)
roiB = 0.8 * base + rng.normal(0, 0.25, T)          # correlated with A
roiC = np.sin(t / 4.0 + 2.0) + rng.normal(0, 0.3, T)  # weakly related

fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
fig.subplots_adjust(hspace=0.55, wspace=0.32, top=0.88)
for ax, sig, name, col in zip(axes[0], [roiA, roiB], ["Example ROI A", "Example ROI B"], [ACCENT, GOOD]):
    ax.plot(t, sig, color=col, lw=1.8)
    ax.axvspan(10, 40, color=ANOMALY, alpha=0.12)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("timepoint (TR)"); ax.set_ylabel("BOLD signal (a.u.)")
axes[0, 0].text(25, axes[0, 0].get_ylim()[1] * 0.85, "30-TR\nwindow", ha="center", fontsize=9, color=ANOMALY, fontweight="bold")

ax = axes[1, 0]
ax.plot(t, roiA, color=ACCENT, lw=1.3, alpha=0.6, label="ROI A")
ax.plot(t, roiB, color=GOOD, lw=1.3, alpha=0.6, label="ROI B")
ax.plot(t, roiC, color=ASD, lw=1.3, alpha=0.6, label="ROI C")
ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("Overlaid example signals", fontsize=11)
ax.set_xlabel("timepoint (TR)")

ax = axes[1, 1]
ex_mat = np.corrcoef(np.vstack([roiA[10:40], roiB[10:40], roiC[10:40]]))
im = ax.imshow(ex_mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["A", "B", "C"])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["A", "B", "C"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{ex_mat[i,j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(ex_mat[i, j]) > 0.5 else INK)
ax.set_title("Pearson correlation\n(this windowed example)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("CONCEPT: ROI time series → 30-TR window → Pearson correlation matrix", fontsize=14, fontweight="bold", y=1.02)
tag_badge(fig, illustrative=True)
footer(fig, "synthetic example signals -- raw ABIDE ROI time series are not present in this repo instance (data/raw/ not shipped); "
            "the REAL 360x360 matrix built the same way is pearson_correlation_matrix.png")
savefig(fig, os.path.join(OUT, "temporal_window_example.png"))

# dump verification numbers
with open(os.path.join(OUT, "_verification.json"), "w") as f:
    json.dump({
        "subject_id": subject_id, "site": site, "label": label, "window": WINDOW,
        "window_start_TR": w_start, "window_end_TR": w_end,
        "num_undirected_edges_kept": n_nonzero_undirected,
        "abs_r_threshold": threshold, "num_nodes": num_nodes,
        "source_file": f"data/processed/graphs/{SUBJECT}.npz",
    }, f, indent=2)
print("done: 01_original_data")
