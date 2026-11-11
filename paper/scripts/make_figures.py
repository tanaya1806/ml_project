"""Paper figures (vector PDF), generated only from real project outputs. Column width 3.45 in, page width 7.16 in."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_curve
ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"; FIG = f"{ROOT}/paper/figures"; RES = f"{ROOT}/paper/results"
sys.path.insert(0, f"{ROOT}/Output_to_be_shown/_scripts")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
  "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
  "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
C = {"ctl": "#1b7f79", "asd": "#b5651d", "gen": "#6a3d9a", "rnd": "#7f7f7f", "acc": "#2a5db0", "bad": "#b2182b"}
W1, W2 = 3.45, 7.16
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
def save(fig, name): fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig); print("saved", name)

# ---------------- Fig 2: example FC matrix, stable subgraph, weight trajectories ----------------
SUBJ, SEED = "UM_1_007", 284
d = np.load(f"{ROOT}/data/processed/graphs/{SUBJ}.npz")
M = np.zeros((360, 360), np.float32); ei, ew = d["edge_indices"][0], d["edge_weights"][0]; M[ei[0], ei[1]] = ew
from common_model import cache_for
c = cache_for(SUBJ); roi = np.asarray(c[f"roi_indices_{SEED}"]); EI = c[f"edge_index_{SEED}"]; EW = c[f"edge_weight_{SEED}"]
fig = plt.figure(figsize=(W2, 2.25)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.34)
ax = fig.add_subplot(gs[0]); im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
ax.set_xlabel("ROI index"); ax.set_ylabel("ROI index"); ax.set_title("Sparsified FC, window 1"); tag(ax, "(a)")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("signed Pearson $r$", fontsize=7); cb.ax.tick_params(labelsize=6)
ax = fig.add_subplot(gs[1]); G = nx.Graph(); G.add_nodes_from(range(len(roi))); seen = set()
for a, b, w in zip(EI[0][0], EI[0][1], EW[0]):
    k = (min(a, b), max(a, b))
    if k in seen: continue
    seen.add(k); G.add_edge(int(a), int(b), weight=float(w))
pos = nx.kamada_kawai_layout(G, weight=None)
for u, v, dd in G.edges(data=True):
    ax.plot(*zip(pos[u], pos[v]), color=C["ctl"] if dd["weight"] > 0 else C["asd"], lw=0.25 + 0.9 * abs(dd["weight"]), alpha=0.45)
sl = int(np.where(roi == SEED)[0][0]); xs = [pos[n][0] for n in G]; ys = [pos[n][1] for n in G]
ax.scatter(xs, ys, s=7, c="#dde3f0", edgecolors=C["acc"], linewidths=0.4, zorder=3); ax.scatter([pos[sl][0]], [pos[sl][1]], s=26, c=C["gen"], zorder=4)
ax.axis("off"); ax.set_title(f"Stable subgraph, seed {SEED}"); ax.text(0.5, -0.04, f"{len(roi)} nodes, {len(seen)} edges (window 1)", transform=ax.transAxes, ha="center", va="top", fontsize=6.5); tag(ax, "(b)")
ax = fig.add_subplot(gs[2]); T = len(EI); pairs = {}
for t in range(T):
    for a, b, w in zip(EI[t][0], EI[t][1], EW[t]):
        k = (min(int(a), int(b)), max(int(a), int(b))); pairs.setdefault(k, np.full(T, np.nan))[t] = w
freq = sorted(pairs, key=lambda k: -np.isfinite(pairs[k]).sum() - 0.001 * abs(np.nanmean(pairs[k])))
inc = [k for k in freq if sl in k][:5]
for i, k in enumerate(inc):
    ax.plot(np.arange(1, T + 1), pairs[k], marker="o", ms=2, lw=0.8, label=f"ROI {roi[k[0]]}$-$ROI {roi[k[1]]}")
ax.set_xlabel("window $t$"); ax.set_ylabel("edge weight $w_{ij}^{(t)}$"); ax.set_title("Edge weights over windows"); ax.axhline(0, color="k", lw=0.3)
ax.legend(fontsize=5.5, ncol=1, frameon=False, loc="lower left"); tag(ax, "(c)")
save(fig, "fig_example")

# ---------------- Fig 3: training curves ----------------
tr = json.load(open(f"{ROOT}/outputs/experiments/phase1/training_report.json"))["history"]
ep = [h["epoch"] for h in tr]
fig, ax = plt.subplots(figsize=(W1, 2.0))
ax.plot(ep, [h["train_pred_loss"] for h in tr], "o-", ms=3.5, color=C["acc"], label="train prediction loss")
ax.plot(ep, [h["val_pred_loss"] for h in tr], "s-", ms=3.5, color=C["bad"], label="validation prediction loss (controls)")
ax.axvline(4, color="k", lw=0.5, ls=":"); ax.text(4.05, 0.163, "checkpoint\n(epoch 4)", fontsize=6, va="top")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE$(z_{t+1},\\hat z_{t+1})$"); ax.set_xticks(ep)
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.plot(ep, [h["train_var_penalty"] for h in tr], "^--", color=C["rnd"], ms=3, label="variance penalty $L_{var}$")
ax2.set_ylabel("$L_{var}$"); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", fontsize=6.2, bbox_to_anchor=(0.5, -0.27), ncol=1)
save(fig, "fig_training")

# ---------------- Fig 4: Phase-1 subject-level score distributions + ROC (+CI) ----------------
yv = np.load(f"{RES}/val_scores_gru.npy"); yt = np.load(f"{RES}/test_scores_gru.npy"); Lt = np.load(f"{RES}/test_labels.npy")
ab = json.load(open(f"{RES}/ablation.json")); m = json.load(open(f"{ROOT}/outputs/experiments/phase1/metrics.json"))
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.25), gridspec_kw={"wspace": 0.28}); ax = axs[0]
bins = np.linspace(min(yv.min(), yt.min()) - 0.5, max(yv.max(), yt.max()) + 0.5, 22)
ax.hist(yv, bins, density=True, alpha=0.55, color=C["rnd"], label=f"val-control ($n$={len(yv)})")
ax.hist(yt[Lt == 0], bins, density=True, histtype="step", color=C["ctl"], lw=1.4, label=f"test-control ($n$={(Lt==0).sum()})")
ax.hist(yt[Lt == 1], bins, density=True, histtype="step", color=C["asd"], lw=1.4, ls="--", label=f"test-ASD ($n$={(Lt==1).sum()})")
ax.axvline(m["tau_subject"], color=C["bad"], lw=1, ls=":"); ax.text(m["tau_subject"] + 0.2, ax.get_ylim()[1] * 0.9, "$\\tau_{subject}$", color=C["bad"], fontsize=7)
ax.set_xlabel("subject score $S=\\max_t A_t$ (over 20 seeds)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=6.3); tag(ax, "(a)")
ax = axs[1]; fpr, tpr, _ = roc_curve(Lt, yt); ax.plot(fpr, tpr, color=C["acc"], label=f"GRU error (AUROC {m['auroc']:.3f})")
rng = np.random.default_rng(0); grid = np.linspace(0, 1, 101); tp = []
for _ in range(500):
    i = rng.integers(0, len(Lt), len(Lt))
    if Lt[i].min() == Lt[i].max(): continue
    f_, t_, _ = roc_curve(Lt[i], yt[i]); tp.append(np.interp(grid, f_, t_))
lo, hi = np.percentile(tp, [2.5, 97.5], axis=0); ax.fill_between(grid, lo, hi, color=C["acc"], alpha=0.15, label="95% bootstrap band")
ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--", label="chance"); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
ax.legend(frameon=False, loc="lower right", fontsize=6.3); tag(ax, "(b)")
save(fig, "fig_phase1")

# ---------------- Fig 5: anomaly timeline (two real subjects) ----------------
from common_model import embedding_sequence, gru_predict, anomaly_scores
fig, axs = plt.subplots(2, 1, figsize=(W1, 2.9), sharex=True, gridspec_kw={"hspace": 0.18})
for ax, sid, col, lab in zip(axs, ["UM_1_007", "UM_1_062"], [C["ctl"], C["asd"]], ["control", "ASD"]):
    z, _ = embedding_sequence(sid, SEED); a = anomaly_scores(z, gru_predict(z)).detach().cpu().numpy(); x = np.arange(1, len(a) + 1)
    ax.plot(x, a, "o-", color=col, ms=2.5); ax.axhline(m["tau_transition"], color=C["bad"], ls="--", lw=0.9)
    o = x[a > m["tau_transition"]]; ax.scatter(o, a[o - 1], marker="x", color=C["bad"], s=22, zorder=4)
    ax.set_ylabel("$A_t$"); ax.text(0.01, 0.9, f"{sid} ({lab}), seed {SEED}", transform=ax.transAxes, fontsize=6.5, va="top")
axs[1].text(0.99, 0.79, "$\\tau_{transition}$=%.2f (dashed)" % m["tau_transition"], transform=axs[1].transAxes, fontsize=6.5, color=C["bad"], ha="right")
axs[1].set_xlabel("transition $t\\rightarrow t+1$"); save(fig, "fig_timeline")

# ---------------- Fig 6: Phase 2 ----------------
V = json.load(open(f"{ROOT}/outputs/experiments/phase2/generator_validation.json")); sev = V["severity_control"]["per_group"]; det = V["detectability"]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw={"wspace": 0.42}); ax = axs[0]; al = [0.25, 0.55, 0.9]
for meth, col, lab, mk in [("structure_aware", C["gen"], "structure-aware", "o"), ("random", C["rnd"], "random", "s")]:
    ax.errorbar(al, [sev[f"{meth}/{l}"]["mean"] for l in ["low", "medium", "high"]], yerr=[sev[f"{meth}/{l}"]["std"] for l in ["low", "medium", "high"]],
                marker=mk, ms=3.5, capsize=2, color=col, label=lab)
ax.axhline(V["tau_transition"], color=C["bad"], ls="--", lw=0.8); ax.text(0.26, V["tau_transition"] + 0.2, "$\\tau_{transition}$", color=C["bad"], fontsize=6.5)
ax.set_xlabel("severity $\\alpha$"); ax.set_ylabel("score after generation"); ax.legend(frameon=False, fontsize=6.3, loc="upper left"); tag(ax, "(a)")
ax = axs[1]; keys = ["real_normal_val_control", "real_asd_test", "random/high", "structure_aware/low", "structure_aware/medium", "structure_aware/high"]
labs = ["real\nnormal", "real\nASD", "random\nhigh", "SA\nlow", "SA\nmed.", "SA\nhigh"]; cols = [C["ctl"], C["asd"], C["rnd"], C["gen"], C["gen"], C["gen"]]
b = ax.bar(range(6), [100 * det[k] for k in keys], color=cols)
for r_, k in zip(b, keys): ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 0.8, f"{100*det[k]:.1f}", ha="center", fontsize=6)
ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=6); ax.set_ylabel("transitions above $\\tau_{transition}$ (%)"); tag(ax, "(b)")
from common_model import get_models, cache_for as cf
import torch
from src.generation.structure_aware_generator import compute_edge_gradient, _undirected_pairs
from src.models.embedding_sequence import seed_embedding_sequence
gin, gru, ckpt, device, p1 = get_models(); cc = cf("KKI_002"); T0 = 1; S0 = 1
with torch.no_grad():
    zs = seed_embedding_sequence(gin, cc, S0, device); zt = gru(zs[:T0 + 1].unsqueeze(0)).squeeze(0)[-1]
eidx = cc[f"edge_index_{S0}"][T0 + 1]; ewt = cc[f"edge_weight_{S0}"][T0 + 1].astype(np.float32)
g0 = compute_edge_gradient(gin, eidx, ewt, len(cc[f"roi_indices_{S0}"]), zt, device); pr = _undirected_pairs(eidx)
sens = np.sort(np.abs([sum(g0[i] for i in v) for v in pr.values()])); n = len(sens); top = round(0.9 * 0.15 * n)
ax = axs[2]; ax.plot(np.arange(n), sens, color=C["acc"]); ax.axvspan(n - top, n, color=C["gen"], alpha=0.15); ax.text(n - top - 8, sens.max() * 0.55, f"selected\n(top {top})", ha="right", fontsize=6.3, color=C["gen"])
ax.set_xlabel(f"edge rank (of {n})"); ax.set_ylabel("$|\\partial A_t/\\partial w_{ij}|$"); tag(ax, "(c)")
save(fig, "fig_phase2")
json.dump({"example_edges": int(n), "example_top": int(top)}, open(f"{RES}/fig_phase2_example.json", "w"))

# ---------------- Fig 7: Phase 3 ----------------
f3 = json.load(open(f"{ROOT}/outputs/experiments/phase3/final_comparison.json")); shots = ["1-shot", "5-shot", "10-shot"]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw={"wspace": 0.34})
for ax, met, ttl, ref in [(axs[0], "f1", "F1", None), (axs[1], "recall", "Recall", None), (axs[2], "auroc", "AUROC", 0.5)]:
    for i, (cond, col, lab, mk, off) in enumerate([("few_shot_only", C["asd"], "few-shot only", "o", -0.12), ("few_shot_plus_structure_aware", C["gen"], "+ structure-aware", "s", 0),
                                              ("few_shot_plus_random", C["rnd"], "+ random", "^", 0.12)]):
        ax.errorbar(np.arange(3) + off, [f3[s][cond][f"{met}_mean"] for s in shots], yerr=[f3[s][cond][f"{met}_std"] for s in shots], marker=mk, ms=3.5, capsize=2, color=col, label=lab, lw=0.9)
    if ref: ax.axhline(ref, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels(["1", "5", "10"]); ax.set_xlabel("real ASD support subjects $k$"); ax.set_ylabel(ttl); tag(ax, "(%s)" % "abc"[[axs[0], axs[1], axs[2]].index(ax)])
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.08)); axs[2].set_ylim(0.38, 0.62); save(fig, "fig_phase3")
print("done")
