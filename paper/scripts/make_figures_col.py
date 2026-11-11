"""Column-width (3.45 in) versions of the detector, generation and few-shot figures, from the same real outputs. PNG into docx_build/assets."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"; RES = f"{ROOT}/paper/results"; OUT = f"{ROOT}/paper/docx_build/assets"
sys.path.insert(0, f"{ROOT}/Output_to_be_shown/_scripts"); sys.path.insert(0, ROOT)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
  "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
  "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
C = {"ctl": "#1b7f79", "asd": "#b5651d", "gen": "#6a3d9a", "rnd": "#7f7f7f", "acc": "#2a5db0", "bad": "#b2182b"}
W1 = 3.45
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")
def save(fig, name): fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved", name)

# ---- detector: score distributions + ROC, stacked ----
yv = np.load(f"{RES}/val_scores_gru.npy"); yt = np.load(f"{RES}/test_scores_gru.npy"); Lt = np.load(f"{RES}/test_labels.npy")
m = json.load(open(f"{ROOT}/outputs/experiments/phase1/metrics.json"))
fig, axs = plt.subplots(2, 1, figsize=(W1, 4.0), gridspec_kw={"hspace": 0.5}); ax = axs[0]
bins = np.linspace(min(yv.min(), yt.min()) - 0.5, max(yv.max(), yt.max()) + 0.5, 22)
ax.hist(yv, bins, density=True, alpha=0.55, color=C["rnd"], label=f"validation controls ($n$={len(yv)})")
ax.hist(yt[Lt == 0], bins, density=True, histtype="step", color=C["ctl"], lw=1.4, label=f"test controls ($n$={(Lt==0).sum()})")
ax.hist(yt[Lt == 1], bins, density=True, histtype="step", color=C["asd"], lw=1.4, ls="--", label=f"test ASD ($n$={(Lt==1).sum()})")
ax.axvline(m["tau_subject"], color=C["bad"], lw=1, ls=":"); ax.text(m["tau_subject"] + 0.5, ax.get_ylim()[1] * 0.55, "threshold", color=C["bad"], fontsize=6.5)
ax.set_xlabel("person score (largest score over 20 regions and all steps)"); ax.set_ylabel("density"); ax.legend(frameon=False, loc="upper right"); tag(ax, "(a)")
ax = axs[1]; fpr, tpr, _ = roc_curve(Lt, yt); ax.plot(fpr, tpr, color=C["acc"], label=f"GRU error (AUROC {m['auroc']:.3f})")
rng = np.random.default_rng(0); grid = np.linspace(0, 1, 101); tp = []
for _ in range(500):
    i = rng.integers(0, len(Lt), len(Lt))
    if Lt[i].min() == Lt[i].max(): continue
    f_, t_, _ = roc_curve(Lt[i], yt[i]); tp.append(np.interp(grid, f_, t_))
lo, hi = np.percentile(tp, [2.5, 97.5], axis=0); ax.fill_between(grid, lo, hi, color=C["acc"], alpha=0.15, label="95% bootstrap band")
ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--", label="chance"); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.legend(frameon=False, loc="lower right"); tag(ax, "(b)")
save(fig, "fig_detector_col")

# ---- generation: (a),(b) top, (c) bottom ----
V = json.load(open(f"{ROOT}/outputs/experiments/phase2/generator_validation.json")); sev = V["severity_control"]["per_group"]; det = V["detectability"]
fig = plt.figure(figsize=(W1, 3.7)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.55); ax = fig.add_subplot(gs[0, 0]); al = [0.25, 0.55, 0.9]
for meth, col, lab, mk in [("structure_aware", C["gen"], "guided", "o"), ("random", C["rnd"], "random", "s")]:
    ax.errorbar(al, [sev[f"{meth}/{l}"]["mean"] for l in ["low", "medium", "high"]], yerr=[sev[f"{meth}/{l}"]["std"] for l in ["low", "medium", "high"]], marker=mk, ms=3, capsize=2, color=col, label=lab)
ax.axhline(V["tau_transition"], color=C["bad"], ls="--", lw=0.8); ax.set_xlabel("severity"); ax.set_ylabel("score after edit"); ax.legend(frameon=False, loc="upper left"); tag(ax, "(a)")
ax = fig.add_subplot(gs[0, 1]); keys = ["real_normal_val_control", "real_asd_test", "random/high", "structure_aware/low", "structure_aware/medium", "structure_aware/high"]
labs = ["real\ncontrol", "real\nASD", "random\nhigh", "guided\nlow", "guided\nmed.", "guided\nhigh"]; cols = [C["ctl"], C["asd"], C["rnd"], C["gen"], C["gen"], C["gen"]]
b = ax.bar(range(6), [100 * det[k] for k in keys], color=cols)
for r_, k in zip(b, keys): ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 0.8, f"{100*det[k]:.1f}", ha="center", fontsize=5.5)
ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=5.2); ax.set_ylabel("steps above threshold (%)"); tag(ax, "(b)")
from src.inference.engine import get_models
from src.data.subgraph_cache import load_subject_cache
import torch
from src.generation.structure_aware_generator import compute_edge_gradient, _undirected_pairs
from src.models.embedding_sequence import seed_embedding_sequence
gin, gru, ckpt, device, p1 = get_models(); cc = load_subject_cache("KKI_002", f"{ROOT}/data/processed/subgraph_cache"); T0 = 1; S0 = 1
with torch.no_grad():
    zs = seed_embedding_sequence(gin, cc, S0, device); zt = gru(zs[:T0 + 1].unsqueeze(0)).squeeze(0)[-1]
eidx = cc[f"edge_index_{S0}"][T0 + 1]; ewt = cc[f"edge_weight_{S0}"][T0 + 1].astype(np.float32)
g0 = compute_edge_gradient(gin, eidx, ewt, len(cc[f"roi_indices_{S0}"]), zt, device); pr = _undirected_pairs(eidx)
sens = np.sort(np.abs([sum(g0[i] for i in v) for v in pr.values()])); n = len(sens); top = round(0.9 * 0.15 * n)
ax = fig.add_subplot(gs[1, :]); ax.plot(np.arange(n), sens, color=C["acc"]); ax.axvspan(n - top, n, color=C["gen"], alpha=0.15)
ax.text(n - top - 8, sens.max() * 0.55, f"chosen: top {top} of {n}", ha="right", fontsize=6.5, color=C["gen"]); ax.set_xlabel("connections ranked by influence"); ax.set_ylabel("influence on score"); tag(ax, "(c)")
save(fig, "fig_generation_col")

# ---- few-shot: F1, recall, AUROC + legend ----
f3 = json.load(open(f"{ROOT}/outputs/experiments/phase3/final_comparison.json")); shots = ["1-shot", "5-shot", "10-shot"]
fig = plt.figure(figsize=(W1, 3.1)); gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5); axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
for ax, met, ttl, ref in [(axs[0], "f1", "F1", None), (axs[1], "recall", "Recall", None), (axs[2], "auroc", "AUROC", 0.5)]:
    for cond, col, lab, mk, off in [("few_shot_only", C["asd"], "ASD scans only", "o", -0.12), ("few_shot_plus_structure_aware", C["gen"], "+ guided synthetic", "s", 0), ("few_shot_plus_random", C["rnd"], "+ random synthetic", "^", 0.12)]:
        ax.errorbar(np.arange(3) + off, [f3[s][cond][f"{met}_mean"] for s in shots], yerr=[f3[s][cond][f"{met}_std"] for s in shots], marker=mk, ms=3, capsize=2, color=col, label=lab, lw=0.9)
    if ref: ax.axhline(ref, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels(["1", "5", "10"]); ax.set_xlabel("real ASD scans"); ax.set_ylabel(ttl); tag(ax, "(%s)" % "abc"[axs.index(ax)])
axs[2].set_ylim(0.38, 0.62)
lg = fig.add_subplot(gs[1, 1]); lg.axis("off"); h_, l_ = axs[0].get_legend_handles_labels(); lg.legend(h_, l_, loc="center", frameon=False, fontsize=6.5)
save(fig, "fig_fewshot_col")
