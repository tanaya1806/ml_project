"""One compact column-width two-panel figure for the paper: (a) synthetic-scan ROC, (b) detection rate by change strength. Reads paper/results/extra_baselines.json."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
J = json.load(open("paper/results/extra_baselines.json")); cur = J["curves"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.3,
  "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 600})
SHORT = {"temporal_subgraph_detector": "Proposed", "whole_brain_gin_gru": "Whole-brain GIN+GRU", "static_baseline": "Static score", "persistence_baseline": "Persistence score",
         "ocsvm": "One-class SVM", "iforest": "Isolation forest", "autoencoder": "Autoencoder"}
COL = {"temporal_subgraph_detector": "#0f7c8f", "whole_brain_gin_gru": "#6a3d9a", "static_baseline": "#2a5db0", "persistence_baseline": "#7f9fd0", "ocsvm": "#b5651d", "iforest": "#d9a066", "autoencoder": "#8c564b"}
y = np.array(cur["y_syn"]); st = np.array(cur["strength"]); xs = np.arange(3)
fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.4), gridspec_kw={"wspace": 0.45})
for k in SHORT:
    lw = 1.8 if k.startswith("temporal") else 1.0
    f, t, _ = roc_curve(y, np.array(cur["syn"][k])); a.plot(f, t, color=COL[k], lw=lw, label=SHORT[k].replace("Whole-brain GIN+GRU","Whole-brain GIN+GRU"))
    fl = np.array(cur["syn_flag"][k]); b.plot(xs, [100 * fl[st == s].mean() for s in ("low", "moderate", "high")], marker="o", ms=2.5, ls="--" if k == "ocsvm" else "-", lw=lw, color=COL[k])
a.plot([0, 1], [0, 1], "k:", lw=0.7); a.set_xlabel("false-positive rate"); a.set_ylabel("true-positive rate")
b.set_xticks(xs); b.set_xticklabels(["low", "mod.", "high"]); b.set_xlabel("strength of change"); b.set_ylabel("changed scans flagged (%)"); b.set_ylim(0, 105)
for ax, t in ((a, "(a)"), (b, "(b)")): ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.4, labelspacing=0.25)
fig.savefig("paper/figures_hires/fig_roc_strength_col.png"); plt.close(fig)
