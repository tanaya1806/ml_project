"""Figures + tables from paper/results/extra_baselines.json (600 dpi, 3.4 in column width, text >= 7 pt). Output: paper/extra_results/."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
O = "paper/extra_results"; os.makedirs(O, exist_ok=True)
J = json.load(open("paper/results/extra_baselines.json")); cur = J["curves"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
  "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
  "savefig.bbox": "tight", "savefig.dpi": 600})
NAME = {"temporal_subgraph_detector": "Temporal subgraph detector (proposed)", "whole_brain_gin_gru": "Whole-brain GIN+GRU", "static_baseline": "Static score", "persistence_baseline": "Persistence score",
        "ocsvm": "One-class SVM", "iforest": "Isolation forest", "autoencoder": "Autoencoder (normative)", "few_shot_only": "Few-shot head", "few_shot_random_anomalies": "Few-shot + random anomalies",
        "few_shot_structure_aware_anomalies": "Few-shot + structure-aware anomalies"}
COL = {"temporal_subgraph_detector": "#0f7c8f", "whole_brain_gin_gru": "#6a3d9a", "static_baseline": "#2a5db0", "persistence_baseline": "#7f9fd0", "ocsvm": "#b5651d", "iforest": "#d9a066",
       "autoencoder": "#8c564b", "few_shot_only": "#999999", "few_shot_random_anomalies": "#666666", "few_shot_structure_aware_anomalies": "#333333"}
LS = {k: ("-" if i < 7 else "--") for i, k in enumerate(NAME)}
W = 3.4; y_re, y_sy = np.array(cur["y_real"]), np.array(cur["y_syn"])
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")
def save(fig, n): fig.savefig(f"{O}/{n}.png"); plt.close(fig); print("saved", n)

# 1. synthetic ROC + PR
fig, axs = plt.subplots(2, 1, figsize=(W, 5.6), gridspec_kw={"hspace": 0.35})
for k in NAME:
    s = np.array(cur["syn"][k]); f, t, _ = roc_curve(y_sy, s); axs[0].plot(f, t, color=COL[k], ls=LS[k], label=f"{NAME[k]} ({J['syn'][k]['auroc']:.2f})")
    p, r, _ = precision_recall_curve(y_sy, s); axs[1].plot(r, p, color=COL[k], ls=LS[k])
axs[0].plot([0, 1], [0, 1], "k:", lw=0.7); axs[0].set_xlabel("false-positive rate"); axs[0].set_ylabel("true-positive rate"); tag(axs[0], "(a)")
axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision"); axs[1].set_ylim(0.4, 1.02); tag(axs[1], "(b)")
axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6.6, title="AUROC in brackets", title_fontsize=6.6)
save(fig, "fig_roc_pr_synthetic")
# 2. real-test ROC
fig, ax = plt.subplots(figsize=(W, 3.2))
for k in ("temporal_subgraph_detector", "whole_brain_gin_gru", "static_baseline", "persistence_baseline", "ocsvm", "iforest", "autoencoder"):
    f, t, _ = roc_curve(y_re, np.array(cur["real"][k])); ax.plot(f, t, color=COL[k], label=f"{NAME[k]} ({J['real'][k]['auroc']:.2f})")
ax.plot([0, 1], [0, 1], "k:", lw=0.7); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), title="AUROC in brackets", title_fontsize=6.6)
save(fig, "fig_roc_real_test")
# 3. metric bars (synthetic): accuracy with CI, F1, MCC
order = sorted(NAME, key=lambda k: -J["syn"][k]["accuracy"]); fig, ax = plt.subplots(figsize=(W, 4.4)); y = np.arange(len(order)); h = 0.26
for j, (m, lab, col) in enumerate((("accuracy", "accuracy", "#0f7c8f"), ("f1", "F1", "#d9a066"), ("mcc", "MCC", "#6a3d9a"))):
    v = np.array([J["syn"][k][m] for k in order]); err = None
    if m in ("accuracy", "f1", "mcc"): lo = np.array([J["syn"][k]["ci"][m][0] for k in order]); hi = np.array([J["syn"][k]["ci"][m][1] for k in order]); err = [np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)]
    ax.barh(y + (j - 1) * h, v, h, color=col, label=lab, xerr=err, error_kw=dict(lw=0.6, capsize=1.2))
ax.set_yticks(y); ax.set_yticklabels([NAME[k] for k in order], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.set_xlabel("score (bars: 95% bootstrap interval)"); ax.legend(frameon=False, loc="lower right")
save(fig, "fig_metrics_synthetic")
# 4. confusion matrices
pick = ["temporal_subgraph_detector", "whole_brain_gin_gru", "static_baseline", "autoencoder", "few_shot_only", "few_shot_structure_aware_anomalies"]
fig, axs = plt.subplots(3, 2, figsize=(W, 5.0), gridspec_kw={"hspace": 0.62, "wspace": 0.3})
for ax, k in zip(axs.ravel(), pick):
    m = J["syn"][k]; M = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]); ax.imshow(M, cmap="Blues", vmin=0, vmax=50)
    for i in range(2):
        for j in range(2): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > 30 else "black", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "flagged"], fontsize=6.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["normal-\nlike", "changed"], fontsize=6.5)
    ax.set_title(NAME[k].replace(" (proposed)", "").replace("Few-shot + structure-aware anomalies", "Few-shot + guided anomalies"), fontsize=6.8); ax.spines[:].set_visible(False)
save(fig, "fig_confusion_synthetic")
# 5. detection rate by strength
st = np.array(cur["strength"]); fig, ax = plt.subplots(figsize=(W, 3.4)); xs = np.arange(3)
for k in NAME:
    fl = np.array(cur["syn_flag"][k]); ax.plot(xs, [100 * fl[st == s].mean() for s in ("low", "moderate", "high")], marker="o", ms=3, color=COL[k], ls=LS[k], label=NAME[k])
ax.set_xticks(xs); ax.set_xticklabels(["low", "moderate", "high"]); ax.set_xlabel("strength of the injected change"); ax.set_ylabel("changed scans flagged (%)"); ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "fig_detection_by_strength")
# 6. threshold sweep for the proposed detector and the whole-brain baseline
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.45}); pc = np.array([80, 85, 90, 95, 97.5, 99, 99.9])
for ax, k in zip(axs, ("temporal_subgraph_detector", "whole_brain_gin_gru")):
    v = np.array(cur["val"][k]); s = np.array(cur["syn"][k]); sens = [100 * (s[y_sy == 1] > np.percentile(v, p)).mean() for p in pc]; fa = [100 * (s[y_sy == 0] > np.percentile(v, p)).mean() for p in pc]
    ax.plot(pc, sens, "o-", ms=3, color="#0f7c8f", label="changed scans flagged"); ax.plot(pc, fa, "s-", ms=3, color="#b2182b", label="normal-like flagged"); ax.axvline(95, color="k", ls=":", lw=0.6)
    ax.set_xlabel("threshold percentile"); ax.set_title("proposed" if "temporal" in k else "whole-brain", fontsize=8); ax.set_ylim(-3, 105)
axs[0].set_ylabel("%"); axs[0].legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(1.15, -0.25), ncol=2)
save(fig, "fig_threshold_sweep")

# tables
def row(k, m):
    ci = m["ci"]; return [NAME[k], f"{m['accuracy']:.3f} [{ci['accuracy'][0]:.2f}, {ci['accuracy'][1]:.2f}]", f"{m['sensitivity']:.3f}", f"{m['specificity']:.3f}", f"{m['balanced_accuracy']:.3f}", f"{m['precision']:.3f}", f"{m['f1']:.3f}", f"{m['mcc']:.3f}",
                           f"{m['auroc']:.3f} [{ci['auroc'][0]:.2f}, {ci['auroc'][1]:.2f}]", f"{m['auprc']:.3f}", f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}"]
H = ["Model", "Accuracy [95% CI]", "Sensitivity", "Specificity", "Balanced acc.", "Precision", "F1", "MCC", "AUROC [95% CI]", "AUPRC", "TP/FP/TN/FN"]
md = []
for split, title in (("syn", "100 synthetic scans (changed vs normal-like)"), ("real", "Real held-out test set (131 patients, ASD vs control)")):
    ks = [k for k in NAME if k in J[split]]; ks.sort(key=lambda k: -J[split][k]["accuracy"]); rows = [row(k, J[split][k]) for k in ks]
    import csv
    with open(f"{O}/table_{'synthetic' if split=='syn' else 'real_test'}.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(H); w.writerows(rows)
    md += [f"### {title}", "| " + " | ".join(H) + " |", "|" + "---|" * len(H)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
st_rows = [[NAME[k]] + [f"{100*np.array(cur['syn_flag'][k])[st==s].mean():.0f}%" for s in ("low", "moderate", "high")] + [f"{100*np.array(cur['syn_flag'][k])[y_sy==0].mean():.0f}%"] for k in NAME]
with open(f"{O}/table_detection_by_strength.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["Model", "Low", "Moderate", "High", "False alarms (normal-like)"]); w.writerows(st_rows)
md += ["### Changed scans flagged by strength", "| Model | Low | Moderate | High | False alarms |", "|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in st_rows]
open(f"{O}/tables.md", "w").write("\n".join(md)); print("tables written")
