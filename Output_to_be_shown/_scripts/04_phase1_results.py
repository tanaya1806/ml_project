"""
05_phase1_results -- real aggregate Phase 1 detection results, recomputed from
scratch by mirroring src/evaluate/detect_baseline.py's real evaluation loop
(same checkpoint, same subjects, same thresholds), so per-subject scores are
available for ROC/PR/confusion-matrix figures -- and cross-checked against the
project's own committed outputs/experiments/phase1/metrics.json.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              average_precision_score, confusion_matrix)
sys.path.insert(0, os.path.dirname(__file__))
from style import *
from common_model import get_models, cache_for, pools

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "05_phase1_results")

gin, gru, ckpt, device, p1_cfg = get_models()
P = pools()
seed_rois = P["seed_rois"]
test_all = P["test_all"]

reported = json.load(open(os.path.join(ROOT, "outputs/experiments/phase1/metrics.json")))
tau_transition = reported["tau_transition"]
tau_subject = reported["tau_subject"]

from src.models.embedding_sequence import seed_embedding_sequence
from src.anomaly.scoring import transition_scores

t0 = time.time()
y_true, y_score, subj_ids = [], [], []
with torch.no_grad():
    for i, sid in enumerate(test_all):
        cache = cache_for(sid)
        max_score = 0.0
        for seed in seed_rois:
            z_seq = seed_embedding_sequence(gin, cache, int(seed), device)
            if z_seq.shape[0] < 2:
                continue
            z_hat = gru(z_seq.unsqueeze(0)).squeeze(0)
            a_t = transition_scores(z_seq, z_hat).cpu().numpy()
            if len(a_t):
                max_score = max(max_score, float(a_t.max()))
        y_true.append(int(cache["label"]))
        y_score.append(max_score)
        subj_ids.append(sid)
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(test_all)} test subjects scored ({time.time()-t0:.0f}s)")

y_true = np.array(y_true); y_score = np.array(y_score)
y_pred = (y_score > tau_subject).astype(int)

recomputed = {
    "precision": float(np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_pred == 1))),
    "recall": float(np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))),
    "auroc": float(roc_auc_score(y_true, y_score)),
    "auprc": float(average_precision_score(y_true, y_score)),
    "num_flagged": int(y_pred.sum()),
}
print("[verify] recomputed vs. reported (outputs/experiments/phase1/metrics.json):")
print("  recomputed:", json.dumps(recomputed, indent=2))
print("  reported  :", json.dumps({k: reported[k] for k in ["precision", "recall", "auroc", "auprc", "num_subjects_flagged_anomalous"]}, indent=2))
assert abs(recomputed["auroc"] - reported["auroc"]) < 1e-6, "AUROC mismatch vs. committed metrics.json!"
assert abs(recomputed["auprc"] - reported["auprc"]) < 1e-6, "AUPRC mismatch vs. committed metrics.json!"
print("[verify] EXACT MATCH to committed metrics.json (bit-for-bit AUROC/AUPRC) -- recomputation confirmed faithful.")

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
assert (tp, tn, fp, fn) == (reported["true_positives"], reported["true_negatives"], reported["false_positives"], reported["false_negatives"])
print(f"[verify] confusion matrix matches exactly: TP={tp} TN={tn} FP={fp} FN={fn}")

# ---------------------------------------------------------------------------
# phase1_metrics.png -- real metrics panel
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.4))
ax.axis("off")
rows = [
    ("Precision", f"{reported['precision']:.3f}"),
    ("Recall", f"{reported['recall']:.3f}"),
    ("F1", f"{reported['f1']:.3f}"),
    ("AUROC", f"{reported['auroc']:.3f}  (0.500 = chance)"),
    ("AUPRC", f"{reported['auprc']:.3f}  (base rate = {sum(y_true)/len(y_true):.3f})"),
    ("τ_transition", f"{reported['tau_transition']:.4f}"),
    ("τ_subject", f"{reported['tau_subject']:.4f}"),
    ("True Positives", str(reported["true_positives"])),
    ("True Negatives", str(reported["true_negatives"])),
    ("False Positives", str(reported["false_positives"])),
    ("False Negatives", str(reported["false_negatives"])),
    ("Test subjects", f"{reported['num_test_subjects']}  (70 control / 61 ASD)"),
    ("Subjects flagged anomalous", str(reported["num_subjects_flagged_anomalous"])),
    ("Anomalous transitions localized", str(reported["num_anomalous_transitions_detected"])),
]
ax.set_title("Phase 1 — Real Baseline Detection Metrics\n(131 held-out test subjects, never used for training or calibration)",
              fontsize=14, pad=18)
y = 0.92
for k, v in rows:
    ax.text(0.03, y, k, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.62, y, v, fontsize=12, family="monospace", transform=ax.transAxes)
    y -= 0.068
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase1/metrics.json  |  code: src/evaluate/detect_baseline.py  |  independently re-verified above")
savefig(fig, os.path.join(OUT, "phase1_metrics.png"))

# ---------------------------------------------------------------------------
# phase1_confusion_matrix.png -- real, from real predictions
# ---------------------------------------------------------------------------
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(6, 5.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nControl", "Predicted\nASD"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True\nControl", "True\nASD"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_title(f"Phase 1 — Real Confusion Matrix (n={len(y_true)} test subjects)", fontsize=13)
tag_badge(fig, illustrative=False)
footer(fig, "source: real recomputed predictions (y_score > τ_subject), verified == outputs/experiments/phase1/metrics.json")
savefig(fig, os.path.join(OUT, "phase1_confusion_matrix.png"))

# ---------------------------------------------------------------------------
# phase1_roc_curve.png / phase1_pr_curve.png -- real curves from real scores
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_score)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"real ROC (AUROC = {reported['auroc']:.3f})")
ax.plot([0, 1], [0, 1], color=LINE, lw=1.5, ls="--", label="chance (AUROC = 0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Phase 1 — Real ROC Curve (131 test subjects)", fontsize=13)
ax.legend(fontsize=10, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
tag_badge(fig, illustrative=False)
footer(fig, "source: real per-subject max A_t scores vs. true diagnosis label  |  code: sklearn.metrics.roc_curve")
savefig(fig, os.path.join(OUT, "phase1_roc_curve.png"))

prec, rec, _ = precision_recall_curve(y_true, y_score)
base_rate = float(np.mean(y_true))
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(rec, prec, color=ASD, lw=2.2, label=f"real PR curve (AUPRC = {reported['auprc']:.3f})")
ax.axhline(base_rate, color=LINE, lw=1.5, ls="--", label=f"chance (base rate = {base_rate:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Phase 1 — Real Precision-Recall Curve (131 test subjects)", fontsize=13)
ax.legend(fontsize=10, loc="upper right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
tag_badge(fig, illustrative=False)
footer(fig, "source: real per-subject max A_t scores vs. true diagnosis label  |  code: sklearn.metrics.precision_recall_curve")
savefig(fig, os.path.join(OUT, "phase1_pr_curve.png"))

with open(os.path.join(OUT, "_verification.json"), "w") as f:
    json.dump({
        "recomputed": recomputed, "reported_metrics_json": reported,
        "exact_match_auroc": True, "exact_match_auprc": True,
        "exact_match_confusion_matrix": True,
        "num_test_subjects_scored": len(y_true),
    }, f, indent=2)
print("done: 04_phase1_results")
