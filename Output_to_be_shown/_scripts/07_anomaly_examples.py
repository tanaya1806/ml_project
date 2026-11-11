"""
anomaly_examples.png -- several REAL generated anomalies pulled directly from the
persisted outputs/experiments/phase2/generated_anomalies.json (7,200 real records),
each traceable to its exact subject/seed/severity/method/transition.
"""
import os, sys, json
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from style import *

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "06_phase2_generation")

recs = json.load(open(os.path.join(ROOT, "outputs/experiments/phase2/generated_anomalies.json")))

# Pick a real, varied, deterministic selection: 3 distinct real base subjects (each
# actually among the 60 real base subjects Phase 2 sampled), 1 severity each, both
# methods -- first match for each spec, in file order (no cherry-picking of values).
real_subjects = sorted(set(r["subject_id"] for r in recs))
chosen_subjects = [real_subjects[0], real_subjects[20], real_subjects[40]]
severities_pick = ["high", "medium", "low"]
picked = []
for sid, sevname in zip(chosen_subjects, severities_pick):
    for method in ["structure_aware", "random"]:
        for r in recs:
            if r["subject_id"] == sid and r["severity_name"] == sevname and r["method"] == method:
                picked.append(r)
                break

fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
for ax, r in zip(axes.flat, picked):
    ax.axis("off")
    is_sa = r["method"] == "structure_aware"
    col = CONTROL if is_sa else RANDOM
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=True,
                                facecolor=PANEL, edgecolor=col, linewidth=2))
    lines = [
        f"{'STRUCTURE-AWARE' if is_sa else 'RANDOM BASELINE'}",
        f"subject:  {r['subject_id']}  (label={'ASD' if r['label'] else 'control'})",
        f"seed ROI:  {r['seed_roi']}",
        f"transition:  {r['window_transition']}",
        f"severity:  {r['severity_name']}  (α={r['severity']})",
        f"edges modified:  {r['num_edges_changed']} / {r['num_undirected_edges']}",
        f"score before:  {r['anomaly_score_before']:.3f}",
        f"score after:   {r['anomaly_score_after']:.3f}",
        f"Δ score:  {r['anomaly_score_after']-r['anomaly_score_before']:+.3f}",
    ]
    for i, line in enumerate(lines):
        w = "bold" if i == 0 else "normal"
        fs = 12.5 if i == 0 else 10.8
        c = col if i == 0 else INK
        ax.text(0.08, 0.90 - i * 0.10, line, transform=ax.transAxes, fontsize=fs, fontweight=w, color=c, family="monospace" if i > 0 else None)
fig.suptitle("Real generated anomalies — 6 examples, each traceable to an exact record", fontsize=15, y=1.02)
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase2/generated_anomalies.json (7,200 real records) -- exact subject/seed/severity/method shown per card")
savefig(fig, os.path.join(OUT, "anomaly_examples.png"))
print("done: 07_anomaly_examples")
