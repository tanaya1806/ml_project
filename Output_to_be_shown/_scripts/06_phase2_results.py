"""
07_phase2_results -- real, aggregate Phase 2 validation results, taken directly
from the project's own committed outputs/experiments/phase2/generator_validation.json
(7,200 real generated records; no recomputation needed since this file already
IS the real experiment output).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from style import *

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "07_phase2_results")

V = json.load(open(os.path.join(ROOT, "outputs/experiments/phase2/generator_validation.json")))
sev = V["severity_control"]["per_group"]
det = V["detectability"]

# ---------------------------------------------------------------------------
# severity_vs_anomaly_score.png
# ---------------------------------------------------------------------------
levels = ["low", "medium", "high"]
alphas = [0.25, 0.55, 0.9]
sa_mean = [sev[f"structure_aware/{l}"]["mean"] for l in levels]
sa_std = [sev[f"structure_aware/{l}"]["std"] for l in levels]
rd_mean = [sev[f"random/{l}"]["mean"] for l in levels]
rd_std = [sev[f"random/{l}"]["std"] for l in levels]

fig, ax = plt.subplots(figsize=(8, 6.4))
ax.errorbar(alphas, sa_mean, yerr=sa_std, marker="o", color=CONTROL, lw=2.2, capsize=5, label="structure-aware")
ax.errorbar(alphas, rd_mean, yerr=rd_std, marker="o", color=RANDOM, lw=2.2, capsize=5, label="random baseline")
ax.axhline(V["tau_transition"], color=BAD, ls="--", lw=1.4, label=f"τ_transition = {V['tau_transition']:.3f}")
ax.set_xlabel("severity (α)"); ax.set_ylabel("anomaly score after generation (mean ± std)")
ax.set_title("Phase 2 — Real Severity vs. Anomaly Score\n(7,200 generated records, both methods)", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
mono_sa = V["severity_control"]["monotonic_in_severity"]["structure_aware"]
mono_rd = V["severity_control"]["monotonic_in_severity"]["random"]
ax.text(0.98, 0.03, f"monotonic in severity: structure-aware={mono_sa}, random={mono_rd}",
        transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase2/generator_validation.json  |  code: src/evaluate/validate_generator.py")
savefig(fig, os.path.join(OUT, "severity_vs_anomaly_score.png"))

# ---------------------------------------------------------------------------
# structure_aware_vs_random.png
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.4))
x = np.arange(3); w = 0.35
b1 = ax.bar(x - w / 2, sa_mean, w, yerr=sa_std, color=CONTROL, label="structure-aware", capsize=4)
b2 = ax.bar(x + w / 2, rd_mean, w, yerr=rd_std, color=RANDOM, label="random", capsize=4)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(α={a})" for l, a in zip(levels, alphas)])
ax.set_ylabel("anomaly score after generation (mean ± std)")
ax.set_title("Phase 2 — Structure-Aware vs. Random Perturbation\n(real, same edit budget both methods)", fontsize=13)
ax.legend(fontsize=10)
for bars, means in [(b1, sa_mean), (b2, rd_mean)]:
    for rect, m in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.15, f"{m:.2f}",
                ha="center", fontsize=9.5)
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase2/generator_validation.json")
savefig(fig, os.path.join(OUT, "structure_aware_vs_random.png"))

# ---------------------------------------------------------------------------
# detection_rate_comparison.png
# ---------------------------------------------------------------------------
cats = ["real normal\n(val-control)", "real ASD\n(test)", "generated\nrandom, high", "generated\nstruct-aware, low",
        "generated\nstruct-aware, med", "generated\nstruct-aware, high"]
vals = [det["real_normal_val_control"], det["real_asd_test"], det["random/high"],
        det["structure_aware/low"], det["structure_aware/medium"], det["structure_aware/high"]]
colors = [CONTROL, ASD, RANDOM, ANOMALY, ANOMALY, ANOMALY]
fig, ax = plt.subplots(figsize=(10.5, 6.4))
bars = ax.bar(cats, vals, color=colors)
ax.set_ylim(0, max(vals) * 1.28)
for rect, v in zip(bars, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.008, f"{100*v:.1f}%", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylabel(f"fraction of transitions exceeding τ_transition ({V['tau_transition']:.3f})")
ax.set_title("Phase 2 — Real Detection (Threshold-Crossing) Rate", fontsize=13)
ratio = det["structure_aware/high"] / det["random/high"]
ax.text(0.5, 0.97, f"structure-aware/high ÷ random/high = {ratio:.1f}×", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, fontweight="bold", color=ANOMALY)
plt.setp(ax.get_xticklabels(), fontsize=9.5)
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase2/generator_validation.json (detectability)  |  code: src/evaluate/validate_generator.py")
savefig(fig, os.path.join(OUT, "detection_rate_comparison.png"))

# ---------------------------------------------------------------------------
# phase2_metrics.png -- summary panel
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.8))
ax.axis("off")
rows = [
    ("Generated records", f"{V['validity']['num_records']:,}"),
    ("Constraint violations", str(V["validity"]["num_constraint_violations"])),
    ("Severities", "low (0.25) · medium (0.55) · high (0.9)"),
    ("Methods", "structure-aware (gradient-guided) · random (baseline)"),
    ("Structure-aware monotonic in severity?", str(mono_sa)),
    ("Random monotonic in severity?", str(mono_rd)),
    ("Detect. rate, structure-aware/high", f"{100*det['structure_aware/high']:.1f}%"),
    ("Detect. rate, random/high", f"{100*det['random/high']:.1f}%"),
    ("Ratio (structure-aware ÷ random)", f"{ratio:.1f}×"),
    ("Detect. rate, real normal (val-control)", f"{100*det['real_normal_val_control']:.1f}%"),
    ("Detect. rate, real ASD (test)", f"{100*det['real_asd_test']:.1f}%"),
    ("Diversity (mean Jaccard distance)", ", ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in V["diversity"].items() if "structure_aware" in k)),
]
ax.set_title("Phase 2 — Real Generation-Effectiveness Summary", fontsize=14.5, pad=16)
y = 0.93
for k, v in rows:
    ax.text(0.02, y, k, fontsize=11.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.60, y, v, fontsize=11, family="monospace", transform=ax.transAxes)
    y -= 0.078
tag_badge(fig, illustrative=False)
footer(fig, "source: outputs/experiments/phase2/generator_validation.json  |  code: src/evaluate/validate_generator.py")
savefig(fig, os.path.join(OUT, "phase2_metrics.png"))

with open(os.path.join(OUT, "_verification.json"), "w") as f:
    json.dump({
        "severity_control": V["severity_control"], "detectability": det,
        "diversity": V["diversity"], "validity": V["validity"],
        "tau_transition": V["tau_transition"], "detection_ratio_high": ratio,
        "source_file": "outputs/experiments/phase2/generator_validation.json",
    }, f, indent=2)
print("done: 06_phase2_results")
