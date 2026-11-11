"""
04_gru_phase1 -- what the real trained GRU actually predicts, and the real
anomaly score it produces, for real subjects.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from style import *
from common_model import embedding_sequence, gru_predict, anomaly_scores

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"
OUT = os.path.join(ROOT, "Output_to_be_shown", "04_gru_phase1")

SUBJECT_CTL = "UM_1_007"
SUBJECT_ASD = "UM_1_062"
SEED_ROI = 284

metrics = json.load(open(os.path.join(ROOT, "outputs/experiments/phase1/metrics.json")))
tau_transition = metrics["tau_transition"]
tau_subject = metrics["tau_subject"]

z_ctl, _ = embedding_sequence(SUBJECT_CTL, SEED_ROI)
zhat_ctl = gru_predict(z_ctl)
a_ctl = anomaly_scores(z_ctl, zhat_ctl).detach().cpu().numpy()

z_asd, _ = embedding_sequence(SUBJECT_ASD, SEED_ROI)
zhat_asd = gru_predict(z_asd)
a_asd = anomaly_scores(z_asd, zhat_asd).detach().cpu().numpy()

print(f"[verify] {SUBJECT_CTL} seed {SEED_ROI}: T={z_ctl.shape[0]} max A_t={a_ctl.max():.3f}")
print(f"[verify] {SUBJECT_ASD} seed {SEED_ROI}: T={z_asd.shape[0]} max A_t={a_asd.max():.3f}")
print(f"[verify] tau_transition={tau_transition:.4f} tau_subject={tau_subject:.4f} (from outputs/experiments/phase1/metrics.json)")

# ---------------------------------------------------------------------------
# temporal_embedding_sequence.png -- real z1..zT heatmap
# ---------------------------------------------------------------------------
z_np = z_ctl.detach().cpu().numpy()
fig, ax = plt.subplots(figsize=(10.5, 5))
im = ax.imshow(z_np, cmap="RdBu_r", vmin=-np.abs(z_np).max(), vmax=np.abs(z_np).max(), aspect="auto")
ax.set_xlabel("embedding dimension (0–127)")
ax.set_ylabel("time window t (0–{})".format(z_np.shape[0] - 1))
ax.set_title(f"Real temporal embedding sequence z₁...z_T — subject {SUBJECT_CTL}, seed {SEED_ROI}\n"
             f"({z_np.shape[0]} windows × 128 dims) — this is what feeds the GRU", fontsize=12.5)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="value")
tag_badge(fig, illustrative=False)
footer(fig, f"source: real forward pass on subgraph_cache/{SUBJECT_CTL}.npz (seed {SEED_ROI})  |  code: src/models/embedding_sequence.py")
savefig(fig, os.path.join(OUT, "temporal_embedding_sequence.png"))

# ---------------------------------------------------------------------------
# gru_actual_vs_predicted.png -- real actual vs predicted + difference + A_t
# ---------------------------------------------------------------------------
zhat_np = zhat_ctl.detach().cpu().numpy()
actual_next = z_np[1:]        # z_2..z_T
predicted_next = zhat_np[:-1]  # zhat predicted at t for t+1, t=0..T-2
diff = actual_next - predicted_next
vmax = max(np.abs(actual_next).max(), np.abs(predicted_next).max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
titles = [f"ACTUAL  z$_{{t+1}}$", f"PREDICTED  ẑ$_{{t+1}}$", "DIFFERENCE"]
mats = [actual_next, predicted_next, diff]
for ax, mat, title in zip(axes, mats, titles):
    vm = vmax if title != "DIFFERENCE" else np.abs(diff).max()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("embedding dim")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
axes[0].set_ylabel("transition index t")
fig.suptitle(f"Real GRU prediction vs. real actual next embedding — subject {SUBJECT_CTL}, seed {SEED_ROI}", fontsize=13.5, y=1.03)
note = (r"$A_t = \|z_{t+1}-\hat z_{t+1}\|_2$" + f"   —  computed per row of the DIFFERENCE panel; "
        f"max over this subject/seed = {a_ctl.max():.3f}")
fig.text(0.5, -0.05, note, ha="center", fontsize=11)
tag_badge(fig, illustrative=False)
footer(fig, f"source: real forward pass, best.pt on subgraph_cache/{SUBJECT_CTL}.npz  |  code: src/anomaly/scoring.py:transition_scores")
savefig(fig, os.path.join(OUT, "gru_actual_vs_predicted.png"))

# ---------------------------------------------------------------------------
# anomaly_score_over_time.png -- real A_t per transition, two real subjects,
# tau_transition (transition-level) vs tau_subject (subject-level) kept distinct.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8.6), sharex=True)
fig.subplots_adjust(bottom=0.16, hspace=0.35, top=0.90)
for ax, a_t, name, col, label in zip(
    axes, [a_ctl, a_asd], [SUBJECT_CTL, SUBJECT_ASD], [CONTROL, ASD], ["control", "ASD"]
):
    x = np.arange(len(a_t))
    ax.plot(x, a_t, "-o", color=col, ms=4, lw=1.5, label=f"{name} ({label}), seed {SEED_ROI}")
    ax.axhline(tau_transition, color=BAD, ls="--", lw=1.3, label=f"τ_transition = {tau_transition:.3f} (per-window)")
    ax.axhline(tau_subject, color=ACCENT, ls=":", lw=1.3, label=f"τ_subject = {tau_subject:.3f} (per-subject max, NOT for this axis)")
    over = x[a_t > tau_transition]
    if len(over):
        ax.scatter(over, a_t[over], color=BAD, s=55, zorder=5, marker="x")
    ax.set_ylabel("A$_t$ (transition anomaly score)")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.set_title(f"{name} ({label})  —  max A$_t$ = {a_t.max():.3f}", fontsize=11.5, loc="left")
axes[1].set_xlabel("transition index t → t+1")
fig.suptitle(f"Real per-transition anomaly score over time — seed ROI {SEED_ROI}", fontsize=14, y=1.0)
note = ("τ_transition is the line an individual transition is compared to (localization).\n"
        "τ_subject (dotted) is only ever compared against this subject's OWN maximum A_t, once per subject — "
        "shown here for reference, not as a per-transition cutoff.")
fig.text(0.5, 0.06, note, ha="center", va="bottom", fontsize=9, color=MUTED)
tag_badge(fig, illustrative=False)
footer(fig, f"source: real forward pass, best.pt  |  τ values: outputs/experiments/phase1/metrics.json  |  code: src/anomaly/scoring.py")
savefig(fig, os.path.join(OUT, "anomaly_score_over_time.png"))

with open(os.path.join(OUT, "_verification.json"), "w") as f:
    json.dump({
        "subject_control": SUBJECT_CTL, "subject_asd": SUBJECT_ASD, "seed_roi": SEED_ROI,
        "tau_transition": tau_transition, "tau_subject": tau_subject,
        "max_A_t_control": float(a_ctl.max()), "max_A_t_asd": float(a_asd.max()),
        "control_flagged_any_transition": bool((a_ctl > tau_transition).any()),
        "asd_flagged_any_transition": bool((a_asd > tau_transition).any()),
    }, f, indent=2)
print("done: 03_gru_phase1")
