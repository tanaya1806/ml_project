"""
Shared presentation style for every figure in Output_to_be_shown/.
Light background, consistent palette, consistent typography -- reused across
every script in this folder so the whole deliverable reads as one system.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Palette (kept consistent with the rest of this project's deliverables)
CONTROL = "#0E9488"      # teal  -- typical control / normal
ASD = "#C1691B"          # amber/brown -- ASD / real abnormal
ANOMALY = "#8B3FD1"      # violet -- Phase 2 generated / synthetic anomaly
RANDOM = "#8A94A6"       # grey -- random-perturbation baseline
ACCENT = "#2E5FD9"       # blue -- structural / neutral accent
BAD = "#C43D3D"          # red -- flags, violations, negative
GOOD = "#2E6B4F"          # green -- positive / validated
INK = "#141A26"
MUTED = "#5B6478"
LINE = "#D8DEEA"
BG = "#FFFFFF"
PANEL = "#F5F7FB"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

REAL_TAG = dict(text="REAL DATA · REAL COMPUTATION", color=GOOD)
ILLUS_TAG = dict(text="ILLUSTRATIVE SCHEMATIC — NOT EXPERIMENTAL DATA", color=BAD)


def tag_badge(ax_or_fig, illustrative=False):
    """Stamp every figure with an unambiguous REAL vs ILLUSTRATIVE provenance badge (top-right,
    clear of the bottom-left source footer)."""
    tag = ILLUS_TAG if illustrative else REAL_TAG
    fig = ax_or_fig if hasattr(ax_or_fig, "text") and not hasattr(ax_or_fig, "transAxes") else ax_or_fig.figure
    fig.text(
        0.995, 0.995, tag["text"],
        ha="right", va="top", fontsize=8.5, fontweight="bold",
        color="white", bbox=dict(facecolor=tag["color"], edgecolor="none", boxstyle="round,pad=0.35,rounding_size=0.2"),
    )


def footer(fig, source_line):
    fig.text(0.01, 0.005, source_line, ha="left", va="bottom", fontsize=7.5, color=MUTED, family="monospace")


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)
