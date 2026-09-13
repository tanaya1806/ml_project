"""Phase 3 final-comparison figures: F1/AUROC across shot counts and conditions."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

COND_LABELS = {
    "few_shot_only": "few-shot only",
    "few_shot_plus_structure_aware": "few-shot + structure-aware",
    "few_shot_plus_random": "few-shot + random",
}


def make_figures(summary: dict, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    shots = sorted(summary.keys(), key=lambda k: int(k.split("-")[0]))
    shot_x = [int(k.split("-")[0]) for k in shots]

    for metric in ["f1", "auroc"]:
        plt.figure(figsize=(6.5, 4.5))
        for cond, label in COND_LABELS.items():
            means = [summary[s][cond][f"{metric}_mean"] for s in shots]
            stds = [summary[s][cond][f"{metric}_std"] for s in shots]
            plt.errorbar(shot_x, means, yerr=stds, marker="o", capsize=4, label=label)
        plt.xlabel("Shot count (real ASD support examples)")
        plt.ylabel(metric.upper())
        plt.title(f"Phase 3: {metric.upper()} vs shot count")
        plt.xscale("log")
        plt.xticks(shot_x, [str(x) for x in shot_x])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"0{1 if metric=='f1' else 2}_{metric}_vs_shots.png"), dpi=150)
        plt.close()


def main():
    base_cfg = yaml.safe_load(open("configs/baseline.yaml"))
    p3_cfg = yaml.safe_load(open("configs/phase3.yaml"))
    summary = json.load(open(os.path.join(p3_cfg["output_dir"], "final_comparison.json")))
    fig_dir = os.path.join(base_cfg["dataset"]["figures_dir"], "phase3")
    make_figures(summary, fig_dir)
    print(f"[*] Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
