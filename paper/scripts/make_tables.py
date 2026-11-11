"""LaTeX table bodies generated directly from result JSON files (no hand-typed numbers)."""
import json, numpy as np
ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH"; R = f"{ROOT}/paper/results"
ab = json.load(open(f"{R}/ablation.json")); ds = json.load(open(f"{R}/derived_stats.json"))
f3 = json.load(open(f"{ROOT}/outputs/experiments/phase3/final_comparison.json"))
f = lambda x, n=3: f"{x:.{n}f}"
rows = [("Static subgraph (distance to control centroid)", "static"), ("Temporal, no learned dynamics (persistence)", "persist"),
        ("Temporal, GRU prediction error (ours)", "gru")]
L = []
for name, k in rows:
    m = ab["methods"][k]; lo, hi = m["auroc_ci95"]
    L.append(f"{name} & {f(m['auroc'])} [{f(lo)}, {f(hi)}] & {f(m['perm_p_vs_chance'],2)} & {f(m['auprc'])} & {f(m['precision'])} & {f(m['recall'])} & {f(m['f1'])} & {m['n_flagged']} & {f(m['auroc_meanA_aggregation'])} \\\\")
open(f"{ROOT}/paper/tab_scoring_body.tex", "w").write("\n".join(L).rstrip() + "\n")
conds = [("few_shot_only", "Few-shot only"), ("few_shot_plus_structure_aware", "+ structure-aware"), ("few_shot_plus_random", "+ random")]
L = []
for s in ["1-shot", "5-shot", "10-shot"]:
    for i, (c, nm) in enumerate(conds):
        d = f3[s][c]; key = f"{s}/{c}"; ch = ds["phase3_chance_f1"][key]
        pm = lambda met: f"{f(d[met+'_mean'])}$\\pm${f(d[met+'_std'],2)}"
        flag = f"{f(ch['mean_flag_rate'],2)}" if ch["mean_flag_rate"] is not None else "n/a"
        chf = f"{f(ch['mean_chance_f1_at_flag_rate'],2)}" if ch["mean_chance_f1_at_flag_rate"] is not None else "n/a"
        lead = f"\\multirow{{3}}{{*}}{{{s.split('-')[0]}}}" if i == 0 else ""
        L.append(f"{lead} & {nm} & {pm('precision')} & {pm('recall')} & {pm('f1')} & {pm('auroc')} & {pm('auprc')} & {flag} & {chf} \\\\")
    L.append("\\midrule" if s != "10-shot" else "\\bottomrule")
open(f"{ROOT}/paper/tab_fewshot_body.tex", "w").write("\n".join(L).rstrip() + "\n")
print(open(f"{ROOT}/paper/tab_scoring_body.tex").read()); print(open(f"{ROOT}/paper/tab_fewshot_body.tex").read())
