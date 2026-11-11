"""Builds tables_figures_explained.tex: every table/figure of main.tex plus a short plain-language note. Run from paper/."""
import re
s = open('main.tex').read()
order = {l.split()[1]: l.split()[0].strip('[]') for l in open('refs_order.txt') if l.strip()}
def cite(m): return "{[" + ", ".join(order[k.strip()] for k in m.group(1).split(",")) + "]}"
def env(label):
    i = s.index("\\label{%s}" % label)
    a = max(s.rfind("\\begin{table*}", 0, i), s.rfind("\\begin{table}", 0, i), s.rfind("\\begin{figure*}", 0, i), s.rfind("\\begin{figure}", 0, i))
    m = re.compile(r"\\end\{(table\*?|figure\*?)\}").search(s, i)
    b = s[a:m.end()]
    b = re.sub(r"\\begin\{(table|figure)\*\}", r"\\begin{\1}", b); b = re.sub(r"\\end\{(table|figure)\*\}", r"\\end{\1}", b)
    b = b.replace("[!t]", "[H]")
    b = re.sub(r"\\cite\{([^}]*)\}", cite, b)
    b = b.replace("Sec.~\\ref{sec:results}", "Sec.~VII")
    if label in ("tab:lit", "tab:scoring", "tab:fewshot"):
        b = b.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
    if label == "fig:pipeline":
        b = b.replace("\\begin{tikzpicture}", "\\resizebox{\\linewidth}{!}{\\begin{tikzpicture}", 1).replace("\\end{tikzpicture}", "\\end{tikzpicture}}", 1)
    return b
N = {  # label: (heading, explanation)
"tab:lit": ("Comparison with closest prior work", "Each row is a published study closest to ours; the columns say what it studied, which method it used, and whether it models time (Temporal), looks below whole-graph level (Loc.), or uses few labelled examples (Few-shot). A tick means addressed, a circle partial, a dash not addressed as we read it. The table shows that each ingredient exists separately but that none of the listed works combines all three, which is the basis of the paper's cautious gap statement (``to our knowledge'')."),
"fig:pipeline": ("End-to-end pipeline", "The whole system as nine steps in one continuous flow, read left to right on rows 1 and 3 and right to left on row 2, with the key technical settings inside each box: 30-TR windows with top-20% signed graphs, 20 selected brain regions tracked as subgraphs, a 2-layer GIN with a GRU, the prediction-error score, a 95th-percentile threshold from control scans, gradient-guided synthetic edits, few-shot logistic regression and ADOS-based severity bands. Test scans are used only at the final detection step."),
"tab:data": ("Dataset and preprocessing summary", "Lists the facts about the data the experiments rely on: the ABIDE I cohort (871 subjects, 403 ASD and 468 controls, 20 sites), 360 regions of interest, scan lengths, the 609/131/131 subject-level split, window and stride settings, how connectivity edges are kept, and the size of the subgraphs. It also states plainly that the brain atlas and preprocessing pipeline are not recorded in the project files."),
"tab:config": ("Experimental configuration", "A reference sheet of every setting needed to reproduce the runs: the two-layer graph encoder (GIN), the recurrent predictor (GRU), training settings (Adam, 5 epochs), how the thresholds are set, the anomaly generator's parameters, the few-shot classifier, and the hardware and software used. Hyperparameters were fixed in advance, not searched."),
"fig:training": ("Training curves", "Shows the prediction error on training controls (blue) and on held-out validation controls (red) over the five training epochs, plus the variance penalty (grey dashed) that stops the embeddings from collapsing to a constant. The validation loss is lowest at epoch 4 (0.113), which is the checkpoint used. It confirms training was stable, not that the model detects ASD."),
"fig:phase1": ("Detector test results", "Panel (a) plots the subject-level anomaly scores of validation controls, test controls and test ASD subjects, with the decision threshold at 19.81; the three distributions overlap almost completely. Panel (b) is the ROC curve on the 131 test subjects with a 95\\% bootstrap band; it hugs the chance diagonal (AUROC 0.453, CI 0.353--0.552). Together they show that the detector does not separate ASD from control subjects."),
"fig:timeline": ("Transition scores over time for two subjects", "Each point is the anomaly score for one window-to-window transition of a single seed region (ROI 284) for one control subject (top) and one ASD subject (bottom). The red dashed line is the transition threshold (7.43) and crosses mark the transitions above it. It illustrates how flagged moments (the localization records) arise; the two subjects are examples only and say nothing about group differences."),
"tab:scoring": ("Scoring-function comparison on the test set", "Compares three ways of scoring the same frozen encoder's output: static distance from the control average, persistence (change from the previous window) and the GRU's prediction error. For each: AUROC with a 95\\% bootstrap interval and permutation p-value, AUPRC, precision, recall, F1 and number of subjects flagged. All three sit near chance (AUROC 0.45--0.49, intervals overlapping), so temporal modelling gives no measurable benefit here."),
"fig:phase2": ("Anomaly generation", "(a) The model's anomaly score after editing connection weights, against edit severity: guided (structure-aware) edits raise the score steadily (about 4.0 to 7.6) while random edits do not. (b) Share of transitions above the threshold: 45.8\\% for guided high-severity edits versus 4.8\\% for random edits and about 5\\% for real data. (c) For one example, the edges ranked by gradient sensitivity, with the 74 of 548 edge pairs chosen for editing. The effect is expected by construction because the same model guides and scores the edits."),
"tab:fewshot": ("Few-shot comparison on the test set", "Results for 1, 5 and 10 real ASD examples under three conditions: few-shot only, plus 30 structure-aware generated anomalies, plus 30 random ones (mean and standard deviation over five draws). Precision, recall, F1, AUROC and AUPRC are shown, together with the fraction of subjects flagged and the F1 a random classifier would get at that flag rate. Adding generated anomalies raises F1, but AUROC stays near 0.50 and F1 is close to the chance level."),
"fig:phase3": ("Few-shot results", "The same few-shot results as a graph: F1 (a), recall (b) and AUROC (c) against the number of real ASD examples, for the three conditions with error bars. F1 and recall jump when generated anomalies are added, but AUROC stays on the dashed chance line (0.5) in every condition. This indicates the classifier flags more subjects rather than ranking ASD subjects better."),
"tab:severity": ("Severity-band estimation (exploratory)", "Tests whether the detector's scores can estimate ADOS-2 based severity bands (low, moderate, high) for ASD subjects. Accuracy and macro-F1 are given for a held-out set (60 subjects) and for repeated cross-validation (193 subjects), next to the accuracy of always guessing the most common band. The model is below that simple rule, so the scores carry no usable severity information here."),
"tab:ablation": ("Component ablation and coverage", "Summarises which parts of the method were actually tested against an alternative and what came out: temporal versus static scoring, generated anomalies, structure-aware versus random, and recurrence were evaluated; subgraph versus whole-graph scoring, alternative encoders, the variance term and a support-weighted score were not. It keeps the paper honest about what is, and is not, supported by evidence."),
"tab:cost": ("Computational cost", "Measured run times on an Apple-silicon laptop: building the subgraph cache (112 s), training (299.7 s), the detector evaluation (27 s), generating 7,200 anomaly records (79 s), the 45 few-shot fits (86 s), and the cost of extracting all 360 seed subgraphs per subject. It shows the whole pipeline is cheap enough to run on a laptop."),
}
def _num():
    order = sorted(N, key=lambda k: s.index("\\label{%s}" % k)); tc = fc = 0; out = {}
    rom = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
    for k in order:
        if k.startswith("tab"): out[k] = "Table " + rom[tc]; tc += 1
        else: fc += 1; out[k] = "Figure %d" % fc
    return out
_L = _num(); lab = lambda k: _L[k]
seq = ["tab:lit", "fig:pipeline", "tab:data", "tab:config", "fig:training", "fig:phase1", "fig:timeline", "tab:scoring", "fig:phase2", "tab:fewshot", "fig:phase3", "tab:severity", "tab:ablation", "tab:cost"]
pre = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.9cm]{geometry}
\usepackage{fontspec}\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb,graphicx,booktabs,multirow,array,xcolor,float,caption,titlesec}
\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
\newcommand{\ok}{\ensuremath{\checkmark}}\renewcommand{\part}{\ensuremath{\circ}}\newcommand{\no}{--}
\renewcommand{\thetable}{\Roman{table}}\renewcommand{\figurename}{Fig.}
\captionsetup{font=small,labelfont=bf}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\begin{document}
\begin{center}{\Large\bfseries Tables and Figures Explained}\\[3pt]
{\normalsize Few-Shot Temporal Subgraph Anomaly Detection and Structure-Aware Anomaly Generation in Functional Brain Networks}\end{center}
Every table and figure in the paper is reproduced below, each with a short note on what it depicts. All numbers come from the project's committed outputs and scripts.
\bigskip
"""
out = pre
for k in seq:
    h, d = N[k]
    out += "\\section*{%s}\n%s\n\n\\smallskip\n%s\n\n\\bigskip\n" % (lab(k) + ": " + h, env(k), d)
    if k in ("fig:pipeline", "tab:config", "fig:timeline", "tab:fewshot"): out += "\\clearpage\n"
out += "\\end{document}\n"
open('tables_figures_explained.tex', 'w').write(out)
