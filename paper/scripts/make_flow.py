"""Builds 3 versions of the single end-to-end serpentine pipeline diagram (fig_flow_{function,data,mono}.pdf/png).
One continuous path, solid arrows only, no numbering, no phases. Run from paper/."""
import subprocess, os
# (header, line, fill-key-function, fill-key-data)
S = [
 ("ABIDE I rs-fMRI", "871 resting-state scans", "pill", "data_none", "pill"),
 ("DATA INTAKE", "QC-passed scans, 360 ROIs", "fdata", "none", ""),
 ("SUBJECT SPLIT", "train / validation / test", "fdata", "none", ""),
 ("SLIDING WINDOWS", "30 TR window, stride 10", "fgraph", "none", ""),
 ("GRAPH BUILD", "signed correlation, top 20\\% edges", "fgraph", "none", ""),
 ("SUBGRAPH TRACKING", "20 seed ROIs, 1-hop", "fgraph", "none", ""),
 ("GRAPH ENCODING", "GIN, 128-D embedding", "fmodel", "tc", ""),
 ("TEMPORAL PREDICTION", "GRU, trained on controls", "fmodel", "tc", ""),
 ("ANOMALY SCORING", "next-embedding error", "fscore", "none", ""),
 ("THRESHOLD CALIBRATION", "95th percentile, validation controls", "fscore", "vc", ""),
 ("ANOMALY GENERATION", "gradient-guided edge edits", "fgen", "tc", ""),
 ("GENERATION CHECKS", "range tests, random baseline", "fgen", "none", ""),
 ("FEW-SHOT ADAPTATION", "ASD scans + generated anomalies", "fgen", "asd", ""),
 ("DETECTION", "subject score vs. threshold", "fscore", "test", ""),
 ("SEVERITY ESTIMATION", "ADOS-based bands", "fsev", "sev", ""),
 ("OUTPUT", "flag + severity band", "pill", "test", ""),
]
COLS_F = dict(fdata="gray!18", fgraph="blue!12", fmodel="violet!14", fscore="orange!18", fgen="green!14", fsev="cyan!16", pill="white")
COLS_D = dict(none="white", tc="blue!12", vc="green!15", asd="violet!14", test="orange!18", sev="cyan!16", data_none="white")
LEG_F = [("gray!18", "data preparation"), ("blue!12", "graph construction"), ("violet!14", "learned model"), ("orange!18", "scoring and decision"), ("green!14", "anomaly generation, few-shot"), ("cyan!16", "severity")]
LEG_D = [("blue!12", "uses train-control scans"), ("green!15", "uses validation controls only"), ("violet!14", "uses ASD support scans"), ("orange!18", "uses test scans (final output only)"), ("cyan!16", "uses severity labels"), ("white", "no labels used")]
W, GAP = 3.7, 1.0
xs = [i * (W + GAP) for i in range(4)]
ys = [0, -2.5, -5.0, -7.5]
pos = []
for r in range(4):
    cols = xs if r % 2 == 0 else xs[::-1]
    for c in range(4): pos.append((cols[c], ys[r]))

def tex(mode):
    L = [r"\documentclass[border=5pt]{standalone}", r"\usepackage{fontspec}\setmainfont{Times New Roman}", r"\usepackage{tikz}\usetikzlibrary{arrows.meta}",
         r"\begin{document}", r"\begin{tikzpicture}[font=\small,>=Stealth,box/.style={draw,rounded corners=3pt,minimum width=%.2fcm,minimum height=1.35cm,align=center,inner sep=3pt,line width=0.7pt},pill/.style={draw,rounded corners=0.67cm,minimum width=%.2fcm,minimum height=1.35cm,align=center,inner sep=3pt,line width=1.2pt},arr/.style={->,line width=1.1pt}]" % (W, W)]
    for i, (h, ln, kf, kd, _) in enumerate(S):
        x, y = pos[i]
        sty = "pill" if kf == "pill" else "box"
        fill = "white" if mode == "mono" else (COLS_F[kf] if mode == "function" else COLS_D[kd])
        if mode == "function" and kf == "pill": fill = "white"
        L.append(r"\node[%s,fill=%s] (n%d) at (%.2f,%.2f) {\textbf{%s}\\%s};" % (sty, fill, i, x, y, h, ln))
    for i in range(15):
        a, b = pos[i], pos[i + 1]
        if a[1] == b[1]:
            d = ("east", "west") if b[0] > a[0] else ("west", "east")
        else:
            d = ("south", "north")
        L.append(r"\draw[arr] (n%d.%s)--(n%d.%s);" % (i, d[0], i + 1, d[1]))
    leg = LEG_F if mode == "function" else (LEG_D if mode == "data" else [])
    for k, (c, t) in enumerate(leg):
        lx = (k % 3) * 6.2 - 1.8; ly = -9.55 - (k // 3) * 0.65
        L.append(r"\node[draw,fill=%s,minimum size=0.3cm,inner sep=0] at (%.2f,%.2f) {}; \node[anchor=west] at (%.2f,%.2f) {%s};" % (c, lx, ly, lx + 0.25, ly, t))
    L += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L)

for mode in ("function", "data", "mono"):
    base = f"fig_flow_{mode}"; open(base + ".tex", "w").write(tex(mode))
    subprocess.run(["tectonic", base + ".tex"], check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", base + ".pdf", "figures/" + base], check=True)
    os.replace(base + ".pdf", "figures/" + base + ".pdf"); os.remove(base + ".tex")
print("ok")
