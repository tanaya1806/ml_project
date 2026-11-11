"""Nine-box black-and-white end-to-end flowchart, headers only, Times New Roman. Thick border = this work's main contribution. Run from paper/."""
import subprocess, os
B = [("TIME-RESOLVED\\\\BRAIN GRAPHS", "One signed brain network per 30-TR window instead of a single static network", False),
     ("BRAIN-REGION SUBGRAPH\\\\TRACKING", "20 regions with fixed 1-hop node sets followed over time, so an anomaly is localised in region and time", False),
     ("CONTROL-ONLY\\\\TEMPORAL MODEL", "GIN + GRU predict the next window from control scans only; no labelled anomalies needed for training", False),
     ("TEMPORAL SUBGRAPH\\\\ANOMALY SCORING", "Prediction error per region and step; patient score = maximum over regions and steps", False),
     ("TWO-LEVEL THRESHOLD\\\\CALIBRATION", "Step and patient thresholds from validation controls; fixed a failure that flagged all 131 test patients", False),
     ("GRADIENT-GUIDED\\\\ANOMALY GENERATION", "Weight-only edits of real control subgraphs; 45.8\\% detected against 4.8\\% for random edits", False),
     ("GENERATION\\\\VALIDITY CHECKS", "7,200 records, 0 rule violations, severity-monotonic, edit diversity close to 1.0", False),
     ("HONEST FEW-SHOT\\\\EVALUATION", "Chance-matched F1 shows the gain comes from the flag rate, not from ranking (AUROC about 0.5)", False),
     ("SEVERITY ESTIMATE AND\\\\DEMO APPLICATION", "ADOS-based severity bands (no better than a majority guess) and a web app for raw time-series upload", False)]
W, GX, GY = 8.6, 1.2, 1.2
HR = [3.2, 3.2, 3.2]  # estimated row heights, used only for vertical spacing (boxes size themselves to content)
IMG = {}
ys = [0, -(HR[0] / 2 + GY + HR[1] / 2), -(HR[0] / 2 + GY + HR[1] + GY + HR[2] / 2)]
cols = [0, W + GX, 2 * (W + GX)]
pos = [((cols if r % 2 == 0 else cols[::-1])[c], ys[r]) for r in range(3) for c in range(3)]
L = [r"\documentclass[border=6pt]{standalone}", r"\usepackage{fontspec}\setmainfont{Times New Roman}", r"\usepackage{amsmath,amssymb,graphicx}", r"\hyphenpenalty=10000 \exhyphenpenalty=10000 \tolerance=3000 \emergencystretch=2em", r"\usepackage{tikz}\usetikzlibrary{arrows.meta}", r"\begin{document}",
     r"\begin{tikzpicture}[font=\normalsize,>=Stealth,arr/.style={->,line width=1.5pt}]"]
for i, (h, det, key) in enumerate(B):
    x, y = pos[i]
    L.append(r"\node[draw,line width=%s,rounded corners=%s,fill=white,text=black,minimum width=%.1fcm,text width=%.1fcm,align=center,inner sep=6pt] (n%d) at (%.2f,%.2f) {{\large\bfseries %s}\\[4pt]{\normalsize\mdseries %s}};" % ("0.9pt", "3pt", W, W - 0.5, i, x, y, h, det + "".join(["\\\\[5pt]"] + [r"\includegraphics[height=%.1fcm]{figures/%s}\hspace{8pt}" % (4.6 if i == 8 else 3.3, n + (".png" if n.endswith("app") else ".pdf")) for n in IMG.get(i, [])]) if i in IMG else det))
for i in range(8):
    a, b = pos[i], pos[i + 1]
    d = (("east", "west") if b[0] > a[0] else ("west", "east")) if a[1] == b[1] else ("south", "north")
    L.append(r"\draw[arr] (n%d.%s)--(n%d.%s);" % (i, d[0], i + 1, d[1]))
L += [r"\end{tikzpicture}", r"\end{document}"]
open("fig_flow_contrib.tex", "w").write("\n".join(L))
subprocess.run(["tectonic", "fig_flow_contrib.tex"], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "fig_flow_contrib.pdf", "figures/fig_flow_contrib"], check=True)
os.replace("fig_flow_contrib.pdf", "figures/fig_flow_contrib.pdf"); os.remove("fig_flow_contrib.tex"); print("ok")
