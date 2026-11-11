"""Nine-box black-and-white end-to-end flowchart, headers only, Times New Roman. Thick border = this work's main contribution. Run from paper/."""
import subprocess, os
B = [("DATA PREPARATION\\\\AND PATIENT SPLIT", "ABIDE I, 871 patient scans, 360 ROIs (Glasser atlas); stratified patient-level split 609 / 131 / 131", False),
     ("SLIDING-WINDOW\\\\BRAIN GRAPH BUILDING", "30-TR windows, stride 10; Pearson FC; signed graph, top 20\\% edges by $|r|$", True),
     ("BRAIN-REGION SUBGRAPH\\\\TRACKING", "20 selected brain regions (ROIs), 1-hop neighbourhood; node set fixed at the first window", True),
     ("GIN ENCODING AND GRU\\\\NEXT-WINDOW PREDICTION", "2-layer weighted GIN, mean pool, $z\\in\\mathbb{R}^{128}$; GRU (hidden 128), MSE + variance loss, controls only", False),
     ("TEMPORAL SUBGRAPH ANOMALY\\\\SCORING FOR ASD DETECTION", "$A_t=\\|z_{t+1}-\\hat z_{t+1}\\|_2$; patient score = max over regions and transitions", True),
     ("CONTROL-BASED DETECTION\\\\THRESHOLD CALIBRATION", "95th percentile of validation-control scores (transition and patient level)", False),
     ("SYNTHETIC SUBGRAPH\\\\ANOMALY GENERATION", "Gradient-guided PGD edge-weight edits, 5 steps, up to 15\\% of edge pairs, 3 severities", True),
     ("FEW-SHOT ASD\\\\ADAPTATION", "1 / 5 / 10 ASD patient scans + 30 generated anomalies; logistic regression on 20-D region scores", True),
     ("ANOMALY DETECTION AND\\\\SEVERITY OUTPUT", "Flag if patient score exceeds threshold $\\tau$; ADOS-2 band (low / moderate / high) by logistic regression", False)]
W, H, GX, GY = 7.9, 2.1, 1.2, 1.4
cols = [0, W + GX, 2 * (W + GX)]
pos = [((cols if r % 2 == 0 else cols[::-1])[c], -r * (H + GY)) for r in range(3) for c in range(3)]
L = [r"\documentclass[border=6pt]{standalone}", r"\usepackage{fontspec}\setmainfont{Times New Roman}", r"\usepackage{amsmath,amssymb}", r"\hyphenpenalty=10000 \exhyphenpenalty=10000 \tolerance=3000 \emergencystretch=2em", r"\usepackage{tikz}\usetikzlibrary{arrows.meta}", r"\begin{document}",
     r"\begin{tikzpicture}[font=\normalsize,>=Stealth,arr/.style={->,line width=1.5pt}]"]
for i, (h, det, key) in enumerate(B):
    x, y = pos[i]
    L.append(r"\node[draw,line width=%s,rounded corners=%s,fill=white,text=black,minimum width=%.1fcm,minimum height=%.1fcm,text width=%.1fcm,align=center,inner sep=6pt] (n%d) at (%.2f,%.2f) {{\large\bfseries %s}};" % ("0.9pt", "0.95cm" if i == 8 else "3pt", W, H, W - 0.5, i, x, y, h))
for i in range(8):
    a, b = pos[i], pos[i + 1]
    d = (("east", "west") if b[0] > a[0] else ("west", "east")) if a[1] == b[1] else ("south", "north")
    L.append(r"\draw[arr] (n%d.%s)--(n%d.%s);" % (i, d[0], i + 1, d[1]))
L += [r"\end{tikzpicture}", r"\end{document}"]
open("fig_flow_headers.tex", "w").write("\n".join(L))
subprocess.run(["tectonic", "fig_flow_headers.tex"], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "fig_flow_headers.pdf", "figures/fig_flow_headers"], check=True)
os.replace("fig_flow_headers.pdf", "figures/fig_flow_headers.pdf"); os.remove("fig_flow_headers.tex"); print("ok")
