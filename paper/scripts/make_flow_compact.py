"""Nine-box black-and-white end-to-end flowchart, headers only, Times New Roman. Thick border = this work's main contribution. Run from paper/."""
import subprocess, os
B = [("DATA PREPARATION\\\\AND PATIENT SPLIT", "ABIDE I, 871 patient scans", False),
     ("SLIDING-WINDOW\\\\BRAIN GRAPH BUILDING", "30-TR windows, stride 10; Pearson correlation; signed graph, top 20\\% edges", True),
     ("BRAIN-REGION SUBGRAPH\\\\TRACKING", "20 selected brain regions (ROIs), 1-hop neighbourhood; node set fixed at the first window", True),
     ("GIN ENCODING AND GRU\\\\NEXT-WINDOW PREDICTION", "2-layer weighted GIN, mean pool, $z\\in\\mathbb{R}^{128}$; GRU (hidden 128), MSE + variance loss, controls only", False),
     ("TEMPORAL SUBGRAPH ANOMALY\\\\SCORING FOR ASD DETECTION", "patient score = max over regions and transitions", True),
     ("CONTROL-BASED DETECTION\\\\THRESHOLD CALIBRATION", "95th percentile of validation-control scores (transition and patient level)", False),
     ("SYNTHETIC SUBGRAPH\\\\ANOMALY GENERATION", "Gradient-guided PGD edge-weight edits, 5 steps, up to 15\\% of edge pairs, 3 severities", True),
     ("FEW-SHOT ASD\\\\ADAPTATION", "1 / 5 / 10 ASD patient scans + 30 generated anomalies; logistic regression on 20-D region scores", True),
     ("ANOMALY DETECTION AND\\\\SEVERITY OUTPUT", "Flag if patient score exceeds threshold $\\tau$; ADOS-2 band (low / moderate / high) by logistic regression", False)]
W, GX, GY = 8.6, 1.2, 1.2
HR = [4.5, 2.2, 6.2]  # estimated row heights, used only for vertical spacing (boxes size themselves to content)
IMG = {0: ["flow_in_graph_ctl", "flow_in_graph_asd"], 8: ["flow_out_app", "flow_out_synth"]}
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
open("fig_flow_compact.tex", "w").write("\n".join(L))
subprocess.run(["tectonic", "fig_flow_compact.tex"], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "fig_flow_compact.pdf", "figures/fig_flow_compact"], check=True)
os.replace("fig_flow_compact.pdf", "figures/fig_flow_compact.pdf"); os.remove("fig_flow_compact.tex"); print("ok")
