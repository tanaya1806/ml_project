"""Column-width (about 8.8 cm) portrait version of the nine-step flowchart, same wording and real example images. Run from paper/."""
import subprocess, os
BW, GAP, VG = 4.15, 0.75, 0.62  # box width, horizontal gap, vertical gap (cm)
H = r"\fontsize{6.8}{7.8}\selectfont\bfseries"; D = r"\fontsize{6}{7}\selectfont\mdseries"
def box(name, pos, head, det, extra="", w=BW, style=""):
    return r"\node[bx,%s,text width=%.2fcm,minimum width=%.2fcm] (%s) %s {{%s %s}\\[1.5pt]{%s %s}%s};" % (style, w - 0.25, w, name, pos, H, head, D, det, extra)
img = lambda f, h: r"\\[2pt]\includegraphics[height=%.2fcm]{%s}" % (h, f)
L = [r"\documentclass[border=3pt]{standalone}", r"\usepackage{fontspec}\setmainfont{Times New Roman}", r"\usepackage{amsmath,amssymb,graphicx}",
     r"\hyphenpenalty=10000 \exhyphenpenalty=10000 \tolerance=3000 \emergencystretch=2em", r"\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning}", r"\begin{document}",
     r"\begin{tikzpicture}[>=Stealth,bx/.style={font=\fontsize{6.2}{7.3}\selectfont,draw,line width=0.45pt,rounded corners=2pt,fill=white,align=center,inner sep=3pt},arr/.style={->,line width=0.7pt}]"]
gi = lambda f: r"\includegraphics[height=1.55cm]{figures/%s.pdf}" % f
L.append(box("n1", "at (0,0)", "DATA PREPARATION\\\\AND PATIENT SPLIT", "ABIDE I, 871 patient scans", img("figures/flow_in_graph_ctl.pdf", 1.7) + r"\hspace{3pt}" + r"\includegraphics[height=1.7cm]{figures/flow_in_graph_asd.pdf}"))
L.append(box("n2", "[right=%.2fcm of n1]" % GAP, "SLIDING-WINDOW\\\\BRAIN GRAPH BUILDING", "30-TR windows, stride 10; Pearson correlation; signed graph, top 20\\% edges"))
L.append(box("n3", "[anchor=north] at ([yshift=-%.2fcm]n2.center |- n1.south)" % VG, "BRAIN-REGION SUBGRAPH\\\\TRACKING", "20 selected brain regions (ROIs), 1-hop neighbourhood; node set fixed at the first window"))
L.append(box("n4", "[left=%.2fcm of n3]" % GAP, "GIN ENCODING AND GRU\\\\NEXT-WINDOW PREDICTION", "2-layer weighted GIN, mean pool, $z\\in\\mathbb{R}^{128}$; GRU (hidden 128), MSE + variance loss, controls only"))
L.append(box("n5", "[below=%.2fcm of n4]" % VG, "TEMPORAL SUBGRAPH ANOMALY\\\\SCORING FOR ASD DETECTION", "patient score = max over regions and transitions"))
L.append(box("n6", "[right=%.2fcm of n5]" % GAP, "CONTROL-BASED DETECTION\\\\THRESHOLD CALIBRATION", "95th percentile of validation-control scores (transition and patient level)"))
L.append(box("n7", "[below=%.2fcm of n6]" % VG, "SYNTHETIC SUBGRAPH\\\\ANOMALY GENERATION", "Gradient-guided PGD edge-weight edits, 5 steps, up to 15\\% of edge pairs, 3 severities"))
L.append(box("n8", "[left=%.2fcm of n7]" % GAP, "FEW-SHOT ASD\\\\ADAPTATION", "1 / 5 / 10 ASD patient scans + 30 generated anomalies; logistic regression on 20-D region scores"))
L.append(box("n9", "[below=%.2fcm of n8.south east, anchor=north]" % VG, "ANOMALY DETECTION AND SEVERITY OUTPUT", "Flag if patient score exceeds threshold $\\tau$; ADOS-2 band (low / moderate / high) by logistic regression",
             img("figures/flow_out_app.png", 2.5) + r"\hspace{6pt}" + r"\includegraphics[height=2.5cm]{figures/flow_out_synth.pdf}", w=2 * BW + GAP))
for a, b, d in [("n1", "n2", "east,west"), ("n2", "n3", "south,north"), ("n3", "n4", "west,east"), ("n4", "n5", "south,north"), ("n5", "n6", "east,west"), ("n6", "n7", "south,north"), ("n7", "n8", "west,east")]:
    p, q = d.split(","); L.append(r"\draw[arr] (%s.%s)--(%s.%s);" % (a, p, b, q))
L.append(r"\draw[arr] (n8.south)--(n8.south |- n9.north);")
L += [r"\end{tikzpicture}", r"\end{document}"]
open("fig_flow_col.tex", "w").write("\n".join(L))
subprocess.run(["tectonic", "fig_flow_col.tex"], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-r", "400", "-png", "-singlefile", "fig_flow_col.pdf", "docx_build/assets/fig_flow_col"], check=True)
os.remove("fig_flow_col.pdf"); os.remove("fig_flow_col.tex"); print("ok")
