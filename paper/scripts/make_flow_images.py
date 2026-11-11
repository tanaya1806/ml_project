"""Real example images for the flowchart. Run from repo root with the web app running on :8800.
 flow_in_graph_ctl/asd.pdf : brain-region subgraph (nodes = ROIs, edges = strongest correlations, first window) of a control and an ASD test scan
 flow_out_synth.pdf        : a real control subgraph and its synthetic anomaly (gradient-guided edge edits; edited edges drawn heavy)
 flow_out_app.png          : screenshot of the web app result for sample_18 (flagged)"""
import os, subprocess, sys
import numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from src.inference import engine as E
from src.generation.structure_aware_generator import generate_anomaly

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 10})
OUT, SEED = "paper/figures", 47
load = lambda f: np.loadtxt(f"testing_data/samples/{f}", delimiter=",")
caches = {n: E.cache_from_timeseries(load(f), n) for n, f in [("ctl", "sample_03.csv"), ("asd", "sample_18.csv")]}

def graph(cache, t=0):
    roi = cache[f"roi_indices_{SEED}"]; ei = cache[f"edge_index_{SEED}"][t]; ew = cache[f"edge_weight_{SEED}"][t]
    G = nx.Graph(); G.add_nodes_from(range(len(roi)))
    for (u, v), w in zip(ei.T, ew):
        if u < v: G.add_edge(int(u), int(v), w=float(w))
    return G

def draw(G, pos, ax, changed=None, title=None):
    pos_e = [(u, v) for u, v, d in G.edges(data=True) if d["w"] >= 0]; neg_e = [(u, v) for u, v, d in G.edges(data=True) if d["w"] < 0]
    nx.draw_networkx_edges(G, pos, edgelist=pos_e, ax=ax, width=0.35, edge_color="0.45", alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=neg_e, ax=ax, width=0.35, edge_color="0.45", style="dashed", alpha=0.8)
    if changed: nx.draw_networkx_edges(G, pos, edgelist=changed, ax=ax, width=1.6, edge_color="black")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=9, node_color="black", linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=[0], ax=ax, node_size=45, node_color="white", edgecolors="black", linewidths=1.2)
    ax.set_axis_off()
    if title: ax.set_title(title, fontsize=10, pad=2)

for name, label in [("ctl", "control patient"), ("asd", "ASD patient")]:
    G = graph(caches[name]); pos = nx.spring_layout(G, seed=3, k=0.3, iterations=200)
    f, a = plt.subplots(figsize=(1.6, 1.6), constrained_layout=True); draw(G, pos, a, title=label); f.savefig(f"{OUT}/flow_in_graph_{name}.pdf"); plt.close(f)

gin, gru, _, device, _ = E.get_models()
rec = generate_anomaly(gin, gru, caches["ctl"], SEED, 3, 0.9, device)
G0 = graph(caches["ctl"], 4); roi = caches["ctl"][f"roi_indices_{SEED}"]; idx = {int(r): i for i, r in enumerate(roi)}
changed = [(idx[e["roi_u"]], idx[e["roi_v"]]) for e in rec["edges_changed"]]
pos = nx.spring_layout(G0, seed=3, k=0.3, iterations=200)
f, ax = plt.subplots(2, 1, figsize=(1.75, 3.2), constrained_layout=True)
draw(G0, pos, ax[0], title="real control"); draw(G0, pos, ax[1], changed=changed, title="synthetic anomaly")
f.savefig(f"{OUT}/flow_out_synth.pdf"); plt.close(f)
print("edited pairs:", len(changed), "score before/after:", round(rec["anomaly_score_before"], 2), round(rec["anomaly_score_after"], 2))

CH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1100,1500", "--virtual-time-budget=25000", "--screenshot=/tmp/_app.png", "http://localhost:8800/#demo=sample_18.csv"], capture_output=True)
from PIL import Image
Image.open("/tmp/_app.png").crop((590, 190, 1060, 742)).save(f"{OUT}/flow_out_app.png")
