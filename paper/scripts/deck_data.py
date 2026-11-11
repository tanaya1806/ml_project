"""Real data for the native (editable) deck figures -> paper/deck/deck_data.json. Run from repo root."""
import json, os, sys
import numpy as np, networkx as nx
sys.path.insert(0, os.getcwd())
from src.inference import engine as E
from src.generation.structure_aware_generator import generate_anomaly
SEED, NN, NE = 47, 34, 70
load = lambda f: np.loadtxt(f"testing_data/samples/{f}", delimiter=",")
caches = {n: E.cache_from_timeseries(load(f), n) for n, f in [("ctl", "sample_03.csv"), ("asd", "sample_18.csv")]}
def graph(cache, t=0, w_override=None):
    ei = cache[f"edge_index_{SEED}"][t]; ew = cache[f"edge_weight_{SEED}"][t] if w_override is None else w_override
    G = nx.Graph(); G.add_nodes_from(range(len(cache[f"roi_indices_{SEED}"])))
    for (u, v), w in zip(ei.T, ew):
        if u < v: G.add_edge(int(u), int(v), w=float(w))
    return G
def pack(G, keep, pos, edges=None, bold=()):
    idx = {n: i for i, n in enumerate(keep)}
    es = edges if edges is not None else sorted([(u, v, d["w"]) for u, v, d in G.edges(data=True) if u in idx and v in idx], key=lambda e: -abs(e[2]))[:NE]
    return {"nodes": [[float(pos[n][0]), float(pos[n][1]), 1 if n == 0 else 0] for n in keep],
            "edges": [[idx[u], idx[v], 1 if w < 0 else 0, 1 if (min(u, v), max(u, v)) in bold else 0] for u, v, w in es if u in idx and v in idx]}
out = {}
# input graphs (first window): same node set rule for both = 0 (seed) + highest-degree nodes
for name in ("ctl", "asd"):
    G = graph(caches[name]); keep = [0] + [n for n, _ in sorted(G.degree, key=lambda x: -x[1]) if n != 0][:NN - 1]
    pos = nx.spring_layout(G.subgraph(keep), seed=3, k=0.6, iterations=200); out[name] = pack(G, keep, pos); out[name]["total_nodes"] = G.number_of_nodes()
# synthetic anomaly before/after on the control scan (same node set)
gin, gru, *_ = E.get_models(); import torch
rec = generate_anomaly(gin, gru, caches["ctl"], SEED, 3, 0.9, E.get_models()[3])
G0 = graph(caches["ctl"], 4); roi = caches["ctl"][f"roi_indices_{SEED}"]; idx = {int(r): i for i, r in enumerate(roi)}
bold = {(min(idx[e["roi_u"]], idx[e["roi_v"]]), max(idx[e["roi_u"]], idx[e["roi_v"]])) for e in rec["edges_changed"]}
from collections import Counter
inc = Counter(n for e in bold for n in e); keep = [0] + [n for n, _ in inc.most_common() if n != 0][:NN - 1]
pos = nx.spring_layout(G0.subgraph(keep), seed=3, k=0.6, iterations=200); ks = set(keep)
allE = sorted([(u, v, d["w"]) for u, v, d in G0.edges(data=True) if u in ks and v in ks], key=lambda e: -abs(e[2]))
ed = [e for e in allE if (min(e[0], e[1]), max(e[0], e[1])) in bold][:45]; rest = [e for e in allE if e not in ed][:25]; es = ed + rest
shown_bold = ed
out["synth_before"] = pack(G0, keep, pos, es); out["synth_after"] = pack(G0, keep, pos, es, bold)
out["synth_info"] = {"edited_pairs": len(bold), "score_before": float(rec["anomaly_score_before"]), "score_after": float(rec["anomaly_score_after"]), "shown_edited": len(shown_bold)}
# example scan outputs
r = E.analyze(load("sample_18.csv"), "sample_18")
out["scan"] = {k: r[k] for k in ("subject_score", "tau_subject", "tau_transition", "top_seed", "top_seed_scores", "seed_max", "n_windows", "n_timepoints", "flagged", "n_flagged_transitions", "n_transitions")}
# detector figure data
RES = "paper/results"; yv = np.load(f"{RES}/val_scores_gru.npy"); yt = np.load(f"{RES}/test_scores_gru.npy"); Lt = np.load(f"{RES}/test_labels.npy")
bins = np.linspace(8, 24, 9).tolist() + [30, 70]; bins = sorted(set(bins))
h = lambda a: (np.histogram(a, bins)[0] / len(a)).tolist()
out["hist"] = {"bins": bins, "val_control": h(yv), "test_control": h(yt[Lt == 0]), "test_asd": h(yt[Lt == 1])}
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(Lt, yt); rng = np.random.default_rng(0); grid = np.linspace(0, 1, 21); tp = []
for _ in range(500):
    i = rng.integers(0, len(Lt), len(Lt))
    if Lt[i].min() == Lt[i].max(): continue
    f_, t_, _ = roc_curve(Lt[i], yt[i]); tp.append(np.interp(grid, f_, t_))
lo, hi = np.percentile(tp, [2.5, 97.5], axis=0)
out["roc"] = {"grid": grid.tolist(), "gru": np.interp(grid, fpr, tpr).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
# generated records for the examples table
recs = json.load(open("outputs/experiments/phase2/generated_anomalies.json")); subs = sorted(set(x["subject_id"] for x in recs)); chosen = [subs[0], subs[20], subs[40]]; picked = []
for sid, sev in zip(chosen, ["high", "medium", "low"]):
    for meth in ["structure_aware", "random"]:
        for x in recs:
            if x["subject_id"] == sid and x["severity_name"] == sev and x["method"] == meth: picked.append({k: x[k] for k in ("subject_id", "method", "severity_name", "severity", "num_edges_changed", "num_undirected_edges", "anomaly_score_before", "anomaly_score_after")}); break
out["records"] = picked
V = json.load(open("outputs/experiments/phase2/generator_validation.json")); sv = V["severity_control"]["per_group"]
out["severity_curve"] = {m: [sv[f"{m}/{l}"]["mean"] for l in ("low", "medium", "high")] for m in ("structure_aware", "random")}; out["detect"] = V["detectability"]; out["tau_transition"] = V["tau_transition"]
json.dump(out, open("paper/deck/deck_data.json", "w")); print("ok", {k: (len(v["nodes"]), len(v["edges"])) for k, v in out.items() if isinstance(v, dict) and "nodes" in v}, out["synth_info"])
