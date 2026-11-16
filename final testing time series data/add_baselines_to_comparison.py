"""Add the four extra baselines (and real-test AUROCs) from paper/results/extra_baselines.json to comparison_results.json. Idempotent. Run from repo root."""
import json
P = "final testing time series data/comparison_results.json"
c = json.load(open(P)); e = json.load(open("paper/results/extra_baselines.json")); cur = e["curves"]
y = cur["y_syn"]; assert y == [1 if s["truth"] == "injected anomaly" else 0 for s in c["scans"]]
for k in ("whole_brain_gin_gru", "ocsvm", "iforest", "autoencoder"):
    fl = cur["syn_flag"][k]; sc = cur["syn"][k]; m = e["syn"][k]
    c["arms"][k] = {"correct": m["tp"] + m["tn"], "accuracy": m["accuracy"], "tp": m["tp"], "fn": m["fn"], "fp": m["fp"], "tn": m["tn"], "flagged": m["tp"] + m["fp"], "per_trial_accuracy": []}
    for s, f, p, t in zip(c["scans"], fl, sc, y):
        s[k] = {"prob": round(float(p), 4), "flagged": bool(f), "correct": bool(f) == bool(t)}
    assert sum(s[k]["correct"] for s in c["scans"]) == c["arms"][k]["correct"], k
real = {}
for k, m in e["real"].items():
    real[{"temporal_subgraph_detector": "screening_detector"}.get(k, k)] = {"auroc": m["auroc"], "ci": m["ci"], "accuracy": m["accuracy"]}
c["real_test"] = {"n": e["n"]["real_test"], "note": "Real held-out ABIDE I patients (ASD vs control).", "models": real}
json.dump(c, open(P, "w")); print("arms:", list(c["arms"]))
