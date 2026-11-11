"""Build the 100-scan SYNTHETIC test set, run the real Phase 1 pipeline on every CSV exactly as the web app does, and write the answer key.
Design fixed before running (no tuning to hit target counts): 50 normal-like scans and 50 injected-anomaly scans in three strength levels.
Run from the repo root:  python3 "final testing time series data/make_final_dataset.py" """
import os, sys
import numpy as np, pandas as pd, joblib
D = "final testing time series data"; sys.path.insert(0, os.getcwd()); sys.path.insert(0, D)
import synth
from src.inference import engine as E
rng = np.random.default_rng(20261110)
LEVELS = {"low": (17, (0.7, 1.1), (1, 1)), "moderate": (17, (1.2, 2.0), (1, 2)), "high": (16, (2.0, 3.5), (2, 3))}   # n scans, amplitude range, n seeds range
plan = [("normal_like", "", 0.0, 0)] * 50
for band, (n, (a0, a1), (s0, s1)) in LEVELS.items():
    plan += [("injected_anomaly", band, float(rng.uniform(a0, a1)), int(rng.integers(s0, s1 + 1))) for _ in range(n)]
order = rng.permutation(len(plan))
sev = joblib.load("outputs/experiments/severity/severity_model.joblib"); os.makedirs(f"{D}/scans", exist_ok=True)
rows = []
for n, i in enumerate(order, 1):
    group, band, amp, ns = plan[i]; T = int(rng.integers(120, 201))
    ts, scale = synth.normal_scan(rng, T)
    if group == "injected_anomaly": ts = synth.inject(rng, ts, scale, amp, ns)
    fn = f"scan_{n:03d}.csv"; np.savetxt(f"{D}/scans/{fn}", ts.T, delimiter=",", fmt="%.6g")      # rows = time points, 360 ROI columns
    r = E.analyze(np.loadtxt(f"{D}/scans/{fn}", delimiter=","), fn)
    pred = sev["model"].predict(np.array(r["features"])[None])[0] if r["flagged"] else ""
    rows.append(dict(file=fn, true_group=group, synthetic_strength=band, injected_seeds=ns, injected_amplitude=round(amp, 2), time_points=T,
                     subject_score=round(r["subject_score"], 3), threshold=round(r["tau_subject"], 3), detector_output="flagged" if r["flagged"] else "not flagged",
                     detector_correct=bool(r["flagged"] == (group == "injected_anomaly")),
                     severity_predicted=pred, severity_matches_synthetic_strength=(pred == band) if (pred and band) else ""))
df = pd.DataFrame(rows).sort_values("file"); df.to_csv(f"{D}/answer_key_and_results.csv", index=False)
try: df.to_excel(f"{D}/answer_key_and_results.xlsx", index=False)
except Exception as e: print("xlsx skipped:", e)
a = df[df.true_group == "injected_anomaly"]; c = df[df.true_group == "normal_like"]
print(f"injected-anomaly scans flagged: {(a.detector_output=='flagged').sum()}/{len(a)} (missed {(a.detector_output!='flagged').sum()})")
print(f"normal-like scans flagged: {(c.detector_output=='flagged').sum()}/{len(c)}")
print("by strength:\n", a.groupby("synthetic_strength").detector_output.apply(lambda s: f"{(s=='flagged').sum()}/{len(s)} flagged"))
f = a[a.detector_output == "flagged"]; print("severity predicted vs synthetic strength on flagged anomalies:\n", pd.crosstab(f.synthetic_strength, f.severity_predicted))
print("severity match:", int((f.severity_matches_synthetic_strength == True).sum()), "of", len(f))
