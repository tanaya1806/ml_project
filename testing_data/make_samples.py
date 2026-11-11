"""Pick 20 held-out TEST-split subjects (10 control, 10 ASD with a known ADOS severity) at random with a fixed seed,
write their raw ROI time series as CSV (rows = time points, 360 columns), run the real pipeline on each CSV and write the
answer key and the actual results. Selection never looks at model outputs. Run from repo root."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src.data.loader import ABIDELoader
from src.inference import engine as E
from src.severity.linkage import build_link, severity_band
import joblib

D = "testing_data"; RAWTS = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/GRAPH_ANAMOLY/data/raw/abide/Time_serie_per_site"
link = build_link(); test_ids = set(json.load(open("data/processed/metadata/subject_splits.json"))["splits"]["test"])
t = link[link.subject_id.isin(test_ids)]
rng = np.random.default_rng(2026)
ctl = rng.choice(t[t.label == 0].subject_id.values, 10, replace=False)
asd = rng.choice(t[(t.label == 1) & t.ados_css.notna()].subject_id.values, 10, replace=False)
chosen = list(ctl) + list(asd); order = rng.permutation(len(chosen))  # shuffle so file number reveals nothing
loader = ABIDELoader(RAWTS); cache = {}
sev = joblib.load("outputs/experiments/severity/severity_model.joblib")
rows = []
for n, i in enumerate(order, 1):
    sid = chosen[i]; site = sid.rsplit("_", 1)[0]
    if site not in cache: cache[site] = {r.subject_id: r for r in loader.load_site(site)}
    rec = cache[site][sid]; fn = f"sample_{n:02d}.csv"
    np.savetxt(f"{D}/samples/{fn}", rec.time_series, delimiter=",", fmt="%.6g")
    r = E.analyze(np.loadtxt(f"{D}/samples/{fn}", delimiter=","), sid)  # exactly what the web app will do with the file
    meta = link[link.subject_id == sid].iloc[0]
    pred = sev["model"].predict(np.array(r["features"])[None])[0] if r["flagged"] else ""
    rows.append(dict(file=fn, subject_id=sid, site=site, true_label="ASD" if meta.label else "Control",
                     true_ados_css=meta.ados_css, true_severity_band=severity_band(meta.ados_css) if meta.label else "",
                     time_points=r["n_timepoints"], subject_score=round(r["subject_score"], 3), threshold=round(r["tau_subject"], 3),
                     detector_output="flagged" if r["flagged"] else "not flagged",
                     detector_correct=bool(r["flagged"] == bool(meta.label)),
                     severity_band_if_flagged=pred, severity_correct=(pred == severity_band(meta.ados_css)) if (meta.label and pred) else ""))
df = pd.DataFrame(rows).sort_values("file"); df.to_csv(f"{D}/answer_key_and_results.csv", index=False)
print(df.to_string(index=False))
acc = df.detector_correct.mean(); print("\nsample accuracy", acc, "| flagged", (df.detector_output == "flagged").sum())
