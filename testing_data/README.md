# testing_data

**Front end (`app/index.html`)** is now a clinic-style screening page: upload a scan, get "autism indicators detected / not detected" and, if detected, the severity level (low / moderate / high). Optional patient details and a local patient-records list are kept only in the browser. The earlier technical page is still served at `/research`; `final testing time series data/scans/*.csv` can be uploaded directly (also reachable at `/demo-scans/<file>`).

Local web app + 20 held-out sample scans for the GRAPH_ANOMALY detector.

```bash
python3 testing_data/app/server.py        # from the repo root, then open http://localhost:8800
```

- `app/` – the website (white / light-blue UI) and its stdlib-only server. Upload a raw ROI time series
  (`.csv .tsv .txt .1D .npy .npz`; rows = time points, 360 ROI columns, or the transpose; at least 40 time points).
- `samples/sample_01.csv … sample_20.csv` – raw ROI series of 20 test-split subjects (10 control, 10 ASD with a known
  ADOS severity), drawn at random with a fixed seed (`make_samples.py`); file numbers were shuffled. Selection never looked at model output.
- `answer_key_and_results.csv` – ground truth for each file and the detector's real output on it (including the wrong ones).
- `make_samples.py` – regenerates all of the above (needs the raw ABIDE time series folder referenced in `src/severity/linkage.py`).

## What the results mean
The detector is the Phase 1 model: test AUROC 0.453 (95% CI 0.353–0.552), accuracy 53.4% on 131 test subjects, i.e. no better than
chance, and it flags very few subjects. On these 20 files it matched the true label for 11 and flagged 1. The severity estimate
(low / moderate / high ADOS-2 bands) was below a majority-class baseline in evaluation. This is a research prototype and must not be used
for diagnosis. Severity labels come from linking the time series to the ABIDE phenotype file by reproducing the nilearn quality filter
(per-site counts match; within-group order is assumed).
