# final testing time series data (SYNTHETIC)

100 synthetic ROI time series (`scans/scan_001.csv … scan_100.csv`; rows = time points, 360 ROI columns, 120–200 time points each).
**This is not patient data and it says nothing about autism detection.** The detector's real result on real ABIDE test subjects is chance level
(AUROC 0.453). Do not present this set as evidence that the detector finds autism or estimates ADOS severity.

## What is in it
- 50 `normal_like` scans: a smooth 12-network factor model with no injected change.
- 50 `injected_anomaly` scans: a shared signal is added to one to three seed ROIs and their 1-hop neighbours for 40 time points, which changes
  that sub-structure's connectivity (the same idea as the paper's guided edits). Strength levels fixed in `make_final_dataset.py` before running:
  low (17 scans), moderate (17), high (16). The label is "injected anomaly", **not** "ASD".
- Scan numbers are shuffled; `answer_key_and_results.csv` / `.xlsx` hold the truth and the pipeline's real output for every file.

## Real results of the Phase 1 pipeline on this set (nothing tuned to hit target counts)
- injected-anomaly scans flagged: 46 / 50 (4 missed, all among the 17 low-strength scans); normal-like scans flagged: 0 / 50.
- Severity: the real severity model (trained on ADOS-2 bands of ABIDE patients) was run on the flagged scans. Its output matched the synthetic strength
  band for 12 of 46 scans, i.e. chance level, as expected because "synthetic strength" is not ADOS severity. Matches and mismatches are both kept in the key
  (`severity_matches_synthetic_strength`).

Regenerate: `python3 "final testing time series data/make_final_dataset.py"` from the repo root. Upload any `scans/scan_xxx.csv` to the web app in `testing_data/app`.
