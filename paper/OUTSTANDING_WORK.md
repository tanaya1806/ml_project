# Outstanding work and open questions

Items the manuscript depends on or explicitly leaves open. "Experiment" = not run, so no claim is made. "Confirm" = needs the authors' input; nothing was assumed.

## A. Questions for the authors (needed before submission)
1. **Atlas, preprocessing, QC (resolved from the extraction script).** The older project folder contains the extraction script: Glasser 360 atlas via Nilearn, ABIDE preprocessed (CPAC, band-pass, global signal regression, quality_checked). The paper now states this. Still unrecorded: the exact filter cut-offs, and the mapping from ROI index to Glasser area names (needed to label flagged ROIs anatomically); the original reordering file `coords_with_order.npz` was not found.
3. **Title "Dr. Prof."** preserved as supplied; confirm the intended form.
4. **Acknowledgment / funding / ethics statement.** None were supplied, so none is written. ABIDE data-use acknowledgment wording should be checked against the ABIDE data-use agreement.
5. **Author contributions, ORCIDs, venue.** Not supplied and not invented. Venue template: IEEEtran conference class compiled; official venue compliance (page limit, copyright block) is unverified. The manuscript is 12 pages, which will exceed most conference limits.
6. **Hardware.** Original runs were on an Apple-silicon laptop (MPS); chip model and RAM were not logged.
7. **Phase 3 framing.** The seminar slides present Phase 3 as future work; the repository and this paper treat it as completed. Confirm the intended framing.

## B. Experiments not run (the paper reports them as not evaluated)
| RQ | Missing experiment | Why it matters |
|---|---|---|
| RQ2 | Whole-graph static and temporal baselines (no subgraph localisation) | Cannot claim localisation helps |
| RQ5 | Sensitivity to window length, stride, sparsification level, subgraph hop/size, number of seeds | Robustness unknown |
| RQ6 | Leave-site-out / site-stratified evaluation, with and without harmonisation | Site generalisation unknown; all 20 sites are in every split |
| RQ1 | Independent encoder for baselines (the static/persistence baselines share the jointly trained encoder) | Baselines are not independent of the temporal objective |
| - | Encoder alternatives (GCN, GAT, unweighted GIN) and lambda_var = 0 ablation | Component contributions unattributed |
| - | Multiple training seeds with variance | Only one training seed; training variance unreported |
| - | Independent validation of generated anomalies (scored by a model not used to guide edits) | Phase 2 detectability is self-referential |
| - | Richer few-shot heads (per-window features, attention, prototype distance); standard-scaled features | Hypothesis that single-entry edits give little signal is untested |
| - | Support-weighted score A(1+lambda C) | Designed, not implemented |
| - | More than 20 seed ROIs (all 360) on at least a subset | Seed coverage is 5.6% of ROIs |
| - | Covariate analysis (age, sex, site, motion) of subject scores | Confounds not modelled |
| - | Anatomical labelling of flagged ROIs/subgraphs | Needs atlas (A1) |

## C. Known methodological caveats to keep visible
- Two design decisions followed first looks at test output (two thresholds after the all-131-flagged failure; 20-D features after identical AUROC across 1-D conditions). Threshold-free metrics are unaffected; operating-point metrics are not strictly blind. Disclosed in Secs. IV-G, IV-J, VII.
- The few-shot test comparison shares one test set across five support draws; SDs are not independent of test-set variability.
- The per-seed recurrence result (seed ROI 109, p = 0.0019 vs Bonferroni threshold 0.0025) is one of 20 exploratory tests.
- Added analyses (scoring-function ablation, bootstrap CIs, permutation tests, scan-length check, recurrence null, chance-matched F1) were run newly for this paper from the committed checkpoint; scripts and JSON are in `scripts/` and `results/`. The GRU row reproduces the committed Phase 1 metrics exactly.

## D. Discrepancies between project artefacts and code (flagged, not silently fixed)
- Seminar slide 6 says "Epochs: 100"; the code and `training_report.json` show 5 epochs.
- Seminar slide 7 writes the update as w + alpha * s_ij; the code takes sign-gradient steps of size alpha * 0.6 / 5 with clipping to [-1, 1] (Eq. 8 in the paper follows the code).
- Phase 1 report timings: cache build 112 s (not the 71 s/subject pilot figure); the paper reports both with their sources.

## E. Housekeeping
- Nothing from `paper/` or `Output_to_be_shown/` has been committed.
- Before submission: re-run `scripts/verify_refs.py`, re-check Table I against full texts, check the five venue-only references, and look for published versions of the two preprints.

## F. Severity estimation and test app (added later)
- Severity labels are ADOS-2 calibrated severity scores for 193 of 403 ASD subjects; the link from time series to phenotype row reproduces the nilearn QC filter (all 20 per-site counts match) but **within-group subject order is assumed**, not verified. Verify with the original download order or subject IDs before relying on the severity result.
- Severity bands (low <=5, moderate 6-7, high >=8) are our grouping, not an official DSM-5 level mapping.
- Result is null (held-out accuracy 0.433 vs 0.550 majority-class; CV macro-F1 permutation p = 0.080). Any claim of severity detection would need new features/models and independent validation.
- The web app's detector flagged 1 of 20 sample files; its practical sensitivity is very low. Do not present it as a diagnostic.
- The 20 sample files are real test-split subjects chosen at random (not chosen for correct outputs). Results, including errors, are in `testing_data/answer_key_and_results.csv`.
