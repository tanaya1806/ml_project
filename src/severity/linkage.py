"""Link time-series subject IDs (SITE_i) to ABIDE phenotype rows (ADOS severity etc.).

The time series were extracted with nilearn fetch_abide_pcp(quality_checked=True), controls first then ASD per site
(see GSP-applied-on-ASD-classification/ABIDE_dataset_time_series.py). Reproducing that QC filter on the phenotype CSV
yields exactly the 871 subjects and the per-site ASD/control counts of the time-series files (verified below).
Within-group order is ASSUMED to follow phenotype-row order (nilearn preserves it); counts are verified, order is not.
"""
import glob, os
import numpy as np
import pandas as pd

RAW = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/GRAPH_ANAMOLY/data/raw/abide"
OK, OKM = ["OK"], ["OK", "maybe"]


def qc_filter(p: pd.DataFrame) -> pd.DataFrame:
    m = (p.qc_rater_1.isin(OK) & p.qc_anat_rater_2.isin(OKM) & p.qc_func_rater_2.isin(OKM)
         & p.qc_anat_rater_3.isin(OK) & p.qc_func_rater_3.isin(OK))
    return p[m]


def build_link() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(RAW, "Phenotypic_V1_0b_preprocessed1.csv"))
    q = qc_filter(p)
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "Time_serie_per_site", "site_*time_series.npz"))):
        site = os.path.basename(f)[5:-len("time_series.npz")]
        lab = np.load(f, allow_pickle=True)["labels"].astype(int)
        s = q[q.SITE_ID == site]
        ctl, asd = s[s.DX_GROUP == 2], s[s.DX_GROUP == 1]
        assert (len(asd), len(ctl)) == (int(lab.sum()), int((lab == 0).sum())), site  # count check
        ordered = pd.concat([ctl, asd])
        for i, (_, r) in enumerate(ordered.iterrows()):
            css = r.ADOS_GOTHAM_SEVERITY
            rows.append(dict(subject_id=f"{site}_{i:03d}", site=site, sub_id=int(r.SUB_ID), label=int(lab[i]),
                             ados_css=float(css) if pd.notna(css) and css > 0 else np.nan,
                             age=float(r.AGE_AT_SCAN), sex=int(r.SEX)))
            assert int(r.DX_GROUP == 1) == int(lab[i])  # label order check
    return pd.DataFrame(rows)


def severity_band(css: float) -> str:
    """Analyst-defined bands over ADOS-2 calibrated severity score (NOT an official map to DSM-5 levels)."""
    return "low" if css <= 5 else ("moderate" if css <= 7 else "high")


if __name__ == "__main__":
    d = build_link(); d.to_csv("data/processed/metadata/subject_link_phenotype.csv", index=False)
    sp = __import__("json").load(open("data/processed/metadata/subject_splits.json"))
    print(d.shape, d.label.sum(), sp.keys())
    a = d[(d.label == 1)]
    print("ASD:", len(a), " with CSS:", a.ados_css.notna().sum()); print(a.ados_css.value_counts().sort_index().to_dict())
