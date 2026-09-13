"""
ABIDE I rs-fMRI Data Loader.

Robustly handles:
- 20 site NPZ files with both fixed-shape arrays and object arrays (variable time series lengths)
- 360 ROIs
- Binary diagnosis labels (0 = Typical Control, 1 = Autism Spectrum Disorder)
- Metadata extraction (site, duration, indexing)
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Generator, Optional, Dict
import numpy as np


@dataclass
class SubjectRecord:
    subject_id: str
    site: str
    site_subject_idx: int
    global_idx: int
    label: int  # 0 = Control, 1 = Autism
    timepoints: int
    num_rois: int
    time_series: np.ndarray  # Shape: (T, 360), dtype: np.float32


class ABIDELoader:
    """
    Loader for ABIDE I resting-state fMRI ROI time-series across all scanning sites.
    """
    def __init__(self, ts_dir: str):
        self.ts_dir = ts_dir
        self.npz_files = sorted(glob.glob(os.path.join(ts_dir, "site_*time_series.npz")))
        if not self.npz_files:
            raise FileNotFoundError(f"No site NPZ files found in directory: {ts_dir}")
        self._site_map = {}
        for f in self.npz_files:
            site_name = os.path.basename(f).replace("site_", "").replace("time_series.npz", "")
            self._site_map[site_name] = f

    @property
    def available_sites(self) -> List[str]:
        return list(self._site_map.keys())

    def load_site(self, site_name: str, dtype=np.float32) -> List[SubjectRecord]:
        """
        Loads all subjects for a specific scanning site.
        """
        if site_name not in self._site_map:
            raise KeyError(f"Site '{site_name}' not found. Available sites: {self.available_sites}")
        
        file_path = self._site_map[site_name]
        data = np.load(file_path, allow_pickle=True)
        ts_all = data["abide_ts"]
        labels = data["labels"]

        if len(ts_all) != len(labels):
            raise ValueError(f"Site {site_name}: length of abide_ts ({len(ts_all)}) != labels ({len(labels)})")

        records = []
        for i in range(len(ts_all)):
            subj_ts = ts_all[i]
            if not isinstance(subj_ts, np.ndarray):
                subj_ts = np.array(subj_ts)
            
            # Ensure float32 for downstream computation
            subj_ts = subj_ts.astype(dtype)
            t_len, rois = subj_ts.shape
            
            lbl = int(round(float(labels[i])))
            if lbl not in (0, 1):
                raise ValueError(f"Unexpected label {lbl} for site {site_name} subject {i}")

            subject_id = f"{site_name}_{i:03d}"
            record = SubjectRecord(
                subject_id=subject_id,
                site=site_name,
                site_subject_idx=i,
                global_idx=-1,  # Populated during full dataset loading
                label=lbl,
                timepoints=t_len,
                num_rois=rois,
                time_series=subj_ts
            )
            records.append(record)

        return records

    def load_all_subjects(self, dtype=np.float32) -> List[SubjectRecord]:
        """
        Loads all subjects across all 20 scanning sites into memory.
        """
        all_records = []
        global_idx = 0
        for site_name in self.available_sites:
            site_records = self.load_site(site_name, dtype=dtype)
            for rec in site_records:
                rec.global_idx = global_idx
                all_records.append(rec)
                global_idx += 1
        return all_records

    def iter_subjects(self, dtype=np.float32) -> Generator[SubjectRecord, None, None]:
        """
        Generator yielding subjects one by one to save memory during streaming/batch processing.
        """
        global_idx = 0
        for site_name in self.available_sites:
            site_records = self.load_site(site_name, dtype=dtype)
            for rec in site_records:
                rec.global_idx = global_idx
                yield rec
                global_idx += 1

    def get_summary_df(self):
        """
        Returns a pandas DataFrame summarizing all subjects across sites without loading full time-series into memory.
        """
        import pandas as pd
        rows = []
        global_idx = 0
        for site_name, fpath in self._site_map.items():
            data = np.load(fpath, allow_pickle=True)
            ts_all = data["abide_ts"]
            labels = data["labels"]
            for i in range(len(ts_all)):
                s = ts_all[i]
                shape = s.shape if hasattr(s, "shape") else (len(s), len(s[0]))
                lbl = int(round(float(labels[i])))
                rows.append({
                    "global_idx": global_idx,
                    "subject_id": f"{site_name}_{i:03d}",
                    "site": site_name,
                    "site_subject_idx": i,
                    "label": lbl,
                    "diagnosis": "ASD" if lbl == 1 else "Control",
                    "timepoints": shape[0],
                    "num_rois": shape[1]
                })
                global_idx += 1
        return pd.DataFrame(rows)
