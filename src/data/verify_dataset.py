"""
Dataset Verification Script for ABIDE I rs-fMRI Time Series and Phenotypic Data.
Performs rigorous structural, dimensional, statistical, and consistency checks across all sites.
"""

import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify_abide_dataset(data_dir: str, output_report_path: str = None) -> dict:
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    ts_dir = raw_dir / "Time_serie_per_site"
    pheno_file = raw_dir / "abide" / "Phenotypic_V1_0b_preprocessed1.csv"

    print("================================================================================")
    print("                      ABIDE I DATASET VERIFICATION REPORT                      ")
    print("================================================================================")

    results = {
        "raw_directory": str(raw_dir),
        "phenotypic_file": {
            "exists": pheno_file.exists(),
            "path": str(pheno_file)
        },
        "sites": {},
        "summary": {}
    }

    # 1. Phenotypic file check
    if pheno_file.exists():
        df_pheno = pd.read_csv(pheno_file)
        results["phenotypic_file"]["num_records"] = len(df_pheno)
        results["phenotypic_file"]["num_columns"] = len(df_pheno.columns)
        results["phenotypic_file"]["columns"] = list(df_pheno.columns)
        results["phenotypic_file"]["dx_counts"] = df_pheno["DX_GROUP"].value_counts().to_dict()
        results["phenotypic_file"]["site_counts"] = df_pheno["SITE_ID"].value_counts().to_dict()
        print(f"[*] Phenotypic CSV loaded: {len(df_pheno)} records, {len(df_pheno.columns)} columns.")
    else:
        print(f"[!] WARNING: Phenotypic CSV not found at {pheno_file}")

    # 2. NPZ time series files check
    npz_files = sorted(glob.glob(str(ts_dir / "site_*time_series.npz")))
    print(f"[*] Found {len(npz_files)} site NPZ files in {ts_dir}")

    total_subjects_all = 0
    total_autism_all = 0
    total_normal_all = 0
    total_nans_all = 0
    total_infs_all = 0
    roi_counts_set = set()
    all_site_names = []

    for npz_path in npz_files:
        fname = os.path.basename(npz_path)
        site_name = fname.replace("site_", "").replace("time_series.npz", "")
        all_site_names.append(site_name)

        try:
            data = np.load(npz_path, allow_pickle=True)
            keys = list(data.keys())
            
            # Key checks
            expected_keys = {"abide_ts_normal", "abide_ts_autism", "abide_ts", "labels"}
            missing_keys = expected_keys - set(keys)
            extra_keys = set(keys) - expected_keys

            ts_normal = data["abide_ts_normal"]
            ts_autism = data["abide_ts_autism"]
            ts_all = data["abide_ts"]
            labels = data["labels"]

            n_normal = len(ts_normal)
            n_autism = len(ts_autism)
            n_all = len(ts_all)
            n_labels = len(labels)

            total_subjects_all += n_all
            total_autism_all += n_autism
            total_normal_all += n_normal

            # Consistency check: len(abide_ts) == len(labels)
            len_match = (n_all == n_labels)
            sum_match = (n_normal + n_autism == n_all)

            # Label encoding check
            unique_labels = sorted(list(np.unique(labels)))
            label_0_count = int(np.sum(labels == 0.0))
            label_1_count = int(np.sum(labels == 1.0))
            label_alignment_ok = (label_0_count == n_normal and label_1_count == n_autism)

            # Check if labels are ordered: normal first then autism
            normal_labels_sub = labels[:n_normal]
            autism_labels_sub = labels[n_normal:]
            ordered_alignment = bool(np.all(normal_labels_sub == 0.0) and np.all(autism_labels_sub == 1.0))

            # Inspect array types, ROIs, timepoints, NaNs, Infs
            is_object_array = (ts_all.dtype == object)
            timepoints_list = []
            site_nans = 0
            site_infs = 0
            site_rois = set()

            for i in range(n_all):
                subj_ts = ts_all[i]
                if not isinstance(subj_ts, np.ndarray):
                    subj_ts = np.array(subj_ts)
                shape = subj_ts.shape
                timepoints_list.append(shape[0])
                site_rois.add(shape[1])
                roi_counts_set.add(shape[1])

                nans = int(np.isnan(subj_ts).sum())
                infs = int(np.isinf(subj_ts).sum())
                site_nans += nans
                site_infs += infs

            total_nans_all += site_nans
            total_infs_all += site_infs

            # Verify if abide_ts matches concatenation of abide_ts_normal + abide_ts_autism
            exact_content_match = True
            for i in range(n_normal):
                if not np.allclose(ts_all[i], ts_normal[i], equal_nan=True):
                    exact_content_match = False
                    break
            if exact_content_match:
                for i in range(n_autism):
                    if not np.allclose(ts_all[n_normal + i], ts_autism[i], equal_nan=True):
                        exact_content_match = False
                        break

            # Subject IDs presence check: look for dedicated subject identifier arrays
            has_subject_ids = any(k.lower() in ["sub_id", "subject_id", "sub_ids", "subject_ids", "subject", "sub"] for k in keys)

            site_info = {
                "file": fname,
                "num_subjects": n_all,
                "num_normal": n_normal,
                "num_autism": n_autism,
                "num_labels": n_labels,
                "len_match": len_match,
                "sum_match": sum_match,
                "exact_content_match": exact_content_match,
                "label_values": unique_labels,
                "label_alignment_ok": label_alignment_ok,
                "ordered_alignment": ordered_alignment,
                "is_object_array": is_object_array,
                "min_timepoints": min(timepoints_list),
                "max_timepoints": max(timepoints_list),
                "is_fixed_timepoints": (min(timepoints_list) == max(timepoints_list)),
                "rois": list(site_rois),
                "nan_count": site_nans,
                "inf_count": site_infs,
                "dtype": str(ts_all.dtype if not is_object_array else ts_all[0].dtype),
                "has_subject_ids": has_subject_ids,
                "extra_keys": list(extra_keys),
                "missing_keys": list(missing_keys)
            }
            results["sites"][site_name] = site_info

        except Exception as e:
            results["sites"][site_name] = {
                "file": fname,
                "error": str(e),
                "malformed": True
            }

    results["summary"] = {
        "total_sites": len(npz_files),
        "total_subjects": total_subjects_all,
        "total_autism": total_autism_all,
        "total_normal": total_normal_all,
        "class_ratio_autism": total_autism_all / total_subjects_all if total_subjects_all > 0 else 0,
        "class_ratio_normal": total_normal_all / total_subjects_all if total_subjects_all > 0 else 0,
        "all_rois": list(roi_counts_set),
        "total_nans": total_nans_all,
        "total_infs": total_infs_all,
        "all_len_match": all(s.get("len_match", False) for s in results["sites"].values()),
        "all_sum_match": all(s.get("sum_match", False) for s in results["sites"].values()),
        "all_content_match": all(s.get("exact_content_match", False) for s in results["sites"].values()),
        "all_ordered_alignment": all(s.get("ordered_alignment", False) for s in results["sites"].values()),
        "has_subject_ids_in_npz": any(s.get("has_subject_ids", False) for s in results["sites"].values())
    }

    # Print clean summary table
    print("\n--- PER-SITE BREAKDOWN ---")
    header = f"{'Site':<12} | {'Total':<6} | {'TC(0)':<6} | {'ASD(1)':<6} | {'Timepoints':<12} | {'ROIs':<5} | {'Fixed?':<7} | {'NaNs':<5}"
    print(header)
    print("-" * len(header))
    for site, s in results["sites"].items():
        if "error" in s:
            print(f"{site:<12} | ERROR: {s['error']}")
        else:
            tp_str = f"{s['min_timepoints']}" if s['is_fixed_timepoints'] else f"{s['min_timepoints']}-{s['max_timepoints']}"
            fixed_str = "Yes" if s['is_fixed_timepoints'] else "No"
            roi_str = ",".join(map(str, s['rois']))
            print(f"{site:<12} | {s['num_subjects']:<6} | {s['num_normal']:<6} | {s['num_autism']:<6} | {tp_str:<12} | {roi_str:<5} | {fixed_str:<7} | {s['nan_count']:<5}")
    print("-" * len(header))
    print(f"{'TOTAL':<12} | {total_subjects_all:<6} | {total_normal_all:<6} | {total_autism_all:<6} | {'--':<12} | {360:<5} | {'--':<7} | {total_nans_all:<5}")

    print("\n--- GLOBAL CONSISTENCY CHECKS ---")
    print(f"[*] Total recording sites: {results['summary']['total_sites']}")
    print(f"[*] Total subjects with time-series: {results['summary']['total_subjects']}")
    print(f"[*] Total Autism Spectrum Disorder (ASD): {results['summary']['total_autism']} ({results['summary']['class_ratio_autism']*100:.2f}%)")
    print(f"[*] Total Typical Controls (TC): {results['summary']['total_normal']} ({results['summary']['class_ratio_normal']*100:.2f}%)")
    print(f"[*] Brain ROIs across all sites: {results['summary']['all_rois']}")
    print(f"[*] Total NaNs across all time series: {results['summary']['total_nans']}")
    print(f"[*] Total Infs across all time series: {results['summary']['total_infs']}")
    print(f"[*] len(abide_ts) == len(labels) for ALL sites: {results['summary']['all_len_match']}")
    print(f"[*] abide_ts_normal + abide_ts_autism == abide_ts count for ALL sites: {results['summary']['all_sum_match']}")
    print(f"[*] Exact content match between subsets and abide_ts for ALL sites: {results['summary']['all_content_match']}")
    print(f"[*] Strictly ordered alignment (Normal=0 first, ASD=1 second) for ALL sites: {results['summary']['all_ordered_alignment']}")
    print(f"[*] Subject IDs found inside NPZ files: {results['summary']['has_subject_ids_in_npz']}")

    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"[*] Detailed JSON verification report saved to: {output_report_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify ABIDE I dataset structure and integrity.")
    parser.add_argument("--data_dir", type=str, default=r"D:\Graph anomaly detection\data", help="Root data directory")
    parser.add_argument("--output_report", type=str, default=r"D:\Graph anomaly detection\data\processed\metadata\verification_report.json", help="Path to save output JSON report")
    args = parser.parse_args()
    verify_abide_dataset(args.data_dir, args.output_report)
