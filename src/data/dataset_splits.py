"""
Subject-Level Dataset Splitting for ABIDE I.

Enforces:
- Strict subject-level splitting (no temporal leakage: all windows of a subject stay in one split)
- Stratification by diagnosis (ASD vs Control)
- Detailed per-site distribution reporting
- Deterministic reproducibility via seed
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_subject_splits(
    subject_summary_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: str = None
) -> Dict[str, dict]:
    """
    Creates subject-level train/val/test splits stratified by diagnosis label.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"

    df = subject_summary_df.copy()
    labels = df["label"].values
    subject_ids = df["subject_id"].values
    sites = df["site"].values

    # Step 1: Split into Train+Val vs Test
    test_size = test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(sss1.split(df, labels))

    # Step 2: Split Train+Val into Train vs Val
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_seed)
    train_sub_idx, val_sub_idx = next(sss2.split(df.iloc[train_val_idx], labels[train_val_idx]))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    split_dict = {
        "train": df.iloc[train_idx]["subject_id"].tolist(),
        "val": df.iloc[val_idx]["subject_id"].tolist(),
        "test": df.iloc[test_idx]["subject_id"].tolist()
    }

    # Verify mutual exclusivity
    s_train = set(split_dict["train"])
    s_val = set(split_dict["val"])
    s_test = set(split_dict["test"])
    assert len(s_train.intersection(s_val)) == 0, "Train and Val overlap!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test overlap!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test overlap!"
    assert len(s_train) + len(s_val) + len(s_test) == len(df), "Total subjects mismatch!"

    # Compute detailed statistics per split
    manifest = {
        "metadata": {
            "total_subjects": len(df),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed
        },
        "splits": split_dict,
        "split_statistics": {}
    }

    print("\n================================================================================")
    print("                         SUBJECT-LEVEL SPLIT SUMMARY                            ")
    print("================================================================================")
    
    for split_name in ["train", "val", "test"]:
        sub_df = df[df["subject_id"].isin(split_dict[split_name])]
        n_tot = len(sub_df)
        n_tc = int(np.sum(sub_df["label"] == 0))
        n_asd = int(np.sum(sub_df["label"] == 1))
        site_dist = sub_df["site"].value_counts().to_dict()

        manifest["split_statistics"][split_name] = {
            "count": n_tot,
            "control_count": n_tc,
            "asd_count": n_asd,
            "asd_percentage": round(n_asd / n_tot * 100, 2),
            "site_distribution": site_dist
        }
        print(f"[{split_name.upper():<5}] Total: {n_tot:<4} | TC(0): {n_tc:<4} | ASD(1): {n_asd:<4} ({n_asd/n_tot*100:.2f}% ASD) | Sites: {len(site_dist)}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fp:
            json.dump(manifest, fp, indent=2)
        print(f"[*] Subject splits manifest saved to: {output_path}")

    return manifest
