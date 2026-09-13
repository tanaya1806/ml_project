"""
Phase 1E: Few-Shot Support Sets.

Support sets (1/5/10-shot) are drawn only from designated train/adaptation ASD
subjects -- never from validation or test subjects (subject-level isolation).
"""

from typing import Dict, List, Sequence

import numpy as np


def sample_support_set(pool_subject_ids: Sequence[str], shot: int, seed: int) -> List[str]:
    """Deterministic, reproducible sample of `shot` subjects from `pool_subject_ids`."""
    if shot > len(pool_subject_ids):
        raise ValueError(f"Requested {shot}-shot support but pool only has {len(pool_subject_ids)} subjects.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool_subject_ids), size=shot, replace=False)
    return [pool_subject_ids[i] for i in sorted(idx)]


def assert_subject_isolation(support_ids: Sequence[str], forbidden_ids: Sequence[str]) -> None:
    overlap = set(support_ids) & set(forbidden_ids)
    if overlap:
        raise ValueError(f"Support set leaks {len(overlap)} forbidden (val/test) subject(s): {sorted(overlap)}")


def compute_recurrence(
    support_ids: Sequence[str],
    subject_seed_anomaly_flags: Dict[str, Dict[int, bool]],
) -> Dict[int, float]:
    """
    C(S) = (# support subjects showing an anomaly in subgraph S) / (# support subjects).
    subject_seed_anomaly_flags[subject_id][seed_roi] = True if that subject had >=1
    anomalous transition in that seed's subgraph.
    """
    if len(support_ids) == 0:
        return {}
    seeds = set()
    for sid in support_ids:
        seeds.update(subject_seed_anomaly_flags.get(sid, {}).keys())

    recurrence = {}
    for seed in seeds:
        count = sum(1 for sid in support_ids if subject_seed_anomaly_flags.get(sid, {}).get(seed, False))
        recurrence[seed] = count / len(support_ids)
    return recurrence
