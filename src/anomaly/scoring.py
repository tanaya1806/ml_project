"""
Phase 1D: Baseline Temporal Anomaly Score.

For each transition t -> t+1:
    A_t = || z_{t+1} - z_hat_{t+1} ||_2

The threshold tau is calibrated from held-out NORMAL (control, validation-split)
subjects only: tau = percentile_95(A_normal_validation). A transition is flagged
anomalous when A_t > tau.
"""

from typing import Sequence

import numpy as np
import torch


def transition_scores(z_seq: torch.Tensor, z_hat_seq: torch.Tensor) -> torch.Tensor:
    """
    z_seq, z_hat_seq: [T, D]. Returns A of shape [T-1]: A[t] = ||z_seq[t+1] - z_hat_seq[t]||_2,
    i.e. the error between the GRU's prediction made at t and the true next embedding.
    """
    if z_seq.shape[0] < 2:
        return torch.zeros(0, device=z_seq.device)
    diff = z_seq[1:] - z_hat_seq[:-1]
    return torch.linalg.norm(diff, dim=-1)


def calibrate_threshold(normal_scores: Sequence[float], percentile: float = 95.0) -> float:
    """tau = percentile_p(A_normal_validation). Never derived from test or ASD subjects."""
    if len(normal_scores) == 0:
        raise ValueError("Cannot calibrate a threshold from an empty normal-score set.")
    return float(np.percentile(np.asarray(normal_scores, dtype=np.float64), percentile))


def flag_anomalous(scores: Sequence[float], tau: float) -> np.ndarray:
    return np.asarray(scores) > tau
