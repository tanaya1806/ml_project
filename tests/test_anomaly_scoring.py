"""Unit tests for Phase 1D anomaly scoring."""

import numpy as np
import pytest
import torch

from src.anomaly.scoring import transition_scores, calibrate_threshold, flag_anomalous


def test_transition_scores_values():
    z_seq = torch.tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])  # [T=3, D=2]
    z_hat_seq = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])  # predicts next step
    a_t = transition_scores(z_seq, z_hat_seq)
    # A[0] = ||z_1 - z_hat_0|| = ||(3,4)-(0,0)|| = 5
    # A[1] = ||z_2 - z_hat_1|| = ||(3,4)-(0,0)|| = 5
    assert a_t.shape == (2,)
    assert torch.allclose(a_t, torch.tensor([5.0, 5.0]), atol=1e-5)


def test_transition_scores_short_sequence():
    z_seq = torch.randn(1, 4)
    z_hat_seq = torch.randn(1, 4)
    a_t = transition_scores(z_seq, z_hat_seq)
    assert a_t.shape == (0,)


def test_calibrate_threshold_percentile():
    scores = list(range(1, 101))  # 1..100
    tau = calibrate_threshold(scores, percentile=95.0)
    assert np.isclose(tau, np.percentile(scores, 95.0))


def test_calibrate_threshold_empty_raises():
    with pytest.raises(ValueError):
        calibrate_threshold([], percentile=95.0)


def test_flag_anomalous():
    scores = [0.1, 5.0, 10.0, 0.5]
    flags = flag_anomalous(scores, tau=4.0)
    assert flags.tolist() == [False, True, True, False]
