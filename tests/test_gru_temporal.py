"""Unit tests for the Phase 1C GRU temporal model."""

import torch

from src.models.gru_temporal import TemporalGRU


def test_output_shape():
    model = TemporalGRU(embedding_dim=128, hidden_dim=64, num_layers=1)
    z_seq = torch.randn(4, 10, 128)  # [B, T, D]
    z_hat = model(z_seq)
    assert z_hat.shape == (4, 10, 128)


def test_no_future_leakage():
    """z_hat[:, t] must depend only on z_seq[:, :t+1], not on future timesteps."""
    torch.manual_seed(0)
    model = TemporalGRU(embedding_dim=8, hidden_dim=16, num_layers=1)
    model.eval()

    z_seq = torch.randn(1, 6, 8)
    z_hat_full = model(z_seq)

    # Truncate the sequence after t=3; the prediction at t=2 must be unchanged.
    z_seq_truncated = z_seq[:, :3, :]
    z_hat_truncated = model(z_seq_truncated)

    assert torch.allclose(z_hat_full[:, :3, :], z_hat_truncated, atol=1e-6)


def test_single_timestep():
    model = TemporalGRU(embedding_dim=16, hidden_dim=16, num_layers=1)
    z_seq = torch.randn(2, 1, 16)
    z_hat = model(z_seq)
    assert z_hat.shape == (2, 1, 16)
