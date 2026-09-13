"""
Phase 1C: Lightweight GRU Temporal Model.

Learns how normal functional brain substructures evolve through time:
    h_t = GRU(z_t, h_{t-1})
    z_hat_{t+1} = f(h_t)

Trained only on control (TC) subjects so h_t / z_hat encode "normal" dynamics.
"""

import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.readout = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, D] GIN embedding sequence z_1..z_T
        Returns:
            z_hat: [B, T, D] where z_hat[:, t] = predicted z_{t+1} given z_1..z_t
                   (z_hat[:, -1] predicts one step beyond the observed sequence and
                   has no ground truth target; callers compare z_hat[:, :-1] to z_seq[:, 1:]).
        """
        h, _ = self.gru(z_seq)
        z_hat = self.readout(h)
        return z_hat
