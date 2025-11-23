"""
WorldModel for RoutingAGI
-------------------------

A simple residual MLP-based world model that rolls the hidden state
forward in "planning time".

Intuition:
    - Input:  h_0  = hidden state from RoutingAGI decoder [B, T, D]
    - World model predicts deltas: Δh = f(h)
    - Next state: h_{k+1} = h_k + Δh_k
    - Unroll for K steps to simulate future routing/network evolution.

This is generic:
    - h can encode capacity, slack, congestion, route structure, etc.
    - WorldModel learns dynamics of those features.

Shapes:
    Input:
        h0: [B, T, D]     (batch x nodes/time x features)

    Output:
        rollout: [B, K, T, D]
            rollout[:, 0] = h_1 (1-step ahead)
            rollout[:, 1] = h_2
            ...
            rollout[:, K-1] = h_K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import FeedForward


class WorldModel(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
    ):
        """
        Args:
            dim:        model dimension D (same as RoutingAGI hidden size)
            hidden_dim: internal FFN dim (default: 4 * dim)
            num_layers: how many residual MLP blocks to use inside the
                        transition function.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        # Stack of small residual MLP blocks to model dynamics
        self.blocks = nn.ModuleList([
            FeedForward(dim, hidden_dim=hidden_dim, activation="silu")
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def transition(self, h: torch.Tensor) -> torch.Tensor:
        """
        One-step transition function:
            h_next = h + f(h)

        h: [B, T, D]
        returns: [B, T, D]
        """
        x = self.norm(h)
        for block in self.blocks:
            x = x + block(x)   # residual inside world model
        return h + x            # global residual: h_next = h + Δh

    def forward(self, h0: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Roll out the world model for `steps` transitions.

        Args:
            h0:    [B, T, D]  initial hidden state
            steps: int, number of rollout steps

        Returns:
            rollout: [B, steps, T, D]
                rollout[:, s] = h_{s+1}
        """
        assert h0.dim() == 3, "WorldModel expects [B, T, D] input."

        B, T, D = h0.shape
        h = h0
        states = []

        for _ in range(steps):
            h = self.transition(h)     # [B, T, D]
            states.append(h)

        # Stack over "future step" dimension
        rollout = torch.stack(states, dim=1)   # [B, steps, T, D]
        return rollout


# Tiny smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 16
    w = WorldModel(dim=D, num_layers=2)
    h0 = torch.randn(B, T, D)
    out = w(h0, steps=3)
    print("input:", h0.shape)
    print("rollout:", out.shape)  # [B, 3, T, D]
