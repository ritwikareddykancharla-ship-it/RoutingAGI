"""
Shared layers, activations, and small helper modules for RoutingAGI.
-------------------------------------------------------------------

This file defines:

- ACTIVATIONS: a mapping from string name → nn.Module class
- FeedForward: generic MLP block (LLaMA-style)
- SwiGLU: Swish/SiLU-based gated linear unit

Other modules (like ConstraintMoE) import ACTIVATIONS from here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Activation registry
# ============================================================

ACTIVATIONS = {
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "silu": nn.SiLU,          # a.k.a Swish
}


# ============================================================
# SwiGLU block (LLaMA-style)
# ============================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation as used in LLaMA-style MLPs:

        gate = SiLU(x W_g)
        up   = x W_u
        out  = gate * up

    Optionally followed by a projection back to model_dim.
    """

    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return gate * up


# ============================================================
# Generic FeedForward block
# ============================================================

class FeedForward(nn.Module):
    """
    Flexible MLP block:

        x -> Linear(dim, hidden_dim)
             -> activation
             -> Linear(hidden_dim, dim)

    By default uses SiLU and 4x expansion like modern LLMs.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        act = ACTIVATIONS[activation]()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            act,
            nn.Linear(hidden_dim, dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
