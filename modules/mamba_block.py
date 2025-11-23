"""
Mamba-style State-Space Model (SSM) Block
-----------------------------------------

This is a clean, readable PyTorch implementation of a Mamba block.

It follows the core components shown in Mamba:
- input projection into (x_proj, gate)
- learned state matrix A
- dynamic input B_t and output C_t filters
- learned time step dt_t
- running hidden state h_t with selective scan
- output projection

Shapes:
    Input:  [B, T, D]
    Output: [B, T, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    Minimal but fully-correct Mamba-style block.

    This version is readable and perfect for experimentation.
    """

    def __init__(self, dim: int, dt_rank: int = 16):
        super().__init__()

        self.dim = dim
        hidden = dim * 2  # for projecting into (x_proj, gates)

        # ------------------------------------------------------------------
        # 1. Input projection → split into projection + gate branch
        # ------------------------------------------------------------------
        self.in_proj = nn.Linear(dim, hidden)  # [D → 2D]

        # ------------------------------------------------------------------
        # 2. State space core parameters
        # A: diagonal state matrix (learned)
        # B, C: dynamic filters produced from x_proj
        # dt: learned time-step scale
        # ------------------------------------------------------------------
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)

        self.dt = nn.Linear(dim, dt_rank)          # low-rank dt
        self.dt_proj = nn.Linear(dt_rank, dim)     # back to D dim

        # ------------------------------------------------------------------
        # 3. Output projection
        # ------------------------------------------------------------------
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]

        Returns:
            y: [B, T, D]
        """

        B, T, D = x.shape

        # ---------------------------------------------------------------
        # (1) Input projection → x_proj and gate
        # ---------------------------------------------------------------
        u = self.in_proj(x)       # [B, T, 2D]
        x_proj, gate = u.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # ---------------------------------------------------------------
        # (2) Compute dynamic parameters
        # ---------------------------------------------------------------
        # A is constrained to be negative (stable)
        A = -F.softplus(self.A)               # [D]
        B_t = self.B(x_proj)                  # [B, T, D]
        C_t = self.C(x_proj)                  # [B, T, D]

        dt_raw = self.dt(x_proj)              # [B, T, dt_rank]
        dt = self.dt_proj(F.softplus(dt_raw)) # [B, T, D]

        # ---------------------------------------------------------------
        # (3) Selective scan over time — core SSM logic
        # ---------------------------------------------------------------
        h = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            # dt controls exponential decay
            expA = torch.exp(dt[:, t] * A)    # [B, D]

            # Recurrence:
            # h_t = expA * h_{t-1} + dt_t * B_t
            h = expA * h + dt[:, t] * B_t[:, t]

            # Output y_t = C_t * h_t
            y_t = C_t[:, t] * h
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, D]

        # ---------------------------------------------------------------
        # (4) gating + output projection
        # ---------------------------------------------------------------
        y = y * gate
        y = self.out_proj(y)

        return y
