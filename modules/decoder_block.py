"""
Decoder Block for RoutingAGI
----------------------------

One layer of the RoutingAGI decoder stack:

    x ──LN────> MambaBlock ──► + x ──LN────> ConstraintMoE ──► +  ──> h_out

Outputs:
    h_out: [B, T, D]   – updated hidden states
    aux:   dict[name] -> [B, T, out_dim] constraint predictions
"""

import torch
import torch.nn as nn

from modules.mamba_block import MambaBlock
from modules.constraint_moe import ConstraintMoE
from modules.layers import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, use_ffn: bool = False):
        """
        dim: model hidden size D
        use_ffn: if True, adds an extra FeedForward after MoE (optional)
        """
        super().__init__()

        self.dim = dim
        self.use_ffn = use_ffn

        # Norms
        self.norm_mamba = nn.LayerNorm(dim)
        self.norm_moe = nn.LayerNorm(dim)

        # Core blocks
        self.mamba = MambaBlock(dim)
        self.moe = ConstraintMoE(dim)

        # Optional FFN (like transformer MLP) on top
        if use_ffn:
            self.norm_ffn = nn.LayerNorm(dim)
            self.ffn = FeedForward(dim, activation="silu")

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]

        Returns:
            h:   [B, T, D]
            aux: dict[name] -> [B, T, out_dim]
        """

        # -------------------------------------------------------
        # 1) Mamba path: temporal / dynamic filtering
        # -------------------------------------------------------
        h_norm = self.norm_mamba(x)
        mamba_out = self.mamba(h_norm)      # [B,T,D]
        h = x + mamba_out                   # residual 1

        # -------------------------------------------------------
        # 2) Constraint MoE: constraint-aware feature shaping
        # -------------------------------------------------------
        h_moe_in = self.norm_moe(h)
        moe_delta, aux = self.moe(h_moe_in) # [B,T,D], dict of [B,T,out_dim]
        h = h + moe_delta                   # residual 2

        # -------------------------------------------------------
        # 3) Optional FeedForward (extra mixing)
        # -------------------------------------------------------
        if self.use_ffn:
            h_ffn_in = self.norm_ffn(h)
            h = h + self.ffn(h_ffn_in)      # residual 3

        return h, aux
