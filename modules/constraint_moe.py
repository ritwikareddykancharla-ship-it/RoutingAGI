"""
Constraint-aware Mixture-of-Experts (MoE)
-----------------------------------------

Each expert corresponds to one MILP constraint family:
capacity, time windows, lane legality, throughput, SLA,
flow conservation, mode constraints, region rules,
load balancing, trailer availability.

Each expert:
    trunk:       nonlinear representation learner
    main_head:   returns delta to add to residual stream
    aux_head:    returns constraint-specific prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.constraint_registry import CONSTRAINTS
from modules.layers import ACTIVATIONS   # we will define activations in layers.py


# ============================================================
# Generic Expert (shared template for all constraint types)
# ============================================================

class GenericExpert(nn.Module):
    """
    Shared architecture for all constraint experts:

        h ─→ trunk (MLP)
           ├── main_head (Δh for residual)
           └── constraint_head (prediction for MILP target)

    trunk: nonlinear mapping tuned by activation from registry
    main_head: produces feature delta added to main residual stream
    constraint_head: produces constraint output for loss
    """

    def __init__(self, hidden_dim: int, output_dim: int, activation: str):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        act = ACTIVATIONS[activation]()

        # Expert trunk: 2-layer MLP, 4x expansion
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            act,
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
            act,
        )

        # Residual delta (same dim as h)
        self.main_head = nn.Linear(4 * hidden_dim, hidden_dim)

        # Constraint-specific prediction
        self.constraint_head = nn.Linear(4 * hidden_dim, output_dim)

    def forward(self, h: torch.Tensor):
        """
        h: [B*T, D]

        Returns:
          main_delta: [B*T, D]
          aux_output: [B*T, output_dim]
        """
        x = self.trunk(h)                      # [B*T, 4D]
        main_delta = self.main_head(x)         # [B*T, D]
        aux_output = self.constraint_head(x)   # [B*T, out_dim]
        return main_delta, aux_output


# ============================================================
# Constraint MoE
# ============================================================

class ConstraintMoE(nn.Module):
    """
    Full MoE with one expert per constraint.

    There is NO gating yet:
        - all experts compute deltas
        - deltas are summed
        - aux outputs returned per constraint family

    Output:
        main_sum: [B, T, D]
        aux_dict: { constraint_name: [B, T, out_dim] }
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleDict()

        # Auto-build experts from registry
        for name, cfg in CONSTRAINTS.items():
            self.experts[name] = GenericExpert(
                hidden_dim=hidden_dim,
                output_dim=cfg["output_dim"],
                activation=cfg["activation"],
            )

    def forward(self, h: torch.Tensor):
        """
        h: [B, T, D]

        Returns:
          main_sum: [B, T, D]
          aux: dict( name -> [B, T, output_dim] )
        """
        B, T, D = h.shape
        h_flat = h.reshape(B * T, D)

        # initialize zero accumulator
        main_sum_flat = torch.zeros_like(h_flat)
        aux_dict_flat = {}

        # Run each expert independently
        for name, expert in self.experts.items():
            main_delta_flat, aux_flat = expert(h_flat)

            main_sum_flat += main_delta_flat    # sum deltas
            aux_dict_flat[name] = aux_flat      # store constraint prediction

        # reshape back to [B, T, ...]
        main_sum = main_sum_flat.reshape(B, T, D)
        aux = {
            name: out.reshape(B, T, CONSTRAINTS[name]["output_dim"])
            for name, out in aux_dict_flat.items()
        }

        return main_sum, aux
