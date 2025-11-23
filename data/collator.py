"""
collator.py
-----------

Collates variable-sized graph samples into padded batches for RoutingAGI.

Each graph may have different number of nodes T.
We pad everything to max_T in the batch.

Output format:
    {
        "node_features": [B, maxT, F],
        "adj_matrix":    [B, maxT, maxT],
        "edge_types":    [B, maxT, maxT],
        "sp_dist":       [B, maxT, maxT],
        "centrality":    [B, maxT],

        "targets":       [B, maxT, D],
        ... (every MILP constraint target): [B, maxT, out_dim]

        "future_states": [B, K, maxT, D] (optional)

        "mask":          [B, maxT]   # 1 for real nodes, 0 for padded
    }
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from config.constraint_registry import CONSTRAINTS


class RoutingCollator:
    def __init__(self):
        pass

    def __call__(self, batch_list):
        """
        batch_list: list of samples from RoutingDataset.__getitem__()

        Build padded batch dict.
        """

        B = len(batch_list)
        T_list = [b["node_features"].shape[0] for b in batch_list]
        maxT = max(T_list)

        # ------------------------------------------
        # Allocate padded tensors
        # ------------------------------------------

        # Find dims
        F = batch_list[0]["node_features"].shape[-1]
        D = batch_list[0]["targets"].shape[-1]

        # Node mask: 1 for valid nodes
        mask = torch.zeros(B, maxT)

        node_features = torch.zeros(B, maxT, F)
        adj_matrix    = torch.zeros(B, maxT, maxT)
        edge_types    = torch.zeros(B, maxT, maxT, dtype=torch.long)
        sp_dist       = torch.zeros(B, maxT, maxT)
        centrality    = torch.zeros(B, maxT)
        targets       = torch.zeros(B, maxT, D)

        # Constraint targets (variable dims)
        constraint_targets = {
            name: []
            for name in CONSTRAINTS
        }

        # Future states (optional)
        has_future = any(("future_states" in b) for b in batch_list)
        future_states = None
        if has_future:
            # K is steps dimension
            K = batch_list[0]["future_states"].shape[0]
            future_states = torch.zeros(B, K, maxT, D)

        # ------------------------------------------
        # Fill padded batch
        # ------------------------------------------
        for i, sample in enumerate(batch_list):
            T = sample["node_features"].shape[0]

            # Mask
            mask[i, :T] = 1

            # Main tensors
            node_features[i, :T] = sample["node_features"]
            adj_matrix[i, :T, :T] = sample["adj_matrix"]
            edge_types[i, :T, :T] = sample["edge_types"]
            sp_dist[i, :T, :T] = sample["sp_dist"]
            centrality[i, :T] = sample["centrality"]
            targets[i, :T] = sample["targets"]

            # Constraint targets
            for name, spec in CONSTRAINTS.items():
                tgt = sample[name]     # [T, out_dim]
                out_dim = tgt.shape[-1] if tgt.dim() == 2 else 1

                # Expand list for this batch
                if len(constraint_targets[name]) == 0:
                    # First time init padded tensor
                    constraint_targets[name] = torch.zeros(B, maxT, out_dim)

                constraint_targets[name][i, :T] = tgt

            # Optional world model targets
            if has_future and "future_states" in sample:
                future_states[i, :, :T] = sample["future_states"]

        # ------------------------------------------
        # Build final batch dict
        # ------------------------------------------
        batch_out = {
            "node_features": node_features,
            "adj_matrix": adj_matrix,
            "edge_types": edge_types,
            "sp_dist": sp_dist,
            "centrality": centrality,

            "targets": targets,
            "mask": mask,
        }

        # Add constraint targets
        for name in CONSTRAINTS:
            batch_out[name] = constraint_targets[name]

        if has_future:
            batch_out["future_states"] = future_states

        return batch_out
