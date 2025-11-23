"""
RoutingAGI: Full Model Assembly
-------------------------------

This model wires together:
    - GraphEncoder (node → D dim)
    - N DecoderBlocks (Mamba + Constraint-MoE)
    - Optional WorldModel (future extension)
    - Final output projection

Outputs:
    y_final: [B, T, D]
    aux_all: list of constraint dicts (one per decoder layer)

"""

import torch
import torch.nn as nn

from modules.graph_encoder import GraphEncoder
from modules.decoder_block import DecoderBlock


class RoutingAGI(nn.Module):
    """
    Full RoutingAGI model.

    Args:
        cfg: dict with keys:
            - model_dim: hidden size
            - depth: number of decoder layers
            - node_feature_dim: input node feature dimension
            - use_ffn: whether to include FFN after MoE in each block
    """

    def __init__(self, cfg):
        super().__init__()

        self.model_dim = cfg.get("model_dim", 256)
        self.depth = cfg.get("depth", 8)
        self.node_feature_dim = cfg.get("node_feature_dim", 32)
        self.use_ffn = cfg.get("use_ffn", False)

        # --------------------------------------------------------------
        # Encoder: turn raw graph/node features → model_dim embeddings
        # --------------------------------------------------------------
        self.encoder = GraphEncoder(
            in_dim=self.node_feature_dim,
            hidden_dim=self.model_dim
        )

        # --------------------------------------------------------------
        # Decoder: stack of Mamba + Constraint-MoE blocks
        # --------------------------------------------------------------
        self.blocks = nn.ModuleList([
            DecoderBlock(self.model_dim, use_ffn=self.use_ffn)
            for _ in range(self.depth)
        ])

        # --------------------------------------------------------------
        # Final projection head
        # --------------------------------------------------------------
        self.final = nn.Linear(self.model_dim, self.model_dim)

    # ==================================================================
    # FORWARD
    # ==================================================================
    def forward(self, batch):
        """
        batch:
            {
                "node_features": [B, T, F],
                ... (other graph metadata if needed)
            }

        Returns:
            y_final: [B, T, D]
            aux_all: list (len=depth) of:
                     dict {constraint_name: [B,T,out_dim]}
        """

        # 1) Encode graph/node features into hidden states
        h = self.encoder(batch)     # [B, T, D]
        aux_all = []

        # 2) Pass through each decoder block
        for block in self.blocks:
            h, aux = block(h)       # h: [B,T,D], aux: dict
            aux_all.append(aux)

        # 3) Final head
        y_final = self.final(h)     # [B, T, D]

        return y_final, aux_all


# ======================================================================
# SMALL TEST
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = {
        "model_dim": 128,
        "depth": 3,
        "node_feature_dim": 32,
        "use_ffn": True,
    }

    model = RoutingAGI(cfg)
    B, T, F = 4, 10, 32

    dummy = {
        "node_features": torch.randn(B, T, F)
    }

    y, aux_all = model(dummy)

    print("y shape:", y.shape)
    print("num decoder layers:", len(aux_all))
    print("constraints in layer 0:", aux_all[0].keys())
