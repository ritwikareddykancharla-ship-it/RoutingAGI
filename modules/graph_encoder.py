"""
GraphEncoder — Graphormer Version
---------------------------------

Graphormer (Ying et al. 2021):
    A transformer encoder with structural encodings for graphs.

Key features implemented:
  - Node features projection
  - Edge type embedding
  - Distance embedding (shortest path distance)
  - Centrality encoding
  - Attention bias injection (Graphormer trick)
  - Multi-head self-attention over nodes

Input structure expected:
    batch["node_features"]: [B, T, F]
    batch["sp_dist"]:       [B, T, T]     # shortest path distances
    batch["edge_types"]:    [B, T, T]     # lane-type IDs (int)
    batch["centrality"]:    [B, T]        # degree or hub centrality

Output:
    [B, T, D] Graphormer-encoded node representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Multi-Head Self Attention with Attention Bias (Graphormer)
# -------------------------------------------------------------
class AttentionWithBias(nn.Module):
    """
    Standard MHSA, but adds a pre-computed attention bias matrix
    to the attention logits before softmax.

    Shapes:
        x:         [B, T, D]
        attn_bias: [B, H, T, T]

    Returns:
        out:       [B, T, D]
    """

    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_bias):
        B, T, D = x.shape
        H = self.num_heads

        q = self.q_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale  # [B,H,T,T]

        # Add Graphormer structural bias
        if attn_bias is not None:
            scores = scores + attn_bias  # [B,H,T,T]

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(out)


# -------------------------------------------------------------
# Graphormer Layer
# -------------------------------------------------------------
class GraphormerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionWithBias(dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        # Attention block
        h = self.norm1(x)
        h = self.attn(h, attn_bias)
        x = x + self.dropout1(h)

        # FFN block
        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout2(h2)

        return x


# -------------------------------------------------------------
# Graphormer Encoder — Full Stack
# -------------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        num_edge_types: int = 16,
        max_dist: int = 20,
    ):
        super().__init__()

        self.node_proj = nn.Linear(in_dim, hidden_dim)

        # Structural encodings
        self.edge_emb = nn.Embedding(num_edge_types, num_heads)
        self.dist_emb = nn.Embedding(max_dist + 1, num_heads)
        self.centrality_emb = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    # ---------------------------------------------------------
    # Compute attention bias matrix
    # ---------------------------------------------------------
    def compute_attn_bias(self, batch, num_heads, device):
        """
        Construct Graphormer attention bias:
            attn_bias[b,h,i,j] =
                  edge_type_embedding[i,j,h] +
                  dist_embedding[i,j,h]

        Shapes:
            edge_types: [B, T, T] ints
            sp_dist:    [B, T, T] ints
        """
        B, T = batch["edge_types"].shape[:2]

        edge_types = batch["edge_types"].to(device)      # [B,T,T]
        sp_dist = batch["sp_dist"].to(device).clamp(max=self.dist_emb.num_embeddings - 1)

        # edge bias: [B, T, T, H]
        edge_bias = self.edge_emb(edge_types)  # Embedding → [B,T,T,H]

        # dist bias: [B, T, T, H]
        dist_bias = self.dist_emb(sp_dist)     # Embedding → [B,T,T,H]

        # Combine & reshape for MH-attn:
        # → [B, H, T, T]
        attn_bias = (edge_bias + dist_bias).permute(0, 3, 1, 2)

        return attn_bias

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, batch):
        """
        Expected fields in batch:
            node_features: [B, T, F]
            edge_types:    [B, T, T]
            sp_dist:       [B, T, T]
            centrality:    [B, T]

        Returns:
            h: [B, T, D]
        """

        x = batch["node_features"]     # [B,T,F]
        B, T, _ = x.shape
        device = x.device

        h = self.node_proj(x)          # [B,T,D]

        # Centrality embedding
        if "centrality" in batch:
            central = batch["centrality"].unsqueeze(-1).float()  # [B,T,1]
            h = h + self.centrality_emb(central)

        # Structural attention bias
        if "edge_types" in batch and "sp_dist" in batch:
            attn_bias = self.compute_attn_bias(batch, num_heads=self.layers[0].attn.num_heads, device=device)
        else:
            attn_bias = None

        # Stack of Graphormer layers
        for layer in self.layers:
            h = layer(h, attn_bias)

        return h
