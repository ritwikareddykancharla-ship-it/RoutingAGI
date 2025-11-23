"""
dataset_builder.py
-------------------

Turns raw middle-mile routing graph samples into model-ready batches.

Pipeline:
    raw graph + MILP labels
        -> node features
        -> adjacency + edge types
        -> shortest path distances
        -> centrality scores
        -> MILP constraint targets
        -> final dict { node_features, edge_types, sp_dist, centrality, targets, ... }

This dataset is fully compatible with:
    - GraphormerEncoder
    - RoutingAGI model
    - training loop / evaluation utils
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import networkx as nx

from data.milp_targets import compute_all_targets


# ============================================================
# Helpers
# ============================================================

def compute_shortest_path_matrix(G, T):
    """All-pairs shortest-path distances."""
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    dist = torch.full((T, T), 10_000.0)  # big number

    for i in range(T):
        for j in range(T):
            if j in lengths.get(i, {}):
                dist[i, j] = lengths[i][j]

    return dist.clamp(max=20)   # Graphormer max_dist


def compute_adjacency_matrix(G, T):
    """Adjacency matrix [T,T] binary."""
    adj = torch.zeros(T, T)
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def compute_edge_type_matrix(G, T):
    """
    Edge type matrix [T,T].
    lane_type stored as integer IDs.
    """
    edge_types = torch.zeros(T, T, dtype=torch.long)
    for u, v, data in G.edges(data=True):
        lane_type = data.get("lane_type", 0)
        edge_types[u, v] = lane_type
        edge_types[v, u] = lane_type
    return edge_types


def compute_centrality(G, T):
    """Degree centrality normalized [0,1]."""
    c_dict = nx.degree_centrality(G)
    central = torch.zeros(T)
    for node, c in c_dict.items():
        central[node] = c
    return central


# ============================================================
# DATASET CLASS
# ============================================================

class RoutingDataset(Dataset):
    """
    Each sample is one routing graph snapshot:
        {
           "G": networkx graph
           "node_features": [T,F]
           "milp_labels": dict of constraint raw labels
           "targets": [T,D]   (optional)
           "future_states": [K,T,D]   (optional)
        }
    """

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        G = sample["G"]
        T = G.number_of_nodes()

        # -----------------------------
        # NODE FEATURES
        # -----------------------------
        node_feats = torch.tensor(sample["node_features"]).float()  # [T,F]

        # -----------------------------
        # GRAPH MATRICES
        # -----------------------------
        adj = compute_adjacency_matrix(G, T)          # [T,T]
        sp_dist = compute_shortest_path_matrix(G, T)  # [T,T]
        edge_types = compute_edge_type_matrix(G, T)   # [T,T]
        centrality = compute_centrality(G, T)         # [T]

        # -----------------------------
        # MILP CONSTRAINT TARGETS
        # -----------------------------
        milp_targets = compute_all_targets(sample["milp_labels"], T)
        # returns a dict:
        # { "capacity": [T,1], "time_window": [T,1], ... }

        # -----------------------------
        # MAIN SUPERVISION TARGET
        # -----------------------------
        if "targets" in sample:
            targets = torch.tensor(sample["targets"]).float()  # [T,D]
        else:
            # placeholder if no target provided
            targets = torch.zeros(T, 32)

        # -----------------------------
        # WORLD MODEL FUTURE STATES
        # -----------------------------
        future_states = None
        if "future_states" in sample:
            future_states = torch.tensor(sample["future_states"]).float()  # [K,T,D]

        # -----------------------------
        # FINAL BATCH STRUCT
        # -----------------------------
        batch = {
            "node_features": node_feats,
            "adj_matrix": adj,
            "edge_types": edge_types,
            "sp_dist": sp_dist,
            "centrality": centrality,
            "targets": targets,
        }

        # add all MILP constraint targets
        batch.update(milp_targets)

        if future_states is not None:
            batch["future_states"] = future_states

        return batch
