"""
main.py
-------

RoutingAGI Full Training Script.

This script:
    - loads config
    - builds dataset + dataloader
    - builds RoutingAGI model
    - builds optimizer
    - runs training loop
    - runs evaluation

Run:
    python main.py
"""

import torch
from torch.utils.data import DataLoader

# Config
from config.model_config import MODEL_CONFIG

# Dataset
from data.dataset_builder import RoutingDataset
from data.collator import RoutingCollator

# Model
from modules.routing_agi_model import RoutingAGI

# Optimizer
from training.optimizer import build_optimizer

# Training + Eval
from training.train_loop import train
from training.evaluation import evaluate


# ============================================================
#  Synthetic Example Generator (Temporary)
#  You will replace this with real MILP/Graph data later.
# ============================================================

import networkx as nx
import random

def generate_dummy_sample(T=10, F=32, D=32):
    """
    Generates a random graph + random node features + dummy MILP labels.
    This lets the entire pipeline run end-to-end.
    """
    G = nx.random_geometric_graph(T, radius=0.5)

    # Add lane types
    for u, v in G.edges():
        G[u][v]["lane_type"] = random.randint(0, 3)

    # Fake node features
    node_features = torch.randn(T, F).tolist()

    # Fake MILP labels (you will replace with real constraints)
    milp_labels = {
        "capacity": torch.rand(T).tolist(),
        "time_window": torch.rand(T).tolist(),
        "lane_legality": torch.randint(0, 2, (T,)).tolist(),
        "throughput": torch.rand(T).tolist(),
        "sla": torch.rand(T).tolist(),
        "flow": torch.rand(T).tolist(),
        "mode": torch.randint(0, 2, (T,)).tolist(),
        "region": torch.randint(0, 2, (T,)).tolist(),
        "load_balance": torch.rand(T).tolist(),
        "trailer": torch.rand(T).tolist(),
    }

    # Fake main target (e.g., routing embedding)
    targets = torch.randn(T, D).tolist()

    # Fake future states for world model
    future_states = torch.randn(3, T, D).tolist()  # K=3

    return {
        "G": G,
        "node_features": node_features,
        "milp_labels": milp_labels,
        "targets": targets,
        "future_states": future_states,
    }


# ============================================================
#  MAIN FUNCTION
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    cfg = MODEL_CONFIG

    # --------------------------------------------------------
    # 1. Build Dataset
    # --------------------------------------------------------
    print("💛 Creating synthetic dataset (replace with real)…")
    samples = [generate_dummy_sample() for _ in range(128)]  # 128 samples

    dataset = RoutingDataset(samples)
    collator = RoutingCollator()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    # --------------------------------------------------------
    # 2. Build Model
    # --------------------------------------------------------
    print("💙 Building RoutingAGI Model…")
    model = RoutingAGI(cfg).to(device)

    # --------------------------------------------------------
    # 3. Optimizer
    # --------------------------------------------------------
    optimizer = build_optimizer(model, cfg)

    # --------------------------------------------------------
    # 4. Train
    # --------------------------------------------------------
    print("❤️ Training RoutingAGI…")
    train(model, dataloader, optimizer, cfg, device)

    # --------------------------------------------------------
    # 5. Evaluate
    # --------------------------------------------------------
    print("💜 Evaluating RoutingAGI…")
    eval_metrics = evaluate(model, dataloader, cfg, device)

    print("\n🎉 DONE! Results:")
    print(eval_metrics)


if __name__ == "__main__":
    main()
