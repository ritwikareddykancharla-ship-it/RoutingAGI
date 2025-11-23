"""
model_config.py
----------------

Central place to store all model hyperparameters.

RoutingAGI loads this file to construct:
    - GraphEncoder
    - Decoder Blocks (Mamba + MoE)
    - World Model
    - Training hyperparams

Everything is clean, structured, and extendable.
"""

MODEL_CONFIG = {

    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================

    "model_dim": 256,              # hidden size D
    "depth": 8,                    # number of decoder blocks
    "node_feature_dim": 32,        # dimension F of input node features
    "use_ffn": True,               # whether DecoderBlock has FFN after MoE

    # Encoder choice: "graphormer" or "gnn"
    "encoder_type": "graphormer",

    # Graphormer settings
    "encoder": {
        "num_layers": 4,
        "num_heads": 8,
        "num_edge_types": 16,
        "max_dist": 20,
    },

    # ============================================================
    # WORLD MODEL SETTINGS
    # ============================================================

    "world_model": {
        "enabled": True,
        "num_layers": 2,
        "hidden_dim": None,     # defaults to 4 * D
        "rollout_steps": 3,     # predict 3 steps into the future
    },

    # ============================================================
    # TRAINING & OPTIMIZATION
    # ============================================================

    "train": {
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "clip_grad": 1.0,
        "epochs": 30,
    },

    # Constraint loss weighting
    "loss_weights": {
        "capacity": 1.0,
        "time_window": 1.0,
        "lane_legality": 2.0,
        "throughput": 1.0,
        "sla": 1.0,
        "flow": 0.5,
        "mode": 1.0,
        "region": 0.5,
        "load_balance": 1.0,
        "trailer": 1.0,
    },

    # ============================================================
    # MAMBA SETTINGS
    # ============================================================

    "mamba": {
        "dt_rank": 16,
    },
}
