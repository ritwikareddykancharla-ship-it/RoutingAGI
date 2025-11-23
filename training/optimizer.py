"""
optimizer.py
------------

Utility for building optimizer & (future) learning rate scheduler
from model_config settings.

Keeps training loop clean and makes optimizer swaps trivial.
"""

import torch
from torch.optim import AdamW, SGD, RMSprop


# Optional Lion optimizer (if you want it, otherwise comment out)
try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False


def build_optimizer(model, cfg):
    """
    Build the optimizer from the config dict.

    cfg["train"] fields:
        - lr
        - weight_decay
        - optimizer (optional): "adamw" | "lion" | "sgd" | "rmsprop"

    Returns:
        optimizer: torch.optim.Optimizer
    """

    train_cfg = cfg.get("train", {})

    lr = train_cfg.get("lr", 3e-4)
    wd = train_cfg.get("weight_decay", 0.01)
    opt_name = train_cfg.get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )

    elif opt_name == "lion":
        if not HAS_LION:
            raise ImportError(
                "Lion optimizer not installed. Run: pip install lion-pytorch"
            )
        optimizer = Lion(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )

    elif opt_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=train_cfg.get("momentum", 0.9)
        )

    elif opt_name == "rmsprop":
        optimizer = RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=train_cfg.get("momentum", 0.9)
        )

    else:
        raise ValueError(f"Unknown optimizer type: {opt_name}")

    return optimizer
