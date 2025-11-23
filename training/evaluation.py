"""
evaluation.py
--------------

Evaluation utilities for RoutingAGI.

Computes:
  - Main regression metrics
  - Constraint-specific metrics
  - World model rollout metrics (optional)
  - Pretty printed reports

This file expects:
    model: RoutingAGI
    dataloader: validation or test loader
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from config.constraint_registry import CONSTRAINTS


# ============================================================
# Utility metrics
# ============================================================

def mse(pred, gt):
    return ((pred - gt) ** 2).mean().item()


def mae(pred, gt):
    return (pred - gt).abs().mean().item()


def bce_accuracy(logits, targets):
    """Binary accuracy for legality / capacity flags."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == targets).float().mean().item()


def ce_accuracy(logits, targets):
    """Accuracy for region rules (multi-class)."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


# ============================================================
# Evaluation Loop
# ============================================================

def evaluate(model, dataloader, cfg, device="cuda"):
    model.eval()

    metrics = {
        "main_mse": [],
        "main_mae": [],
        "constraints": {name: [] for name in CONSTRAINTS},
        "constraint_acc": {name: [] for name in CONSTRAINTS},
        "world_mse": [],
    }

    pbar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Forward pass
            y_pred, aux_all = model(batch)
            aux = aux_all[-1]        # last decoder layer

            # --------------------------------------------------
            # MAIN METRICS
            # --------------------------------------------------
            gt = batch["targets"]
            metrics["main_mse"].append(mse(y_pred, gt))
            metrics["main_mae"].append(mae(y_pred, gt))

            # --------------------------------------------------
            # CONSTRAINT METRICS
            # --------------------------------------------------
            for name, spec in CONSTRAINTS.items():
                out = aux[name]      # predicted
                g = batch[name]      # ground truth

                # Regression constraints
                if spec["loss"] == "mse":
                    metrics["constraints"][name].append(mse(out, g))

                # Binary constraints
                elif spec["loss"] == "bce":
                    acc = bce_accuracy(out, g)
                    metrics["constraint_acc"][name].append(acc)

                # Categorical constraints
                elif spec["loss"] == "ce":
                    # CE expects [B*T,C] logits and [B*T] labels
                    B, T, C = out.shape
                    logits = out.reshape(B * T, C)
                    labels = g.reshape(B * T)
                    acc = ce_accuracy(logits, labels)
                    metrics["constraint_acc"][name].append(acc)

            # --------------------------------------------------
            # WORLD MODEL METRICS
            # --------------------------------------------------
            if cfg["world_model"]["enabled"] and "future_states" in batch:
                from modules.world_model import WorldModel

                if not hasattr(model, "world_model"):
                    model.world_model = WorldModel(
                        dim=cfg["model_dim"],
                        num_layers=cfg["world_model"]["num_layers"],
                        hidden_dim=cfg["world_model"]["hidden_dim"]
                    ).to(device)

                rollout = model.world_model(
                    y_pred, steps=cfg["world_model"]["rollout_steps"]
                )

                gt_future = batch["future_states"]
                wm_mse = mse(rollout, gt_future)
                metrics["world_mse"].append(wm_mse)

    # Collapse lists → averages
    final = {
        "main_mse": sum(metrics["main_mse"]) / len(metrics["main_mse"]),
        "main_mae": sum(metrics["main_mae"]) / len(metrics["main_mae"]),
        "constraints": {},
        "constraint_acc": {},
    }

    # Constraint losses
    for name in CONSTRAINTS:
        if len(metrics["constraints"][name]) > 0:
            final["constraints"][name] = (
                sum(metrics["constraints"][name]) / len(metrics["constraints"][name])
            )
        if len(metrics["constraint_acc"][name]) > 0:
            final["constraint_acc"][name] = (
                sum(metrics["constraint_acc"][name]) / len(metrics["constraint_acc"][name])
            )

    # World model
    if len(metrics["world_mse"]) > 0:
        final["world_mse"] = sum(metrics["world_mse"]) / len(metrics["world_mse"])

    # Pretty print
    print("\n====== Evaluation Report ======")
    print(f"Main MSE: {final['main_mse']:.4f}")
    print(f"Main MAE: {final['main_mae']:.4f}")

    print("\nConstraint Losses:")
    for name, value in final["constraints"].items():
        print(f"  {name:15s}  Loss = {value:.4f}")

    print("\nConstraint Accuracies:")
    for name, value in final["constraint_acc"].items():
        print(f"  {name:15s}  Acc = {value:.4f}")

    if "world_mse" in final:
        print(f"\nWorld Model MSE: {final['world_mse']:.4f}")

    print("================================\n")

    return final
