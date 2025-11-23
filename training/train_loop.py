"""
train_loop.py
--------------

Clean, production-ready PyTorch training loop for RoutingAGI.

Handles:
  - batching
  - forward pass
  - main loss
  - constraint (MoE aux) losses
  - world model rollout loss
  - gradient clipping
  - logging

This file expects:
    model: RoutingAGI
    dataloader: yields batches from dataset_builder.py
    optimizer: built by optimizer.py
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from config.constraint_registry import CONSTRAINTS


class RoutingLoss(nn.Module):
    """
    Computes:
        - main supervised loss
        - constraint-specific losses using aux outputs
        - optional world-model rollout loss
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.loss_weights = cfg["loss_weights"]

        # Main task loss (regression)
        self.main_loss_fn = nn.MSELoss()

        # Each constraint gets its own loss head based on type:
        self.constraint_losses = {}
        for name, spec in CONSTRAINTS.items():
            if spec["loss"] == "mse":
                self.constraint_losses[name] = nn.MSELoss()
            elif spec["loss"] == "bce":
                self.constraint_losses[name] = nn.BCEWithLogitsLoss()
            elif spec["loss"] == "ce":
                self.constraint_losses[name] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type for {name}: {spec['loss']}")

    def forward(self, batch, y_pred, aux_all, rollout=None):
        """
        batch:
            - "targets": [B,T,D] (main task)
            - constraint targets stored as:
                - "capacity"
                - "time_window"
                - ...
            - optional: "future_states" for world model comparison

        y_pred:   [B,T,D]
        aux_all:  list of dicts{constraint_name: [B,T,out_dim]}
                  one dict per decoder layer

        rollout:  [B,K,T,D] optional world model output
        """

        total_loss = 0.0
        logs = {}

        # ---------------------------------------------------------
        # 1) Main task loss (regression)
        # ---------------------------------------------------------
        main_gt = batch["targets"]            # [B,T,D]
        main_pred = y_pred                    # [B,T,D]

        main_loss = self.main_loss_fn(main_pred, main_gt)
        logs["main_loss"] = main_loss.item()
        total_loss += main_loss

        # ---------------------------------------------------------
        # 2) Constraint losses (MoE aux)
        # ---------------------------------------------------------
        # Use only the *last* decoder layer's aux outputs (cleanest)
        aux_last = aux_all[-1]

        for name, aux_pred in aux_last.items():
            gt = batch[name]  # shape must match output_dim

            # BCE expect shape [B,T,1]; CE expect [B*T,classes]
            if CONSTRAINTS[name]["loss"] == "ce":
                B, T, C = aux_pred.shape
                aux_loss = self.constraint_losses[name](
                    aux_pred.reshape(B * T, C),
                    gt.reshape(B * T)
                )
            else:
                aux_loss = self.constraint_losses[name](aux_pred, gt)

            w = self.loss_weights.get(name, 1.0)
            total_loss += w * aux_loss
            logs[f"{name}_loss"] = aux_loss.item()

        # ---------------------------------------------------------
        # 3) World Model rollout loss (optional)
        # ---------------------------------------------------------
        if rollout is not None and "future_states" in batch:
            # rollout: [B,K,T,D]
            # gt_future: [B,K,T,D]
            gt_future = batch["future_states"]
            wm_loss = nn.MSELoss()(rollout, gt_future)
            logs["world_loss"] = wm_loss.item()
            total_loss += wm_loss

        return total_loss, logs


# ======================================================================
# Training Loop
# ======================================================================

def train(model, dataloader, optimizer, cfg, device="cuda"):
    model.train()
    criterion = RoutingLoss(cfg)

    for epoch in range(cfg["train"]["epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)

            optimizer.zero_grad()

            # -----------------------------------------------------
            # Forward pass
            # -----------------------------------------------------
            y_pred, aux_all = model(batch)

            # Optional world model
            rollout = None
            if cfg["world_model"]["enabled"]:
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

            # -----------------------------------------------------
            # Compute loss
            # -----------------------------------------------------
            loss, logs = criterion(batch, y_pred, aux_all, rollout)

            # -----------------------------------------------------
            # Backward pass
            # -----------------------------------------------------
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train"]["clip_grad"]
            )

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} completed | avg loss: {epoch_loss/len(dataloader):.4f}")

    print("Training Finished ✨")
