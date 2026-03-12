from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """多分类 Focal Loss，适合类别不均衡。"""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            loss = self.alpha * loss

        return loss.mean()


def create_loss(loss_name: str, label_smoothing: float = 0.0, focal_gamma: float = 2.0) -> nn.Module:
    name = loss_name.lower().strip()
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma)
    raise ValueError(f"Unknown loss: {loss_name}")


def create_optimizer(
    optimizer_name: str,
    params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower().strip()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
):
    name = scheduler_name.lower().strip()
    if name == "none":
        return None, False
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False
    if name == "onecycle":
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=0.2,
                anneal_strategy="cos",
            ),
            True,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, math.ceil(epochs * 0.1)),
        ), False
    raise ValueError(f"Unknown scheduler: {scheduler_name}")
