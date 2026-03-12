from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """多分类 Focal Loss，适合类别不均衡。"""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | float | None = None):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        elif alpha is not None:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            if self.alpha.ndim == 0:
                loss = self.alpha * loss
            else:
                loss = self.alpha[targets] * loss

        return loss.mean()


def build_class_weights(
    counts: Sequence[int],
    strategy: str = "none",
    effective_num_beta: float = 0.999,
) -> torch.Tensor | None:
    name = strategy.lower().strip()
    if name == "none":
        return None

    counts_t = torch.tensor([max(1, int(c)) for c in counts], dtype=torch.float32)

    if name == "inverse":
        weights = 1.0 / counts_t
    elif name == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(counts_t)
    elif name == "effective_num":
        beta = min(max(effective_num_beta, 0.0), 0.99999)
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts_t)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    weights = weights / weights.sum() * len(counts)
    return weights


def create_loss(
    loss_name: str,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    name = loss_name.lower().strip()
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
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
