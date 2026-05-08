from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _build_cross_entropy_loss(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    kwargs = {"weight": class_weights}
    if label_smoothing > 0:
        try:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing, **kwargs)
        except TypeError:
            pass
    return nn.CrossEntropyLoss(**kwargs)


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


class ClassSpecificFocalLoss(nn.Module):
    """为不同类别设置不同 focal gamma 的 Focal Loss。

    对难分类的类别（如良性）使用更大的 gamma，更聚焦难样本；
    对容易分类的类别使用较小的 gamma。
    """

    def __init__(
        self,
        gammas: Sequence[float] | None = None,
        alpha: torch.Tensor | None = None,
    ):
        super().__init__()
        if gammas is None:
            gammas = [2.0]
        gammas_t = torch.tensor(gammas, dtype=torch.float32)
        self.register_buffer("gammas", gammas_t)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 每个样本使用其对应类别的 gamma
        per_sample_gamma = self.gammas[targets]
        focal_factor = (1 - target_probs).pow(per_sample_gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            if self.alpha.ndim == 0:
                loss = self.alpha * loss
            else:
                loss = self.alpha[targets] * loss

        return loss.mean()


class MaskAwareClassificationLoss(nn.Module):
    """Mask-aware 分类损失。

    总损失 = CE(logits, y)
           + mask_loss_weight * CE(masked_logits, y)
           + consistency_weight * KL(masked || full)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        mask_loss_weight: float = 0.5,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.ce = _build_cross_entropy_loss(class_weights=class_weights, label_smoothing=label_smoothing)
        self.mask_loss_weight = mask_loss_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masked_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_loss = self.ce(logits, targets)
        if masked_logits is None:
            return base_loss

        mask_ce = self.ce(masked_logits, targets)
        kl = F.kl_div(
            F.log_softmax(masked_logits, dim=1),
            F.softmax(logits.detach(), dim=1),
            reduction="batchmean",
        )
        return base_loss + self.mask_loss_weight * mask_ce + self.consistency_weight * kl


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


class HierarchicalMultiTaskLoss(nn.Module):
    """层次化多任务损失。

    受 Subregion-Unet 思想启发，通过类别分组并行训练三个任务：
      - Task A: 健康(0) vs 非健康(1,2) — 所有样本参与
      - Task B: 三分类(0,1,2)         — 所有样本参与（主任务）
      - Task C: 良性(1) vs 恶性(2)    — 仅非健康样本参与

    模型 forward 训练时返回 (logits_main, logits_abnormal, logits_bm)。
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        weight_abnormal: float = 1.0,
        weight_main: float = 1.0,
        weight_benign_malignant: float = 1.0,
        focal_gammas: Sequence[float] | None = None,
    ):
        super().__init__()
        if focal_gammas is not None:
            self.ce_main = ClassSpecificFocalLoss(gammas=focal_gammas, alpha=class_weights)
        else:
            self.ce_main = _build_cross_entropy_loss(class_weights=class_weights, label_smoothing=label_smoothing)
        self.ce_abnormal = _build_cross_entropy_loss(label_smoothing=label_smoothing)
        self.ce_bm = _build_cross_entropy_loss(label_smoothing=label_smoothing)
        self.w_abnormal = weight_abnormal
        self.w_main = weight_main
        self.w_bm = weight_benign_malignant

    def forward(
        self,
        model_output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(model_output, (tuple, list)):
            logits_main, logits_abnormal, logits_bm = model_output
        else:
            return self.ce_main(model_output, targets)

        # Task B: 三分类（主损失）
        loss_main = self.ce_main(logits_main, targets)

        # Task A: 健康 vs 非健康
        targets_abnormal = (targets > 0).long()
        loss_abnormal = self.ce_abnormal(logits_abnormal, targets_abnormal)

        # Task C: 良性 vs 恶性（仅非健康样本）
        non_healthy_mask = targets > 0
        loss_bm = torch.tensor(0.0, device=logits_main.device)
        if non_healthy_mask.any():
            sub_logits = logits_bm[non_healthy_mask]
            sub_targets = targets[non_healthy_mask] - 1  # 1→0, 2→1
            loss_bm = self.ce_bm(sub_logits, sub_targets)

        return (self.w_abnormal * loss_abnormal
                + self.w_main * loss_main
                + self.w_bm * loss_bm)


def create_loss(
    loss_name: str,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    mask_loss_weight: float = 0.5,
    consistency_weight: float = 0.1,
    weight_abnormal: float = 1.0,
    weight_main: float = 1.0,
    weight_benign_malignant: float = 1.0,
    focal_gammas: Sequence[float] | None = None,
) -> nn.Module:
    name = loss_name.lower().strip()
    if name == "ce":
        return _build_cross_entropy_loss(class_weights=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    if name == "class_specific_focal":
        return ClassSpecificFocalLoss(gammas=focal_gammas, alpha=class_weights)
    if name == "mask_aware":
        return MaskAwareClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            mask_loss_weight=mask_loss_weight,
            consistency_weight=consistency_weight,
        )
    if name == "hierarchical":
        return HierarchicalMultiTaskLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            weight_abnormal=weight_abnormal,
            weight_main=weight_main,
            weight_benign_malignant=weight_benign_malignant,
        )
    if name == "hierarchical_class_specific_focal":
        return HierarchicalMultiTaskLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            weight_abnormal=weight_abnormal,
            weight_main=weight_main,
            weight_benign_malignant=weight_benign_malignant,
            focal_gammas=focal_gammas,
        )
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
