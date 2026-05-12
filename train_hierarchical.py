"""层次化多任务分类训练入口。

受 Subregion-Unet 思想启发，通过类别分组并行训练：
  - Head A: 健康(0) vs 非健康(1,2)
  - Head B: 标准三分类(0,1,2) — 主输出
  - Head C: 良性(1) vs 恶性(2) — 仅非健康样本

附加功能：
  - 良性-恶性 Mixup (--bm-mixup): 在良性与恶性样本之间做 Beta 混合
  - 混淆感知微调 (--ca-ft): 先正常训练，再根据混淆矩阵对难分类别加权微调
  - Class-Specific Focal Loss (--focal-gammas): 为不同类别设置不同 focal gamma

用法:
  # 基础多任务训练
  python train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root <CT_ROOT> \
    --metadata-csv <METADATA_CSV> \
    --ct-root <CT_ROOT> \
    --output-dir outputs/hierarchical_resnet3d18_mc \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --scheduler cosine \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --selection-metric balanced_accuracy

  # 多任务 + 良性-恶性 Mixup
  python train_hierarchical.py \
    ... (同上) \
    --bm-mixup \
    --bm-mixup-alpha 0.4

  # 多任务 + 混淆感知微调
  python train_hierarchical.py \
    ... (同上) \
    --ca-ft \
    --ca-ft-start 24 \
    --ca-ft-boost 2.0

  # 多任务 + Class-Specific Focal Loss (良性 gamma=3.0)
  python train_hierarchical.py \
    ... (同上) \
    --loss hierarchical \
    --focal-gammas 2.0 3.0 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from lung_cancer_cls.dataset import (
        DatasetType,
        create_dataset,
        get_default_transforms,
        get_default_volume_transforms,
        IntranetCTDataset,
    )
    from lung_cancer_cls.model import HierarchicalMultiTaskClassifier
    from lung_cancer_cls.train import (
        build_epoch_log,
        compute_classification_metrics,
        remap_samples_by_class_mode,
        resolve_selection_metric,
        resolve_selection_score,
        set_seed,
        stratified_split,
        stratified_train_val_split,
    )
    from lung_cancer_cls.training_components import (
        HierarchicalMultiTaskLoss,
        build_class_weights,
        create_optimizer,
        create_scheduler,
    )
except ModuleNotFoundError:
    # Fallback for running this script directly from repository root (src/ layout).
    REPO_ROOT = Path(__file__).resolve().parent
    SRC_DIR = REPO_ROOT / "src"
    if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from lung_cancer_cls.dataset import (
        DatasetType,
        create_dataset,
        get_default_transforms,
        get_default_volume_transforms,
        IntranetCTDataset,
    )
    from lung_cancer_cls.model import HierarchicalMultiTaskClassifier
    from lung_cancer_cls.train import (
        build_epoch_log,
        compute_classification_metrics,
        remap_samples_by_class_mode,
        resolve_selection_metric,
        resolve_selection_score,
        set_seed,
        stratified_split,
        stratified_train_val_split,
    )
    from lung_cancer_cls.training_components import (
        HierarchicalMultiTaskLoss,
        build_class_weights,
        create_optimizer,
        create_scheduler,
    )


@dataclass
class HierarchicalConfig:
    dataset_type: DatasetType
    data_root: str
    metadata_csv: str | None
    ct_root: str | None
    intranet_source: str
    bundle_nm_path: str | None
    bundle_bn_path: str | None
    bundle_mt_path: str | None
    split_manifest_csv: str | None
    split_fold: int
    output_dir: str
    class_mode: str
    binary_task: str
    split_mode: str
    train_ratio: float
    use_predefined_split: bool
    selection_metric: str
    backbone: str
    pretrained: bool
    use_3d_input: bool
    depth_size: int
    volume_hw: int
    image_size: int
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    aug_profile: str
    sampling_strategy: str
    class_weight_strategy: str
    effective_num_beta: float
    label_smoothing: float
    cpu: bool
    seed: int
    weight_abnormal: float
    weight_main: float
    weight_benign_malignant: float
    eval_mode: str
    bm_mixup: bool
    bm_mixup_alpha: float
    ca_ft: bool
    ca_ft_start: int | None
    ca_ft_boost: float
    focal_gammas: list[float] | None


def _build_config(ns: argparse.Namespace) -> HierarchicalConfig:
    return HierarchicalConfig(**vars(ns))


# ---------------------------------------------------------------------------
# 多任务训练 / 评估
# ---------------------------------------------------------------------------

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unexpected batch format: {type(batch)}, len={len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")


def train_epoch_hierarchical(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)  # (logits_main, logits_abnormal, logits_bm)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
    return running_loss / max(seen, 1)



def _weighted_soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy for probabilistic targets, with optional per-class weights."""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs)
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1).to(logits.device)
    return loss.sum(dim=1).mean()


def hierarchical_bm_mixup_loss(
    model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lam: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Soft-label hierarchical loss for benign-malignant Mixup samples.

    Mixed samples are always non-healthy for Head A, while Head B and Head C
    receive probability targets according to the sampled benign proportion ``lam``.
    """
    logits_main, logits_abnormal, logits_bm = model_output
    lam = lam.to(logits_main.device).float().clamp(0.0, 1.0)
    n_mix = logits_main.size(0)

    main_targets = torch.zeros(n_mix, logits_main.size(1), device=logits_main.device)
    main_targets[:, 1] = lam
    main_targets[:, 2] = 1.0 - lam

    abnormal_targets = torch.zeros(n_mix, 2, device=logits_abnormal.device)
    abnormal_targets[:, 1] = 1.0

    bm_targets = torch.zeros(n_mix, 2, device=logits_bm.device)
    bm_targets[:, 0] = lam
    bm_targets[:, 1] = 1.0 - lam

    ce_main = getattr(criterion, "ce_main", None)
    class_weights = getattr(ce_main, "weight", None)
    loss_main = _weighted_soft_cross_entropy(logits_main, main_targets, class_weights)
    loss_abnormal = _weighted_soft_cross_entropy(logits_abnormal, abnormal_targets)
    loss_bm = _weighted_soft_cross_entropy(logits_bm, bm_targets)

    w_ab = getattr(criterion, "w_abnormal", 1.0)
    w_main = getattr(criterion, "w_main", 1.0)
    w_bm = getattr(criterion, "w_bm", 1.0)
    return w_ab * loss_abnormal + w_main * loss_main + w_bm * loss_bm

def train_epoch_hierarchical_mixup(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    mixup_alpha: float = 0.4,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
) -> float:
    """层次化多任务训练 + 良性-恶性 Mixup。

    每个 batch：
      1. 正常前向 + 多任务损失
      2. 找出良性(1)和恶性(2)样本，随机配对做 Mixup
      3. 混合样本过模型，用 soft target 计算附加损失：
         - Head B 三分类目标为 [0, lam, 1-lam]
         - Head A 始终为非健康
         - Head C 良/恶目标为 [lam, 1-lam]
      4. 两部分损失合并反传
    """
    model.train()
    running_loss = 0.0
    seen = 0
    beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)

    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # ---- 正常多任务损失 ----
        outputs = model(x)
        loss = criterion(outputs, y)

        # ---- 良性-恶性 Mixup ----
        benign_idx = (y == 1).nonzero(as_tuple=True)[0]
        malignant_idx = (y == 2).nonzero(as_tuple=True)[0]
        n_benign = benign_idx.shape[0]
        n_malignant = malignant_idx.shape[0]

        if n_benign > 0 and n_malignant > 0:
            n_mix = min(n_benign, n_malignant)
            # 随机采样配对
            perm_b = benign_idx[torch.randperm(n_benign)[:n_mix]]
            perm_m = malignant_idx[torch.randperm(n_malignant)[:n_mix]]
            lam = beta_dist.sample((n_mix, 1, 1, 1)).to(device)  # [n_mix,1,1,1]
            if x.dim() == 5:
                lam = lam.unsqueeze(-1)  # [n_mix,1,1,1,1] for 3D

            x_mix = lam * x[perm_b] + (1 - lam) * x[perm_m]

            outputs_mix = model(x_mix)
            loss_mix = hierarchical_bm_mixup_loss(outputs_mix, lam.view(-1), criterion)

            loss = loss + loss_mix

        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate_hierarchical(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels, all_probs = [], []
    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        # 推理模式：只用主 head
        logits_main = model(x)
        loss = F.cross_entropy(logits_main, y)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        all_labels.append(y.cpu())
        all_probs.append(torch.softmax(logits_main, dim=1).cpu())
    if total == 0:
        return {"loss": None, "accuracy": None, "num_samples": 0}
    y_np = torch.cat(all_labels).numpy().astype(np.int64)
    p_np = torch.cat(all_probs).numpy()
    return compute_classification_metrics(
        y_np, p_np, loss=total_loss / total, class_names=class_names
    )


# ---------------------------------------------------------------------------
# 三分类推理（cascade: Head A → Head C）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hierarchical_cascade(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """用 Head A + Head C 做级联推理：先判断健康/非健康，再区分良/恶。"""
    model.eval()
    total = 0
    all_labels, all_probs = [], []
    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        feat = model.extract_features(x)
        logits_abnormal = model.head_abnormal(feat)      # [B, 2]
        logits_main = model.head_main(feat)              # [B, 3]
        logits_bm = model.head_benign_malignant(feat)    # [B, 2]

        probs_main = torch.softmax(logits_main, dim=1)   # [B, 3]
        probs_abnormal = torch.softmax(logits_abnormal, dim=1)  # [B, 2]
        probs_bm = torch.softmax(logits_bm, dim=1)       # [B, 2]

        # 级联逻辑：如果 Head A 判定为健康 → class 0，否则用 Head C 区分 1/2
        pred_abnormal = torch.argmax(probs_abnormal, dim=1)  # 0=健康, 1=非健康
        final_probs = torch.zeros(x.size(0), 3, device=device)
        # 健康样本：直接用主 head 的 class 0 概率
        healthy_mask = pred_abnormal == 0
        final_probs[healthy_mask] = probs_main[healthy_mask]
        # 非健康样本：用 Head C 的 1 vs 2 概率替换 class 1/2
        abnormal_mask = pred_abnormal == 1
        if abnormal_mask.any():
            final_probs[abnormal_mask, 0] = 0.0
            final_probs[abnormal_mask, 1] = probs_bm[abnormal_mask, 0]
            final_probs[abnormal_mask, 2] = probs_bm[abnormal_mask, 1]

        # 归一化
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        total += y.size(0)
        all_labels.append(y.cpu())
        all_probs.append(final_probs.cpu())
    if total == 0:
        return {"loss": None, "accuracy": None, "num_samples": 0}
    y_np = torch.cat(all_labels).numpy().astype(np.int64)
    p_np = torch.cat(all_probs).numpy()
    return compute_classification_metrics(y_np, p_np, loss=0.0, class_names=class_names)


# ---------------------------------------------------------------------------
# 混淆感知微调 (Confusion-Aware Fine-tuning)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_confusion_weights(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    boost_factor: float = 2.0,
) -> torch.Tensor:
    """分析模型在验证集上的混淆矩阵，为易混淆类别生成加权权重。

    返回 shape=[num_classes] 的权重张量，混淆越多的类别权重越高。
    """
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # 构建混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(labels_np, preds_np):
        cm[t, p] += 1.0

    # 每个类别的错误率（行归一化后，非对角线之和）
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    error_rates = 1.0 - np.diag(cm) / row_sums.squeeze()

    # 权重 = 1 + boost_factor * 归一化错误率
    max_err = error_rates.max()
    if max_err > 0:
        normalized = error_rates / max_err
    else:
        normalized = error_rates
    weights = 1.0 + boost_factor * normalized
    weights_t = torch.tensor(weights, dtype=torch.float32)

    print(f"  Confusion-aware weights: {[f'{w:.3f}' for w in weights_t.tolist()]}")
    print(f"  Per-class error rates: {[f'{e:.3f}' for e in error_rates.tolist()]}")
    return weights_t


def train_epoch_confusion_aware(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    confusion_weights: torch.Tensor,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
) -> float:
    """混淆感知微调训练 epoch。

    在正常多任务损失基础上，对每个样本额外加权：混淆越多的类别权重越高。
    """
    model.train()
    running_loss = 0.0
    seen = 0
    cw = confusion_weights.to(device)

    for batch in loader:
        x, y, mask = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)

        # 计算每个样本的损失，然后按类别加权
        if isinstance(outputs, (tuple, list)):
            logits_main = outputs[0]
        else:
            logits_main = outputs

        # 逐样本 CE
        per_sample_loss = F.cross_entropy(logits_main, y, reduction="none")
        # 按类别加权
        sample_weights = cw[y]
        loss = (per_sample_loss * sample_weights).mean()

        # 多任务子损失（不加权）
        if isinstance(outputs, (tuple, list)):
            targets_abnormal = (y > 0).long()
            loss_abnormal = F.cross_entropy(outputs[1], targets_abnormal)
            non_healthy_mask = y > 0
            loss_bm = torch.tensor(0.0, device=device)
            if non_healthy_mask.any():
                sub_logits = outputs[2][non_healthy_mask]
                sub_targets = y[non_healthy_mask] - 1
                loss_bm = F.cross_entropy(sub_logits, sub_targets)
            w_ab = getattr(criterion, "w_abnormal", 1.0)
            w_bm = getattr(criterion, "w_bm", 1.0)
            loss = loss + w_ab * loss_abnormal + w_bm * loss_bm

        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
    return running_loss / max(seen, 1)


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------

def _build_dataset(config):
    dataset_kwargs: Dict[str, Any] = {}
    if config.dataset_type == DatasetType.INTRANET_CT:
        if config.metadata_csv:
            dataset_kwargs["metadata_csv"] = Path(config.metadata_csv)
        if config.ct_root:
            dataset_kwargs["ct_root"] = Path(config.ct_root)
        dataset_kwargs["intranet_source"] = config.intranet_source
        if config.bundle_nm_path:
            dataset_kwargs["bundle_nm_path"] = Path(config.bundle_nm_path)
        if config.bundle_bn_path:
            dataset_kwargs["bundle_bn_path"] = Path(config.bundle_bn_path)
        if config.bundle_mt_path:
            dataset_kwargs["bundle_mt_path"] = Path(config.bundle_mt_path)
    if config.use_3d_input:
        dataset_kwargs["use_3d"] = True
        dataset_kwargs["depth_size"] = config.depth_size
        dataset_kwargs["volume_hw"] = config.volume_hw
    if config.split_manifest_csv:
        dataset_kwargs["split_manifest_csv"] = Path(config.split_manifest_csv)
        dataset_kwargs["split_fold"] = config.split_fold
    root = Path(config.data_root)
    return create_dataset(config.dataset_type, root, **dataset_kwargs)




def _parse_dataset_type(dataset_type: str) -> DatasetType:
    mapping = {
        "iqothnccd": DatasetType.IQ_OTHNCCD,
        "luna16": DatasetType.LUNA16,
        "lidc_idri": DatasetType.LIDC_IDRI,
        "intranet_ct": DatasetType.INTRANET_CT,
    }
    key = str(dataset_type).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return mapping[key]
def _build_transforms(config):
    if config.use_3d_input:
        train_tf, val_tf = get_default_volume_transforms(aug_profile=config.aug_profile)
    else:
        train_tf = get_default_transforms(config.image_size, aug_profile=config.aug_profile, is_train=True)
        val_tf = get_default_transforms(config.image_size, aug_profile="basic", is_train=False)
    return train_tf, val_tf






def _build_dataset_from_samples(base_dataset, samples, transform, config):
    kwargs = {"samples": samples, "transform": transform}
    if isinstance(base_dataset, IntranetCTDataset):
        kwargs.update(
            use_3d=config.use_3d_input,
            depth_size=config.depth_size,
            volume_hw=config.volume_hw,
        )
    return type(base_dataset)(**kwargs)

def _resolve_split_indices(samples, config):
    if config.split_mode == "train_val":
        train_idx, val_idx = stratified_train_val_split(samples, train_ratio=config.train_ratio, seed=config.seed)
        test_idx = []
    else:
        train_idx, val_idx, test_idx = stratified_split(samples, train_ratio=config.train_ratio, seed=config.seed)

    if not config.use_predefined_split:
        return train_idx, val_idx, test_idx

    split_to_idx = defaultdict(list)
    for idx, sample in enumerate(samples):
        split = ""
        if sample.metadata is not None:
            split = str(sample.metadata.get("split", "")).lower().strip()
        split_to_idx[split].append(idx)

    predefined_train = split_to_idx.get("train", [])
    predefined_val = split_to_idx.get("val", []) + split_to_idx.get("valid", [])
    predefined_test = split_to_idx.get("test", [])

    if config.split_mode == "train_val":
        if predefined_train and (predefined_val or predefined_test):
            eval_idx = predefined_val + predefined_test
            if predefined_val and predefined_test:
                print("  Using metadata predefined val+test split as validation")
            elif predefined_test:
                print("  Using metadata predefined train/test split as train/val")
            else:
                print("  Using metadata predefined split (train/val)")
            return predefined_train, eval_idx, []
        print("  Predefined split unavailable for train_val; fallback to stratified split")
        return train_idx, val_idx, test_idx

    if predefined_train and (predefined_val or predefined_test):
        if not predefined_val and predefined_test:
            print("  Using metadata predefined train/test split; validation set is empty")
        else:
            print("  Using metadata predefined split")
        return predefined_train, predefined_val, predefined_test

    print("  Predefined split unavailable; fallback to stratified split")
    return train_idx, val_idx, test_idx

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="层次化多任务分类训练")

    # 数据
    parser.add_argument("--dataset-type", type=str, default="intranet_ct",
                        choices=["iqothnccd", "luna16", "lidc_idri", "intranet_ct"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--ct-root", type=str, default=None)
    parser.add_argument("--intranet-source", type=str, default="csv", choices=["csv", "bundle", "both"])
    parser.add_argument("--bundle-nm-path", type=str, default=None)
    parser.add_argument("--bundle-bn-path", type=str, default=None)
    parser.add_argument("--bundle-mt-path", type=str, default=None)
    parser.add_argument("--split-manifest-csv", type=str, default=None)
    parser.add_argument("--split-fold", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--class-mode", type=str, default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--binary-task", type=str, default="malignant_vs_normal")
    parser.add_argument("--split-mode", type=str, default="train_val_test", choices=["train_val_test", "train_val"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--use-predefined-split", action="store_true")
    parser.add_argument("--selection-metric", type=str, default="balanced_accuracy")

    # 模型
    parser.add_argument("--backbone", type=str, default="resnet3d18",
                        choices=["simple", "resnet18", "resnet18_se", "resnet18_cbam",
                                 "efficientnet_b0", "convnext_tiny", "resnet3d18",
                                 "mc3_18", "r2plus1d_18", "swin3d_tiny",
                                 "densenet3d", "densenet3d_121", "densenet121",
                                 "densenet3d121", "attention3d_cnn"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use-3d-input", action="store_true")
    parser.add_argument("--depth-size", type=int, default=128)
    parser.add_argument("--volume-hw", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    # 训练
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "onecycle", "plateau"])
    parser.add_argument("--aug-profile", type=str, default="strong", choices=["basic", "strong"])
    parser.add_argument("--sampling-strategy", type=str, default="default", choices=["default", "weighted"])
    parser.add_argument("--class-weight-strategy", type=str, default="none",
                        choices=["none", "inverse", "sqrt_inverse", "effective_num"])
    parser.add_argument("--effective-num-beta", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 多任务权重
    parser.add_argument("--weight-abnormal", type=float, default=1.0,
                        help="Head A (健康 vs 非健康) 损失权重")
    parser.add_argument("--weight-main", type=float, default=1.0,
                        help="Head B (三分类) 损失权重")
    parser.add_argument("--weight-benign-malignant", type=float, default=1.0,
                        help="Head C (良性 vs 恶性) 损失权重")

    # 推理模式
    parser.add_argument("--eval-mode", type=str, default="main", choices=["main", "cascade"],
                        help="推理模式：main=仅用主 head，cascade=Head A + Head C 级联")

    # 良性-恶性 Mixup
    parser.add_argument("--bm-mixup", action="store_true",
                        help="启用良性-恶性 Mixup 数据增强")
    parser.add_argument("--bm-mixup-alpha", type=float, default=0.4,
                        help="Mixup Beta 分布的 alpha 参数 (默认: 0.4)")

    # 混淆感知微调
    parser.add_argument("--ca-ft", action="store_true",
                        help="启用混淆感知微调 (Confusion-Aware Fine-tuning)")
    parser.add_argument("--ca-ft-start", type=int, default=None,
                        help="从第几个 epoch 开始混淆感知微调 (默认: epochs*0.6)")
    parser.add_argument("--ca-ft-boost", type=float, default=2.0,
                        help="混淆类别权重提升因子 (默认: 2.0)")

    # Class-Specific Focal Loss
    parser.add_argument("--focal-gammas", type=float, nargs="+", default=None,
                        help="每个类别的 focal gamma，如 --focal-gammas 2.0 3.0 2.0")

    args = parser.parse_args()
    args.dataset_type = _parse_dataset_type(args.dataset_type)
    config = _build_config(args)

    # 自动启用 3D
    auto_3d = {"resnet3d18", "mc3_18", "r2plus1d_18", "swin3d_tiny",
               "densenet3d", "densenet3d_121", "densenet121", "densenet3d121",
               "attention3d_cnn"}
    if config.backbone in auto_3d:
        config.use_3d_input = True

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if config.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # ---- 数据 ----
    print("\nLoading dataset...")
    ds = _build_dataset(config)
    samples = ds.get_samples()
    samples, class_names = remap_samples_by_class_mode(
        samples, class_mode=config.class_mode, binary_task=config.binary_task
    )
    num_classes = len(class_names)
    print(f"  Samples: {len(samples)}, Classes: {num_classes} ({class_names})")

    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s.label] += 1
    for lbl in range(num_classes):
        print(f"    {class_names[lbl]}: {label_counts[lbl]}")

    # 划分
    train_idx, val_idx, test_idx = _resolve_split_indices(samples, config)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx) if test_idx else 0}")

    train_tf, val_tf = _build_transforms(config)
    train_base = _build_dataset_from_samples(ds, [samples[i] for i in train_idx], train_tf, config)
    val_base = _build_dataset_from_samples(ds, [samples[i] for i in val_idx], val_tf, config)
    train_ds = Subset(train_base, range(len(train_idx)))
    val_ds = Subset(val_base, range(len(val_idx)))
    test_ds = None
    if test_idx:
        test_base = _build_dataset_from_samples(ds, [samples[i] for i in test_idx], val_tf, config)
        test_ds = Subset(test_base, range(len(test_idx)))

    # 采样器
    if config.sampling_strategy == "weighted":
        train_labels = [samples[i].label for i in train_idx]
        class_sample_counts = [max(1, train_labels.count(c)) for c in range(num_classes)]
        weights_per_class = [1.0 / c for c in class_sample_counts]
        sample_weights = [weights_per_class[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)
    test_loader = None
    if test_ds:
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True)

    # ---- 模型 ----
    print(f"\nBuilding HierarchicalMultiTaskClassifier (backbone={config.backbone})...")
    model = HierarchicalMultiTaskClassifier(
        backbone_name=config.backbone,
        num_classes=num_classes,
        pretrained=config.pretrained,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失 ----
    class_weights = None
    if config.class_weight_strategy != "none":
        counts = [label_counts[c] for c in range(num_classes)]
        class_weights = build_class_weights(counts, strategy=config.class_weight_strategy,
                                            effective_num_beta=config.effective_num_beta)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"  Class weights: {class_weights.tolist()}")

    criterion = HierarchicalMultiTaskLoss(
        class_weights=class_weights,
        label_smoothing=config.label_smoothing,
        weight_abnormal=config.weight_abnormal,
        weight_main=config.weight_main,
        weight_benign_malignant=config.weight_benign_malignant,
        focal_gammas=config.focal_gammas,
    )
    loss_desc = "hierarchical"
    if config.focal_gammas:
        loss_desc += f"+class_specific_focal(gammas={config.focal_gammas})"
    print(f"  Loss: {loss_desc} (w_abnormal={config.weight_abnormal}, w_main={config.weight_main}, w_bm={config.weight_benign_malignant})")

    # ---- 优化器 & 调度器 ----
    optimizer = create_optimizer(config.optimizer, model.parameters(), config.lr, config.weight_decay)
    scheduler, step_per_batch = create_scheduler(config.scheduler, optimizer, config.epochs, len(train_loader))

    # ---- 训练循环 ----
    selection_metric = resolve_selection_metric(config.selection_metric, num_classes)
    ca_ft_start = config.ca_ft_start if config.ca_ft_start is not None else int(config.epochs * 0.6)
    print(f"\nTraining for {config.epochs} epochs, selection_metric={selection_metric}")
    if config.bm_mixup:
        print(f"  BM-Mixup: enabled (alpha={config.bm_mixup_alpha})")
    if config.ca_ft:
        print(f"  Confusion-Aware FT: enabled (start_epoch={ca_ft_start}, boost={config.ca_ft_boost})")
    print("-" * 60)

    best_score = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    start_time = time.time()
    confusion_weights = None  # 混淆感知权重，按需计算

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # 混淆感知微调：到达 start epoch 时计算混淆权重
        if config.ca_ft and epoch == ca_ft_start and confusion_weights is None:
            print(f"\n  >>> Computing confusion weights at epoch {epoch+1}...")
            confusion_weights = _compute_confusion_weights(
                model, val_loader, device, num_classes, boost_factor=config.ca_ft_boost
            )
            confusion_weights = confusion_weights.to(device)

        # 选择训练函数
        if confusion_weights is not None:
            train_loss = train_epoch_confusion_aware(
                model, train_loader, device, criterion, optimizer,
                confusion_weights=confusion_weights,
                scheduler=scheduler, scheduler_step_per_batch=step_per_batch,
            )
        elif config.bm_mixup:
            train_loss = train_epoch_hierarchical_mixup(
                model, train_loader, device, criterion, optimizer,
                mixup_alpha=config.bm_mixup_alpha,
                scheduler=scheduler, scheduler_step_per_batch=step_per_batch,
            )
        else:
            train_loss = train_epoch_hierarchical(
                model, train_loader, device, criterion, optimizer, scheduler, step_per_batch
            )
        if scheduler and not step_per_batch:
            scheduler.step()

        val_metrics = evaluate_hierarchical(model, val_loader, device, criterion, class_names)
        val_score, used_metric = resolve_selection_score(val_metrics, selection_metric)
        elapsed = time.time() - epoch_start

        epoch_payload = {
            "loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            **{k: v for k, v in val_metrics.items() if k != "loss"},
            "selection_score": val_score,
            "selection_metric_used": used_metric,
            "epoch": epoch,
            "elapsed_sec": elapsed,
        }
        epoch_log = build_epoch_log(epoch_payload)
        epoch_log.update({"epoch": epoch, "selection_score": val_score, "elapsed_sec": elapsed})
        history.append(epoch_log)
        status = ""

        if val_score is not None and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            status = " *best"

        ca_tag = " [CA-FT]" if confusion_weights is not None else ""
        print(f"  Epoch {epoch+1:3d}/{config.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics.get('loss', 0):.4f} | val_acc={val_metrics.get('accuracy', 0):.4f} | "
              f"val_bacc={val_metrics.get('balanced_accuracy', 0):.4f} | "
              f"{selection_metric}={val_score:.4f} | {elapsed:.1f}s{status}{ca_tag}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best {selection_metric}={best_score:.4f} at epoch {best_epoch+1}")

    # ---- 测试评估 ----
    print("\n--- Test Evaluation ---")
    if best_score > -1.0 and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))

    results: Dict[str, Any] = {
        "best_epoch": best_epoch + 1,
        f"best_val_{selection_metric}": best_score,
        "total_training_time_sec": total_time,
        "history": history,
        "config": vars(args),
    }

    # 主 head 评估
    if test_loader:
        test_main = evaluate_hierarchical(model, test_loader, device, criterion, class_names)
        results.update({f"test_{k}": v for k, v in test_main.items() if v is not None})
        print(f"  [Main Head] test_acc={test_main.get('accuracy', 0):.4f} | "
              f"test_bacc={test_main.get('balanced_accuracy', 0):.4f} | "
              f"test_auroc={test_main.get('auroc', 0):.4f} | "
              f"test_f1={test_main.get('f1', 0):.4f}")
    else:
        test_main = evaluate_hierarchical(model, val_loader, device, criterion, class_names)
        results.update({f"val_test_{k}": v for k, v in test_main.items() if v is not None})

    # Cascade 评估
    if test_loader:
        test_cascade = evaluate_hierarchical_cascade(model, test_loader, device, class_names)
        results.update({f"cascade_test_{k}": v for k, v in test_cascade.items() if v is not None})
        print(f"  [Cascade]   test_acc={test_cascade.get('accuracy', 0):.4f} | "
              f"test_bacc={test_cascade.get('balanced_accuracy', 0):.4f} | "
              f"test_auroc={test_cascade.get('auroc', 0):.4f} | "
              f"test_f1={test_cascade.get('f1', 0):.4f}")

    # 保存
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'metrics.json'}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
