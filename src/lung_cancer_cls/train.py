from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from lung_cancer_cls.dataset import (
    DatasetType,
    BaseCTDataset,
    Sample,
    create_dataset,
    get_default_transforms,
    get_default_volume_transforms,
    DataGenerator,
)
from lung_cancer_cls.training_components import (
    MaskAwareClassificationLoss,
    build_class_weights,
    create_loss,
    create_optimizer,
    create_scheduler,
)


@dataclass
class TrainConfig:
    """训练配置"""
    dataset_type: DatasetType
    data_root: Path
    output_dir: Path
    image_size: int = 224
    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    split_mode: str = "train_val_test"
    seed: int = 42
    cpu: bool = False
    model: str = "simple"
    pretrained: bool = False
    aug_profile: str = "basic"
    loss_name: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    optimizer_name: str = "adamw"
    scheduler_name: str = "none"
    sampling_strategy: str = "default"
    class_weight_strategy: str = "none"
    effective_num_beta: float = 0.999
    metadata_csv: Path | None = None
    ct_root: Path | None = None
    use_predefined_split: bool = False
    use_3d_input: bool = False
    depth_size: int = 32
    volume_hw: int = 128
    mask_txt: Path | None = None
    mask_loss_weight: float = 0.5
    consistency_weight: float = 0.1
    use_mask_guided_input: bool = False
    class_mode: str = "multiclass"
    binary_task: str = "malignant_vs_rest"
    selection_metric: str = "auto"
    intranet_source: str = "csv"
    bundle_nm_path: Path | None = None
    bundle_bn_path: Path | None = None
    bundle_mt_path: Path | None = None
    two_stage_bundle_to_csv: bool = False
    finetune_epochs: int = 10
    finetune_lr: float = 1e-4
    init_checkpoint: Path | None = None
    init_checkpoint_prefix: str | None = None
    group_split_mode: str = "auto"


MULTICLASS_NAMES = {
    0: "normal",
    1: "benign",
    2: "malignant",
}

SELECTION_METRICS = {
    "auto",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "loss",
}


def remap_samples_by_class_mode(
    samples: Sequence[Sample],
    class_mode: str,
    binary_task: str,
) -> Tuple[List[Sample], Dict[int, str]]:
    """Optionally collapse three-class labels into a binary task."""

    mode = class_mode.lower().strip()
    if mode == "multiclass":
        return list(samples), dict(MULTICLASS_NAMES)

    if mode != "binary":
        raise ValueError(f"Unknown class_mode: {class_mode}")

    task = binary_task.lower().strip()
    if task == "abnormal_vs_normal":
        class_names = {0: "normal", 1: "abnormal"}

        def remap(label: int) -> int:
            return 0 if label == 0 else 1

    elif task == "malignant_vs_rest":
        class_names = {0: "non_malignant", 1: "malignant"}

        def remap(label: int) -> int:
            return 1 if label == 2 else 0

    elif task == "malignant_vs_normal":
        class_names = {0: "normal", 1: "malignant"}
        remapped = [
            replace(sample, label=1 if sample.label == 2 else 0)
            for sample in samples
            if sample.label in {0, 2}
        ]
        return remapped, class_names

    elif task == "benign_vs_malignant":
        class_names = {0: "benign", 1: "malignant"}
        remapped = [
            replace(sample, label=1 if sample.label == 2 else 0)
            for sample in samples
            if sample.label in {1, 2}
        ]
        return remapped, class_names

    else:
        raise ValueError(f"Unknown binary_task: {binary_task}")

    remapped = [replace(sample, label=remap(sample.label)) for sample in samples]
    return remapped, class_names


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """兼容 (x,y) 与 (x,y,mask) 两种 batch 格式。"""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        return x, y, None
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, mask = batch
        return x, y, mask
    raise ValueError(f"Unsupported batch format: {type(batch)}")


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_checkpoint_state(raw_state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw_state, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            nested = raw_state.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(k, str) for k in raw_state.keys()):
            return raw_state
    raise ValueError("Unsupported checkpoint format; expected a state_dict-like mapping.")


def load_compatible_init_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    prefix: str | None = None,
) -> Dict[str, Any]:
    raw_state = torch.load(checkpoint_path, map_location=device)
    state_dict = _unwrap_checkpoint_state(raw_state)

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        normalized[key[7:] if key.startswith("module.") else key] = value

    if prefix:
        normalized = {
            key[len(prefix):]: value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in normalized.items()
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape
    }

    if not compatible:
        print(f"未找到可兼容的初始权重: {checkpoint_path}")
        if prefix:
            print(f"已按前缀过滤: {prefix}")
        return {
            "loaded_keys": 0,
            "skipped_keys": int(len(normalized)),
            "used_prefix": prefix,
        }

    model.load_state_dict(compatible, strict=False)
    skipped = max(0, len(normalized) - len(compatible))
    print(f"加载初始权重: {checkpoint_path}")
    if prefix:
        print(f"加载前缀: {prefix}")
    print(f"兼容参数: {len(compatible)}, 跳过参数: {skipped}")
    return {
        "loaded_keys": int(len(compatible)),
        "skipped_keys": int(skipped),
        "used_prefix": prefix,
    }


def infer_lidc_group_id(sample: Sample, mode: str = "auto") -> str | None:
    mode = mode.lower().strip()
    if mode == "none":
        return None

    if sample.metadata and sample.metadata.get("group_id"):
        return str(sample.metadata["group_id"])

    rel_parent = ""
    if sample.metadata and sample.metadata.get("relative_parent"):
        rel_parent = str(sample.metadata["relative_parent"])

    path_text = f"{sample.image_path.as_posix()}::{rel_parent}"
    stem = sample.image_path.stem
    stem = stem[:-4] if stem.lower().endswith(".nii") else stem

    patient_match = re.search(r"(LIDC-IDRI-\d{4})", path_text, flags=re.IGNORECASE)
    if patient_match:
        patient_id = patient_match.group(1).upper()
    else:
        patient_match = re.search(r"\b(patient|case|scan|subject)[_-]?(\d{2,})\b", path_text, flags=re.IGNORECASE)
        patient_id = f"{patient_match.group(1).lower()}_{patient_match.group(2)}" if patient_match else None

    nodule_match = re.search(r"\b(nodule|nod|lesion|roi|cluster|ann|annotation)[_-]?([A-Za-z0-9]+)\b", path_text, flags=re.IGNORECASE)
    nodule_id = f"{nodule_match.group(1).lower()}_{nodule_match.group(2)}" if nodule_match else None

    tail = stem
    if patient_id and patient_id in tail:
        tail = tail.split(patient_id, 1)[1].lstrip("_- ")
    tail = re.sub(r"(?i)([_-]?(slice|img|image|patch|frame|axial|z)[_-]?\d+)$", "", tail).strip("_- ")
    tail = re.sub(r"[_-]\d{1,4}$", "", tail).strip("_- ")

    if mode == "patient":
        return patient_id or rel_parent or None

    if mode in {"nodule", "auto"}:
        if patient_id and nodule_id:
            return f"{patient_id}__{nodule_id}"
        if patient_id and tail:
            return f"{patient_id}__{tail}"
        if rel_parent:
            return rel_parent.replace("/", "__")
        if tail and tail != stem:
            return tail
        if mode == "nodule":
            return None

    return patient_id or rel_parent or None


def infer_group_ids(
    samples: Sequence[Sample],
    dataset_type: DatasetType,
    mode: str,
) -> List[str] | None:
    mode = mode.lower().strip()
    if mode == "none":
        return None
    if dataset_type != DatasetType.LIDC_IDRI:
        return None

    group_ids = [infer_lidc_group_id(sample, mode=mode) for sample in samples]
    if any(group_id is None for group_id in group_ids):
        return None
    if len(set(group_ids)) == len(group_ids):
        return None
    return [str(group_id) for group_id in group_ids]


def _safe_group_train_test_split(
    group_items: List[Tuple[str, List[int], str]],
    test_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, List[int], str]], List[Tuple[str, List[int], str]]]:
    from sklearn.model_selection import train_test_split

    if len(group_items) <= 1:
        return group_items, []

    labels = [item[2] for item in group_items]
    use_stratify = len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) >= 2
    train_items, test_items = train_test_split(
        group_items,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels if use_stratify else None,
    )
    return list(train_items), list(test_items)


def stratified_group_split(
    samples: Sequence[Sample],
    group_ids: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        group_to_indices[str(group_id)].append(idx)

    group_items: List[Tuple[str, List[int], str]] = []
    for group_id, idxs in group_to_indices.items():
        labels = sorted({samples[idx].label for idx in idxs})
        label_signature = "|".join(str(label) for label in labels)
        group_items.append((group_id, idxs, label_signature))

    train_groups, temp_groups = _safe_group_train_test_split(group_items, test_size=(1 - train_ratio), seed=seed)
    if len(temp_groups) <= 1:
        train_idx = [idx for _, idxs, _ in group_items for idx in idxs]
        return train_idx, [], []

    val_groups, test_groups = _safe_group_train_test_split(temp_groups, test_size=0.5, seed=seed)
    train_idx = [idx for _, idxs, _ in train_groups for idx in idxs]
    val_idx = [idx for _, idxs, _ in val_groups for idx in idxs]
    test_idx = [idx for _, idxs, _ in test_groups for idx in idxs]
    return train_idx, val_idx, test_idx


def stratified_group_train_val_split(
    samples: Sequence[Sample],
    group_ids: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        group_to_indices[str(group_id)].append(idx)

    group_items: List[Tuple[str, List[int], str]] = []
    for group_id, idxs in group_to_indices.items():
        labels = sorted({samples[idx].label for idx in idxs})
        label_signature = "|".join(str(label) for label in labels)
        group_items.append((group_id, idxs, label_signature))

    train_groups, val_groups = _safe_group_train_test_split(group_items, test_size=(1 - train_ratio), seed=seed)
    train_idx = [idx for _, idxs, _ in train_groups for idx in idxs]
    val_idx = [idx for _, idxs, _ in val_groups for idx in idxs]
    return train_idx, val_idx


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    统一的数据划分方法（与 project366.ipynb 一致）
    先 80-20 划分为训练集和临时集，再将临时集 50-50 划分为验证集和测试集
    最终比例：80-10-10

    处理小类别样本数不足的情况
    """
    from sklearn.model_selection import train_test_split
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        by_label[s.label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_label.items():
        # 处理样本数极少的情况
        if len(idxs) <= 2:
            # 如果类别样本数 <= 2，则全部放入训练集
            train_idx.extend(idxs)
            continue

        # 第一次划分：训练集 vs 临时集 (80-20)
        train_imgs, temp_imgs = train_test_split(
            idxs, test_size=(1 - train_ratio),
            random_state=seed, shuffle=True
        )

        # 处理临时集样本数不足的情况
        if len(temp_imgs) <= 1:
            train_idx.extend(idxs)
            continue

        # 第二次划分：临时集分为验证集和测试集 (50-50)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5,
            random_state=seed, shuffle=True
        )

        train_idx.extend(train_imgs)
        val_idx.extend(val_imgs)
        test_idx.extend(test_imgs)

    # 检查是否有验证集和测试集，如果没有，随机从训练集中分配一些
    if len(val_idx) == 0 or len(test_idx) == 0:
        # 确保至少有一些样本用于验证和测试
        if len(val_idx) == 0:
            # 从训练集中随机选择一些作为验证集
            num_val = min(5, max(1, len(train_idx) // 20))
            val_indices = random.sample(train_idx, num_val)
            for idx in val_indices:
                val_idx.append(idx)
                train_idx.remove(idx)
        if len(test_idx) == 0:
            # 从训练集中随机选择一些作为测试集
            num_test = min(5, max(1, len(train_idx) // 20))
            test_indices = random.sample(train_idx, num_test)
            for idx in test_indices:
                test_idx.append(idx)
                train_idx.remove(idx)

    return train_idx, val_idx, test_idx


def stratified_train_val_split(
    samples: Sequence[Sample],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """按类别分层做 train/val 划分，不生成测试集（例如 8:2）。"""
    from sklearn.model_selection import train_test_split

    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        by_label[s.label].append(idx)

    train_idx, val_idx = [], []
    for _, idxs in by_label.items():
        if len(idxs) <= 1:
            train_idx.extend(idxs)
            continue

        train_imgs, val_imgs = train_test_split(
            idxs,
            test_size=(1 - train_ratio),
            random_state=seed,
            shuffle=True,
        )
        train_idx.extend(train_imgs)
        val_idx.extend(val_imgs)

    if len(val_idx) == 0 and train_idx:
        num_val = min(5, max(1, len(train_idx) // 10))
        val_indices = random.sample(train_idx, num_val)
        for idx in val_indices:
            val_idx.append(idx)
            train_idx.remove(idx)

    return train_idx, val_idx


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _metric_with_fallback(func: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return _safe_float(func(*args, **kwargs))
    except ValueError:
        return None


def _format_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def resolve_selection_metric(selection_metric: str, num_classes: int) -> str:
    metric = selection_metric.lower().strip()
    if metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    if metric == "auto":
        return "auroc" if num_classes == 2 else "balanced_accuracy"
    return metric


def get_selection_score(metrics: Dict[str, Any], metric_name: str) -> float | None:
    value = _safe_float(metrics.get(metric_name))
    if value is None:
        return None
    if metric_name == "loss":
        return -value
    return value


def resolve_selection_score(metrics: Dict[str, Any], metric_name: str) -> Tuple[float, str]:
    score = get_selection_score(metrics, metric_name)
    if score is not None:
        return score, metric_name

    fallback_metric = "accuracy"
    fallback_score = get_selection_score(metrics, fallback_metric)
    if fallback_score is not None:
        return fallback_score, fallback_metric

    return -float("inf"), metric_name


def build_epoch_log(metrics: Dict[str, Any]) -> Dict[str, float | None]:
    keys = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "npv",
        "fpr",
        "fnr",
        "youden_j",
        "brier_score",
        "ece",
    ]
    return {key: _safe_float(metrics.get(key)) for key in keys if key in metrics}


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    num_classes = probabilities.shape[1]
    labels = list(range(num_classes))
    predictions = probabilities.argmax(axis=1)

    metrics: Dict[str, Any] = {
        "loss": float(loss),
        "accuracy": _safe_float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": _metric_with_fallback(balanced_accuracy_score, y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "mcc": _metric_with_fallback(matthews_corrcoef, y_true, predictions),
        "num_samples": int(y_true.shape[0]),
    }

    if num_classes == 2:
        positive_probs = probabilities[:, 1]
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        specificity = None
        if (tn + fp) > 0:
            specificity = float(tn / (tn + fp))
        npv = None
        if (tn + fn) > 0:
            npv = float(tn / (tn + fn))
        fpr = None if (fp + tn) == 0 else float(fp / (fp + tn))
        fnr = None if (fn + tp) == 0 else float(fn / (fn + tp))
        prevalence = float(np.mean(y_true.astype(np.float32)))
        predicted_positive_rate = float(np.mean(predictions.astype(np.float32)))
        youden_j = None
        if specificity is not None:
            youden_j = float(recall + specificity - 1.0)
        mcc = None
        mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denom > 0:
            mcc = float((tp * tn - fp * fn) / np.sqrt(mcc_denom))

        # Expected calibration error with equal-width bins.
        num_bins = 10
        ece = 0.0
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        for bin_idx in range(num_bins):
            lower = bin_edges[bin_idx]
            upper = bin_edges[bin_idx + 1]
            if bin_idx == num_bins - 1:
                mask = (positive_probs >= lower) & (positive_probs <= upper)
            else:
                mask = (positive_probs >= lower) & (positive_probs < upper)
            if not np.any(mask):
                continue
            bin_conf = float(np.mean(positive_probs[mask]))
            bin_acc = float(np.mean(y_true[mask].astype(np.float32)))
            ece += abs(bin_acc - bin_conf) * (float(np.sum(mask)) / float(len(y_true)))

        metrics.update(
            {
                "auroc": _metric_with_fallback(roc_auc_score, y_true, positive_probs),
                "auprc": _metric_with_fallback(average_precision_score, y_true, positive_probs),
                "mcc": mcc,
                "precision": _safe_float(precision),
                "ppv": _safe_float(precision),
                "precision_positive": _safe_float(precision),
                "recall": _safe_float(recall),
                "sensitivity": _safe_float(recall),
                "specificity": specificity,
                "npv": npv,
                "fpr": fpr,
                "fnr": fnr,
                "youden_j": youden_j,
                "prevalence": prevalence,
                "predicted_positive_rate": predicted_positive_rate,
                "f1": _safe_float(f1),
                "brier_score": float(np.mean((positive_probs - y_true.astype(np.float32)) ** 2)),
                "ece": float(ece),
                "positive_class": class_names.get(1, "class_1"),
                "negative_class": class_names.get(0, "class_0"),
            }
        )
    else:
        y_true_onehot = np.eye(num_classes, dtype=np.float32)[y_true]
        metrics.update(
            {
                "auroc": _metric_with_fallback(
                    roc_auc_score,
                    y_true_onehot,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                ),
                "auprc": _metric_with_fallback(
                    average_precision_score,
                    y_true_onehot,
                    probabilities,
                    average="macro",
                ),
                "precision_macro": _safe_float(
                    precision_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "precision_weighted": _safe_float(
                    precision_score(y_true, predictions, average="weighted", zero_division=0)
                ),
                "recall_macro": _safe_float(
                    recall_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "recall_weighted": _safe_float(
                    recall_score(y_true, predictions, average="weighted", zero_division=0)
                ),
                "f1": _safe_float(f1_score(y_true, predictions, average="macro", zero_division=0)),
                "f1_weighted": _safe_float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
            }
        )

    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_names: Dict[int, str],
    use_mask_guided_input: bool = False,
) -> Dict[str, Any]:
    """统一的评估函数"""
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: List[torch.Tensor] = []
    all_probabilities: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x, y, mask = unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            mask_t = mask.to(device) if isinstance(mask, torch.Tensor) else None
            if use_mask_guided_input and mask_t is not None:
                x = x * mask_t

            logits = model(x)
            if isinstance(criterion, MaskAwareClassificationLoss) and mask_t is not None:
                masked_logits = model(x * mask_t)
                loss = criterion(logits, y, masked_logits=masked_logits)
            else:
                loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_labels.append(y.detach().cpu())
            all_probabilities.append(torch.softmax(logits, dim=1).detach().cpu())

    if total == 0:
        return {
            "loss": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "auroc": None,
            "auprc": None,
            "mcc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "npv": None,
            "fpr": None,
            "fnr": None,
            "youden_j": None,
            "brier_score": None,
            "ece": None,
            "confusion_matrix": [],
            "num_samples": 0,
        }

    y_true_np = torch.cat(all_labels).numpy().astype(np.int64)
    probabilities_np = torch.cat(all_probabilities).numpy()
    average_loss = total_loss / max(total, 1)
    return compute_classification_metrics(
        y_true_np,
        probabilities_np,
        loss=average_loss,
        class_names=class_names,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scheduler_step_per_batch: bool = False,
    use_mask_guided_input: bool = False,
) -> float:
    """统一的单轮训练函数"""
    model.train()
    running_loss = 0.0
    seen = 0

    for batch in loader:
        x, y, mask = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        mask_t = mask.to(device) if isinstance(mask, torch.Tensor) else None
        if use_mask_guided_input and mask_t is not None:
            x = x * mask_t

        optimizer.zero_grad()
        logits = model(x)
        if isinstance(criterion, MaskAwareClassificationLoss) and mask_t is not None:
            masked_logits = model(x * mask_t)
            loss = criterion(logits, y, masked_logits=masked_logits)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)

    return running_loss / max(seen, 1)


def train_model(config: TrainConfig) -> Dict[str, Any]:
    """
    统一的训练主函数

    Args:
        config: 训练配置

    Returns:
        包含训练历史和指标的字典
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")

    print("=" * 60)
    print(f"数据集类型: {config.dataset_type.name}")
    print(f"数据根目录: {config.data_root}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 60)

    auto_3d_models = {"resnet3d18", "mc3_18", "r2plus1d_18", "swin3d_tiny", "densenet3d", "densenet3d_121", "attention3d_cnn"}
    use_3d = config.use_3d_input or config.model in auto_3d_models

    # 1. 创建数据集
    print("\n正在加载数据集...")
    dataset_kwargs: Dict[str, Any] = {}
    full_dataset: Any
    if config.mask_txt is not None:
        full_dataset = DataGenerator(txtpath=config.mask_txt)
        samples = full_dataset.get_samples()
    else:
        if config.dataset_type == DatasetType.INTRANET_CT:
            if config.metadata_csv is not None:
                dataset_kwargs["metadata_csv"] = config.metadata_csv
            if config.ct_root is not None:
                dataset_kwargs["ct_root"] = config.ct_root
            dataset_kwargs["intranet_source"] = config.intranet_source
            if config.bundle_nm_path is not None:
                dataset_kwargs["bundle_nm_path"] = config.bundle_nm_path
            if config.bundle_bn_path is not None:
                dataset_kwargs["bundle_bn_path"] = config.bundle_bn_path
            if config.bundle_mt_path is not None:
                dataset_kwargs["bundle_mt_path"] = config.bundle_mt_path
        if config.dataset_type in {DatasetType.LIDC_IDRI, DatasetType.INTRANET_CT} and use_3d:
            dataset_kwargs["use_3d"] = True
            dataset_kwargs["depth_size"] = config.depth_size
            dataset_kwargs["volume_hw"] = config.volume_hw
        full_dataset = create_dataset(config.dataset_type, config.data_root, **dataset_kwargs)
        samples = full_dataset.get_samples()
    samples, class_names = remap_samples_by_class_mode(
        samples,
        class_mode=config.class_mode,
        binary_task=config.binary_task,
    )
    num_classes = len(class_names)
    selection_metric = resolve_selection_metric(config.selection_metric, num_classes)
    print(f"Found {len(samples)} samples")

    # 打印类别分布
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s.label] += 1
    print("类别分布:")
    for label in range(num_classes):
        class_name = class_names.get(label, f"class_{label}")
        print(f"  {class_name} (label={label}): {label_counts.get(label, 0)}")

    # 2. 数据划分
    split_info: Dict[str, Any] = {
        "requested_group_split_mode": config.group_split_mode,
        "used_group_split_mode": "none",
        "num_groups": None,
        "num_repeated_groups": None,
    }
    group_ids = infer_group_ids(samples, config.dataset_type, config.group_split_mode)
    if group_ids is not None:
        repeated_groups = sum(1 for count in Counter(group_ids).values() if count > 1)
        split_info.update(
            {
                "used_group_split_mode": config.group_split_mode,
                "num_groups": int(len(set(group_ids))),
                "num_repeated_groups": int(repeated_groups),
            }
        )
        if config.split_mode == "train_val":
            train_idx, val_idx = stratified_group_train_val_split(samples, group_ids, config.train_ratio, config.seed)
            test_idx = []
        else:
            train_idx, val_idx, test_idx = stratified_group_split(samples, group_ids, config.train_ratio, config.seed)
        print(
            f"使用 group split: mode={config.group_split_mode}, "
            f"groups={split_info['num_groups']}, repeated_groups={split_info['num_repeated_groups']}"
        )
    else:
        if config.group_split_mode != "none" and config.dataset_type == DatasetType.LIDC_IDRI:
            print(f"group split 未生效，回退到样本级分层划分: mode={config.group_split_mode}")
        if config.split_mode == "train_val":
            train_idx, val_idx = stratified_train_val_split(samples, config.train_ratio, config.seed)
            test_idx = []
        else:
            train_idx, val_idx, test_idx = stratified_split(samples, config.train_ratio, config.seed)

    # 可选：使用数据表中的预定义划分（适合内网数据）
    if config.use_predefined_split:
        split_to_idx = defaultdict(list)
        for idx, s in enumerate(samples):
            split = ""
            if s.metadata is not None:
                split = str(s.metadata.get("split", "")).lower().strip()
            split_to_idx[split].append(idx)

        predefined_train = split_to_idx.get("train", [])
        predefined_val = split_to_idx.get("val", []) + split_to_idx.get("valid", [])
        predefined_test = split_to_idx.get("test", [])

        if config.split_mode == "train_val":
            if predefined_train and predefined_val:
                train_idx = predefined_train
                val_idx = predefined_val
                test_idx = []
                print("使用 metadata 预定义划分（train/val）")
            else:
                print("预定义 train/val 划分不可用，回退为分层划分")
        elif predefined_train and (predefined_val or predefined_test):
            train_idx = predefined_train
            val_idx = predefined_val
            test_idx = predefined_test
            print("使用 metadata 预定义划分")
        else:
            print("预定义划分不可用，回退为统一分层划分")
    if config.split_mode == "train_val":
        print(f"\n数据划分: train={len(train_idx)}, val={len(val_idx)} (无测试集)")
    else:
        print(f"\n数据划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 3. 获取数据增强
    train_tf, val_test_tf = get_default_transforms(
        config.dataset_type, config.image_size, aug_profile=config.aug_profile
    )
    train_volume_tf, val_volume_tf = get_default_volume_transforms(config.aug_profile)

    # 4. 创建 DataLoader
    # 重新应用 transform（因为 create_dataset 返回的可能没有 transform）
    # 这里我们使用相同的 samples，但创建新的带 transform 的 dataset
    # 先获取样本列表，然后创建带 transform 的数据集
    from lung_cancer_cls.dataset import IQOTHNCCDDataset, LUNA16Dataset, LIDCIDRIDataset, IntranetCTDataset
    from torchvision import transforms
    from PIL import Image

    interpolation_mode = getattr(transforms, "InterpolationMode", None)
    nearest_interp = interpolation_mode.NEAREST if interpolation_mode is not None else Image.NEAREST

    mask_tf = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size), interpolation=nearest_interp),
        transforms.ToTensor(),
    ])

    test_ds = None
    if config.mask_txt is not None:
        train_ds = Subset(DataGenerator(txtpath=config.mask_txt, image_transform=train_tf, mask_transform=mask_tf), train_idx)
        val_ds = Subset(DataGenerator(txtpath=config.mask_txt, image_transform=val_test_tf, mask_transform=mask_tf), val_idx)
        if config.split_mode == "train_val_test":
            test_ds = Subset(DataGenerator(txtpath=config.mask_txt, image_transform=val_test_tf, mask_transform=mask_tf), test_idx)
    elif config.dataset_type == DatasetType.IQ_OTHNCCD:
        if use_3d:
            raise ValueError("IQ-OTH/NCCD 是 2D 数据，不能使用 3D 输入模式")
        train_ds = Subset(IQOTHNCCDDataset(samples, transform=train_tf), train_idx)
        val_ds = Subset(IQOTHNCCDDataset(samples, transform=val_test_tf), val_idx)
        if config.split_mode == "train_val_test":
            test_ds = Subset(IQOTHNCCDDataset(samples, transform=val_test_tf), test_idx)
    else:
        if config.dataset_type == DatasetType.LUNA16:
            if use_3d:
                raise ValueError("当前 LUNA16 流程使用 2D 切片，不能使用 3D 输入模式")
            train_ds = Subset(LUNA16Dataset(samples, transform=train_tf), train_idx)
            val_ds = Subset(LUNA16Dataset(samples, transform=val_test_tf), val_idx)
            if config.split_mode == "train_val_test":
                test_ds = Subset(LUNA16Dataset(samples, transform=val_test_tf), test_idx)
        elif config.dataset_type == DatasetType.LIDC_IDRI:
            train_ds = Subset(
                LIDCIDRIDataset(
                    samples,
                    transform=train_volume_tf if use_3d else train_tf,
                    use_3d=use_3d,
                    depth_size=config.depth_size,
                    volume_hw=config.volume_hw,
                ),
                train_idx,
            )
            val_ds = Subset(
                LIDCIDRIDataset(
                    samples,
                    transform=val_volume_tf if use_3d else val_test_tf,
                    use_3d=use_3d,
                    depth_size=config.depth_size,
                    volume_hw=config.volume_hw,
                ),
                val_idx,
            )
            if config.split_mode == "train_val_test":
                test_ds = Subset(
                    LIDCIDRIDataset(
                        samples,
                        transform=val_volume_tf if use_3d else val_test_tf,
                        use_3d=use_3d,
                        depth_size=config.depth_size,
                        volume_hw=config.volume_hw,
                    ),
                    test_idx,
                )
        else:
            train_ds = Subset(
                IntranetCTDataset(
                    samples,
                    transform=train_volume_tf if use_3d else train_tf,
                    use_3d=use_3d,
                    depth_size=config.depth_size,
                    volume_hw=config.volume_hw,
                ),
                train_idx,
            )
            val_ds = Subset(
                IntranetCTDataset(
                    samples,
                    transform=val_volume_tf if use_3d else val_test_tf,
                    use_3d=use_3d,
                    depth_size=config.depth_size,
                    volume_hw=config.volume_hw,
                ),
                val_idx,
            )
            if config.split_mode == "train_val_test":
                test_ds = Subset(
                    IntranetCTDataset(
                        samples,
                        transform=val_volume_tf if use_3d else val_test_tf,
                        use_3d=use_3d,
                        depth_size=config.depth_size,
                        volume_hw=config.volume_hw,
                    ),
                    test_idx,
                )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )

    # 数据不平衡处理：可选加权采样
    train_label_counts = [0 for _ in range(num_classes)]
    for i in train_idx:
        train_label_counts[samples[i].label] += 1

    if config.sampling_strategy == "weighted":
        per_class_weights = [1.0 / max(1, c) for c in train_label_counts]
        sample_weights = [per_class_weights[samples[i].label] for i in train_idx]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=config.num_workers,
        )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers
        )

    # 5. 创建模型
    print(f"\n模型: {config.model} (pretrained={config.pretrained})")
    label_summary = ", ".join(
        f"{class_names[idx]}={train_label_counts[idx]}" for idx in range(num_classes)
    )
    print(f"训练集类别计数: {label_summary}")
    print(f"不平衡策略: sampler={config.sampling_strategy}, class_weight={config.class_weight_strategy}")
    print(f"最佳模型选择指标: {selection_metric}")
    from lung_cancer_cls.model import build_model
    model = build_model(
        config.model,
        num_classes=num_classes,
        pretrained=config.pretrained,
    ).to(device)
    init_load_info = None
    if config.init_checkpoint is not None and config.init_checkpoint.exists():
        init_load_info = load_compatible_init_weights(
            model,
            config.init_checkpoint,
            device,
            prefix=config.init_checkpoint_prefix,
        )

    optimizer = create_optimizer(
        config.optimizer_name,
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = create_loss(
        config.loss_name,
        label_smoothing=config.label_smoothing,
        focal_gamma=config.focal_gamma,
        class_weights=build_class_weights(
            train_label_counts,
            strategy=config.class_weight_strategy,
            effective_num_beta=config.effective_num_beta,
        ),
        mask_loss_weight=config.mask_loss_weight,
        consistency_weight=config.consistency_weight,
    ).to(device)
    scheduler, scheduler_step_per_batch = create_scheduler(
        config.scheduler_name,
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    # 6. 准备输出目录
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7. 训练循环
    best_val_score = -float("inf")
    best_epoch = 0
    best_val_metrics: Dict[str, Any] | None = None
    history = []

    print(f"\n开始训练（共 {config.epochs} 轮）")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
            use_mask_guided_input=config.use_mask_guided_input,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            class_names=class_names,
            use_mask_guided_input=config.use_mask_guided_input,
        )
        selection_score, used_metric = resolve_selection_score(val_metrics, selection_metric)

        if scheduler is not None and not scheduler_step_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(selection_score)
            else:
                scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_metric": selection_metric,
            "selection_metric_used": used_metric,
            "selection_score": _safe_float(selection_score),
        }
        epoch_log.update({f"val_{k}": v for k, v in build_epoch_log(val_metrics).items()})
        history.append(epoch_log)

        print(f"[Epoch {epoch}/{config.epochs}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={_format_metric(_safe_float(val_metrics.get('loss')))} "
              f"val_acc={_format_metric(_safe_float(val_metrics.get('accuracy')))} "
              f"val_auc={_format_metric(_safe_float(val_metrics.get('auroc')))} "
              f"val_bacc={_format_metric(_safe_float(val_metrics.get('balanced_accuracy')))} "
              f"val_f1={_format_metric(_safe_float(val_metrics.get('f1')))} "
              f"monitor({used_metric})={_format_metric(_safe_float(val_metrics.get(used_metric)))} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  保存最佳模型 (epoch {epoch})")

    # 8. 评估测试集
    test_loss: float | None = None
    test_acc: float | None = None
    test_metrics: Dict[str, Any] | None = None
    if test_loader is not None:
        print("\n" + "-" * 60)
        print("在测试集上评估最佳模型...")
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            criterion,
            class_names=class_names,
            use_mask_guided_input=config.use_mask_guided_input,
        )
        test_loss = _safe_float(test_metrics.get("loss"))
        test_acc = _safe_float(test_metrics.get("accuracy"))
        print(f"测试结果: loss={test_loss:.4f}, acc={test_acc:.4f}")
    else:
        print("\n" + "-" * 60)
        print("跳过测试集评估（split_mode=train_val）")

    # 9. 保存指标
    best_val_acc = _safe_float(best_val_metrics.get("accuracy")) if best_val_metrics else None
    best_val_auroc = _safe_float(best_val_metrics.get("auroc")) if best_val_metrics else None
    metrics = {
        "dataset_type": config.dataset_type.name,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metrics": best_val_metrics,
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "test_metrics": test_metrics,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "class_names": class_names,
        "config": {
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model": config.model,
            "pretrained": config.pretrained,
            "class_mode": config.class_mode,
            "binary_task": config.binary_task,
            "selection_metric": config.selection_metric,
            "sampling_strategy": config.sampling_strategy,
            "class_weight_strategy": config.class_weight_strategy,
            "effective_num_beta": config.effective_num_beta,
            "group_split_mode": config.group_split_mode,
            "split_mode": config.split_mode,
            "loss_name": config.loss_name,
            "mask_txt": str(config.mask_txt) if config.mask_txt else None,
            "mask_loss_weight": config.mask_loss_weight,
            "consistency_weight": config.consistency_weight,
            "use_mask_guided_input": config.use_mask_guided_input,
            "volume_hw": config.volume_hw,
            "intranet_source": config.intranet_source,
            "bundle_nm_path": str(config.bundle_nm_path) if config.bundle_nm_path else None,
            "bundle_bn_path": str(config.bundle_bn_path) if config.bundle_bn_path else None,
            "bundle_mt_path": str(config.bundle_mt_path) if config.bundle_mt_path else None,
            "two_stage_bundle_to_csv": config.two_stage_bundle_to_csv,
            "finetune_epochs": config.finetune_epochs,
            "finetune_lr": config.finetune_lr,
            "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else None,
            "init_checkpoint_prefix": config.init_checkpoint_prefix,
        }
    }
    metrics["split_info"] = split_info
    if init_load_info is not None:
        metrics["init_load_info"] = init_load_info

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    if test_acc is not None:
        print(f"测试准确率: {test_acc:.4f}")
    print(f"结果已保存到: {out_dir}")
    print("=" * 60)

    return metrics


def train_bundle_then_finetune_csv(config: TrainConfig) -> Dict[str, Any]:
    """先用 bundle 三文件预训练，再用 csv 索引数据微调。"""
    if config.dataset_type != DatasetType.INTRANET_CT:
        raise ValueError("two-stage bundle->csv 仅支持 intranet_ct")

    stage1_cfg = replace(
        config,
        intranet_source="bundle",
        output_dir=config.output_dir / "stage1_bundle_pretrain",
        use_predefined_split=False,
        init_checkpoint=None,
    )
    print("\n" + "#" * 60)
    print("Stage-1: 使用 processed/NM_all.npy + BN_all.npy + MT_all.npy 预训练")
    print("#" * 60)
    stage1_metrics = train_model(stage1_cfg)

    stage1_ckpt = stage1_cfg.output_dir / "best_model.pt"
    if not stage1_ckpt.exists():
        raise RuntimeError(f"Stage-1 未找到最佳权重: {stage1_ckpt}")

    stage2_cfg = replace(
        config,
        intranet_source="csv",
        output_dir=config.output_dir / "stage2_csv_finetune",
        epochs=config.finetune_epochs,
        lr=config.finetune_lr,
        init_checkpoint=stage1_ckpt,
    )
    print("\n" + "#" * 60)
    print("Stage-2: 使用多模态统一检索表对应数据微调")
    print("#" * 60)
    stage2_metrics = train_model(stage2_cfg)

    summary = {
        "two_stage": True,
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "stage1_checkpoint": str(stage1_ckpt),
    }
    with open(config.output_dir / "two_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="肺癌 CT 三分类统一训练框架 "
                    "(支持 IQ-OTH/NCCD、LUNA16、LIDC-IDRI、intranet_ct)"
    )

    # 必需参数
    parser.add_argument(
        "--dataset-type", type=str, choices=["iqothnccd", "luna16", "lidc_idri", "intranet_ct"],
        required=True, help="数据集类型: iqothnccd / luna16 / lidc_idri / intranet_ct"
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录"
    )
    parser.add_argument("--metadata-csv", type=str, default=None, help="内网数据索引 CSV 路径")
    parser.add_argument("--ct-root", type=str, default=None, help="内网 CT .npy 根目录")
    parser.add_argument("--use-predefined-split", action="store_true", help="使用索引表中的 train/val/test 划分")
    parser.add_argument("--intranet-source", type=str, choices=["csv", "bundle", "both"], default="csv", help="内网数据来源：csv / bundle / both")
    parser.add_argument("--bundle-nm-path", type=str, default="/home/apulis-dev/userdata/processed/NM_all.npy", help="NM_all.npy 路径")
    parser.add_argument("--bundle-bn-path", type=str, default="/home/apulis-dev/userdata/processed/BN_all.npy", help="BN_all.npy 路径")
    parser.add_argument("--bundle-mt-path", type=str, default="/home/apulis-dev/userdata/processed/MT_all.npy", help="MT_all.npy 路径")
    parser.add_argument("--two-stage-bundle-to-csv", action="store_true", help="先 bundle 预训练，再 csv 微调")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="二阶段微调轮数")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="二阶段微调学习率")
    parser.add_argument("--mask-txt", type=str, default=None, help="mask-aware txt 列表（每行: mask_path image_path [label]）")

    # 训练参数
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="输入图像尺寸 (默认: 224)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="训练轮数 (默认: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="批次大小 (默认: 16)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="数据加载的 worker 数 (默认: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="学习率 (默认: 1e-3)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="权重衰减 (默认: 1e-4)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="训练集比例 (默认: 0.8)；在 train_val_test 中剩余用于 val/test，在 train_val 中剩余全部用于 val"
    )
    parser.add_argument(
        "--split-mode", type=str, choices=["train_val_test", "train_val"], default="train_val_test",
        help="数据划分模式：train_val_test(80/10/10) 或 train_val(8/2，无测试集)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )

    # 模型参数
    parser.add_argument(
        "--cpu", action="store_true",
        help="强制使用 CPU (即使有 GPU)"
    )
    parser.add_argument(
        "--model", type=str, choices=[
            "simple",
            "resnet18",
            "resnet18_se",
            "resnet18_cbam",
            "efficientnet_b0",
            "convnext_tiny",
            "resnet3d18",
            "mc3_18",
            "r2plus1d_18",
            "swin3d_tiny",
            "densenet3d",
            "densenet3d_121",
            "attention3d_cnn",
        ],
        default="simple", help="模型架构 (默认: simple)"
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="使用预训练权重（2D 使用 ImageNet，3D 视频模型使用 Kinetics400）"
    )
    parser.add_argument(
        "--aug-profile", type=str, choices=["basic", "strong"], default="basic",
        help="数据增强配置：basic / strong"
    )
    parser.add_argument(
        "--loss", type=str, choices=["ce", "focal", "mask_aware"], default="ce",
        help="损失函数：ce / focal / mask_aware"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="CE 标签平滑，仅 loss=ce 时有效"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal Loss gamma，仅 loss=focal 时有效"
    )
    parser.add_argument("--mask-loss-weight", type=float, default=0.5, help="mask_aware 中掩膜分支 CE 权重")
    parser.add_argument("--consistency-weight", type=float, default=0.1, help="mask_aware 中一致性 KL 权重")
    parser.add_argument("--use-mask-guided-input", action="store_true", help="用 mask 先过滤输入以降低背景噪声")
    parser.add_argument(
        "--optimizer", type=str, choices=["adamw", "sgd"], default="adamw",
        help="优化器：adamw / sgd"
    )
    parser.add_argument(
        "--scheduler", type=str, choices=["none", "cosine", "onecycle", "plateau"], default="none",
        help="学习率调度器"
    )
    parser.add_argument(
        "--sampling-strategy", type=str, choices=["default", "weighted"], default="default",
        help="训练采样策略：default / weighted（类别不平衡时推荐）"
    )
    parser.add_argument(
        "--class-weight-strategy", type=str,
        choices=["none", "inverse", "sqrt_inverse", "effective_num"], default="none",
        help="损失函数类别权重策略"
    )
    parser.add_argument(
        "--class-mode", type=str, choices=["multiclass", "binary"], default="multiclass",
        help="分类任务模式：multiclass / binary"
    )
    parser.add_argument(
        "--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal", "benign_vs_malignant"], default="malignant_vs_rest",
        help="二分类标签折叠方式"
    )
    parser.add_argument(
        "--selection-metric", type=str, choices=sorted(SELECTION_METRICS), default="auto",
        help="鏈€浣虫ā鍨嬮€夋嫨鎸囨爣锛歛uto 鍦?binary 鏃朵娇鐢?auroc锛屽湪 multiclass 鏃朵娇鐢?balanced_accuracy"
    )
    parser.add_argument(
        "--effective-num-beta", type=float, default=0.999,
        help="effective_num 权重的 beta 参数"
    )
    parser.add_argument(
        "--init-checkpoint", type=str, default=None,
        help="Optional checkpoint used to initialize compatible weights before fine-tuning."
    )
    parser.add_argument(
        "--init-checkpoint-prefix", type=str, default=None,
        help="Optional key prefix to strip when loading init weights, e.g. ct_encoder."
    )
    parser.add_argument(
        "--group-split-mode", type=str, choices=["none", "auto", "patient", "nodule"], default="auto",
        help="Split LIDC-IDRI by detected patient or nodule groups when possible."
    )
    parser.add_argument("--use-3d-input", action="store_true", help="启用 3D 体输入（仅内网 .npy）")
    parser.add_argument("--depth-size", type=int, default=32, help="3D 输入重采样深度")
    parser.add_argument("--volume-hw", type=int, default=128, help="3D 输入重采样后的平面分辨率")

    return parser


def train_main(args: argparse.Namespace) -> Dict[str, Any]:
    """主函数"""

    # 将字符串转换为枚举
    dataset_str = args.dataset_type.lower().strip()
    if dataset_str == "iqothnccd":
        dataset_type = DatasetType.IQ_OTHNCCD
    elif dataset_str == "luna16":
        dataset_type = DatasetType.LUNA16
    elif dataset_str == "lidc_idri":
        dataset_type = DatasetType.LIDC_IDRI
    elif dataset_str == "intranet_ct":
        dataset_type = DatasetType.INTRANET_CT
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    config = TrainConfig(
        dataset_type=dataset_type,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
        cpu=args.cpu,
        model=args.model,
        pretrained=args.pretrained,
        aug_profile=args.aug_profile,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        mask_loss_weight=args.mask_loss_weight,
        consistency_weight=args.consistency_weight,
        use_mask_guided_input=args.use_mask_guided_input,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        sampling_strategy=args.sampling_strategy,
        class_weight_strategy=args.class_weight_strategy,
        class_mode=args.class_mode,
        binary_task=args.binary_task,
        selection_metric=args.selection_metric,
        effective_num_beta=args.effective_num_beta,
        use_3d_input=args.use_3d_input,
        depth_size=args.depth_size,
        volume_hw=args.volume_hw,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        ct_root=Path(args.ct_root) if args.ct_root else None,
        use_predefined_split=args.use_predefined_split,
        mask_txt=Path(args.mask_txt) if args.mask_txt else None,
        intranet_source=args.intranet_source,
        bundle_nm_path=Path(args.bundle_nm_path) if args.bundle_nm_path else None,
        bundle_bn_path=Path(args.bundle_bn_path) if args.bundle_bn_path else None,
        bundle_mt_path=Path(args.bundle_mt_path) if args.bundle_mt_path else None,
        two_stage_bundle_to_csv=args.two_stage_bundle_to_csv,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        init_checkpoint_prefix=args.init_checkpoint_prefix,
        group_split_mode=args.group_split_mode,
    )
    if config.two_stage_bundle_to_csv:
        return train_bundle_then_finetune_csv(config)
    return train_model(config)


def main():
    """CLI entrypoint."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
