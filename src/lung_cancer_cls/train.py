from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from lung_cancer_cls.dataset import (
    DatasetType,
    BaseCTDataset,
    Sample,
    create_dataset,
    get_default_transforms,
)
from lung_cancer_cls.model import SimpleCTClassifier, ResNet18CTClassifier


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
    seed: int = 42
    cpu: bool = False
    model: str = "simple"
    pretrained: bool = False


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    统一的数据划分方法（与 project366.ipynb 一致）
    先 80-20 划分为训练集和临时集，再将临时集 50-50 划分为验证集和测试集
    最终比例：80-10-10
    """
    from sklearn.model_selection import train_test_split
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        by_label[s.label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_label.items():
        # 第一次划分：训练集 vs 临时集 (80-20)
        train_imgs, temp_imgs = train_test_split(
            idxs, test_size=(1 - train_ratio),
            random_state=seed, shuffle=True
        )
        # 第二次划分：临时集分为验证集和测试集 (50-50)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5,
            random_state=seed, shuffle=True
        )

        train_idx.extend(train_imgs)
        val_idx.extend(val_imgs)
        test_idx.extend(test_imgs)
    return train_idx, val_idx, test_idx


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float]:
    """统一的评估函数"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    """统一的单轮训练函数"""
    model.train()
    running_loss = 0.0
    seen = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
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

    # 1. 创建数据集
    print("\n正在加载数据集...")
    full_dataset = create_dataset(config.dataset_type, config.data_root)
    samples = full_dataset.get_samples()
    print(f"找到 {len(samples)} 个样本")

    # 打印类别分布
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s.label] += 1
    print("类别分布:")
    for label in sorted(label_counts.keys()):
        class_name = ["normal", "benign", "malignant"][label]
        print(f"  {class_name} (label={label}): {label_counts[label]}")

    # 2. 数据划分
    train_idx, val_idx, test_idx = stratified_split(
        samples, config.train_ratio, config.seed
    )
    print(f"\n数据划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 3. 获取数据增强
    train_tf, val_test_tf = get_default_transforms(
        config.dataset_type, config.image_size
    )

    # 4. 创建 DataLoader
    # 重新应用 transform（因为 create_dataset 返回的可能没有 transform）
    # 这里我们使用相同的 samples，但创建新的带 transform 的 dataset
    # 先获取样本列表，然后创建带 transform 的数据集
    from lung_cancer_cls.dataset import IQOTHNCCDDataset, LUNA16Dataset

    if config.dataset_type == DatasetType.IQ_OTHNCCD:
        train_ds = Subset(IQOTHNCCDDataset(samples, transform=train_tf), train_idx)
        val_ds = Subset(IQOTHNCCDDataset(samples, transform=val_test_tf), val_idx)
        test_ds = Subset(IQOTHNCCDDataset(samples, transform=val_test_tf), test_idx)
    else:
        train_ds = Subset(LUNA16Dataset(samples, transform=train_tf), train_idx)
        val_ds = Subset(LUNA16Dataset(samples, transform=val_test_tf), val_idx)
        test_ds = Subset(LUNA16Dataset(samples, transform=val_test_tf), test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )

    # 5. 创建模型
    print(f"\n模型: {config.model} (pretrained={config.pretrained})")
    if config.model == "resnet18":
        model = ResNet18CTClassifier(
            num_classes=3, pretrained=config.pretrained
        ).to(device)
    else:
        model = SimpleCTClassifier(num_classes=3).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # 6. 准备输出目录
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7. 训练循环
    best_val_acc = -1.0
    history = []

    print(f"\n开始训练（共 {config.epochs} 轮）")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"[Epoch {epoch}/{config.epochs}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  保存最佳模型 (epoch {epoch})")

    # 8. 评估测试集
    print("\n" + "-" * 60)
    print("在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"测试结果: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 9. 保存指标
    metrics = {
        "dataset_type": config.dataset_type.name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "config": {
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model": config.model,
            "pretrained": config.pretrained,
        }
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"结果已保存到: {out_dir}")
    print("=" * 60)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="肺癌 CT 三分类统一训练框架 "
                    "(支持 IQ-OTH/NCCD 和 LUNA16)"
    )

    # 必需参数
    parser.add_argument(
        "--dataset-type", type=str, choices=["iqothnccd", "luna16"],
        required=True, help="数据集类型: iqothnccd 或 luna16"
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录"
    )

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
        help="训练集比例 (默认: 0.8)，剩余 20% 平分为验证集和测试集"
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
        "--model", type=str, choices=["simple", "resnet18"],
        default="simple", help="模型架构 (默认: simple)"
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="使用 ImageNet 预训练权重 (仅对 resnet18 有效)"
    )

    return parser


def main():
    """主函数"""
    args = build_parser().parse_args()

    # 将字符串转换为枚举
    dataset_str = args.dataset_type.lower().strip()
    if dataset_str == "iqothnccd":
        dataset_type = DatasetType.IQ_OTHNCCD
    elif dataset_str == "luna16":
        dataset_type = DatasetType.LUNA16
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
        seed=args.seed,
        cpu=args.cpu,
        model=args.model,
        pretrained=args.pretrained,
    )

    train_model(config)


if __name__ == "__main__":
    main()
