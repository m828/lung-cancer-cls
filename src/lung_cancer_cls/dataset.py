from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DatasetType(Enum):
    """支持的数据集类型"""
    IQ_OTHNCCD = auto()
    LUNA16 = auto()


@dataclass
class Sample:
    """通用样本表示"""
    image_path: Path
    label: int
    metadata: Optional[Dict] = None


class BaseCTDataset(Dataset, ABC):
    """CT 数据集的基类，定义通用接口"""

    @abstractmethod
    def get_samples(self) -> List[Sample]:
        """获取所有样本"""
        pass

    @classmethod
    @abstractmethod
    def discover(cls, root: Path, **kwargs) -> 'BaseCTDataset':
        """从目录发现并加载数据集"""
        pass


# ======================================
# 标签映射和工具函数
# ======================================

CLASS_NAME_TO_ID: Dict[str, int] = {
    "normal": 0,
    "benign": 1,
    "malignant": 2,
}

ALIASES = {
    "normal": "normal",
    "healthy": "normal",
    "n": "normal",
    "benign": "benign",
    "b": "benign",
    "malignant": "malignant",
    "m": "malignant",
    "cancer": "malignant",
    "normal cases": "normal",
    "benign cases": "benign",
    "malignant cases": "malignant",
    "bengin": "benign",
    "bengin case": "benign",
    "bengin cases": "benign",
    "normal case": "normal",
    "benign case": "benign",
    "malignant case": "malignant",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def canonicalize_class_name(name: str) -> str | None:
    return ALIASES.get(name.strip().lower())


# ======================================
# IQ-OTH/NCCD 数据集
# ======================================

class IQOTHNCCDDataset(BaseCTDataset):
    """IQ-OTH/NCCD 2D 图像数据集"""

    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None):
        self.samples = list(samples)
        self.transform = transform

    def get_samples(self) -> List[Sample]:
        return self.samples

    @classmethod
    def discover(cls, root: Path, **kwargs) -> 'IQOTHNCCDDataset':
        """发现 IQ-OTH/NCCD 样本"""
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")

        # 文件夹关键词到标签的映射
        folder_label_map = [
            ("malignant cases", 2),
            ("malignant case", 2),
            ("cancer cases", 2),
            ("cancer case", 2),
            ("benign cases", 1),
            ("benign case", 1),
            ("bengin cases", 1),
            ("bengin case", 1),
            ("normal cases", 0),
            ("normal case", 0),
            ("healthy cases", 0),
            ("healthy case", 0),
            ("malignant", 2),
            ("cancer", 2),
            ("bengin", 1),
            ("benign", 1),
            ("normal", 0),
            ("healthy", 0),
            ("m", 2),
            ("b", 1),
            ("n", 0),
        ]

        samples: List[Sample] = []

        for category_dir in root.iterdir():
            if not category_dir.is_dir():
                continue

            dir_name_lower = category_dir.name.lower()
            label = -1

            for keyword, l in folder_label_map:
                if keyword in dir_name_lower:
                    label = l
                    break

            if label == -1:
                continue

            for p in category_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                    continue
                samples.append(Sample(image_path=p, label=label))

        if not samples:
            raise RuntimeError(
                "No images were discovered. Check folder names include normal/benign/malignant."
            )
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label


# ======================================
# LUNA16 数据集
# ======================================

try:
    import SimpleITK as sitk
    import pandas as pd
    HAS_LUNA16_DEPS = True
except ImportError:
    HAS_LUNA16_DEPS = False


class LUNA16Dataset(BaseCTDataset):
    """LUNA16 数据集（使用预提取的 2D 切片）"""

    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None):
        self.samples = list(samples)
        self.transform = transform

    def get_samples(self) -> List[Sample]:
        return self.samples

    @classmethod
    def discover(cls, root: Path, **kwargs) -> 'LUNA16Dataset':
        """发现 LUNA16 样本（从预提取的切片目录）"""
        # 检查是否是原始 LUNA16 目录
        if (root / "annotations.csv").exists() and any(root.glob("subset*")):
            # 检查是否有预提取的切片
            extracted_dir = root / "extracted_slices"
            if extracted_dir.exists() and any(extracted_dir.glob("*/*.png")):
                return cls._discover_from_extracted(extracted_dir)
            else:
                # 需要先提取切片
                raise RuntimeError(
                    "No extracted slices found. Please first run:\n"
                    "python prepare_luna16_slices.py --luna16-root /path/to/luna16"
                )
        else:
            # 直接是提取的切片目录
            return cls._discover_from_extracted(root)

    @classmethod
    def _discover_from_extracted(cls, root: Path) -> 'LUNA16Dataset':
        """从预提取的切片目录发现样本"""
        samples: List[Sample] = []

        for label_dir in root.iterdir():
            if not label_dir.is_dir():
                continue

            label = -1
            dir_name = label_dir.name.lower()
            if "normal" in dir_name:
                label = 0
            elif "benign" in dir_name:
                label = 1
            elif "malignant" in dir_name or "nodule" in dir_name:
                label = 2

            if label == -1:
                continue

            for img_path in label_dir.glob("*.png"):
                samples.append(Sample(image_path=img_path, label=label))

        if not samples:
            raise RuntimeError("No extracted slices found in directory")
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label


# ======================================
# 工厂函数
# ======================================

def create_dataset(dataset_type: DatasetType, root: Path, **kwargs) -> BaseCTDataset:
    """创建指定类型的数据集"""
    if dataset_type == DatasetType.IQ_OTHNCCD:
        return IQOTHNCCDDataset.discover(root, **kwargs)
    elif dataset_type == DatasetType.LUNA16:
        return LUNA16Dataset.discover(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_default_transforms(dataset_type: DatasetType, image_size: int = 224):
    """获取指定数据集类型的默认数据增强"""
    from torchvision import transforms

    if dataset_type == DatasetType.IQ_OTHNCCD:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        val_test_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        # LUNA16 使用略有不同的归一化
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.3]),
        ])
        val_test_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.3]),
        ])
    return train_tf, val_test_tf
