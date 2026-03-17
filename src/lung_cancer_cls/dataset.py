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
    LIDC_IDRI = auto()
    INTRANET_CT = auto()


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
    "abnormal": 1,  # 二分类时 abnormal 对应标签 1
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
    "abnormal": "abnormal",
    "a": "abnormal",
    "benign or malignant": "abnormal",
    "normal cases": "normal",
    "benign cases": "benign",
    "malignant cases": "malignant",
    "abnormal cases": "abnormal",
    "bengin": "benign",
    "bengin case": "benign",
    "bengin cases": "benign",
    "normal case": "normal",
    "benign case": "benign",
    "malignant case": "malignant",
    "abnormal case": "abnormal",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def canonicalize_class_name(name: str) -> str | None:
    return ALIASES.get(name.strip().lower())


class DataGenerator(Dataset):
    """Mask-aware 数据读取器（借鉴 Subregion-Unet 的 txt 配置方式）。

    txt 每行支持以下格式之一：
    1) `mask_path image_path label`
    2) `mask_path image_path`（标签从 image_path 父目录名推断）
    """

    def __init__(
        self,
        txtpath: str | Path,
        image_transform: Callable | None = None,
        mask_transform: Callable | None = None,
    ):
        self.datatxtpath = Path(txtpath)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.samples = self.read_txt()

    def read_txt(self) -> List[Sample]:
        samples: List[Sample] = []
        if not self.datatxtpath.exists():
            raise FileNotFoundError(f"Mask txt not found: {self.datatxtpath}")

        with open(self.datatxtpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                mask_path = Path(parts[0])
                image_path = Path(parts[1])

                if len(parts) >= 3:
                    label_token = parts[2].strip().lower()
                    if label_token.isdigit():
                        label = int(label_token)
                    else:
                        canonical = canonicalize_class_name(label_token)
                        if canonical is None:
                            continue
                        label = CLASS_NAME_TO_ID[canonical]
                else:
                    canonical = canonicalize_class_name(image_path.parent.name)
                    if canonical is None:
                        continue
                    label = CLASS_NAME_TO_ID[canonical]

                if not image_path.exists() or not mask_path.exists():
                    continue

                samples.append(
                    Sample(
                        image_path=image_path,
                        label=label,
                        metadata={"mask_path": str(mask_path)},
                    )
                )

        if not samples:
            raise RuntimeError("No valid mask-aware samples parsed from txt")
        return samples

    def get_samples(self) -> List[Sample]:
        return self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mask_path = Path(sample.metadata["mask_path"]) if sample.metadata else None
        if mask_path is None:
            raise ValueError("Mask path missing in sample metadata")

        img = Image.open(sample.image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        mask_arr = np.array(mask)
        mask_arr[mask_arr != 0] = 255
        mask = Image.fromarray(mask_arr.astype(np.uint8))

        if self.image_transform is not None:
            img = self.image_transform(img)
        else:
            from torchvision import transforms

            img = transforms.ToTensor()(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            from torchvision import transforms

            mask = transforms.ToTensor()(mask)

        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()

        return img, sample.label, mask


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
# LIDC-IDRI 数据集
# ======================================


class LIDCIDRIDataset(BaseCTDataset):
    """LIDC-IDRI 数据集（支持 2D 切片和 3D `.npy` 体数据）。"""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable | None = None,
        use_3d: bool = False,
        depth_size: int = 32,
        volume_hw: int = 128,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.use_3d = use_3d
        self.depth_size = depth_size
        self.volume_hw = volume_hw

    def get_samples(self) -> List[Sample]:
        return self.samples

    @classmethod
    def discover(cls, root: Path, **kwargs) -> 'LIDCIDRIDataset':
        """发现 LIDC-IDRI 样本（normal/benign/malignant 目录结构）。"""
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")

        use_3d = bool(kwargs.get("use_3d", False))
        depth_size = int(kwargs.get("depth_size", 32))
        volume_hw = int(kwargs.get("volume_hw", 128))

        samples: List[Sample] = []
        for label_dir in root.iterdir():
            if not label_dir.is_dir():
                continue

            canonical = canonicalize_class_name(label_dir.name)
            if canonical is None:
                continue
            label = CLASS_NAME_TO_ID[canonical]

            for p in label_dir.iterdir():
                if not p.is_file():
                    continue
                if use_3d:
                    if p.suffix.lower() == ".npy":
                        samples.append(Sample(image_path=p, label=label))
                else:
                    if p.suffix.lower() in IMG_EXTS:
                        samples.append(Sample(image_path=p, label=label))

        if not samples:
            raise RuntimeError(
                "No LIDC-IDRI samples discovered. Expect subfolders like normal/benign/malignant with images (2D) or .npy volumes (3D)."
            )
        return cls(samples, use_3d=use_3d, depth_size=depth_size, volume_hw=volume_hw)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        if self.use_3d:
            arr = np.load(sample.image_path).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise ValueError(f"Unsupported LIDC-IDRI 3D array shape: {arr.shape}, path={sample.image_path}")

            arr = arr - arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val

            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.depth_size, self.volume_hw, self.volume_hw),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            tensor = (tensor - 0.5) / 0.5
            return tensor, sample.label

        img = Image.open(sample.image_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label


# ======================================
# 内网 CT 数据集
# ======================================

INTRANET_LABEL_MAP: Dict[str, int] = {
    "健康对照": 0,
    "良性结节": 1,
    "肺癌": 2,
}


class IntranetCTDataset(BaseCTDataset):
    """内网 CT 三分类数据集（基于索引表读取 .npy）"""

    _bundle_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable | None = None,
        use_3d: bool = False,
        depth_size: int = 32,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.use_3d = use_3d
        self.depth_size = depth_size

    def get_samples(self) -> List[Sample]:
        return self.samples

    @classmethod
    def discover(cls, root: Path, **kwargs) -> "IntranetCTDataset":
        """
        从内网索引表发现样本。

        kwargs:
            metadata_csv: 索引 CSV 路径（默认 root / "多模态统一检索表_CT本地路径_CT划分.csv"）
            ct_root: CT 数据根目录（默认 root）
            ct_path_col: CT 路径列名（默认 "CT_numpy_cloud路径"）
            label_col: 标签列名（默认 "样本类型"）
            split_col: 划分列名（默认 "CT_train_val_split"）
        """
        source = str(kwargs.get("intranet_source", "csv")).lower().strip()

        if source == "csv":
            samples = cls._discover_from_csv(root, **kwargs)
        elif source == "bundle":
            samples = cls._discover_from_bundles(root, **kwargs)
        elif source == "both":
            samples = cls._discover_from_csv(root, **kwargs) + cls._discover_from_bundles(root, **kwargs)
        else:
            raise ValueError(f"Unknown intranet_source: {source}")

        if not samples:
            raise RuntimeError("No valid intranet CT samples discovered")
        return cls(samples)

    @classmethod
    def _discover_from_csv(cls, root: Path, **kwargs) -> List[Sample]:
        import pandas as pd

        metadata_csv = Path(kwargs.get("metadata_csv", root / "多模态统一检索表_CT本地路径_CT划分.csv"))
        ct_root = Path(kwargs.get("ct_root", root))
        ct_path_col = kwargs.get("ct_path_col", "CT_numpy_cloud路径")
        label_col = kwargs.get("label_col", "样本类型")
        split_col = kwargs.get("split_col", "CT_train_val_split")

        if not metadata_csv.exists():
            return []
        if not ct_root.exists():
            return []

        df = pd.read_csv(metadata_csv).fillna("PANDASNAN")
        samples: List[Sample] = []
        for _, row in df.iterrows():
            label_name = str(row.get(label_col, "")).strip()
            if label_name not in INTRANET_LABEL_MAP:
                continue
            label = INTRANET_LABEL_MAP[label_name]
            ct_rel = row.get(ct_path_col, "PANDASNAN")
            if ct_rel == "PANDASNAN":
                continue
            rel_path = str(ct_rel).replace("\\", "/").lstrip("/")
            ct_path = ct_root / rel_path
            if not ct_path.exists():
                continue
            split = str(row.get(split_col, "")).strip().lower()
            samples.append(Sample(image_path=ct_path, label=label, metadata={"split": split} if split else None))
        return samples

    @classmethod
    def _discover_from_bundles(cls, root: Path, **kwargs) -> List[Sample]:
        nm_path = Path(kwargs.get("bundle_nm_path", root / "processed/NM_all.npy"))
        bn_path = Path(kwargs.get("bundle_bn_path", root / "processed/BN_all.npy"))
        mt_path = Path(kwargs.get("bundle_mt_path", root / "processed/MT_all.npy"))

        mapping = [(nm_path, 0), (bn_path, 1), (mt_path, 2)]
        samples: List[Sample] = []
        for p, label in mapping:
            if not p.exists():
                continue
            arr = np.load(p, mmap_mode="r")
            n = int(arr.shape[0]) if arr.ndim >= 3 else 1
            for i in range(n):
                samples.append(
                    Sample(
                        image_path=p,
                        label=label,
                        metadata={"bundle_path": str(p), "bundle_index": i},
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        arr: np.ndarray
        if sample.metadata is not None and "bundle_path" in sample.metadata:
            bundle_path = str(sample.metadata["bundle_path"])
            bundle_index = int(sample.metadata.get("bundle_index", 0))
            if bundle_path not in self._bundle_cache:
                self._bundle_cache[bundle_path] = np.load(bundle_path)
            bundle_arr = self._bundle_cache[bundle_path]
            if bundle_arr.ndim >= 3:
                arr = bundle_arr[bundle_index].astype(np.float32)
            else:
                arr = bundle_arr.astype(np.float32)
        else:
            arr = np.load(sample.image_path).astype(np.float32)

        if self.use_3d:
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise ValueError(f"Unsupported CT array shape for 3D mode: {arr.shape}, path={sample.image_path}")

            arr = arr - arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val

            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.depth_size, 128, 128),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            tensor = (tensor - 0.5) / 0.5
            return tensor, sample.label

        if self.use_3d:
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise ValueError(f"Unsupported CT array shape for 3D mode: {arr.shape}, path={sample.image_path}")

            arr = arr - arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val

            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.depth_size, 128, 128),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            tensor = (tensor - 0.5) / 0.5
            return tensor, sample.label

        # 兼容 3D 体数据：取中间层作为 2D 输入
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported CT array shape: {arr.shape}, path={sample.image_path}")

        # min-max 到 [0, 255]，再转 PIL 走统一 transform
        arr = arr - arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        arr = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

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
    elif dataset_type == DatasetType.LIDC_IDRI:
        return LIDCIDRIDataset.discover(root, **kwargs)
    elif dataset_type == DatasetType.INTRANET_CT:
        return IntranetCTDataset.discover(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_default_transforms(
    dataset_type: DatasetType,
    image_size: int = 224,
    aug_profile: str = "basic",
):
    """获取指定数据集类型的数据增强（可切换 profile 便于消融）。"""
    from torchvision import transforms

    profile = aug_profile.lower().strip()

    if profile == "strong":
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        val_test_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return train_tf, val_test_tf

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
