from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


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
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    image_path: Path
    label: int


def canonicalize_class_name(name: str) -> str | None:
    return ALIASES.get(name.strip().lower())


def discover_iqothnccd_samples(root: str | Path) -> List[Sample]:
    """Discover image samples for IQ-OTH/NCCD style datasets.

    This supports both:
    - root/{normal,benign,malignant}/*.png
    - root/**/{Normal,Benign,Malignant}/**/*.jpg
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    samples: List[Sample] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue

        matched_label = None
        for part in p.parts:
            cname = canonicalize_class_name(part)
            if cname is not None:
                matched_label = CLASS_NAME_TO_ID[cname]
        if matched_label is None:
            continue
        samples.append(Sample(image_path=p, label=matched_label))

    if not samples:
        raise RuntimeError(
            "No images were discovered. Check folder names include normal/benign/malignant."
        )
    return samples


class LungCT2DDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label
