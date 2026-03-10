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
    "normal cases": "normal",
    "benign cases": "benign",
    "malignant cases": "malignant",
    "bengin": "benign",  # handle misspelling
    "bengin case": "benign",
    "bengin cases": "benign",
    "normal case": "normal",
    "benign case": "benign",
    "malignant case": "malignant",
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
    - Also checks filenames for category keywords
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Direct map from folder keywords to label - order matters! Longer patterns first!
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

    # Get all subdirectories in root
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue

        # Determine label from category directory name
        dir_name_lower = category_dir.name.lower()
        label = -1

        # Find matching category keyword in directory name
        for keyword, l in folder_label_map:
            if keyword in dir_name_lower:
                label = l
                break

        if label == -1:
            continue  # skip directories that don't match any category

        # Find all image files in this category directory
        for p in category_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            samples.append(Sample(image_path=p, label=label))

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
