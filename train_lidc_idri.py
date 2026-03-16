#!/usr/bin/env python3
"""LIDC-IDRI 数据集快捷训练脚本。"""

import sys
import subprocess
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    lidc_root = "/workspace/data-lung/lidc_idri_slices"

    default_args = [
        "python", "train.py",
        "--dataset-type", "lidc_idri",
        "--data-root", lidc_root,
        "--output-dir", "outputs/lidc_idri_default",
        "--model", "resnet18",
        "--pretrained",
        "--split-mode", "train_val_test",
        "--epochs", "30",
        "--batch-size", "16",
        "--image-size", "224",
        "--num-workers", "4",
        "--lr", "1e-3",
    ]

    user_args = sys.argv[1:] if len(sys.argv) > 1 else []
    subprocess.check_call(default_args + user_args)


if __name__ == "__main__":
    main()
