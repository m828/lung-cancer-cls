#!/usr/bin/env python3
"""
LUNA16 数据集快捷训练脚本
"""

import sys
import subprocess
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def auto_extract_slices(luna16_root: str = "/workspace/data-lung/luna16"):
    """自动检查并提取切片"""
    import os
    luna16_root = Path(luna16_root)
    extracted_dir = luna16_root / "extracted_slices"

    # 检查是否已存在预提取的切片
    if extracted_dir.exists() and any(extracted_dir.glob("*/*.png")):
        print("✓ 发现预提取的切片")
        return str(extracted_dir)

    # 检查是否是原始 LUNA16 目录
    if (luna16_root / "annotations.csv").exists() and any(luna16_root.glob("subset*")):
        print("✗ 未发现预提取切片，正在提取...")
        cmd = [
            "python", "prepare_luna16_slices.py",
            "--luna16-root", str(luna16_root),
            "--output-dir", str(extracted_dir),
            "--subsets", "0,1",
            "--slices-per-series", "20"
        ]
        subprocess.check_call(cmd)
        return str(extracted_dir)

    raise RuntimeError("无法识别的 LUNA16 目录结构")


def main():
    """训练 LUNA16 数据集"""
    luna16_root = "/workspace/data-lung/luna16"

    # 自动检查并提取切片
    try:
        extracted_dir = auto_extract_slices(luna16_root)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 默认训练参数
    default_args = [
        "python", "train.py",
        "--dataset-type", "luna16",
        "--data-root", extracted_dir,
        "--output-dir", "outputs/luna16_default",
        "--model", "resnet18",
        "--pretrained",
        "--epochs", "30",
        "--batch-size", "32",
        "--image-size", "224",
        "--num-workers", "4",
        "--lr", "1e-3",
    ]

    # 处理命令行参数
    if len(sys.argv) > 1:
        user_args = sys.argv[1:]
    else:
        user_args = []

    # 运行训练
    subprocess.check_call(default_args + user_args)


if __name__ == "__main__":
    main()
