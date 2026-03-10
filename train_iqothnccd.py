#!/usr/bin/env python3
"""
IQ-OTH/NCCD 数据集快捷训练脚本
"""

import sys
import subprocess
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """训练 IQ-OTH/NCCD 数据集"""
    default_args = [
        "python", "train.py",
        "--dataset-type", "iqothnccd",
        "--data-root", "/workspace/data-lung/IQ-OTHNCCD Lung Cancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset",
        "--output-dir", "outputs/iqothnccd_default",
        "--model", "resnet18",
        "--pretrained",
        "--epochs", "50",
        "--batch-size", "16",
        "--image-size", "224",
        "--num-workers", "4",
        "--lr", "1e-3",
    ]

    if len(sys.argv) > 1:
        # 使用用户提供的参数
        args = sys.argv[1:]
        # 检查是否需要自动添加输出目录后缀
        if "--output-dir" not in args and any(arg in args for arg in ["--model", "--epochs"]):
            # 可以添加逻辑自动生成输出目录
            pass
    else:
        # 使用默认参数
        args = []

    # 运行训练
    subprocess.check_call(default_args + args)


if __name__ == "__main__":
    main()
