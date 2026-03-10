#!/usr/bin/env python3
"""
肺癌 CT 三分类统一训练框架 - 入口脚本

使用方式:

1. 训练 IQ-OTH/NCCD 数据集:
   python train.py --dataset-type iqothnccd \
     --data-root "/workspace/data-lung/IQ-OTHNCCD Lung Cancer/..." \
     --output-dir outputs/iqothnccd_resnet18 \
     --model resnet18 --pretrained --epochs 50

2. 训练 LUNA16 数据集:
   # 首先提取切片
   python prepare_luna16_slices.py --luna16-root /workspace/data-lung/luna16 \
     --output-dir /workspace/data-lung/luna16/extracted_slices --subsets 0,1,2

   # 然后训练
   python train.py --dataset-type luna16 \
     --data-root /workspace/data-lung/luna16/extracted_slices \
     --output-dir outputs/luna16_resnet18 \
     --model resnet18 --pretrained --epochs 50
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.train import main

if __name__ == "__main__":
    main()
