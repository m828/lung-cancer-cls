#!/usr/bin/env python3
"""测试二分类数据集"""

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.dataset import create_dataset, DatasetType


def main():
    data_dir = Path("/workspace/data-lung/lidc_idri_full_binary")

    print("=" * 60)
    print("测试二分类数据集 (normal/abnormal)")
    print("=" * 60)

    # 测试加载
    dataset = create_dataset(DatasetType.LIDC_IDRI, data_dir, use_3d=True)
    print(f"✓ 数据集加载成功")
    print(f"  样本总数: {len(dataset)}")

    # 统计标签分布
    label_counts = {}
    for sample in dataset.get_samples():
        label = sample.label
        label_counts[label] = label_counts.get(label, 0) + 1

    class_names = ["normal", "abnormal"]
    for label in sorted(label_counts.keys()):
        display_name = "normal" if label == 0 else "abnormal"
        print(f"  {display_name} (label={label}): {label_counts[label]}")

    # 测试加载一个样本
    img, label = dataset[0]
    print(f"\n  第一个样本:")
    display_name = "normal" if label == 0 else "abnormal"
    print(f"    标签: {label} ({display_name})")
    print(f"    图像形状: {img.shape}")
    print(f"    图像类型: {type(img)}")
    print(f"    数值范围: [{img.min():.3f}, {img.max():.3f}]")

    # 验证所有数据都是有效的
    valid_count = 0
    invalid_count = 0
    print(f"\n  验证所有数据...")
    for i in range(len(dataset)):
        try:
            img, label = dataset[i]
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            print(f"    错误 ({i}): {e}")

    print(f"  有效样本: {valid_count}")
    print(f"  无效样本: {invalid_count}")

    print("\n" + "=" * 60)
    print("训练命令示例:")
    print(f"  python train.py --dataset-type lidc_idri --data-root {data_dir} --output-dir outputs/lidc_binary --model attention3d_cnn --split-mode train_val --epochs 5 --batch-size 4 --image-size 128")


if __name__ == "__main__":
    main()
