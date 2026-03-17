#!/usr/bin/env python3
"""测试标注数据是否可以正确加载。"""

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.dataset import create_dataset, DatasetType


def test_3class():
    """测试三分类数据加载"""
    print("=" * 60)
    print("测试三分类数据集 (normal/benign/malignant)")
    print("=" * 60)

    data_root = Path("/workspace/data-lung/lidc_idri_3class")

    dataset = create_dataset(DatasetType.LIDC_IDRI, data_root, use_3d=True)
    print(f"✓ 数据集加载成功")
    print(f"  样本总数: {len(dataset)}")

    # 统计标签分布
    label_counts = {}
    for sample in dataset.get_samples():
        label = sample.label
        label_counts[label] = label_counts.get(label, 0) + 1

    class_names = ["normal", "benign", "malignant"]
    for label in sorted(label_counts.keys()):
        print(f"  {class_names[label]} (label={label}): {label_counts[label]}")

    # 测试加载一个样本
    img, label = dataset[0]
    print(f"\n  第一个样本:")
    print(f"    标签: {label} ({class_names[label]})")
    print(f"    图像形状: {img.shape}")
    print(f"    图像类型: {type(img)}")
    print(f"    数值范围: [{img.min():.3f}, {img.max():.3f}]")

    return True


def test_2class():
    """测试二分类数据加载"""
    print("\n" + "=" * 60)
    print("测试二分类数据集 (normal/malignant)")
    print("=" * 60)

    data_root = Path("/workspace/data-lung/lidc_idri_2class")

    dataset = create_dataset(DatasetType.LIDC_IDRI, data_root, use_3d=True)
    print(f"✓ 数据集加载成功")
    print(f"  样本总数: {len(dataset)}")

    # 统计标签分布
    label_counts = {}
    for sample in dataset.get_samples():
        label = sample.label
        label_counts[label] = label_counts.get(label, 0) + 1

    class_names = ["normal", "benign", "malignant"]
    for label in sorted(label_counts.keys()):
        if label == 2:
            display_name = "malignant (benign+malignant)"
        else:
            display_name = class_names[label]
        print(f"  {display_name} (label={label}): {label_counts[label]}")

    # 测试加载一个样本
    img, label = dataset[0]
    print(f"\n  第一个样本:")
    if label == 2:
        display_name = "malignant (benign+malignant)"
    else:
        display_name = class_names[label]
    print(f"    标签: {label} ({display_name})")
    print(f"    图像形状: {img.shape}")
    print(f"    图像类型: {type(img)}")
    print(f"    数值范围: [{img.min():.3f}, {img.max():.3f}]")

    return True


def main():
    try:
        success_3 = test_3class()
        success_2 = test_2class()

        print("\n" + "=" * 60)
        if success_3 and success_2:
            print("✓ 所有测试通过！")
            print("\n数据路径:")
            print("  三分类: /workspace/data-lung/lidc_idri_3class")
            print("  二分类: /workspace/data-lung/lidc_idri_2class")
            print("\n训练命令示例:")
            print("  三分类:")
            print("    python train.py --dataset-type lidc_idri --data-root /workspace/data-lung/lidc_idri_3class --output-dir outputs/lidc_3class --model attention3d_cnn --split-mode train_val --epochs 5 --batch-size 4 --image-size 128")
            print("\n  二分类:")
            print("    python train.py --dataset-type lidc_idri --data-root /workspace/data-lung/lidc_idri_2class --output-dir outputs/lidc_2class --model attention3d_cnn --split-mode train_val --epochs 5 --batch-size 4 --image-size 128")
        else:
            print("✗ 部分测试失败")
        print("=" * 60)
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
