#!/usr/bin/env python3
"""
从三分类数据创建二分类数据（normal vs abnormal）
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="从三分类数据创建二分类数据（normal vs abnormal）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/workspace/data-lung/lidc_idri_full_3class",
        help="三分类数据输入目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/data-lung/lidc_idri_full_binary",
        help="二分类数据输出目录"
    )
    parser.add_argument(
        "--normal-class",
        type=str,
        default="normal",
        help="正常类别的目录名"
    )
    parser.add_argument(
        "--abnormal-classes",
        type=str,
        default="benign,malignant",
        help="异常类别的目录名（逗号分隔）"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 创建输出目录
    (output_dir / "normal").mkdir(parents=True, exist_ok=True)
    (output_dir / "abnormal").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("从三分类数据创建二分类数据")
    print("=" * 60)
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"正常类别: {args.normal_class}")
    print(f"异常类别: {args.abnormal_classes}")
    print("=" * 60)

    # 复制正常类别
    normal_src = input_dir / args.normal_class
    normal_dst = output_dir / "normal"

    if not normal_src.exists():
        print(f"警告: 正常类别目录 {normal_src} 不存在")
        return

    normal_files = list(normal_src.glob("*.npy"))
    print(f"复制正常类别: {len(normal_files)} 个文件")
    for src_file in tqdm(normal_files, desc="正常类别"):
        shutil.copy2(src_file, normal_dst / src_file.name)

    # 复制异常类别（合并 benign 和 malignant）
    abnormal_classes = args.abnormal_classes.split(",")
    abnormal_dst = output_dir / "abnormal"

    total_abnormal = 0
    for cls in abnormal_classes:
        cls_src = input_dir / cls.strip()
        if not cls_src.exists():
            print(f"警告: 异常类别目录 {cls_src} 不存在")
            continue

        cls_files = list(cls_src.glob("*.npy"))
        print(f"复制 {cls} 到异常类别: {len(cls_files)} 个文件")

        for src_file in tqdm(cls_files, desc=f"异常类别 ({cls})"):
            # 重命名文件以避免重名
            new_name = f"{cls}_{src_file.name}"
            shutil.copy2(src_file, abnormal_dst / new_name)
            total_abnormal += 1

    print("=" * 60)
    print("数据统计:")
    print(f"  正常类别: {len(normal_files)} 个文件")
    print(f"  异常类别: {total_abnormal} 个文件")
    print(f"  总计: {len(normal_files) + total_abnormal} 个文件")
    print("=" * 60)
    print("完成!")
    print("二分类数据已创建在:")
    print(f"  正常类别: {normal_dst}")
    print(f"  异常类别: {abnormal_dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
