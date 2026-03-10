#!/usr/bin/env python3
"""
LUNA16 数据预处理脚本：提取2D切片并准备分类数据集

该脚本会：
1. 读取 LUNA16 原始 MHD/RAW 文件
2. 提取并保存 2D 切片
3. 根据结节标注为切片分配标签
4. 创建数据集目录结构

注意：LUNA16 主要是检测数据集，没有直接的良恶性标注。
这里我们使用简化的分类策略：
- 不含结节的切片 -> 正常 (normal, 0)
- 仅含小结节（直径<6mm）的切片 -> 良性 (benign, 1)
- 含大结节（直径>=6mm）的切片 -> 恶性 (malignant, 2)
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def load_annotations(annotations_path: Path) -> Dict[str, pd.DataFrame]:
    """加载 LUNA16 标注数据并按 seriesuid 分组."""
    df = pd.read_csv(annotations_path)
    # 按 seriesuid 分组
    annotations = {}
    for seriesuid, group in df.groupby("seriesuid"):
        annotations[seriesuid] = group
    return annotations


def load_mhd_image(mhd_path: Path) -> Tuple[np.ndarray, Dict]:
    """加载 MHD/RAW 数据."""
    image = sitk.ReadImage(str(mhd_path))
    array = sitk.GetArrayFromImage(image)
    metadata = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
    }
    return array, metadata


def is_nodule_in_slice(nodule_z: float, slice_idx: int, spacing: Tuple[float, float, float],
                        origin: Tuple[float, float, float]) -> bool:
    """判断结节是否在当前切片中（简化方法）."""
    z_spacing = spacing[2]
    z_origin = origin[2]
    slice_z = z_origin + slice_idx * z_spacing
    # 结节中心距离切片中心小于一个层厚视为在切片中
    return abs(nodule_z - slice_z) < z_spacing / 2


def normalize_ct_slice(slice_img: np.ndarray) -> np.ndarray:
    """标准化 CT 切片（肺部窗宽窗位）."""
    # 肺部窗宽窗位: 窗宽 1500, 窗位 -600
    min_val, max_val = -1200, 300
    slice_img = np.clip(slice_img, min_val, max_val)
    # 归一化到 0-255
    slice_img = (slice_img - min_val) / (max_val - min_val) * 255
    return slice_img.astype(np.uint8)


def get_slice_label(slice_idx: int, annotations: pd.DataFrame,
                   spacing: Tuple[float, float, float],
                   origin: Tuple[float, float, float]) -> int:
    """根据标注确定切片的标签.

    标签策略:
    - 0: 无结节 (normal)
    - 1: 仅有小结节 (直径 < 6mm, benign)
    - 2: 有大结节 (直径 >= 6mm, malignant)
    """
    max_diameter = 0.0
    has_nodule = False

    for _, ann in annotations.iterrows():
        if is_nodule_in_slice(ann["coordZ"], slice_idx, spacing, origin):
            has_nodule = True
            diameter = ann["diameter_mm"]
            if diameter > max_diameter:
                max_diameter = diameter

    if not has_nodule:
        return 0
    elif max_diameter < 6.0:
        return 1
    else:
        return 2


def process_series(mhd_path: Path, annotations: pd.DataFrame,
                  output_dir: Path, slices_per_series: int = 20) -> List[Tuple[Path, int]]:
    """处理一个 CT 序列并保存切片."""
    try:
        ct_array, metadata = load_mhd_image(mhd_path)
    except Exception as e:
        print(f"  加载失败 {mhd_path}: {e}")
        return []

    series_uid = mhd_path.stem
    spacing = metadata["spacing"]
    origin = metadata["origin"]
    num_slices = ct_array.shape[0]

    # 选择切片：不是每个切片都处理，避免数据不均衡
    slice_indices = np.linspace(num_slices // 4, num_slices * 3 // 4,
                                  min(slices_per_series, num_slices // 2)).astype(int)

    saved_samples = []

    for slice_idx in slice_indices:
        if slice_idx < 0 or slice_idx >= num_slices:
            continue

        # 获取标签
        label = get_slice_label(slice_idx, annotations, spacing, origin)

        # 提取并保存切片
        slice_img = ct_array[slice_idx]
        slice_img = normalize_ct_slice(slice_img)

        # 创建目录
        label_dir = output_dir / ["normal", "benign", "malignant"][label]
        label_dir.mkdir(exist_ok=True)

        slice_path = label_dir / f"{series_uid}_slice{slice_idx:04d}.png"
        Image.fromarray(slice_img).save(slice_path)
        saved_samples.append((slice_path, label))

    return saved_samples


def main():
    parser = argparse.ArgumentParser(description="准备 LUNA16 2D 分类数据集")
    parser.add_argument("--luna16-root", type=str, required=True,
                        help="LUNA16 数据集根目录")
    parser.add_argument("--output-dir", type=str, default="/workspace/data-lung/luna16/extracted_slices",
                        help="输出目录")
    parser.add_argument("--slices-per-series", type=int, default=20,
                        help="每个序列提取的切片数")
    parser.add_argument("--subsets", type=str, default="0,1,2",
                        help="使用的子集（逗号分隔）")
    args = parser.parse_args()

    if not HAS_SITK:
        raise ImportError("需要 SimpleITK，请运行: pip install SimpleITK")

    luna16_root = Path(args.luna16_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注
    annotations_path = luna16_root / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotations_path}")
    annotations = load_annotations(annotations_path)
    print(f"加载了 {len(annotations)} 个序列的标注")

    # 确定要处理的子集
    subset_ids = args.subsets.split(",")
    print(f"处理子集: {subset_ids}")

    # 收集所有 MHD 文件
    mhd_files = []
    for subset_id in subset_ids:
        subset_dir = luna16_root / f"subset{subset_id}"
        if not subset_dir.exists():
            print(f"警告: 子集目录不存在: {subset_dir}")
            continue
        # 检查是否在嵌套目录中
        nested_subset_dir = subset_dir / f"subset{subset_id}"
        if nested_subset_dir.exists():
            subset_dir = nested_subset_dir
        mhd_files.extend(subset_dir.glob("*.mhd"))

    print(f"找到 {len(mhd_files)} 个 MHD 文件")

    # 处理每个文件
    all_samples = []
    for i, mhd_path in enumerate(mhd_files, 1):
        series_uid = mhd_path.stem
        print(f"[{i}/{len(mhd_files)}] 处理: {series_uid}")

        series_annotations = annotations.get(series_uid, pd.DataFrame())
        samples = process_series(
            mhd_path, series_annotations, output_dir,
            slices_per_series=args.slices_per_series
        )
        all_samples.extend(samples)

    # 统计
    label_counts = [0, 0, 0]
    for _, label in all_samples:
        label_counts[label] += 1

    print("\n" + "=" * 60)
    print("数据集统计:")
    print(f"  正常 (normal): {label_counts[0]}")
    print(f"  良性 (benign): {label_counts[1]}")
    print(f"  恶性 (malignant): {label_counts[2]}")
    print(f"  总计: {len(all_samples)} 张切片")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 保存样本列表（可选）
    sample_list_path = output_dir / "samples.csv"
    with open(sample_list_path, "w") as f:
        f.write("path,label\n")
        for path, label in all_samples:
            f.write(f"{path},{label}\n")
    print(f"样本列表已保存到: {sample_list_path}")


if __name__ == "__main__":
    main()
