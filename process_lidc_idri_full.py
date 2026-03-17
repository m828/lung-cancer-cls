#!/usr/bin/env python3
"""
LIDC-IDRI 完整数据预处理脚本。

对 LIDC-IDRI 原始 DICOM 数据进行预处理：
1. 解析 XML 标注获取恶性度评分
2. 对 DICOM 进行 HU 值截断、归一化
3. 重采样到统一尺寸
4. 按三分类（normal/benign/malignant）划分
5. 保存为 npy 文件
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm


def parse_malignancy_from_xml(xml_path: Path) -> Optional[List[int]]:
    """
    从 LIDC-IDRI XML 标注文件中解析恶性度评分。

    malignancy 评分: 1=高度良性, 2=中度良性, 3=不确定, 4=中度恶性, 5=高度恶性

    返回: 所有结节的恶性度评分列表
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 定义命名空间
        ns = {'nih': 'http://www.nih.gov'}

        malignancies = []

        # 查找所有结节
        for nodule in root.findall('.//nih:unblindedReadNodule', ns):
            # 查找恶性度特征
            chars = nodule.find('nih:characteristics', ns)
            if chars is not None:
                malignancy_elem = chars.find('nih:malignancy', ns)
                if malignancy_elem is not None and malignancy_elem.text:
                    try:
                        mal = int(malignancy_elem.text.strip())
                        malignancies.append(mal)
                    except ValueError:
                        pass

        return malignancies if malignancies else None
    except Exception as e:
        print(f"  解析 XML 失败 {xml_path}: {e}")
        return None


def determine_label_3class(malignancies: List[int]) -> int:
    """
    根据恶性度评分确定三分类标签。

    策略:
    - 如果没有结节: normal(0)
    - 如果所有结节的最大恶性度 <= 2: benign(1)
    - 如果有结节的恶性度 >= 4: malignant(2)
    - 如果只有 3 (不确定)，偏向良性
    """
    if not malignancies:
        return 0  # normal

    max_mal = max(malignancies)

    if max_mal <= 2:
        return 1  # benign
    elif max_mal >= 4:
        return 2  # malignant
    else:  # max_mal == 3
        return 1  # 偏向良性


def load_dicom_series_as_volume(series_dir: Path) -> Optional[np.ndarray]:
    """
    从 DICOM 序列目录加载 3D 体积数据。
    """
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_names:
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)  # [D, H, W]
        return arr.astype(np.float32)
    except Exception as e:
        print(f"  加载 DICOM 失败 {series_dir}: {e}")
        return None


def preprocess_volume(arr: np.ndarray, hu_min: int = -1000, hu_max: int = 400) -> np.ndarray:
    """
    预处理 CT 体积数据：HU值截断、归一化。

    Args:
        arr: 原始 CT 体积
        hu_min: 最低 HU 值
        hu_max: 最高 HU 值

    Returns:
        预处理后的体积（[0,1] 范围）
    """
    # HU 值截断
    arr = np.clip(arr, hu_min, hu_max)
    # 归一化到 [0,1]
    arr = (arr - hu_min) / (hu_max - hu_min)
    return arr.astype(np.float32)


def resample_volume(arr: np.ndarray, target_depth: int = 32, target_hw: int = 128) -> np.ndarray:
    """重采样到统一尺寸"""
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported volume shape: {arr.shape}")

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    resized = torch.nn.functional.interpolate(
        t,
        size=(target_depth, target_hw, target_hw),
        mode='trilinear',
        align_corners=False
    )
    return resized.squeeze().numpy()


def find_ct_series_and_xmls(patient_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    查找患者目录下的 CT 序列和对应的 XML 标注文件。
    返回: [(ct_series_dir, xml_file), ...]
    """
    results = []

    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue

        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue

            # 检查是否是 CT 序列（有 DICOM 文件）
            dicom_files = list(series_dir.glob("*.dcm"))
            if not dicom_files:
                continue

            # 查找 XML 标注
            xml_files = list(series_dir.glob("*.xml"))
            xml_file = xml_files[0] if xml_files else None

            results.append((series_dir, xml_file))

    return results


def main():
    parser = argparse.ArgumentParser(description="LIDC-IDRI 完整数据预处理")
    parser.add_argument("--input-root", type=str, default="/workspace/data-lung/LIDC-IDRI/LIDC-IDRI",
                        help="LIDC-IDRI 原始数据根目录")
    parser.add_argument("--output-root", type=str, default="/workspace/data-lung/lidc_idri_full_3class",
                        help="输出根目录")
    parser.add_argument("--target-depth", type=int, default=32, help="目标深度")
    parser.add_argument("--target-hw", type=int, default=128, help="目标高度/宽度")
    parser.add_argument("--hu-min", type=int, default=-1000, help="HU 值截断下限")
    parser.add_argument("--hu-max", type=int, default=400, help="HU 值截断上限")
    parser.add_argument("--max-patients", type=int, default=10, help="最大处理患者数（用于测试）")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for cls_name in ["normal", "benign", "malignant"]:
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("LIDC-IDRI-")])
    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]

    print(f"找到 {len(patient_dirs)} 个患者目录")
    print("=" * 80)

    stats = defaultdict(int)
    skipped = 0

    for patient_dir in tqdm(patient_dirs, desc="处理患者"):
        patient_id = patient_dir.name

        # 查找 CT 序列和 XML
        ct_xml_pairs = find_ct_series_and_xmls(patient_dir)

        if not ct_xml_pairs:
            skipped += 1
            continue

        for ct_idx, (ct_dir, xml_file) in enumerate(ct_xml_pairs):
            # 解析标注
            malignancies = None
            if xml_file:
                malignancies = parse_malignancy_from_xml(xml_file)

            label = determine_label_3class(malignancies) if malignancies else 0

            # 加载和预处理
            volume = load_dicom_series_as_volume(ct_dir)
            if volume is None:
                skipped += 1
                continue

            volume = preprocess_volume(volume, args.hu_min, args.hu_max)
            volume = resample_volume(volume, args.target_depth, args.target_hw)

            # 保存
            class_names = ["normal", "benign", "malignant"]
            cls_name = class_names[label]
            out_path = output_root / cls_name / f"{patient_id}_ct{ct_idx}.npy"
            np.save(out_path, volume)
            stats[label] += 1

    print("\n" + "=" * 80)
    print("数据统计:")
    class_names = ["normal", "benign", "malignant"]
    for label, count in sorted(stats.items()):
        print(f"  {class_names[label]} (label={label}): {count} 个体积")

    print(f"\n跳过: {skipped} 个序列")
    print(f"预处理完成!")
    print(f"输出目录: {output_root}")
    print(f"训练命令示例:")
    print(f"  python train.py --dataset-type lidc_idri --data-root {output_root} --output-dir outputs/lidc_full --model attention3d_cnn --split-mode train_val --epochs 5 --batch-size 4 --image-size 128")


if __name__ == "__main__":
    main()
