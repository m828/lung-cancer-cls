#!/usr/bin/env python3
"""
准备 LIDC-IDRI 标注数据（二分类和三分类）。

从原始 LIDC-IDRI 数据中提取 CT 体积并根据结节恶性度标注：
- 三分类：normal(0) / benign(1) / malignant(2)
- 二分类：normal(0) / abnormal(1) 或 benign(0) / malignant(1)
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
                        mal = int(malignancy_elem.text)
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
    - 如果只有 3 (不确定)，偏向 benign(1)
    """
    if not malignancies:
        return 0  # normal

    max_mal = max(malignancies)

    if max_mal <= 2:
        return 1  # benign
    elif max_mal >= 4:
        return 2  # malignant
    else:  # max_mal == 3
        # 只有不确定的，偏向良性
        return 1


def determine_label_2class(malignancies: List[int], mode: str = "normal_abnormal") -> int:
    """
    根据恶性度评分确定二分类标签。

    mode:
    - "normal_abnormal": normal(0) vs abnormal(1) (有任何结节都算异常)
    - "benign_malignant": benign(0) vs malignant(1) (只看有结节的情况)
    """
    if mode == "normal_abnormal":
        return 0 if not malignancies else 1
    else:  # benign_malignant
        if not malignancies:
            return -1  # skip (无结节)
        max_mal = max(malignancies)
        return 0 if max_mal <= 3 else 1


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


def preprocess_volume(arr: np.ndarray, depth: int, hw: int) -> np.ndarray:
    """预处理体积：归一化 + 调整大小"""
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported volume shape: {arr.shape}")

    # 归一化到 [0, 1]
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val

    # 调整大小
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    t = torch.nn.functional.interpolate(
        t,
        size=(depth, hw, hw),
        mode="trilinear",
        align_corners=False,
    )
    out = t.squeeze(0).squeeze(0).numpy().astype(np.float32)
    return out


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
                # 检查子目录
                for subdir in series_dir.iterdir():
                    if subdir.is_dir():
                        sub_dicoms = list(subdir.glob("*.dcm"))
                        if sub_dicoms:
                            # 查找 XML
                            xml_file = None
                            for xml_candidate in series_dir.glob("*.xml"):
                                xml_file = xml_candidate
                                break
                            for xml_candidate in subdir.glob("*.xml"):
                                xml_file = xml_candidate
                                break
                            results.append((subdir, xml_file))
            else:
                # 查找 XML
                xml_file = None
                for xml_candidate in series_dir.glob("*.xml"):
                    xml_file = xml_candidate
                    break
                results.append((series_dir, xml_file))

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare LIDC-IDRI annotated 3D dataset")
    parser.add_argument("--input-root", type=str, default="/workspace/data-lung/LIDC-IDRI/LIDC-IDRI",
                        help="LIDC-IDRI 原始数据根目录")
    parser.add_argument("--output-root-3class", type=str, default="/workspace/data-lung/lidc_idri_3class",
                        help="三分类输出根目录")
    parser.add_argument("--output-root-2class", type=str, default="/workspace/data-lung/lidc_idri_2class",
                        help="二分类输出根目录")
    parser.add_argument("--2class-mode", type=str, choices=["normal_abnormal", "benign_malignant"],
                        default="normal_abnormal", help="二分类模式")
    parser.add_argument("--depth-size", type=int, default=32, help="目标深度")
    parser.add_argument("--image-size", type=int, default=128, help="目标图像大小")
    parser.add_argument("--max-patients", type=int, default=None, help="最大处理患者数（用于测试）")
    args = parser.parse_args()

    input_root = Path(args.input_root)

    # 创建输出目录
    output_root_3class = Path(args.output_root_3class)
    output_root_2class = Path(args.output_root_2class)

    for cls_name in ["normal", "benign", "malignant"]:
        (output_root_3class / cls_name).mkdir(parents=True, exist_ok=True)

    if args.two_class_mode == "normal_abnormal":
        for cls_name in ["normal", "abnormal"]:
            (output_root_2class / cls_name).mkdir(parents=True, exist_ok=True)
    else:
        for cls_name in ["benign", "malignant"]:
            (output_root_2class / cls_name).mkdir(parents=True, exist_ok=True)

    # 获取所有患者目录
    patient_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("LIDC-IDRI-")])
    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]

    print(f"找到 {len(patient_dirs)} 个患者目录")
    print("=" * 80)

    stats_3class = defaultdict(int)
    stats_2class = defaultdict(int)
    skipped = 0

    for patient_dir in tqdm(patient_dirs, desc="处理患者"):
        patient_id = patient_dir.name

        # 查找 CT 序列和 XML
        ct_xml_pairs = find_ct_series_and_xmls(patient_dir)

        if not ct_xml_pairs:
            skipped += 1
            continue

        # 对每个 CT 序列处理
        for ct_idx, (ct_series_dir, xml_file) in enumerate(ct_xml_pairs):
            # 解析恶性度
            malignancies = None
            if xml_file:
                malignancies = parse_malignancy_from_xml(xml_file)

            # 确定标签
            label_3class = determine_label_3class(malignancies) if malignancies else 0
            label_2class = determine_label_2class(malignancies, args.two_class_mode) if malignancies else 0

            # 跳过二分类中不需要的样本
            if args.two_class_mode == "benign_malignant" and label_2class == -1:
                continue

            # 加载和预处理 CT
            volume = load_dicom_series_as_volume(ct_series_dir)
            if volume is None:
                skipped += 1
                continue

            try:
                volume = preprocess_volume(volume, args.depth_size, args.image_size)
            except Exception as e:
                print(f"  预处理失败 {patient_id}: {e}")
                skipped += 1
                continue

            # 保存三分类
            class_names_3class = ["normal", "benign", "malignant"]
            cls_name_3class = class_names_3class[label_3class]
            out_path_3class = output_root_3class / cls_name_3class / f"{patient_id}_{ct_idx}.npy"
            np.save(out_path_3class, volume)
            stats_3class[label_3class] += 1

            # 保存二分类
            if args.two_class_mode == "normal_abnormal":
                class_names_2class = ["normal", "abnormal"]
            else:
                class_names_2class = ["benign", "malignant"]
            cls_name_2class = class_names_2class[label_2class]
            out_path_2class = output_root_2class / cls_name_2class / f"{patient_id}_{ct_idx}.npy"
            np.save(out_path_2class, volume)
            stats_2class[label_2class] += 1

    # 打印统计
    print("\n" + "=" * 80)
    print("三分类统计:")
    class_names_3class = ["normal", "benign", "malignant"]
    for label, count in sorted(stats_3class.items()):
        print(f"  {class_names_3class[label]} (label={label}): {count} 个体积")

    print(f"\n二分类统计 (模式={args.two_class_mode}):")
    if args.two_class_mode == "normal_abnormal":
        class_names_2class = ["normal", "abnormal"]
    else:
        class_names_2class = ["benign", "malignant"]
    for label, count in sorted(stats_2class.items()):
        print(f"  {class_names_2class[label]} (label={label}): {count} 个体积")

    print(f"\n跳过: {skipped} 个序列")
    print(f"三分类数据保存到: {output_root_3class}")
    print(f"二分类数据保存到: {output_root_2class}")
    print("=" * 80)


if __name__ == "__main__":
    main()
