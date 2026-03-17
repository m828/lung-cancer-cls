#!/usr/bin/env python3
"""
从 LUNA16 extracted_slices 创建 3D 体积数据集的简化版本。
将同一病例的连续切片合并成 3D 体积。
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image


def extract_case_uid_from_filename(filename: str) -> str:
    """从切片文件名中提取病例ID。格式类似：1.3.6.1.4.1.14519.5.2.1.6279.6001.313835996725364342034830119490_slice0116.png"""
    if "_slice" in filename:
        case_uid = filename.split("_slice")[0]
        return case_uid
    return filename.split(".")[0]


def group_slices_by_case(slice_dir: Path) -> dict:
    """按病例分组切片。"""
    case_groups = defaultdict(list)
    slice_paths = list(slice_dir.glob("*.png"))
    print(f"找到 {len(slice_paths)} 个切片")

    for path in slice_paths:
        case_uid = extract_case_uid_from_filename(path.name)
        case_groups[case_uid].append(path)

    print(f"包含 {len(case_groups)} 个病例")
    return case_groups


def create_3d_volume_from_slices(slice_paths, target_depth=32):
    """将连续切片合并成3D体积。"""
    # 按切片号排序
    def get_slice_number(path):
        if "_slice" in path.name:
            try:
                slice_part = path.name.split("_slice")[1]
                slice_num = int(slice_part.split(".")[0])
                return slice_num
            except:
                pass
        return 0

    sorted_paths = sorted(slice_paths, key=get_slice_number)

    # 选择中间的 target_depth 个切片
    num_slices = len(sorted_paths)
    if num_slices < target_depth:
        # 切片不足，复制填充
        repeated_paths = []
        for i in range(target_depth):
            repeated_paths.append(sorted_paths[i % num_slices])
        sorted_paths = repeated_paths
    else:
        # 取中间的 target_depth 个切片
        start_idx = (num_slices - target_depth) // 2
        sorted_paths = sorted_paths[start_idx:start_idx + target_depth]

    # 读取并转换为灰度
    volume = []
    for path in sorted_paths:
        img = Image.open(path).convert("L")
        volume.append(np.array(img))

    volume = np.array(volume)

    # 调整大小为 128x128
    from torch import nn
    import torch
    vol_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resized = nn.functional.interpolate(
        vol_tensor,
        size=(target_depth, 128, 128),
        mode='trilinear',
        align_corners=False
    )
    return resized.squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description="从 LUNA16 2D 切片创建 3D 体积数据集")
    parser.add_argument("--input-root", type=str, default="/workspace/data-lung/luna16/extracted_slices",
                        help="LUNA16 提取的 2D 切片根目录")
    parser.add_argument("--output-root", type=str, default="/workspace/data-lung/luna16/3d_volumes",
                        help="输出的 3D 体积根目录")
    parser.add_argument("--depth-size", type=int, default=32, help="目标深度")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)

    # 处理每个类别
    for cls_name in ["normal", "benign", "malignant"]:
        src_dir = input_root / cls_name
        if not src_dir.exists():
            print(f"警告: 未找到类别目录 {src_dir}")
            continue

        dst_dir = output_root / cls_name
        dst_dir.mkdir(exist_ok=True)

        case_groups = group_slices_by_case(src_dir)

        for case_uid, slice_paths in case_groups.items():
            print(f"处理类别 {cls_name} 的病例 {case_uid}，包含 {len(slice_paths)} 个切片")
            volume = create_3d_volume_from_slices(slice_paths, args.depth_size)

            # 标准化
            volume = volume / 255.0
            volume = (volume - 0.5) / 0.5

            output_path = dst_dir / f"{case_uid}.npy"
            np.save(output_path, volume.astype(np.float32))

    # 统计数量
    print("\n" + "="*60)
    for cls_name in ["normal", "benign", "malignant"]:
        cls_dir = output_root / cls_name
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.npy")))
            print(f"  {cls_name}: {count} 个体积")
    print("="*60)


if __name__ == "__main__":
    main()
