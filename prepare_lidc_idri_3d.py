#!/usr/bin/env python3
"""将 LIDC-IDRI 3D 体数据预处理为可训练的 `.npy` 体块。

输入目录结构（按类别分目录）:
root/
  normal/
  benign/
  malignant/

支持输入文件：.npy / .mhd / .nii / .nii.gz
输出目录同结构，文件保存为 `.npy`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_volume(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)

    if path.name.lower().endswith(".nii.gz") or suffix in {".nii", ".mhd"}:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
        return arr

    raise ValueError(f"Unsupported volume format: {path}")


def preprocess(arr: np.ndarray, depth: int, hw: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported volume shape: {arr.shape}")

    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    t = torch.nn.functional.interpolate(
        t,
        size=(depth, hw, hw),
        mode="trilinear",
        align_corners=False,
    )
    out = t.squeeze(0).squeeze(0).numpy().astype(np.float32)
    return out


def build_output_stem(src_dir: Path, path: Path) -> str:
    rel = path.relative_to(src_dir)
    rel_no_suffix = rel
    if rel.name.lower().endswith(".nii.gz"):
        rel_no_suffix = rel.with_name(rel.name[:-7])
    else:
        rel_no_suffix = rel.with_suffix("")
    parts = [part for part in rel_no_suffix.parts if part not in {"."}]
    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LIDC-IDRI 3D volumes into normalized .npy tensors")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--depth-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cls_names = ["normal", "benign", "malignant"]
    count = 0

    for cls_name in cls_names:
        src_dir = input_root / cls_name
        if not src_dir.exists():
            continue
        dst_dir = output_root / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for p in src_dir.rglob("*"):
            if not p.is_file():
                continue
            if not (
                p.suffix.lower() in {".npy", ".mhd", ".nii"}
                or p.name.lower().endswith(".nii.gz")
            ):
                continue
            vol = load_volume(p)
            vol = preprocess(vol, depth=args.depth_size, hw=args.image_size)
            out_stem = build_output_stem(src_dir, p)
            out_path = dst_dir / f"{out_stem}.npy"
            np.save(out_path, vol)
            count += 1

    print(f"Done. Saved {count} volumes to: {output_root}")


if __name__ == "__main__":
    main()
