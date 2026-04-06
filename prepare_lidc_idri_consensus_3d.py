#!/usr/bin/env python3
"""Preprocess raw LIDC-IDRI consensus nodules into cropped 3D `.npy` tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.lidc_consensus_preprocess import (  # noqa: E402
    LIDCCropConfig,
    preprocess_lidc_consensus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop consensus LIDC-IDRI nodules from raw DICOM series according to "
            "nodule_manifest.csv and export fixed-size 3D .npy tensors."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/data-lung/LIDC-IDRI"),
        help="Raw LIDC-IDRI root that contains the nested patient folders.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="Consensus nodule_manifest.csv produced by build_lidc_idri_split_manifest.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory that will receive benign/ and malignant/ .npy crops.",
    )
    parser.add_argument(
        "--split-manifest-csv",
        type=Path,
        default=None,
        help="Optional split_manifest.csv used to filter/export a specific fold.",
    )
    parser.add_argument(
        "--split-fold",
        type=int,
        default=None,
        help="Optional fold index when split_manifest.csv contains multiple folds.",
    )
    parser.add_argument("--depth-size", type=int, default=32, help="Output depth after trilinear resize.")
    parser.add_argument("--volume-hw", type=int, default=128, help="Output H/W after trilinear resize.")
    parser.add_argument("--hu-min", type=float, default=-1000.0, help="Lower HU clipping bound.")
    parser.add_argument("--hu-max", type=float, default=400.0, help="Upper HU clipping bound.")
    parser.add_argument(
        "--context-scale",
        type=float,
        default=1.5,
        help="Expand the in-plane nodule box by this factor before cropping.",
    )
    parser.add_argument("--min-size-xy", type=int, default=32, help="Minimum crop width/height in pixels.")
    parser.add_argument("--min-size-z", type=int, default=8, help="Minimum crop depth in slices.")
    parser.add_argument(
        "--z-margin-mm",
        type=float,
        default=1.5,
        help="Extra axial margin added to the annotation z-range before slice lookup.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional debug cap on processed samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LIDCCropConfig(
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_depth=args.depth_size,
        target_hw=args.volume_hw,
        context_scale=args.context_scale,
        min_size_xy=args.min_size_xy,
        min_size_z=args.min_size_z,
        z_margin_mm=args.z_margin_mm,
    )

    print("=" * 72)
    print("LIDC-IDRI consensus 3D crop preprocessing")
    print("=" * 72)
    print(f"input_root:         {args.input_root}")
    print(f"manifest_csv:       {args.manifest_csv}")
    print(f"split_manifest_csv: {args.split_manifest_csv if args.split_manifest_csv else '(none)'}")
    print(f"split_fold:         {args.split_fold if args.split_fold is not None else '(all)'}")
    print(f"output_root:        {args.output_root}")
    print(f"target_size:        ({config.target_depth}, {config.target_hw}, {config.target_hw})")
    print(f"context_scale:      {config.context_scale}")
    print("=" * 72)

    outputs = preprocess_lidc_consensus_manifest(
        input_root=args.input_root,
        manifest_csv=args.manifest_csv,
        output_root=args.output_root,
        config=config,
        split_manifest_csv=args.split_manifest_csv,
        split_fold=args.split_fold,
        limit=args.limit,
    )

    print("Done.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
