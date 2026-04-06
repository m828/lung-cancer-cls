#!/usr/bin/env python3
"""Build a literature-style LIDC-IDRI split manifest from raw data and metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.lidc_split import (  # noqa: E402
    LIDCSplitConfig,
    build_and_write_lidc_splits,
    resolve_patient_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build patient-wise LIDC-IDRI split manifests using the literature-common "
            "1-2 benign vs 4-5 malignant protocol."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/data-lung/LIDC-IDRI"),
        help="Root directory that contains metadata.csv and/or the nested LIDC-IDRI patient folders.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional metadata.csv path. Defaults to <input-root>/metadata.csv when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive nodule_manifest.csv, split_manifest.csv and summary.json.",
    )
    parser.add_argument(
        "--metadata-source",
        type=str,
        choices=["auto", "csv", "xml"],
        default="auto",
        help="Prefer metadata.csv, raw XML, or auto-detect. metadata.csv is recommended when available.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["benign_vs_malignant"],
        default="benign_vs_malignant",
        help="Currently the script targets the common LIDC benign-vs-malignant benchmark.",
    )
    parser.add_argument(
        "--label-policy",
        type=str,
        choices=["score12_vs_score45", "score123_vs_score45", "score12_vs_score345"],
        default="score12_vs_score45",
        help="How malignancy scores are collapsed into the binary benchmark labels.",
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["patient_kfold", "patient_holdout"],
        default="patient_kfold",
        help="Literature-style 5-fold CV or a single patient-wise holdout split.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of outer folds for patient_kfold.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio used by patient_holdout. The remainder is split equally into val/test.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help=(
            "Validation ratio inside each training fold. This is used for early stopping and "
            "checkpoint selection after the outer patient-wise split is built."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patient-col", type=str, default=None, help="Optional explicit patient column name.")
    parser.add_argument("--nodule-col", type=str, default=None, help="Optional explicit nodule column name.")
    parser.add_argument("--malignancy-col", type=str, default=None, help="Optional explicit malignancy column name.")
    parser.add_argument("--path-col", type=str, default=None, help="Optional explicit path column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        candidate = args.input_root / "metadata.csv"
        metadata_csv = candidate if candidate.exists() else None

    config = LIDCSplitConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        metadata_csv=metadata_csv,
        metadata_source=args.metadata_source,
        task=args.task,
        label_policy=args.label_policy,
        split_scheme=args.split_scheme,
        n_splits=args.n_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        patient_col=args.patient_col,
        nodule_col=args.nodule_col,
        malignancy_col=args.malignancy_col,
        path_col=args.path_col,
    )

    patient_root = resolve_patient_root(config.input_root)
    print("=" * 72)
    print("LIDC-IDRI 文献风格 split manifest 生成")
    print("=" * 72)
    print(f"input_root:     {config.input_root}")
    print(f"patient_root:   {patient_root}")
    print(f"metadata_csv:   {config.metadata_csv if config.metadata_csv is not None else '(none)'}")
    print(f"metadata_source:{config.metadata_source}")
    print(f"label_policy:   {config.label_policy}")
    print(f"split_scheme:   {config.split_scheme}")
    print(f"output_dir:     {config.output_dir}")
    print("=" * 72)

    outputs = build_and_write_lidc_splits(config)
    print("Done.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print("\n建议下一步：")
    print("1. 先打开 summary.json 检查每折的 benign / malignant / patient 数量。")
    print("2. 再据此做结节裁剪或 3D 预处理，确保输出文件名保留 patient_id / nodule_id。")
    print("3. 训练时固定使用这份 split_manifest.csv，不要再让训练脚本临时随机切分。")


if __name__ == "__main__":
    main()
