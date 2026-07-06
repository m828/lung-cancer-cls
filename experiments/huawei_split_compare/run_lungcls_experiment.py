#!/usr/bin/env python3
"""Run lung-cancer-cls on a fixed Huawei split manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from huawei_split_tools import DEFAULT_CT_CSV, DEFAULT_CT_ROOT, DEFAULT_GENE_TSV, modality_counts, split_label_distribution


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch lung-cancer-cls with a fixed Huawei comparison split.")
    parser.add_argument("--split_name", choices=["huawei_8_2", "split_7_1_2"], required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ct_csv", type=Path, default=DEFAULT_CT_CSV)
    parser.add_argument("--ct_root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--gene_tsv", type=Path, default=DEFAULT_GENE_TSV)
    parser.add_argument("--text_feature_tsv", type=Path, default=None)
    parser.add_argument("--text_health_csv", type=Path, default=None)
    parser.add_argument("--text_disease_csv", type=Path, default=None)
    parser.add_argument("--bert_model_path", type=Path, default=None)
    parser.add_argument("--text_embedding_backend", choices=["bert", "hash"], default="bert")
    parser.add_argument("--base_output_dir", type=Path, default=EXP_DIR / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--modalities", type=str, default="ct,text,cnv")
    parser.add_argument("--ct_model", type=str, default="resnet3d18")
    parser.add_argument("--metadata_sample_id_col", type=str, default="SampleID")
    parser.add_argument("--metadata_text_id_col", type=str, default="样本编号")
    parser.add_argument("--text_record_id_col", type=str, default="样本编号")
    parser.add_argument("--gene_id_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default="样本类型")
    parser.add_argument("--split_col", type=str, default="CT_train_val_split")
    parser.add_argument("--ct_path_col", type=str, default="CT_numpy_cloud路径")
    parser.add_argument("--strict_no_leakage", action="store_true")
    parser.add_argument("--disable_3d_input", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest:
        return args.manifest.resolve()
    return (EXP_DIR / "splits" / args.split_name / "split_manifest.csv").resolve()


def write_input_audit(manifest: Path, output_dir: Path) -> None:
    df = pd.read_csv(manifest).fillna("")
    audit = {
        "manifest": str(manifest),
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "label_distribution": split_label_distribution(df),
        "modality_counts": modality_counts(df),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_command(args: argparse.Namespace, manifest: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train_multimodal.py"),
        "--data-root",
        str(args.ct_root),
        "--metadata-csv",
        str(args.ct_csv),
        "--ct-root",
        str(args.ct_root),
        "--output-dir",
        str(output_dir),
        "--modalities",
        args.modalities,
        "--reference-manifest",
        str(manifest),
        "--label-col",
        args.label_col,
        "--metadata-sample-id-col",
        args.metadata_sample_id_col,
        "--metadata-text-id-col",
        args.metadata_text_id_col,
        "--split-col",
        args.split_col,
        "--ct-path-col",
        args.ct_path_col,
        "--text-record-id-col",
        args.text_record_id_col,
        "--class-mode",
        "multiclass",
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--ct-model",
        args.ct_model,
    ]

    modalities = {part.strip() for part in args.modalities.split(",") if part.strip()}
    if "cnv" in modalities:
        cmd.extend(["--gene-tsv", str(args.gene_tsv)])
        if args.gene_id_col:
            cmd.extend(["--gene-id-col", args.gene_id_col])

    if "text" in modalities:
        if args.text_feature_tsv and args.text_feature_tsv.exists():
            cmd.extend(["--text-feature-tsv", str(args.text_feature_tsv)])
        else:
            if args.text_feature_tsv:
                raise FileNotFoundError(f"text_feature_tsv not found: {args.text_feature_tsv}")
            if not (args.text_health_csv and args.text_disease_csv):
                raise RuntimeError(
                    "Text modality is enabled, but no usable text source was provided. "
                    "Pass --text_feature_tsv, or pass both --text_health_csv and --text_disease_csv. "
                    "For Huawei cache, first run export_huawei_text_features.py."
                )
            if args.text_health_csv:
                cmd.extend(["--text-health-csv", str(args.text_health_csv)])
            if args.text_disease_csv:
                cmd.extend(["--text-disease-csv", str(args.text_disease_csv)])
            if args.bert_model_path:
                cmd.extend(["--bert-model-path", str(args.bert_model_path)])
            cmd.extend(["--text-embedding-backend", args.text_embedding_backend])

    if args.strict_no_leakage:
        cmd.append("--strict-no-leakage")
    if args.disable_3d_input:
        cmd.append("--disable-3d-input")
    return cmd


def main() -> None:
    args = parse_args()
    manifest = resolve_manifest(args)
    if not manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest}. Run generate_split_manifests.py first.")

    output_dir = (args.base_output_dir / args.split_name / f"seed_{args.seed}" / args.modalities.replace(",", "_")).resolve()
    write_input_audit(manifest, output_dir)
    cmd = build_command(args, manifest, output_dir)
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    print("[run_lungcls_experiment] command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
