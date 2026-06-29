#!/usr/bin/env python3
"""Cache teacher logits / probabilities for strict CT+Text KD search.

This script is intentionally independent from train_student_kd.py.  It reads a
trained teacher run, rebuilds the strict-no-leakage cohort/split, forwards the
teacher once over train/val/test, and writes reusable soft targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from lung_cancer_cls.multimodal_teacher_student import (  # noqa: E402
    MultiModalTrainConfig,
    apply_text_feature_safety_after_split,
    build_multimodal_cohort,
    config_from_dict,
    create_model_from_config,
    filter_cohort_to_manifest,
    load_split_manifest_indices,
    load_teacher_bundle,
    move_inputs_to_device,
    normalize_modalities,
    prepare_dataloaders,
    save_split_manifest,
    set_seed,
    split_cohort,
    validate_patient_split_integrity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-name", type=str, required=True)
    p.add_argument("--reference-manifest", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--ct-root", type=Path, default=None)
    p.add_argument("--gene-tsv", type=Path, default=None)
    p.add_argument("--text-feature-tsv", type=Path, default=None)
    p.add_argument("--text-health-csv", type=Path, default=None)
    p.add_argument("--text-disease-csv", type=Path, default=None)
    p.add_argument("--text-cache-tsv", type=Path, default=None)
    p.add_argument("--label-col", type=str, default=None)
    p.add_argument("--metadata-sample-id-col", type=str, default=None)
    p.add_argument("--patient-id-col", type=str, default=None)
    p.add_argument("--metadata-text-id-col", type=str, default=None)
    p.add_argument("--split-col", type=str, default=None)
    p.add_argument("--ct-path-col", type=str, default=None)
    p.add_argument("--text-record-id-col", type=str, default=None)
    p.add_argument("--gene-id-col", type=str, default=None)
    p.add_argument("--gene-label-col", type=str, default=None)
    p.add_argument("--class-mode", type=str, default=None)
    p.add_argument("--binary-task", type=str, default=None)
    p.add_argument("--split-mode", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--strict-no-leakage", action="store_true")
    p.add_argument("--disable-text-numeric-features", action="store_true")
    p.add_argument("--allowed-text-cols", type=str, default=None)
    p.add_argument("--allowed-numeric-cols", type=str, default=None)
    p.add_argument("--forbidden-feature-keywords", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def apply_overrides(config: MultiModalTrainConfig, args: argparse.Namespace) -> MultiModalTrainConfig:
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.cpu = bool(args.cpu)
    config.strict_no_leakage = bool(args.strict_no_leakage)
    config.disable_text_numeric_features = bool(args.disable_text_numeric_features)
    for key in [
        "data_root",
        "metadata_csv",
        "reference_manifest",
        "ct_root",
        "gene_tsv",
        "text_feature_tsv",
        "text_health_csv",
        "text_disease_csv",
        "text_cache_tsv",
        "label_col",
        "metadata_sample_id_col",
        "patient_id_col",
        "metadata_text_id_col",
        "split_col",
        "ct_path_col",
        "text_record_id_col",
        "gene_id_col",
        "gene_label_col",
        "class_mode",
        "binary_task",
        "split_mode",
        "allowed_text_cols",
        "allowed_numeric_cols",
        "forbidden_feature_keywords",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    return config


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def forward_split(model, loader, dataset, split_name: str, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            bs = int(labels.size(0))
            inputs = move_inputs_to_device(inputs, device)
            outputs = model.forward_outputs(inputs) if hasattr(model, "forward_outputs") else {"logits": model(inputs)}
            logits = outputs["logits"].detach().cpu()
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            margin = (probs[:, 1] - probs[:, 0]).abs() if probs.shape[1] == 2 else conf
            for i in range(bs):
                ds_i = offset + i
                y = int(labels[i].item())
                row = {
                    "sample_id": dataset.sample_ids[ds_i],
                    "record_id": dataset.record_ids[ds_i],
                    "split": split_name,
                    "y_true": y,
                    "teacher_logit_0": float(logits[i, 0].item()),
                    "teacher_prob_0": float(probs[i, 0].item()),
                    "teacher_confidence": float(conf[i].item()),
                    "teacher_margin": float(margin[i].item()),
                    "teacher_correct": int(int(pred[i].item()) == y),
                }
                if logits.shape[1] > 1:
                    row["teacher_logit_1"] = float(logits[i, 1].item())
                    row["teacher_prob_1"] = float(probs[i, 1].item())
                rows.append(row)
            offset += bs
    return rows


def main() -> int:
    args = parse_args()
    out_csv = args.output_dir / f"{args.cache_name}.csv"
    out_json = args.output_dir / f"{args.cache_name}.metadata.json"
    if out_csv.exists() and not args.overwrite:
        print(f"[SKIP] cached targets exist: {out_csv}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    teacher_metrics, teacher_config, teacher_modalities, _ = load_teacher_bundle(args.teacher_run_dir)
    teacher_config = config_from_dict(teacher_metrics.get("config", {}), MultiModalTrainConfig)
    teacher_config.modalities = normalize_modalities(teacher_modalities)
    config = apply_overrides(teacher_config, args)
    config.modalities = normalize_modalities(teacher_modalities)

    cohort, feature_info, class_names, cohort_stats = build_multimodal_cohort(config, config.modalities)
    manifest = args.reference_manifest or (args.teacher_run_dir / "split_manifest.csv")
    if manifest.is_file():
        cohort = filter_cohort_to_manifest(cohort, manifest)
        train_idx, val_idx, test_idx = load_split_manifest_indices(cohort, manifest)
        split_source = "reference_manifest" if args.reference_manifest else "teacher_manifest"
    else:
        train_idx, val_idx, test_idx, split_source = split_cohort(cohort, config)
    if not train_idx:
        raise RuntimeError("Training split is empty.")

    leakage_warnings = validate_patient_split_integrity(
        cohort,
        train_idx,
        val_idx,
        test_idx,
        "patient_id" if config.patient_id_col else None,
        strict=config.strict_no_leakage,
    )
    cohort, feature_info, text_warnings = apply_text_feature_safety_after_split(
        config, cohort, feature_info, train_idx, val_idx, test_idx
    )
    leakage_warnings.extend(text_warnings)

    split_manifest_out = args.output_dir / f"{args.cache_name}.split_manifest.csv"
    save_split_manifest(split_manifest_out, cohort, train_idx, val_idx, test_idx)
    train_ds, val_ds, test_ds, _, val_loader, test_loader, train_eval_loader = prepare_dataloaders(
        config, cohort, feature_info, config.modalities, (train_idx, val_idx, test_idx)
    )

    model = create_model_from_config(config, feature_info, class_names).to(device)
    ckpt = args.teacher_run_dir / "best_model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    rows = []
    rows.extend(forward_split(model, train_eval_loader, train_ds, "train", device))
    rows.extend(forward_split(model, val_loader, val_ds, "val", device))
    if test_loader is not None and test_ds is not None:
        rows.extend(forward_split(model, test_loader, test_ds, "test", device))

    fieldnames = [
        "sample_id",
        "record_id",
        "split",
        "y_true",
        "teacher_logit_0",
        "teacher_logit_1",
        "teacher_prob_0",
        "teacher_prob_1",
        "teacher_confidence",
        "teacher_margin",
        "teacher_correct",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metadata = {
        "teacher_run_dir": str(args.teacher_run_dir),
        "teacher_modalities": list(config.modalities),
        "cache_csv": str(out_csv),
        "split_manifest": str(split_manifest_out),
        "split_source": split_source,
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "class_names": class_names,
        "cohort_stats": cohort_stats,
        "strict_no_leakage": bool(config.strict_no_leakage),
        "disable_text_numeric_features": bool(config.disable_text_numeric_features),
        "leakage_warnings": leakage_warnings,
        "config": jsonable(config.__dict__),
    }
    out_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] cached teacher targets: {out_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
