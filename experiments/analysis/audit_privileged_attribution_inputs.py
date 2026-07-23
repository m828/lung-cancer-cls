#!/usr/bin/env python3
"""Inventory existing artifacts used by the 0542 attribution suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import sha256_file, write_csv_rows  # noqa: E402


SEEDS = (42, 43, 44, 45)
TEACHER_MATCH_FIELDS = (
    "ct_model",
    "batch_size",
    "lr",
    "weight_decay",
    "optimizer_name",
    "scheduler_name",
    "loss_name",
    "label_smoothing",
    "sampling_strategy",
    "class_weight_strategy",
    "effective_num_beta",
    "epochs",
    "selection_metric",
    "aug_profile",
    "depth_size",
    "volume_hw",
    "ct_feature_dim",
    "text_feature_dim",
    "fusion_hidden_dim",
    "dropout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def csv_info(path: Path, id_columns=("sample_id", "record_id")) -> tuple[int, str, str]:
    if not path.is_file():
        return 0, "", ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0, "", sha256_file(path)
    id_col = next((column for column in id_columns if column in rows[0]), "")
    identity_hash = hashlib.sha256("\n".join(sorted(str(row[id_col]).strip() for row in rows)).encode("utf-8")).hexdigest() if id_col else ""
    return len(rows), identity_hash, sha256_file(path)


def cache_path(root: Path, teacher: str, seed: int) -> Path:
    candidates = [
        root / "outputs0534_best_student_kd_search" / "cached_teacher_targets" / f"{teacher}_seed{seed}.csv",
        root / "outputs0535_student_kd_refinement" / "cached_teacher_targets" / f"{teacher}_seed{seed}.csv",
    ]
    return next((path for path in candidates if path.is_file()), candidates[-1])


def student_training_value(config: dict[str, Any], cached: dict[str, Any], field: str) -> Any:
    cached_alias = {
        "optimizer_name": "optimizer",
        "scheduler_name": "scheduler",
        "loss_name": "loss",
    }.get(field, field)
    return cached.get(cached_alias, config.get(field))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_rows = []
    prediction_rows = []
    manifest_rows = []
    for teacher, directory_prefix, modalities in (
        ("T0", "densenet3d121_ct_text_teacher_strict_seed", "ct,text"),
        ("T1", "densenet3d121_ct_cnv_text_teacher_strict_seed", "ct,text,cnv"),
    ):
        for seed in SEEDS:
            run_dir = args.results_root / "outputs0531_teacher_homogeneous_gene_test" / f"{directory_prefix}{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "best_model.pt"
            cache = cache_path(args.results_root, teacher, seed)
            metrics = read_json(metrics_path)
            config = metrics.get("config") or {}
            teacher_rows.append(
                {
                    "teacher": teacher,
                    "seed": seed,
                    "modalities": modalities,
                    "run_dir": str(run_dir),
                    "metrics_available": metrics_path.is_file(),
                    "checkpoint_available_local": checkpoint_path.is_file(),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
                    "cached_logits_available": cache.is_file(),
                    "cached_logits_path": str(cache),
                    "cached_logits_sha256": sha256_file(cache) if cache.is_file() else "",
                    "selection_metric": config.get("selection_metric", ""),
                    "split_manifest": str(run_dir / "split_manifest.csv"),
                }
            )
            for split in ("train", "val", "test"):
                pred_path = run_dir / f"{split}_predictions.csv"
                count, identity_hash, digest = csv_info(pred_path)
                prediction_rows.append({"family": "teacher", "model": teacher, "seed": seed, "split": split, "path": str(pred_path), "available": pred_path.is_file(), "row_count": count, "identity_hash": identity_hash, "sha256": digest})
            manifest = run_dir / "split_manifest.csv"
            count, identity_hash, digest = csv_info(manifest)
            manifest_rows.append({"model": teacher, "seed": seed, "path": str(manifest), "available": manifest.is_file(), "row_count": count, "identity_hash": identity_hash, "sha256": digest})

    student_rows = []
    for seed in SEEDS:
        run_dir = args.results_root / "outputs0535_student_kd_refinement" / "refined_candidates" / f"R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed{seed}"
        metrics = read_json(run_dir / "metrics.json")
        config = metrics.get("config") or {}
        cached = metrics.get("cached_kd_config") or {}
        student_rows.append(
            {
                "model": "R3",
                "seed": seed,
                "run_dir": str(run_dir),
                "metrics_available": (run_dir / "metrics.json").is_file(),
                "checkpoint_available_local": (run_dir / "best_model.pt").is_file(),
                "checkpoint_path": str(run_dir / "best_model.pt"),
                "multiple_checkpoint_criteria_available": (run_dir / "checkpoint_evaluations.json").is_file(),
                "batch_size": cached.get("batch_size", config.get("batch_size", "")),
                "learning_rate": cached.get("lr", config.get("lr", "")),
                "alpha": cached.get("distillation_alpha", metrics.get("distillation_alpha", "")),
                "temperature": cached.get("distillation_temperature", metrics.get("distillation_temperature", "")),
                "selection_metric": metrics.get("selection_metric", ""),
            }
        )
        for split in ("train", "val", "test"):
            pred_path = run_dir / f"{split}_predictions.csv"
            count, identity_hash, digest = csv_info(pred_path)
            prediction_rows.append({"family": "student", "model": "R3", "seed": seed, "split": split, "path": str(pred_path), "available": pred_path.is_file(), "row_count": count, "identity_hash": identity_hash, "sha256": digest})

    tri_models = {
        "TRI_T_CACHE": lambda seed: args.results_root / "outputs0541_triclass_teacher_student_selection_4seed" / "cached_teacher_targets" / f"teacher_select_accuracy_seed{seed}.csv",
        "TRI_S0": lambda seed: args.results_root / "outputs0541_triclass_teacher_student_selection_4seed" / "supervised_ct_text" / f"TRI-S0_ct_text_densenet3d121_seed{seed}" / "test_predictions.csv",
        "TRI_SKD": lambda seed: args.results_root / "outputs0541_triclass_teacher_student_selection_4seed" / "student_4seed" / "teacher_select_accuracy" / f"TRI-SKD_teacher_accuracy_student_macro_f1_seed{seed}" / "test_predictions.csv",
    }
    for model, resolver in tri_models.items():
        for seed in SEEDS:
            path = resolver(seed)
            count, identity_hash, digest = csv_info(path)
            prediction_rows.append({"family": "triclass", "model": model, "seed": seed, "split": "all_or_test", "path": str(path), "available": path.is_file(), "row_count": count, "identity_hash": identity_hash, "sha256": digest})

    write_csv_rows(args.output_dir / "teacher_checkpoint_inventory.csv", teacher_rows, list(teacher_rows[0]))
    write_csv_rows(args.output_dir / "student_checkpoint_inventory.csv", student_rows, list(student_rows[0]))
    write_csv_rows(args.output_dir / "data_manifest_inventory.csv", manifest_rows, list(manifest_rows[0]))
    write_csv_rows(args.output_dir / "prediction_file_inventory.csv", prediction_rows, list(prediction_rows[0]))

    teacher_protocol_mismatches: list[str] = []
    teacher_manifest_mismatches: list[str] = []
    for seed in SEEDS:
        t0_dir = args.results_root / "outputs0531_teacher_homogeneous_gene_test" / f"densenet3d121_ct_text_teacher_strict_seed{seed}"
        t1_dir = args.results_root / "outputs0531_teacher_homogeneous_gene_test" / f"densenet3d121_ct_cnv_text_teacher_strict_seed{seed}"
        t0_seed_config = read_json(t0_dir / "metrics.json").get("config") or {}
        t1_seed_config = read_json(t1_dir / "metrics.json").get("config") or {}
        for field in TEACHER_MATCH_FIELDS:
            if t0_seed_config.get(field) != t1_seed_config.get(field):
                teacher_protocol_mismatches.append(
                    f"seed{seed}:{field}={t0_seed_config.get(field)!r} vs {t1_seed_config.get(field)!r}"
                )
        t0_manifest = t0_dir / "split_manifest.csv"
        t1_manifest = t1_dir / "split_manifest.csv"
        if not t0_manifest.is_file() or not t1_manifest.is_file() or sha256_file(t0_manifest) != sha256_file(t1_manifest):
            teacher_manifest_mismatches.append(f"seed{seed}")

    t0_config = read_json(args.results_root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_text_teacher_strict_seed42" / "metrics.json").get("config") or {}
    r3_metrics = read_json(args.results_root / "outputs0535_student_kd_refinement" / "refined_candidates" / "R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42" / "metrics.json")
    r3_config = r3_metrics.get("config") or {}
    r3_cached = r3_metrics.get("cached_kd_config") or {}
    baseline_comparison_fields = (
        "ct_model",
        "batch_size",
        "lr",
        "weight_decay",
        "optimizer_name",
        "scheduler_name",
        "loss_name",
        "label_smoothing",
        "sampling_strategy",
        "class_weight_strategy",
        "effective_num_beta",
        "epochs",
        "aug_profile",
        "depth_size",
        "volume_hw",
        "ct_feature_dim",
        "text_feature_dim",
        "fusion_hidden_dim",
        "dropout",
    )
    baseline_mismatches = [
        f"{field}: locked={t0_config.get(field)!r}, R3={student_training_value(r3_config, r3_cached, field)!r}"
        for field in baseline_comparison_fields
        if t0_config.get(field) != student_training_value(r3_config, r3_cached, field)
    ]
    if str(t0_config.get("selection_metric")) != str(r3_metrics.get("selection_metric")):
        baseline_mismatches.append(
            f"checkpoint criterion: locked={t0_config.get('selection_metric')!r}, R3={r3_metrics.get('selection_metric')!r}"
        )
    if bool(r3_cached.get("amp", False)):
        baseline_mismatches.append("mixed precision: locked generic trainer=disabled, R3 cached-KD trainer=enabled")
    baseline_matched = not baseline_mismatches
    teacher_complete = all(row["metrics_available"] for row in teacher_rows)
    teacher_weights_complete = all(row["checkpoint_available_local"] for row in teacher_rows)
    caches_complete = all(row["cached_logits_available"] for row in teacher_rows)
    tri_complete = all(row["available"] for row in prediction_rows if row["family"] == "triclass")
    audit = f"""# Existing Pipeline Audit

- CT-text and CT-text-CNV teachers have byte-identical split manifests for seeds 42-45: **{not teacher_manifest_mismatches}**.
- The two teacher families match on the audited training protocol fields for seeds 42-45: **{not teacher_protocol_mismatches}**. The CT-text-CNV teacher additionally contains the CNV branch and associated fusion parameters.
- Both teacher families have metrics and predictions for seeds 42-45: **{teacher_complete}**.
- Local teacher checkpoint binaries cover seeds 42-45: **{teacher_weights_complete}**. The local synchronized result copy contains metrics/predictions but no `.pt` files; checkpoint-dependent full runs must execute where the original checkpoints exist.
- Cached T0/T1 teacher logits cover seeds 42-45: **{caches_complete}**.
- The locked CT-text reference is not training-matched to R3: **{not baseline_matched}**. Verified differences: {"; ".join(baseline_mismatches) if baseline_mismatches else "none"}.
- Existing cached KD correctly retains hard-label CE and normalizes weighted KL by the within-batch weight sum.
- Existing R3 runs save only `best_model.pt`; the 0542 extension adds validation-loss/AUROC/F1/BAcc/composite and last checkpoints without changing legacy defaults.
- Both original and repeat seed-44 teacher directories exist. R3 cache metadata points to the original seed-44 teachers, so teacher-cache factorial arms follow the original artifact and do not merge it with the repeat. The read-only teacher-correction analysis separately uses the paper-locked `seed44_repeat` run as its hard-label student reference.
- Triclass TRI-T cache, TRI-S0 predictions, and TRI-SKD predictions are present for seeds 42-45: **{tri_complete}**.
- TRI-S0 contains 36 test identities per seed outside the teacher/KD aligned cohort; the confusion comparison explicitly uses the 227 common identities and reports this restriction.

## Reuse

Existing manifests, T0/T1 cached logits, teacher predictions, locked CT-text predictions, R3 predictions, and triclass predictions are reused read-only. New training is required for `S0_MATCHED`, uniform/confidence matched arms not already run under R3 settings, shuffled confidence, and the permuted-CNV teacher/student.

## Audit exceptions

- Teacher protocol mismatches: {teacher_protocol_mismatches if teacher_protocol_mismatches else "none"}.
- Teacher manifest mismatches: {teacher_manifest_mismatches if teacher_manifest_mismatches else "none"}.
"""
    (args.output_dir / "existing_pipeline_audit.md").write_text(audit, encoding="utf-8")
    reports_dir = args.output_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "existing_pipeline_audit.md").write_text(audit, encoding="utf-8")
    plan = """# Implementation Plan

1. Reuse the existing cached-logits student trainer and add explicit attribution modes while preserving legacy option semantics.
2. Reuse each teacher cache identically across uniform and confidence arms.
3. Add a deterministic training-only weight permutation for shuffled confidence.
4. Materialize split-local, whole-row CNV permutations before teacher training and preserve target-to-donor mappings.
5. Save all validation-selected checkpoints atomically for new student runs; keep composite as primary.
6. Analyze effects with common-identity paired bootstrap inside seed, followed by seed-level averaging.
7. Reuse existing predictions for teacher-correction and triclass confusion analyses; never start training in analysis stages.
"""
    (args.output_dir / "implementation_plan.md").write_text(plan, encoding="utf-8")
    print("[OK] wrote existing pipeline audit inventories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
