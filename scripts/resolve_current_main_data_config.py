#!/usr/bin/env python3
"""Resolve data inputs from the current successful binary main experiments.

This script is intentionally read-only.  It extracts the data contract used by
the current malignant-vs-normal CT+Text / CT+CNV+Text line and writes a JSON
file that wrappers can consume instead of asking users to re-export many paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PATH_KEYS = [
    "data_root",
    "metadata_csv",
    "ct_root",
    "gene_tsv",
    "text_feature_tsv",
    "text_health_csv",
    "text_disease_csv",
    "bert_model_path",
    "text_cache_tsv",
]
COLUMN_KEYS = [
    "metadata_sample_id_col",
    "patient_id_col",
    "metadata_text_id_col",
    "text_record_id_col",
    "ct_path_col",
    "label_col",
    "split_col",
    "gene_id_col",
    "gene_label_col",
]
TEXT_KEYS = [
    "text_embedding_backend",
    "text_hash_dim",
    "text_batch_size",
    "text_max_length",
    "disable_text_numeric_features",
    "allowed_text_cols",
    "allowed_numeric_cols",
    "forbidden_feature_keywords",
]
OTHER_KEYS = ["strict_no_leakage", "class_mode", "binary_task"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return None
    return {
        "path": str(path),
        "config": cfg,
        "class_names": data.get("class_names"),
        "modalities": data.get("modalities", cfg.get("modalities")),
        "cohort_stats": data.get("cohort_stats"),
        "feature_info": data.get("feature_info"),
    }


def unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def candidate_roots(project_root: Path, results_root: Path | None) -> list[Path]:
    roots = []
    if results_root is not None:
        roots.append(results_root)
        roots.append(results_root.parent)
    roots.extend([project_root, project_root.parent])
    return unique_paths(roots)


def candidate_metrics(project_root: Path, results_root: Path | None) -> list[Path]:
    rels = [
        "outputs0535_student_kd_refinement/refined_candidates/"
        "R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42/metrics.json",
        "outputs0535_student_kd_refinement/refined_candidates/"
        "R3_repeat1_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42/metrics.json",
        "outputs0534_best_student_kd_search/s2_gene_teacher_kd/"
        "S2_confidence_a0.1_T8_bs12_lr1e-4_seed42/metrics.json",
        "outputs0531_teacher_homogeneous_gene_test/"
        "densenet3d121_ct_cnv_text_teacher_strict_seed42/metrics.json",
        "outputs0531_teacher_homogeneous_gene_test/"
        "densenet3d121_ct_text_teacher_strict_seed42/metrics.json",
        "outputs0531_gene_privileged_ablation/"
        "ct_text_sc_densenet3d121_strict_bs4_seed42/metrics.json",
        "outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/metrics.json",
    ]
    paths = []
    for root in candidate_roots(project_root, results_root):
        for rel in rels:
            paths.append(root / rel)
    return unique_paths(paths)


def first_config_value(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        value = record["config"].get(key)
        if value not in (None, ""):
            return value
    return None


def choose_text_feature_tsv(records: list[dict[str, Any]], gene_tsv: str | None) -> str | None:
    candidates: list[str] = []
    for record in records:
        value = record["config"].get("text_feature_tsv")
        if not value:
            continue
        text = str(value)
        if gene_tsv and text == gene_tsv:
            continue
        candidates.append(text)
    if not candidates:
        return None
    text_like = [p for p in candidates if "/text/" in p.lower() or "text" in Path(p).name.lower()]
    return text_like[0] if text_like else candidates[0]


def source_kind(config: dict[str, Any]) -> str:
    if config.get("text_feature_tsv"):
        return "text_feature_tsv"
    if config.get("text_health_csv") or config.get("text_disease_csv"):
        return "raw_text_csv"
    if config.get("text_cache_tsv"):
        return "text_cache_tsv"
    return "unknown"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve() if args.results_root else None

    records = [record for path in candidate_metrics(project_root, results_root) if (record := read_metrics(path))]
    if not records:
        raise SystemExit("No current binary main metrics.json found for data-config resolution.")

    resolved: dict[str, Any] = {
        "project_root": str(project_root),
        "results_root_hint": str(results_root) if results_root else None,
        "source_metrics": [record["path"] for record in records],
        "source_priority": "outputs0535 R3 -> outputs0534 best KD -> outputs0531 homogeneous -> sensitivity fallback",
    }

    for key in PATH_KEYS + COLUMN_KEYS + TEXT_KEYS + OTHER_KEYS:
        resolved[key] = first_config_value(records, key)

    gene_tsv = str(resolved.get("gene_tsv") or "") or None
    resolved["text_feature_tsv"] = choose_text_feature_tsv(records, gene_tsv)
    text_feature_tsv = str(resolved.get("text_feature_tsv") or "") or None
    if text_feature_tsv:
        resolved["text_source_type"] = "text_feature_tsv"
        resolved["text_source"] = text_feature_tsv
    else:
        donor_cfg = next((r["config"] for r in records if source_kind(r["config"]) != "unknown"), records[0]["config"])
        resolved["text_source_type"] = source_kind(donor_cfg)
        resolved["text_source"] = (
            donor_cfg.get("text_cache_tsv")
            or donor_cfg.get("text_health_csv")
            or donor_cfg.get("text_disease_csv")
        )

    resolved["gene_source"] = gene_tsv
    resolved["same_text_and_gene_source"] = bool(text_feature_tsv and gene_tsv and text_feature_tsv == gene_tsv)
    resolved["main_binary_class_mode"] = first_config_value(records, "class_mode")
    resolved["main_binary_task"] = first_config_value(records, "binary_task")
    resolved["triclass_class_mode"] = "multiclass"
    resolved["triclass_label_mapping"] = {
        "健康对照": "normal",
        "良性结节": "benign",
        "肺癌": "malignant",
    }
    resolved["num_classes"] = 3

    missing_required = [
        key
        for key in ["data_root", "metadata_csv", "ct_root", "text_source"]
        if not resolved.get(key)
    ]
    if not resolved.get("gene_tsv"):
        missing_required.append("gene_tsv")
    resolved["missing_required_config_keys"] = missing_required

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    if resolved["same_text_and_gene_source"]:
        raise SystemExit("[ERROR] text feature source equals gene feature source. This is almost certainly wrong.")
    if missing_required:
        raise SystemExit(f"Missing required config keys: {', '.join(missing_required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
