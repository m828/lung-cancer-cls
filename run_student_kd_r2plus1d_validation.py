#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_TEMPLATE_CONFIG = Path("configs/student_kd_backbone_sweep_mvn_formal.json")
DEFAULT_VALIDATION_CONFIG = Path("configs/student_kd_r2plus1d_validation_campaign.json")
DEFAULT_REPORT_DIR = "../outputs/batch_student_kd_r2plus1d_validation_campaign"
DEFAULT_ANALYSIS_DIR = Path("outputs/r2plus1d_validation_analysis")

PRIMARY_NAME = "ct_text_student_kd_mvn_r2plus1d_18_full_combo"
RESNET_CT_TEXT_NAME = "ct_text_student_kd_mvn_resnet3d18_full_combo"
CT_ONLY_R2P1D_NAME = "ct_student_kd_mvn_r2plus1d_18_full_combo"
LOGITS_ONLY_NAME = "ct_text_student_kd_mvn_r2plus1d_18_logits_only"
TEACHER_SPLIT = Path("outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv")
LEGACY_CT_TEXT_SC = "ct_text_mvn_sc_tvt"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_experiment(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    for experiment in config.get("experiments", []):
        if experiment.get("name") == name:
            return experiment
    raise KeyError(f"Experiment not found in template config: {name}")


def flag_index(args: Sequence[str], flag: str) -> int | None:
    for idx, item in enumerate(args):
        if item == flag:
            return idx
    return None


def set_flag(args: List[str], flag: str, value: str) -> None:
    idx = flag_index(args, flag)
    if idx is None:
        args.extend([flag, str(value)])
        return
    if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
        args.insert(idx + 1, str(value))
    else:
        args[idx + 1] = str(value)


def get_flag(args: Sequence[str], flag: str, default: str = "") -> str:
    idx = flag_index(args, flag)
    if idx is None or idx + 1 >= len(args):
        return default
    return str(args[idx + 1])


def remove_flag(args: List[str], flag: str, takes_value: bool = True) -> None:
    while True:
        idx = flag_index(args, flag)
        if idx is None:
            return
        del args[idx]
        if takes_value and idx < len(args) and not args[idx].startswith("--"):
            del args[idx]


def set_run_name_and_output(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec["name"] = name
    spec["run_dir"] = f"../outputs/{name}"
    set_flag(spec["args"], "--output-dir", f"outputs/{name}")
    return spec


def apply_path_overrides(spec: Dict[str, Any], overrides: Dict[str, str | None]) -> None:
    args = spec["args"]
    flag_map = {
        "data_root": "--data-root",
        "metadata_csv": "--metadata-csv",
        "ct_root": "--ct-root",
        "gene_tsv": "--gene-tsv",
        "text_feature_tsv": "--text-feature-tsv",
        "teacher_run_dir": "--teacher-run-dir",
        "reference_manifest": "--reference-manifest",
    }
    for key, flag in flag_map.items():
        value = overrides.get(key)
        if value:
            set_flag(args, flag, value)


def spec_tags(*items: str) -> List[str]:
    return [item for item in items if item]


def build_validation_config(
    template_config: Path,
    report_dir: str,
    seeds: Sequence[int],
    overrides: Dict[str, str | None],
) -> Dict[str, Any]:
    template = read_json(template_config)
    primary_template = find_experiment(template, PRIMARY_NAME)
    resnet_template = find_experiment(template, RESNET_CT_TEXT_NAME)
    ct_only_template = find_experiment(template, CT_ONLY_R2P1D_NAME)

    experiments: List[Dict[str, Any]] = []

    primary = copy.deepcopy(primary_template)
    set_flag(primary["args"], "--seed", "42")
    primary["tags"] = spec_tags("binary", "student", "ct_text", "r2plus1d_18", "validation", "primary", "seed42")
    experiments.append(primary)

    for seed in seeds:
        name = f"{PRIMARY_NAME}_seed{int(seed)}"
        spec = set_run_name_and_output(primary_template, name)
        set_flag(spec["args"], "--seed", str(int(seed)))
        spec["tags"] = spec_tags(
            "binary",
            "student",
            "ct_text",
            "r2plus1d_18",
            "validation",
            "seed_repeat",
            f"seed{int(seed)}",
        )
        experiments.append(spec)

    logits_only = set_run_name_and_output(primary_template, LOGITS_ONLY_NAME)
    set_flag(logits_only["args"], "--seed", "42")
    set_flag(logits_only["args"], "--distill-methods", "logits")
    remove_flag(logits_only["args"], "--distill-method-weights", takes_value=True)
    remove_flag(logits_only["args"], "--distill-feature-loss", takes_value=True)
    remove_flag(logits_only["args"], "--distill-normalize-features", takes_value=False)
    logits_only["tags"] = spec_tags(
        "binary",
        "student",
        "ct_text",
        "r2plus1d_18",
        "validation",
        "ablation",
        "logits_only",
    )
    experiments.append(logits_only)

    ct_only = copy.deepcopy(ct_only_template)
    set_flag(ct_only["args"], "--seed", "42")
    ct_only["tags"] = spec_tags(
        "binary",
        "student",
        "ct",
        "r2plus1d_18",
        "validation",
        "ablation",
        "ct_only",
        "full_combo",
    )
    experiments.append(ct_only)

    resnet = copy.deepcopy(resnet_template)
    set_flag(resnet["args"], "--seed", "42")
    resnet["tags"] = spec_tags(
        "binary",
        "student",
        "ct_text",
        "resnet3d18",
        "validation",
        "comparator",
        "full_combo",
    )
    experiments.append(resnet)

    for spec in experiments:
        apply_path_overrides(spec, overrides)

    return {
        "description": (
            "Validation campaign for the candidate CT+Text R2Plus1D student KD result. "
            "Runs seed repeats and minimal ablations while keeping the 1019-case teacher split fixed."
        ),
        "report_dir": report_dir,
        "metrics_name": "metrics.json",
        "continue_on_error": True,
        "skip_existing_success": True,
        "export_table": True,
        "export_plots": True,
        "experiments": experiments,
    }


def command_make_config(args: argparse.Namespace) -> None:
    seeds = parse_int_list(args.seeds)
    config = build_validation_config(
        template_config=args.template_config,
        report_dir=args.report_dir,
        seeds=seeds,
        overrides={
            "data_root": args.data_root,
            "metadata_csv": args.metadata_csv,
            "ct_root": args.ct_root,
            "gene_tsv": args.gene_tsv,
            "text_feature_tsv": args.text_feature_tsv,
            "teacher_run_dir": args.teacher_run_dir,
            "reference_manifest": args.reference_manifest,
        },
    )
    write_json(args.output_config, config)
    print(f"Wrote validation batch config: {args.output_config}")
    for experiment in config["experiments"]:
        print(f"- {experiment['name']}: {get_flag(experiment['args'], '--output-dir')}")


def command_run_batch(args: argparse.Namespace) -> None:
    if args.make_config:
        command_make_config(args)
    cmd = [
        args.python_executable or sys.executable,
        "run_experiment_batch.py",
        "--config",
        str(args.output_config),
    ]
    if args.only:
        cmd.extend(["--only", args.only])
    if args.report_only:
        cmd.append("--report-only")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_int_list(raw: str) -> List[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return [int(item) for item in values]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_metrics(run_dir: Path) -> Dict[str, Any] | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return read_json(path)


def metric_row(name: str, role: str, run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    config = metrics.get("config", {})
    cohort = metrics.get("cohort_stats", {})
    modalities = ",".join(metrics.get("modalities") or config.get("modalities") or [])
    return {
        "name": name,
        "role": role,
        "run_dir": str(run_dir),
        "modalities": modalities,
        "ct_model": config.get("ct_model", metrics.get("ct_model", "")),
        "seed": config.get("seed", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "num_total": cohort.get("num_total", ""),
        "split_source": metrics.get("split_source", ""),
        "val_auroc": nested(metrics, "val_metrics", "auroc"),
        "test_auroc": nested(metrics, "test_metrics", "auroc"),
        "test_bacc": nested(metrics, "test_metrics", "balanced_accuracy"),
        "test_f1": nested(metrics, "test_metrics", "f1"),
        "test_sensitivity": nested(metrics, "test_metrics", "sensitivity"),
        "test_specificity": nested(metrics, "test_metrics", "specificity"),
        "test_confusion_matrix": nested(metrics, "test_metrics", "confusion_matrix", default=""),
    }


def display_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    displayed = dict(row)
    for key in [
        "val_auroc",
        "test_auroc",
        "test_bacc",
        "test_f1",
        "test_sensitivity",
        "test_specificity",
    ]:
        displayed[key] = format_float(displayed.get(key))
    return displayed


def run_refs(outputs_root: Path, legacy_root: Path | None, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    refs = [
        {"name": PRIMARY_NAME, "role": "primary_seed42", "path": outputs_root / PRIMARY_NAME},
        {
            "name": LOGITS_ONLY_NAME,
            "role": "ablation_logits_only",
            "path": outputs_root / LOGITS_ONLY_NAME,
        },
        {
            "name": CT_ONLY_R2P1D_NAME,
            "role": "ablation_ct_only",
            "path": outputs_root / CT_ONLY_R2P1D_NAME,
        },
        {
            "name": RESNET_CT_TEXT_NAME,
            "role": "comparator_resnet3d18",
            "path": outputs_root / RESNET_CT_TEXT_NAME,
        },
    ]
    for seed in seeds:
        name = f"{PRIMARY_NAME}_seed{int(seed)}"
        refs.append({"name": name, "role": f"primary_seed{int(seed)}", "path": outputs_root / name})
    if legacy_root is not None:
        refs.append(
            {
                "name": LEGACY_CT_TEXT_SC,
                "role": "legacy_0417_ct_text_sc",
                "path": legacy_root / LEGACY_CT_TEXT_SC,
            }
        )
    return refs


def summarize_metrics(outputs_root: Path, legacy_root: Path | None, seeds: Sequence[int], analysis_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ref in run_refs(outputs_root, legacy_root, seeds):
        metrics = load_metrics(ref["path"])
        if metrics is None:
            continue
        rows.append(metric_row(ref["name"], ref["role"], ref["path"], metrics))

    fieldnames = [
        "name",
        "role",
        "modalities",
        "ct_model",
        "seed",
        "best_epoch",
        "num_total",
        "split_source",
        "val_auroc",
        "test_auroc",
        "test_bacc",
        "test_f1",
        "test_sensitivity",
        "test_specificity",
        "test_confusion_matrix",
        "run_dir",
    ]
    write_csv(analysis_dir / "validation_metrics.csv", rows, fieldnames)
    write_markdown_table(
        analysis_dir / "validation_metrics.md",
        [display_metric_row(row) for row in rows],
        fieldnames,
    )
    return rows


def summarize_seed_stability(metric_rows: Sequence[Dict[str, Any]], analysis_dir: Path) -> None:
    seed_rows = [
        row
        for row in metric_rows
        if row["name"] == PRIMARY_NAME or row["name"].startswith(f"{PRIMARY_NAME}_seed")
    ]
    metrics = ["test_auroc", "test_bacc", "test_f1", "test_sensitivity", "test_specificity"]
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        values = [float(row[metric]) for row in seed_rows if row.get(metric) not in ("", None)]
        if not values:
            continue
        rows.append(
            {
                "metric": metric,
                "num_seeds": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    fieldnames = ["metric", "num_seeds", "mean", "std", "min", "max"]
    write_csv(analysis_dir / "seed_stability.csv", rows, fieldnames)
    write_markdown_table(
        analysis_dir / "seed_stability.md",
        [{**row, **{key: format_float(row[key]) for key in ["mean", "std", "min", "max"]}} for row in rows],
        fieldnames,
    )


def split_duplicate_summary(split_path: Path) -> Dict[str, int]:
    if not split_path.exists():
        return {"num_rows": 0, "sample_cross_split_duplicates": 0, "record_cross_split_duplicates": 0}
    rows = read_csv_rows(split_path)
    sample_splits: Dict[str, set[str]] = {}
    record_splits: Dict[str, set[str]] = {}
    for row in rows:
        split = row.get("assigned_split") or row.get("split") or ""
        sample_id = row.get("sample_id") or ""
        record_id = row.get("record_id") or ""
        if sample_id:
            sample_splits.setdefault(sample_id, set()).add(split)
        if record_id:
            record_splits.setdefault(record_id, set()).add(split)
    return {
        "num_rows": len(rows),
        "sample_cross_split_duplicates": sum(1 for value in sample_splits.values() if len(value) > 1),
        "record_cross_split_duplicates": sum(1 for value in record_splits.values() if len(value) > 1),
    }


def summarize_split_integrity(
    outputs_root: Path,
    legacy_root: Path | None,
    seeds: Sequence[int],
    teacher_split: Path,
    analysis_dir: Path,
) -> None:
    teacher_md5 = md5_file(teacher_split) if teacher_split.exists() else ""
    rows: List[Dict[str, Any]] = []
    for ref in run_refs(outputs_root, legacy_root, seeds):
        split_path = ref["path"] / "split_manifest.csv"
        if not split_path.exists():
            continue
        dup = split_duplicate_summary(split_path)
        split_md5 = md5_file(split_path)
        rows.append(
            {
                "name": ref["name"],
                "role": ref["role"],
                "split_path": str(split_path),
                "split_md5": split_md5,
                "teacher_split_md5": teacher_md5,
                "matches_teacher_split": bool(teacher_md5 and split_md5 == teacher_md5),
                **dup,
            }
        )
    fieldnames = [
        "name",
        "role",
        "matches_teacher_split",
        "num_rows",
        "sample_cross_split_duplicates",
        "record_cross_split_duplicates",
        "split_md5",
        "teacher_split_md5",
        "split_path",
    ]
    write_csv(analysis_dir / "split_integrity.csv", rows, fieldnames)
    write_markdown_table(analysis_dir / "split_integrity.md", rows, fieldnames)


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "record_id": row.get("record_id", ""),
                "label_name": row.get("label_name", ""),
                "label": int(row["label"]),
                "split": row.get("split", ""),
                "prediction": int(row["prediction"]),
                "prob_malignant": float(row["prob_malignant"]),
            }
        )
    return rows


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [score for label, score in zip(labels, scores) if label == 1]
    neg = [score for label, score in zip(labels, scores) if label == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def bacc_score(labels: Sequence[int], predictions: Sequence[int]) -> float:
    pos_idx = [idx for idx, label in enumerate(labels) if label == 1]
    neg_idx = [idx for idx, label in enumerate(labels) if label == 0]
    if not pos_idx or not neg_idx:
        return float("nan")
    sensitivity = sum(1 for idx in pos_idx if predictions[idx] == 1) / len(pos_idx)
    specificity = sum(1 for idx in neg_idx if predictions[idx] == 0) / len(neg_idx)
    return 0.5 * (sensitivity + specificity)


def f1_score(labels: Sequence[int], predictions: Sequence[int]) -> float:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def quantile(values: Sequence[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def pair_predictions(a_path: Path, b_path: Path) -> List[Dict[str, Any]]:
    a_rows = load_predictions(a_path)
    b_rows = {(row["sample_id"], row["record_id"]): row for row in load_predictions(b_path)}
    paired = []
    for a_row in a_rows:
        key = (a_row["sample_id"], a_row["record_id"])
        b_row = b_rows.get(key)
        if b_row is None:
            continue
        if a_row["label"] != b_row["label"]:
            raise ValueError(f"Label mismatch for {key}")
        paired.append({"a": a_row, "b": b_row})
    return paired


def metric_diffs(paired: Sequence[Dict[str, Any]], indices: Sequence[int]) -> Dict[str, float] | None:
    labels = [paired[idx]["a"]["label"] for idx in indices]
    if len(set(labels)) < 2:
        return None
    a_scores = [paired[idx]["a"]["prob_malignant"] for idx in indices]
    b_scores = [paired[idx]["b"]["prob_malignant"] for idx in indices]
    a_pred = [paired[idx]["a"]["prediction"] for idx in indices]
    b_pred = [paired[idx]["b"]["prediction"] for idx in indices]
    return {
        "auroc": auc_score(labels, a_scores) - auc_score(labels, b_scores),
        "bacc": bacc_score(labels, a_pred) - bacc_score(labels, b_pred),
        "f1": f1_score(labels, a_pred) - f1_score(labels, b_pred),
    }


def paired_bootstrap(
    a_name: str,
    a_dir: Path,
    b_name: str,
    b_dir: Path,
    analysis_dir: Path,
    n_bootstrap: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    a_pred = a_dir / "test_predictions.csv"
    b_pred = b_dir / "test_predictions.csv"
    if not a_pred.exists() or not b_pred.exists():
        return []

    paired = pair_predictions(a_pred, b_pred)
    if not paired:
        return []
    rng = random.Random(random_seed)
    obs = metric_diffs(paired, list(range(len(paired))))
    if obs is None:
        return []

    boot: Dict[str, List[float]] = {key: [] for key in obs}
    for _ in range(n_bootstrap):
        indices = [rng.randrange(len(paired)) for _ in range(len(paired))]
        diff = metric_diffs(paired, indices)
        if diff is None:
            continue
        for key, value in diff.items():
            boot[key].append(value)

    error_rows = []
    summary = {"both_correct": 0, "a_correct_b_wrong": 0, "a_wrong_b_correct": 0, "both_wrong": 0}
    for item in paired:
        a = item["a"]
        b = item["b"]
        a_correct = int(a["prediction"] == a["label"])
        b_correct = int(b["prediction"] == b["label"])
        if a_correct and b_correct:
            bucket = "both_correct"
        elif a_correct and not b_correct:
            bucket = "a_correct_b_wrong"
        elif not a_correct and b_correct:
            bucket = "a_wrong_b_correct"
        else:
            bucket = "both_wrong"
        summary[bucket] += 1
        error_rows.append(
            {
                "sample_id": a["sample_id"],
                "record_id": a["record_id"],
                "label_name": a["label_name"],
                "label": a["label"],
                "split": a["split"],
                "a_name": a_name,
                "b_name": b_name,
                "a_prediction": a["prediction"],
                "b_prediction": b["prediction"],
                "a_prob_malignant": a["prob_malignant"],
                "b_prob_malignant": b["prob_malignant"],
                "bucket": bucket,
            }
        )
    safe_pair = f"{a_name}_vs_{b_name}".replace("/", "_")
    write_csv(
        analysis_dir / f"paired_errors_{safe_pair}.csv",
        error_rows,
        [
            "sample_id",
            "record_id",
            "label_name",
            "label",
            "split",
            "a_name",
            "b_name",
            "a_prediction",
            "b_prediction",
            "a_prob_malignant",
            "b_prob_malignant",
            "bucket",
        ],
    )

    rows = []
    for metric, observed in obs.items():
        values = boot[metric]
        rows.append(
            {
                "a_name": a_name,
                "b_name": b_name,
                "metric": metric,
                "n_paired": len(paired),
                "diff_a_minus_b": observed,
                "ci95_low": quantile(values, 0.025),
                "ci95_high": quantile(values, 0.975),
                "p_diff_gt_0": sum(1 for value in values if value > 0) / len(values) if values else "",
                **summary,
            }
        )
    return rows


def summarize_bootstrap(
    outputs_root: Path,
    legacy_root: Path | None,
    analysis_dir: Path,
    n_bootstrap: int,
    random_seed: int,
) -> None:
    pairs = [
        (PRIMARY_NAME, outputs_root / PRIMARY_NAME, RESNET_CT_TEXT_NAME, outputs_root / RESNET_CT_TEXT_NAME),
        (PRIMARY_NAME, outputs_root / PRIMARY_NAME, LOGITS_ONLY_NAME, outputs_root / LOGITS_ONLY_NAME),
        (PRIMARY_NAME, outputs_root / PRIMARY_NAME, CT_ONLY_R2P1D_NAME, outputs_root / CT_ONLY_R2P1D_NAME),
    ]
    if legacy_root is not None:
        pairs.append((PRIMARY_NAME, outputs_root / PRIMARY_NAME, LEGACY_CT_TEXT_SC, legacy_root / LEGACY_CT_TEXT_SC))

    rows: List[Dict[str, Any]] = []
    for idx, (a_name, a_dir, b_name, b_dir) in enumerate(pairs):
        rows.extend(
            paired_bootstrap(
                a_name=a_name,
                a_dir=a_dir,
                b_name=b_name,
                b_dir=b_dir,
                analysis_dir=analysis_dir,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed + idx,
            )
        )

    fieldnames = [
        "a_name",
        "b_name",
        "metric",
        "n_paired",
        "diff_a_minus_b",
        "ci95_low",
        "ci95_high",
        "p_diff_gt_0",
        "both_correct",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "both_wrong",
    ]
    write_csv(analysis_dir / "paired_bootstrap.csv", rows, fieldnames)
    display_rows = []
    for row in rows:
        shown = dict(row)
        for key in ["diff_a_minus_b", "ci95_low", "ci95_high", "p_diff_gt_0"]:
            shown[key] = format_float(shown.get(key))
        display_rows.append(shown)
    write_markdown_table(analysis_dir / "paired_bootstrap.md", display_rows, fieldnames)

    summary_fields = ["a_name", "b_name", "both_correct", "a_correct_b_wrong", "a_wrong_b_correct", "both_wrong"]
    seen = set()
    summary_rows = []
    for row in rows:
        key = (row["a_name"], row["b_name"])
        if key in seen:
            continue
        seen.add(key)
        summary_rows.append({field: row.get(field, "") for field in summary_fields})
    write_csv(analysis_dir / "paired_error_summary.csv", summary_rows, summary_fields)


def command_summarize(args: argparse.Namespace) -> None:
    seeds = parse_int_list(args.seeds)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    legacy_root = args.legacy_root if args.legacy_root else None
    metric_rows = summarize_metrics(args.outputs_root, legacy_root, seeds, args.analysis_dir)
    summarize_seed_stability(metric_rows, args.analysis_dir)
    summarize_split_integrity(args.outputs_root, legacy_root, seeds, args.teacher_split, args.analysis_dir)
    summarize_bootstrap(
        outputs_root=args.outputs_root,
        legacy_root=legacy_root,
        analysis_dir=args.analysis_dir,
        n_bootstrap=args.bootstrap,
        random_seed=args.random_seed,
    )
    print(f"Wrote validation analysis to: {args.analysis_dir}")


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_VALIDATION_CONFIG)
    parser.add_argument("--report-dir", type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seeds", type=str, default="43,44,45", help="Comma-separated repeat seeds.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--ct-root", type=str, default=None)
    parser.add_argument("--gene-tsv", type=str, default=None)
    parser.add_argument("--text-feature-tsv", type=str, default=None)
    parser.add_argument("--teacher-run-dir", type=str, default=None)
    parser.add_argument("--reference-manifest", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, run, and summarize the R2Plus1D CT+Text student KD validation campaign."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_config = subparsers.add_parser("make-config", help="Write the validation batch JSON config.")
    add_common_config_args(make_config)
    make_config.set_defaults(func=command_make_config)

    run_batch = subparsers.add_parser("run-batch", help="Optionally write the config, then run run_experiment_batch.py.")
    add_common_config_args(run_batch)
    run_batch.add_argument("--make-config", action=argparse.BooleanOptionalAction, default=True)
    run_batch.add_argument("--only", type=str, default="")
    run_batch.add_argument("--report-only", action="store_true")
    run_batch.add_argument("--python-executable", type=str, default=None)
    run_batch.set_defaults(func=command_run_batch)

    summarize = subparsers.add_parser("summarize", help="Summarize metrics, split integrity, bootstrap, and error deltas.")
    summarize.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    summarize.add_argument("--legacy-root", type=Path, default=None)
    summarize.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    summarize.add_argument("--teacher-split", type=Path, default=TEACHER_SPLIT)
    summarize.add_argument("--seeds", type=str, default="43,44,45")
    summarize.add_argument("--bootstrap", type=int, default=1000)
    summarize.add_argument("--random-seed", type=int, default=20260517)
    summarize.set_defaults(func=command_summarize)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
