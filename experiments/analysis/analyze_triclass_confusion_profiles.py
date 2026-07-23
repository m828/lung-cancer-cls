#!/usr/bin/env python3
"""Compute four-seed triclass confusion profiles from existing predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import (  # noqa: E402
    AttributionAuditError,
    Prediction,
    align_prediction_maps,
    load_prediction_csv,
    multiclass_class_metrics,
    multiclass_confusion,
    row_normalize_confusion,
    write_csv_rows,
)


MODEL_NAMES = ("TRI_T", "TRI_S0", "TRI_SKD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45")
    return parser.parse_args()


def load_teacher_cache(path: Path) -> dict[str, Prediction]:
    output: dict[str, Prediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")).strip().lower() != "test":
                continue
            sid = str(row["sample_id"]).strip()
            if sid in output:
                raise AttributionAuditError(f"duplicate teacher cache test identity: {sid}")
            probabilities = tuple(float(row[f"teacher_prob_{index}"]) for index in range(3))
            output[sid] = Prediction(sid, int(row["y_true"]), probabilities, int(max(range(3), key=lambda index: probabilities[index])))
    if not output:
        raise AttributionAuditError(f"teacher cache has no test rows: {path}")
    return output


def audit_class_mapping(source_root: Path, seeds: list[int]) -> tuple[dict[int, str], list[str]]:
    evidence: list[str] = []
    mappings: list[dict[int, str]] = []
    for seed in seeds:
        candidates = [
            source_root / "teacher_selected" / f"teacher_select_accuracy_seed{seed}" / "metrics.json",
            source_root / "teacher_ct_cnv_text" / f"TRI-T_multi_select_seed{seed}" / "metrics.json",
            source_root / "supervised_ct_text" / f"TRI-S0_ct_text_densenet3d121_seed{seed}" / "metrics.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("class_names") or {}
            mapping = {int(index): str(name).strip().lower() for index, name in raw.items()}
            if mapping:
                mappings.append(mapping)
                evidence.append(f"`{path}` -> {mapping}")
    if not mappings:
        raise AttributionAuditError("class mapping unavailable from metrics.json")
    reference = mappings[0]
    if any(mapping != reference for mapping in mappings[1:]):
        raise AttributionAuditError(f"inconsistent triclass mappings: {mappings}")
    expected = {0: "normal", 1: "benign", 2: "malignant"}
    if reference != expected:
        raise AttributionAuditError(f"verified class mapping differs from requested order: {reference}")
    return reference, evidence


def model_predictions(source_root: Path, seed: int) -> dict[str, dict[str, Prediction]]:
    return {
        "TRI_T": load_teacher_cache(source_root / "cached_teacher_targets" / f"teacher_select_accuracy_seed{seed}.csv"),
        "TRI_S0": load_prediction_csv(source_root / "supervised_ct_text" / f"TRI-S0_ct_text_densenet3d121_seed{seed}" / "test_predictions.csv"),
        "TRI_SKD": load_prediction_csv(source_root / "student_4seed" / "teacher_select_accuracy" / f"TRI-SKD_teacher_accuracy_student_macro_f1_seed{seed}" / "test_predictions.csv"),
    }


def matrix_rows(model: str, seed: int, matrix: list[list[float]], class_names: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "seed": seed,
            "true_class_index": true_index,
            "true_class": class_names[true_index],
            "predicted_class_index": pred_index,
            "predicted_class": class_names[pred_index],
            "value": matrix[true_index][pred_index],
        }
        for true_index in range(len(matrix))
        for pred_index in range(len(matrix))
    ]


def plot_matrix(path_stem: Path, title: str, mean_matrix, std_matrix, class_labels) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    mean_array = np.asarray(mean_matrix, dtype=float)
    std_array = np.asarray(std_matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(mean_array, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    for row in range(mean_array.shape[0]):
        for col in range(mean_array.shape[1]):
            color = "white" if mean_array[row, col] > 0.55 else "#17202A"
            axis.text(col, row, f"{mean_array[row, col]:.2f}\n+/-{std_array[row, col]:.2f}", ha="center", va="center", fontsize=9, color=color)
    axis.set_xticks(range(len(class_labels)), class_labels)
    axis.set_yticks(range(len(class_labels)), class_labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title, fontsize=11)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(path_stem: Path, matrices, stds, class_labels) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    image = None
    for axis, model in zip(axes, MODEL_NAMES):
        mean_array = np.asarray(matrices[model], dtype=float)
        std_array = np.asarray(stds[model], dtype=float)
        image = axis.imshow(mean_array, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        for row in range(3):
            for col in range(3):
                color = "white" if mean_array[row, col] > 0.55 else "#17202A"
                axis.text(col, row, f"{mean_array[row, col]:.2f}\n+/-{std_array[row, col]:.2f}", ha="center", va="center", fontsize=8, color=color)
        axis.set_xticks(range(3), class_labels, rotation=20, ha="right")
        axis.set_yticks(range(3), class_labels)
        axis.set_title(model.replace("_", "-"), fontsize=10)
        axis.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized proportion")
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    class_names, evidence = audit_class_mapping(args.source_root, seeds)
    (args.output_dir / "class_mapping_audit.md").write_text(
        "# Triclass Class Mapping Audit\n\nVerified mapping: `0=Normal, 1=Benign, 2=Malignant`.\n\nEvidence:\n" + "\n".join(f"- {line}" for line in evidence) + "\n",
        encoding="utf-8",
    )

    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    class_metric_rows: list[dict[str, Any]] = []
    matrices: dict[str, list[list[list[float]]]] = {model: [] for model in MODEL_NAMES}
    alignment_notes: list[str] = []
    for seed in seeds:
        predictions = model_predictions(args.source_root, seed)
        identity_sets = {model: set(records) for model, records in predictions.items()}
        ids, aligned = align_prediction_maps(predictions, require_identical=False)
        excluded = {model: len(identity_sets[model] - set(ids)) for model in MODEL_NAMES}
        alignment_notes.append(
            f"seed{seed}: {len(ids)} common test identities; model-specific identities excluded before comparison: {excluded}"
        )
        for model in MODEL_NAMES:
            labels = [record.label for record in aligned[model]]
            predicted = [record.prediction if record.prediction is not None else max(range(3), key=lambda index: record.probabilities[index]) for record in aligned[model]]
            matrix = multiclass_confusion(labels, predicted, 3)
            normalized = row_normalize_confusion(matrix)
            matrices[model].append(normalized)
            raw_rows.extend(matrix_rows(model, seed, matrix, class_names))
            normalized_rows.extend(matrix_rows(model, seed, normalized, class_names))
            per_class_metrics = multiclass_class_metrics(matrix)
            for metric in per_class_metrics:
                class_index = int(metric["class_index"])
                class_metric_rows.append({"model": model, "seed": seed, "class_index": class_index, "class_name": class_names[class_index], **metric})
            class_metric_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "class_index": -1,
                    "class_name": "macro",
                    "precision": statistics.mean(float(row["precision"]) for row in per_class_metrics),
                    "recall": statistics.mean(float(row["recall"]) for row in per_class_metrics),
                    "f1": statistics.mean(float(row["f1"]) for row in per_class_metrics),
                    "support": sum(float(row["support"]) for row in per_class_metrics),
                }
            )

    mean_matrices: dict[str, list[list[float]]] = {}
    std_matrices: dict[str, list[list[float]]] = {}
    mean_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    for model, model_matrices in matrices.items():
        mean_matrix = [[statistics.mean(matrix[row][col] for matrix in model_matrices) for col in range(3)] for row in range(3)]
        std_matrix = [[statistics.stdev(matrix[row][col] for matrix in model_matrices) if len(model_matrices) > 1 else 0.0 for col in range(3)] for row in range(3)]
        mean_matrices[model] = mean_matrix
        std_matrices[model] = std_matrix
        mean_rows.extend(matrix_rows(model, -1, mean_matrix, class_names))
        std_rows.extend(matrix_rows(model, -1, std_matrix, class_names))

    summary_rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for class_index, class_name in [*class_names.items(), (-1, "macro")]:
            records = [row for row in class_metric_rows if row["model"] == model and row["class_index"] == class_index]
            record: dict[str, Any] = {"model": model, "class_index": class_index, "class_name": class_name, "n_seeds": len(records)}
            for metric in ("precision", "recall", "f1", "support"):
                values = [float(row[metric]) for row in records]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary_rows.append(record)

    matrix_fields = ["model", "seed", "true_class_index", "true_class", "predicted_class_index", "predicted_class", "value"]
    metric_fields = ["model", "seed", "class_index", "class_name", "precision", "recall", "f1", "support"]
    summary_fields = ["model", "class_index", "class_name", "n_seeds", "precision_mean", "precision_sd", "recall_mean", "recall_sd", "f1_mean", "f1_sd", "support_mean", "support_sd"]
    write_csv_rows(args.output_dir / "triclass_confusion_raw_by_seed.csv", raw_rows, matrix_fields)
    write_csv_rows(args.output_dir / "triclass_confusion_normalized_by_seed.csv", normalized_rows, matrix_fields)
    write_csv_rows(args.output_dir / "triclass_confusion_mean.csv", mean_rows, matrix_fields)
    write_csv_rows(args.output_dir / "triclass_confusion_std.csv", std_rows, matrix_fields)
    write_csv_rows(args.output_dir / "triclass_class_metrics_by_seed.csv", class_metric_rows, metric_fields)
    write_csv_rows(args.output_dir / "triclass_class_metrics_summary.csv", summary_rows, summary_fields)

    labels = ["Normal", "Benign", "Malignant"]
    for model in MODEL_NAMES:
        plot_matrix(args.output_dir / f"triclass_confusion_{model}", model.replace("_", "-"), mean_matrices[model], std_matrices[model], labels)
    plot_comparison(args.output_dir / "triclass_confusion_comparison", mean_matrices, std_matrices, labels)

    def strongest_error(model: str, true_index: int) -> tuple[str, float]:
        values = [(class_names[pred], mean_matrices[model][true_index][pred]) for pred in range(3) if pred != true_index]
        return max(values, key=lambda item: item[1])

    teacher_benign = strongest_error("TRI_T", 1)
    student_malignant = strongest_error("TRI_SKD", 2)
    report = [
        "# Triclass Confusion Analysis Report",
        "",
        *[f"- {line}" for line in alignment_notes],
        "",
        f"TRI-T benign cases were most often assigned to `{teacher_benign[0]}` (mean row proportion {teacher_benign[1]:.4f}).",
        f"TRI-SKD malignant cases were most often assigned to `{student_malignant[0]}` (mean row proportion {student_malignant[1]:.4f}).",
        "The matrices show class-specific redistribution across teacher and students. They support a descriptive statement that teacher class capabilities were not reproduced identically by the distilled student.",
        "No causal attribution to CNV is made because the comparison does not isolate CNV information from branch capacity and other teacher differences.",
    ]
    (args.output_dir / "triclass_confusion_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[OK] triclass confusion analysis seeds={len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
