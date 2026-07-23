#!/usr/bin/env python3
"""Analyze whether full-modality teacher corrections are reflected in the KD student."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from attribution_common import percentile, positive_probability  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import (  # noqa: E402
    AttributionAuditError,
    Prediction,
    align_prediction_maps,
    binary_metrics,
    load_prediction_csv,
    partition_teacher_correction_groups,
    select_binary_threshold,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ct-text-teacher-pattern",
        default="../outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed{seed}",
    )
    parser.add_argument(
        "--full-teacher-pattern",
        default="../outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed{seed}",
    )
    parser.add_argument(
        "--supervised-pattern",
        default="../outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed{seed}",
    )
    parser.add_argument(
        "--kd-pattern",
        default="../outputs0535_student_kd_refinement/refined_candidates/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed{seed}",
    )
    parser.add_argument(
        "--supervised-run-override",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Override the supervised run for a logical seed (for example the locked seed-44 repeat).",
    )
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_run(pattern: str, seed: int) -> Path:
    return Path(pattern.format(seed=seed)).expanduser().resolve()


def parse_run_overrides(values: Sequence[str]) -> dict[int, Path]:
    overrides: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run override must use SEED=PATH: {value}")
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text.strip())
        if seed in overrides:
            raise ValueError(f"duplicate supervised override for seed {seed}")
        overrides[seed] = Path(path_text).expanduser().resolve()
    return overrides


def model_threshold(run_dir: Path) -> float:
    val = load_prediction_csv(run_dir / "val_predictions.csv")
    ids = sorted(val)
    return select_binary_threshold(
        [val[sid].label for sid in ids],
        [positive_probability(val[sid]) for sid in ids],
    )


def predictions_at(records: Sequence[Prediction], threshold: float) -> list[int]:
    return [int(positive_probability(record) >= threshold) for record in records]


def group_metrics(
    group: str,
    ids: Sequence[str],
    labels: Sequence[int],
    student_records: Sequence[Prediction],
    full_teacher_records: Sequence[Prediction],
    student_threshold: float,
) -> dict[str, Any]:
    student_probs = [positive_probability(record) for record in student_records]
    student_predictions = [int(value >= student_threshold) for value in student_probs]
    teacher_predictions = [int(max(record.probabilities) == record.probabilities[1]) for record in full_teacher_records]
    metrics = binary_metrics(labels, student_probs, student_threshold) if labels else {}
    correctness = [int(prediction == label) for prediction, label in zip(student_predictions, labels)]
    return {
        "group": group,
        "n_cases": len(ids),
        "accuracy": statistics.mean(correctness) if correctness else float("nan"),
        "sensitivity": metrics.get("sensitivity", float("nan")),
        "specificity": metrics.get("specificity", float("nan")),
        "mean_predicted_probability": statistics.mean(student_probs) if student_probs else float("nan"),
        "mean_correctness": statistics.mean(correctness) if correctness else float("nan"),
        "mean_teacher_confidence": statistics.mean(max(record.probabilities) for record in full_teacher_records) if full_teacher_records else float("nan"),
        "student_teacher_disagreement": statistics.mean(int(student != teacher) for student, teacher in zip(student_predictions, teacher_predictions)) if labels else float("nan"),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = int(b) + int(c)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def seed_analysis(seed: int, paths: Mapping[str, Path], teacher_threshold_mode: str) -> dict[str, Any]:
    prediction_maps = {
        name: load_prediction_csv(path / "test_predictions.csv")
        for name, path in paths.items()
    }
    ids, aligned = align_prediction_maps(prediction_maps, require_identical=True)
    labels = [record.label for record in aligned["ct_text_teacher"]]
    if teacher_threshold_mode == "fixed_0.5":
        ct_threshold = full_threshold = 0.5
    else:
        ct_threshold = model_threshold(paths["ct_text_teacher"])
        full_threshold = model_threshold(paths["full_teacher"])
    supervised_threshold = model_threshold(paths["supervised"])
    kd_threshold = model_threshold(paths["kd"])
    ct_predictions = predictions_at(aligned["ct_text_teacher"], ct_threshold)
    full_predictions = predictions_at(aligned["full_teacher"], full_threshold)
    groups = partition_teacher_correction_groups(labels, ct_predictions, full_predictions)

    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for index, sid in enumerate(ids):
        row: dict[str, Any] = {
            "seed": seed,
            "threshold_mode": teacher_threshold_mode,
            "sample_id": sid,
            "label": labels[index],
            "group": groups[index],
            "ct_text_teacher_probability": positive_probability(aligned["ct_text_teacher"][index]),
            "full_teacher_probability": positive_probability(aligned["full_teacher"][index]),
            "full_teacher_confidence": max(aligned["full_teacher"][index].probabilities),
            "supervised_probability": positive_probability(aligned["supervised"][index]),
            "kd_probability": positive_probability(aligned["kd"][index]),
            "supervised_correct": int(int(positive_probability(aligned["supervised"][index]) >= supervised_threshold) == labels[index]),
            "kd_correct": int(int(positive_probability(aligned["kd"][index]) >= kd_threshold) == labels[index]),
        }
        case_rows.append(row)

    for group in ("A", "B", "C", "D"):
        indices = [index for index, value in enumerate(groups) if value == group]
        for student_name, threshold in (("supervised", supervised_threshold), ("kd", kd_threshold)):
            metrics = group_metrics(
                group,
                [ids[index] for index in indices],
                [labels[index] for index in indices],
                [aligned[student_name][index] for index in indices],
                [aligned["full_teacher"][index] for index in indices],
                threshold,
            )
            metric_rows.append(
                {
                    "seed": seed,
                    "threshold_mode": teacher_threshold_mode,
                    "student": student_name,
                    "student_threshold": threshold,
                    **metrics,
                }
            )

    group_a = [row for row in case_rows if row["group"] == "A"]
    b = sum(row["kd_correct"] == 1 and row["supervised_correct"] == 0 for row in group_a)
    c = sum(row["kd_correct"] == 0 and row["supervised_correct"] == 1 for row in group_a)
    return {
        "seed": seed,
        "ids": ids,
        "aligned": aligned,
        "labels": labels,
        "groups": groups,
        "case_rows": case_rows,
        "metric_rows": metric_rows,
        "thresholds": {
            "ct_text_teacher": ct_threshold,
            "full_teacher": full_threshold,
            "supervised": supervised_threshold,
            "kd": kd_threshold,
        },
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": exact_mcnemar_p(b, c),
    }


def bootstrap_group_a(seed_results: Sequence[dict[str, Any]], iterations: int) -> list[dict[str, Any]]:
    rng = random.Random(45402)
    deltas: list[float] = []
    sample_sizes: list[int] = []
    for _ in range(iterations):
        per_seed: list[float] = []
        for result in seed_results:
            count = len(result["ids"])
            sample = [rng.randrange(count) for _ in range(count)]
            labels = [result["labels"][index] for index in sample]
            ct_predictions = predictions_at([result["aligned"]["ct_text_teacher"][index] for index in sample], result["thresholds"]["ct_text_teacher"])
            full_predictions = predictions_at([result["aligned"]["full_teacher"][index] for index in sample], result["thresholds"]["full_teacher"])
            groups = partition_teacher_correction_groups(labels, ct_predictions, full_predictions)
            group_indices = [position for position, group in enumerate(groups) if group == "A"]
            if not group_indices:
                continue
            supervised = predictions_at([result["aligned"]["supervised"][sample[position]] for position in group_indices], result["thresholds"]["supervised"])
            kd = predictions_at([result["aligned"]["kd"][sample[position]] for position in group_indices], result["thresholds"]["kd"])
            group_labels = [labels[position] for position in group_indices]
            supervised_accuracy = statistics.mean(int(pred == label) for pred, label in zip(supervised, group_labels))
            kd_accuracy = statistics.mean(int(pred == label) for pred, label in zip(kd, group_labels))
            per_seed.append(kd_accuracy - supervised_accuracy)
            sample_sizes.append(len(group_indices))
        if per_seed:
            deltas.append(statistics.mean(per_seed))
    if not deltas:
        return [{"metric": "accuracy", "delta_kd_minus_supervised": "", "ci95_low": "", "ci95_high": "", "ci_crosses_zero": "", "n_bootstrap_valid": 0, "status": "GROUP_A_TOO_SMALL"}]
    observed_seed_deltas = []
    for result in seed_results:
        group_rows = [row for row in result["case_rows"] if row["group"] == "A"]
        if group_rows:
            observed_seed_deltas.append(statistics.mean(row["kd_correct"] - row["supervised_correct"] for row in group_rows))
    point = statistics.mean(observed_seed_deltas) if observed_seed_deltas else float("nan")
    lower, upper = percentile(deltas, 0.025), percentile(deltas, 0.975)
    return [
        {
            "metric": "accuracy",
            "delta_kd_minus_supervised": point,
            "ci95_low": lower,
            "ci95_high": upper,
            "ci_crosses_zero": lower <= 0 <= upper,
            "n_bootstrap_valid": len(deltas),
            "median_group_a_resample_size": statistics.median(sample_sizes) if sample_sizes else 0,
            "status": "descriptive_with_bootstrap_interval",
        }
    ]


def ensemble_sensitivity(primary_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not primary_results:
        return []
    common = sorted(set.intersection(*(set(result["ids"]) for result in primary_results)))
    rows: list[dict[str, Any]] = []
    for sid in common:
        records: dict[str, list[Prediction]] = {name: [] for name in ("ct_text_teacher", "full_teacher", "supervised", "kd")}
        for result in primary_results:
            index = result["ids"].index(sid)
            for name in records:
                records[name].append(result["aligned"][name][index])
        label = records["ct_text_teacher"][0].label
        probabilities = {name: statistics.mean(positive_probability(record) for record in values) for name, values in records.items()}
        ct_correct = int(probabilities["ct_text_teacher"] >= 0.5) == label
        full_correct = int(probabilities["full_teacher"] >= 0.5) == label
        group = "A" if not ct_correct and full_correct else ("B" if ct_correct and full_correct else ("C" if ct_correct and not full_correct else "D"))
        rows.append({"sample_id": sid, "label": label, "group": group, **probabilities})
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    supervised_overrides = parse_run_overrides(args.supervised_run_override)
    primary_results: list[dict[str, Any]] = []
    sensitivity_results: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        paths = {
            "ct_text_teacher": resolve_run(args.ct_text_teacher_pattern, seed),
            "full_teacher": resolve_run(args.full_teacher_pattern, seed),
            "supervised": supervised_overrides.get(seed, resolve_run(args.supervised_pattern, seed)),
            "kd": resolve_run(args.kd_pattern, seed),
        }
        absent = [f"{name}:{path}" for name, path in paths.items() if not (path / "test_predictions.csv").is_file() or not (path / "val_predictions.csv").is_file()]
        if absent:
            missing.append(f"seed{seed}: " + ", ".join(absent))
            continue
        primary_results.append(seed_analysis(seed, paths, "fixed_0.5"))
        sensitivity_results.append(seed_analysis(seed, paths, "validation_selected"))

    all_results = primary_results + sensitivity_results
    case_rows = [row for result in all_results for row in result["case_rows"]]
    metric_rows = [row for result in all_results for row in result["metric_rows"]]
    count_rows = []
    for result in all_results:
        for group in ("A", "B", "C", "D"):
            count_rows.append(
                {
                    "seed": result["seed"],
                    "threshold_mode": result["case_rows"][0]["threshold_mode"],
                    "group": group,
                    "count": result["groups"].count(group),
                    "mcnemar_b_group_a": result["mcnemar_b"] if group == "A" else "",
                    "mcnemar_c_group_a": result["mcnemar_c"] if group == "A" else "",
                    "exact_mcnemar_p_group_a": result["mcnemar_p"] if group == "A" else "",
                }
            )
    iterations = min(args.bootstrap_iters, 100) if args.smoke else args.bootstrap_iters
    bootstrap_rows = bootstrap_group_a(primary_results, iterations)
    ensemble_rows = ensemble_sensitivity(primary_results)

    write_csv_rows(args.output_dir / "case_level_predictions.csv", case_rows, list(case_rows[0]) if case_rows else ["seed", "sample_id", "status"])
    write_csv_rows(args.output_dir / "teacher_correction_groups_by_seed.csv", case_rows, list(case_rows[0]) if case_rows else ["seed", "sample_id", "group"])
    write_csv_rows(args.output_dir / "teacher_correction_group_counts.csv", count_rows, ["seed", "threshold_mode", "group", "count", "mcnemar_b_group_a", "mcnemar_c_group_a", "exact_mcnemar_p_group_a"])
    write_csv_rows(args.output_dir / "teacher_correction_student_metrics.csv", metric_rows, list(metric_rows[0]) if metric_rows else ["seed", "group", "student", "status"])
    write_csv_rows(args.output_dir / "teacher_correction_bootstrap.csv", bootstrap_rows, list(bootstrap_rows[0]))
    write_csv_rows(args.output_dir / "teacher_correction_ensemble_sensitivity.csv", ensemble_rows, list(ensemble_rows[0]) if ensemble_rows else ["sample_id", "group", "status"])
    group_a_cases = [row for row in case_rows if row.get("threshold_mode") == "fixed_0.5" and row.get("group") == "A"]
    write_csv_rows(args.output_dir / "teacher_correction_case_list.csv", group_a_cases, list(group_a_cases[0]) if group_a_cases else ["seed", "sample_id", "group"])

    primary_counts = [result["groups"].count("A") for result in primary_results]
    primary_group_a = [
        row
        for result in primary_results
        for row in result["case_rows"]
        if row["group"] == "A"
    ]
    kd_correct_count = sum(int(row["kd_correct"]) for row in primary_group_a)
    supervised_correct_count = sum(int(row["supervised_correct"]) for row in primary_group_a)
    kd_only_correct = sum(
        int(row["kd_correct"] == 1 and row["supervised_correct"] == 0)
        for row in primary_group_a
    )
    supervised_only_correct = sum(
        int(row["kd_correct"] == 0 and row["supervised_correct"] == 1)
        for row in primary_group_a
    )
    inherited_confidence = [
        float(row["full_teacher_confidence"])
        for row in primary_group_a
        if int(row["kd_correct"]) == 1
    ]
    not_inherited_confidence = [
        float(row["full_teacher_confidence"])
        for row in primary_group_a
        if int(row["kd_correct"]) == 0
    ]
    bootstrap_result = bootstrap_rows[0] if bootstrap_rows else {}
    ensemble_group_a_count = sum(row.get("group") == "A" for row in ensemble_rows)
    report = [
        "# Teacher-Correction Transfer Report",
        "",
        f"Complete seeds: {len(primary_results)}/{len(seeds)}.",
        f"Group A counts by seed: {primary_counts if primary_counts else 'MISSING'}.",
        "Group A denotes cases misclassified by the CT-text teacher and correctly classified by the CT-text-CNV teacher.",
        "Student comparisons use each student's validation-selected threshold; primary teacher grouping uses threshold 0.5.",
        "The analysis asks whether students use additional full-modality teacher supervision; it does not demonstrate that the student learned genomic representations.",
        f"Supervised-run overrides: {', '.join(f'seed{seed}={path}' for seed, path in sorted(supervised_overrides.items())) if supervised_overrides else 'none'}.",
        "",
        "## Primary descriptive findings",
        "",
        f"Across seeds, Group A contained {len(primary_group_a)} seed-case observations; these are not treated as independent patients.",
        f"The KD student was correct for {kd_correct_count}/{len(primary_group_a) if primary_group_a else 0} Group A observations, compared with {supervised_correct_count}/{len(primary_group_a) if primary_group_a else 0} for the hard-label student.",
        f"KD-only correct observations: {kd_only_correct}; supervised-only correct observations: {supervised_only_correct}.",
        f"The ensemble sensitivity grouping contained {ensemble_group_a_count} Group A identities.",
        "Cases for which the KD student was correct are interpreted as full-teacher corrections reflected in student behavior; incorrect KD cases are corrections not inherited under this operational definition.",
        "",
        "## Confidence and interval assessment",
        "",
        f"Mean full-teacher confidence was {statistics.mean(inherited_confidence):.4f} when the KD student was correct and {statistics.mean(not_inherited_confidence):.4f} when it was incorrect." if inherited_confidence and not_inherited_confidence else "Teacher-confidence stratification was unavailable because one outcome group was empty.",
        f"The seed-averaged Group A accuracy difference (KD minus supervised) was {bootstrap_result.get('delta_kd_minus_supervised', 'MISSING')}; the 95% bootstrap interval was [{bootstrap_result.get('ci95_low', 'MISSING')}, {bootstrap_result.get('ci95_high', 'MISSING')}].",
        "Because the interval includes zero, the transfer comparison remains descriptive rather than interval-supported.",
        "Confidence associations are descriptive and do not establish that confidence measures genomic knowledge transferability.",
        "",
        "## Seed-level exact paired comparisons",
        "",
    ]
    for result in primary_results:
        report.append(
            f"- seed{result['seed']}: Group A n={result['groups'].count('A')}, "
            f"KD-only correct={result['mcnemar_b']}, supervised-only correct={result['mcnemar_c']}, "
            f"exact McNemar p={result['mcnemar_p']:.4f}."
        )
    report.append("")
    if missing:
        report.extend(["## Missing inputs", *[f"- {item}" for item in missing], ""])
    if primary_counts and min(primary_counts) < 5:
        report.append("Group A is small in at least one seed; inference is treated as descriptive and exact McNemar results are reported without forced asymptotic claims.")
    (args.output_dir / "teacher_correction_transfer_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[OK] teacher-correction analysis seeds={len(primary_results)}")
    return 0 if len(primary_results) == len(seeds) else 2


if __name__ == "__main__":
    raise SystemExit(main())
