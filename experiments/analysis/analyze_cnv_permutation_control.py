#!/usr/bin/env python3
"""Analyze real-CNV, split-local permuted-CNV, and CT-text controls."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from attribution_common import (  # type: ignore
    discover_arm_runs,
    expected_run_completeness,
    paired_contrast_bootstrap,
    parse_seed_list,
    write_markdown_table,
    write_standard_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import AttributionAuditError, sha256_file, write_csv_rows  # noqa: E402


TEACHER_ARMS = ("ct_text", "permuted_cnv", "real_cnv")
STUDENT_ARMS = ("ct_text_confidence", "permuted_cnv_confidence", "real_cnv_confidence")
TEACHER_EFFECTS = {
    "real_cnv_minus_permuted_cnv": {"real_cnv": 1.0, "permuted_cnv": -1.0},
    "real_cnv_minus_ct_text": {"real_cnv": 1.0, "ct_text": -1.0},
    "permuted_cnv_minus_ct_text": {"permuted_cnv": 1.0, "ct_text": -1.0},
}
STUDENT_EFFECTS = {
    "real_cnv_kd_minus_permuted_cnv_kd": {"real_cnv_confidence": 1.0, "permuted_cnv_confidence": -1.0},
    "real_cnv_kd_minus_ct_text_teacher_kd": {"real_cnv_confidence": 1.0, "ct_text_confidence": -1.0},
    "permuted_cnv_kd_minus_ct_text_teacher_kd": {"permuted_cnv_confidence": 1.0, "ct_text_confidence": -1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--expected-seeds", default="42,43,44,45")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def audit_permutations(root: Path, run_mode: str) -> tuple[list[dict[str, object]], list[str]]:
    consolidated: list[dict[str, object]] = []
    notes = ["# CNV Permutation Integrity Audit", ""]
    for path in sorted((root / "permutation_manifests").rglob("*.csv")) if (root / "permutation_manifests").is_dir() else []:
        if not path.name.startswith(f"{run_mode}_seed"):
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        split_by_id = {str(row.get("sample_id", "")): str(row.get("split", "")) for row in rows}
        duplicate_targets = len(split_by_id) != len(rows)
        fixed = [row for row in rows if int(float(row.get("is_fixed_point", 0) or 0)) != 0]
        cross = [
            row
            for row in rows
            if split_by_id.get(str(row.get("donor_sample_id", ""))) != str(row.get("split", ""))
        ]
        hash_mismatches = [
            row
            for row in rows
            if row.get("donor_feature_hash") and row.get("target_feature_hash_after")
            and row["donor_feature_hash"] != row["target_feature_hash_after"]
        ]
        if duplicate_targets or fixed or cross or hash_mismatches:
            raise AttributionAuditError(
                f"invalid CNV permutation {path}: duplicates={int(duplicate_targets)}, "
                f"fixed={len(fixed)}, cross_split={len(cross)}, row_hash_mismatch={len(hash_mismatches)}"
            )
        for row in rows:
            consolidated.append({"mapping_file": str(path), **row})
        notes.append(f"- `{path}`: rows={len(rows)}, fixed points=0, cross-split donors=0, sha256=`{sha256_file(path)}`.")
    if len(notes) == 2:
        notes.append("MISSING: no permutation manifests are available yet.")
    return consolidated, notes


def bootstrap_effects(runs, definitions, iterations, seed_base):
    output = []
    for index, (name, coefficients) in enumerate(definitions.items()):
        rows, _ = paired_contrast_bootstrap(
            runs,
            coefficients,
            iterations=iterations,
            random_seed=seed_base + index,
            effect_name=name,
        )
        output.extend(rows)
    return output


def main() -> int:
    args = parse_args()
    iterations = min(args.bootstrap_iters, 100) if args.smoke else args.bootstrap_iters
    requested_mode = "smoke" if args.smoke else "full"
    teacher_runs, teacher_audit = discover_arm_runs(args.root / "teachers", TEACHER_ARMS, run_mode=requested_mode)
    student_runs, student_audit = discover_arm_runs(args.root / "students", STUDENT_ARMS, run_mode=requested_mode)
    expected_seeds = parse_seed_list(args.expected_seeds)
    teacher_completion, teachers_complete = expected_run_completeness(teacher_runs, TEACHER_ARMS, expected_seeds)
    student_completion, students_complete = expected_run_completeness(student_runs, STUDENT_ARMS, expected_seeds)
    write_standard_outputs(args.root, "teacher_seed_metrics.csv", "teacher_summary.csv", teacher_runs)
    write_standard_outputs(args.root, "student_seed_metrics.csv", "student_summary.csv", student_runs)
    teacher_bootstrap = bootstrap_effects(teacher_runs, TEACHER_EFFECTS, iterations, 25402)
    student_bootstrap = bootstrap_effects(student_runs, STUDENT_EFFECTS, iterations, 35402)
    fields = ["effect", "metric", "point_estimate", "ci95_low", "ci95_high", "ci_crosses_zero", "n_seeds", "identity_count_by_seed", "n_bootstrap", "status"]
    write_csv_rows(args.root / "teacher_bootstrap.csv", teacher_bootstrap, fields)
    write_csv_rows(args.root / "student_bootstrap.csv", student_bootstrap, fields)
    mapping_rows, audit_notes = audit_permutations(args.root, requested_mode)
    mapping_fields = sorted({key for row in mapping_rows for key in row}) if mapping_rows else ["mapping_file", "split", "sample_id", "donor_sample_id"]
    write_csv_rows(args.root / "cnv_permutation_mapping.csv", mapping_rows, mapping_fields)
    (args.root / "cnv_permutation_integrity_audit.md").write_text("\n".join(audit_notes) + "\n", encoding="utf-8")
    report_rows = [{"level": "teacher", **row} for row in teacher_bootstrap] + [{"level": "student", **row} for row in student_bootstrap]
    write_markdown_table(args.root / "cnv_permutation_report.md", "CNV Permutation Control", ["level", *fields], report_rows)
    write_csv_rows(args.root / "run_inventory.csv", teacher_completion + student_completion, ["arm", "seed", "status", "run_dir", "reason", "threshold", "test_rows"])
    print(f"[OK] CNV permutation analysis complete mappings={len(mapping_rows)}")
    mapping_files = {str(row["mapping_file"]) for row in mapping_rows}
    return 0 if teachers_complete and students_complete and len(mapping_files) == len(expected_seeds) else 2


if __name__ == "__main__":
    raise SystemExit(main())
