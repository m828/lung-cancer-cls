#!/usr/bin/env python3
"""Analyze uniform, true-confidence, and shuffled-confidence controls."""

from __future__ import annotations

import argparse
import csv
import math
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
from experiments.utils.attribution_audit import AttributionAuditError, write_csv_rows  # noqa: E402


ARMS = ("uniform", "true_confidence", "shuffled_confidence")
EFFECTS = {
    "true_minus_uniform": {"true_confidence": 1.0, "uniform": -1.0},
    "true_minus_shuffled": {"true_confidence": 1.0, "shuffled_confidence": -1.0},
    "shuffled_minus_uniform": {"shuffled_confidence": 1.0, "uniform": -1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--expected-seeds", default="42,43,44,45")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def audit_weights(root: Path, run_mode: str) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    notes: list[str] = []
    for mapping_path in sorted(root.rglob("kd_weight_mapping.csv")):
        if "shuffled_confidence" not in str(mapping_path):
            continue
        if run_mode not in mapping_path.parts:
            continue
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        original = sorted(float(row["original_weight"]) for row in records)
        shuffled = sorted(float(row["effective_weight"]) for row in records)
        equal = len(original) == len(shuffled) and all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(original, shuffled))
        changed = sum(str(row.get("sample_id")) != str(row.get("donor_sample_id")) for row in records)
        if not equal:
            raise AttributionAuditError(f"weight distribution changed in {mapping_path}")
        rows.append(
            {
                "mapping_path": str(mapping_path),
                "n_samples": len(records),
                "distribution_equal": equal,
                "changed_assignments": changed,
                "all_assignments_changed": changed == len(records),
            }
        )
        notes.append(f"- `{mapping_path}`: multiset preserved; {changed}/{len(records)} assignments changed.")
    return rows, notes


def main() -> int:
    args = parse_args()
    iterations = min(args.bootstrap_iters, 100) if args.smoke else args.bootstrap_iters
    runs, audit = discover_arm_runs(args.root, ARMS, run_mode="smoke" if args.smoke else "full")
    completion_rows, all_complete = expected_run_completeness(runs, ARMS, parse_seed_list(args.expected_seeds))
    _, summaries = write_standard_outputs(
        args.root,
        "shuffled_confidence_seed_metrics.csv",
        "shuffled_confidence_summary.csv",
        runs,
    )
    bootstrap_rows = []
    for index, (name, coefficients) in enumerate(EFFECTS.items()):
        rows, _ = paired_contrast_bootstrap(
            runs,
            coefficients,
            iterations=iterations,
            random_seed=15402 + index,
            effect_name=name,
        )
        bootstrap_rows.extend(rows)
    fields = ["effect", "metric", "point_estimate", "ci95_low", "ci95_high", "ci_crosses_zero", "n_seeds", "identity_count_by_seed", "n_bootstrap", "status"]
    write_csv_rows(args.root / "shuffled_confidence_bootstrap.csv", bootstrap_rows, fields)
    weight_rows, notes = audit_weights(args.root, "smoke" if args.smoke else "full")
    write_csv_rows(
        args.root / "weight_distribution_audit.csv",
        weight_rows,
        ["mapping_path", "n_samples", "distribution_equal", "changed_assignments", "all_assignments_changed"],
    )
    (args.root / "weight_mapping_audit.md").write_text(
        "# Shuffled-Confidence Weight Mapping Audit\n\n" + ("\n".join(notes) if notes else "MISSING: no shuffled mapping files found.") + "\n",
        encoding="utf-8",
    )
    write_csv_rows(
        args.root / "run_inventory.csv",
        completion_rows,
        ["arm", "seed", "status", "run_dir", "reason", "threshold", "test_rows"],
    )
    write_markdown_table(args.root / "shuffled_confidence_report.md", "Shuffled-Confidence Control", fields, bootstrap_rows)
    print(f"[OK] shuffled-confidence analysis complete, mappings={len(weight_rows)}")
    return 0 if all_complete and len(weight_rows) == len(parse_seed_list(args.expected_seeds)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
