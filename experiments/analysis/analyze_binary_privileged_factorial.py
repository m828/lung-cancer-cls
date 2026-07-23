#!/usr/bin/env python3
"""Analyze the matched teacher-modality x KD-weight binary factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attribution_common import (  # type: ignore
    BINARY_METRICS,
    discover_arm_runs,
    expected_run_completeness,
    parse_seed_list,
    paired_contrast_bootstrap,
    standard_summary_fields,
    write_markdown_table,
    write_standard_outputs,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import atomic_write_json, write_csv_rows  # noqa: E402


ARMS = (
    "S0_MATCHED",
    "KD_CT_TEXT_UNIFORM",
    "KD_CT_TEXT_CONFIDENCE",
    "KD_CT_TEXT_CNV_UNIFORM",
    "KD_CT_TEXT_CNV_CONFIDENCE",
)

EFFECTS = {
    "cnv_teacher_effect_uniform": {"KD_CT_TEXT_CNV_UNIFORM": 1.0, "KD_CT_TEXT_UNIFORM": -1.0},
    "cnv_teacher_effect_confidence": {"KD_CT_TEXT_CNV_CONFIDENCE": 1.0, "KD_CT_TEXT_CONFIDENCE": -1.0},
    "confidence_effect_ct_text_teacher": {"KD_CT_TEXT_CONFIDENCE": 1.0, "KD_CT_TEXT_UNIFORM": -1.0},
    "confidence_effect_ct_text_cnv_teacher": {"KD_CT_TEXT_CNV_CONFIDENCE": 1.0, "KD_CT_TEXT_CNV_UNIFORM": -1.0},
    "final_vs_matched_supervised": {"KD_CT_TEXT_CNV_CONFIDENCE": 1.0, "S0_MATCHED": -1.0},
    "factorial_interaction": {
        "KD_CT_TEXT_CNV_CONFIDENCE": 1.0,
        "KD_CT_TEXT_CNV_UNIFORM": -1.0,
        "KD_CT_TEXT_CONFIDENCE": -1.0,
        "KD_CT_TEXT_UNIFORM": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=5402)
    parser.add_argument("--expected-seeds", default="42,43,44,45")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    iterations = min(args.bootstrap_iters, 100) if args.smoke else args.bootstrap_iters
    runs, audit_rows = discover_arm_runs(args.root, ARMS, run_mode="smoke" if args.smoke else "full")
    expected_rows, all_complete = expected_run_completeness(runs, ARMS, parse_seed_list(args.expected_seeds))
    _, summaries = write_standard_outputs(
        args.root,
        "binary_factorial_seed_metrics.csv",
        "binary_factorial_summary.csv",
        runs,
    )

    effect_rows = []
    alignment_rows = []
    for offset, (effect_name, coefficients) in enumerate(EFFECTS.items()):
        rows, alignment = paired_contrast_bootstrap(
            runs,
            coefficients,
            iterations=iterations,
            random_seed=args.bootstrap_seed + offset,
            effect_name=effect_name,
        )
        effect_rows.extend(rows)
        alignment_rows.extend(alignment)

    effect_fields = [
        "effect",
        "metric",
        "point_estimate",
        "ci95_low",
        "ci95_high",
        "ci_crosses_zero",
        "n_seeds",
        "identity_count_by_seed",
        "n_bootstrap",
        "status",
    ]
    write_csv_rows(args.root / "binary_factorial_effects.csv", effect_rows, effect_fields)
    write_csv_rows(args.root / "binary_factorial_bootstrap.csv", effect_rows, effect_fields)
    write_csv_rows(
        args.root / "binary_factorial_run_manifest.csv",
        expected_rows,
        ["arm", "seed", "status", "run_dir", "reason", "threshold", "test_rows"],
    )

    paper_rows = []
    for row in summaries:
        record = {"arm": row["arm"], "n_seeds": row["n_seeds"]}
        for metric in BINARY_METRICS:
            record[metric] = f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"
        paper_rows.append(record)
    write_markdown_table(
        args.root / "binary_factorial_paper_table.md",
        "Matched Binary Factorial Results",
        ["arm", "n_seeds", *BINARY_METRICS],
        paper_rows,
    )

    interaction = [row for row in effect_rows if row["effect"] == "factorial_interaction"]
    write_markdown_table(
        args.root / "binary_factorial_interaction_report.md",
        "Teacher Modality x KD Weighting Interaction",
        effect_fields,
        interaction,
    )
    alignment_lines = ["# Prediction Alignment Audit", ""]
    for row in expected_rows:
        alignment_lines.append(
            f"- `{row.get('arm')}` seed `{row.get('seed')}`: {row.get('status')} "
            f"({row.get('test_rows', '')} test identities) {row.get('reason', '')}"
        )
    alignment_lines.extend(["", "Bootstrap contrasts resample common test identities within each seed and then average seed-level effects; seed x identity rows are never stacked."])
    (args.root / "prediction_alignment_audit.md").write_text("\n".join(alignment_lines) + "\n", encoding="utf-8")
    atomic_write_json(
        args.root / "analysis_config.json",
        {
            "arms": list(ARMS),
            "effects": EFFECTS,
            "metrics": list(BINARY_METRICS),
            "bootstrap_iterations": iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "resampling": "common identity indices within each seed, then mean seed contrast",
            "smoke": args.smoke,
        },
    )
    complete_arms = sum(bool(runs[arm]) for arm in ARMS)
    print(f"[OK] factorial analysis complete arms={complete_arms}/{len(ARMS)} all_expected={all_complete}")
    return 0 if all_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
