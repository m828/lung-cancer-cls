#!/usr/bin/env python3
"""Summarize test evaluations of validation-selected checkpoints only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.attribution_audit import (  # noqa: E402
    binary_metrics,
    load_prediction_csv,
    select_binary_threshold,
    sha256_file,
    write_csv_rows,
)


METRICS = ("auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--run-mode", choices=["smoke", "full"], default=None)
    return parser.parse_args()


def seed_from_metrics(run_dir: Path) -> int | None:
    try:
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = (payload.get("config") or {}).get("seed")
    return int(value) if value is not None else None


def arm_from_run_dir(search_root: Path, run_dir: Path) -> str:
    """Recover the experiment arm from factorial, shuffle, or CNV trees."""

    relative = run_dir.relative_to(search_root)
    parts = relative.parts
    if not parts:
        return run_dir.name
    if parts[0] in {"students", "teachers"} and len(parts) > 1:
        return parts[1]
    return parts[0]


def validation_selected_test_metrics(run_dir: Path, criterion: str) -> tuple[float | None, dict[str, Any] | None]:
    """Recompute test metrics at a threshold selected only from validation predictions."""

    prediction_dir = run_dir / "checkpoint_predictions"
    val_path = prediction_dir / f"{criterion}_val_predictions.csv"
    test_path = prediction_dir / f"{criterion}_test_predictions.csv"
    if not val_path.is_file() or not test_path.is_file():
        return None, None
    val = load_prediction_csv(val_path)
    test = load_prediction_csv(test_path)
    val_ids = sorted(val)
    test_ids = sorted(test)
    val_probs = [float(val[sid].probabilities[1]) for sid in val_ids]
    test_probs = [float(test[sid].probabilities[1]) for sid in test_ids]
    threshold = select_binary_threshold([val[sid].label for sid in val_ids], val_probs)
    metrics = binary_metrics([test[sid].label for sid in test_ids], test_probs, threshold)
    return threshold, metrics


def main() -> int:
    args = parse_args()
    search_roots = args.search_root or [args.root.parent / "01_binary_factorial", args.root.parent / "02_shuffled_confidence", args.root.parent / "03_cnv_permutation"]
    inventory: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_evaluation_files: set[Path] = set()
    for search_root in search_roots:
        for evaluation_path in sorted(search_root.rglob("checkpoint_evaluations.json")) if search_root.is_dir() else []:
            canonical_evaluation_path = evaluation_path.resolve()
            if canonical_evaluation_path in seen_evaluation_files:
                continue
            seen_evaluation_files.add(canonical_evaluation_path)
            run_dir = evaluation_path.parent
            run_complete = {}
            try:
                run_complete = json.loads((run_dir / "run_complete.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            metrics_payload = {}
            try:
                metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            observed_mode = run_complete.get("run_mode") or (metrics_payload.get("cached_kd_config") or {}).get("run_mode") or "legacy"
            if observed_mode == "full_candidate":
                observed_mode = "full"
            if args.run_mode is not None and observed_mode not in {args.run_mode, "legacy"}:
                continue
            seed = seed_from_metrics(run_dir)
            arm = arm_from_run_dir(search_root, run_dir)
            evaluations = json.loads(evaluation_path.read_text(encoding="utf-8"))
            for record in evaluations:
                checkpoint_path = Path(str(record.get("checkpoint_path", "")))
                status = str(record.get("status", "MISSING"))
                if status == "complete" and not checkpoint_path.is_file():
                    status = "MISSING_FILE"
                criterion = str(record.get("checkpoint_criterion", ""))
                selected_threshold, selected_metrics = validation_selected_test_metrics(run_dir, criterion)
                threshold_source = "validation_predictions" if selected_metrics is not None else "checkpoint_record_fallback"
                inventory.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "criterion": criterion,
                        "status": status,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
                        "test_used_for_selection": bool(record.get("test_metrics_used_for_selection", False)),
                        "validation_threshold": selected_threshold if selected_threshold is not None else record.get("validation_threshold", ""),
                        "threshold_source": threshold_source,
                    }
                )
                if status != "complete":
                    missing.append(f"{arm} seed{seed} {record.get('checkpoint_criterion')}: {status}")
                    continue
                test_metrics = selected_metrics or record.get("test_metrics") or {}
                metric_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "checkpoint_criterion": criterion,
                        "epoch": record.get("epoch"),
                        "validation_threshold": selected_threshold if selected_threshold is not None else record.get("validation_threshold", ""),
                        "threshold_source": threshold_source,
                        **{metric: test_metrics.get(metric, "") for metric in METRICS},
                        "checkpoint_path": str(checkpoint_path),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["arm"]), str(row["checkpoint_criterion"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (arm, criterion), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"arm": arm, "checkpoint_criterion": criterion, "n_seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""} and math.isfinite(float(row[metric]))]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        summaries.append(summary)

    inventory_fields = ["arm", "seed", "criterion", "status", "checkpoint_path", "checkpoint_sha256", "test_used_for_selection", "validation_threshold", "threshold_source"]
    metric_fields = ["arm", "seed", "checkpoint_criterion", "epoch", "validation_threshold", "threshold_source", *METRICS, "checkpoint_path"]
    summary_fields = ["arm", "checkpoint_criterion", "n_seeds", *[field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_sd")]]
    write_csv_rows(args.root / "checkpoint_inventory.csv", inventory, inventory_fields)
    write_csv_rows(args.root / "checkpoint_test_metrics.csv", metric_rows, metric_fields)
    write_csv_rows(args.root / "checkpoint_sensitivity_by_seed.csv", metric_rows, metric_fields)
    write_csv_rows(args.root / "checkpoint_sensitivity_summary.csv", summaries, summary_fields)

    direction_lines = [
        "# Checkpoint Stability Report",
        "",
        "Test results are reported after validation-only checkpoint and operating-threshold selection; they are never used to choose either quantity.",
        "When checkpoint prediction files are available, the operating threshold is recomputed from that checkpoint's validation predictions and then applied to its test predictions.",
        "The primary criterion remains `val_composite`; this analysis does not recommend a criterion from test performance.",
        "",
    ]
    for arm in sorted({str(row["arm"]) for row in metric_rows}):
        direction_lines.append(f"## {arm}")
        arm_rows = [row for row in metric_rows if row["arm"] == arm]
        criteria = sorted({str(row["checkpoint_criterion"]) for row in arm_rows})
        direction_lines.append(f"Available criteria: {', '.join(criteria) if criteria else 'none'}.")
        primary_rows = [row for row in arm_rows if row["checkpoint_criterion"] == "val_composite"]
        direction_lines.append(f"Primary-checkpoint seeds available: {len(primary_rows)}.")
        direction_lines.append("")

    baseline_rows = {
        (int(row["seed"]), str(row["checkpoint_criterion"])): row
        for row in metric_rows
        if row["arm"] == "S0_MATCHED" and row.get("seed") is not None
    }
    if baseline_rows:
        direction_lines.extend(["## Direction relative to S0_MATCHED", ""])
        for arm in sorted({str(row["arm"]) for row in metric_rows if row["arm"] != "S0_MATCHED"}):
            direction_lines.append(f"### {arm}")
            arm_rows = [row for row in metric_rows if row["arm"] == arm]
            for criterion in sorted({str(row["checkpoint_criterion"]) for row in arm_rows}):
                differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
                for row in arm_rows:
                    if row["checkpoint_criterion"] != criterion or row.get("seed") is None:
                        continue
                    baseline = baseline_rows.get((int(row["seed"]), criterion))
                    if baseline is None:
                        continue
                    for metric in METRICS:
                        if row.get(metric) not in {None, ""} and baseline.get(metric) not in {None, ""}:
                            differences[metric].append(float(row[metric]) - float(baseline[metric]))
                signs = []
                for metric, values in differences.items():
                    if not values:
                        continue
                    mean_difference = statistics.mean(values)
                    favorable = mean_difference < 0 if metric in {"ece", "brier_score", "loss"} else mean_difference > 0
                    signs.append(f"{metric}={'favorable' if favorable else 'unfavorable'} ({mean_difference:+.4f})")
                if signs:
                    direction_lines.append(f"- `{criterion}`: " + "; ".join(signs))
            direction_lines.append("")
    (args.root / "checkpoint_stability_report.md").write_text("\n".join(direction_lines) + "\n", encoding="utf-8")
    (args.root / "missing_checkpoint_report.md").write_text(
        "# Missing Checkpoint Report\n\n" + ("\n".join(f"- {line}" for line in missing) if missing else "No missing checkpoint records among discovered runs.") + "\n",
        encoding="utf-8",
    )
    print(f"[OK] checkpoint sensitivity records={len(metric_rows)} missing={len(missing)}")
    return 0 if metric_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
