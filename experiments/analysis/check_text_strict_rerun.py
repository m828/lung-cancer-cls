#!/usr/bin/env python3
"""Summary / sanity-check script for the text strict-no-leakage rerun batch.

Reads every run directory under ``--root`` (default
``outputs/text_strict_ref1019_rerun``) and emits two summary artefacts:

    <root>/strict_text_rerun_summary.csv
    <root>/strict_text_rerun_summary.md

For each run we check:
  * metrics.json exists and contains test metrics
  * split_manifest.csv exists with the expected 1019 / 652 / 163 / 204 layout
  * test_predictions.csv exists and has exactly 204 rows
  * text_feature_audit.json + leakage_warnings.json exist (strict-no-leakage)

The script is pure read-only - it never re-trains and can be invoked
independently after the bash batch finishes (or after a partial failure).

Usage:
    python experiments/analysis/check_text_strict_rerun.py \
        --root outputs/text_strict_ref1019_rerun \
        --reference-manifest outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

EXPECTED_TOTAL = 1019
EXPECTED_TRAIN = 652
EXPECTED_VAL = 163
EXPECTED_TEST = 204

METRIC_FIELDS = [
    "auroc",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "ece",
    "brier_score",
]

SUMMARY_FIELDS = [
    "run_name",
    "output_dir",
    "has_metrics",
    "has_split_manifest",
    "split_total",
    "train",
    "val",
    "test",
    "has_test_predictions",
    "test_pred_rows",
    "auroc",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "ece",
    "brier",
    "best_epoch",
    "has_text_feature_audit",
    "has_leakage_warnings",
    "warnings_count",
    "split_check_status",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
        return None


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    except OSError as exc:
        print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
        return -1


def _split_counts(manifest_path: Path) -> tuple[int, int, int, int] | None:
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return None
            split_col = None
            for cand in ("assigned_split", "split", "Split"):
                if cand in reader.fieldnames:
                    split_col = cand
                    break
            if split_col is None:
                return None
            counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
            total = 0
            for row in reader:
                total += 1
                val = (row.get(split_col) or "").strip().lower()
                if val in counts:
                    counts[val] += 1
            return total, counts["train"], counts["val"], counts["test"]
    except OSError as exc:
        print(f"[WARN] failed to read {manifest_path}: {exc}", file=sys.stderr)
        return None


def _metric_value(test_metrics: dict[str, Any], key: str) -> Any:
    if not isinstance(test_metrics, dict):
        return ""
    if key in test_metrics:
        return test_metrics[key]
    upper = key.upper()
    if upper in test_metrics:
        return test_metrics[upper]
    return ""


def _format_metric(v: Any) -> str:
    if v in (None, ""):
        return ""
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return str(v)


def analyse_run(run_dir: Path, expected_counts: tuple[int, int, int, int]) -> dict[str, Any]:
    record: dict[str, Any] = {f: "" for f in SUMMARY_FIELDS}
    record["run_name"] = run_dir.name
    record["output_dir"] = str(run_dir)
    notes: list[str] = []

    metrics_path = run_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    record["has_metrics"] = bool(metrics)
    if metrics:
        test_metrics = metrics.get("test_metrics") or metrics.get("metrics") or {}
        for src_key, dst_key in (
            ("auroc", "auroc"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("f1", "f1"),
            ("recall", "recall"),
            ("specificity", "specificity"),
            ("ece", "ece"),
            ("brier_score", "brier"),
        ):
            record[dst_key] = _format_metric(_metric_value(test_metrics, src_key))
        record["best_epoch"] = metrics.get("best_epoch", "")
    else:
        notes.append("missing metrics.json")

    manifest_path = run_dir / "split_manifest.csv"
    counts = _split_counts(manifest_path)
    record["has_split_manifest"] = manifest_path.is_file()
    if counts is None:
        if manifest_path.is_file():
            notes.append("split_manifest.csv unreadable")
        else:
            notes.append("missing split_manifest.csv")
        split_status = "MISSING"
    else:
        total, tr, va, te = counts
        record["split_total"] = total
        record["train"] = tr
        record["val"] = va
        record["test"] = te
        if (total, tr, va, te) != expected_counts:
            split_status = "SPLIT_MISMATCH"
            notes.append(
                f"split counts {total}/{tr}/{va}/{te} != "
                f"{expected_counts[0]}/{expected_counts[1]}/{expected_counts[2]}/{expected_counts[3]}"
            )
        else:
            split_status = "OK"
    record["split_check_status"] = split_status

    pred_path = run_dir / "test_predictions.csv"
    record["has_test_predictions"] = pred_path.is_file()
    if pred_path.is_file():
        rows = _count_rows(pred_path)
        record["test_pred_rows"] = rows
        if rows != EXPECTED_TEST:
            notes.append(f"TEST_PRED_ROWS_NOT_204 ({rows})")
    else:
        record["test_pred_rows"] = ""
        notes.append("missing test_predictions.csv")

    audit_path = run_dir / "text_feature_audit.json"
    warn_path = run_dir / "leakage_warnings.json"
    record["has_text_feature_audit"] = audit_path.is_file()
    record["has_leakage_warnings"] = warn_path.is_file()
    if not audit_path.is_file() and not warn_path.is_file():
        notes.append("STRICT_AUDIT_MISSING")
    elif not audit_path.is_file():
        notes.append("missing text_feature_audit.json")
    elif not warn_path.is_file():
        notes.append("missing leakage_warnings.json")

    if warn_path.is_file():
        warn_data = _read_json(warn_path) or {}
        warns = warn_data.get("warnings") or []
        record["warnings_count"] = len(warns) if isinstance(warns, list) else ""
    else:
        record["warnings_count"] = ""

    record["notes"] = "; ".join(notes)
    return record


def discover_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"logs", "summaries", "__pycache__"}:
            continue
        runs.append(entry)
    return runs


def write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in SUMMARY_FIELDS})


def _md_cell(v: Any) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if v == "" or v is None:
        return "-"
    return str(v).replace("|", "\\|")


def write_markdown(
    records: list[dict[str, Any]],
    out_path: Path,
    root: Path,
    reference_manifest: Path | None,
    ref_counts: tuple[int, int, int, int] | None,
) -> None:
    lines: list[str] = []
    lines.append("# Text Strict-No-Leakage Rerun Summary\n")
    lines.append(f"- Root: `{root}`")
    if reference_manifest is not None:
        lines.append(f"- Reference manifest: `{reference_manifest}`")
    if ref_counts is not None:
        lines.append(
            f"- Reference counts: total={ref_counts[0]}, "
            f"train={ref_counts[1]}, val={ref_counts[2]}, test={ref_counts[3]}"
        )
    lines.append(
        f"- Expected split: total={EXPECTED_TOTAL}, train={EXPECTED_TRAIN}, "
        f"val={EXPECTED_VAL}, test={EXPECTED_TEST}"
    )
    lines.append(f"- Runs discovered: {len(records)}\n")

    headers = [
        "run",
        "metrics",
        "split(total/tr/va/te)",
        "split_status",
        "test_pred_rows",
        "AUROC",
        "BAcc",
        "F1",
        "ECE",
        "Brier",
        "audit",
        "warnings",
        "warn_n",
        "notes",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for rec in records:
        split_repr = (
            f"{rec.get('split_total', '-')}/"
            f"{rec.get('train', '-')}/"
            f"{rec.get('val', '-')}/"
            f"{rec.get('test', '-')}"
        )
        row = [
            rec.get("run_name", ""),
            _md_cell(rec.get("has_metrics")),
            split_repr,
            rec.get("split_check_status", ""),
            _md_cell(rec.get("test_pred_rows")),
            _md_cell(rec.get("auroc")),
            _md_cell(rec.get("balanced_accuracy")),
            _md_cell(rec.get("f1")),
            _md_cell(rec.get("ece")),
            _md_cell(rec.get("brier")),
            _md_cell(rec.get("has_text_feature_audit")),
            _md_cell(rec.get("has_leakage_warnings")),
            _md_cell(rec.get("warnings_count")),
            _md_cell(rec.get("notes")),
        ]
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")

    lines.append("\n## Status flags\n")
    lines.append("- `SPLIT_MISMATCH` — split_manifest.csv counts differ from the 1019/652/163/204 reference.")
    lines.append("- `TEST_PRED_ROWS_NOT_204` — test_predictions.csv row count differs from 204.")
    lines.append("- `STRICT_AUDIT_MISSING` — neither text_feature_audit.json nor leakage_warnings.json found.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/text_strict_ref1019_rerun"),
        help="Directory containing per-run subdirectories",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"),
        help="Reference 1019 split manifest (informational only)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional explicit CSV output path",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Optional explicit Markdown output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root directory not found: {root}", file=sys.stderr)
        return 1

    ref_path: Path | None = args.reference_manifest.resolve() if args.reference_manifest else None
    ref_counts = _split_counts(ref_path) if ref_path and ref_path.is_file() else None
    if ref_path is not None and not ref_path.is_file():
        print(
            f"[WARN] reference manifest not found at {ref_path}; continuing with expected defaults",
            file=sys.stderr,
        )
    if ref_counts is not None and ref_counts != (
        EXPECTED_TOTAL,
        EXPECTED_TRAIN,
        EXPECTED_VAL,
        EXPECTED_TEST,
    ):
        print(
            f"[WARN] reference manifest counts {ref_counts} differ from expected "
            f"{(EXPECTED_TOTAL, EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST)}; "
            "checks still use the expected layout",
            file=sys.stderr,
        )

    expected = (EXPECTED_TOTAL, EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST)

    runs = discover_runs(root)
    if not runs:
        print(f"[WARN] no run directories found under {root}", file=sys.stderr)

    records = [analyse_run(run, expected) for run in runs]

    csv_path = args.csv_out.resolve() if args.csv_out else root / "strict_text_rerun_summary.csv"
    md_path = args.md_out.resolve() if args.md_out else root / "strict_text_rerun_summary.md"

    write_csv(records, csv_path)
    write_markdown(records, md_path, root, ref_path, ref_counts)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Runs analysed: {len(records)}")
    bad = [r for r in records if r.get("split_check_status") not in ("OK", "")]
    if bad:
        print(f"Runs with split issues: {len(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
