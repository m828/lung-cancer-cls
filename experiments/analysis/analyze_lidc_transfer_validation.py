#!/usr/bin/env python3
"""Summarize LIDC transfer validation runs under outputs0537."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


METRICS = ["accuracy", "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score"]
GROUP_DIRS = {"LIDC-B": "baseline", "LIDC-KDInit": "kd_init"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0537_lidc_transfer_validation"))
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fold_from_name(name: str) -> str:
    if "fold" not in name:
        return ""
    return name.rsplit("fold", 1)[-1].split("_", 1)[0]


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for group, subdir in GROUP_DIRS.items():
        for run_dir in sorted((root / subdir).glob("*")):
            if not run_dir.is_dir():
                continue
            metrics = read_json(run_dir / "metrics.json")
            tm = (metrics or {}).get("test_metrics") or {}
            row = {
                "group": group,
                "fold": fold_from_name(run_dir.name),
                "run_name": run_dir.name,
                "status": "complete" if metrics else "missing_metrics",
                "run_dir": str(run_dir),
            }
            for metric in METRICS:
                row[metric] = safe_float(tm.get(metric))
            rows.append(row)
    return rows


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if math.isfinite(v) else "-"
    return "-" if v is None else str(v)


def mean_std(values: list[Any]) -> str:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}±0.0000"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def paired_fold_rows(rows: list[dict[str, Any]], n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group_fold = {(r["group"], r["fold"]): r for r in rows if r["status"] == "complete"}
    folds = sorted({fold for group, fold in by_group_fold if group == "LIDC-B"} & {fold for group, fold in by_group_fold if group == "LIDC-KDInit"})
    comparison = []
    boot_rows = []
    for fold in folds:
        base = by_group_fold[("LIDC-B", fold)]
        kd = by_group_fold[("LIDC-KDInit", fold)]
        row = {"fold": fold}
        for metric in METRICS:
            b = base.get(metric)
            k = kd.get(metric)
            row[f"baseline_{metric}"] = b
            row[f"kd_init_{metric}"] = k
            row[f"delta_{metric}"] = None if b is None or k is None else k - b
        comparison.append(row)
    if comparison:
        rng = random.Random(42)
        for metric in METRICS:
            deltas = [r[f"delta_{metric}"] for r in comparison if isinstance(r.get(f"delta_{metric}"), float)]
            if not deltas:
                continue
            observed = statistics.mean(deltas)
            boot = []
            for _ in range(max(1, n_boot)):
                boot.append(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas))
            boot.sort()
            lo = boot[int(0.025 * (len(boot) - 1))]
            hi = boot[int(0.975 * (len(boot) - 1))]
            boot_rows.append({
                "comparison": "LIDC-KDInit vs LIDC-B",
                "metric": metric,
                "delta": observed,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0 <= hi,
                "n_folds": len(deltas),
                "n_bootstrap": n_boot,
                "bootstrap_level": "paired_fold",
            })
    return comparison, boot_rows


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    fields = ["group", "fold", "run_name", "status", *METRICS, "run_dir"]
    write_csv(root / "lidc_metrics.csv", rows, fields)
    (root / "lidc_metrics.md").write_text(
        "# LIDC Metrics\n\n"
        + md_table(["group", "fold", "status", *METRICS], [[r["group"], r["fold"], r["status"], *(fmt(r.get(m)) for m in METRICS)] for r in rows])
        + "\n",
        encoding="utf-8",
    )
    fold_rows, boot_rows = paired_fold_rows(rows, args.bootstrap_iters)
    comp_fields = ["fold", *[f"baseline_{m}" for m in METRICS], *[f"kd_init_{m}" for m in METRICS], *[f"delta_{m}" for m in METRICS]]
    write_csv(root / "lidc_foldwise_comparison.csv", fold_rows, comp_fields)
    write_csv(root / "lidc_bootstrap.csv", boot_rows, ["comparison", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_folds", "n_bootstrap", "bootstrap_level"])
    summary_rows = []
    for group in GROUP_DIRS:
        gr = [r for r in rows if r["group"] == group and r["status"] == "complete"]
        summary_rows.append([group, str(len(gr)), *(mean_std([r.get(m) for r in gr]) for m in METRICS)])
    lines = ["# LIDC Transfer Validation Summary\n"]
    lines.append(md_table(["group", "n_folds", *METRICS], summary_rows))
    lines.append("\n## Interpretation Questions")
    lines.append("- R3 CT encoder 初始化是否优于 LIDC baseline: use `lidc_foldwise_comparison.csv` and `lidc_bootstrap.csv`.")
    lines.append("- 提升体现在哪些指标: inspect deltas for AUROC, BAcc, F1 and Recall.")
    lines.append("- 是否说明更可迁移 CT 表征: only if KDInit improves held-out folds without selecting by LIDC test.")
    lines.append("- 定位: public transfer validation, not strict same-task external validation.")
    lines.append("\nNote: bootstrap is paired-fold bootstrap because `train.py` records fold metrics but does not export LIDC sample-level prediction CSVs.")
    (root / "lidc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote LIDC analysis under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
