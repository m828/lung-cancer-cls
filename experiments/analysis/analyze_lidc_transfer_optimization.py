#!/usr/bin/env python3
"""Analyze outputs0539 LIDC transfer optimization profiles."""

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
RANK_METRICS = ["balanced_accuracy", "f1", "auroc", "recall"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0539_lidc_transfer_optimization"))
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fold_from_name(name: str) -> str:
    return name.rsplit("fold", 1)[-1].split("_", 1)[0] if "fold" in name else ""


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for profile_dir in sorted((root / "profiles").glob("*")):
        if not profile_dir.is_dir():
            continue
        for run_dir in sorted(profile_dir.glob("fold*")):
            if not run_dir.is_dir():
                continue
            metrics = read_json(run_dir / "metrics.json")
            tm = (metrics or {}).get("test_metrics") or {}
            row = {
                "profile": profile_dir.name,
                "fold": fold_from_name(run_dir.name),
                "run_name": run_dir.name,
                "status": "complete" if metrics else "missing_metrics",
                "selection_metric": (metrics or {}).get("selection_metric", ""),
                "transfer_mode": ((metrics or {}).get("config") or {}).get("transfer_mode", ""),
                "init_checkpoint": ((metrics or {}).get("config") or {}).get("init_checkpoint", ""),
                "run_dir": str(run_dir),
                "audit_json": str(run_dir / "lidc_kdinit_loading_audit.json") if (run_dir / "lidc_kdinit_loading_audit.json").exists() else "",
            }
            for metric in METRICS:
                row[metric] = safe_float(tm.get(metric))
            rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "-"
    return "-" if value is None else str(value)


def mean(values: list[Any]) -> float | None:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def mean_std(values: list[Any]) -> str:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}±0.0000"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def profile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for profile in sorted({r["profile"] for r in rows if r["status"] == "complete"}):
        prof_rows = [r for r in rows if r["profile"] == profile and r["status"] == "complete"]
        row = {"profile": profile, "n_folds": len(prof_rows)}
        for metric in METRICS:
            row[metric] = mean([r.get(metric) for r in prof_rows])
        out.append(row)
    return out


def foldwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_profile_fold = {(r["profile"], r["fold"]): r for r in rows if r["status"] == "complete"}
    baseline_folds = {fold for profile, fold in by_profile_fold if profile == "baseline_default"}
    out = []
    for profile in sorted({profile for profile, _ in by_profile_fold if profile != "baseline_default"}):
        folds = sorted(baseline_folds & {fold for p, fold in by_profile_fold if p == profile})
        for fold in folds:
            base = by_profile_fold[("baseline_default", fold)]
            prof = by_profile_fold[(profile, fold)]
            row = {"profile": profile, "fold": fold}
            for metric in METRICS:
                b = base.get(metric)
                v = prof.get(metric)
                row[f"baseline_{metric}"] = b
                row[f"profile_{metric}"] = v
                row[f"delta_{metric}"] = None if b is None or v is None else v - b
            out.append(row)
    return out


def bootstrap(fold_rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    out = []
    rng = random.Random(42)
    for profile in sorted({r["profile"] for r in fold_rows}):
        prof_rows = [r for r in fold_rows if r["profile"] == profile]
        for metric in METRICS:
            deltas = [r.get(f"delta_{metric}") for r in prof_rows if isinstance(r.get(f"delta_{metric}"), float)]
            if not deltas:
                continue
            observed = statistics.mean(deltas)
            samples = []
            for _ in range(max(1, n_boot)):
                samples.append(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas))
            samples.sort()
            lo = samples[int(0.025 * (len(samples) - 1))]
            hi = samples[int(0.975 * (len(samples) - 1))]
            out.append({
                "comparison": f"{profile} vs baseline_default",
                "profile": profile,
                "metric": metric,
                "delta": observed,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0 <= hi,
                "n_folds": len(deltas),
                "n_bootstrap": n_boot,
                "bootstrap_level": "paired_fold",
            })
    return out


def win_counts(fold_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in sorted({r["profile"] for r in fold_rows}):
        score = 0
        for row in [r for r in fold_rows if r["profile"] == profile]:
            for metric in RANK_METRICS:
                delta = row.get(f"delta_{metric}")
                if isinstance(delta, float) and delta > 0:
                    score += 1
            ece_delta = row.get("delta_ece")
            if isinstance(ece_delta, float) and ece_delta < 0:
                score += 1
        counts[profile] = score
    return counts


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    if not rows:
        message = (
            "# LIDC Transfer Optimization Summary\n\n"
            "MISSING: no profile/fold metrics were found. Run the suite first or check --root.\n"
        )
        for name in [
            "lidc_transfer_optimization_metrics.md",
            "lidc_transfer_optimization_ranking.md",
            "lidc_transfer_optimization_summary.md",
        ]:
            (root / name).write_text(message, encoding="utf-8")
        write_csv(root / "lidc_transfer_optimization_metrics.csv", [], ["profile", "fold", "status", "run_dir"])
        write_csv(root / "lidc_transfer_optimization_foldwise.csv", [], ["profile", "fold"])
        write_csv(root / "lidc_transfer_optimization_bootstrap.csv", [], ["comparison", "profile", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0"])
        print(f"[MISSING] no LIDC optimization inputs found under {root}")
        return 0
    fields = ["profile", "fold", "run_name", "status", "selection_metric", "transfer_mode", *METRICS, "audit_json", "run_dir"]
    write_csv(root / "lidc_transfer_optimization_metrics.csv", rows, fields)
    (root / "lidc_transfer_optimization_metrics.md").write_text(
        "# LIDC Transfer Optimization Metrics\n\n"
        + md_table(["profile", "fold", "status", *METRICS], [[r["profile"], r["fold"], r["status"], *(fmt(r.get(m)) for m in METRICS)] for r in rows])
        + "\n",
        encoding="utf-8",
    )
    summary = profile_summary(rows)
    ranking = sorted(summary, key=lambda r: tuple(-(r.get(metric) or -999.0) for metric in RANK_METRICS))
    fold_rows = foldwise(rows)
    wins = win_counts(fold_rows)
    rank_rows = [
        [str(i), r["profile"], str(r["n_folds"]), *(fmt(r.get(m)) for m in RANK_METRICS), str(wins.get(r["profile"], 0))]
        for i, r in enumerate(ranking, start=1)
    ]
    (root / "lidc_transfer_optimization_ranking.md").write_text(
        "# LIDC Transfer Optimization Ranking\n\n"
        + md_table(["rank", "profile", "n_folds", *RANK_METRICS, "fold_metric_win_count"], rank_rows)
        + "\n",
        encoding="utf-8",
    )
    fold_fields = ["profile", "fold", *[f"baseline_{m}" for m in METRICS], *[f"profile_{m}" for m in METRICS], *[f"delta_{m}" for m in METRICS]]
    write_csv(root / "lidc_transfer_optimization_foldwise.csv", fold_rows, fold_fields)
    boot_rows = bootstrap(fold_rows, args.bootstrap_iters)
    write_csv(root / "lidc_transfer_optimization_bootstrap.csv", boot_rows, ["comparison", "profile", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_folds", "n_bootstrap", "bootstrap_level"])
    lines = ["# LIDC Transfer Optimization Summary\n"]
    n_folds = max([int(r["fold"]) for r in rows if str(r.get("fold", "")).isdigit()] + [-1]) + 1
    if n_folds <= 1:
        lines.append("single-fold smoke result, not final evidence")
        lines.append("")
    lines.append("This is public-data transfer validation, not strict same-task external validation.")
    lines.append("")
    lines.append(md_table(["profile", "n_folds", *RANK_METRICS, "ece"], [[r["profile"], str(r["n_folds"]), *(fmt(r.get(m)) for m in RANK_METRICS), fmt(r.get("ece"))] for r in summary]))
    lines.append("")
    lines.append("Use fold-wise deltas and paired-fold bootstrap before deciding whether KDInit supports transferable CT representations.")
    (root / "lidc_transfer_optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote LIDC transfer optimization analysis under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
