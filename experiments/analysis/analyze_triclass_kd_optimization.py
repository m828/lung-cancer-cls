#!/usr/bin/env python3
"""Analyze outputs0538 triclass KD optimization profiles.

This is read-only analysis. It compares optimization profiles against the
TRI-S0 supervised baseline from outputs0536 when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


CLASS_NAMES = ["normal", "benign", "malignant"]
METRICS = [
    "accuracy",
    "macro_auroc",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "normal_recall",
    "benign_recall",
    "malignant_recall",
    "clinical_composite",
    "ece",
    "brier_score",
]
RANK_METRICS = ["balanced_accuracy", "macro_f1", "malignant_recall", "macro_auroc", "clinical_composite"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0538_triclass_kd_optimization"))
    p.add_argument("--source-root", type=Path, default=Path("../outputs0536_triclass_extension"))
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


def seed_from_name(name: str) -> str:
    return name.rsplit("seed", 1)[-1].split("_", 1)[0] if "seed" in name else ""


def auc_binary(labels: list[int], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2.0) * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metrics_from_predictions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows: list[tuple[int, list[float]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        prob_cols = [f"prob_{name}" for name in CLASS_NAMES if f"prob_{name}" in fields]
        if len(prob_cols) != 3:
            prob_cols = [f"prob_{idx}" for idx in range(3) if f"prob_{idx}" in fields]
        if len(prob_cols) != 3 or "label" not in fields:
            return {}
        for row in reader:
            rows.append((int(float(row["label"])), [float(row[col]) for col in prob_cols]))
    if not rows:
        return {}
    labels = [row[0] for row in rows]
    probs = [row[1] for row in rows]
    preds = [max(range(3), key=lambda idx: row[idx]) for row in probs]
    cm = [[0 for _ in range(3)] for _ in range(3)]
    for y, pred in zip(labels, preds):
        cm[y][pred] += 1
    recalls, precisions, f1s = [], [], []
    for cls in range(3):
        tp = cm[cls][cls]
        fn = sum(cm[cls][j] for j in range(3) if j != cls)
        fp = sum(cm[i][cls] for i in range(3) if i != cls)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    aucs = []
    for cls in range(3):
        auc = auc_binary([1 if y == cls else 0 for y in labels], [row[cls] for row in probs])
        if auc is not None:
            aucs.append(auc)
    confidences = [max(row) for row in probs]
    correct = [1 if y == pred else 0 for y, pred in zip(labels, preds)]
    ece = 0.0
    for bin_idx in range(10):
        lo, hi = bin_idx / 10.0, (bin_idx + 1) / 10.0
        ids = [idx for idx, conf in enumerate(confidences) if lo <= conf < hi or (bin_idx == 9 and conf == 1.0)]
        if ids:
            ece += len(ids) / len(labels) * abs(statistics.mean(confidences[i] for i in ids) - statistics.mean(correct[i] for i in ids))
    brier = statistics.mean(sum((row[c] - (1.0 if y == c else 0.0)) ** 2 for c in range(3)) / 3.0 for y, row in zip(labels, probs))
    out = {
        "accuracy": sum(1 for y, pred in zip(labels, preds) if y == pred) / len(labels),
        "macro_auroc": statistics.mean(aucs) if aucs else None,
        "macro_f1": statistics.mean(f1s),
        "balanced_accuracy": statistics.mean(recalls),
        "macro_precision": statistics.mean(precisions),
        "normal_recall": recalls[0],
        "benign_recall": recalls[1],
        "malignant_recall": recalls[2],
        "ece": ece,
        "brier_score": brier,
        "confusion_matrix": cm,
        "num_samples": len(labels),
    }
    out["clinical_composite"] = clinical_composite(out)
    return out


def clinical_composite(row: dict[str, Any]) -> float | None:
    vals = [safe_float(row.get(k)) for k in ["balanced_accuracy", "macro_f1", "malignant_recall", "macro_auroc", "benign_recall"]]
    if any(v is None for v in vals):
        return None
    return 0.25 * vals[0] + 0.25 * vals[1] + 0.20 * vals[2] + 0.20 * vals[3] + 0.10 * vals[4]


def metric_from_json(metrics: dict[str, Any], key: str) -> Any:
    tm = metrics.get("test_metrics") or {}
    if key == "macro_auroc":
        return tm.get("macro_auroc", tm.get("auroc"))
    if key == "macro_f1":
        return tm.get("macro_f1", tm.get("f1"))
    if key == "macro_precision":
        return tm.get("precision_macro")
    return tm.get(key)


def row_from_run(group: str, profile: str, run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    pred_metrics = metrics_from_predictions(run_dir / "test_predictions.csv")
    row = {
        "group": group,
        "profile": profile,
        "seed": seed_from_name(run_dir.name),
        "run_name": run_dir.name,
        "status": "complete" if metrics else "missing_metrics",
        "run_dir": str(run_dir),
        "selection_metric": (metrics or {}).get("selection_metric", ""),
        "num_samples": pred_metrics.get("num_samples") or ((metrics or {}).get("test_metrics") or {}).get("num_samples"),
        "confusion_matrix": pred_metrics.get("confusion_matrix") or ((metrics or {}).get("test_metrics") or {}).get("confusion_matrix"),
    }
    for metric in METRICS:
        value = pred_metrics.get(metric)
        if value is None and metrics:
            value = metric_from_json(metrics, metric)
        row[metric] = safe_float(value)
    if row["clinical_composite"] is None:
        row["clinical_composite"] = clinical_composite(row)
    return row


def collect(root: Path, source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s0_root = source_root / "supervised_ct_text"
    for run_dir in sorted(s0_root.glob("*")):
        if run_dir.is_dir():
            rows.append(row_from_run("TRI-S0", "source_tris0", run_dir))
    for profile_dir in sorted((root / "profiles").glob("*")):
        if not profile_dir.is_dir():
            continue
        for run_dir in sorted(profile_dir.glob("*")):
            if run_dir.is_dir():
                rows.append(row_from_run("TRI-SKD-OPT", profile_dir.name, run_dir))
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
        group_rows = [r for r in rows if r["profile"] == profile and r["status"] == "complete"]
        summary = {"profile": profile, "n": len(group_rows)}
        for metric in METRICS:
            summary[metric] = mean([r.get(metric) for r in group_rows])
        out.append(summary)
    return out


def bootstrap_vs_s0(rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    s0 = [r for r in rows if r["group"] == "TRI-S0" and r["status"] == "complete"]
    s0_mean = {metric: mean([r.get(metric) for r in s0]) for metric in METRICS}
    out = []
    rng = random.Random(42)
    for profile in sorted({r["profile"] for r in rows if r["group"] == "TRI-SKD-OPT"}):
        prof = [r for r in rows if r["profile"] == profile and r["status"] == "complete"]
        if not prof or not s0:
            continue
        for metric in ["balanced_accuracy", "macro_f1", "malignant_recall", "macro_auroc"]:
            vals = [r.get(metric) for r in prof if isinstance(r.get(metric), float)]
            base = s0_mean.get(metric)
            if not vals or base is None:
                continue
            observed = statistics.mean(vals) - base
            boot = []
            for _ in range(max(1, n_boot)):
                boot.append(statistics.mean(vals[rng.randrange(len(vals))] for _ in vals) - base)
            boot.sort()
            lo = boot[int(0.025 * (len(boot) - 1))]
            hi = boot[int(0.975 * (len(boot) - 1))]
            out.append({
                "comparison": f"{profile} vs TRI-S0",
                "profile": profile,
                "metric": metric,
                "delta": observed,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0 <= hi,
                "n_profile_runs": len(vals),
                "n_s0_runs": len(s0),
                "bootstrap_level": "seed_level",
                "n_bootstrap": n_boot,
            })
    return out


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root, source_root)
    if not rows:
        message = (
            "# Triclass KD Optimization Summary\n\n"
            "MISSING: no TRI-S0 source runs or TRI-SKD optimization runs were found. "
            "Run the suite first or check --root/--source-root.\n"
        )
        for name in [
            "triclass_kd_optimization_metrics.md",
            "triclass_kd_optimization_ranking.md",
            "triclass_kd_optimization_summary.md",
            "triclass_kd_optimization_confusion_matrices.md",
        ]:
            (root / name).write_text(message, encoding="utf-8")
        write_csv(root / "triclass_kd_optimization_metrics.csv", [], ["group", "profile", "seed", "status", "run_dir"])
        write_csv(root / "triclass_kd_optimization_per_class_recall.csv", [], ["group", "profile", "seed", "normal_recall", "benign_recall", "malignant_recall", "run_dir"])
        write_csv(root / "triclass_kd_optimization_bootstrap.csv", [], ["comparison", "profile", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0"])
        print(f"[MISSING] no triclass optimization inputs found under {root} or {source_root}")
        return 0
    fields = ["group", "profile", "seed", "run_name", "status", "selection_metric", "num_samples", *METRICS, "run_dir"]
    write_csv(root / "triclass_kd_optimization_metrics.csv", rows, fields)
    metric_md_rows = [[r["group"], r["profile"], r["seed"], r["status"], *(fmt(r.get(m)) for m in METRICS)] for r in rows]
    (root / "triclass_kd_optimization_metrics.md").write_text(
        "# Triclass KD Optimization Metrics\n\n" + md_table(["group", "profile", "seed", "status", *METRICS], metric_md_rows) + "\n",
        encoding="utf-8",
    )
    summary = profile_summary(rows)
    s0_summary = next((s for s in summary if s["profile"] == "source_tris0"), {})
    ranking = [s for s in summary if s["profile"] != "source_tris0"]
    ranking.sort(key=lambda r: tuple(-(r.get(metric) or -999.0) for metric in RANK_METRICS))
    rank_rows = []
    for idx, row in enumerate(ranking, start=1):
        rank_rows.append([
            str(idx), row["profile"], str(row["n"]),
            *(fmt(row.get(metric)) for metric in RANK_METRICS),
            fmt((row.get("balanced_accuracy") or 0.0) - (s0_summary.get("balanced_accuracy") or 0.0)) if s0_summary else "-",
            fmt((row.get("macro_f1") or 0.0) - (s0_summary.get("macro_f1") or 0.0)) if s0_summary else "-",
            fmt((row.get("malignant_recall") or 0.0) - (s0_summary.get("malignant_recall") or 0.0)) if s0_summary else "-",
            fmt((row.get("macro_auroc") or 0.0) - (s0_summary.get("macro_auroc") or 0.0)) if s0_summary else "-",
        ])
    (root / "triclass_kd_optimization_ranking.md").write_text(
        "# Triclass KD Optimization Ranking\n\n"
        + md_table(["rank", "profile", "n", *RANK_METRICS, "delta_BAcc", "delta_macro_F1", "delta_malignant_recall", "delta_macro_AUROC"], rank_rows)
        + "\n\nNegative deltas versus TRI-S0 are negative extension results and should not be framed as support.\n",
        encoding="utf-8",
    )
    write_csv(root / "triclass_kd_optimization_per_class_recall.csv", rows, ["group", "profile", "seed", "normal_recall", "benign_recall", "malignant_recall", "run_dir"])
    cm_lines = ["# Triclass KD Optimization Confusion Matrices\n"]
    for row in rows:
        cm_lines.append(f"## {row['profile']} seed{row['seed']}\n")
        cm_lines.append(f"`{row.get('confusion_matrix')}`\n")
    (root / "triclass_kd_optimization_confusion_matrices.md").write_text("\n".join(cm_lines), encoding="utf-8")
    boot_rows = bootstrap_vs_s0(rows, args.bootstrap_iters)
    write_csv(root / "triclass_kd_optimization_bootstrap.csv", boot_rows, ["comparison", "profile", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_profile_runs", "n_s0_runs", "bootstrap_level", "n_bootstrap"])
    lines = ["# Triclass KD Optimization Summary\n"]
    lines.append("This is an extension validation optimization suite, not the locked binary main result.")
    lines.append("")
    lines.append(md_table(["profile", "n", *RANK_METRICS], [[s["profile"], str(s["n"]), *(fmt(s.get(m)) for m in RANK_METRICS)] for s in summary]))
    lines.append("")
    lines.append("Use `triclass_kd_optimization_ranking.md` for profile ordering and inspect negative deltas before making claims.")
    (root / "triclass_kd_optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote triclass KD optimization analysis under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
