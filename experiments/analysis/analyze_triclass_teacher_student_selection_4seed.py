#!/usr/bin/env python3
"""Analyze triclass teacher/student checkpoint-selection sensitivity runs.

Read-only analysis for outputs0541.  It expects teacher metrics written by
train_multimodal.py with --extra-checkpoint-metrics and student cached-KD runs
from scripts/run_triclass_teacher_student_selection_4seed.sh.
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
    "normal_recall",
    "benign_recall",
    "malignant_recall",
    "clinical_composite",
    "ece",
    "brier_score",
]
BOOT_METRICS = ["macro_f1", "balanced_accuracy", "malignant_recall", "ece", "brier_score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0541_triclass_teacher_student_selection_4seed"))
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
    if "seed" not in name:
        return ""
    return name.rsplit("seed", 1)[-1].split("_", 1)[0]


def clinical_composite(metrics: dict[str, Any]) -> float | None:
    values = {
        "balanced_accuracy": scalar(metrics, "balanced_accuracy"),
        "macro_f1": scalar(metrics, "macro_f1"),
        "malignant_recall": scalar(metrics, "malignant_recall"),
        "macro_auroc": scalar(metrics, "macro_auroc"),
        "benign_recall": scalar(metrics, "benign_recall"),
    }
    if any(v is None for v in values.values()):
        return None
    return (
        0.25 * values["balanced_accuracy"]
        + 0.25 * values["macro_f1"]
        + 0.20 * values["malignant_recall"]
        + 0.20 * values["macro_auroc"]
        + 0.10 * values["benign_recall"]
    )


def scalar(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    if key == "macro_auroc":
        return safe_float(metrics.get("macro_auroc", metrics.get("auroc")))
    if key == "macro_f1":
        return safe_float(metrics.get("macro_f1", metrics.get("f1")))
    if key == "clinical_composite":
        return safe_float(metrics.get("clinical_composite")) or clinical_composite(metrics)
    return safe_float(metrics.get(key))


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
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metrics_from_arrays(labels: list[int], probs: list[list[float]]) -> dict[str, Any]:
    if not labels or not probs:
        return {}
    preds = [max(range(len(row)), key=lambda i: row[i]) for row in probs]
    cm = [[0 for _ in range(3)] for _ in range(3)]
    for y, pred in zip(labels, preds):
        if 0 <= y < 3 and 0 <= pred < 3:
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
    aucs = [
        auc
        for cls in range(3)
        if (auc := auc_binary([1 if y == cls else 0 for y in labels], [row[cls] for row in probs])) is not None
    ]
    confidences = [max(row) for row in probs]
    correct = [1 if y == pred else 0 for y, pred in zip(labels, preds)]
    ece = 0.0
    for bin_idx in range(10):
        lo, hi = bin_idx / 10.0, (bin_idx + 1) / 10.0
        idx = [i for i, conf in enumerate(confidences) if (lo <= conf < hi) or (bin_idx == 9 and conf == 1.0)]
        if idx:
            ece += len(idx) / len(labels) * abs(
                statistics.mean(confidences[i] for i in idx) - statistics.mean(correct[i] for i in idx)
            )
    brier = statistics.mean(
        sum((row[cls] - (1.0 if y == cls else 0.0)) ** 2 for cls in range(3)) / 3.0
        for y, row in zip(labels, probs)
    )
    out = {
        "accuracy": sum(1 for y, pred in zip(labels, preds) if y == pred) / len(labels),
        "macro_auroc": statistics.mean(aucs) if aucs else None,
        "macro_f1": statistics.mean(f1s),
        "balanced_accuracy": statistics.mean(recalls),
        "normal_recall": recalls[0],
        "benign_recall": recalls[1],
        "malignant_recall": recalls[2],
        "ece": ece,
        "brier_score": brier,
        "num_samples": len(labels),
        "confusion_matrix": cm,
    }
    out["clinical_composite"] = clinical_composite(out)
    return out


def load_predictions(path: Path) -> dict[str, tuple[int, list[float]]]:
    if not path.is_file():
        return {}
    out: dict[str, tuple[int, list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        id_col = "sample_id" if "sample_id" in fields else "record_id"
        prob_cols = [f"prob_{name}" for name in CLASS_NAMES if f"prob_{name}" in fields]
        if len(prob_cols) != 3:
            prob_cols = [f"prob_{idx}" for idx in range(3) if f"prob_{idx}" in fields]
        if len(prob_cols) != 3 or "label" not in fields:
            return {}
        for row in reader:
            sid = str(row.get(id_col, "")).strip()
            if sid:
                out[sid] = (int(float(row["label"])), [float(row[col]) for col in prob_cols])
    return out


def row_from_metrics(prefix: dict[str, Any], metrics: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(prefix)
    for metric in METRICS:
        row[metric] = scalar(metrics, metric)
    row["num_samples"] = scalar(metrics, "num_samples")
    return row


def test_metrics_for_run(run_dir: Path, metrics_json: dict[str, Any] | None) -> dict[str, Any]:
    metrics = dict((metrics_json or {}).get("test_metrics") or {})
    pred = load_predictions(run_dir / "test_predictions.csv")
    if pred:
        computed = metrics_from_arrays(
            [value[0] for _, value in sorted(pred.items())],
            [value[1] for _, value in sorted(pred.items())],
        )
        for key, value in computed.items():
            if value is not None:
                metrics[key] = value
    return metrics


def teacher_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((root / "teacher_ct_cnv_text").glob("*")):
        data = read_json(run_dir / "metrics.json")
        if not data:
            continue
        seed = str(data.get("config", {}).get("seed") or seed_from_name(run_dir.name))
        for rec in data.get("multi_metric_checkpoints") or []:
            val = rec.get("val_metrics") or rec.get("val_metrics_at_selection") or {}
            test = rec.get("test_metrics") or {}
            row: dict[str, Any] = {
                "status": "complete",
                "seed": seed,
                "selection_metric": rec.get("selection_metric", ""),
                "best_epoch": rec.get("best_epoch", ""),
                "checkpoint_path": rec.get("checkpoint_path", ""),
                "run_dir": str(run_dir),
            }
            for metric in METRICS:
                row[f"val_{metric}"] = scalar(val, metric)
                row[f"test_{metric}"] = scalar(test, metric)
            rows.append(row)
    if not rows:
        rows.append({"status": "MISSING", "seed": "", "selection_metric": "", "best_epoch": "", "checkpoint_path": "", "run_dir": ""})
    return rows


def student_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        ("seed42_sensitivity", root / "seed42_teacher_selection_sensitivity"),
        ("four_seed", root / "student_4seed"),
    ]
    for mode, base in roots:
        for metric_dir in sorted(base.glob("teacher_select_*")):
            teacher_metric = metric_dir.name.replace("teacher_select_", "")
            for run_dir in sorted(metric_dir.glob("*")):
                data = read_json(run_dir / "metrics.json")
                if not data:
                    continue
                seed = str(data.get("config", {}).get("seed") or seed_from_name(run_dir.name))
                row = row_from_metrics(
                    {
                        "status": "complete",
                        "mode": mode,
                        "group": "TRI-SKD",
                        "teacher_selection_metric": teacher_metric,
                        "student_selection_metric": data.get("selection_metric", ""),
                        "seed": seed,
                        "best_epoch": data.get("best_epoch", ""),
                        "run_dir": str(run_dir),
                    },
                    test_metrics_for_run(run_dir, data),
                )
                rows.append(row)
    if not rows:
        rows.append({"status": "MISSING", "mode": "", "group": "TRI-SKD", "teacher_selection_metric": "", "student_selection_metric": "", "seed": "", "best_epoch": "", "run_dir": ""})
    return rows


def baseline_rows(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((source_root / "supervised_ct_text").glob("*")):
        data = read_json(run_dir / "metrics.json")
        if not data:
            continue
        seed = str(data.get("config", {}).get("seed") or seed_from_name(run_dir.name))
        rows.append(row_from_metrics(
            {"status": "complete", "mode": "baseline", "group": "TRI-S0", "teacher_selection_metric": "", "student_selection_metric": data.get("selection_metric", ""), "seed": seed, "best_epoch": data.get("best_epoch", ""), "run_dir": str(run_dir)},
            test_metrics_for_run(run_dir, data),
        ))
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "-"
    if value is None:
        return "-"
    return str(value)


def mean_std(values: list[Any]) -> str:
    vals = [float(v) for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}+/-0.0000"
    return f"{statistics.mean(vals):.4f}+/-{statistics.stdev(vals):.4f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_teacher_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["status", "seed", "selection_metric", "best_epoch"]
    for split in ["val", "test"]:
        fields.extend(f"{split}_{metric}" for metric in METRICS)
    fields.extend(["checkpoint_path", "run_dir"])
    write_csv(root / "triclass_teacher_selection_summary.csv", rows, fields)
    write_csv(root / "teacher_checkpoint_selection_summary.csv", rows, fields)
    if rows and rows[0].get("status") == "MISSING":
        text = "# Triclass Teacher Selection Summary\n\nMISSING: no teacher multi-metric checkpoint records found.\n"
    else:
        headers = ["seed", "metric", "epoch", "val macro-F1", "val BAcc", "val mal-rec", "test macro-F1", "test BAcc", "test mal-rec", "ckpt"]
        body = [[
            str(r.get("seed", "")),
            str(r.get("selection_metric", "")),
            str(r.get("best_epoch", "")),
            fmt(r.get("val_macro_f1")),
            fmt(r.get("val_balanced_accuracy")),
            fmt(r.get("val_malignant_recall")),
            fmt(r.get("test_macro_f1")),
            fmt(r.get("test_balanced_accuracy")),
            fmt(r.get("test_malignant_recall")),
            str(r.get("checkpoint_path", "")),
        ] for r in rows]
        text = "# Triclass Teacher Selection Summary\n\n" + md_table(headers, body) + "\n"
    (root / "triclass_teacher_selection_summary.md").write_text(text, encoding="utf-8")
    (root / "teacher_checkpoint_selection_summary.md").write_text(text, encoding="utf-8")


def write_student_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["status", "mode", "group", "teacher_selection_metric", "student_selection_metric", "seed", "best_epoch", *METRICS, "num_samples", "run_dir"]
    write_csv(root / "triclass_student_4seed_metrics.csv", rows, fields)
    if rows and rows[0].get("status") == "MISSING":
        text = "# Triclass Student 4-Seed Metrics\n\nMISSING: no student metrics found.\n"
    else:
        headers = ["mode", "teacher metric", "seed", "Acc", "macro-AUROC", "macro-F1", "BAcc", "mal-rec", "ECE", "Brier"]
        body = [[
            str(r.get("mode", "")),
            str(r.get("teacher_selection_metric", "")),
            str(r.get("seed", "")),
            fmt(r.get("accuracy")),
            fmt(r.get("macro_auroc")),
            fmt(r.get("macro_f1")),
            fmt(r.get("balanced_accuracy")),
            fmt(r.get("malignant_recall")),
            fmt(r.get("ece")),
            fmt(r.get("brier_score")),
        ] for r in rows if r.get("status") != "MISSING"]
        text = "# Triclass Student 4-Seed Metrics\n\n" + md_table(headers, body) + "\n"
    (root / "triclass_student_4seed_metrics.md").write_text(text, encoding="utf-8")


def grouped_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = tuple(str(row.get(k, "")) for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(vals)
        for metric in METRICS:
            rec[metric] = mean_std([v.get(metric) for v in vals])
        out.append(rec)
    return out


def bootstrap(root: Path, source_root: Path, rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    baseline_preds = {
        seed_from_name(path.parent.name): load_predictions(path)
        for path in (source_root / "supervised_ct_text").glob("*/test_predictions.csv")
    }
    student_pred_by_key: dict[tuple[str, str], dict[str, tuple[int, list[float]]]] = {}
    for row in rows:
        if row.get("status") != "complete" or row.get("mode") != "four_seed":
            continue
        run_dir = Path(str(row.get("run_dir", "")))
        pred = load_predictions(run_dir / "test_predictions.csv")
        if pred:
            student_pred_by_key[(str(row.get("teacher_selection_metric")), str(row.get("seed")))] = pred
    rng = random.Random(42)
    for teacher_metric in sorted({k[0] for k in student_pred_by_key}):
        seeds = sorted({seed for metric, seed in student_pred_by_key if metric == teacher_metric} & set(baseline_preds))
        if not seeds:
            continue
        common_ids = set.intersection(*(set(student_pred_by_key[(teacher_metric, seed)]) & set(baseline_preds[seed]) for seed in seeds))
        ids = sorted(common_ids)
        if not ids:
            continue

        def group_metrics(sample_ids: list[str], side: str) -> dict[str, float]:
            per_seed = []
            for seed in seeds:
                pred = student_pred_by_key[(teacher_metric, seed)] if side == "student" else baseline_preds[seed]
                labels = [pred[sid][0] for sid in sample_ids]
                probs = [pred[sid][1] for sid in sample_ids]
                per_seed.append(metrics_from_arrays(labels, probs))
            return {metric: statistics.mean(float(m[metric]) for m in per_seed if m.get(metric) is not None) for metric in BOOT_METRICS}

        observed_student = group_metrics(ids, "student")
        observed_base = group_metrics(ids, "baseline")
        deltas = {metric: [] for metric in BOOT_METRICS}
        for _ in range(max(1, n_boot)):
            sample_ids = [ids[rng.randrange(len(ids))] for _ in ids]
            ms = group_metrics(sample_ids, "student")
            mb = group_metrics(sample_ids, "baseline")
            for metric in BOOT_METRICS:
                deltas[metric].append(ms[metric] - mb[metric])
        for metric in BOOT_METRICS:
            vals = sorted(deltas[metric])
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            delta = observed_student[metric] - observed_base[metric]
            out.append({
                "comparison": f"TRI-SKD teacher_select_{teacher_metric} vs TRI-S0",
                "teacher_selection_metric": teacher_metric,
                "metric": metric,
                "delta": delta,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0.0 <= hi,
                "n_seeds": len(seeds),
                "n_samples": len(ids),
                "n_bootstrap": n_boot,
            })
    if not out:
        out.append({"comparison": "MISSING", "teacher_selection_metric": "", "metric": "", "delta": "", "ci95_low": "", "ci95_high": "", "ci_crosses_0": "", "n_seeds": "", "n_samples": "", "n_bootstrap": n_boot})
    return out


def write_comparison(root: Path, teacher_rows_: list[dict[str, Any]], student_rows_: list[dict[str, Any]], baseline_rows_: list[dict[str, Any]], boot_rows: list[dict[str, Any]]) -> None:
    lines = ["# Triclass Teacher/Student Comparison\n"]
    if teacher_rows_ and teacher_rows_[0].get("status") == "MISSING":
        lines.append("MISSING: teacher multi-metric checkpoint records are not available.")
        lines.append("")
    if student_rows_ and student_rows_[0].get("status") == "MISSING":
        lines.append("MISSING: student KD runs are not available.")
        lines.append("")

    lines.append("## Student Summary")
    summary = grouped_summary(student_rows_, ["mode", "teacher_selection_metric", "student_selection_metric"])
    if summary:
        headers = ["mode", "teacher metric", "student metric", "n", "Acc", "macro-AUROC", "macro-F1", "BAcc", "mal-rec", "ECE", "Brier"]
        lines.append(md_table(headers, [[
            r["mode"], r["teacher_selection_metric"], r["student_selection_metric"], str(r["n"]),
            r["accuracy"], r["macro_auroc"], r["macro_f1"], r["balanced_accuracy"], r["malignant_recall"], r["ece"], r["brier_score"],
        ] for r in summary]))
    else:
        lines.append("MISSING: no complete student summary rows.")
    lines.append("")

    lines.append("## TRI-S0 Baseline")
    base_summary = grouped_summary(baseline_rows_, ["group"])
    if base_summary:
        lines.append(md_table(["group", "n", "Acc", "macro-AUROC", "macro-F1", "BAcc", "mal-rec", "ECE", "Brier"], [[
            r["group"], str(r["n"]), r["accuracy"], r["macro_auroc"], r["macro_f1"], r["balanced_accuracy"], r["malignant_recall"], r["ece"], r["brier_score"],
        ] for r in base_summary]))
    else:
        lines.append("MISSING: no TRI-S0 baseline rows found in source root.")
    lines.append("")

    lines.append("## Bootstrap")
    if boot_rows and boot_rows[0].get("comparison") != "MISSING":
        lines.append(md_table(["comparison", "metric", "delta", "95% CI", "crosses 0", "n seeds", "n samples"], [[
            str(r["comparison"]), str(r["metric"]), fmt(r["delta"]), f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]", str(r["ci_crosses_0"]), str(r["n_seeds"]), str(r["n_samples"]),
        ] for r in boot_rows]))
    else:
        lines.append("MISSING: bootstrap requires matched four-seed student and TRI-S0 prediction CSVs.")
    (root / "triclass_teacher_student_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paper_lines = ["# Triclass 4-Seed Paper Table\n"]
    if base_summary:
        paper_lines.append("## Baseline")
        paper_lines.append(md_table(["group", "n", "macro-F1", "BAcc", "mal-rec", "ECE", "Brier"], [[
            r["group"], str(r["n"]), r["macro_f1"], r["balanced_accuracy"], r["malignant_recall"], r["ece"], r["brier_score"],
        ] for r in base_summary]))
    four = [r for r in summary if r.get("mode") == "four_seed"]
    if four:
        paper_lines.append("\n## TRI-SKD")
        paper_lines.append(md_table(["teacher metric", "student metric", "n", "macro-F1", "BAcc", "mal-rec", "ECE", "Brier"], [[
            r["teacher_selection_metric"], r["student_selection_metric"], str(r["n"]), r["macro_f1"], r["balanced_accuracy"], r["malignant_recall"], r["ece"], r["brier_score"],
        ] for r in four]))
    else:
        paper_lines.append("\nMISSING: no four-seed student rows; any seed42 sensitivity remains preliminary.")
    (root / "triclass_4seed_paper_table.md").write_text("\n".join(paper_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    t_rows = teacher_rows(root)
    s_rows = student_rows(root)
    b_rows = baseline_rows(source_root)
    boot_rows = bootstrap(root, source_root, s_rows, args.bootstrap_iters)

    write_teacher_outputs(root, t_rows)
    write_student_outputs(root, s_rows)
    write_csv(root / "triclass_4seed_bootstrap.csv", boot_rows, ["comparison", "teacher_selection_metric", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_seeds", "n_samples", "n_bootstrap"])
    write_comparison(root, t_rows, s_rows, b_rows, boot_rows)
    print(f"[OK] wrote triclass teacher/student selection analysis under {root}")
    if (t_rows and t_rows[0].get("status") == "MISSING") or (s_rows and s_rows[0].get("status") == "MISSING"):
        print("[WARN] Some outputs are MISSING because the corresponding runs have not completed yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
