#!/usr/bin/env python3
"""Summarize internal triclass extension runs under outputs0536.

This is read-only analysis: it scans completed runs, computes multiclass
metrics from prediction CSVs when available, and writes summary tables.
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
GROUPS = {
    "TRI-T": "teacher_ct_cnv_text",
    "TRI-S0": "supervised_ct_text",
    "TRI-SKD": "student_kd_ct_text_from_gene_teacher",
}
METRICS = [
    "accuracy",
    "macro_auroc",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "normal_recall",
    "benign_recall",
    "malignant_recall",
    "ece",
    "brier_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0536_triclass_extension"))
    p.add_argument("--bootstrap-iters", type=int, default=2000)
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


def seed_from_name(name: str) -> str:
    if "seed" not in name:
        return ""
    return name.rsplit("seed", 1)[-1].split("_", 1)[0]


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


def confusion(labels: list[int], preds: list[int], k: int = 3) -> list[list[int]]:
    cm = [[0 for _ in range(k)] for _ in range(k)]
    for y, p in zip(labels, preds):
        if 0 <= y < k and 0 <= p < k:
            cm[y][p] += 1
    return cm


def metrics_from_arrays(labels: list[int], probs: list[list[float]]) -> dict[str, Any]:
    if not labels or not probs:
        return {}
    preds = [max(range(len(row)), key=lambda i: row[i]) for row in probs]
    cm = confusion(labels, preds, 3)
    recalls = []
    precisions = []
    f1s = []
    for c in range(3):
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(3) if j != c)
        fp = sum(cm[i][c] for i in range(3) if i != c)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    aucs = []
    for c in range(3):
        auc = auc_binary([1 if y == c else 0 for y in labels], [row[c] for row in probs])
        if auc is not None:
            aucs.append(auc)
    confidences = [max(row) for row in probs]
    correct = [1 if y == p else 0 for y, p in zip(labels, preds)]
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, conf in enumerate(confidences) if (lo <= conf < hi) or (b == 9 and conf == 1.0)]
        if not idx:
            continue
        ece += len(idx) / len(labels) * abs(statistics.mean(confidences[i] for i in idx) - statistics.mean(correct[i] for i in idx))
    brier = statistics.mean(
        sum((row[c] - (1.0 if y == c else 0.0)) ** 2 for c in range(3)) / 3.0
        for y, row in zip(labels, probs)
    )
    return {
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / len(labels),
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


def load_predictions(path: Path) -> dict[str, tuple[int, list[float]]]:
    if not path.is_file():
        return {}
    out: dict[str, tuple[int, list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        id_col = "sample_id" if "sample_id" in fields else "record_id"
        prob_cols = []
        for name in CLASS_NAMES:
            col = f"prob_{name}"
            if col in fields:
                prob_cols.append(col)
        if len(prob_cols) != 3:
            prob_cols = [f"prob_{i}" for i in range(3) if f"prob_{i}" in fields]
        if len(prob_cols) != 3 or "label" not in fields:
            return {}
        for row in reader:
            sid = str(row.get(id_col, "")).strip()
            if not sid:
                continue
            out[sid] = (int(float(row["label"])), [float(row[c]) for c in prob_cols])
    return out


def run_records(root: Path) -> list[dict[str, Any]]:
    rows = []
    for group, subdir in GROUPS.items():
        base = root / subdir
        for run_dir in sorted(base.glob("*")):
            if not run_dir.is_dir():
                continue
            metrics_json = read_json(run_dir / "metrics.json")
            pred = load_predictions(run_dir / "test_predictions.csv")
            computed = metrics_from_arrays([v[0] for _, v in sorted(pred.items())], [v[1] for _, v in sorted(pred.items())]) if pred else {}
            tm = (metrics_json or {}).get("test_metrics") or {}
            row = {
                "group": group,
                "run_name": run_dir.name,
                "seed": seed_from_name(run_dir.name),
                "status": "complete" if metrics_json else "missing_metrics",
                "run_dir": str(run_dir),
                "num_samples": computed.get("num_samples") or tm.get("num_samples"),
                "confusion_matrix": computed.get("confusion_matrix") or tm.get("confusion_matrix"),
            }
            row["accuracy"] = safe_float(computed.get("accuracy", tm.get("accuracy")))
            row["macro_auroc"] = safe_float(computed.get("macro_auroc", tm.get("auroc")))
            row["macro_f1"] = safe_float(computed.get("macro_f1", tm.get("f1")))
            row["balanced_accuracy"] = safe_float(computed.get("balanced_accuracy", tm.get("balanced_accuracy")))
            row["macro_precision"] = safe_float(computed.get("macro_precision", tm.get("precision_macro")))
            row["normal_recall"] = safe_float(computed.get("normal_recall"))
            row["benign_recall"] = safe_float(computed.get("benign_recall"))
            row["malignant_recall"] = safe_float(computed.get("malignant_recall"))
            row["ece"] = safe_float(computed.get("ece", tm.get("ece")))
            row["brier_score"] = safe_float(computed.get("brier_score", tm.get("brier_score")))
            rows.append(row)
    return rows


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if math.isfinite(v) else "-"
    return "-" if v is None else str(v)


def mean_std(values: list[float]) -> str:
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


def bootstrap_rows(root: Path, n_boot: int) -> list[dict[str, Any]]:
    comparisons = [("TRI-SKD", "TRI-S0"), ("TRI-T", "TRI-S0")]
    out = []
    pred_by_group_seed: dict[tuple[str, str], dict[str, tuple[int, list[float]]]] = {}
    for group, subdir in GROUPS.items():
        for run_dir in sorted((root / subdir).glob("*")):
            seed = seed_from_name(run_dir.name)
            pred = load_predictions(run_dir / "test_predictions.csv")
            if pred:
                pred_by_group_seed[(group, seed)] = pred
    rng = random.Random(42)
    for a, b in comparisons:
        seeds = sorted({s for g, s in pred_by_group_seed if g == a} & {s for g, s in pred_by_group_seed if g == b})
        if not seeds:
            continue
        common_ids = set.intersection(*(set(pred_by_group_seed[(a, s)]) & set(pred_by_group_seed[(b, s)]) for s in seeds))
        ids = sorted(common_ids)
        if not ids:
            continue
        def group_metrics(sample_ids: list[str], group: str) -> dict[str, float]:
            per_seed = []
            for seed in seeds:
                pred = pred_by_group_seed[(group, seed)]
                per_seed.append(metrics_from_arrays([pred[i][0] for i in sample_ids], [pred[i][1] for i in sample_ids]))
            return {m: statistics.mean(x[m] for x in per_seed if x.get(m) is not None) for m in ["macro_auroc", "balanced_accuracy", "macro_f1", "malignant_recall"]}
        obs_a, obs_b = group_metrics(ids, a), group_metrics(ids, b)
        samples = {m: [] for m in obs_a}
        for _ in range(max(1, n_boot)):
            sampled = [ids[rng.randrange(len(ids))] for _ in ids]
            ma, mb = group_metrics(sampled, a), group_metrics(sampled, b)
            for m in samples:
                samples[m].append(ma[m] - mb[m])
        for m, vals in samples.items():
            vals.sort()
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            out.append({
                "comparison": f"{a} vs {b}",
                "metric": m,
                "delta": obs_a[m] - obs_b[m],
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0 <= hi,
                "n_samples": len(ids),
                "n_bootstrap": n_boot,
            })
    return out


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = run_records(root)
    fields = ["group", "run_name", "seed", "status", "num_samples", *METRICS, "run_dir"]
    write_csv(root / "triclass_metrics.csv", rows, fields)
    metric_md_rows = [[r["group"], r["seed"], r["status"], *(fmt(r.get(m)) for m in METRICS)] for r in rows]
    (root / "triclass_metrics.md").write_text(
        "# Triclass Metrics\n\n" + md_table(["group", "seed", "status", *METRICS], metric_md_rows) + "\n",
        encoding="utf-8",
    )
    per_class = []
    for r in rows:
        per_class.append({k: r.get(k) for k in ["group", "run_name", "seed", "normal_recall", "benign_recall", "malignant_recall"]})
    write_csv(root / "triclass_per_class_recall.csv", per_class, ["group", "run_name", "seed", "normal_recall", "benign_recall", "malignant_recall"])
    cm_lines = ["# Triclass Confusion Matrices\n"]
    for r in rows:
        cm_lines.append(f"## {r['group']} seed{r['seed']}")
        cm_lines.append("```json")
        cm_lines.append(json.dumps(r.get("confusion_matrix") or [], ensure_ascii=False))
        cm_lines.append("```\n")
    (root / "triclass_confusion_matrices.md").write_text("\n".join(cm_lines), encoding="utf-8")
    boot = bootstrap_rows(root, args.bootstrap_iters)
    write_csv(root / "triclass_bootstrap.csv", boot, ["comparison", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_samples", "n_bootstrap"])
    summary_rows = []
    for group in GROUPS:
        gr = [r for r in rows if r["group"] == group and r["status"] == "complete"]
        summary_rows.append([group, str(len(gr)), *(mean_std([r.get(m) for r in gr]) for m in METRICS)])
    lines = ["# Triclass Extension Summary\n"]
    lines.append(md_table(["group", "n", *METRICS], summary_rows))
    lines.append("\n## Core Questions")
    lines.append("- Teacher 是否保留 malignant recall: see `TRI-T` malignant_recall.")
    lines.append("- CT+Text supervised 是否出现 malignant->benign 类别塌陷: inspect `TRI-S0` confusion matrix row for malignant.")
    lines.append("- 当前选择性 KD 是否改善 malignant recall: compare `TRI-SKD` vs `TRI-S0` malignant_recall and bootstrap rows.")
    lines.append("- KD 是否提高 macro-F1 / BAcc: compare macro_f1 and balanced_accuracy.")
    lines.append("- CNV/基因信息是否帮助良恶性边界: compare `TRI-T` and `TRI-SKD` against `TRI-S0` on malignant recall and macro AUROC.")
    lines.append("- 三分类定位: extension / failure-mode analysis, not the binary main result.")
    (root / "triclass_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote triclass analysis under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
