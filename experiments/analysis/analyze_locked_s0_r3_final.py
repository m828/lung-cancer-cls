#!/usr/bin/env python3
"""Lock S0_homogeneous and compare R3 refinements against it.

Read-only analysis.  This script never starts training and never edits original
training outputs.  It writes locked-baseline and comparison reports under the
outputs0535 refinement root.
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


METRICS = [
    "auroc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "ece",
    "brier_score",
    "nll",
    "composite",
]
BOOT_METRICS = ["auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score"]
SEED_LABELS = ["42", "43", "44", "45"]
LOCKED_S0_LABELS = {"42": "42", "43": "43", "44": "44_repeat", "45": "45"}
BOOT_SEED = 42

ID_COLUMNS = ("sample_id", "record_id")
LABEL_COLUMNS = ("label", "y_true", "target", "gt", "label_int")
PROB_COLUMNS = (
    "prob_malignant",
    "prob_1",
    "prob_pos",
    "y_prob",
    "score",
    "probability_malignant",
    "p",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0535_student_kd_refinement"))
    p.add_argument("--homogeneous-root", type=Path, default=Path("../outputs0531_teacher_homogeneous_gene_test"))
    p.add_argument("--sensitivity-root", type=Path, default=Path("../outputs0531_gene_privileged_ablation"))
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_predictions(path: Path) -> dict[str, tuple[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, tuple[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        id_col = next((c for c in ID_COLUMNS if c in fields), None)
        label_col = next((c for c in LABEL_COLUMNS if c in fields), None)
        prob_col = next((c for c in PROB_COLUMNS if c in fields), None)
        if id_col is None or label_col is None or prob_col is None:
            raise ValueError(f"Cannot identify id/label/prob columns in {path}: {fields}")
        for row in reader:
            sid = str(row.get(id_col, "")).strip()
            if not sid:
                continue
            out[sid] = (int(float(row[label_col])), max(0.0, min(1.0, float(row[prob_col]))))
    return out


def auroc(labels: list[int], probs: list[float]) -> float:
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(lab for _, lab in pairs[i:j])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for lab, prob in zip(labels, probs):
        bins[min(int(prob * n_bins), n_bins - 1)].append((lab, prob))
    total = len(labels)
    if total == 0:
        return float("nan")
    return sum(
        len(bin_rows) / total
        * abs(statistics.mean(prob for _, prob in bin_rows) - statistics.mean(lab for lab, _ in bin_rows))
        for bin_rows in bins
        if bin_rows
    )


def nll(labels: list[int], probs: list[float], eps: float = 1e-7) -> float:
    return -statistics.mean(
        lab * math.log(max(min(prob, 1.0 - eps), eps))
        + (1 - lab) * math.log(max(1.0 - prob, eps))
        for lab, prob in zip(labels, probs)
    )


def compute_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for lab, pred in zip(labels, preds) if lab == 1 and pred == 1)
    fp = sum(1 for lab, pred in zip(labels, preds) if lab == 0 and pred == 1)
    fn = sum(1 for lab, pred in zip(labels, preds) if lab == 1 and pred == 0)
    tn = sum(1 for lab, pred in zip(labels, preds) if lab == 0 and pred == 0)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    bacc = (recall + specificity) / 2.0
    out = {
        "auroc": auroc(labels, probs),
        "accuracy": accuracy,
        "balanced_accuracy": bacc,
        "f1": f1,
        "recall": recall,
        "specificity": specificity,
        "ece": ece(labels, probs),
        "brier_score": statistics.mean((prob - lab) ** 2 for lab, prob in zip(labels, probs)) if labels else float("nan"),
        "nll": nll(labels, probs),
    }
    out["composite"] = composite(out)
    return out


def composite(metrics: dict[str, float]) -> float:
    return (
        metrics["auroc"]
        + 0.5 * metrics["balanced_accuracy"]
        + 0.5 * metrics["f1"]
        + 0.25 * metrics["recall"]
        - 0.25 * metrics["ece"]
    )


def threshold_score(metrics: dict[str, float]) -> float:
    return 0.5 * metrics["balanced_accuracy"] + 0.5 * metrics["f1"] + 0.25 * metrics["recall"]


def select_val_threshold(pred: dict[str, tuple[int, float]]) -> float:
    items = sorted(pred.items())
    labels = [v[0] for _, v in items]
    probs = [v[1] for _, v in items]
    best_t = 0.5
    best_score = -float("inf")
    for i in range(1, 100):
        t = i / 100.0
        score = threshold_score(compute_metrics(labels, probs, threshold=t))
        if score > best_score or (math.isclose(score, best_score) and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t = t
            best_score = score
    return best_t


def metrics_for_pred(pred: dict[str, tuple[int, float]], threshold: float) -> dict[str, float]:
    items = sorted(pred.items())
    labels = [v[0] for _, v in items]
    probs = [v[1] for _, v in items]
    return compute_metrics(labels, probs, threshold)


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_ms(values: list[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.4f}±{std:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def locked_s0_dir(root: Path, seed: str) -> Path:
    suffix = LOCKED_S0_LABELS[seed]
    return root / f"densenet3d121_ct_text_teacher_strict_seed{suffix}"


def sensitivity_s0_dir(root: Path, seed: str) -> Path:
    return root / f"ct_text_sc_densenet3d121_strict_bs4_seed{seed}"


def candidate_dir(root: Path, prefix: str, seed: str) -> Path | None:
    candidates = [
        root / "refined_candidates" / f"{prefix}_seed{seed}",
        root / f"{prefix}_seed{seed}",
    ]
    for cand in candidates:
        if (cand / "test_predictions.csv").is_file():
            return cand
    return None


def load_run(run_dir: Path, threshold_mode: str) -> dict[str, Any]:
    test_pred = load_predictions(run_dir / "test_predictions.csv")
    val_pred = load_predictions(run_dir / "val_predictions.csv")
    threshold = 0.5 if threshold_mode == "default" else select_val_threshold(val_pred)
    return {
        "dir": run_dir,
        "test_pred": test_pred,
        "val_pred": val_pred,
        "threshold": threshold,
        "metrics": metrics_for_pred(test_pred, threshold),
        "n_test": len(test_pred),
    }


def summarize_group(run_records: dict[str, dict[str, Any]], model_name: str, threshold_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, rec in run_records.items():
        row = {
            "model": model_name,
            "threshold_mode": threshold_mode,
            "seed": seed,
            "source_seed": rec.get("source_seed", seed),
            "threshold": rec["threshold"],
            "n_test": rec["n_test"],
            "run_dir": str(rec["dir"]),
        }
        row.update(rec["metrics"])
        rows.append(row)
    summary = {
        "model": model_name,
        "threshold_mode": threshold_mode,
        "seed": "mean±std",
        "source_seed": "",
        "threshold": "",
        "n_test": "",
        "run_dir": "",
    }
    for metric in METRICS:
        summary[metric] = fmt_ms([r[metric] for r in rows])
    rows.append(summary)
    return rows


def bootstrap_group_delta(
    a_runs: dict[str, dict[str, Any]],
    b_runs: dict[str, dict[str, Any]],
    n_boot: int,
) -> dict[str, tuple[float, float, float, bool]]:
    seed_keys = sorted(set(a_runs) & set(b_runs))
    if not seed_keys:
        raise ValueError("No common seeds for bootstrap")
    common_ids = set.intersection(*(set(a_runs[s]["test_pred"]) & set(b_runs[s]["test_pred"]) for s in seed_keys))
    ids = sorted(common_ids)
    if len(ids) != 204:
        raise ValueError(f"Expected 204 aligned test samples, found {len(ids)}")
    for seed in seed_keys:
        for sid in ids:
            if a_runs[seed]["test_pred"][sid][0] != b_runs[seed]["test_pred"][sid][0]:
                raise ValueError(f"Label mismatch for seed {seed}, sample {sid}")

    def group_metrics(sample_ids: list[str], side: str) -> dict[str, float]:
        per_seed = []
        for seed in seed_keys:
            run = a_runs[seed] if side == "a" else b_runs[seed]
            labels = [run["test_pred"][sid][0] for sid in sample_ids]
            probs = [run["test_pred"][sid][1] for sid in sample_ids]
            per_seed.append(compute_metrics(labels, probs, run["threshold"]))
        return {
            metric: statistics.mean(seed_metrics[metric] for seed_metrics in per_seed)
            for metric in BOOT_METRICS
        }

    observed_a = group_metrics(ids, "a")
    observed_b = group_metrics(ids, "b")
    observed = {metric: observed_a[metric] - observed_b[metric] for metric in BOOT_METRICS}

    rng = random.Random(BOOT_SEED)
    deltas: dict[str, list[float]] = {metric: [] for metric in BOOT_METRICS}
    for _ in range(max(1, n_boot)):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        ma = group_metrics(sampled_ids, "a")
        mb = group_metrics(sampled_ids, "b")
        for metric in BOOT_METRICS:
            deltas[metric].append(ma[metric] - mb[metric])

    out: dict[str, tuple[float, float, float, bool]] = {}
    for metric in BOOT_METRICS:
        values = sorted(deltas[metric])
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        out[metric] = (observed[metric], lo, hi, lo <= 0.0 <= hi)
    return out


def write_baseline_lock(
    root: Path,
    s0_default_summary: list[dict[str, Any]],
    s0_val_summary: list[dict[str, Any]],
    sensitivity_val_summary: list[dict[str, Any]],
) -> None:
    default_mean = next(r for r in s0_default_summary if r["seed"] == "mean±std")
    val_mean = next(r for r in s0_val_summary if r["seed"] == "mean±std")
    sens_mean = next(r for r in sensitivity_val_summary if r["seed"] == "mean±std")
    text = f"""# Baseline Lock: S0_homogeneous

## Primary Baseline

The paper-facing S0 baseline is locked to `S0_homogeneous`:

- `outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42`
- `outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed43`
- `outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed44_repeat`
- `outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed45`

This baseline is used because it is the CT+Text supervised/no-gene reference model from the homogeneous teacher-student transfer experiment line.  It is therefore the fairest paper-facing comparator for a CT+Text student distilled from the CT+CNV+Text gene teacher within the same transfer chain.

## Seed 44 Policy

The original `seed44` run is excluded from the locked baseline.  The locked seed set uses `42/43/44_repeat/45` because the current decision treats `seed44_repeat` as the accepted replacement for the homogeneous CT+Text reference line.  All main R3 comparisons map student `seed44` to locked baseline `seed44_repeat`.

## Locked Baseline Metrics

Default threshold:

- AUROC: {default_mean['auroc']}
- Accuracy: {default_mean['accuracy']}
- BAcc: {default_mean['balanced_accuracy']}
- F1: {default_mean['f1']}
- Recall: {default_mean['recall']}
- Specificity: {default_mean['specificity']}
- ECE: {default_mean['ece']}
- Brier: {default_mean['brier_score']}
- NLL: {default_mean['nll']}
- Composite: {default_mean['composite']}

Validation-selected threshold:

- AUROC: {val_mean['auroc']}
- Accuracy: {val_mean['accuracy']}
- BAcc: {val_mean['balanced_accuracy']}
- F1: {val_mean['f1']}
- Recall: {val_mean['recall']}
- Specificity: {val_mean['specificity']}
- ECE: {val_mean['ece']}
- Brier: {val_mean['brier_score']}
- NLL: {val_mean['nll']}
- Composite: {val_mean['composite']}

## Relationship to Supplementary S0

`outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed42-45` is also a valid DenseNet3D121 CT+Text supervised strict baseline.  It is not used as the main comparison baseline because it belongs to the gene-privileged ablation line rather than the homogeneous teacher-student transfer line.

Supplementary sensitivity baseline metrics:

- AUROC: {sens_mean['auroc']}
- BAcc: {sens_mean['balanced_accuracy']}
- F1: {sens_mean['f1']}
- Recall: {sens_mean['recall']}
- ECE: {sens_mean['ece']}
- Brier: {sens_mean['brier_score']}

## Main Text Name

Use: **S0_homogeneous: DenseNet3D121 CT+Text supervised / no-gene reference model**.

## Supplementary Reporting

Report `outputs0531_gene_privileged_ablation` as **supplementary S0 sensitivity baseline** and state that both baselines are legal CT+Text supervised strict baselines, while the homogeneous baseline is primary for consistency with the teacher/student transfer experiment chain.
"""
    (root / "baseline_lock.md").write_text(text, encoding="utf-8")


def write_metric_files(root: Path, rows: list[dict[str, Any]], csv_name: str, md_name: str, title: str) -> None:
    fields = ["model", "threshold_mode", "seed", "source_seed", "threshold", "n_test", *METRICS, "run_dir"]
    write_csv(root / csv_name, rows, fields)
    headers = ["seed", "source", "threshold", "AUROC", "Acc", "BAcc", "F1", "Recall", "Spec", "ECE", "Brier", "NLL", "Composite"]
    md_rows = []
    for row in rows:
        md_rows.append([
            str(row["seed"]),
            str(row.get("source_seed", "")),
            fmt(row.get("threshold")),
            str(row["auroc"]) if row["seed"] == "mean±std" else fmt(row["auroc"]),
            str(row["accuracy"]) if row["seed"] == "mean±std" else fmt(row["accuracy"]),
            str(row["balanced_accuracy"]) if row["seed"] == "mean±std" else fmt(row["balanced_accuracy"]),
            str(row["f1"]) if row["seed"] == "mean±std" else fmt(row["f1"]),
            str(row["recall"]) if row["seed"] == "mean±std" else fmt(row["recall"]),
            str(row["specificity"]) if row["seed"] == "mean±std" else fmt(row["specificity"]),
            str(row["ece"]) if row["seed"] == "mean±std" else fmt(row["ece"]),
            str(row["brier_score"]) if row["seed"] == "mean±std" else fmt(row["brier_score"]),
            str(row["nll"]) if row["seed"] == "mean±std" else fmt(row["nll"]),
            str(row["composite"]) if row["seed"] == "mean±std" else fmt(row["composite"]),
        ])
    (root / md_name).write_text(f"# {title}\n\n" + md_table(headers, md_rows) + "\n", encoding="utf-8")


def rows_for_comparison(
    name: str,
    threshold_mode: str,
    a_name: str,
    b_name: str,
    a_runs: dict[str, dict[str, Any]],
    b_runs: dict[str, dict[str, Any]],
    n_boot: int,
) -> list[dict[str, Any]]:
    boot = bootstrap_group_delta(a_runs, b_runs, n_boot)
    rows = []
    for metric in BOOT_METRICS:
        a_values = [a_runs[seed]["metrics"][metric] for seed in sorted(a_runs)]
        b_values = [b_runs[seed]["metrics"][metric] for seed in sorted(b_runs)]
        delta, lo, hi, crosses = boot[metric]
        rows.append({
            "comparison": name,
            "threshold_mode": threshold_mode,
            "model_a": a_name,
            "model_b": b_name,
            "metric": metric,
            "model_a_mean_std": fmt_ms(a_values),
            "model_b_mean_std": fmt_ms(b_values),
            "delta_a_minus_b": delta,
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_crosses_0": crosses,
            "n_samples": 204,
            "n_bootstrap": n_boot,
        })
    return rows


def load_group(run_dirs: dict[str, Path], threshold_mode: str, source_seed_map: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for seed, run_dir in run_dirs.items():
        rec = load_run(run_dir, threshold_mode)
        rec["source_seed"] = (source_seed_map or {}).get(seed, seed)
        out[seed] = rec
    return out


def write_final_tables(root: Path, rows: list[dict[str, Any]], missing: list[str]) -> None:
    fields = [
        "comparison", "threshold_mode", "model_a", "model_b", "metric",
        "model_a_mean_std", "model_b_mean_std", "delta_a_minus_b",
        "ci95_low", "ci95_high", "ci_crosses_0", "n_samples", "n_bootstrap",
    ]
    write_csv(root / "r3_final_vs_locked_s0_table.csv", rows, fields)
    headers = ["comparison", "threshold", "metric", "A", "B", "delta", "95% CI", "crosses 0"]
    md_rows = []
    for row in rows:
        md_rows.append([
            row["comparison"],
            row["threshold_mode"],
            row["metric"],
            row["model_a_mean_std"],
            row["model_b_mean_std"],
            fmt(row["delta_a_minus_b"]),
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
            str(row["ci_crosses_0"]),
        ])
    body = "# R3 Final vs Locked S0 Table\n\n"
    if missing:
        body += "## Missing Inputs\n\n" + "\n".join(f"- {m}" for m in missing) + "\n\n"
    body += md_table(headers, md_rows) + "\n"
    (root / "r3_final_vs_locked_s0_table.md").write_text(body, encoding="utf-8")


def write_review(root: Path, rows: list[dict[str, Any]], missing: list[str]) -> None:
    lines = ["# R3 Final vs Locked S0 Review\n"]
    lines.append("## Baseline")
    lines.append("Primary S0 is locked to `S0_homogeneous` using seeds `42/43/44_repeat/45` from `outputs0531_teacher_homogeneous_gene_test`.")
    lines.append("")
    lines.append("## Threshold Policy")
    lines.append("- `default`: both S0 and R3 use threshold 0.5.")
    lines.append("- `val_selected`: each model uses its own validation-selected threshold, fit on val predictions and applied to test predictions.")
    lines.append("- The main comparison should use `val_selected` when discussing threshold-optimized deployable performance; default-threshold rows are supplementary.")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.extend(f"- {m}" for m in missing)
        lines.append("")
        lines.append("R3 bootstrap claims were not computed for missing comparisons. Copy or generate the listed `test_predictions.csv` and `val_predictions.csv` files, then rerun this script.")
    else:
        def subset(comp: str, mode: str) -> list[dict[str, Any]]:
            return [r for r in rows if r["comparison"] == comp and r["threshold_mode"] == mode]

        for comp in ["R3 vs locked S0", "R3_repeat vs locked S0", "R3_repeat vs R3"]:
            rows_val = subset(comp, "val_selected")
            if not rows_val:
                continue
            lines.append(f"## {comp} Val-Selected")
            for metric in BOOT_METRICS:
                row = next(r for r in rows_val if r["metric"] == metric)
                lines.append(
                    f"- {metric}: delta={fmt(row['delta_a_minus_b'])}, "
                    f"95% CI=[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}], "
                    f"crosses_0={row['ci_crosses_0']}"
                )
            lines.append("")

        r3_val = subset("R3 vs locked S0", "val_selected")
        supported = r3_val and all(
            (not r["ci_crosses_0"])
            and ((r["delta_a_minus_b"] > 0 and r["metric"] not in {"ece", "brier_score"})
                 or (r["delta_a_minus_b"] < 0 and r["metric"] in {"ece", "brier_score"}))
            for r in r3_val
        )
        partial = r3_val and any(not r["ci_crosses_0"] for r in r3_val)
        lines.append("## Paper Wording")
        lines.append("A. If bootstrap supports comprehensive R3 improvement:")
        lines.append("置信度加权的基因增强教师蒸馏学生模型在验证集阈值选择后，相比同构 CT-文本有监督参照模型，在 AUROC、BAcc、F1、Recall 和校准指标上均取得改善。")
        lines.append("")
        lines.append("B. If bootstrap supports only partial metrics:")
        lines.append("该蒸馏策略主要改善 AUROC、Recall 和校准，而 BAcc/F1 的提升需谨慎解释。")
        lines.append("")
        lines.append(f"Current automated route: {'A' if supported else 'B' if partial else 'insufficient/statistically mixed'}")
    (root / "r3_final_vs_locked_s0_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sensitivity(root: Path, locked_rows: list[dict[str, Any]], supp_rows: list[dict[str, Any]], sensitivity_boot: list[dict[str, Any]]) -> None:
    locked_mean = next(r for r in locked_rows if r["seed"] == "mean±std")
    supp_mean = next(r for r in supp_rows if r["seed"] == "mean±std")
    lines = ["# Supplementary S0 Sensitivity\n"]
    lines.append("Both baselines are legal DenseNet3D121 CT+Text supervised strict baselines.")
    lines.append("The main text uses `S0_homogeneous` because it is aligned with the homogeneous teacher/student transfer experiment chain.")
    lines.append("The `outputs0531_gene_privileged_ablation` S0 is reported as a sensitivity baseline.")
    lines.append("")
    lines.append("## Group Metrics")
    headers = ["baseline", "AUROC", "BAcc", "F1", "Recall", "ECE", "Brier", "Composite"]
    md_rows = [
        ["S0_homogeneous", locked_mean["auroc"], locked_mean["balanced_accuracy"], locked_mean["f1"], locked_mean["recall"], locked_mean["ece"], locked_mean["brier_score"], locked_mean["composite"]],
        ["S0_gene_privileged_sensitivity", supp_mean["auroc"], supp_mean["balanced_accuracy"], supp_mean["f1"], supp_mean["recall"], supp_mean["ece"], supp_mean["brier_score"], supp_mean["composite"]],
    ]
    lines.append(md_table(headers, md_rows))
    if sensitivity_boot:
        lines.append("")
        lines.append("## Paired Bootstrap: S0_homogeneous minus Sensitivity S0")
        boot_rows = []
        for row in sensitivity_boot:
            boot_rows.append([
                row["threshold_mode"],
                row["metric"],
                fmt(row["delta_a_minus_b"]),
                f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
                str(row["ci_crosses_0"]),
            ])
        lines.append(md_table(["threshold", "metric", "delta", "95% CI", "crosses 0"], boot_rows))
    (root / "supplementary_s0_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    homogeneous_root = args.homogeneous_root.expanduser().resolve()
    sensitivity_root = args.sensitivity_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    locked_dirs = {seed: locked_s0_dir(homogeneous_root, seed) for seed in SEED_LABELS}
    sensitivity_dirs = {seed: sensitivity_s0_dir(sensitivity_root, seed) for seed in SEED_LABELS}
    locked_source_map = {seed: LOCKED_S0_LABELS[seed] for seed in SEED_LABELS}

    locked_default = load_group(locked_dirs, "default", locked_source_map)
    locked_val = load_group(locked_dirs, "val_selected", locked_source_map)
    sensitivity_default = load_group(sensitivity_dirs, "default")
    sensitivity_val = load_group(sensitivity_dirs, "val_selected")

    locked_default_rows = summarize_group(locked_default, "S0_homogeneous", "default")
    locked_val_rows = summarize_group(locked_val, "S0_homogeneous", "val_selected")
    locked_rows = locked_default_rows + locked_val_rows
    write_metric_files(
        root,
        locked_rows,
        "s0_homogeneous_locked_metrics.csv",
        "s0_homogeneous_locked_metrics.md",
        "S0 Homogeneous Locked Metrics",
    )
    sensitivity_rows = summarize_group(sensitivity_val, "S0_gene_privileged_sensitivity", "val_selected")
    write_baseline_lock(root, locked_default_rows, locked_val_rows, sensitivity_rows)

    r3_prefix = "R3_confidence_a0.1_T8_bs12_lr1e-4_composite"
    r3_repeat_prefix = "R3_repeat1_confidence_a0.1_T8_bs12_lr1e-4_composite"
    r3_dirs = {seed: candidate_dir(root, r3_prefix, seed) for seed in SEED_LABELS}
    r3_repeat_dirs = {seed: candidate_dir(root, r3_repeat_prefix, seed) for seed in SEED_LABELS}
    missing = []
    for label, dirs in [("R3", r3_dirs), ("R3_repeat", r3_repeat_dirs)]:
        for seed, run_dir in dirs.items():
            if run_dir is None:
                missing.append(f"{label} seed{seed}: missing `{label.lower()}.../test_predictions.csv` under {root}/refined_candidates")
            elif not (run_dir / "val_predictions.csv").is_file():
                missing.append(f"{label} seed{seed}: missing `{run_dir / 'val_predictions.csv'}`")

    final_rows: list[dict[str, Any]] = []
    sensitivity_boot: list[dict[str, Any]] = []
    for mode in ["default", "val_selected"]:
        locked_runs = locked_default if mode == "default" else locked_val
        sensitivity_runs = sensitivity_default if mode == "default" else sensitivity_val
        sensitivity_boot.extend(
            rows_for_comparison(
                "S0_homogeneous vs sensitivity S0",
                mode,
                "S0_homogeneous",
                "S0_gene_privileged_sensitivity",
                locked_runs,
                sensitivity_runs,
                args.bootstrap_iters,
            )
        )

        if not any(m.startswith("R3 ") for m in missing):
            r3_runs = load_group({seed: r3_dirs[seed] for seed in SEED_LABELS if r3_dirs[seed] is not None}, mode)
            final_rows.extend(
                rows_for_comparison("R3 vs locked S0", mode, "R3", "S0_homogeneous", r3_runs, locked_runs, args.bootstrap_iters)
            )
        if not any(m.startswith("R3_repeat ") for m in missing):
            r3_repeat_runs = load_group({seed: r3_repeat_dirs[seed] for seed in SEED_LABELS if r3_repeat_dirs[seed] is not None}, mode)
            final_rows.extend(
                rows_for_comparison("R3_repeat vs locked S0", mode, "R3_repeat", "S0_homogeneous", r3_repeat_runs, locked_runs, args.bootstrap_iters)
            )
        if not any(m.startswith("R3 ") for m in missing) and not any(m.startswith("R3_repeat ") for m in missing):
            r3_runs = load_group({seed: r3_dirs[seed] for seed in SEED_LABELS if r3_dirs[seed] is not None}, mode)
            r3_repeat_runs = load_group({seed: r3_repeat_dirs[seed] for seed in SEED_LABELS if r3_repeat_dirs[seed] is not None}, mode)
            final_rows.extend(
                rows_for_comparison("R3_repeat vs R3", mode, "R3_repeat", "R3", r3_repeat_runs, r3_runs, args.bootstrap_iters)
            )

    write_csv(
        root / "r3_final_vs_locked_s0_bootstrap.csv",
        final_rows,
        [
            "comparison", "threshold_mode", "model_a", "model_b", "metric",
            "model_a_mean_std", "model_b_mean_std", "delta_a_minus_b",
            "ci95_low", "ci95_high", "ci_crosses_0", "n_samples", "n_bootstrap",
        ],
    )
    write_final_tables(root, final_rows, missing)
    write_review(root, final_rows, missing)
    write_sensitivity(root, locked_val_rows, sensitivity_rows, sensitivity_boot)

    print(f"[OK] wrote locked baseline reports under {root}")
    if missing:
        print("[WARN] R3 comparisons incomplete because inputs are missing:")
        for item in missing:
            print(f"  - {item}")
        return 2
    print("[OK] R3 comparisons completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
