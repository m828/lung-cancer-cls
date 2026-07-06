#!/usr/bin/env python3
"""Analyze outputs0535 local CT+Text student KD refinement results.

Read-only analysis.  This script never starts training and never edits model
artifacts; it only summarizes completed runs under outputs0535.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_kd_optimization_suite import (  # noqa: E402
    METRICS,
    _compute_metrics,
    _fmt,
    _fmt_ms,
    _mean_std,
    _safe_float,
    load_predictions_csv,
    md_table,
    parse_seed,
    parse_split_manifest,
    write_csv,
)

BOOT_SEED = 42
EXPECTED_SPLIT = (1019, 652, 163, 204)
PRIMARY_METRICS = ["auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0535_student_kd_refinement"))
    p.add_argument("--baseline-root", type=Path, default=Path("outputs0531_gene_privileged_ablation"))
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_runs(root: Path) -> list[Path]:
    stage = root / "refined_candidates"
    if not stage.is_dir():
        return []
    return sorted(p for p in stage.iterdir() if p.is_dir() and (p / "metrics.json").is_file())


def candidate_key(run_name: str) -> str:
    seed = parse_seed(run_name)
    if seed is None:
        return run_name
    suffix = f"_seed{seed}"
    return run_name[: -len(suffix)] if run_name.endswith(suffix) else run_name


def composite_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics.get("auroc", 0.0))
        + 0.5 * float(metrics.get("balanced_accuracy", 0.0))
        + 0.5 * float(metrics.get("f1", 0.0))
        + 0.25 * float(metrics.get("recall", 0.0))
        - 0.25 * float(metrics.get("ece", 1.0))
    )


def threshold_selection_score(metrics: dict[str, float]) -> float:
    return (
        0.5 * float(metrics.get("balanced_accuracy", 0.0))
        + 0.5 * float(metrics.get("f1", 0.0))
        + 0.25 * float(metrics.get("recall", 0.0))
    )


def select_val_threshold(pred: dict[str, tuple[int, float]] | None) -> float:
    if not pred:
        return 0.5
    items = sorted(pred.items(), key=lambda x: x[0])
    labels = [v[0] for _, v in items]
    probs = [v[1] for _, v in items]
    best_t = 0.5
    best_score = -float("inf")
    for i in range(1, 100):
        t = i / 100.0
        score = threshold_selection_score(_compute_metrics(labels, probs, threshold=t))
        if score > best_score or (math.isclose(score, best_score) and abs(t - 0.5) < abs(best_t - 0.5)):
            best_score = score
            best_t = t
    return best_t


def metrics_at(pred: dict[str, tuple[int, float]] | None, threshold: float) -> dict[str, float] | None:
    if not pred:
        return None
    items = sorted(pred.items(), key=lambda x: x[0])
    labels = [v[0] for _, v in items]
    probs = [v[1] for _, v in items]
    return _compute_metrics(labels, probs, threshold=threshold)


def baseline_run_map(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("ct_text_sc_densenet3d121_strict_bs4_seed"):
            continue
        seed = parse_seed(p.name)
        if seed is not None:
            out[seed] = p
    return out


def paired_bootstrap_thresholds(
    run_a: dict[str, tuple[int, float]],
    run_b: dict[str, tuple[int, float]],
    threshold_a: float,
    threshold_b: float,
    n_boot: int,
    seed: int = BOOT_SEED,
) -> dict[str, tuple[float, float, float, float, int, str]]:
    common = sorted(set(run_a) & set(run_b))
    if not common:
        return {k: (float("nan"), float("nan"), float("nan"), 1.0, 0, "NO_COMMON_IDS") for k in PRIMARY_METRICS}
    labels = [run_a[sid][0] for sid in common]
    probs_a = [run_a[sid][1] for sid in common]
    probs_b = [run_b[sid][1] for sid in common]
    n = len(common)
    ma = _compute_metrics(labels, probs_a, threshold=threshold_a)
    mb = _compute_metrics(labels, probs_b, threshold=threshold_b)
    observed = {k: ma[k] - mb[k] for k in PRIMARY_METRICS}

    rng = random.Random(seed)
    idxs = list(range(n))
    deltas: dict[str, list[float]] = {k: [] for k in PRIMARY_METRICS}
    for _ in range(max(1, int(n_boot))):
        sample_idx = [rng.choice(idxs) for _ in range(n)]
        sample_labels = [labels[i] for i in sample_idx]
        sample_a = [probs_a[i] for i in sample_idx]
        sample_b = [probs_b[i] for i in sample_idx]
        sa = _compute_metrics(sample_labels, sample_a, threshold=threshold_a)
        sb = _compute_metrics(sample_labels, sample_b, threshold=threshold_b)
        for k in PRIMARY_METRICS:
            deltas[k].append(sa[k] - sb[k])

    out: dict[str, tuple[float, float, float, float, int, str]] = {}
    for k in PRIMARY_METRICS:
        d = sorted(deltas[k])
        lo = d[int(0.025 * (len(d) - 1))]
        hi = d[int(0.975 * (len(d) - 1))]
        obs = observed[k]
        if obs >= 0:
            p = sum(1 for x in d if x <= -abs(obs)) / len(d) * 2
        else:
            p = sum(1 for x in d if x >= abs(obs)) / len(d) * 2
        out[k] = (obs, lo, hi, min(1.0, max(0.0, p)), n, "")
    return out


def run_record(run_dir: Path) -> dict[str, Any]:
    m = read_json(run_dir / "metrics.json") or {}
    cfg = m.get("cached_kd_config") or {}
    base_cfg = m.get("config") or {}
    split_total, split_counts = parse_split_manifest(run_dir / "split_manifest.csv")
    pred = load_predictions_csv(run_dir / "test_predictions.csv")
    kd_summary = m.get("kd_sample_weights_summary") or {}
    seed = parse_seed(run_dir.name)
    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "candidate": candidate_key(run_dir.name),
        "seed": seed,
        "best_epoch": m.get("best_epoch", ""),
        "selection_metric": m.get("selection_metric", ""),
        "alpha": m.get("distillation_alpha", cfg.get("distillation_alpha", "")),
        "temperature": m.get("distillation_temperature", cfg.get("distillation_temperature", "")),
        "kd_weighting": m.get("kd_weighting", cfg.get("kd_weighting", "")),
        "batch_size": cfg.get("batch_size", base_cfg.get("batch_size", "")),
        "grad_accum_steps": cfg.get("grad_accum_steps", ""),
        "lr": cfg.get("lr", base_cfg.get("lr", "")),
        "weight_decay": cfg.get("weight_decay", base_cfg.get("weight_decay", "")),
        "scheduler": cfg.get("scheduler", base_cfg.get("scheduler_name", "")),
        "ct_model": base_cfg.get("ct_model", ""),
        "modalities": ",".join(m.get("modalities") or base_cfg.get("modalities") or []),
        "strict_no_leakage": bool(base_cfg.get("strict_no_leakage", False)),
        "disable_text_numeric_features": bool(base_cfg.get("disable_text_numeric_features", False)),
        "split_status": "OK" if split_counts == EXPECTED_SPLIT else "CHECK",
        "split_counts": "/".join(str(x) for x in split_counts) if split_counts else "",
        "test_pred_rows": len(pred) if pred else 0,
        "kd_weight_mean": _safe_float(kd_summary.get("mean")),
        "kd_weight_std": _safe_float(kd_summary.get("std")),
        "zero_weight_samples_count": kd_summary.get("zero_weight_samples_count", ""),
        "T1_advantage_samples_count": kd_summary.get("T1_advantage_samples_count", ""),
        "learnable_advantage_samples_count": kd_summary.get("learnable_advantage_samples_count", ""),
        "cached_teacher_targets": m.get("cached_teacher_targets", ""),
        "reference_teacher_targets": m.get("reference_teacher_targets", ""),
        "s0_predictions": m.get("s0_predictions", ""),
    }
    row["valid"] = (
        row["strict_no_leakage"] is True
        and row["disable_text_numeric_features"] is True
        and row["ct_model"] == "densenet3d_121"
        and row["modalities"] == "ct,text"
        and row["test_pred_rows"] == 204
        and row["split_status"] == "OK"
    )
    return row


def write_inventory(records: list[dict[str, Any]], root: Path) -> None:
    fields = [
        "candidate", "run_name", "seed", "valid", "split_status", "split_counts", "test_pred_rows",
        "selection_metric", "best_epoch", "alpha", "temperature", "kd_weighting", "batch_size",
        "grad_accum_steps", "lr", "weight_decay", "scheduler", "ct_model", "modalities",
        "strict_no_leakage", "disable_text_numeric_features", "kd_weight_mean", "kd_weight_std",
        "zero_weight_samples_count", "T1_advantage_samples_count", "learnable_advantage_samples_count",
        "cached_teacher_targets", "reference_teacher_targets", "s0_predictions",
    ]
    write_csv(records, root / "refinement_run_inventory.csv", fields)
    rows = [[str(r.get(k, "")) for k in fields[:24]] for r in records]
    (root / "refinement_run_inventory.md").write_text(
        "# Refinement Run Inventory\n\n" + md_table(rows, fields[:24]) + "\n",
        encoding="utf-8",
    )


def write_metric_tables(
    metric_rows: list[dict[str, Any]],
    root: Path,
    mode: str,
) -> None:
    fields = [
        "threshold_mode", "candidate", "run_name", "seed", "threshold", "baseline_threshold",
        "composite_score", "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll",
        "s0_auroc", "s0_balanced_accuracy", "s0_f1", "s0_recall", "s0_specificity", "s0_ece", "s0_brier_score", "s0_nll",
        "delta_auroc", "delta_balanced_accuracy", "delta_f1", "delta_recall", "delta_specificity", "delta_ece", "delta_brier_score", "delta_nll",
    ]
    stem = "refinement_default_threshold_metrics" if mode == "default" else "refinement_val_selected_threshold_metrics"
    rows = [r for r in metric_rows if r["threshold_mode"] == mode]
    write_csv(rows, root / f"{stem}.csv", fields)
    md_headers = ["candidate", "seed", "threshold", "AUROC", "BAcc", "F1", "Recall", "ECE", "Brier", "dAUROC", "dBAcc", "dF1", "dRecall", "dECE"]
    md_rows = []
    for r in rows:
        md_rows.append([
            str(r["candidate"]),
            str(r["seed"]),
            _fmt(r["threshold"]),
            _fmt(r["auroc"]),
            _fmt(r["balanced_accuracy"]),
            _fmt(r["f1"]),
            _fmt(r["recall"]),
            _fmt(r["ece"]),
            _fmt(r["brier_score"]),
            _fmt(r["delta_auroc"]),
            _fmt(r["delta_balanced_accuracy"]),
            _fmt(r["delta_f1"]),
            _fmt(r["delta_recall"]),
            _fmt(r["delta_ece"]),
        ])
    title = "Default Threshold Metrics" if mode == "default" else "Val-Selected Threshold Metrics"
    (root / f"{stem}.md").write_text("# " + title + "\n\n" + md_table(md_rows, md_headers) + "\n", encoding="utf-8")


def build_metric_rows(records: list[dict[str, Any]], run_dirs: dict[str, Path], s0_map: dict[int, Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        seed = r.get("seed")
        if seed is None or seed not in s0_map:
            continue
        run_dir = run_dirs[r["run_name"]]
        pred = load_predictions_csv(run_dir / "test_predictions.csv")
        val_pred = load_predictions_csv(run_dir / "val_predictions.csv")
        s0_dir = s0_map[int(seed)]
        s0_pred = load_predictions_csv(s0_dir / "test_predictions.csv")
        s0_val_pred = load_predictions_csv(s0_dir / "val_predictions.csv")
        if not pred or not s0_pred:
            continue
        thresholds = {
            "default": (0.5, 0.5),
            "val_selected": (select_val_threshold(val_pred), select_val_threshold(s0_val_pred)),
        }
        for mode, (th, s0_th) in thresholds.items():
            m = metrics_at(pred, th)
            b = metrics_at(s0_pred, s0_th)
            if not m or not b:
                continue
            row = {
                "threshold_mode": mode,
                "candidate": r["candidate"],
                "run_name": r["run_name"],
                "seed": seed,
                "threshold": th,
                "baseline_threshold": s0_th,
                "composite_score": composite_score(m),
            }
            for metric in PRIMARY_METRICS:
                row[metric] = m.get(metric)
                row[f"s0_{metric}"] = b.get(metric)
                row[f"delta_{metric}"] = (m.get(metric) - b.get(metric)) if m.get(metric) is not None and b.get(metric) is not None else ""
            out.append(row)
    return out


def write_bootstrap(metric_rows: list[dict[str, Any]], run_dirs: dict[str, Path], s0_map: dict[int, Path], root: Path, n_boot: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in metric_rows:
        seed = int(r["seed"])
        pred = load_predictions_csv(run_dirs[str(r["run_name"])] / "test_predictions.csv")
        s0_pred = load_predictions_csv(s0_map[seed] / "test_predictions.csv")
        if not pred or not s0_pred:
            continue
        boot = paired_bootstrap_thresholds(
            pred,
            s0_pred,
            float(r["threshold"]),
            float(r["baseline_threshold"]),
            n_boot,
        )
        for metric in PRIMARY_METRICS:
            delta, lo, hi, p, n, note = boot[metric]
            rows.append({
                "candidate": r["candidate"],
                "run_name": r["run_name"],
                "seed": seed,
                "threshold_mode": r["threshold_mode"],
                "metric": metric,
                "delta": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": p,
                "n_samples": n,
                "note": note,
            })
    fields = ["candidate", "run_name", "seed", "threshold_mode", "metric", "delta", "ci_lo", "ci_hi", "p_value", "n_samples", "note"]
    write_csv(rows, root / "refinement_paired_bootstrap_vs_s0.csv", fields)
    md_rows = []
    for r in rows:
        if r["metric"] in {"auroc", "balanced_accuracy", "f1", "recall", "ece", "brier_score"}:
            md_rows.append([
                str(r["candidate"]),
                str(r["seed"]),
                str(r["threshold_mode"]),
                str(r["metric"]),
                _fmt(r["delta"]),
                _fmt(r["ci_lo"]),
                _fmt(r["ci_hi"]),
                _fmt(r["p_value"]),
                str(r["n_samples"]),
            ])
    (root / "refinement_paired_bootstrap_vs_s0.md").write_text(
        "# Paired Bootstrap vs S0 Supervised\n\n"
        + md_table(md_rows, ["candidate", "seed", "threshold_mode", "metric", "delta", "ci_lo", "ci_hi", "p", "n"])
        + "\n",
        encoding="utf-8",
    )
    return rows


def mean_std_from(rows: list[dict[str, Any]], key: str) -> str:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    ms = _mean_std(vals)
    if ms is None:
        return "-"
    return _fmt_ms(ms[0], ms[1])


def summarize_candidates(metric_rows: list[dict[str, Any]], boot_rows: list[dict[str, Any]], root: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        by_key[(str(r["candidate"]), str(r["threshold_mode"]))].append(r)

    boot_by_key_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in boot_rows:
        if isinstance(r.get("delta"), (int, float)):
            boot_by_key_metric[(str(r["candidate"]), str(r["threshold_mode"]), str(r["metric"]))].append(float(r["delta"]))

    summaries: list[dict[str, Any]] = []
    for (cand, mode), rows in sorted(by_key.items()):
        metric_means = {}
        for metric in PRIMARY_METRICS:
            vals = [float(r[metric]) for r in rows if isinstance(r.get(metric), (int, float))]
            metric_means[metric] = statistics.mean(vals) if vals else float("nan")
        delta_means = {}
        for metric in PRIMARY_METRICS:
            vals = boot_by_key_metric.get((cand, mode, metric), [])
            delta_means[metric] = statistics.mean(vals) if vals else float("nan")
        summary = {
            "candidate": cand,
            "threshold_mode": mode,
            "n_seeds": len({r["seed"] for r in rows}),
            "mean_composite_score": composite_score(metric_means),
            "auroc": mean_std_from(rows, "auroc"),
            "balanced_accuracy": mean_std_from(rows, "balanced_accuracy"),
            "f1": mean_std_from(rows, "f1"),
            "recall": mean_std_from(rows, "recall"),
            "specificity": mean_std_from(rows, "specificity"),
            "ece": mean_std_from(rows, "ece"),
            "brier_score": mean_std_from(rows, "brier_score"),
            "nll": mean_std_from(rows, "nll"),
            "delta_auroc_mean": delta_means["auroc"],
            "delta_balanced_accuracy_mean": delta_means["balanced_accuracy"],
            "delta_f1_mean": delta_means["f1"],
            "delta_recall_mean": delta_means["recall"],
            "delta_ece_mean": delta_means["ece"],
            "delta_brier_score_mean": delta_means["brier_score"],
        }
        summary["mean_noninferior_to_s0"] = (
            summary["delta_auroc_mean"] >= 0
            and summary["delta_balanced_accuracy_mean"] >= 0
            and summary["delta_f1_mean"] >= 0
            and summary["delta_recall_mean"] >= 0
        )
        summaries.append(summary)

    summaries.sort(key=lambda r: (r["threshold_mode"] == "val_selected", r["mean_noninferior_to_s0"], r["mean_composite_score"]), reverse=True)
    fields = [
        "candidate", "threshold_mode", "n_seeds", "mean_composite_score",
        "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll",
        "delta_auroc_mean", "delta_balanced_accuracy_mean", "delta_f1_mean", "delta_recall_mean", "delta_ece_mean", "delta_brier_score_mean",
        "mean_noninferior_to_s0",
    ]
    write_csv(summaries, root / "refinement_candidate_summary.csv", fields)
    md_rows = []
    for r in summaries:
        md_rows.append([
            str(r["candidate"]),
            str(r["threshold_mode"]),
            str(r["n_seeds"]),
            _fmt(r["mean_composite_score"]),
            str(r["auroc"]),
            str(r["balanced_accuracy"]),
            str(r["f1"]),
            str(r["recall"]),
            str(r["ece"]),
            _fmt(r["delta_auroc_mean"]),
            _fmt(r["delta_balanced_accuracy_mean"]),
            _fmt(r["delta_f1_mean"]),
            _fmt(r["delta_recall_mean"]),
            _fmt(r["delta_ece_mean"]),
            str(r["mean_noninferior_to_s0"]),
        ])
    (root / "refinement_candidate_summary.md").write_text(
        "# Refinement Candidate Summary\n\n"
        + md_table(md_rows, ["candidate", "threshold_mode", "seeds", "score", "AUROC", "BAcc", "F1", "Recall", "ECE", "dAUROC", "dBAcc", "dF1", "dRecall", "dECE", "noninferior"])
        + "\n",
        encoding="utf-8",
    )

    val_rows = [r for r in summaries if r["threshold_mode"] == "val_selected"]
    viable = [r for r in val_rows if r["mean_noninferior_to_s0"]]
    best = max(viable or val_rows or summaries, key=lambda r: r["mean_composite_score"], default=None)
    return summaries, best


def write_best_report(root: Path, records: list[dict[str, Any]], summaries: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = ["# Best Refined Student Candidate\n"]
    lines.append(f"- completed refined runs: {len(records)}")
    lines.append(f"- valid refined runs: {sum(1 for r in records if r.get('valid'))}")
    if not best:
        lines.extend(["", "No completed valid candidate was found."])
        (root / "best_refined_student_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend([
        "",
        "## Selected Candidate",
        f"- candidate: {best['candidate']}",
        f"- threshold mode: {best['threshold_mode']}",
        f"- composite score: {_fmt(best['mean_composite_score'])}",
        f"- AUROC: {best['auroc']}",
        f"- BAcc: {best['balanced_accuracy']}",
        f"- F1: {best['f1']}",
        f"- Recall: {best['recall']}",
        f"- ECE: {best['ece']}",
        f"- Brier: {best['brier_score']}",
        "",
        "## Mean Delta vs S0",
        f"- dAUROC: {_fmt(best['delta_auroc_mean'])}",
        f"- dBAcc: {_fmt(best['delta_balanced_accuracy_mean'])}",
        f"- dF1: {_fmt(best['delta_f1_mean'])}",
        f"- dRecall: {_fmt(best['delta_recall_mean'])}",
        f"- dECE: {_fmt(best['delta_ece_mean'])}",
        f"- dBrier: {_fmt(best['delta_brier_score_mean'])}",
        "",
        "## Decision",
        "- Stable non-inferiority target met: " + str(best["mean_noninferior_to_s0"]),
        "- Required target: AUROC higher than S0 and BAcc/F1/Recall not lower than S0.",
    ])
    (root / "best_refined_student_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1
    runs = discover_runs(root)
    run_dirs = {p.name: p for p in runs}
    records = [run_record(p) for p in runs]
    write_inventory(records, root)
    s0_map = baseline_run_map(args.baseline_root.expanduser().resolve())
    metric_rows = build_metric_rows(records, run_dirs, s0_map)
    write_metric_tables(metric_rows, root, "default")
    write_metric_tables(metric_rows, root, "val_selected")
    boot_rows = write_bootstrap(metric_rows, run_dirs, s0_map, root, args.bootstrap_iters)
    summaries, best = summarize_candidates(metric_rows, boot_rows, root)
    write_best_report(root, records, summaries, best)
    print(f"[OK] analyzed {len(records)} refined runs under {root}")
    if best:
        print(f"[OK] best refined candidate: {best['candidate']} ({best['threshold_mode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
