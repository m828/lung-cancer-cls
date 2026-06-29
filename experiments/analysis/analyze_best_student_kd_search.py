#!/usr/bin/env python3
"""Analyze outputs0534 best CT+Text student KD search.

Read-only analysis: no training is launched and no model artifacts are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    apply_temperature,
    best_thresholds,
    fit_temperature_scale,
    load_predictions_csv,
    md_table,
    metric_rows_at_threshold,
    paired_bootstrap,
    parse_seed,
    write_csv,
)

STAGE_DIRS = {
    "s1_ct_text_teacher_kd",
    "s2_gene_teacher_kd",
    "selective_kd",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0534_best_student_kd_search"))
    p.add_argument("--baseline-root", type=Path, default=Path("outputs0531_gene_privileged_ablation"))
    p.add_argument("--s1-baseline-root", type=Path, default=Path("outputs0532_privileged_student_kd_optimization"))
    p.add_argument("--gene-baseline-root", type=Path, default=Path("outputs0533_kd_optimization_suite"))
    p.add_argument("--teacher-root", type=Path, default=Path("outputs0531_teacher_homogeneous_gene_test"))
    p.add_argument("--bootstrap-iters", type=int, default=5000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_runs(root: Path) -> list[tuple[Path, str]]:
    runs = []
    for stage in sorted(STAGE_DIRS):
        d = root / stage
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_dir() and (p / "metrics.json").is_file():
                runs.append((p, stage))
    return runs


def classify(stage: str, name: str) -> str:
    if stage == "s1_ct_text_teacher_kd":
        return "S1_cached_ct_text_teacher"
    if stage == "s2_gene_teacher_kd":
        return "S2_cached_gene_teacher"
    if stage == "selective_kd":
        if "advantage" in name:
            return "S2_selective_advantage"
        if "confidence" in name:
            return "S2_selective_confidence"
        if "margin" in name:
            return "S2_selective_margin"
    return stage


def metric_from_predictions(run_dir: Path) -> dict[str, float] | None:
    pred = load_predictions_csv(run_dir / "test_predictions.csv")
    if not pred:
        return None
    labels = [v[0] for _, v in sorted(pred.items(), key=lambda x: x[0])]
    probs = [v[1] for _, v in sorted(pred.items(), key=lambda x: x[0])]
    return _compute_metrics(labels, probs)


def record_for(run_dir: Path, stage: str) -> dict[str, Any]:
    m = read_json(run_dir / "metrics.json") or {}
    cfg = m.get("cached_kd_config") or m.get("config") or {}
    tm = m.get("test_metrics") or {}
    vm = m.get("val_metrics") or {}
    fallback = metric_from_predictions(run_dir) or {}
    r: dict[str, Any] = {
        "run_name": run_dir.name,
        "stage": stage,
        "group": classify(stage, run_dir.name),
        "seed": parse_seed(run_dir.name),
        "best_epoch": m.get("best_epoch", ""),
        "selection_metric": m.get("selection_metric", ""),
        "cached_teacher_targets": m.get("cached_teacher_targets", ""),
        "reference_teacher_targets": m.get("reference_teacher_targets", ""),
        "alpha": m.get("distillation_alpha", cfg.get("distillation_alpha", "")),
        "temperature": m.get("distillation_temperature", cfg.get("distillation_temperature", "")),
        "kd_weighting": m.get("kd_weighting", cfg.get("kd_weighting", "none")),
        "batch_size": cfg.get("batch_size", ""),
        "grad_accum_steps": cfg.get("grad_accum_steps", ""),
        "lr": cfg.get("lr", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "scheduler": cfg.get("scheduler", ""),
        "strict_no_leakage": bool((m.get("config") or {}).get("strict_no_leakage", False)),
        "disable_text_numeric_features": bool((m.get("config") or {}).get("disable_text_numeric_features", False)),
    }
    for metric in METRICS:
        r[metric] = _safe_float(tm.get(metric, fallback.get(metric)))
        r[f"val_{metric}"] = _safe_float(vm.get(metric))
    pred = load_predictions_csv(run_dir / "test_predictions.csv")
    r["test_pred_rows"] = len(pred) if pred else ""
    return r


def read_ref_runs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            seed = parse_seed(p.name)
            if seed is not None:
                out[seed] = p
    return out


def discover_gene_baselines(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for sub in ["alpha_sweep", "temperature_sweep", "optimizer_sweep", "light_combo_variants", "confidence_weighted_kd"]:
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_dir() or not p.name.startswith("S2"):
                continue
            seed = parse_seed(p.name)
            if seed is not None and seed not in out:
                out[seed] = p
    return out


def write_metrics(records: list[dict[str, Any]], root: Path) -> None:
    fields = [
        "stage", "group", "run_name", "seed", "best_epoch", "selection_metric",
        "alpha", "temperature", "kd_weighting", "batch_size", "grad_accum_steps",
        "lr", "weight_decay", "scheduler", "strict_no_leakage", "disable_text_numeric_features",
        "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll",
        "val_auroc", "val_balanced_accuracy", "val_f1", "val_recall", "val_ece", "test_pred_rows",
        "cached_teacher_targets", "reference_teacher_targets",
    ]
    write_csv(records, root / "best_student_metrics.csv", fields)
    rows = []
    for r in records:
        rows.append([str(r.get(k, "")) if k not in METRICS else _fmt(r.get(k)) for k in fields[:24]])
    (root / "best_student_metrics.md").write_text(
        "# Best Student Metrics\n\n" + md_table(rows, fields[:24]) + "\n",
        encoding="utf-8",
    )


def rank_records(records: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    ranked = []
    for r in records:
        auroc = r.get("auroc") or 0.0
        bacc = r.get("balanced_accuracy") or 0.0
        f1 = r.get("f1") or 0.0
        recall = r.get("recall") or 0.0
        ece = r.get("ece")
        brier = r.get("brier_score")
        nll = r.get("nll")
        ranked.append({
            **r,
            "auroc_rank_score": auroc - 0.05 * max(0.0, 0.75 - bacc) - 0.05 * max(0.0, 0.75 - f1),
            "balanced_rank_score": bacc + f1 + recall,
            "calibrated_rank_score": auroc - 0.5 * (ece if isinstance(ece, float) else 1.0) - 0.25 * (brier if isinstance(brier, float) else 1.0) - 0.1 * (nll if isinstance(nll, float) else 1.0),
        })
    ranked.sort(key=lambda x: (x["auroc_rank_score"], x["balanced_rank_score"]), reverse=True)
    write_csv(
        ranked,
        root / "best_student_ranking.csv",
        ["run_name", "stage", "group", "seed", "auroc_rank_score", "balanced_rank_score", "calibrated_rank_score", "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll"],
    )
    return ranked


def write_bootstrap(records: list[dict[str, Any]], runs: list[tuple[Path, str]], args: argparse.Namespace, root: Path) -> None:
    refs = {
        "S0_supervised": read_ref_runs(args.baseline_root.resolve(), "ct_text_sc_densenet3d121_strict_bs4_seed"),
        "S1_logits_alpha02": read_ref_runs(args.s1_baseline_root.resolve(), "S1_logits_alpha02_T4_seed"),
        "gene_teacher_kd_baseline": discover_gene_baselines(args.gene_baseline_root.resolve()),
        "teacher_upper_bound": read_ref_runs(args.teacher_root.resolve(), "densenet3d121_ct_cnv_text_teacher_strict_seed"),
    }
    run_dir_by_name = {p.name: p for p, _ in runs}
    rows = []
    for r in records:
        seed = r.get("seed")
        if seed is None:
            continue
        pred = load_predictions_csv(run_dir_by_name[str(r["run_name"])] / "test_predictions.csv")
        if not pred:
            continue
        for comp, ref_map in refs.items():
            ref_dir = ref_map.get(int(seed))
            if not ref_dir:
                continue
            ref_pred = load_predictions_csv(ref_dir / "test_predictions.csv")
            if not ref_pred:
                continue
            boot = paired_bootstrap(pred, ref_pred, args.bootstrap_iters)
            for metric in METRICS:
                delta, lo, hi, p, n, note = boot[metric]
                rows.append({
                    "run_name": r["run_name"],
                    "seed": seed,
                    "comparison": comp,
                    "metric": metric,
                    "delta": delta,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_value": p,
                    "n_samples": n,
                    "note": note,
                })
    write_csv(rows, root / "paired_bootstrap_vs_supervised.csv", ["run_name", "seed", "comparison", "metric", "delta", "ci_lo", "ci_hi", "p_value", "n_samples", "note"])


def write_thresholds(records: list[dict[str, Any]], runs: list[tuple[Path, str]], root: Path) -> None:
    run_dir_by_name = {p.name: p for p, _ in runs}
    lines = ["# Threshold and Calibration Summary\n", "| run | seed | BAcc@0.5 | F1@0.5 | best BAcc th | test BAcc | best F1 th | test F1 | T | ECE before | ECE after | NLL before | NLL after |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for r in records:
        d = run_dir_by_name[str(r["run_name"])]
        val = load_predictions_csv(d / "val_predictions.csv")
        test = load_predictions_csv(d / "test_predictions.csv")
        if not val or not test:
            continue
        val_items = sorted(val.items(), key=lambda x: x[0])
        test_items = sorted(test.items(), key=lambda x: x[0])
        val_labels = [x[1][0] for x in val_items]
        val_probs = [x[1][1] for x in val_items]
        test_labels = [x[1][0] for x in test_items]
        test_probs = [x[1][1] for x in test_items]
        default = metric_rows_at_threshold(test_labels, test_probs, 0.5)
        opt = best_thresholds(val_labels, val_probs)
        bacc_best = metric_rows_at_threshold(test_labels, test_probs, opt["best_bacc_threshold"])
        f1_best = metric_rows_at_threshold(test_labels, test_probs, opt["best_f1_threshold"])
        temp = fit_temperature_scale(val_labels, val_probs)
        before = _compute_metrics(test_labels, test_probs)
        after = _compute_metrics(test_labels, apply_temperature(test_probs, temp))
        row = {
            "run_name": r["run_name"],
            "seed": r.get("seed", ""),
            "bacc_0.5": default["balanced_accuracy"],
            "f1_0.5": default["f1"],
            "best_bacc_threshold": opt["best_bacc_threshold"],
            "bacc_best_threshold": bacc_best["balanced_accuracy"],
            "best_f1_threshold": opt["best_f1_threshold"],
            "f1_best_threshold": f1_best["f1"],
            "temperature": temp,
            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "nll_before": before["nll"],
            "nll_after": after["nll"],
        }
        csv_rows.append(row)
        lines.append("| " + " | ".join([str(row["run_name"]), str(row["seed"]), _fmt(row["bacc_0.5"]), _fmt(row["f1_0.5"]), _fmt(row["best_bacc_threshold"]), _fmt(row["bacc_best_threshold"]), _fmt(row["best_f1_threshold"]), _fmt(row["f1_best_threshold"]), _fmt(row["temperature"]), _fmt(row["ece_before"]), _fmt(row["ece_after"]), _fmt(row["nll_before"]), _fmt(row["nll_after"])]) + " |")
    write_csv(csv_rows, root / "threshold_calibration_summary.csv", ["run_name", "seed", "bacc_0.5", "f1_0.5", "best_bacc_threshold", "bacc_best_threshold", "best_f1_threshold", "f1_best_threshold", "temperature", "ece_before", "ece_after", "nll_before", "nll_after"])
    (root / "threshold_calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_transfer(root: Path) -> None:
    cache_dir = root / "cached_teacher_targets"
    rows = []
    if cache_dir.is_dir():
        for p in sorted(cache_dir.glob("*.csv")):
            vals = []
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        vals.append((row.get("split", ""), float(row.get("teacher_confidence", 0.0)), float(row.get("teacher_margin", 0.0)), int(float(row.get("teacher_correct", 0)))))
                    except ValueError:
                        pass
            for split in ["train", "val", "test"]:
                sub = [v for v in vals if v[0] == split]
                if not sub:
                    continue
                rows.append({
                    "cache": p.name,
                    "split": split,
                    "n": len(sub),
                    "mean_confidence": statistics.mean(v[1] for v in sub),
                    "mean_margin": statistics.mean(v[2] for v in sub),
                    "teacher_accuracy": statistics.mean(v[3] for v in sub),
                })
    write_csv(rows, root / "sample_transfer_summary.csv", ["cache", "split", "n", "mean_confidence", "mean_margin", "teacher_accuracy"])
    lines = ["# Sample Transfer Summary\n", "| cache | split | n | confidence | margin | teacher_acc |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join([str(r["cache"]), str(r["split"]), str(r["n"]), _fmt(r["mean_confidence"]), _fmt(r["mean_margin"]), _fmt(r["teacher_accuracy"])]) + " |")
    (root / "sample_transfer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(records: list[dict[str, Any]], ranked: list[dict[str, Any]], root: Path) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[str(r["group"])].append(r)
    lines = ["# Best Student KD Search Summary\n", f"- completed runs: {len(records)}", ""]
    lines.append("## Group Means\n")
    lines.append("| group | n | AUROC | BAcc | F1 | Recall | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in sorted(by_group):
        vals = by_group[g]
        lines.append("| " + " | ".join([
            g,
            str(len(vals)),
            _fmt_ms(*_mean_std([v["auroc"] for v in vals if isinstance(v.get("auroc"), float)])),
            _fmt_ms(*_mean_std([v["balanced_accuracy"] for v in vals if isinstance(v.get("balanced_accuracy"), float)])),
            _fmt_ms(*_mean_std([v["f1"] for v in vals if isinstance(v.get("f1"), float)])),
            _fmt_ms(*_mean_std([v["recall"] for v in vals if isinstance(v.get("recall"), float)])),
            _fmt_ms(*_mean_std([v["ece"] for v in vals if isinstance(v.get("ece"), float)])),
            _fmt_ms(*_mean_std([v["brier_score"] for v in vals if isinstance(v.get("brier_score"), float)])),
        ]) + " |")
    if ranked:
        best_auroc = max(ranked, key=lambda x: x.get("auroc") or -1)
        best_bal = max(ranked, key=lambda x: x.get("balanced_rank_score") or -1)
        best_cal = max(ranked, key=lambda x: x.get("calibrated_rank_score") or -1)
        lines.extend([
            "",
            "## Recommended Candidates",
            f"- best deployable student: {ranked[0]['run_name']}",
            f"- best AUROC student: {best_auroc['run_name']} AUROC={_fmt(best_auroc.get('auroc'))}",
            f"- best balanced student: {best_bal['run_name']} score={_fmt(best_bal.get('balanced_rank_score'))}",
            f"- best calibrated student: {best_cal['run_name']} score={_fmt(best_cal.get('calibrated_rank_score'))}",
            "",
            "Use paired_bootstrap_vs_supervised.csv before claiming superiority over supervised/S1/gene-teacher baselines.",
        ])
    (root / "best_student_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1
    runs = discover_runs(root)
    records = [record_for(p, stage) for p, stage in runs]
    write_metrics(records, root)
    ranked = rank_records(records, root)
    write_bootstrap(records, runs, args, root)
    write_thresholds(records, runs, root)
    write_sample_transfer(root)
    write_summary(records, ranked, root)
    print(f"[OK] analyzed {len(records)} runs under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
