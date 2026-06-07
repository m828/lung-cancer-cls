#!/usr/bin/env python3
"""Analyze teacher homogeneous gene test results.

Reads results from --root (default outputs0531_teacher_homogeneous_gene_test).

Groups:
  Teacher:
    T0: DenseNet3D121 CT+Text teacher strict
    T1: DenseNet3D121 CT+CNV+Text teacher strict
  Student:
    S0: DenseNet3D121 CT+Text supervised (reuse Group A)
    S1-logits: KD from T0, logits-only
    S2-logits: KD from T1, logits-only
    S1-light:  KD from T0, light-combo (no hint)
    S2-light:  KD from T1, light-combo (no hint)

Outputs:
  teacher_homogeneous_metrics.csv/md
  teacher_homogeneous_summary.md
  student_transfer_metrics.csv/md
  student_transfer_summary.md
  paired_bootstrap_teacher.csv
  paired_bootstrap_student.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

MATCHED_SEEDS = {42, 43, 44, 45}
N_BOOT = 10000
BOOT_SEED = 42

METRICS = ["auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score"]
METRIC_LABELS = {"auroc": "AUROC", "balanced_accuracy": "BAcc", "f1": "F1",
                 "recall": "Recall", "specificity": "Spec", "ece": "ECE", "brier_score": "Brier"}


# ====================== Utilities ======================

def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mean_std(vals: list[float]) -> tuple[float, float] | None:
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def _fmt(v: Any, d: int = 4) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{d}f}"
    return str(v)


def _fmt_ms(m: float | None, s: float | None, d: int = 4) -> str:
    if m is None:
        return "-"
    if s == 0 or s is None:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}±{s:.{d}f}"


# ====================== Classification ======================

TEACHER_PATTERNS = {
    "T0_ct_text": "densenet3d121_ct_text_teacher_strict_seed",
    "T1_ct_cnv_text": "densenet3d121_ct_cnv_text_teacher_strict_seed",
}

STUDENT_PATTERNS = {
    "S0_supervised": "ct_text_sc_densenet3d121_strict_bs4_seed",
    "S1_logits": "densenet3d121_kd_from_ct_text_teacher_logits_only_seed",
    "S2_logits": "densenet3d121_kd_from_ct_cnv_text_teacher_logits_only_seed",
    "S1_light": "densenet3d121_kd_from_ct_text_teacher_light_combo_seed",
    "S2_light": "densenet3d121_kd_from_ct_cnv_text_teacher_light_combo_seed",
}


def classify(name: str) -> str | None:
    for group, prefix in {**TEACHER_PATTERNS, **STUDENT_PATTERNS}.items():
        if name.startswith(prefix):
            return group
    return None


# ====================== Metric computation ======================

def _auroc(labels: list[int], probs: list[float]) -> float:
    pos = [(p, 1) for p, l in zip(probs, labels) if l == 1]
    neg = [(p, 0) for p, l in zip(probs, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    concordant = 0
    for pp, _ in pos:
        for pn, _ in neg:
            if pp > pn:
                concordant += 1
            elif pp == pn:
                concordant += 0.5
    return concordant / (len(pos) * len(neg))


def _ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for l, p in zip(labels, probs):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((l, p))
    total = len(labels)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = statistics.mean(p for _, p in b)
        avg_l = statistics.mean(l for l, _ in b)
        ece += len(b) / total * abs(avg_p - avg_l)
    return ece


def _metrics_from_subset(labels: list[int], probs: list[float]) -> dict[str, float]:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 1)
    fp = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 1)
    fn = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 0)
    tn = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bacc = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auroc = _auroc(labels, probs)
    ece = _ece(labels, probs)
    brier = statistics.mean((p - l) ** 2 for p, l in zip(probs, labels))
    return {"auroc": auroc, "balanced_accuracy": bacc, "f1": f1,
            "recall": recall, "specificity": specificity, "ece": ece, "brier_score": brier}


def load_predictions(run_dir: Path) -> dict[str, Any] | None:
    tp = run_dir / "test_predictions.csv"
    if not tp.is_file():
        return None
    try:
        with tp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = {}
            id_col = None
            for c in ("sample_id", "record_id"):
                if c in (reader.fieldnames or []):
                    id_col = c
                    break
            if not id_col:
                return None
            for row in reader:
                sid = row[id_col]
                label = int(row["label"])
                prob = float(row["prob_malignant"])
                rows[sid] = (label, prob)
        return {"rows": rows, "id_col": id_col}
    except Exception:
        return None


# ====================== Analyse one run ======================

def analyse(run_dir: Path, group: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"group": group, "run_name": run_dir.name}
    m = _read_json(run_dir / "metrics.json")
    if m:
        cfg = m.get("config") or {}
        tm = m.get("test_metrics") or {}
        rec["seed"] = cfg.get("seed")
        rec["ct_model"] = cfg.get("ct_model")
        rec["modalities"] = ",".join(cfg.get("modalities") or [])
        rec["batch_size"] = cfg.get("batch_size")
        rec["teacher_run_dir"] = cfg.get("teacher_run_dir", "none")
        rec["distill_methods"] = ",".join(cfg.get("distill_methods") or [])
        rec["best_epoch"] = m.get("best_epoch")
        for k in METRICS:
            rec[k] = tm.get(k)
    return rec


# ====================== Discover ======================

def discover(root: Path) -> list[tuple[Path, str]]:
    out = []
    if not root.is_dir():
        return out
    skip = {"logs", "scripts_used", "__pycache__"}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in skip or p.name.endswith((".csv", ".md")):
            continue
        target = p.resolve()
        group = classify(p.name)
        if group:
            out.append((target if target.is_dir() else p, group))
    return out


# ====================== Paired Bootstrap ======================

def paired_bootstrap(
    pred_index: dict[str, dict[str, tuple[int, float]]],
    group1_runs: list[dict[str, Any]],
    group2_runs: list[dict[str, Any]],
    metrics: list[str],
    n_boot: int = N_BOOT,
    boot_seed: int = BOOT_SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(boot_seed)
    g1_names = {r["run_name"] for r in group1_runs}
    g2_names = {r["run_name"] for r in group2_runs}
    common_ids = None
    for name in g1_names | g2_names:
        if name not in pred_index:
            continue
        ids = set(pred_index[name]["rows"].keys())
        if common_ids is None:
            common_ids = ids
        else:
            common_ids &= ids
    if not common_ids or len(common_ids) < 10:
        return [{"metric": m, "n_samples": 0, "delta": "", "ci_lo": "", "ci_hi": "",
                 "p_value": "", "significant": "", "note": "insufficient common samples"} for m in metrics]

    common_ids = sorted(common_ids)
    n = len(common_ids)

    def avg_prob(name: str, sid: str) -> float:
        return pred_index[name]["rows"][sid][1]

    results = []
    for metric in metrics:
        observed_g1, observed_g2 = [], []
        for sid in common_ids:
            g1_probs = [avg_prob(r["run_name"], sid) for r in group1_runs if r["run_name"] in pred_index]
            g2_probs = [avg_prob(r["run_name"], sid) for r in group2_runs if r["run_name"] in pred_index]
            if g1_probs:
                observed_g1.append(statistics.mean(g1_probs))
            if g2_probs:
                observed_g2.append(statistics.mean(g2_probs))

        labels = [pred_index[group1_runs[0]["run_name"]]["rows"][sid][0] for sid in common_ids]

        if len(observed_g1) < 2 or len(observed_g2) < 2:
            results.append({"metric": metric, "n_samples": n, "delta": "", "ci_lo": "", "ci_hi": "",
                            "p_value": "", "significant": "", "note": "insufficient data"})
            continue

        m1 = _metrics_from_subset(labels, observed_g1)
        m2 = _metrics_from_subset(labels, observed_g2)
        observed_delta = m1[metric] - m2[metric]

        deltas = []
        for _ in range(n_boot):
            idxs = [rng.randint(0, n - 1) for _ in range(n)]
            sub_labels = [labels[i] for i in idxs]
            sub_g1 = [observed_g1[i] for i in idxs]
            sub_g2 = [observed_g2[i] for i in idxs]
            sm1 = _metrics_from_subset(sub_labels, sub_g1)
            sm2 = _metrics_from_subset(sub_labels, sub_g2)
            deltas.append(sm1[metric] - sm2[metric])

        deltas.sort()
        ci_lo = deltas[int(0.025 * n_boot)]
        ci_hi = deltas[int(0.975 * n_boot)]
        if observed_delta >= 0:
            p = sum(1 for d in deltas if d <= -abs(observed_delta)) / n_boot * 2
        else:
            p = sum(1 for d in deltas if d >= abs(observed_delta)) / n_boot * 2
        p = min(p, 1.0)

        results.append({
            "metric": metric, "n_samples": n,
            "delta": f"{observed_delta:+.4f}", "ci_lo": f"{ci_lo:+.4f}", "ci_hi": f"{ci_hi:+.4f}",
            "p_value": f"{p:.4f}", "significant": "YES" if p < 0.05 else "NO", "note": "",
        })
    return results


# ====================== Writers ======================

def write_csv(records: list[dict[str, Any]], out: Path, fields: list[str]) -> None:
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def write_teacher_summary(records: list[dict[str, Any]], bootstrap: list[dict[str, Any]], out: Path) -> None:
    L = []
    a = L.append
    a("# Teacher Homogeneous Gene Test — Teacher Summary\n")
    a("比较 DenseNet3D121 CT+Text teacher vs CT+CNV+Text teacher。\n")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    a("## Per-Run Metrics\n")
    a("| group | seed | modalities | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for g in ["T0_ct_text", "T1_ct_cnv_text"]:
        for r in sorted(by_group.get(g, []), key=lambda x: x.get("seed") or 0):
            def fv(k): return _fmt(r.get(k))
            a(f"| {g} | {r.get('seed','-')} | {r.get('modalities','')} | "
              f"{fv('auroc')} | {fv('balanced_accuracy')} | {fv('f1')} | "
              f"{fv('recall')} | {fv('specificity')} | {fv('ece')} | {fv('brier_score')} |")

    a("\n## Group Summary (matched seeds 42-45)\n")
    a("| group | n | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    a("|---|---|---|---|---|---|---|---|---|")
    group_stats = {}
    for g in ["T0_ct_text", "T1_ct_cnv_text"]:
        matched = [r for r in by_group.get(g, []) if isinstance(r.get("seed"), int) and r["seed"] in MATCHED_SEEDS]
        stats = {}
        for m in METRICS:
            vals = [r[m] for r in matched if isinstance(r.get(m), (int, float))]
            stats[m] = _mean_std(vals)
        group_stats[g] = stats
        n = len(matched)
        a(f"| {g} | {n} | " + " | ".join(_fmt_ms(stats[m][0], stats[m][1]) for m in METRICS) + " |")

    a("\n## Delta: T1 - T0 (gene/CNV teacher advantage)\n")
    a("| metric | Δ | 95% CI | p-value | significant |")
    a("|---|---|---|---|---|")
    for br in bootstrap:
        ci = f"[{br['ci_lo']}, {br['ci_hi']}]" if br['ci_lo'] else "-"
        a(f"| {br['metric']} | {br['delta']} | {ci} | {br['p_value']} | {br['significant']} |")

    a("\n## Interpretation\n")
    auc_br = next((b for b in bootstrap if b["metric"] == "auroc"), None)
    if auc_br and auc_br["delta"]:
        d = float(auc_br["delta"])
        p = float(auc_br["p_value"])
        if d > 0.005 and p < 0.05:
            a("**结论**: CT+CNV+Text teacher 在 AUROC 上显著优于 CT+Text teacher，支持基因信息提升 teacher 质量。")
        elif d > 0:
            a("**结论**: CT+CNV+Text teacher AUROC 略高但未达统计显著，基因信息对 teacher 的贡献证据不足。")
        else:
            a("**结论**: CT+CNV+Text teacher 未显示 AUROC 优势，基因信息未提升 teacher 质量。")
    else:
        a("**结论**: 数据不足，无法判断。")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_student_summary(records: list[dict[str, Any]], bootstrap: list[dict[str, Any]], out: Path) -> None:
    L = []
    a = L.append
    a("# Teacher Homogeneous Gene Test — Student Summary\n")
    a("比较不同 teacher 对 DenseNet3D121 CT+Text student 的影响。\n")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    STUDENT_GROUPS = ["S0_supervised", "S1_logits", "S2_logits", "S1_light", "S2_light"]

    a("## Per-Run Metrics\n")
    a("| group | seed | teacher | distill | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in STUDENT_GROUPS:
        for r in sorted(by_group.get(g, []), key=lambda x: x.get("seed") or 0):
            def fv(k): return _fmt(r.get(k))
            teacher = r.get("teacher_run_dir", "none")
            teacher_short = teacher.split("/")[-1][:30] if teacher else "none"
            a(f"| {g} | {r.get('seed','-')} | {teacher_short} | {r.get('distill_methods','')} | "
              f"{fv('auroc')} | {fv('balanced_accuracy')} | {fv('f1')} | "
              f"{fv('recall')} | {fv('specificity')} | {fv('ece')} | {fv('brier_score')} |")

    a("\n## Group Summary (matched seeds 42-45)\n")
    a("| group | n | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    a("|---|---|---|---|---|---|---|---|---|")
    group_stats = {}
    for g in STUDENT_GROUPS:
        matched = [r for r in by_group.get(g, []) if isinstance(r.get("seed"), int) and r["seed"] in MATCHED_SEEDS]
        if not matched:
            continue
        stats = {}
        for m in METRICS:
            vals = [r[m] for r in matched if isinstance(r.get(m), (int, float))]
            stats[m] = _mean_std(vals)
        group_stats[g] = stats
        a(f"| {g} | {len(matched)} | " + " | ".join(_fmt_ms(stats[m][0], stats[m][1]) for m in METRICS) + " |")

    a("\n## Bootstrap Comparisons\n")
    a("| comparison | metric | Δ | 95% CI | p-value | sig |")
    a("|---|---|---|---|---|---|")
    for br in bootstrap:
        ci = f"[{br['ci_lo']}, {br['ci_hi']}]" if br['ci_lo'] else "-"
        a(f"| {br.get('comparison','')} | {br['metric']} | {br['delta']} | {ci} | {br['p_value']} | {br['significant']} |")

    a("\n## Interpretation\n")
    comparisons = {}
    for br in bootstrap:
        comparisons.setdefault(br.get("comparison", ""), []).append(br)

    for comp, brs in comparisons.items():
        auc_br = next((b for b in brs if b["metric"] == "auroc"), None)
        if not auc_br or not auc_br["delta"]:
            continue
        d = float(auc_br["delta"])
        p = float(auc_br["p_value"])
        if d > 0.005 and p < 0.05:
            a(f"**{comp}**: AUROC Δ={d:+.4f}, p={p:.4f} — **显著正向** ✅")
        elif d > 0:
            a(f"**{comp}**: AUROC Δ={d:+.4f}, p={p:.4f} — 正向但不显著 ⚠️")
        else:
            a(f"**{comp}**: AUROC Δ={d:+.4f}, p={p:.4f} — 无正向增益 ❌")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_bootstrap_csv(results: list[dict[str, Any]], out: Path) -> None:
    fields = ["comparison", "metric", "n_samples", "delta", "ci_lo", "ci_hi", "p_value", "significant", "note"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})


# ====================== Main ======================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0531_teacher_homogeneous_gene_test"))
    p.add_argument("--results-root", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    discovered = discover(root)
    records = [analyse(d, g) for d, g in discovered]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    # Build pred_index for sample-level bootstrap
    pred_index: dict[str, dict[str, tuple[int, float]]] = {}
    for d, g in discovered:
        pr = load_predictions(d.resolve())
        if pr:
            # Use the link name for classification consistency
            pred_index[d.name] = pr

    # ==================== Teacher analysis ====================
    teacher_records = [r for r in records if r["group"] in ("T0_ct_text", "T1_ct_cnv_text")]
    teacher_csv = root / "teacher_homogeneous_metrics.csv"
    teacher_md = root / "teacher_homogeneous_metrics.md"
    teacher_summary = root / "teacher_homogeneous_summary.md"
    bootstrap_teacher_csv = root / "paired_bootstrap_teacher.csv"

    teacher_fields = ["group", "run_name", "seed", "ct_model", "modalities", "batch_size",
                      "teacher_run_dir", "best_epoch"] + METRICS
    write_csv(teacher_records, teacher_csv, teacher_fields)

    # Teacher bootstrap: T1 - T0
    t0_runs = by_group.get("T0_ct_text", [])
    t1_runs = by_group.get("T1_ct_cnv_text", [])
    teacher_bootstrap = []
    if t0_runs and t1_runs:
        teacher_bootstrap = paired_bootstrap(pred_index, t1_runs, t0_runs, METRICS)
    write_bootstrap_csv(teacher_bootstrap, bootstrap_teacher_csv)
    write_teacher_summary(teacher_records, teacher_bootstrap, teacher_summary)

    # ==================== Student analysis ====================
    student_groups = ["S0_supervised", "S1_logits", "S2_logits", "S1_light", "S2_light"]
    student_records = [r for r in records if r["group"] in student_groups]
    student_csv = root / "student_transfer_metrics.csv"
    student_md = root / "student_transfer_metrics.md"
    student_summary = root / "student_transfer_summary.md"
    bootstrap_student_csv = root / "paired_bootstrap_student.csv"

    student_fields = ["group", "run_name", "seed", "ct_model", "modalities", "batch_size",
                      "teacher_run_dir", "distill_methods", "best_epoch"] + METRICS
    write_csv(student_records, student_csv, student_fields)

    # Student bootstrap comparisons
    student_bootstrap = []
    comparisons = [
        ("S1_logits - S0 (KD vs supervised)", "S1_logits", "S0_supervised"),
        ("S2_logits - S1_logits (gene teacher vs no-gene teacher)", "S2_logits", "S1_logits"),
        ("S2_logits - S0 (gene teacher KD vs supervised)", "S2_logits", "S0_supervised"),
    ]
    # Add light-combo comparisons if available
    if by_group.get("S1_light") and by_group.get("S2_light"):
        comparisons.extend([
            ("S1_light - S0 (light-combo KD vs supervised)", "S1_light", "S0_supervised"),
            ("S2_light - S1_light (gene teacher vs no-gene, light-combo)", "S2_light", "S1_light"),
            ("S2_light - S0 (gene teacher light-combo vs supervised)", "S2_light", "S0_supervised"),
        ])

    for comp_name, g1, g2 in comparisons:
        g1_runs = by_group.get(g1, [])
        g2_runs = by_group.get(g2, [])
        if not g1_runs or not g2_runs:
            student_bootstrap.append({"comparison": comp_name, "metric": "all",
                                      "n_samples": 0, "delta": "", "ci_lo": "", "ci_hi": "",
                                      "p_value": "", "significant": "", "note": "missing group"})
            continue
        boot = paired_bootstrap(pred_index, g1_runs, g2_runs, METRICS)
        for b in boot:
            b["comparison"] = comp_name
        student_bootstrap.extend(boot)

    write_bootstrap_csv(student_bootstrap, bootstrap_student_csv)
    write_student_summary(student_records, student_bootstrap, student_summary)

    # Print summary
    print(f"Wrote {teacher_csv}")
    print(f"Wrote {teacher_summary}")
    print(f"Wrote {bootstrap_teacher_csv}")
    print(f"Wrote {student_csv}")
    print(f"Wrote {student_summary}")
    print(f"Wrote {bootstrap_student_csv}")
    print(f"Runs analysed: {len(records)}")
    for g in sorted(by_group):
        print(f"  {g}: {len(by_group[g])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
