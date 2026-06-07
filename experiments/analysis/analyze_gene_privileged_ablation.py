#!/usr/bin/env python3
"""Analyse gene privileged information ablation — DenseNet3D121 strict.

Reads results from ``--root`` (default ``outputs0531_gene_privileged_ablation``):

  Group A: ct_text_sc_densenet3d121_strict_bs4_seed{42..45}   (supervised, no KD)
  Group B: ct_text_teacher_strict_ref1019_seed42              (CT+Text teacher)
  Group C: densenet3d121_kd_from_ct_text_teacher_bs4_seed{42..45}  (KD from CT+Text teacher)
  Group D: densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed{42..45} (KD from CT+CNV+Text teacher)

Produces:
  gene_privileged_ablation_metrics.csv
  gene_privileged_ablation_metrics.md
  gene_privileged_ablation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

EXPECTED = (1019, 652, 163, 204)

CSV_FIELDS = [
    "group", "run_name", "seed", "teacher_type", "student_backbone",
    "modalities", "batch_size",
    "auroc", "balanced_accuracy", "f1", "recall", "specificity",
    "precision", "accuracy", "ece", "brier",
    "best_epoch", "test_num_samples",
    "split_status", "strict_no_leakage", "disable_text_numeric_features",
    "text_num_features", "notes",
]


def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _split_counts(p: Path) -> tuple[int, int, int, int] | None:
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None
            col = None
            for c in ("assigned_split", "split", "Split"):
                if c in reader.fieldnames:
                    col = c; break
            if col is None:
                return None
            cnt = {"train": 0, "val": 0, "test": 0}
            total = 0
            for row in reader:
                total += 1
                v = (row.get(col) or "").strip().lower()
                if v in cnt:
                    cnt[v] += 1
            return total, cnt["train"], cnt["val"], cnt["test"]
    except OSError:
        return None


def _count_rows(p: Path) -> int:
    if not p.is_file():
        return -1
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            try:
                next(r)
            except StopIteration:
                return 0
            return sum(1 for _ in r)
    except OSError:
        return -1


def _fmt(v: Any, d: int = 4) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{d}f}"
    return str(v)


def _mean_std(xs: Iterable[float]) -> tuple[float, float] | None:
    vals = [v for v in xs if isinstance(v, (int, float))]
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


# ====================== classify runs ======================

GROUP_PATTERNS = {
    "A_supervised": "ct_text_sc_densenet3d121_strict_bs4_seed",
    "B_teacher": "ct_text_teacher_strict_ref1019_seed",
    "C_kd_ct_text": "densenet3d121_kd_from_ct_text_teacher_bs4_seed",
    "D_kd_ct_cnv_text": "densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed",
}


def classify_run(name: str) -> str | None:
    for group, prefix in GROUP_PATTERNS.items():
        if name.startswith(prefix):
            return group
    return None


def teacher_type_of(group: str) -> str:
    return {
        "A_supervised": "none (supervised)",
        "B_teacher": "N/A (is teacher)",
        "C_kd_ct_text": "CT+Text (no CNV)",
        "D_kd_ct_cnv_text": "CT+CNV+Text (with gene)",
    }.get(group, "unknown")


# ====================== analyse one run ======================

def analyse(run_dir: Path, group: str) -> dict[str, Any]:
    rec: dict[str, Any] = {k: "" for k in CSV_FIELDS}
    rec["group"] = group
    rec["run_name"] = run_dir.name
    rec["teacher_type"] = teacher_type_of(group)
    notes: list[str] = []

    metrics = _read_json(run_dir / "metrics.json")
    if metrics:
        cfg = metrics.get("config") or {}
        tm = metrics.get("test_metrics") or {}
        rec["student_backbone"] = cfg.get("ct_model", "")
        rec["modalities"] = ",".join(cfg.get("modalities") or [])
        rec["batch_size"] = cfg.get("batch_size")
        rec["seed"] = cfg.get("seed")
        rec["best_epoch"] = metrics.get("best_epoch")
        rec["test_num_samples"] = tm.get("num_samples")
        rec["strict_no_leakage"] = cfg.get("strict_no_leakage")
        rec["disable_text_numeric_features"] = cfg.get("disable_text_numeric_features")
        mfd = metrics.get("modality_feature_dims") or {}
        rec["text_num_features"] = mfd.get("text_num", "")

        rec["auroc"] = tm.get("auroc")
        rec["balanced_accuracy"] = tm.get("balanced_accuracy")
        rec["f1"] = tm.get("f1")
        rec["recall"] = tm.get("recall")
        rec["specificity"] = tm.get("specificity")
        rec["precision"] = tm.get("precision")
        rec["accuracy"] = tm.get("accuracy")
        rec["ece"] = tm.get("ece")
        rec["brier"] = tm.get("brier_score")
    else:
        notes.append("missing metrics.json")

    split_counts = _split_counts(run_dir / "split_manifest.csv")
    if split_counts is None:
        rec["split_status"] = "MISSING"
    elif split_counts == EXPECTED:
        rec["split_status"] = "OK"
    else:
        rec["split_status"] = "MISMATCH"
        notes.append(f"split {split_counts} != {EXPECTED}")

    if rec.get("strict_no_leakage") is not True:
        notes.append("STRICT_NOT_ENABLED")
    if rec.get("disable_text_numeric_features") is not True:
        notes.append("NUM_FEATURES_NOT_DISABLED")

    rec["notes"] = "; ".join(notes)
    return rec


# ====================== discover runs ======================

def discover_runs(root: Path) -> list[tuple[Path, str]]:
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in ("logs", "scripts_used", "__pycache__"):
            continue
        # resolve symlinks
        target = p.resolve()
        group = classify_run(p.name)
        if group:
            out.append((target if target.is_dir() else p, group))
    return out


# ====================== output writers ======================

def write_csv(records: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def write_metrics_md(records: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Gene Privileged Ablation — Per-Run Metrics\n"]
    lines.append("| group | run_name | seed | teacher | backbone | bs | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier | best_ep | split | strict | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| " + " | ".join([
            r["group"], r["run_name"], _fmt(r.get("seed")),
            r["teacher_type"], r["student_backbone"],
            _fmt(r.get("batch_size")),
            _fmt(r.get("auroc")), _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("f1")), _fmt(r.get("recall")),
            _fmt(r.get("specificity")), _fmt(r.get("ece"), 3),
            _fmt(r.get("brier"), 4), _fmt(r.get("best_epoch")),
            r["split_status"], str(r.get("strict_no_leakage", "")),
            r["notes"],
        ]) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(records: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Gene Privileged Information Ablation — Summary\n"]
    lines.append("实验目的：判断基因特权信息 (CNV) 在 teacher 训练中是否对 DenseNet3D121 CT+Text student KD 带来额外增益。\n")

    # group by group
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    # ===================== Table 1: Per-run =====================
    lines.append("## Table 1: Per-Run Metrics\n")
    lines.append("| group | run_name | seed | teacher | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| " + " | ".join([
            r["group"], r["run_name"], _fmt(r.get("seed")),
            r["teacher_type"],
            _fmt(r.get("auroc")), _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("f1")), _fmt(r.get("recall")),
            _fmt(r.get("specificity")), _fmt(r.get("ece"), 3),
            _fmt(r.get("brier"), 4),
        ]) + " |")

    # ===================== Table 2: Group summary =====================
    lines.append("\n## Table 2: Group Summary\n")
    metrics_keys = [
        ("auroc", "AUROC"), ("balanced_accuracy", "BAcc"), ("f1", "F1"),
        ("recall", "Recall"), ("specificity", "Spec"),
        ("ece", "ECE"), ("brier", "Brier"),
    ]
    header = "| group | n | " + " | ".join(f"{label} mean±std" for _, label in metrics_keys) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(metrics_keys)) + "|"
    lines.append(header)
    lines.append(sep)

    MATCHED_SEEDS = {42, 43, 44, 45}

    group_stats: dict[str, dict[str, tuple[float, float] | None]] = {}
    for g in ["A_supervised", "B_teacher", "C_kd_ct_text", "D_kd_ct_cnv_text"]:
        runs = by_group.get(g, [])
        if not runs:
            continue
        # Use only matched seeds (42-45) for fair comparison
        matched = [r for r in runs if isinstance(r.get("seed"), int) and r["seed"] in MATCHED_SEEDS]
        if not matched:
            matched = runs  # fallback if no seed info
        stats = {}
        row_vals = [g, str(len(matched))]
        for key, label in metrics_keys:
            vals = [r[key] for r in matched if isinstance(r.get(key), (int, float))]
            ms = _mean_std(vals)
            stats[key] = ms
            if ms:
                row_vals.append(f"{ms[0]:.4f}±{ms[1]:.4f}")
            else:
                row_vals.append("-")
        group_stats[g] = stats
        lines.append("| " + " | ".join(row_vals) + " |")

    # Table 2b: D 5-seed extension (seed42-46) — NOT used in paired comparison
    d_all = by_group.get("D_kd_ct_cnv_text", [])
    d_extra = [r for r in d_all if isinstance(r.get("seed"), int) and r["seed"] not in MATCHED_SEEDS]
    if d_extra:
        lines.append("\n### Table 2b: Group D 5-Seed Extension (not used in paired comparison)\n")
        lines.append("| group | seeds | n | " + " | ".join(f"{label} mean±std" for _, label in metrics_keys) + " |")
        lines.append("|---|---|---|" + "|".join(["---"] * len(metrics_keys)) + "|")
        row_vals = ["D_kd_ct_cnv_text", "42-46", str(len(d_all))]
        for key, label in metrics_keys:
            vals = [r[key] for r in d_all if isinstance(r.get(key), (int, float))]
            ms = _mean_std(vals)
            row_vals.append(f"{ms[0]:.4f}±{ms[1]:.4f}" if ms else "-")
        lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("\n*Note: seed46 is included only in Group D extension, not in A/C/D paired comparison.*")

    # ===================== Table 3: Delta comparison =====================
    lines.append("\n## Table 3: Delta Comparison\n")
    lines.append("| comparison | ΔAUROC | ΔBAcc | ΔF1 | ΔRecall | ΔSpec | ΔECE | ΔBrier | interpretation |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    comparisons = [
        ("C - A", "C_kd_ct_text", "A_supervised", "KD (no gene) vs supervised"),
        ("D - C", "D_kd_ct_cnv_text", "C_kd_ct_text", "gene teacher vs no-gene teacher"),
        ("D - A", "D_kd_ct_cnv_text", "A_supervised", "KD (with gene) vs supervised"),
    ]

    for label, g1, g2, interp_base in comparisons:
        s1 = group_stats.get(g1, {})
        s2 = group_stats.get(g2, {})
        row = [label]
        for key, _ in metrics_keys:
            v1 = s1.get(key)
            v2 = s2.get(key)
            if v1 and v2:
                delta = v1[0] - v2[0]
                row.append(f"{delta:+.4f}")
            else:
                row.append("-")
        # interpretation
        auc1 = s1.get("auroc")
        auc2 = s2.get("auroc")
        bacc1 = s1.get("balanced_accuracy")
        bacc2 = s2.get("balanced_accuracy")
        if auc1 and auc2 and bacc1 and bacc2:
            d_auc = auc1[0] - auc2[0]
            d_bacc = bacc1[0] - bacc2[0]
            if d_auc > 0.005 and d_bacc > 0.01:
                interp = f"✅ {interp_base}: 有增益 (AUROC {d_auc:+.4f}, BAcc {d_bacc:+.4f})"
            elif d_auc > 0 and d_bacc > 0:
                interp = f"⚠️ {interp_base}: 微弱增益"
            else:
                interp = f"❌ {interp_base}: 无明显增益"
        else:
            interp = f"数据不足"
        row.append(interp)
        lines.append("| " + " | ".join(row) + " |")

    # ===================== Detailed analysis =====================
    lines.append("\n## Detailed Analysis\n")

    # Q1: Supervised baseline
    lines.append("### Q1. Supervised Baseline (Group A)\n")
    a_runs = by_group.get("A_supervised", [])
    if a_runs:
        a_aucs = [r["auroc"] for r in a_runs if isinstance(r.get("auroc"), (int, float))]
        a_baccs = [r["balanced_accuracy"] for r in a_runs if isinstance(r.get("balanced_accuracy"), (int, float))]
        a_f1s = [r["f1"] for r in a_runs if isinstance(r.get("f1"), (int, float))]
        am, asd = _mean_std(a_aucs)
        bm, bsd = _mean_std(a_baccs)
        fm, fsd = _mean_std(a_f1s)
        if am:
            lines.append(f"- AUROC: {am:.4f} ± {asd:.4f} (min={min(a_aucs):.4f}, max={max(a_aucs):.4f})")
        if bm:
            lines.append(f"- BAcc:  {bm:.4f} ± {bsd:.4f} (min={min(a_baccs):.4f}, max={max(a_baccs):.4f})")
        if fm:
            lines.append(f"- F1:    {fm:.4f} ± {fsd:.4f} (min={min(a_f1s):.4f}, max={max(a_f1s):.4f})")
        lines.append(f"- n = {len(a_aucs)}")
    else:
        lines.append("- 无数据")
    lines.append("")

    # Q2: KD from CT+Text teacher vs supervised
    lines.append("### Q2. KD from CT+Text teacher vs Supervised (C vs A)\n")
    c_runs = by_group.get("C_kd_ct_text", [])
    if c_runs and a_runs:
        c_aucs = [r["auroc"] for r in c_runs if isinstance(r.get("auroc"), (int, float))]
        c_baccs = [r["balanced_accuracy"] for r in c_runs if isinstance(r.get("balanced_accuracy"), (int, float))]
        cm, csd = _mean_std(c_aucs)
        c_bm, c_bsd = _mean_std(c_baccs)
        if cm and am:
            d_auc = cm - am
            d_bacc = c_bm - bm
            lines.append(f"- ΔAUROC = {d_auc:+.4f}  (Group C {cm:.4f} vs Group A {am:.4f})")
            lines.append(f"- ΔBAcc  = {d_bacc:+.4f}  (Group C {c_bm:.4f} vs Group A {bm:.4f})")
            if d_auc > 0.005:
                lines.append("- 结论: KD 本身带来 AUROC 提升")
            else:
                lines.append("- 结论: KD 本身对 AUROC 增益有限")
    else:
        lines.append("- 数据不足")
    lines.append("")

    # Q3: KD from CT+CNV+Text vs CT+Text teacher
    lines.append("### Q3. Gene Teacher vs No-Gene Teacher (D vs C)\n")
    d_runs = by_group.get("D_kd_ct_cnv_text", [])
    if d_runs and c_runs:
        d_aucs = [r["auroc"] for r in d_runs if isinstance(r.get("auroc"), (int, float))]
        d_baccs = [r["balanced_accuracy"] for r in d_runs if isinstance(r.get("balanced_accuracy"), (int, float))]
        dm, dsd = _mean_std(d_aucs)
        d_bm, d_bsd = _mean_std(d_baccs)
        if dm and cm:
            d_auc = dm - cm
            d_bacc = d_bm - c_bm
            lines.append(f"- ΔAUROC = {d_auc:+.4f}  (Group D {dm:.4f} vs Group C {cm:.4f})")
            lines.append(f"- ΔBAcc  = {d_bacc:+.4f}  (Group D {d_bm:.4f} vs Group C {c_bm:.4f})")
            if d_auc > 0.005 and d_bacc > 0.01:
                lines.append("- 结论: ✅ 基因特权信息带来额外增益")
            elif d_auc > 0:
                lines.append("- 结论: ⚠️ 微弱增益，需更多证据")
            else:
                lines.append("- 结论: ❌ 基因特权信息未带来额外增益")
    else:
        lines.append("- 数据不足")
    lines.append("")

    # Q4: Overall
    lines.append("### Q4. Overall: KD (with gene) vs Supervised (D vs A)\n")
    if d_runs and a_runs:
        if dm and am:
            d_auc = dm - am
            d_bacc = d_bm - bm
            lines.append(f"- ΔAUROC = {d_auc:+.4f}  (Group D {dm:.4f} vs Group A {am:.4f})")
            lines.append(f"- ΔBAcc  = {d_bacc:+.4f}  (Group D {d_bm:.4f} vs Group A {bm:.4f})")
    else:
        lines.append("- 数据不足")
    lines.append("")

    # Q5: Outlier detection
    lines.append("### Q5. Outlier Detection\n")
    for g in ["A_supervised", "C_kd_ct_text", "D_kd_ct_cnv_text"]:
        runs = by_group.get(g, [])
        if not runs:
            continue
        aucs = [(r.get("seed"), r.get("auroc")) for r in runs if isinstance(r.get("auroc"), (int, float))]
        baccs = [(r.get("seed"), r.get("balanced_accuracy")) for r in runs if isinstance(r.get("balanced_accuracy"), (int, float))]
        if len(aucs) < 2:
            continue
        auc_vals = [v for _, v in aucs]
        bacc_vals = [v for _, v in baccs]
        auc_mean = statistics.mean(auc_vals)
        bacc_mean = statistics.mean(bacc_vals)
        lines.append(f"**{g}:**")
        for seed, auc in aucs:
            flag = ""
            if abs(auc - auc_mean) > 0.02:
                flag = " ⚠️ OUTLIER"
            bacc = next((v for s, v in baccs if s == seed), None)
            lines.append(f"- seed {seed}: AUROC={auc:.4f}, BAcc={bacc:.4f}{flag}")
        lines.append("")

    # Q6: Claim boundary
    lines.append("### Q6. Claim Boundary\n")
    if group_stats.get("D_kd_ct_cnv_text") and group_stats.get("C_kd_ct_text") and group_stats.get("A_supervised"):
        d_auc = group_stats["D_kd_ct_cnv_text"].get("auroc")
        c_auc = group_stats["C_kd_ct_text"].get("auroc")
        a_auc = group_stats["A_supervised"].get("auroc")
        d_bacc = group_stats["D_kd_ct_cnv_text"].get("balanced_accuracy")
        c_bacc = group_stats["C_kd_ct_text"].get("balanced_accuracy")
        a_bacc = group_stats["A_supervised"].get("balanced_accuracy")

        if d_auc and c_auc and a_auc and d_bacc and c_bacc and a_bacc:
            d_gt_c = d_auc[0] > c_auc[0] + 0.005 and d_bacc[0] > c_bacc[0] + 0.01
            c_gt_a = c_auc[0] > a_auc[0] + 0.005

            if d_gt_c and c_gt_a:
                claim = "A"
                text = ("**Claim A**: CT+CNV+Text teacher 的特权蒸馏相较 CT+Text teacher 和 "
                        "supervised baseline 带来稳定增益，支持基因信息作为训练期特权知识。")
            elif d_auc[0] > a_auc[0] + 0.005 and not d_gt_c:
                claim = "B"
                text = ("**Claim B**: 蒸馏训练带来增益，但基因信息的独立贡献仍需更多证据。")
            elif c_auc[0] > d_auc[0]:
                claim = "C"
                text = ("**Claim C**: KD recipe 对 teacher modality 组成敏感，"
                        "CNV teacher 未表现出额外优势。")
            else:
                claim = "D"
                text = ("**Claim D**: DenseNet3D121 CT+Text supervised 已经接近最优，"
                        "蒸馏增益有限。")
            lines.append(f"判定: {text}")
            lines.append(f"\n依据:")
            lines.append(f"- Group A (supervised): AUROC {a_auc[0]:.4f}±{a_auc[1]:.4f}, BAcc {a_bacc[0]:.4f}±{a_bacc[1]:.4f}")
            lines.append(f"- Group C (KD no gene): AUROC {c_auc[0]:.4f}±{c_auc[1]:.4f}, BAcc {c_bacc[0]:.4f}±{c_bacc[1]:.4f}")
            lines.append(f"- Group D (KD with gene): AUROC {d_auc[0]:.4f}±{d_auc[1]:.4f}, BAcc {d_bacc[0]:.4f}±{d_bacc[1]:.4f}")
        else:
            lines.append("数据不足，无法判定。")
    else:
        lines.append("数据不足，无法判定。")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ====================== main ======================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0531_gene_privileged_ablation"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    discovered = discover_runs(root)
    records = [analyse(d, g) for d, g in discovered]

    csv_path = root / "gene_privileged_ablation_metrics.csv"
    md_path = root / "gene_privileged_ablation_metrics.md"
    summary_path = root / "gene_privileged_ablation_summary.md"

    write_csv(records, csv_path)
    write_metrics_md(records, md_path)
    write_summary(records, summary_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Runs analysed: {len(records)}")
    for g in ["A_supervised", "B_teacher", "C_kd_ct_text", "D_kd_ct_cnv_text"]:
        n = sum(1 for r in records if r["group"] == g)
        print(f"  {g}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
