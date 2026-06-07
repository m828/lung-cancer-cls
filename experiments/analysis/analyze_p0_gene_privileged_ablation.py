#!/usr/bin/env python3
"""P0 gene privileged ablation analysis with paired bootstrap.

Reads results from ``--root`` (default ``outputs0531_gene_privileged_ablation``).

Groups:
  A: DenseNet3D121 CT+Text supervised strict bs=4
  C: DenseNet3D121 CT+Text KD from CT+Text teacher bs=4
  D: DenseNet3D121 CT+Text KD from CT+CNV+Text teacher bs=4

Outputs:
  p0_gene_privileged_metrics.csv
  p0_gene_privileged_metrics.md
  p0_gene_privileged_summary.md
  p0_paired_bootstrap.csv
  p0_delong_auroc.csv (TODO placeholder)
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

EXPECTED = (1019, 652, 163, 204)
N_BOOT = 10000
SEED = 42

CSV_FIELDS = [
    "group", "run_name", "seed", "teacher_type", "student_backbone",
    "modalities", "batch_size",
    "auroc", "balanced_accuracy", "f1", "recall", "specificity",
    "precision", "accuracy", "ece", "brier",
    "best_epoch", "test_num_samples",
    "split_status", "strict_no_leakage", "disable_text_numeric_features",
    "notes",
]

GROUP_PATTERNS = {
    "A_supervised": "ct_text_sc_densenet3d121_strict_bs4_seed",
    "B_teacher_ct_text": "ct_text_teacher_strict_ref1019_seed",
    "B_teacher_ct_cnv_text": "ct_cnv_text_teacher_strict_ref1019_seed",
    "C_kd_ct_text": "densenet3d121_kd_from_ct_text_teacher_bs4_seed",
    "D_kd_ct_cnv_text": "densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed",
}


def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _split_ok(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            col = None
            for c in ("assigned_split", "split", "Split"):
                if reader.fieldnames and c in reader.fieldnames:
                    col = c; break
            if not col:
                return "NO_COL"
            cnt = {"train": 0, "val": 0, "test": 0}
            total = 0
            for row in reader:
                total += 1
                v = (row.get(col) or "").strip().lower()
                if v in cnt:
                    cnt[v] += 1
            if (total, cnt["train"], cnt["val"], cnt["test"]) == EXPECTED:
                return "OK"
            return "MISMATCH"
    except:
        return "ERROR"


def classify_run(name: str) -> str | None:
    for group, prefix in GROUP_PATTERNS.items():
        if name.startswith(prefix):
            return group
    return None


def teacher_type_of(group: str) -> str:
    return {
        "A_supervised": "none (supervised)",
        "B_teacher_ct_text": "N/A (is teacher, no CNV)",
        "B_teacher_ct_cnv_text": "N/A (is teacher, with CNV)",
        "C_kd_ct_text": "CT+Text (no CNV)",
        "D_kd_ct_cnv_text": "CT+CNV+Text (with gene)",
    }.get(group, "unknown")


def analyse(run_dir: Path, group: str) -> dict[str, Any]:
    rec: dict[str, Any] = {k: "" for k in CSV_FIELDS}
    rec["group"] = group
    rec["run_name"] = run_dir.name
    rec["teacher_type"] = teacher_type_of(group)
    notes = []

    m = _read_json(run_dir / "metrics.json")
    if m:
        cfg = m.get("config") or {}
        tm = m.get("test_metrics") or {}
        rec["student_backbone"] = cfg.get("ct_model", "")
        rec["modalities"] = ",".join(cfg.get("modalities") or [])
        rec["batch_size"] = cfg.get("batch_size")
        rec["seed"] = cfg.get("seed")
        rec["best_epoch"] = m.get("best_epoch")
        rec["test_num_samples"] = tm.get("num_samples")
        rec["strict_no_leakage"] = cfg.get("strict_no_leakage")
        rec["disable_text_numeric_features"] = cfg.get("disable_text_numeric_features")
        for k in ("auroc", "balanced_accuracy", "f1", "recall", "specificity",
                   "precision", "accuracy"):
            rec[k] = tm.get(k)
        rec["ece"] = tm.get("ece")
        rec["brier"] = tm.get("brier_score")
    else:
        notes.append("missing metrics.json")

    rec["split_status"] = _split_ok(run_dir / "split_manifest.csv")
    rec["notes"] = "; ".join(notes)
    return rec


def discover(root: Path) -> list[tuple[Path, str]]:
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in ("logs", "scripts_used", "__pycache__"):
            continue
        if p.name.endswith((".csv", ".md")):
            continue
        target = p.resolve()
        group = classify_run(p.name)
        if group:
            out.append((target if target.is_dir() else p, group))
    return out


def _mean_std(xs: list[float]) -> tuple[float, float] | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _fmt(v: Any, d: int = 4) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{d}f}"
    return str(v)


# ====================== Sample-level prediction loading ======================

POS_PROB_COLS = ("prob_malignant", "prob_1", "prob_pos", "y_prob", "score")
LABEL_COLS = ("label", "y_true", "target")
ID_COLS = ("sample_id", "record_id")


def load_predictions(run_dir: Path) -> dict[str, Any] | None:
    """Load test_predictions.csv -> {id_col, rows: {id: (label:int, prob:float)}}.

    Aligns by sample_id (preferred) or record_id. Returns None on failure.
    """
    p = run_dir / "test_predictions.csv"
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            id_col = next((c for c in ID_COLS if c in fields), None)
            prob_col = next((c for c in POS_PROB_COLS if c in fields), None)
            label_col = next((c for c in LABEL_COLS if c in fields), None)
            if not id_col or prob_col is None or label_col is None:
                return {"id_col": None, "rows": {},
                        "warning": f"missing id/prob/label cols (have {fields})"}
            rows: dict[str, tuple[int, float]] = {}
            for r in reader:
                rid = (r.get(id_col) or "").strip()
                if not rid:
                    continue
                try:
                    lab = int(float(r.get(label_col)))
                    prob = float(r.get(prob_col))
                except (TypeError, ValueError):
                    continue
                rows[rid] = (lab, prob)
            return {"id_col": id_col, "rows": rows, "warning": ""}
    except OSError as e:
        return {"id_col": None, "rows": {}, "warning": f"read error: {e}"}


# ---- pure-stdlib metrics on (labels, probs) over a sample-index subset ----

def _auroc(labels: list[int], probs: list[float]) -> float:
    pos = [p for l, p in zip(labels, probs) if l == 1]
    neg = [p for l, p in zip(labels, probs) if l == 0]
    if not pos or not neg:
        return float("nan")
    # rank-based Mann-Whitney U (handles ties via average ranks)
    paired = sorted(zip(probs, labels), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(rk for rk, (_, l) in zip(ranks, paired) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _confusion(labels: list[int], preds: list[int]) -> tuple[int, int, int, int]:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    return tp, tn, fp, fn


def _metrics_from_subset(labels: list[int], probs: list[float], thr: float = 0.5) -> dict[str, float]:
    preds = [1 if p >= thr else 0 for p in probs]
    tp, tn, fp, fn = _confusion(labels, preds)
    recall = tp / (tp + fn) if (tp + fn) else float("nan")       # sensitivity
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * recall / (prec + recall)
          if prec == prec and recall == recall and (prec + recall) else float("nan"))
    bacc = ((recall + spec) / 2.0
            if recall == recall and spec == spec else float("nan"))
    n = len(labels)
    brier = sum((p - l) ** 2 for l, p in zip(labels, probs)) / n if n else float("nan")
    # ECE (10 equal-width bins)
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p > lo or (b == 0 and p >= lo)) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return {"auroc": _auroc(labels, probs), "balanced_accuracy": bacc,
            "f1": f1, "recall": recall, "specificity": spec,
            "ece": ece, "brier": brier}


def _avg_probs_over_seeds(pred_list: list[dict[str, Any]], ids: list[str]) -> dict[str, float]:
    """Average positive-class prob across seeds for each id (id present in all)."""
    out = {}
    for rid in ids:
        vals = [pl["rows"][rid][1] for pl in pred_list if rid in pl["rows"]]
        if len(vals) == len(pred_list):
            out[rid] = sum(vals) / len(vals)
    return out


def sample_level_paired_bootstrap(
    preds1: list[dict[str, Any]],
    preds2: list[dict[str, Any]],
    metrics: list[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Sample-level paired bootstrap of metric deltas (group1 - group2).

    preds1/preds2 are lists of per-seed prediction dicts (already seed-matched
    upstream). Probabilities are averaged across seeds per sample, then the SAME
    204 test samples are resampled with replacement each iteration.
    """
    if not preds1 or not preds2:
        return {"error": "no predictions for one side"}
    # id space = intersection across all runs on both sides
    id_sets = [set(pl["rows"].keys()) for pl in (preds1 + preds2)]
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if len(common_ids) < 2:
        return {"error": f"insufficient aligned samples (n={len(common_ids)})"}

    p1 = _avg_probs_over_seeds(preds1, common_ids)
    p2 = _avg_probs_over_seeds(preds2, common_ids)
    common_ids = [i for i in common_ids if i in p1 and i in p2]
    # labels from first run on side 1 (identical split across runs)
    labels = {i: preds1[0]["rows"][i][0] for i in common_ids}

    lab = [labels[i] for i in common_ids]
    pr1 = [p1[i] for i in common_ids]
    pr2 = [p2[i] for i in common_ids]
    n = len(common_ids)

    obs1 = _metrics_from_subset(lab, pr1)
    obs2 = _metrics_from_subset(lab, pr2)
    observed = {m: obs1[m] - obs2[m] for m in metrics}

    rng = random.Random(seed)
    boot_deltas: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idxs = [rng.randint(0, n - 1) for _ in range(n)]
        bl = [lab[i] for i in idxs]
        b1 = [pr1[i] for i in idxs]
        b2 = [pr2[i] for i in idxs]
        m1 = _metrics_from_subset(bl, b1)
        m2 = _metrics_from_subset(bl, b2)
        for m in metrics:
            d = m1[m] - m2[m]
            if d == d:  # skip NaN
                boot_deltas[m].append(d)

    res: dict[str, Any] = {"n_samples": n, "metrics": {}}
    for m in metrics:
        ds = sorted(boot_deltas[m])
        if len(ds) < 100:
            res["metrics"][m] = {"delta": observed[m], "ci_lo": float("nan"),
                                 "ci_hi": float("nan"), "p_value": float("nan"),
                                 "significant": False}
            continue
        lo = ds[int(0.025 * len(ds))]
        hi = ds[int(0.975 * len(ds))]
        od = observed[m]
        if od >= 0:
            p = sum(1 for d in ds if d <= 0) / len(ds) * 2
        else:
            p = sum(1 for d in ds if d >= 0) / len(ds) * 2
        p = min(p, 1.0)
        res["metrics"][m] = {"delta": od, "ci_lo": lo, "ci_hi": hi,
                             "p_value": p, "significant": p < 0.05}
    return res


MATCHED_SEEDS = [42, 43, 44, 45]   # A/C/D fair paired comparison
D_EXT_SEEDS = [42, 43, 44, 45, 46]  # Group D 5-seed main-result extension


def run_bootstrap_comparisons(
    records: list[dict[str, Any]],
    pred_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sample-level paired bootstrap for A/C/D comparisons over matched seeds.

    Uses ONLY seeds 42-45 (intersection of available seeds across the two
    compared groups) so seed46 (Group-D-only) never enters a paired comparison.
    ``pred_index`` maps run_name -> loaded predictions dict.
    """
    metrics = ["auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier"]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    def seed_map(group_runs):
        return {r["seed"]: r for r in group_runs if isinstance(r.get("seed"), int)}

    a_sm = seed_map(by_group.get("A_supervised", []))
    c_sm = seed_map(by_group.get("C_kd_ct_text", []))
    d_sm = seed_map(by_group.get("D_kd_ct_cnv_text", []))

    comparisons = [
        ("C - A (KD vs supervised)", c_sm, a_sm),
        ("D - C (gene teacher vs no-gene teacher)", d_sm, c_sm),
        ("D - A (KD with gene vs supervised)", d_sm, a_sm),
    ]

    results: list[dict[str, Any]] = []
    for comp_name, sm1, sm2 in comparisons:
        # Matched seeds only, and only within MATCHED_SEEDS (42-45)
        common_seeds = sorted((set(sm1) & set(sm2)) & set(MATCHED_SEEDS))
        # collect predictions for these seeds; warn on missing/unaligned
        preds1, preds2, warns = [], [], []
        for s in common_seeds:
            pr1 = pred_index.get(sm1[s]["run_name"])
            pr2 = pred_index.get(sm2[s]["run_name"])
            for tag, pr in (("g1", pr1), ("g2", pr2)):
                if not pr or not pr.get("rows"):
                    warns.append(f"seed{s} {tag}: no test_predictions.csv / no aligned rows")
            if pr1 and pr1.get("rows") and pr2 and pr2.get("rows"):
                preds1.append(pr1)
                preds2.append(pr2)
        note = "; ".join(warns)

        if len(preds1) < 1 or len(preds2) < 1:
            results.append({
                "comparison": comp_name, "metric": "all", "n_samples": 0,
                "n_seeds": len(common_seeds), "delta": "", "ci_lo": "", "ci_hi": "",
                "p_value": "", "significant": "",
                "note": (note + "; " if note else "") + "insufficient aligned predictions",
            })
            continue

        boot = sample_level_paired_bootstrap(preds1, preds2, metrics)
        if "error" in boot:
            results.append({
                "comparison": comp_name, "metric": "all", "n_samples": 0,
                "n_seeds": len(common_seeds), "delta": "", "ci_lo": "", "ci_hi": "",
                "p_value": "", "significant": "",
                "note": (note + "; " if note else "") + boot["error"],
            })
            continue

        for m in metrics:
            md = boot["metrics"][m]
            results.append({
                "comparison": comp_name,
                "metric": m,
                "n_samples": boot["n_samples"],
                "n_seeds": len(common_seeds),
                "delta": f"{md['delta']:+.4f}" if md["delta"] == md["delta"] else "",
                "ci_lo": f"{md['ci_lo']:+.4f}" if md["ci_lo"] == md["ci_lo"] else "",
                "ci_hi": f"{md['ci_hi']:+.4f}" if md["ci_hi"] == md["ci_hi"] else "",
                "p_value": f"{md['p_value']:.4f}" if md["p_value"] == md["p_value"] else "",
                "significant": "YES" if md["significant"] else "NO",
                "note": note,
            })

    return results


# ====================== Writers ======================

def write_csv(records: list[dict[str, Any]], out: Path) -> None:
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def write_metrics_md(records: list[dict[str, Any]], out: Path) -> None:
    lines = ["# P0 Gene Privileged Ablation — Per-Run Metrics\n"]
    lines.append("| group | run_name | seed | teacher | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier | split | strict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| " + " | ".join([
            r["group"], r["run_name"], _fmt(r.get("seed")),
            r["teacher_type"],
            _fmt(r.get("auroc")), _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("f1")), _fmt(r.get("recall")),
            _fmt(r.get("specificity")), _fmt(r.get("ece"), 3),
            _fmt(r.get("brier"), 4), r["split_status"],
            str(r.get("strict_no_leakage", "")),
        ]) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    records: list[dict[str, Any]],
    bootstrap_results: list[dict[str, Any]],
    out: Path,
) -> None:
    lines = ["# P0 Gene Privileged Ablation — Summary\n"]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    metrics_keys = [
        ("auroc", "AUROC"), ("balanced_accuracy", "BAcc"), ("f1", "F1"),
        ("recall", "Recall"), ("specificity", "Spec"),
        ("ece", "ECE"), ("brier", "Brier"),
    ]

    # Table 1: Per-run
    lines.append("## Table 1: Per-Run Metrics\n")
    lines.append("| group | seed | teacher | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| " + " | ".join([
            r["group"], _fmt(r.get("seed")), r["teacher_type"],
            _fmt(r.get("auroc")), _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("f1")), _fmt(r.get("recall")),
            _fmt(r.get("specificity")), _fmt(r.get("ece"), 3),
            _fmt(r.get("brier"), 4),
        ]) + " |")

    # Table 2: Group summary (MATCHED 4-seed view: A/C/D restricted to seeds 42-45)
    lines.append("\n## Table 2: Group Summary — matched 4-seed (42-45)\n")
    lines.append("> Group D is restricted to seeds 42-45 here so A/C/D are directly comparable. "
                 "seed46 (Group-D-only) is reported separately in Table 2b.\n")
    header = "| group | n | " + " | ".join(f"{label} mean±std" for _, label in metrics_keys) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(metrics_keys)) + "|"
    lines.append(header)
    lines.append(sep)

    def _seed_of(r):
        return r.get("seed") if isinstance(r.get("seed"), int) else None

    group_stats: dict[str, dict[str, tuple[float, float] | None]] = {}
    for g in ["A_supervised", "B_teacher_ct_text", "B_teacher_ct_cnv_text",
              "C_kd_ct_text", "D_kd_ct_cnv_text"]:
        runs = by_group.get(g, [])
        # For paired groups restrict to MATCHED_SEEDS; teachers kept as-is
        if g in ("A_supervised", "C_kd_ct_text", "D_kd_ct_cnv_text"):
            runs = [r for r in runs if _seed_of(r) in MATCHED_SEEDS]
        if not runs:
            continue
        stats = {}
        row_vals = [g, str(len(runs))]
        for key, label in metrics_keys:
            vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
            ms = _mean_std(vals)
            stats[key] = ms
            row_vals.append(f"{ms[0]:.4f}±{ms[1]:.4f}" if ms else "-")
        group_stats[g] = stats
        lines.append("| " + " | ".join(row_vals) + " |")

    # Table 2b: Group D 5-seed main-result extension (42-46) — NOT used in paired comparisons
    d_ext = [r for r in by_group.get("D_kd_ct_cnv_text", []) if _seed_of(r) in D_EXT_SEEDS]
    if d_ext:
        lines.append("\n## Table 2b: Group D 5-seed main-result extension (42-46)\n")
        lines.append("> Stability/main-result reporting only. Does NOT enter A vs C vs D paired "
                     "comparisons (Table 4) unless A/C also have seed46.\n")
        lines.append(header)
        lines.append(sep)
        seeds_present = sorted(s for s in (_seed_of(r) for r in d_ext) if s is not None)
        row_vals = ["D_kd_ct_cnv_text (seeds " + ",".join(map(str, seeds_present)) + ")", str(len(d_ext))]
        for key, label in metrics_keys:
            vals = [r[key] for r in d_ext if isinstance(r.get(key), (int, float))]
            ms = _mean_std(vals)
            row_vals.append(f"{ms[0]:.4f}±{ms[1]:.4f}" if ms else "-")
        lines.append("| " + " | ".join(row_vals) + " |")

    # Table 3: Delta comparison
    lines.append("\n## Table 3: Delta Comparison\n")
    lines.append("| comparison | ΔAUROC | ΔBAcc | ΔF1 | ΔRecall | ΔSpec | ΔECE | ΔBrier | interpretation |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    comparisons = [
        ("C - A", "C_kd_ct_text", "A_supervised", "KD vs supervised"),
        ("D - C", "D_kd_ct_cnv_text", "C_kd_ct_text", "gene teacher vs no-gene"),
        ("D - A", "D_kd_ct_cnv_text", "A_supervised", "KD with gene vs supervised"),
    ]
    for label, g1, g2, interp_base in comparisons:
        s1 = group_stats.get(g1, {})
        s2 = group_stats.get(g2, {})
        row = [label]
        for key, _ in metrics_keys:
            v1 = s1.get(key)
            v2 = s2.get(key)
            if v1 and v2:
                row.append(f"{v1[0] - v2[0]:+.4f}")
            else:
                row.append("-")
        a1 = s1.get("auroc")
        a2 = s2.get("auroc")
        b1 = s1.get("balanced_accuracy")
        b2 = s2.get("balanced_accuracy")
        if a1 and a2 and b1 and b2:
            da = a1[0] - a2[0]
            db = b1[0] - b2[0]
            if da > 0.005 and db > 0.01:
                interp = f"✅ {interp_base}: 有增益"
            elif da > 0 and db > 0:
                interp = f"⚠️ {interp_base}: 微弱增益"
            else:
                interp = f"❌ {interp_base}: 无明显增益"
        else:
            interp = "数据不足"
        row.append(interp)
        lines.append("| " + " | ".join(row) + " |")

    # Table 4: Sample-level paired bootstrap results
    lines.append("\n## Table 4: Paired Bootstrap Results (sample-level, matched seeds 42-45)\n")
    lines.append("> Probabilities averaged across matched seeds per sample, aligned by "
                 "sample_id/record_id; the same test samples are resampled with replacement "
                 f"({N_BOOT} iters). n_samples = aligned test cases (expect 204).\n")
    lines.append("| comparison | metric | n_samples | n_seeds | delta | 95% CI | p-value | significant | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for br in bootstrap_results:
        ci = f"[{br['ci_lo']}, {br['ci_hi']}]" if br.get('ci_lo') else "-"
        lines.append("| " + " | ".join([
            br["comparison"], br["metric"],
            str(br.get("n_samples", "")), str(br.get("n_seeds", "")),
            str(br["delta"]), ci, str(br["p_value"]),
            str(br["significant"]), str(br.get("note", "")),
        ]) + " |")

    # Claim boundary
    lines.append("\n## Claim Boundary\n")
    if group_stats.get("D_kd_ct_cnv_text") and group_stats.get("C_kd_ct_text") and group_stats.get("A_supervised"):
        d_a = group_stats["D_kd_ct_cnv_text"].get("auroc")
        c_a = group_stats["C_kd_ct_text"].get("auroc")
        a_a = group_stats["A_supervised"].get("auroc")
        d_b = group_stats["D_kd_ct_cnv_text"].get("balanced_accuracy")
        c_b = group_stats["C_kd_ct_text"].get("balanced_accuracy")
        a_b = group_stats["A_supervised"].get("balanced_accuracy")
        if d_a and c_a and a_a and d_b and c_b and a_b:
            d_gt_c = d_a[0] > c_a[0] + 0.005 and d_b[0] > c_b[0] + 0.01
            c_gt_a = c_a[0] > a_a[0] + 0.005
            if d_gt_c and c_gt_a:
                lines.append("**Claim A**: CT+CNV+Text teacher 的特权蒸馏相较 CT+Text teacher 和 supervised baseline 带来稳定增益，支持基因信息作为训练期特权知识。")
            elif d_a[0] > a_a[0] + 0.005 and not d_gt_c:
                lines.append("**Claim B**: 蒸馏训练带来增益，但基因信息的独立贡献仍需更多证据。")
            elif c_a[0] > d_a[0]:
                lines.append("**Claim C**: KD recipe 对 teacher modality 组成敏感，CNV teacher 未表现出额外优势。")
            else:
                lines.append("**Claim D**: DenseNet3D121 CT+Text supervised 已经接近最优，蒸馏增益有限。")
            lines.append(f"\nGroup A: AUROC {a_a[0]:.4f}±{a_a[1]:.4f}, BAcc {a_b[0]:.4f}±{a_b[1]:.4f}")
            lines.append(f"Group C: AUROC {c_a[0]:.4f}±{c_a[1]:.4f}, BAcc {c_b[0]:.4f}±{c_b[1]:.4f}")
            lines.append(f"Group D: AUROC {d_a[0]:.4f}±{d_a[1]:.4f}, BAcc {d_b[0]:.4f}±{d_b[1]:.4f}")
        else:
            lines.append("数据不足，无法判定。")
    else:
        lines.append("数据不足，无法判定。")

    lines.append("\n## DeLong AUROC Test\n")
    lines.append("TODO: DeLong test requires roc_auc implementation with covariance matrix.")
    lines.append("Currently using paired bootstrap as primary statistical test.")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bootstrap_csv(bootstrap_results: list[dict[str, Any]], out: Path) -> None:
    fields = ["comparison", "metric", "n_samples", "n_seeds", "delta", "ci_lo", "ci_hi", "p_value", "significant", "note"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in bootstrap_results:
            w.writerow({k: r.get(k, "") for k in fields})


def write_delong_csv(out: Path) -> None:
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "z_stat", "p_value", "note"])
        w.writerow(["TODO", "auroc", "", "", "DeLong test not yet implemented; use paired bootstrap instead"])


# ====================== Main ======================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0531_gene_privileged_ablation"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    discovered = discover(root)
    records = []
    pred_index: dict[str, dict[str, Any]] = {}
    for d, g in discovered:
        rec = analyse(d, g)
        records.append(rec)
        pred_index[rec["run_name"]] = load_predictions(d) or {"id_col": None, "rows": {}, "warning": "no predictions"}

    csv_path = root / "p0_gene_privileged_metrics.csv"
    md_path = root / "p0_gene_privileged_metrics.md"
    summary_path = root / "p0_gene_privileged_summary.md"
    bootstrap_path = root / "p0_paired_bootstrap.csv"
    delong_path = root / "p0_delong_auroc.csv"

    write_csv(records, csv_path)
    write_metrics_md(records, md_path)

    bootstrap_results = run_bootstrap_comparisons(records, pred_index)
    write_bootstrap_csv(bootstrap_results, bootstrap_path)
    write_delong_csv(delong_path)
    write_summary(records, bootstrap_results, summary_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {bootstrap_path}")
    print(f"Wrote {delong_path}")
    print(f"Runs analysed: {len(records)}")
    for g in sorted(set(r["group"] for r in records)):
        n = sum(1 for r in records if r["group"] == g)
        print(f"  {g}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
