#!/usr/bin/env python3
"""Analyze privileged student KD optimization results.

Reads results from --root (default outputs0532_privileged_student_kd_optimization)
and comparison groups from --comparison-root (outputs0531_teacher_homogeneous_gene_test)
and --baseline-root (outputs0531_gene_privileged_ablation).

Outputs:
  privileged_student_kd_metrics.csv/md
  privileged_student_kd_summary.md
  threshold_calibration_summary.md
  sample_transfer_error_analysis.md
  paired_bootstrap_student_kd.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import warnings
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
    if not p.is_file(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return None

def _mean_std(vals: list[float]) -> tuple[float, float] | None:
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals: return None
    if len(vals) == 1: return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))

def _fmt(v: Any, d: int = 4) -> str:
    if v is None or v == "": return "-"
    if isinstance(v, float):
        if math.isnan(v): return "-"
        return f"{v:.{d}f}"
    return str(v)

def _fmt_ms(m: float | None, s: float | None, d: int = 4) -> str:
    if m is None: return "-"
    if s == 0 or s is None: return f"{m:.{d}f}"
    return f"{m:.{d}f}±{s:.{d}f}"


# ====================== Classification ======================

OPTIMIZATION_PATTERNS = {
    "S0_supervised": "S0_supervised_seed",
    "S1_logits_alpha02": "S1_logits_alpha02_T4_seed",
    "S1_light_full_no_hint": "S1_light_full_no_hint_alpha02_T4_seed",
    "S2_logits_alpha01": "S2_logits_alpha01_T4_seed",
    "S2_logits_alpha02": "S2_logits_alpha02_T4_seed",
    "S2_logits_alpha03": "S2_logits_alpha03_T4_seed",
    "S2_logits_alpha05": "S2_logits_alpha05_T4_seed",
    "S2_logits_alpha02_T2": "S2_logits_alpha02_T2_seed",
    "S2_logits_alpha02_T6": "S2_logits_alpha02_T6_seed",
    "S2_logits_alpha02_T8": "S2_logits_alpha02_T8_seed",
    "S2_light_logits_fused": "S2_light_logits_fused_alpha02_T4_seed",
    "S2_light_logits_fused_ct_text": "S2_light_logits_fused_ct_text_alpha02_T4_seed",
    "S2_light_full_no_hint": "S2_light_full_no_hint_alpha02_T4_seed",
}

def classify(name: str) -> str | None:
    for group, prefix in sorted(OPTIMIZATION_PATTERNS.items(), key=lambda x: -len(x[1])):
        if name.startswith(prefix):
            return group
    return None


# ====================== Metric computation ======================

def _auroc(labels: list[int], probs: list[float]) -> float:
    pos = [(p, 1) for p, l in zip(probs, labels) if l == 1]
    neg = [(p, 0) for p, l in zip(probs, labels) if l == 0]
    if not pos or not neg: return float("nan")
    concordant = sum(1 for pp, _ in pos for pn, _ in neg if pp > pn) + \
                 0.5 * sum(1 for pp, _ in pos for pn, _ in neg if pp == pn)
    return concordant / (len(pos) * len(neg))

def _ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for l, p in zip(labels, probs):
        bins[min(int(p * n_bins), n_bins - 1)].append((l, p))
    total = len(labels)
    if total == 0: return float("nan")
    return sum(len(b) / total * abs(statistics.mean(p for _, p in b) - statistics.mean(l for l, _ in b))
               for b in bins if b)

def _nll(labels: list[int], probs: list[float], eps: float = 1e-7) -> float:
    return -statistics.mean(l * math.log(max(p, eps)) + (1 - l) * math.log(max(1 - p, eps))
                           for l, p in zip(labels, probs))

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
    return {"auroc": _auroc(labels, probs), "balanced_accuracy": bacc, "f1": f1,
            "recall": recall, "specificity": specificity, "ece": _ece(labels, probs),
            "brier_score": statistics.mean((p - l) ** 2 for p, l in zip(probs, labels)),
            "nll": _nll(labels, probs)}

def _metrics_at_threshold(labels: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 1)
    fp = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 1)
    fn = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 0)
    tn = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bacc = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"threshold": threshold, "balanced_accuracy": bacc, "f1": f1,
            "recall": recall, "specificity": specificity, "accuracy": (tp + tn) / len(labels)}


# ====================== Prediction loading ======================

def load_predictions(run_dir: Path) -> dict[str, Any] | None:
    tp = run_dir / "test_predictions.csv"
    if not tp.is_file():
        warnings.warn(f"Missing test_predictions.csv: {run_dir}")
        return None
    try:
        with tp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            id_col = None
            for c in ("sample_id", "record_id"):
                if c in (reader.fieldnames or []):
                    id_col = c; break
            if not id_col:
                warnings.warn(f"No sample_id/record_id in {tp}")
                return None
            rows = {}
            for row in reader:
                sid = row[id_col]
                label = int(row["label"])
                prob = float(row["prob_malignant"])
                rows[sid] = (label, prob)
        return {"rows": rows, "id_col": id_col}
    except Exception as e:
        warnings.warn(f"Failed to read {tp}: {e}")
        return None

def load_val_predictions(run_dir: Path) -> dict[str, Any] | None:
    vp = run_dir / "val_predictions.csv"
    if not vp.is_file(): return None
    try:
        with vp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            id_col = None
            for c in ("sample_id", "record_id"):
                if c in (reader.fieldnames or []):
                    id_col = c; break
            if not id_col: return None
            rows = {}
            for row in reader:
                sid = row[id_col]
                label = int(row["label"])
                prob = float(row["prob_malignant"])
                rows[sid] = (label, prob)
        return {"rows": rows, "id_col": id_col}
    except:
        return None


# ====================== Analyse ======================

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
        rec["distillation_alpha"] = cfg.get("distillation_alpha")
        rec["distillation_temperature"] = cfg.get("distillation_temperature")
        rec["best_epoch"] = m.get("best_epoch")
        for k in METRICS:
            rec[k] = tm.get(k)
        # Fallback: compute from predictions if missing
        if any(rec.get(k) is None for k in METRICS):
            pr = load_predictions(run_dir)
            if pr:
                labels = [v[0] for v in pr["rows"].values()]
                probs = [v[1] for v in pr["rows"].values()]
                computed = _metrics_from_subset(labels, probs)
                for k in METRICS:
                    if rec.get(k) is None:
                        rec[k] = computed.get(k)
    return rec


# ====================== Discover ======================

def discover(root: Path) -> list[tuple[Path, str]]:
    out = []
    if not root.is_dir(): return out
    skip = {"logs", "scripts_used", "__pycache__"}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in skip or p.name.endswith((".csv", ".md")): continue
        target = p.resolve()
        group = classify(p.name)
        if group:
            out.append((target if target.is_dir() else p, group))
    return out


# ====================== Bootstrap ======================

def paired_bootstrap(
    pred_index: dict[str, dict[str, tuple[int, float]]],
    g1_runs: list[dict[str, Any]], g2_runs: list[dict[str, Any]],
    n_boot: int = N_BOOT, boot_seed: int = BOOT_SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(boot_seed)
    g1_names = {r["run_name"] for r in g1_runs}
    g2_names = {r["run_name"] for r in g2_runs}
    common_ids = None
    for name in g1_names | g2_names:
        if name not in pred_index: continue
        ids = set(pred_index[name]["rows"].keys())
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids or len(common_ids) < 10:
        return [{"metric": m, "n_samples": 0, "delta": "", "ci_lo": "", "ci_hi": "",
                 "p_value": "", "significant": "", "note": "insufficient samples"} for m in METRICS]
    common_ids = sorted(common_ids)
    n = len(common_ids)

    def avg_prob(name, sid):
        return pred_index[name]["rows"][sid][1]

    results = []
    for metric in METRICS:
        g1_probs = [statistics.mean(avg_prob(r["run_name"], sid) for r in g1_runs if r["run_name"] in pred_index)
                    for sid in common_ids]
        g2_probs = [statistics.mean(avg_prob(r["run_name"], sid) for r in g2_runs if r["run_name"] in pred_index)
                    for sid in common_ids]
        labels = [pred_index[g1_runs[0]["run_name"]]["rows"][sid][0] for sid in common_ids]
        m1 = _metrics_from_subset(labels, g1_probs)
        m2 = _metrics_from_subset(labels, g2_probs)
        observed_delta = m1[metric] - m2[metric]

        deltas = []
        for _ in range(n_boot):
            idxs = [rng.randint(0, n - 1) for _ in range(n)]
            sub_l = [labels[i] for i in idxs]
            sm1 = _metrics_from_subset(sub_l, [g1_probs[i] for i in idxs])
            sm2 = _metrics_from_subset(sub_l, [g2_probs[i] for i in idxs])
            deltas.append(sm1[metric] - sm2[metric])
        deltas.sort()
        ci_lo = deltas[int(0.025 * n_boot)]
        ci_hi = deltas[int(0.975 * n_boot)]
        if observed_delta >= 0:
            p = min(sum(1 for d in deltas if d <= -abs(observed_delta)) / n_boot * 2, 1.0)
        else:
            p = min(sum(1 for d in deltas if d >= abs(observed_delta)) / n_boot * 2, 1.0)
        results.append({"metric": metric, "n_samples": n,
                        "delta": f"{observed_delta:+.4f}", "ci_lo": f"{ci_lo:+.4f}", "ci_hi": f"{ci_hi:+.4f}",
                        "p_value": f"{p:.4f}", "significant": "YES" if p < 0.05 else "NO", "note": ""})
    return results


# ====================== Threshold optimization ======================

def optimize_threshold(val_labels: list[int], val_probs: list[float]) -> dict[str, float]:
    """Find optimal thresholds on validation set."""
    best_bacc_t, best_f1_t, best_youden_t = 0.5, 0.5, 0.5
    best_bacc, best_f1, best_youden = 0, 0, 0
    for t_int in range(5, 96):
        t = t_int / 100.0
        m = _metrics_at_threshold(val_labels, val_probs, t)
        youden = m["recall"] + m["specificity"] - 1
        if m["balanced_accuracy"] > best_bacc:
            best_bacc = m["balanced_accuracy"]; best_bacc_t = t
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_f1_t = t
        if youden > best_youden:
            best_youden = youden; best_youden_t = t
    return {"best_bacc_threshold": best_bacc_t, "best_f1_threshold": best_f1_t,
            "best_youden_threshold": best_youden_t}


def temperature_scale(val_labels: list[int], val_probs: list[float]) -> float:
    """Find optimal temperature on validation set using grid search."""
    best_t, best_nll = 1.0, float("inf")
    for t_int in range(5, 201):
        t = t_int / 50.0
        scaled = [min(max(p / t, 1e-7), 1 - 1e-7) for p in val_probs]
        nll = _nll(val_labels, scaled)
        if nll < best_nll:
            best_nll = nll; best_t = t
    return best_t


# ====================== Sample transfer analysis ======================

def sample_transfer_analysis(
    pred_index: dict[str, dict[str, tuple[int, float]]],
    t0_name: str, t1_name: str, s0_name: str, s2_name: str,
) -> dict[str, Any]:
    """Analyze which samples T1 corrects vs T0, and whether S2 inherits."""
    common = None
    for name in [t0_name, t1_name, s0_name, s2_name]:
        if name not in pred_index: return {"error": f"missing {name}"}
        ids = set(pred_index[name]["rows"].keys())
        common = ids if common is None else common & ids
    if not common: return {"error": "no common samples"}

    results = {"total": len(common), "t1_fixes_t0": 0, "s2_fixes_s0": 0,
               "s2_inherits_t1_fix": 0, "s2_new_errors": 0,
               "sample_details": []}

    for sid in sorted(common):
        t0_l, t0_p = pred_index[t0_name]["rows"][sid]
        t1_l, t1_p = pred_index[t1_name]["rows"][sid]
        s0_l, s0_p = pred_index[s0_name]["rows"][sid]
        s2_l, s2_p = pred_index[s2_name]["rows"][sid]
        label = t0_l  # all should be same

        t0_ok = (1 if t0_p >= 0.5 else 0) == label
        t1_ok = (1 if t1_p >= 0.5 else 0) == label
        s0_ok = (1 if s0_p >= 0.5 else 0) == label
        s2_ok = (1 if s2_p >= 0.5 else 0) == label

        t1_fixes_t0 = (not t0_ok) and t1_ok
        s2_fixes_s0 = (not s0_ok) and s2_ok
        s2_inherits = t1_fixes_t0 and s2_fixes_s0
        s2_new_err = s0_ok and (not s2_ok)

        if t1_fixes_t0: results["t1_fixes_t0"] += 1
        if s2_fixes_s0: results["s2_fixes_s0"] += 1
        if s2_inherits: results["s2_inherits_t1_fix"] += 1
        if s2_new_err: results["s2_new_errors"] += 1

    return results


# ====================== Writers ======================

def write_csv(records: list[dict[str, Any]], out: Path, fields: list[str]) -> None:
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})

def write_markdown_table(records: list[dict[str, Any]], out: Path, title: str,
                         group_col: str = "group") -> None:
    lines = [f"# {title}\n"]
    lines.append("| group | seed | alpha | T | teacher | AUROC | BAcc | F1 | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: (x.get(group_col, ""), x.get("seed") or 0)):
        teacher = str(r.get("teacher_run_dir", ""))[-30:]
        lines.append(f"| {r.get(group_col,'')} | {r.get('seed','-')} | "
                     f"{_fmt(r.get('distillation_alpha'))} | {_fmt(r.get('distillation_temperature'))} | "
                     f"{teacher} | {_fmt(r.get('auroc'))} | {_fmt(r.get('balanced_accuracy'))} | "
                     f"{_fmt(r.get('f1'))} | {_fmt(r.get('ece'))} | {_fmt(r.get('brier_score'))} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    p.add_argument("--root", type=Path, default=Path("outputs0532_privileged_student_kd_optimization"))
    p.add_argument("--comparison-root", type=Path, default=Path("outputs0531_teacher_homogeneous_gene_test"))
    p.add_argument("--baseline-root", type=Path, default=Path("outputs0531_gene_privileged_ablation"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    # Discover optimization runs
    discovered = discover(root)
    records = [analyse(d, g) for d, g in discovered]

    # Build pred_index
    pred_index: dict[str, dict[str, tuple[int, float]]] = {}
    for d, g in discovered:
        pr = load_predictions(d.resolve())
        if pr: pred_index[d.name] = pr

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    # Write metrics
    fields = ["group", "run_name", "seed", "ct_model", "modalities", "batch_size",
              "teacher_run_dir", "distill_methods", "distillation_alpha",
              "distillation_temperature", "best_epoch"] + METRICS
    write_csv(records, root / "privileged_student_kd_metrics.csv", fields)
    write_markdown_table(records, root / "privileged_student_kd_metrics.md",
                         "Privileged Student KD Optimization — Per-Run Metrics")

    # ==================== Summary ====================
    L = []
    a = L.append
    a("# Privileged Student KD Optimization — Summary\n")

    # Group summary
    a("## Group Summary (matched seeds 42-45)\n")
    a("| group | n | alpha | T | AUROC | BAcc | F1 | ECE | Brier |")
    a("|---|---|---|---|---|---|---|---|---|")
    group_stats = {}
    for g in sorted(by_group):
        matched = [r for r in by_group[g] if isinstance(r.get("seed"), int) and r["seed"] in MATCHED_SEEDS]
        if not matched: continue
        stats = {}
        for m in METRICS:
            vals = [r[m] for r in matched if isinstance(r.get(m), (int, float))]
            stats[m] = _mean_std(vals)
        group_stats[g] = stats
        alpha = matched[0].get("distillation_alpha", "-")
        temp = matched[0].get("distillation_temperature", "-")
        a(f"| {g} | {len(matched)} | {alpha} | {temp} | " +
          " | ".join(_fmt_ms(stats[m][0], stats[m][1]) for m in METRICS) + " |")

    # ==================== Bootstrap ====================
    bootstrap_results = []

    # Load comparison groups
    s0_runs = by_group.get("S0_supervised", [])
    s1_runs = by_group.get("S1_logits_alpha02", [])

    # Compare each S2 variant vs S0 and S1
    s2_groups = [g for g in by_group if g.startswith("S2_")]
    for s2g in sorted(s2_groups):
        s2_runs = by_group.get(s2g, [])
        if not s2_runs: continue
        for comp_name, ref_runs in [("S0_supervised", s0_runs), ("S1_logits_alpha02", s1_runs)]:
            if not ref_runs: continue
            boot = paired_bootstrap(pred_index, s2_runs, ref_runs)
            for b in boot:
                b["comparison"] = f"{s2g} - {comp_name}"
            bootstrap_results.extend(boot)

    write_bootstrap_csv(bootstrap_results, root / "paired_bootstrap_student_kd.csv")

    # ==================== Threshold optimization ====================
    threshold_lines = ["# Threshold Optimization Summary\n"]
    threshold_lines.append("For each run, optimal thresholds found on validation set.\n")
    threshold_lines.append("| run | group | seed | default_0.5_BAcc | default_0.5_F1 | "
                          "best_BAcc_th | best_BAcc | best_F1_th | best_F1 | best_Youden_th |")
    threshold_lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for d, g in discovered:
        target = d.resolve()
        val_pr = load_val_predictions(target)
        test_pr = load_predictions(target)
        if not val_pr or not test_pr: continue

        val_labels = [v[0] for v in val_pr["rows"].values()]
        val_probs = [v[1] for v in val_pr["rows"].values()]
        test_labels = [v[0] for v in test_pr["rows"].values()]
        test_probs = [v[1] for v in test_pr["rows"].values()]

        # Default 0.5
        default_m = _metrics_at_threshold(test_labels, test_probs, 0.5)
        # Optimize on val
        opt = optimize_threshold(val_labels, val_probs)
        # Apply to test
        bacc_m = _metrics_at_threshold(test_labels, test_probs, opt["best_bacc_threshold"])
        f1_m = _metrics_at_threshold(test_labels, test_probs, opt["best_f1_threshold"])
        youden_m = _metrics_at_threshold(test_labels, test_probs, opt["best_youden_threshold"])

        seed = _read_json(target / "metrics.json")
        seed_val = (seed.get("config", {}).get("seed", "-")) if seed else "-"

        threshold_lines.append(
            f"| {d.name} | {g} | {seed_val} | "
            f"{default_m['balanced_accuracy']:.4f} | {default_m['f1']:.4f} | "
            f"{opt['best_bacc_threshold']:.2f} | {bacc_m['balanced_accuracy']:.4f} | "
            f"{opt['best_f1_threshold']:.2f} | {f1_m['f1']:.4f} | "
            f"{opt['best_youden_threshold']:.2f} |")

    # Temperature scaling
    threshold_lines.append("\n## Temperature Scaling\n")
    threshold_lines.append("| run | group | seed | T_opt | ECE_before | ECE_after | NLL_before | NLL_after |")
    threshold_lines.append("|---|---|---|---|---|---|---|---|")

    for d, g in discovered:
        target = d.resolve()
        val_pr = load_val_predictions(target)
        test_pr = load_predictions(target)
        if not val_pr or not test_pr: continue

        val_labels = [v[0] for v in val_pr["rows"].values()]
        val_probs = [v[1] for v in val_pr["rows"].values()]
        test_labels = [v[0] for v in test_pr["rows"].values()]
        test_probs = [v[1] for v in test_pr["rows"].values()]

        T = temperature_scale(val_labels, val_probs)
        scaled_test = [min(max(p / T, 1e-7), 1 - 1e-7) for p in test_probs]
        ece_before = _ece(test_labels, test_probs)
        ece_after = _ece(test_labels, scaled_test)
        nll_before = _nll(test_labels, test_probs)
        nll_after = _nll(test_labels, scaled_test)

        seed = _read_json(target / "metrics.json")
        seed_val = (seed.get("config", {}).get("seed", "-")) if seed else "-"

        threshold_lines.append(
            f"| {d.name} | {g} | {seed_val} | {T:.2f} | "
            f"{ece_before:.4f} | {ece_after:.4f} | {nll_before:.4f} | {nll_after:.4f} |")

    (root / "threshold_calibration_summary.md").write_text("\n".join(threshold_lines) + "\n", encoding="utf-8")

    # ==================== Sample transfer analysis ====================
    transfer_lines = ["# Sample Transfer Error Analysis\n"]

    # Find best S2 group (by AUROC)
    best_s2g = None
    best_s2_auc = 0
    for g in s2_groups:
        matched = [r for r in by_group.get(g, []) if isinstance(r.get("auroc"), (int, float)) and r.get("seed") in MATCHED_SEEDS]
        if matched:
            mean_auc = statistics.mean(r["auroc"] for r in matched)
            if mean_auc > best_s2_auc:
                best_s2_auc = mean_auc
                best_s2g = g

    if best_s2g:
        transfer_lines.append(f"## Best S2 variant: {best_s2g} (mean AUROC={best_s2_auc:.4f})\n")

        # For each seed, do transfer analysis
        transfer_lines.append("| seed | T1_fixes_T0 | S2_fixes_S0 | S2_inherits_T1 | S2_new_errors |")
        transfer_lines.append("|---|---|---|---|---|")

        for seed in sorted(MATCHED_SEEDS):
            t0_name = f"densenet3d121_ct_text_teacher_strict_seed{seed}"
            t1_name = f"densenet3d121_ct_cnv_text_teacher_strict_seed{seed}"
            s0_name = f"S0_supervised_seed{seed}"
            s2_cands = [r["run_name"] for r in by_group.get(best_s2g, []) if r.get("seed") == seed]
            s2_name = s2_cands[0] if s2_cands else ""

            if all(n in pred_index for n in [t0_name, t1_name, s0_name, s2_name]):
                result = sample_transfer_analysis(pred_index, t0_name, t1_name, s0_name, s2_name)
                transfer_lines.append(
                    f"| {seed} | {result.get('t1_fixes_t0', '-')} | {result.get('s2_fixes_s0', '-')} | "
                    f"{result.get('s2_inherits_t1_fix', '-')} | {result.get('s2_new_errors', '-')} |")
            else:
                transfer_lines.append(f"| {seed} | - | - | - | - | (missing predictions) |")

    (root / "sample_transfer_error_analysis.md").write_text("\n".join(transfer_lines) + "\n", encoding="utf-8")

    # ==================== Full summary ====================
    # Bootstrap summary
    a("\n## Bootstrap Comparisons (sample-level, 10000 iterations)\n")
    a("| comparison | metric | Δ | 95% CI | p-value | sig |")
    a("|---|---|---|---|---|---|")
    for br in bootstrap_results:
        ci = f"[{br['ci_lo']}, {br['ci_hi']}]" if br['ci_lo'] else "-"
        a(f"| {br.get('comparison','')} | {br['metric']} | {br['delta']} | {ci} | {br['p_value']} | {br['significant']} |")

    # Interpretation
    a("\n## Interpretation\n")
    a("### Alpha sweep")
    a("比较不同 alpha 下 S2 (from T1) 的 AUROC/BAcc/F1，找出最优 KD 强度。")
    a("alpha 太大可能导致 negative transfer（teacher 含 student 无法观测的 CNV 信息）。")
    a("\n### Light-combo variants")
    a("比较 logits-only vs logits+fused vs logits+fused,ct,text vs full-no-hint。")
    a("更多 distill 方法是否帮助 student 从 gene teacher 学到更多信息。")
    a("\n### T0 vs T1 control")
    a("S2 - S1 = gene teacher 额外收益。如果 S2 不优于 S1，gene teacher 优势未传递给 student。")

    (root / "privileged_student_kd_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"Wrote {root / 'privileged_student_kd_metrics.csv'}")
    print(f"Wrote {root / 'privileged_student_kd_metrics.md'}")
    print(f"Wrote {root / 'privileged_student_kd_summary.md'}")
    print(f"Wrote {root / 'threshold_calibration_summary.md'}")
    print(f"Wrote {root / 'sample_transfer_error_analysis.md'}")
    print(f"Wrote {root / 'paired_bootstrap_student_kd.csv'}")
    print(f"Runs analysed: {len(records)}")
    for g in sorted(by_group):
        print(f"  {g}: {len(by_group[g])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
