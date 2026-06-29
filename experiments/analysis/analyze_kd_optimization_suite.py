#!/usr/bin/env python3
"""Analyze CT+Text student KD optimization suite results.

This script only reads existing experiment outputs and never launches training.

Outputs (all written under suite root):
  - kd_suite_run_records.csv / kd_suite_run_records.md
  - kd_suite_bootstrap_vs_references.csv
  - kd_suite_threshold_calibration.md
  - kd_suite_threshold_calibration.csv
  - kd_suite_candidate_summary.md
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
import re


EXPECTED_SPLIT = (1019, 652, 163, 204)
N_BOOT_DEFAULT = 10000
BOOT_SEED = 42
STAGE_DIRS = {
    "alpha_sweep",
    "temperature_sweep",
    "optimizer_sweep",
    "batch_size_sweep",
    "light_combo_variants",
    "calibration_kd",
    "confidence_weighted_kd",
}
METRICS = ["auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score", "nll"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "balanced_accuracy": "BAcc",
    "f1": "F1",
    "recall": "Recall",
    "specificity": "Specificity",
    "ece": "ECE",
    "brier_score": "Brier",
    "nll": "NLL",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isfinite(x):
            return x
        return None
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return None


def _mean_std(values: list[float]) -> tuple[float, float] | None:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def _fmt(v: Any, d: int = 4) -> str:
    if v is None or v == "" or v is False or v is True:
        if v is False:
            return str(v)
        if v is True:
            return str(v)
        if v == "":
            return "-"
        if v is None:
            return "-"
        return str(v)
    if isinstance(v, (float, int)):
        if isinstance(v, float) and math.isnan(v):
            return "-"
        return f"{v:.{d}f}"
    return str(v)


def _fmt_ms(mean: float | None, std: float | None, d: int = 4) -> str:
    if mean is None:
        return "-"
    if std is None or std == 0:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f}±{std:.{d}f}"


# ====================== Parsing and discovery ======================


def parse_seed(run_name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", run_name)
    if not m:
        return None
    return int(m.group(1))


def classify_run(name: str) -> str:
    if name.startswith("S0_supervised"):
        return "S0_supervised"
    if name.startswith("S1_logits_alpha02_T4"):
        return "S1_logits_alpha02"
    if name.startswith("S2_light_logits_fused_ct_text"):
        return "S2_light_logits_fused_ct_text"
    if name.startswith("S2_light_logits_fused"):
        return "S2_light_logits_fused"
    if name.startswith("S2_light_"):
        return "S2_light_other"
    if name.startswith("S2_alpha"):
        return "S2_optimizer"
    if "calibT" in name:
        return "S2_calibration"
    if "confsoft" in name or "confhard" in name:
        return "S2_confidence_weighted"
    if "_bs" in name:
        return "S2_batch"
    if name.startswith("S2_logits"):
        return "S2_logits"
    return "S2_other"


def discover_stage_runs(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    if not root.is_dir():
        return out
    for stage in sorted(STAGE_DIRS):
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        for p in sorted(stage_dir.iterdir()):
            if not p.is_dir():
                continue
            if p.name.startswith("."):
                continue
            out.append((p, stage))
    return out


# ====================== Split / config checks ======================


def parse_split_manifest(path: Path) -> tuple[int, tuple[int, int, int, int] | None]:
    if not path.is_file():
        return 0, None
    total = 0
    cnt = {"train": 0, "val": 0, "test": 0}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            lines = f.read().splitlines()
            if not lines:
                return 0, None
            header = lines[0].split(",")
            fields = [h.strip() for h in header]
            col = None
            for c in ("assigned_split", "split", "Split"):
                if c in fields:
                    col = c
                    break
            if col is None:
                return 0, None
            idx = fields.index(col)
            for line in lines[1:]:
                if not line:
                    continue
                parts = line.split(",")
                if idx >= len(parts):
                    continue
                v = parts[idx].strip().lower()
                total += 1
                if v in cnt:
                    cnt[v] += 1
        return total, (total, cnt["train"], cnt["val"], cnt["test"])
    except OSError:
        return 0, None


def _has_text_num_features(m: dict[str, Any] | None) -> str:
    if not m:
        return ""
    cfg = m.get("config") or {}
    if cfg.get("disable_text_numeric_features") is True:
        return "0"
    if isinstance(cfg.get("allowed_numeric_cols"), str):
        v = cfg.get("allowed_numeric_cols").strip()
        if not v or v in {"none", "null", ""}:
            return "0"
        return str(len(v.split(",")))
    return ""


def check_run_flags(run_dir: Path, record: dict[str, Any], warnings: list[str]) -> None:
    manifest_path = run_dir / "split_manifest.csv"
    total, split = parse_split_manifest(manifest_path)
    if split is None:
        record["split_status"] = "MISSING_SPLIT"
        warnings.append("split_manifest_missing")
    else:
        record["split_status"] = "OK" if split == EXPECTED_SPLIT else "SPLIT_MISMATCH"
        if split != EXPECTED_SPLIT:
            warnings.append("split_mismatch")
    record["split_total"] = total

    m = _read_json(run_dir / "metrics.json")
    cfg = (m or {}).get("config", {}) if m else {}
    record["split_mode"] = cfg.get("split_mode", "")
    record["strict_no_leakage"] = bool(cfg.get("strict_no_leakage", False))
    if not record["strict_no_leakage"]:
        warnings.append("strict_no_leakage_false")
    record["disable_text_numeric_features"] = bool(cfg.get("disable_text_numeric_features", False))
    if not record["disable_text_numeric_features"]:
        warnings.append("disable_text_numeric_features_false")
    record["text_num_features"] = _has_text_num_features(m)
    if record["text_num_features"] and record["text_num_features"] != "0":
        warnings.append("text_num_features_used")
    record["ct_model"] = cfg.get("ct_model", "")
    if record["ct_model"] and record["ct_model"] not in {"densenet3d_121", "densenet3d121"}:
        warnings.append("ct_model_not_densenet3d_121")
    record["batch_size"] = cfg.get("batch_size", "")
    methods = cfg.get("distill_methods", [])
    if isinstance(methods, list):
        record["distill_methods"] = ",".join(methods)
    else:
        record["distill_methods"] = str(methods)
    if "hint" in record["distill_methods"]:
        warnings.append("forbidden_hint")
    teacher_dir = cfg.get("teacher_run_dir", "")
    record["teacher_run_dir"] = teacher_dir


# ====================== Prediction loading / metrics ======================


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


def load_predictions_csv(path: Path) -> dict[str, tuple[int, float]] | None:
    if not path.is_file():
        return None
    rows: dict[str, tuple[int, float]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            id_col = next((c for c in ID_COLUMNS if c in fields), None)
            lab_col = next((c for c in LABEL_COLUMNS if c in fields), None)
            prob_col = next((c for c in PROB_COLUMNS if c in fields), None)
            if id_col is None or lab_col is None or prob_col is None:
                return None
            for row in reader:
                sid = (row.get(id_col) or "").strip()
                if not sid:
                    continue
                try:
                    lab = int(float(row.get(lab_col)))
                    prob = float(row.get(prob_col))
                    rows[sid] = (lab, max(0.0, min(1.0, prob)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return None
    return rows


def _auroc(labels: list[int], probs: list[float]) -> float:
    pos = [(p, 1) for p, l in zip(probs, labels) if l == 1]
    neg = [(p, 0) for p, l in zip(probs, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    concordant = 0.0
    for pp, _ in pos:
        for pn, _ in neg:
            if pp > pn:
                concordant += 1.0
            elif pp == pn:
                concordant += 0.5
    return concordant / (len(pos) * len(neg))


def _ece(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for l, p in zip(labels, probs):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((l, p))
    total = len(labels)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = statistics.mean(v[1] for v in b)
        avg_l = statistics.mean(v[0] for v in b)
        ece += len(b) / total * abs(avg_p - avg_l)
    return float(ece)


def _nll(labels: list[int], probs: list[float], eps: float = 1e-7) -> float:
    return -statistics.mean(
        lab * math.log(max(min(p, 1.0 - eps), eps)) + (1 - lab) * math.log(max(1.0 - p, eps))
        for lab, p in zip(labels, probs)
    )


def _compute_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 1)
    fp = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 1)
    fn = sum(1 for l, pr in zip(labels, preds) if l == 1 and pr == 0)
    tn = sum(1 for l, pr in zip(labels, preds) if l == 0 and pr == 0)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    bacc = (recall + spec) / 2.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "auroc": _auroc(labels, probs),
        "balanced_accuracy": bacc,
        "f1": f1,
        "recall": recall,
        "specificity": spec,
        "ece": _ece(labels, probs),
        "brier_score": statistics.mean((p - l) ** 2 for l, p in zip(labels, probs)) if labels else float("nan"),
        "nll": _nll(labels, probs),
        "accuracy": accuracy,
    }


def _to_float_map(values: dict[str, Any]) -> dict[str, float | None]:
    return {k: _safe_float(v) for k, v in values.items()}


def compute_run_record(run_dir: Path, stage: str) -> dict[str, Any]:
    name = run_dir.name
    seed = parse_seed(name)
    m = _read_json(run_dir / "metrics.json")
    cfg = (m or {}).get("config", {})
    tm = (m or {}).get("test_metrics", {}) if m else {}
    vm = (m or {}).get("val_metrics", {}) if m else {}

    record: dict[str, Any] = {
        "run_name": name,
        "stage": stage,
        "seed": seed,
        "group": classify_run(name),
        "split_status": "MISSING",
        "split_total": "",
        "split_mode": "",
        "strict_no_leakage": "",
        "disable_text_numeric_features": "",
        "text_num_features": "",
        "ct_model": "",
        "batch_size": "",
        "distill_methods": "",
        "distillation_alpha": cfg.get("distillation_alpha", ""),
        "distillation_temperature": cfg.get("distillation_temperature", ""),
        "teacher_run_dir": cfg.get("teacher_run_dir", ""),
        "best_epoch": (m or {}).get("best_epoch", ""),
        "warnings": "",
    }
    check_run_flags(run_dir, record, _tmp := [])
    record["warnings"] = ";".join(_tmp) if _tmp else ""

    # copy primary metrics
    metrics = _to_float_map(tm or {})
    for k in METRICS:
        key = {
            "brier_score": "brier_score",
            "nll": "nll",
        }.get(k, k)
        record[k] = metrics.get(key)

    # fallback from predictions if missing any metric
    if any(record[k] is None for k in METRICS):
        pred = load_predictions_csv(run_dir / "test_predictions.csv")
        if pred:
            labels = [v[0] for _, v in sorted(pred.items(), key=lambda x: x[0])]
            probs = [v[1] for _, v in sorted(pred.items(), key=lambda x: x[0])]
            calced = _compute_metrics(labels, probs)
            for k in METRICS:
                if record.get(k) is None:
                    record[k] = calced.get(k)
            record["test_pred_rows"] = len(pred)
        else:
            record["test_pred_rows"] = "-"
    else:
        # still expose test sample count if available
        pred = load_predictions_csv(run_dir / "test_predictions.csv")
        record["test_pred_rows"] = len(pred) if pred is not None else ""

    # keep val metrics if present for quick inspection
    val_metrics = _to_float_map(vm or {})
    for k, v in val_metrics.items():
        record[f"val_{k}"] = v
    return record


# ====================== Paired bootstrap ======================


def load_prediction_index(runs: list[tuple[Path, str]]) -> dict[str, dict[str, tuple[int, float]]]:
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for p, _ in runs:
        rows = load_predictions_csv(p / "test_predictions.csv")
        if rows is None:
            continue
        out[p.name] = rows
    return out


def paired_bootstrap(
    run_a: dict[str, tuple[int, float]],
    run_b: dict[str, tuple[int, float]],
    n_boot: int,
    seed: int = BOOT_SEED,
) -> dict[str, tuple[float, float, float, float, float, str]]:
    """
    return:
      metric -> (mean_delta, ci_lo, ci_hi, p_two_sided, n, note)
    """
    common = set(run_a) & set(run_b)
    if not common:
        return {k: (float("nan"), float("nan"), float("nan"), 1.0, 0, "NO_COMMON_IDS") for k in METRICS}

    ids = sorted(common)
    # labels should match; prefer from run_a
    labels = [run_a[sid][0] for sid in ids]
    probs_a = [run_a[sid][1] for sid in ids]
    probs_b = [run_b[sid][1] for sid in ids]
    n = len(ids)
    if n == 0:
        return {k: (float("nan"), float("nan"), float("nan"), 1.0, 0, "NO_COMMON_IDS") for k in METRICS}

    m1 = _compute_metrics(labels, probs_a)
    m2 = _compute_metrics(labels, probs_b)
    observed = {k: m1[k] - m2[k] for k in METRICS}

    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {k: [] for k in METRICS}
    idxs = list(range(n))
    for _ in range(max(1, int(n_boot))):
        sample_idx = [rng.choice(idxs) for _ in range(n)]
        sample_labels = [labels[i] for i in sample_idx]
        sample_a = [probs_a[i] for i in sample_idx]
        sample_b = [probs_b[i] for i in sample_idx]
        sa = _compute_metrics(sample_labels, sample_a)
        sb = _compute_metrics(sample_labels, sample_b)
        for k in METRICS:
            deltas[k].append(sa[k] - sb[k])

    out: dict[str, tuple[float, float, float, float, int, str]] = {}
    for k in METRICS:
        d = deltas[k]
        d.sort()
        ci_lo = d[int(0.025 * (len(d) - 1))]
        ci_hi = d[int(0.975 * (len(d) - 1))]
        obs = observed[k]
        if obs >= 0:
            p = sum(1 for x in d if x <= -abs(obs)) / len(d) * 2
        else:
            p = sum(1 for x in d if x >= abs(obs)) / len(d) * 2
        p = min(1.0, max(0.0, p))
        out[k] = (obs, ci_lo, ci_hi, p, n, "")
    return out


# ====================== Threshold + calibration ======================


def metric_rows_at_threshold(labels: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    return _compute_metrics(labels, probs, threshold=threshold)


def best_thresholds(labels: list[int], probs: list[float]) -> dict[str, float]:
    best_bacc_t, best_f1_t, best_you_t = 0.5, 0.5, 0.5
    best_bacc, best_f1, best_you = -1e9, -1e9, -1e9
    for t_int in range(1, 100):
        t = t_int / 100.0
        m = metric_rows_at_threshold(labels, probs, t)
        youden = m["recall"] + m["specificity"] - 1.0
        if m["balanced_accuracy"] > best_bacc:
            best_bacc = m["balanced_accuracy"]
            best_bacc_t = t
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_t = t
        if youden > best_you:
            best_you = youden
            best_you_t = t
    return {
        "best_bacc_threshold": best_bacc_t,
        "best_f1_threshold": best_f1_t,
        "best_youden_threshold": best_you_t,
    }


def fit_temperature_scale(labels: list[int], probs: list[float]) -> float:
    """
    Fit temperature on probability-only predictions by applying temperature
    scaling on logits p = sigmoid(logit(p)/T).
    """
    eps = 1e-7
    best_t = 1.0
    best_nll = float("inf")
    for t_raw in range(50, 801):
        t = t_raw / 100.0  # 0.5 -> 8.0
        if t <= 0:
            continue
        scaled = []
        for p in probs:
            p = min(1.0 - eps, max(eps, p))
            z = math.log(p / (1 - p))
            z = z / t
            s = 1.0 / (1.0 + math.exp(-z))
            scaled.append(s)
        nll = _nll(labels, scaled)
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    eps = 1e-7
    out: list[float] = []
    for p in probs:
        p = min(1.0 - eps, max(eps, p))
        z = math.log(p / (1 - p)) / temperature
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def read_reference_runs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for p in root.iterdir():
        if not p.is_dir():
            continue
        seed = parse_seed(p.name)
        if seed is None:
            continue
        if p.name.startswith(prefix):
            out[seed] = p
    return out


def build_reference_maps(root: Path, teacher_root: Path) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    s0_map = read_reference_runs(root, "ct_text_sc_densenet3d121_strict_bs4_seed")
    s1_map = read_reference_runs(root, "S1_logits_alpha02_T4_seed")
    t1_map = read_reference_runs(teacher_root, "densenet3d121_ct_cnv_text_teacher_strict_seed")
    return s0_map, s1_map, t1_map


# ====================== Writers ======================


def write_csv(records: list[dict[str, Any]], out_path: Path, fields: list[str]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def md_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_run_records(records: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "stage",
        "group",
        "run_name",
        "seed",
        "split_status",
        "split_total",
        "strict_no_leakage",
        "disable_text_numeric_features",
        "text_num_features",
        "ct_model",
        "batch_size",
        "distill_methods",
        "distillation_alpha",
        "distillation_temperature",
        "teacher_run_dir",
        "auroc",
        "balanced_accuracy",
        "f1",
        "recall",
        "specificity",
        "ece",
        "brier_score",
        "nll",
        "test_pred_rows",
        "warnings",
    ]
    lines = [
        "# KD Optimization Suite — Per-run Summary\n",
        f"- total runs: {len(records)}",
        "",
    ]
    table_rows = []
    for r in sorted(records, key=lambda x: (x.get("stage", ""), x.get("group", ""), str(x.get("seed", "")), x.get("run_name", ""))):
        table_rows.append([
            str(r.get("stage", "")),
            str(r.get("group", "")),
            str(r.get("run_name", "")),
            str(r.get("seed", "")),
            str(r.get("split_status", "")),
            str(r.get("split_total", "")),
            str(r.get("strict_no_leakage", "")),
            str(r.get("disable_text_numeric_features", "")),
            str(r.get("text_num_features", "")),
            str(r.get("ct_model", "")),
            str(r.get("batch_size", "")),
            str(r.get("distill_methods", "")),
            _fmt(r.get("distillation_alpha")),
            _fmt(r.get("distillation_temperature")),
            str(r.get("teacher_run_dir", "")),
            _fmt(r.get("auroc")),
            _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("f1")),
            _fmt(r.get("recall")),
            _fmt(r.get("specificity")),
            _fmt(r.get("ece")),
            _fmt(r.get("brier_score")),
            _fmt(r.get("nll")),
            str(r.get("test_pred_rows", "")),
            str(r.get("warnings", "")),
        ])
    lines.append(md_table(table_rows, ["stage", "group", "run", "seed", "split", "split_total", "strict", "no_leak", "text_num", "ct_model", "bs", "methods", "alpha", "T", "teacher", "AUROC", "BAcc", "F1", "Recall", "Spec", "ECE", "Brier", "NLL", "rows", "warns"]))
    lines.append("")

    # group summary
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[r.get("group", "unknown")].append(r)
    lines.append("## Group Mean ± Std (matched seeds)\n")
    lines.append("| group | n | AUROC | BAcc | F1 | Recall | Specificity | ECE | Brier | NLL |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(by_group):
        vals = by_group[g]
        lines.append(
            "| " + " | ".join([
                g,
                str(len(vals)),
                _fmt_ms(*_mean_std([x for x in (v.get("auroc") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("balanced_accuracy") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("f1") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("recall") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("specificity") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("ece") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("brier_score") for v in vals) if isinstance(x, (int, float))])),
                _fmt_ms(*_mean_std([x for x in (v.get("nll") for v in vals) if isinstance(x, (int, float))])),
            ]) + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # keep raw csv too
    write_csv(records, out.with_suffix(".csv"), fields=fields)


def write_bootstrap_results(
    bootstrap_rows: list[dict[str, Any]],
    out_csv: Path,
    out_md: Path,
) -> None:
    fields = ["run_name", "stage", "seed", "candidate_group", "comparison",
              "metric", "n_samples", "delta", "ci_lo", "ci_hi", "p_value", "significant", "note"]
    write_csv(bootstrap_rows, out_csv, fields)

    lines = [
        "# Paired Bootstrap vs References\n",
        "| run | seed | comparison | metric | delta | 95% CI | p-value | sig | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(bootstrap_rows, key=lambda x: (x.get("run_name", ""), str(x.get("seed", "")), x.get("comparison", ""), x.get("metric", ""))):
        ci = f"[{r.get('ci_lo')}, {r.get('ci_hi')}]" if r.get("ci_lo") != "" else "-"
        lines.append(
            "| " + " | ".join([
                str(r.get("run_name", "")),
                str(r.get("seed", "")),
                str(r.get("comparison", "")),
                str(r.get("metric", "")),
                _fmt(r.get("delta")),
                ci,
                _fmt(r.get("p_value"), 4),
                str(r.get("significant", "")),
                str(r.get("n_samples", "")),
            ]) + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_calibration(records: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    fields = [
        "run_name",
        "stage",
        "seed",
        "group",
        "split_available",
        "bacc_0.5",
        "f1_0.5",
        "spec_0.5",
        "best_bacc_t",
        "bacc_best_t",
        "best_f1_t",
        "f1_best_t",
        "best_youden_t",
        "youden_bacc",
        "ece_before",
        "ece_after",
        "nll_before",
        "nll_after",
        "temp",
    ]
    rows = []
    lines = [
        "# Threshold and Temperature Calibration",
        "",
        "| run | seed | group | split_ok | BAcc@0.5 | F1@0.5 | Spec@0.5 | best BAcc th | BAcc* | best F1 th | F1* | best Youden th | YoudenBAcc | ECE before | ECE after | NLL before | NLL after | Tfit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in sorted(records, key=lambda x: (x.get("stage", ""), x.get("group", ""), str(x.get("seed", "")), x.get("run_name", ""))):
        rows.append(rec)
        lines.append(
            "| " + " | ".join([
                str(rec.get("run_name", "")),
                str(rec.get("seed", "")),
                str(rec.get("group", "")),
                str(rec.get("split_available", False)),
                _fmt(rec.get("bacc_0.5")),
                _fmt(rec.get("f1_0.5")),
                _fmt(rec.get("spec_0.5")),
                _fmt(rec.get("best_bacc_threshold")),
                _fmt(rec.get("bacc_best_t")),
                _fmt(rec.get("best_f1_threshold")),
                _fmt(rec.get("f1_best_t")),
                _fmt(rec.get("best_youden_threshold")),
                _fmt(rec.get("youden_best")),
                _fmt(rec.get("ece_before")),
                _fmt(rec.get("ece_after")),
                _fmt(rec.get("nll_before")),
                _fmt(rec.get("nll_after")),
                _fmt(rec.get("temperature")),
            ]) + " |"
        )
    write_csv(rows, out_csv, fields=fields)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_summary(
    records: list[dict[str, Any]],
    out_md: Path,
) -> None:
    candidate_groups = sorted({r.get("group") for r in records if r.get("group", "").startswith("S2")})
    lines = [
        "# KD Optimization Suite — Candidate Summary\n",
        "",
    ]
    by_group = defaultdict(list)
    for r in records:
        by_group[r.get("group", "unknown")].append(r)
    lines.append("## Candidate family means (4 seeds preferred)\n")
    lines.append("| family | n | AUROC | BAcc | F1 | Recall | Spec | ECE | Brier | NLL |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(candidate_groups):
        vals = by_group.get(g, [])
        lines.append(
            "| " + " | ".join([
                g,
                str(len(vals)),
                _fmt_ms(*_mean_std([v["auroc"] for v in vals if isinstance(v.get("auroc"), (int, float))])),
                _fmt_ms(*_mean_std([v["balanced_accuracy"] for v in vals if isinstance(v.get("balanced_accuracy"), (int, float))])),
                _fmt_ms(*_mean_std([v["f1"] for v in vals if isinstance(v.get("f1"), (int, float))])),
                _fmt_ms(*_mean_std([v["recall"] for v in vals if isinstance(v.get("recall"), (int, float))])),
                _fmt_ms(*_mean_std([v["specificity"] for v in vals if isinstance(v.get("specificity"), (int, float))])),
                _fmt_ms(*_mean_std([v["ece"] for v in vals if isinstance(v.get("ece"), (int, float))])),
                _fmt_ms(*_mean_std([v["brier_score"] for v in vals if isinstance(v.get("brier_score"), (int, float))])),
                _fmt_ms(*_mean_std([v["nll"] for v in vals if isinstance(v.get("nll"), (int, float))])),
            ]) + " |"
        )
    lines.append("")
    lines.append("## Baseline comparison anchors\n")
    lines.append("- S0 baseline: CT+Text supervised strict")
    lines.append("- S1 baseline: S1_logits_alpha02_T4")
    lines.append("- Best teacher: teacher_root / densenet3d121_ct_cnv_text_teacher_strict_seed{42,43,44,45}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ====================== Main ======================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0533_kd_optimization_suite"), help="Suite root")
    p.add_argument("--baseline-root", type=Path, default=Path("outputs0531_gene_privileged_ablation"), help="Baseline results for S0/S1")
    p.add_argument("--teacher-root", type=Path, default=Path("outputs0531_teacher_homogeneous_gene_test"), help="Teacher results root")
    p.add_argument("--bootstrap-iters", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--expected-split", type=int, nargs=4, default=list(EXPECTED_SPLIT), help="expected (total, train, val, test)")
    p.add_argument("--seed-cutoff", type=int, default=0, help="only analyse seeds >= cutoff? keep 0 for all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    teacher_root = args.teacher_root.expanduser().resolve()

    global EXPECTED_SPLIT
    if len(args.expected_split) == 4:
        EXPECTED_SPLIT = tuple(args.expected_split)  # type: ignore[assignment]

    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    run_records_raw = discover_stage_runs(root)
    if not run_records_raw:
        print(f"[WARN] No run directories found under {root}. Nothing to analyze.")
        return 0

    records = [compute_run_record(p, stage) for p, stage in run_records_raw]
    records.sort(key=lambda x: (x.get("seed", 0), x.get("stage", ""), x.get("run_name", "")))

    # filter by seed cutoff if requested
    if args.seed_cutoff > 0:
        records = [r for r in records if (r.get("seed") or 0) >= args.seed_cutoff]

    write_markdown_run_records(records, root / "kd_suite_run_records.md")

    # paired bootstrap references
    s0_map, s1_map, t1_map = build_reference_maps(baseline_root, teacher_root)
    all_pred = {Path(p).name: load_predictions_csv((Path(p) / "test_predictions.csv")) for p, _ in run_records_raw}
    all_pred = {k: v for k, v in all_pred.items() if v}

    bootstrap_rows: list[dict[str, Any]] = []
    candidate_records = [r for r in records if r.get("group", "").startswith("S2")]
    for r in candidate_records:
        seed = r.get("seed")
        if seed is None:
            continue
        run_name = str(r.get("run_name"))
        if run_name not in all_pred:
            continue

        for comp_name, ref_map in [
            ("S0_supervised", s0_map),
            ("S1_logits_alpha02", s1_map),
            ("T1_teacher", t1_map),
        ]:
            ref_run = ref_map.get(seed)
            if ref_run is None:
                continue
            if ref_run.name not in all_pred:
                # candidate in different tree; try load reference predictions lazily and keep
                ref_rows = load_predictions_csv(ref_run / "test_predictions.csv")
                if ref_rows is None:
                    continue
                all_pred[ref_run.name] = ref_rows
            ref_name = ref_run.name
            if ref_name not in all_pred:
                continue
            boot = paired_bootstrap(all_pred[run_name], all_pred[ref_name], args.bootstrap_iters, BOOT_SEED)
            for metric in METRICS:
                delta, ci_lo, ci_hi, p_val, n, note = boot[metric]
                bootstrap_rows.append({
                    "run_name": run_name,
                    "stage": r.get("stage"),
                    "seed": seed,
                    "candidate_group": r.get("group"),
                    "comparison": comp_name,
                    "metric": metric,
                    "n_samples": n,
                    "delta": delta,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": p_val,
                    "significant": "YES" if p_val < 0.05 else "NO",
                    "note": note,
                })

    write_bootstrap_results(bootstrap_rows, root / "kd_suite_bootstrap_vs_references.csv", root / "kd_suite_bootstrap_vs_references.md")

    # threshold + calibration analysis
    threshold_rows: list[dict[str, Any]] = []
    for r in records:
        run_name = str(r.get("run_name"))
        p = root / r.get("stage", "") / run_name
        if not p.is_dir():
            continue
        val = load_predictions_csv(p / "val_predictions.csv")
        test = load_predictions_csv(p / "test_predictions.csv")
        if not val or not test:
            threshold_rows.append({
                "run_name": run_name,
                "stage": r.get("stage"),
                "seed": r.get("seed"),
                "group": r.get("group"),
                "split_available": False,
            })
            continue

        val_labels = [x[0] for sid, x in sorted(val.items(), key=lambda z: z[0])]
        val_probs = [x[1] for sid, x in sorted(val.items(), key=lambda z: z[0])]
        test_labels = [x[0] for sid, x in sorted(test.items(), key=lambda z: z[0])]
        test_probs = [x[1] for sid, x in sorted(test.items(), key=lambda z: z[0])]

        default = metric_rows_at_threshold(test_labels, test_probs, 0.5)
        opt = best_thresholds(val_labels, val_probs)
        bacc_best = metric_rows_at_threshold(test_labels, test_probs, opt["best_bacc_threshold"])
        f1_best = metric_rows_at_threshold(test_labels, test_probs, opt["best_f1_threshold"])
        you_best = metric_rows_at_threshold(test_labels, test_probs, opt["best_youden_threshold"])

        temp = fit_temperature_scale(val_labels, val_probs)
        calib_probs = apply_temperature(test_probs, temp)
        calib_metrics = _compute_metrics(test_labels, calib_probs)

        baseline_temp_metrics = _compute_metrics(test_labels, test_probs)

        threshold_rows.append({
            "run_name": run_name,
            "stage": r.get("stage"),
            "seed": r.get("seed"),
            "group": r.get("group"),
            "split_available": True,
            "bacc_0.5": default["balanced_accuracy"],
            "f1_0.5": default["f1"],
            "spec_0.5": default["specificity"],
            "best_bacc_threshold": opt["best_bacc_threshold"],
            "bacc_best_t": bacc_best["balanced_accuracy"],
            "best_f1_threshold": opt["best_f1_threshold"],
            "f1_best_t": f1_best["f1"],
            "best_youden_threshold": opt["best_youden_threshold"],
            "youden_best": you_best["balanced_accuracy"] + you_best["recall"] - 1.0,
            "ece_before": baseline_temp_metrics["ece"],
            "ece_after": calib_metrics["ece"],
            "nll_before": baseline_temp_metrics["nll"],
            "nll_after": calib_metrics["nll"],
            "temperature": temp,
        })

    write_threshold_calibration(threshold_rows, root / "kd_suite_threshold_calibration.csv", root / "kd_suite_threshold_calibration.md")
    write_candidate_summary(records, root / "kd_suite_candidate_summary.md")

    # concise console summary
    print(f"Analyzed runs: {len(records)}")
    print(f"Outputs written under: {root}")
    print(f"  - {root / 'kd_suite_run_records.md'}")
    print(f"  - {root / 'kd_suite_bootstrap_vs_references.csv'}")
    print(f"  - {root / 'kd_suite_bootstrap_vs_references.md'}")
    print(f"  - {root / 'kd_suite_threshold_calibration.csv'}")
    print(f"  - {root / 'kd_suite_threshold_calibration.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
