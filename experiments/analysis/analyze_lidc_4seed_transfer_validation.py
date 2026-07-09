#!/usr/bin/env python3
"""Analyze fixed 4-seed LIDC transfer validation runs.

Read-only analyzer for:
- baseline_default
- kdinit_diff_lr_01
across profile/fold/seed. Supports optional sample-level paired bootstrap when
test prediction CSVs are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ["accuracy", "auroc", "balanced_accuracy", "f1", "recall", "specificity", "ece", "brier_score"]
RANK_METRICS = ["balanced_accuracy", "f1", "specificity", "ece", "brier_score", "auroc"]
TARGET_PROFILES = ("baseline_default", "kdinit_diff_lr_01")
DISPLAY_PROFILES = {"baseline_default": "LIDC-B", "kdinit_diff_lr_01": "LIDC-KDInit"}
LABEL_COLUMNS = ("label", "y_true", "target", "gt", "label_int")
PROB_COLUMNS = ("prob_malignant", "prob_1", "prob_pos", "y_prob", "probability_malignant", "p", "score")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../outputs0540_lidc_4seed_transfer_validation"))
    p.add_argument("--bootstrap-iters", type=int, default=10000)
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


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if math.isfinite(v) else "-"
    return "-" if v is None else str(v)


def mean_std(values: list[Any]) -> str:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}±0.0000"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


def mean(values: list[Any]) -> float | None:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fold_from_dir(name: str) -> str:
    if "fold" not in name:
        return ""
    return name.rsplit("fold", 1)[-1].split("_", 1)[0]


def seed_from_dir(name: str) -> str:
    return name.rsplit("seed", 1)[-1].split("_", 1)[0] if "seed" in name else ""


def resolve_metric(tm: dict[str, Any], metric: str) -> float | None:
    return safe_float(tm.get(metric))


def ece_from_probs(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for lab, prob in zip(labels, probs):
        bins[min(int(prob * n_bins), n_bins - 1)].append((lab, prob))
    total = len(labels)
    if total == 0:
        return float("nan")
    return sum(
        len(bin_rows) / total
        * abs((sum(prob for _, prob in bin_rows) / len(bin_rows)) - (sum(lab for lab, _ in bin_rows) / len(bin_rows)))
        for bin_rows in bins
        if bin_rows
    )


def brier_from_probs(labels: list[int], probs: list[float]) -> float:
    return statistics.mean((lab - p) ** 2 for lab, p in zip(labels, probs)) if labels else float("nan")


def auroc_binary(labels: list[int], probs: list[float]) -> float | None:
    pos = [1 for y in labels if y == 1]
    if not pos or len(pos) == len(labels):
        return None
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_predictions(path: Path) -> dict[str, tuple[int, float]]:
    if not path.is_file():
        return {}
    fields = None
    rows: dict[str, tuple[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        label_col = next((c for c in LABEL_COLUMNS if c in fields), None)
        prob_col = next((c for c in PROB_COLUMNS if c in fields), None)
        if label_col is None or prob_col is None:
            return {}
        for row in reader:
            sid = str(row.get("sample_id", row.get("id", row.get("record_id", "")))).strip()
            if not sid:
                continue
            rows[sid] = (int(float(row[label_col])), max(0.0, min(1.0, float(row[prob_col]))))
    return rows


def compute_binary_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    if not labels:
        return {metric: float("nan") for metric in METRICS}
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for lab, pred in zip(labels, preds) if lab == 1 and pred == 1)
    tn = sum(1 for lab, pred in zip(labels, preds) if lab == 0 and pred == 0)
    fp = sum(1 for lab, pred in zip(labels, preds) if lab == 0 and pred == 1)
    fn = sum(1 for lab, pred in zip(labels, preds) if lab == 1 and pred == 0)
    accuracy = (tp + tn) / len(labels)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    auroc = auroc_binary(labels, probs)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc if auroc is not None else float("nan"),
        "ece": ece_from_probs(labels, probs),
        "brier_score": brier_from_probs(labels, probs),
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles_root = root / "profiles"
    if not profiles_root.is_dir():
        return rows

    for profile_dir in sorted(profiles_root.glob("*")):
        profile = profile_dir.name
        if profile not in TARGET_PROFILES:
            continue
        for fold_dir in sorted(profile_dir.glob("fold*")):
            fold = fold_from_dir(fold_dir.name)
            if not fold_dir.is_dir():
                continue
            for seed_dir in sorted(fold_dir.glob("seed*")):
                seed = seed_from_dir(seed_dir.name)
                if not seed_dir.is_dir():
                    continue
                metrics = read_json(seed_dir / "metrics.json") or {}
                tm = (metrics.get("test_metrics") or {})
                audit_json = seed_dir / "lidc_kdinit_loading_audit.json"
                pred_path = seed_dir / "test_predictions.csv"
                pred = load_predictions(pred_path)
                row: dict[str, Any] = {
                    "profile": profile,
                    "group": DISPLAY_PROFILES.get(profile, profile),
                    "fold": fold,
                    "seed": seed,
                    "run_name": f"{profile}_{fold_dir.name}_{seed_dir.name}",
                    "run_dir": str(seed_dir),
                    "status": "complete" if metrics else "missing_metrics",
                    "selection_metric": (metrics or {}).get("selection_metric", ""),
                    "best_epoch": (metrics or {}).get("best_epoch", ""),
                    "init_checkpoint": ((metrics.get("config") or {}).get("init_checkpoint", "")),
                    "init_checkpoint_prefix": ((metrics.get("config") or {}).get("init_checkpoint_prefix", "")),
                    "monitor_metric": ((metrics.get("config") or {}).get("selection_metric", "")),
                    "transfer_mode": ((metrics.get("config") or {}).get("transfer_mode", "")),
                    "seed": seed,
                    "fold": fold,
                    "audit_json": str(audit_json) if audit_json.is_file() else "",
                    "audit_ratio": "",
                    "selection_metric_value": None,
                    "prediction_path": str(pred_path) if pred_path.is_file() else "",
                }

                for metric in METRICS:
                    row[metric] = resolve_metric(tm, metric)
                row["status"] = "complete" if all(
                    row.get(metric) is not None for metric in ["accuracy", "auroc", "balanced_accuracy", "f1", "recall", "specificity"]
                ) and metrics else "missing_metrics"
                if row["status"] == "missing_metrics":
                    # keep partial metrics for visibility
                    row["status"] = "missing_metrics"

                if row.get("audit_json"):
                    audit = read_json(audit_json)
                    if audit:
                        row["audit_loaded_ratio"] = safe_float(audit.get("loaded_key_ratio"))
                else:
                    row["audit_loaded_ratio"] = ""
                row["n_samples"] = len(pred) if pred else None
                rows.append(row)
    return rows


def index_by_profile_fold_seed(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (row["profile"], row["fold"], row["seed"])
        out[key] = row
    return out


def paired_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = index_by_profile_fold_seed(rows)
    seedwise: list[dict[str, Any]] = []
    baseline_keys = { (fold, seed) for (profile, fold, seed), _ in by_key.items() if profile == "baseline_default" }
    for fold, seed in sorted(baseline_keys):
        base = by_key.get(("baseline_default", fold, seed))
        kd = by_key.get(("kdinit_diff_lr_01", fold, seed))
        if not (base and kd):
            continue
        row = {
            "fold": fold,
            "seed": seed,
            "baseline_group": base["group"],
            "kdinit_group": kd["group"],
            "baseline_run_dir": base["run_dir"],
            "kdinit_run_dir": kd["run_dir"],
            "baseline_best_epoch": base.get("best_epoch"),
            "kdinit_best_epoch": kd.get("best_epoch"),
            "baseline_selection_metric": base.get("selection_metric"),
            "kdinit_selection_metric": kd.get("selection_metric"),
        }
        for m in METRICS:
            b = base.get(m)
            k = kd.get(m)
            row[f"baseline_{m}"] = b
            row[f"kdinit_{m}"] = k
            row[f"delta_{m}"] = None if not isinstance(b, float) or not isinstance(k, float) else (k - b)
        if base.get("audit_loaded_ratio") != "":
            row["baseline_audit_loaded_ratio"] = base["audit_loaded_ratio"]
        if kd.get("audit_loaded_ratio") != "":
            row["kdinit_audit_loaded_ratio"] = kd["audit_loaded_ratio"]
        seedwise.append(row)
    return seedwise


def paired_fold_rows(seedwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seedwise:
        grouped[(row["fold"], row["seed"])].append(row)
    out = []
    for (fold, seed), pairs in sorted(grouped.items()):
        # in this suite pairwise should be exactly one pair
        if len(pairs) != 1:
            continue
        pair = pairs[0]
        row = {"fold": fold, "seed": seed}
        for m in METRICS:
            row[f"baseline_{m}"] = pair.get(f"baseline_{m}")
            row[f"kdinit_{m}"] = pair.get(f"kdinit_{m}")
            row[f"delta_{m}"] = pair.get(f"delta_{m}")
        out.append(row)
    return out


def bootstrap_seed_level(seedwise: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not seedwise:
        return out
    rng = random.Random(42)
    for metric in METRICS:
        vals = [row.get(f"delta_{metric}") for row in seedwise if isinstance(row.get(f"delta_{metric}"), float)]
        if not vals:
            continue
        observed = statistics.mean(vals)
        samples = []
        for _ in range(max(1, n_boot)):
            sample = [vals[rng.randrange(len(vals))] for _ in vals]
            samples.append(statistics.mean(sample))
        samples.sort()
        lo = samples[int(0.025 * (len(samples) - 1))]
        hi = samples[int(0.975 * (len(samples) - 1))]
        out.append({
            "comparison": "LIDC-KDInit vs LIDC-B",
            "metric": metric,
            "delta": observed,
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_crosses_0": lo <= 0 <= hi,
            "n_units": len(vals),
            "unit": "seed",
            "n_bootstrap": n_boot,
            "bootstrap_level": "seed",
        })
    return out


def load_seed_predictions(root: Path, profile: str, fold: str, seed: str) -> dict[str, tuple[int, float]]:
    pred_path = root / "profiles" / profile / f"fold{fold}" / f"seed{seed}" / "test_predictions.csv"
    return load_predictions(pred_path)


def seed_predictions_for_pair(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[list[int], list[float], list[float], str]:
    base, kd = pair
    fbase = Path(base["run_dir"]) / "test_predictions.csv"
    fk = Path(kd["run_dir"]) / "test_predictions.csv"
    pb = load_predictions(fbase)
    pk = load_predictions(fk)
    if not pb or not pk:
        return [], [], [], ""
    common = sorted(set(pb) & set(pk))
    if not common:
        return [], [], [], ""
    labels = [pb[sid][0] for sid in common]
    probs_b = [pb[sid][1] for sid in common]
    probs_k = [pk[sid][1] for sid in common]
    if any(l != pk[sid][0] for sid, l in zip(common, labels)):
        return [], [], [], ""
    return labels, probs_b, probs_k, ""


def bootstrap_sample_level(seedwise: list[dict[str, Any]], rows: list[dict[str, Any]], root: Path, n_boot: int) -> list[dict[str, Any]]:
    # require full seedwise and predictions for all pairs
    # Build lookup for baseline/kd profile complete rows
    by_key = index_by_profile_fold_seed(rows)
    # group by fold for bootstrap
    by_fold_seed: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in seedwise:
        fold = row["fold"]
        seed = row["seed"]
        base = by_key.get(("baseline_default", fold, seed))
        kd = by_key.get(("kdinit_diff_lr_01", fold, seed))
        if not (base and kd):
            continue
        labels, pb, pk, _ = seed_predictions_for_pair((base, kd))
        if not labels:
            return []
        by_fold_seed[fold].append((base, kd))

    if not by_fold_seed:
        return []

    by_fold_pairs: dict[str, list[tuple[dict[str, Any], list[tuple[str, float]], list[tuple[str, float]]]]] = {}
    for fold, pairs in by_fold_seed.items():
        seed_pair_data = []
        for base, kd in pairs:
            seed = base["seed"]
            base_pred = load_predictions(Path(base["run_dir"]) / "test_predictions.csv")
            kd_pred = load_predictions(Path(kd["run_dir"]) / "test_predictions.csv")
            if not base_pred or not kd_pred:
                seed_pair_data = []
                break
            common = sorted(set(base_pred) & set(kd_pred))
            if not common:
                seed_pair_data = []
                break
            labels = []
            bp = []
            kp = []
            for sid in common:
                if base_pred[sid][0] != kd_pred[sid][0]:
                    return []
                labels.append(base_pred[sid][0])
                bp.append(base_pred[sid][1])
                kp.append(kd_pred[sid][1])
            seed_pair_data.append((seed, labels, bp, kp))
        if not seed_pair_data:
            return []
        by_fold_pairs[fold] = seed_pair_data

    out: list[dict[str, Any]] = []
    rng = random.Random(42)
    for fold, seed_pairs in by_fold_pairs.items():
        if not seed_pairs:
            continue
        n_samples = len(seed_pairs[0][1])
        if n_samples == 0:
            continue
        # all seeds in same fold must align by sample count
        for _, labels, _, _ in seed_pairs:
            if len(labels) != n_samples:
                return []
        for metric in METRICS:
            deltas: list[float] = []
            for seed, labels, _, _ in seed_pairs:
                m = safe_float(seed)  # keep loop var safe
                _ = m
            vals = []
            observed_pairs = []
            for _ in range(max(1, n_boot)):
                seed_metrics = []
                for _, labels, base_probs, kd_probs in seed_pairs:
                    idxs = [rng.randrange(len(labels)) for _ in range(n_samples)]
                    bl = [labels[i] for i in idxs]
                    bp = [base_probs[i] for i in idxs]
                    kp = [kd_probs[i] for i in idxs]
                    mb = compute_binary_metrics(bl, bp)
                    mk = compute_binary_metrics(bl, kp)
                    dv = mk.get(metric)
                    bv = mb.get(metric)
                    if isinstance(dv, float) and isinstance(bv, float) and math.isfinite(dv) and math.isfinite(bv):
                        seed_metrics.append(dv - bv)
                if seed_metrics:
                    vals.append(statistics.mean(seed_metrics))
            if not vals:
                continue
            observed = statistics.mean(vals)
            vals.sort()
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            out.append({
                "comparison": f"LIDC-KDInit vs LIDC-B",
                "metric": metric,
                "delta": observed,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_crosses_0": lo <= 0 <= hi,
                "n_units": len(seed_pairs),
                "n_samples": n_samples,
                "unit": "sample_within_seed",
                "bootstrap_level": f"sample_fold{fold}",
                "n_bootstrap": n_boot,
            })
    return out


def summarize(seedwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_profile = {profile: [r for r in seedwise if r["seed"] and r[f"baseline_{METRICS[0]}"] is not None]}
    for profile in [TARGET_PROFILES[0], TARGET_PROFILES[1]]:
        if profile == "baseline_default":
            # keep compatible row in source order; baseline is used only for completeness
            runs = [r for r in seedwise if r["seed"]]
            row = {"profile": profile, "display_profile": DISPLAY_PROFILES.get(profile, profile)}
            n = len(runs)
            row["n_runs"] = n
            for m in METRICS:
                vals = [r[f"baseline_{m}"] for r in runs if isinstance(r.get(f"baseline_{m}"), float)]
                row[m] = mean(vals)
            out.append(row)
            continue
        runs = [r for r in seedwise]
        row = {"profile": profile, "display_profile": DISPLAY_PROFILES.get(profile, profile)}
        row["n_runs"] = len(runs)
        for m in METRICS:
            vals = [r[f"delta_{m}"] for r in seedwise if isinstance(r.get(f"delta_{m}"), float)]
            row[m] = mean(vals)
        out.append(row)
    return out


def seed_win_counts(seedwise: list[dict[str, Any]]) -> dict[str, int]:
    wins: dict[str, int] = {"kd_init_positive": 0, "kd_init_negative": 0, "mixed": 0}
    for row in seedwise:
        score = 0
        for m in RANK_METRICS:
            d = row.get(f"delta_{m}")
            if isinstance(d, float) and ((m != "ece" and m != "brier_score" and d > 0) or ((m in {"ece", "brier_score"} and d < 0))):
                score += 1
        if score >= 3:
            wins["kd_init_positive"] += 1
        elif score <= 1:
            wins["kd_init_negative"] += 1
        else:
            wins["mixed"] += 1
    return wins


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    if not rows:
        msg = (
            "# LIDC 4-seed Transfer Validation Summary\n\n"
            "MISSING: no profile/fold/seed metrics found in this root. "
            "Run the suite first (STAGE=train) and then STAGE=analyze.\n"
        )
        for name in [
            "lidc_4seed_metrics.md",
            "lidc_4seed_summary.md",
            "lidc_4seed_seedwise.csv",
            "lidc_4seed_comparison.md",
            "lidc_4seed_paper_table.md",
        ]:
            (root / name).write_text(msg, encoding="utf-8")
        write_csv(root / "lidc_4seed_metrics.csv", [], [
            "profile", "group", "fold", "seed", "status", "run_dir", "selection_metric", "best_epoch", "transfer_mode", "monitor_metric",
            *METRICS,
        ])
        write_csv(root / "lidc_4seed_bootstrap.csv", [], [
            "comparison", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_units", "unit", "n_bootstrap", "bootstrap_level", "n_samples",
        ])
        print(f"[MISSING] no LIDC 4-seed inputs found under {root}")
        return 0

    field_names = [
        "profile", "group", "fold", "seed", "run_name", "status", "run_dir", "selection_metric",
        "best_epoch", "transfer_mode", "monitor_metric", "n_samples", "prediction_path",
        "audit_json", "audit_loaded_ratio",
        *METRICS,
    ]
    write_csv(root / "lidc_4seed_metrics.csv", rows, field_names)
    (root / "lidc_4seed_metrics.md").write_text(
        "# LIDC 4-seed Transfer Metrics\n\n"
        + md_table(
            ["profile", "fold", "seed", "status", *METRICS],
            [[r["profile"], str(r["fold"]), str(r["seed"]), r["status"], *(fmt(r.get(m)) for m in METRICS)] for r in rows],
        )
        + "\n",
        encoding="utf-8",
    )

    seedwise = paired_seed_rows(rows)
    if not seedwise:
        print(f"[WARN] no paired baseline vs kdinit seed rows under {root}")
    seedwise_fields = ["fold", "seed"]
    for profile in ["baseline", "kdinit"]:
        seedwise_fields.extend([f"{profile}_{m}" for m in METRICS] if profile == "baseline" else [])
    # rebuild explicit fields
    seedwise_fields = ["fold", "seed"]
    for prefix in ("baseline", "kdinit"):
        seedwise_fields += [f"{prefix}_{m}" for m in METRICS]
    for prefix in ("baseline", "kdinit"):
        if prefix == "baseline":
            seedwise_fields += ["baseline_best_epoch", "baseline_selection_metric"]
        else:
            seedwise_fields += ["kdinit_best_epoch", "kdinit_selection_metric"]
    seedwise_fields += [f"delta_{m}" for m in METRICS]
    write_csv(root / "lidc_4seed_seedwise.csv", seedwise, seedwise_fields)

    # seed-level or sample-level bootstrap
    bootstrap_rows = bootstrap_sample_level(seedwise, rows, root, args.bootstrap_iters)
    if not bootstrap_rows:
        bootstrap_rows = bootstrap_seed_level(seedwise, args.bootstrap_iters)

    write_csv(
        root / "lidc_4seed_bootstrap.csv",
        bootstrap_rows,
        ["comparison", "metric", "delta", "ci95_low", "ci95_high", "ci_crosses_0", "n_units", "n_samples", "unit", "bootstrap_level", "n_bootstrap"],
    )

    # seed-wise comparison table
    comp_rows = []
    for row in seedwise:
        run = [str(row["fold"]), str(row["seed"])]
        for m in METRICS:
            run.extend([fmt(row.get(f"baseline_{m}")), fmt(row.get(f"kdinit_{m}")), fmt(row.get(f"delta_{m}"))])
        comp_rows.append(run)
    comp_headers = ["fold", "seed"]
    for m in METRICS:
        comp_headers.extend([f"baseline_{m}", f"kdinit_{m}", f"delta_{m}"])
    (root / "lidc_4seed_comparison.md").write_text(
        "# LIDC 4-seed Seed-wise Comparison\n\n"
        + md_table(comp_headers, [[str(c) for c in r] for r in comp_rows])
        + "\n",
        encoding="utf-8",
    )

    # overall summary
    lines = ["# LIDC 4-seed Transfer Validation Summary\n"]
    folds = sorted({r["fold"] for r in rows if r.get("fold") != ""})
    if folds == ["0"]:
        lines.append("single-fold 4-seed repeated transfer validation, not full 5-fold external validation.\n")
    else:
        lines.append(f"folds: {','.join(folds)}\n")
    lines.append("")
    # summary table
    summary_base = [r for r in seedwise]
    base_vals = {m: [r.get(f"baseline_{m}") for r in summary_base if isinstance(r.get(f"baseline_{m}"), float)] for m in METRICS}
    kd_vals = {m: [r.get(f"kdinit_{m}") for r in summary_base if isinstance(r.get(f"kdinit_{m}"), float)] for m in METRICS}
    delta_vals = {m: [r.get(f"delta_{m}") for r in summary_base if isinstance(r.get(f"delta_{m}"), float)] for m in METRICS}
    summary_rows = [["group", "n_runs", *RANK_METRICS, *["delta"]]]
    base_row = ["LIDC-B", str(len(base_vals.get("accuracy", [])))] + [mean_std(base_vals[m]) for m in RANK_METRICS]
    base_row.append("-")
    summary_rows.append(base_row)
    kd_row = ["LIDC-KDInit", str(len(kd_vals.get("accuracy", [])))] + [mean_std(kd_vals[m]) for m in RANK_METRICS]
    kd_row.append("")
    summary_rows.append(kd_row)
    lines.append(md_table(["group", "n_runs", *RANK_METRICS], summary_rows[:2]))
    lines.append("")

    lines.append("## Seed-wise delta summary")
    lines.append("| metric | LIDC-KDInit - LIDC-B (4-seed mean±std) |")
    lines.append("| --- | --- |")
    for m in METRICS:
        lines.append(f"| {m} | {mean_std(delta_vals.get(m, []))} |")
    lines.append("")
    win = seed_win_counts(seedwise)
    lines.append("### Seed-wise win count (heuristic)")
    lines.append(f"- KDInit clear positives: {win['kd_init_positive']}")
    lines.append(f"- KDInit clear negatives: {win['kd_init_negative']}")
    lines.append(f"- Mixed/neutral: {win['mixed']}")

    lines.append("")
    if bootstrap_rows:
        lines.append("## Bootstrap (paired)")
        lines.append("| metric | delta | 95%CI | crosses0 | level | n_units |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for b in bootstrap_rows:
            lines.append(
                "| "
                + " | ".join([
                str(b["metric"]),
                fmt(b["delta"]),
                f"[{fmt(b['ci95_low'])}, {fmt(b['ci95_high'])}]",
                str(b["ci_crosses_0"]),
                str(b["bootstrap_level"]),
                str(b["n_units"]),
                ])
                + " |"
            )
    lines.append("")
    if folds == ["0"] and summary_base:
        lines.append("Interpretation: this is a fixed-split single-fold repeated transfer validation.")
    lines.append("Use bootstrap and seed-wise deltas to assess whether R3-initialized CT encoder improves transfer behavior.")
    (root / "lidc_4seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # paper table by priority metrics
    rank_order = RANK_METRICS
    paper_rows = [["metric", "LIDC-B (mean±std)", "LIDC-KDInit (mean±std)", "Delta (mean±std)", "Crosses 0"]]
    for m in rank_order:
        b = mean(base_vals.get(m, []))
        k = mean(kd_vals.get(m, []))
        d = mean(delta_vals.get(m, []))
        # for CI row from bootstrap (seed-level if sample-level not available)
        bval = mean_std([v for v in base_vals.get(m, []) if isinstance(v, float)]) if base_vals.get(m) else "-"
        kval = mean_std([v for v in kd_vals.get(m, []) if isinstance(v, float)]) if kd_vals.get(m) else "-"
        dvals = [v for v in delta_vals.get(m, []) if isinstance(v, float)]
        d_std = "-"
        if dvals:
            if len(dvals) > 1:
                d_std = f"{statistics.mean(dvals):.4f}±{statistics.stdev(dvals):.4f}"
            else:
                d_std = f"{dvals[0]:.4f}±0.0000"
        bci = "-"
        for brow in bootstrap_rows:
            if brow["metric"] == m:
                bci = f"[{fmt(brow['ci95_low'])}, {fmt(brow['ci95_high'])}]"
                break
        paper_rows.append([m, bval, kval, d_std, bci])
    (root / "lidc_4seed_paper_table.md").write_text(
        "# LIDC 4-seed Paper-ready Table\n\n"
        + md_table(["metric", "LIDC-B", "LIDC-KDInit", "Delta", "95%CI"], [r[1:] for r in paper_rows])
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote 4-seed LIDC analysis under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
