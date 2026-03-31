from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if value_f != value_f:
        return None
    return value_f


def _get_metric(metrics: Dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    return _safe_float(metrics.get(key))


def _detect_family(metrics: Dict[str, Any]) -> str:
    config = metrics.get("config", {})
    if isinstance(config, dict):
        if "ct_model" in config or "gene_hidden_dim" in config:
            return "ct_cnv"
        if "n_estimators" in config or "model_path" in metrics or "estimator_diagnostics" in metrics:
            return "cnv_only"
        if "model" in config or "dataset_type" in metrics:
            return "ct_only"
    return "unknown"


def _resolve_backbone(family: str, metrics: Dict[str, Any]) -> str:
    config = metrics.get("config", {})
    if family == "ct_cnv":
        return str(config.get("ct_model", "unknown"))
    if family == "ct_only":
        return str(config.get("model", "unknown"))
    if family == "cnv_only":
        return "xgboost"
    return "unknown"


def _resolve_input_mode(family: str, metrics: Dict[str, Any]) -> str:
    config = metrics.get("config", {})
    if family == "cnv_only":
        return "tabular"
    use_3d = bool(config.get("use_3d_input"))
    if family == "ct_only" and not use_3d:
        model_name = str(config.get("model", "")).lower()
        if any(token in model_name for token in ["3d", "mc3", "r2plus1d", "swin3d", "attention3d"]):
            use_3d = True
    return "3d" if use_3d else "2d"


def _collect_paths(run_specs: Sequence[str], run_dirs: Sequence[str], metrics_name: str) -> List[Tuple[str, Path]]:
    collected: List[Tuple[str, Path]] = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH for --run, got: {spec}")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        metrics_path = path if path.is_file() else path / metrics_name
        collected.append((name.strip(), metrics_path))
    for raw_path in run_dirs:
        path = Path(raw_path).expanduser().resolve()
        metrics_path = path if path.is_file() else path / metrics_name
        collected.append((path.parent.name if path.is_file() else path.name, metrics_path))
    deduped: List[Tuple[str, Path]] = []
    seen = set()
    for name, path in collected:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, path))
    return deduped


def _load_row(label: str, metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    family = _detect_family(metrics)
    config = metrics.get("config", {})
    val_metrics = metrics.get("val_metrics") or metrics.get("best_val_metrics") or {}
    test_metrics = metrics.get("test_metrics") or {}
    train_metrics = metrics.get("train_metrics") or {}
    cohort_stats = metrics.get("cohort_stats", {})
    num_total = cohort_stats.get("num_total")
    if num_total is None and "rows_after_dedup" in cohort_stats:
        num_total = cohort_stats.get("rows_after_dedup")

    row = {
        "experiment": label,
        "family": family,
        "backbone": _resolve_backbone(family, metrics),
        "input_mode": _resolve_input_mode(family, metrics),
        "selection_metric": metrics.get("selection_metric") or metrics.get("selection_metric_used"),
        "class_mode": config.get("class_mode"),
        "binary_task": config.get("binary_task"),
        "split_source": metrics.get("split_source"),
        "num_total": num_total,
        "num_train": cohort_stats.get("num_train"),
        "num_val": cohort_stats.get("num_val"),
        "num_test": cohort_stats.get("num_test"),
        "best_epoch": metrics.get("best_epoch"),
        "feature_dim": metrics.get("feature_dim"),
        "val_auroc": _get_metric(val_metrics, "auroc"),
        "val_auprc": _get_metric(val_metrics, "auprc"),
        "val_bacc": _get_metric(val_metrics, "balanced_accuracy"),
        "val_f1": _get_metric(val_metrics, "f1"),
        "val_sensitivity": _get_metric(val_metrics, "sensitivity"),
        "val_specificity": _get_metric(val_metrics, "specificity"),
        "test_auroc": _get_metric(test_metrics, "auroc"),
        "test_auprc": _get_metric(test_metrics, "auprc"),
        "test_bacc": _get_metric(test_metrics, "balanced_accuracy"),
        "test_f1": _get_metric(test_metrics, "f1"),
        "test_sensitivity": _get_metric(test_metrics, "sensitivity"),
        "test_specificity": _get_metric(test_metrics, "specificity"),
        "train_auroc": _get_metric(train_metrics, "auroc"),
        "run_dir": str(metrics_path.parent),
    }
    if family == "cnv_only":
        diagnostics = metrics.get("estimator_diagnostics", {})
        row["best_iteration"] = diagnostics.get("best_iteration", metrics.get("best_iteration"))
        row["used_boosting_rounds"] = diagnostics.get("used_boosting_rounds")
    return row


def _write_markdown(path: Path, table: pd.DataFrame) -> None:
    columns = [
        "experiment",
        "family",
        "backbone",
        "input_mode",
        "test_auroc",
        "val_auroc",
        "test_bacc",
        "test_f1",
        "num_total",
    ]
    available = [col for col in columns if col in table.columns]
    lines = [
        "# Experiment Table",
        "",
        "| " + " | ".join(available) + " |",
        "| " + " | ".join(["---"] * len(available)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for col in available:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_experiment_table(
    run_specs: Sequence[str],
    run_dirs: Sequence[str],
    output_dir: Path,
    metrics_name: str = "metrics.json",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _collect_paths(run_specs, run_dirs, metrics_name)
    if not paths:
        raise RuntimeError("No experiment runs were provided.")

    rows = [_load_row(label, metrics_path) for label, metrics_path in paths]
    table = pd.DataFrame(rows)
    sort_cols = [col for col in ["test_auroc", "val_auroc", "test_bacc"] if col in table.columns]
    if sort_cols:
        table = table.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    table = table.reset_index(drop=True)

    csv_path = output_dir / "experiment_table.csv"
    md_path = output_dir / "experiment_table.md"
    summary_path = output_dir / "summary.json"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_markdown(md_path, table)

    summary = {
        "num_runs": int(len(table)),
        "families": sorted(set(table["family"].dropna().astype(str).tolist())),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Exported experiment table for {len(table)} runs")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print("=" * 60)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a unified result table from experiment metrics.")
    parser.add_argument("--run", action="append", default=[], help="Named run in NAME=PATH format. Can be passed multiple times.")
    parser.add_argument("--run-dir", action="append", default=[], help="Experiment directory or metrics.json path. Can be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the exported table.")
    parser.add_argument("--metrics-name", type=str, default="metrics.json", help="Metrics filename when --run-dir points to a directory.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    export_experiment_table(args.run, args.run_dir, args.output_dir, metrics_name=args.metrics_name)


if __name__ == "__main__":
    main()
