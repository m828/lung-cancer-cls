from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "matplotlib is required for batch experiment visualization. "
        "Install it with `pip install matplotlib`."
    ) from exc

from .experiment_table import export_experiment_table


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned or "run"


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _normalize_cli_args(args: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    for item in args:
        normalized.append(str(item))
    return normalized


def _extract_flag_value(args: Sequence[str], flag: str) -> str | None:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


@dataclass
class BatchExperimentSpec:
    name: str
    entrypoint: str
    args: List[str]
    run_dir: str | None = None
    completion_file: str = "metrics.json"
    report_metrics_path: str | None = None
    plot_split: str = "test"
    enabled: bool = True
    skip_if_exists: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class BatchRunnerConfig:
    experiments: List[BatchExperimentSpec]
    report_dir: Path
    metrics_name: str = "metrics.json"
    continue_on_error: bool = True
    skip_existing_success: bool = True
    export_table: bool = True
    export_plots: bool = True


@dataclass
class BatchRunResult:
    name: str
    entrypoint: str
    status: str
    command: List[str]
    run_dir: str | None
    completion_path: str | None
    report_metrics_path: str | None
    plot_split: str
    log_path: str
    started_at: str
    finished_at: str
    duration_seconds: float
    exit_code: int | None = None
    error_message: str | None = None
    tags: List[str] = field(default_factory=list)
    plot_status: str | None = None
    plot_dir: str | None = None


def _load_experiment_spec(raw: Dict[str, Any]) -> BatchExperimentSpec:
    if "name" not in raw or "entrypoint" not in raw:
        raise ValueError("Each experiment needs at least `name` and `entrypoint`.")
    args = raw.get("args", [])
    if not isinstance(args, list):
        raise ValueError(f"`args` must be a list for experiment {raw.get('name')}.")
    return BatchExperimentSpec(
        name=str(raw["name"]),
        entrypoint=str(raw["entrypoint"]),
        args=_normalize_cli_args(args),
        run_dir=raw.get("run_dir"),
        completion_file=str(raw.get("completion_file", "metrics.json")),
        report_metrics_path=raw.get("report_metrics_path"),
        plot_split=str(raw.get("plot_split", "test")),
        enabled=bool(raw.get("enabled", True)),
        skip_if_exists=bool(raw.get("skip_if_exists", True)),
        tags=[str(item) for item in raw.get("tags", [])],
    )


def load_batch_runner_config(config_path: Path) -> BatchRunnerConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_dir = config_path.parent
    experiments = [_load_experiment_spec(item) for item in data.get("experiments", [])]
    if not experiments:
        raise ValueError("The batch config does not contain any experiments.")
    report_dir = _resolve_path(data.get("report_dir", "outputs/batch_runs"), base_dir)
    assert report_dir is not None
    return BatchRunnerConfig(
        experiments=experiments,
        report_dir=report_dir,
        metrics_name=str(data.get("metrics_name", "metrics.json")),
        continue_on_error=bool(data.get("continue_on_error", True)),
        skip_existing_success=bool(data.get("skip_existing_success", True)),
        export_table=bool(data.get("export_table", True)),
        export_plots=bool(data.get("export_plots", True)),
    )


def write_template_config(path: Path) -> Path:
    template = {
        "report_dir": "outputs/batch_runs_demo",
        "metrics_name": "metrics.json",
        "continue_on_error": True,
        "skip_existing_success": True,
        "export_table": True,
        "export_plots": True,
        "experiments": [
            {
                "name": "teacher_resnet3d18_binary",
                "entrypoint": "train_multimodal.py",
                "args": [
                    "--data-root",
                    "/path/to/data_root",
                    "--metadata-csv",
                    "/path/to/metadata.csv",
                    "--ct-root",
                    "/path/to/ct_root",
                    "--gene-tsv",
                    "/path/to/gene.tsv",
                    "--text-feature-tsv",
                    "/path/to/text_features.tsv",
                    "--output-dir",
                    "outputs/teacher_resnet3d18_binary",
                    "--modalities",
                    "ct,text,cnv",
                    "--ct-model",
                    "resnet3d18",
                    "--class-mode",
                    "binary",
                    "--binary-task",
                    "malignant_vs_normal",
                    "--selection-metric",
                    "auroc",
                    "--use-predefined-split"
                ],
                "plot_split": "test",
                "tags": ["teacher", "binary", "resnet3d18"]
            },
            {
                "name": "ct_student_resnet3d18_binary",
                "entrypoint": "train_student_kd.py",
                "args": [
                    "--teacher-run-dir",
                    "outputs/teacher_resnet3d18_binary",
                    "--reference-manifest",
                    "outputs/teacher_resnet3d18_binary/split_manifest.csv",
                    "--ct-root",
                    "/path/to/ct_root",
                    "--output-dir",
                    "outputs/ct_student_resnet3d18_binary",
                    "--modalities",
                    "ct",
                    "--ct-model",
                    "resnet3d18",
                    "--class-mode",
                    "binary",
                    "--binary-task",
                    "malignant_vs_normal",
                    "--selection-metric",
                    "auroc"
                ],
                "plot_split": "test",
                "tags": ["student", "binary", "resnet3d18"]
            },
            {
                "name": "ct_baseline_swin3d_multiclass",
                "entrypoint": "train.py",
                "args": [
                    "--dataset-type",
                    "intranet_ct",
                    "--data-root",
                    "/path/to/data_root",
                    "--metadata-csv",
                    "/path/to/metadata.csv",
                    "--ct-root",
                    "/path/to/ct_root",
                    "--output-dir",
                    "outputs/ct_baseline_swin3d_multiclass",
                    "--model",
                    "swin3d_tiny",
                    "--use-3d-input",
                    "--class-mode",
                    "multiclass",
                    "--use-predefined-split",
                    "--selection-metric",
                    "balanced_accuracy"
                ],
                "plot_split": "test",
                "tags": ["baseline", "multiclass", "swin3d"]
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolve_run_dir(spec: BatchExperimentSpec, config_base_dir: Path) -> Path | None:
    explicit = _resolve_path(spec.run_dir, config_base_dir)
    if explicit is not None:
        return explicit
    output_dir = _extract_flag_value(spec.args, "--output-dir")
    return _resolve_path(output_dir, config_base_dir)


def _resolve_completion_path(spec: BatchExperimentSpec, run_dir: Path | None, config_base_dir: Path) -> Path | None:
    if run_dir is None:
        return None
    completion_file = Path(spec.completion_file)
    if completion_file.is_absolute():
        return completion_file
    return (run_dir / completion_file).resolve()


def _resolve_report_metrics_path(
    spec: BatchExperimentSpec,
    run_dir: Path | None,
    config_base_dir: Path,
    default_metrics_name: str,
) -> Path | None:
    explicit = _resolve_path(spec.report_metrics_path, config_base_dir)
    if explicit is not None:
        return explicit
    if run_dir is None:
        return None
    return (run_dir / default_metrics_name).resolve()


def _should_skip_existing(
    spec: BatchExperimentSpec,
    completion_path: Path | None,
    config: BatchRunnerConfig,
) -> bool:
    if not config.skip_existing_success or not spec.skip_if_exists:
        return False
    return completion_path is not None and completion_path.exists()


def _build_command(
    spec: BatchExperimentSpec,
    entrypoint_path: Path,
    python_executable: str,
) -> List[str]:
    return [python_executable, str(entrypoint_path)] + list(spec.args)


def _write_log(log_path: Path, command: Sequence[str], stdout: str, stderr: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "COMMAND:",
        " ".join(command),
        "",
        "STDOUT:",
        stdout.rstrip(),
        "",
        "STDERR:",
        stderr.rstrip(),
        "",
    ]
    log_path.write_text("\n".join(body), encoding="utf-8")


def _plot_overview_from_table(table_csv: Path, output_dir: Path) -> List[str]:
    df = pd.read_csv(table_csv)
    created: List[str] = []
    if df.empty or "experiment" not in df.columns:
        return created

    plot_df = df.copy()
    plot_df = plot_df.sort_values(
        by=[col for col in ["test_auroc", "test_bacc", "test_f1"] if col in plot_df.columns],
        ascending=False,
        na_position="last",
    )
    plot_df["experiment"] = plot_df["experiment"].astype(str)

    metric_cols = [col for col in ["test_auroc", "test_bacc", "test_f1"] if col in plot_df.columns]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.8), 5))
        x = range(len(plot_df))
        width = 0.25 if len(metric_cols) >= 3 else 0.35
        offsets = {
            1: [0.0],
            2: [-width / 2.0, width / 2.0],
            3: [-width, 0.0, width],
        }[len(metric_cols)]
        for offset, col in zip(offsets, metric_cols):
            ax.bar(
                [idx + offset for idx in x],
                plot_df[col].fillna(0.0).tolist(),
                width=width,
                label=col,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df["experiment"].tolist(), rotation=35, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Batch Test Metrics")
        ax.legend()
        fig.tight_layout()
        metric_plot = output_dir / "batch_test_metrics.png"
        fig.savefig(metric_plot, dpi=200)
        plt.close(fig)
        created.append(metric_plot.name)

    gap_specs = [
        ("auroc", "val_auroc", "test_auroc"),
        ("bacc", "val_bacc", "test_bacc"),
        ("f1", "val_f1", "test_f1"),
    ]
    gap_rows: List[Dict[str, Any]] = []
    for _, row in plot_df.iterrows():
        item: Dict[str, Any] = {"experiment": row["experiment"]}
        has_gap = False
        for label, val_col, test_col in gap_specs:
            val_value = row.get(val_col)
            test_value = row.get(test_col)
            if pd.notna(val_value) and pd.notna(test_value):
                item[f"{label}_gap"] = float(test_value) - float(val_value)
                has_gap = True
        if has_gap:
            gap_rows.append(item)
    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        fig, ax = plt.subplots(figsize=(max(8, len(gap_df) * 0.8), 5))
        x = range(len(gap_df))
        gap_cols = [col for col in ["auroc_gap", "bacc_gap", "f1_gap"] if col in gap_df.columns]
        width = 0.25 if len(gap_cols) >= 3 else 0.35
        offsets = {
            1: [0.0],
            2: [-width / 2.0, width / 2.0],
            3: [-width, 0.0, width],
        }[len(gap_cols)]
        for offset, col in zip(offsets, gap_cols):
            ax.bar(
                [idx + offset for idx in x],
                gap_df[col].fillna(0.0).tolist(),
                width=width,
                label=col,
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(gap_df["experiment"].tolist(), rotation=35, ha="right")
        ax.set_title("Test Minus Val Gap")
        ax.legend()
        fig.tight_layout()
        gap_plot = output_dir / "batch_val_test_gap.png"
        fig.savefig(gap_plot, dpi=200)
        plt.close(fig)
        created.append(gap_plot.name)

    return created


def _export_batch_reports(
    results: Sequence[BatchRunResult],
    config: BatchRunnerConfig,
) -> Dict[str, Any]:
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(asdict(item) for item in results)
    if not results_df.empty:
        results_df.to_csv(report_dir / "batch_results.csv", index=False, encoding="utf-8-sig")

    summary: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "num_runs": int(len(results)),
        "status_counts": results_df["status"].value_counts().to_dict() if not results_df.empty else {},
        "report_dir": str(report_dir),
        "batch_results_csv": str(report_dir / "batch_results.csv"),
    }

    reportable = [
        item
        for item in results
        if item.report_metrics_path
        and Path(item.report_metrics_path).exists()
        and item.status in {"success", "skipped_existing", "report_only", "completed_without_metrics"}
    ]

    if config.export_table and reportable:
        table_dir = report_dir / "table"
        table_summary = export_experiment_table(
            run_specs=[f"{item.name}={item.report_metrics_path}" for item in reportable],
            run_dirs=[],
            output_dir=table_dir,
            metrics_name=config.metrics_name,
        )
        summary["table_summary"] = table_summary
        created = _plot_overview_from_table(table_dir / "experiment_table.csv", report_dir)
        if created:
            summary["overview_plots"] = created

    plot_summaries: List[Dict[str, Any]] = []
    plot_warning: str | None = None
    if config.export_plots:
        try:
            from .visualize_experiment import export_experiment_plots
        except Exception as exc:  # pragma: no cover - depends on optional plotting deps
            export_experiment_plots = None
            plot_warning = f"Per-run plots skipped: {type(exc).__name__}: {exc}"
            summary["plot_warning"] = plot_warning
        else:
            plot_warning = None
        if export_experiment_plots is None:
            for item in reportable:
                item.plot_status = "skipped_missing_plot_dependencies"
        else:
            for item in reportable:
                if not item.run_dir:
                    continue
                run_dir = Path(item.run_dir)
                if not (run_dir / config.metrics_name).exists():
                    continue
                out_dir = report_dir / "plots" / _safe_name(item.name)
                try:
                    plot_summary = export_experiment_plots(run_dir, out_dir, split=item.plot_split)
                    item.plot_status = "success"
                    item.plot_dir = str(out_dir)
                    plot_summaries.append(plot_summary)
                except Exception as exc:
                    item.plot_status = f"failed:{type(exc).__name__}"
                    item.plot_dir = str(out_dir)
        if plot_summaries:
            summary["per_run_plots"] = plot_summaries

    with open(report_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not results_df.empty:
        refreshed_df = pd.DataFrame(asdict(item) for item in results)
        refreshed_df.to_csv(report_dir / "batch_results.csv", index=False, encoding="utf-8-sig")

    return summary


def run_batch_experiments(
    config: BatchRunnerConfig,
    config_base_dir: Path,
    project_root: Path,
    only_names: Sequence[str] | None = None,
    report_only: bool = False,
    python_executable: str | None = None,
) -> Dict[str, Any]:
    selected = set(str(name) for name in (only_names or []))
    py_exec = python_executable or sys.executable
    results: List[BatchRunResult] = []

    for spec in config.experiments:
        if not spec.enabled:
            continue
        if selected and spec.name not in selected:
            continue

        entrypoint_path = _resolve_path(spec.entrypoint, project_root)
        if entrypoint_path is None:
            raise RuntimeError(f"Failed to resolve entrypoint for {spec.name}")
        run_dir = _resolve_run_dir(spec, config_base_dir)
        completion_path = _resolve_completion_path(spec, run_dir, config_base_dir)
        report_metrics_path = _resolve_report_metrics_path(spec, run_dir, config_base_dir, config.metrics_name)
        command = _build_command(spec, entrypoint_path, py_exec)
        log_path = config.report_dir / "logs" / f"{_safe_name(spec.name)}.log"
        started_at = _utc_now_iso()
        start_time = time.perf_counter()

        if report_only:
            status = "report_only"
            exit_code = 0
            error_message = None
            stdout = "Report-only mode: command not executed."
            stderr = ""
        elif _should_skip_existing(spec, completion_path, config):
            status = "skipped_existing"
            exit_code = 0
            error_message = None
            stdout = "Skipped existing successful run because completion file already exists."
            stderr = ""
        else:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                exit_code = int(completed.returncode)
                stdout = completed.stdout
                stderr = completed.stderr
                error_message = None
                if exit_code == 0:
                    if completion_path is None or completion_path.exists():
                        status = "success"
                    else:
                        status = "completed_without_metrics"
                        error_message = (
                            "Command exited successfully, but the configured completion file "
                            f"was not found: {completion_path}"
                        )
                else:
                    status = "failed"
                    error_message = f"Command exited with code {exit_code}."
            except Exception as exc:  # pragma: no cover - subprocess failures are environment dependent
                exit_code = None
                stdout = ""
                stderr = str(exc)
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"

        finished_at = _utc_now_iso()
        duration = time.perf_counter() - start_time
        _write_log(log_path, command, stdout, stderr)

        result = BatchRunResult(
            name=spec.name,
            entrypoint=str(entrypoint_path),
            status=status,
            command=command,
            run_dir=str(run_dir) if run_dir is not None else None,
            completion_path=str(completion_path) if completion_path is not None else None,
            report_metrics_path=str(report_metrics_path) if report_metrics_path is not None else None,
            plot_split=spec.plot_split,
            log_path=str(log_path),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=round(duration, 3),
            exit_code=exit_code,
            error_message=error_message,
            tags=list(spec.tags),
        )
        results.append(result)

        if status == "failed" and not config.continue_on_error:
            break

    return _export_batch_reports(results, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batch of training experiments sequentially, skip failures, and export a unified report."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config describing the experiment batch.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to resolve entrypoints.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated subset of experiment names to run.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Do not execute commands; only aggregate reports for the runs listed in the config.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Optional Python executable used to launch child processes.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        default=None,
        help="Write a starter JSON config to this path and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.write_template is not None:
        path = write_template_config(args.write_template)
        print(f"Wrote batch config template: {path}")
        return
    if args.config is None:
        raise ValueError("--config is required unless --write-template is used.")

    config_path = args.config.expanduser().resolve()
    config = load_batch_runner_config(config_path)
    only_names = [item.strip() for item in args.only.split(",") if item.strip()]
    summary = run_batch_experiments(
        config=config,
        config_base_dir=config_path.parent,
        project_root=args.project_root.expanduser().resolve(),
        only_names=only_names,
        report_only=bool(args.report_only),
        python_executable=args.python_executable,
    )
    print("=" * 60)
    print("Batch experiment run complete")
    print(f"Summary: {config.report_dir / 'batch_summary.json'}")
    if "table_summary" in summary:
        table_summary = summary["table_summary"]
        print(f"Table CSV: {table_summary.get('csv_path')}")
        print(f"Table Markdown: {table_summary.get('markdown_path')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
