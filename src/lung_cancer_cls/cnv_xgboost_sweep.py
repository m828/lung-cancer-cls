from __future__ import annotations

import argparse
import json
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from lung_cancer_cls.cnv_xgboost import CNVXGBoostConfig, train_xgboost_baseline


def _parse_list(raw: str, cast: Any) -> List[Any]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [cast(item) for item in values if item]
    if not parsed:
        raise ValueError(f"Expected a non-empty comma-separated list, got: {raw}")
    return parsed


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if value_f != value_f:
        return None
    return value_f


@dataclass
class CNVXGBoostSweepConfig:
    base_config: CNVXGBoostConfig
    output_dir: Path
    preset: str
    seeds: List[int]
    max_depth_grid: List[int]
    min_child_weight_grid: List[float]
    subsample_grid: List[float]
    colsample_bytree_grid: List[float]
    learning_rate_grid: List[float]
    reg_lambda_grid: List[float]
    gamma_grid: List[float]
    max_runs: int | None = None
    save_predictions: bool = False


CNV_SWEEP_PRESETS: Dict[str, Dict[str, List[Any]]] = {
    "fast": {
        "seeds": [42, 52, 62],
        "max_depth_grid": [3, 4],
        "min_child_weight_grid": [1.0, 2.0],
        "subsample_grid": [0.8],
        "colsample_bytree_grid": [0.8],
        "learning_rate_grid": [0.05],
        "reg_lambda_grid": [1.0, 2.0],
        "gamma_grid": [0.0],
    },
    "formal": {
        "seeds": [42, 52, 62],
        "max_depth_grid": [3, 4],
        "min_child_weight_grid": [1.0, 2.0],
        "subsample_grid": [0.8, 1.0],
        "colsample_bytree_grid": [0.8],
        "learning_rate_grid": [0.03, 0.05],
        "reg_lambda_grid": [1.0, 2.0],
        "gamma_grid": [0.0],
    },
}


def resolve_sweep_grid(
    preset: str,
    raw_value: str | None,
    cast: Any,
    key: str,
) -> List[Any]:
    if raw_value:
        return _parse_list(raw_value, cast)
    if preset not in CNV_SWEEP_PRESETS:
        raise ValueError(f"Unknown sweep preset: {preset}")
    return list(CNV_SWEEP_PRESETS[preset][key])


def _write_markdown_leaderboard(path: Path, leaderboard: pd.DataFrame) -> None:
    columns = [
        "seed",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "learning_rate",
        "reg_lambda",
        "val_auroc",
        "test_auroc",
        "best_iteration",
    ]
    available = [col for col in columns if col in leaderboard.columns]
    lines = [
        "# CNV Sweep Leaderboard",
        "",
        "| " + " | ".join(available) + " |",
        "| " + " | ".join(["---"] * len(available)) + " |",
    ]
    for _, row in leaderboard.iterrows():
        values = []
        for col in available:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_run_configs(config: CNVXGBoostSweepConfig) -> Iterable[CNVXGBoostConfig]:
    run_idx = 0
    for (
        seed,
        max_depth,
        min_child_weight,
        subsample,
        colsample_bytree,
        learning_rate,
        reg_lambda,
        gamma,
    ) in itertools.product(
        config.seeds,
        config.max_depth_grid,
        config.min_child_weight_grid,
        config.subsample_grid,
        config.colsample_bytree_grid,
        config.learning_rate_grid,
        config.reg_lambda_grid,
        config.gamma_grid,
    ):
        if config.max_runs is not None and run_idx >= config.max_runs:
            break

        run_name = (
            f"seed{seed}"
            f"_md{max_depth}"
            f"_mcw{min_child_weight:g}"
            f"_ss{subsample:g}"
            f"_cs{colsample_bytree:g}"
            f"_lr{learning_rate:g}"
            f"_l2{reg_lambda:g}"
            f"_gm{gamma:g}"
        )
        base_kwargs = asdict(config.base_config)
        base_kwargs["output_dir"] = config.output_dir / run_name
        base_kwargs["seed"] = seed
        base_kwargs["max_depth"] = max_depth
        base_kwargs["min_child_weight"] = min_child_weight
        base_kwargs["subsample"] = subsample
        base_kwargs["colsample_bytree"] = colsample_bytree
        base_kwargs["learning_rate"] = learning_rate
        base_kwargs["reg_lambda"] = reg_lambda
        base_kwargs["gamma"] = gamma
        base_kwargs["save_predictions"] = config.save_predictions
        yield CNVXGBoostConfig(**base_kwargs)
        run_idx += 1


def run_sweep(config: CNVXGBoostSweepConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    total_candidates = (
        len(config.seeds)
        * len(config.max_depth_grid)
        * len(config.min_child_weight_grid)
        * len(config.subsample_grid)
        * len(config.colsample_bytree_grid)
        * len(config.learning_rate_grid)
        * len(config.reg_lambda_grid)
        * len(config.gamma_grid)
    )
    planned_runs = min(total_candidates, config.max_runs) if config.max_runs is not None else total_candidates
    print("=" * 60)
    print(f"CNV sweep preset: {config.preset}")
    print(f"Planned runs: {planned_runs}")
    print("=" * 60)

    leaderboard_rows: List[Dict[str, Any]] = []
    for run_id, run_config in enumerate(iter_run_configs(config), start=1):
        print("=" * 60)
        print(f"CNV sweep run {run_id}")
        print(f"Output dir: {run_config.output_dir}")
        print(
            "Params: "
            f"seed={run_config.seed} max_depth={run_config.max_depth} "
            f"min_child_weight={run_config.min_child_weight} "
            f"subsample={run_config.subsample} "
            f"colsample_bytree={run_config.colsample_bytree} "
            f"lr={run_config.learning_rate} "
            f"reg_lambda={run_config.reg_lambda} "
            f"gamma={run_config.gamma}"
        )
        metrics = train_xgboost_baseline(run_config)
        diagnostics = metrics.get("estimator_diagnostics", {})
        leaderboard_rows.append(
            {
                "run_dir": str(run_config.output_dir),
                "seed": run_config.seed,
                "max_depth": run_config.max_depth,
                "min_child_weight": run_config.min_child_weight,
                "subsample": run_config.subsample,
                "colsample_bytree": run_config.colsample_bytree,
                "learning_rate": run_config.learning_rate,
                "reg_lambda": run_config.reg_lambda,
                "gamma": run_config.gamma,
                "selection_metric": metrics.get("selection_metric_used"),
                "selection_score": _safe_float(metrics.get("selection_score")),
                "val_auroc": _safe_float(metrics.get("val_metrics", {}).get("auroc")),
                "val_auprc": _safe_float(metrics.get("val_metrics", {}).get("auprc")),
                "val_bacc": _safe_float(metrics.get("val_metrics", {}).get("balanced_accuracy")),
                "test_auroc": _safe_float(metrics.get("test_metrics", {}).get("auroc")),
                "test_auprc": _safe_float(metrics.get("test_metrics", {}).get("auprc")),
                "test_bacc": _safe_float(metrics.get("test_metrics", {}).get("balanced_accuracy")),
                "best_iteration": diagnostics.get("best_iteration"),
                "used_boosting_rounds": diagnostics.get("used_boosting_rounds"),
                "best_score": _safe_float(diagnostics.get("best_score")),
            }
        )

    if not leaderboard_rows:
        raise RuntimeError("Sweep produced no runs.")

    leaderboard = pd.DataFrame(leaderboard_rows)
    rank_cols = ["test_auroc", "val_auroc", "selection_score"]
    available_rank_cols = [col for col in rank_cols if col in leaderboard.columns]
    leaderboard = leaderboard.sort_values(
        by=available_rank_cols,
        ascending=[False] * len(available_rank_cols),
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.to_csv(config.output_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    _write_markdown_leaderboard(config.output_dir / "leaderboard.md", leaderboard.head(20))

    numeric_cols = [
        "val_auroc",
        "val_auprc",
        "val_bacc",
        "test_auroc",
        "test_auprc",
        "test_bacc",
        "selection_score",
    ]
    aggregate = {}
    for col in numeric_cols:
        series = leaderboard[col].dropna()
        if not series.empty:
            aggregate[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "max": float(series.max()),
                "min": float(series.min()),
            }

    best_row = leaderboard.iloc[0].to_dict()
    summary = {
        "preset": config.preset,
        "num_runs": int(len(leaderboard)),
        "planned_runs": int(planned_runs),
        "best_run": best_row,
        "aggregate": aggregate,
        "leaderboard_path": str(config.output_dir / "leaderboard.csv"),
        "leaderboard_markdown_path": str(config.output_dir / "leaderboard.md"),
    }
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Sweep complete: {len(leaderboard)} runs")
    print(f"Best run: {best_row['run_dir']}")
    print(
        "Best metrics: "
        f"val_auroc={best_row.get('val_auroc')} "
        f"test_auroc={best_row.get('test_auroc')} "
        f"best_iteration={best_row.get('best_iteration')}"
    )
    print(f"Leaderboard: {config.output_dir / 'leaderboard.csv'}")
    print("=" * 60)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a light CNV XGBoost sweep.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Metadata CSV path.")
    parser.add_argument("--gene-tsv", type=Path, required=True, help="CNV / gene TSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Sweep output directory.")
    parser.add_argument("--label-col", type=str, default="样本类型", help="Metadata label column.")
    parser.add_argument("--metadata-gene-id-col", type=str, default="SampleID", help="Metadata gene ID column.")
    parser.add_argument("--split-col", type=str, default="CT_train_val_split", help="Metadata split column.")
    parser.add_argument("--gene-id-col", type=str, default=None, help="Gene TSV ID column.")
    parser.add_argument("--gene-label-col", type=str, default=None, help="Gene TSV label column to drop from features.")
    parser.add_argument("--class-mode", type=str, choices=["multiclass", "binary"], default="binary")
    parser.add_argument("--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal", "benign_vs_malignant"], default="malignant_vs_normal")
    parser.add_argument("--selection-metric", type=str, choices=["auto", "accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "loss"], default="auto")
    parser.add_argument("--split-mode", type=str, choices=["train_val", "train_val_test"], default="train_val_test")
    parser.add_argument("--use-predefined-split", action="store_true", help="Use split labels from the metadata CSV when available.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--class-weight-strategy", type=str, choices=["none", "balanced"], default="balanced")
    parser.add_argument("--preset", type=str, choices=sorted(CNV_SWEEP_PRESETS), default="formal", help="Recommended sweep preset.")
    parser.add_argument("--seed-list", type=str, default=None, help="Comma-separated seed list. Overrides the preset.")
    parser.add_argument("--max-depth-grid", type=str, default=None, help="Comma-separated max_depth grid. Overrides the preset.")
    parser.add_argument("--min-child-weight-grid", type=str, default=None, help="Comma-separated min_child_weight grid. Overrides the preset.")
    parser.add_argument("--subsample-grid", type=str, default=None, help="Comma-separated subsample grid. Overrides the preset.")
    parser.add_argument("--colsample-bytree-grid", type=str, default=None, help="Comma-separated colsample_bytree grid. Overrides the preset.")
    parser.add_argument("--learning-rate-grid", type=str, default=None, help="Comma-separated learning_rate grid. Overrides the preset.")
    parser.add_argument("--reg-lambda-grid", type=str, default=None, help="Comma-separated reg_lambda grid. Overrides the preset.")
    parser.add_argument("--gamma-grid", type=str, default=None, help="Comma-separated gamma grid. Overrides the preset.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional hard cap on the number of sweep runs.")
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction CSVs for each run.")
    return parser


def parse_args() -> CNVXGBoostSweepConfig:
    args = build_parser().parse_args()
    base_config = CNVXGBoostConfig(
        metadata_csv=args.metadata_csv,
        gene_tsv=args.gene_tsv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        metadata_gene_id_col=args.metadata_gene_id_col,
        split_col=args.split_col,
        gene_id_col=args.gene_id_col,
        gene_label_col=args.gene_label_col,
        class_mode=args.class_mode,
        binary_task=args.binary_task,
        selection_metric=args.selection_metric,
        split_mode=args.split_mode,
        use_predefined_split=args.use_predefined_split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        class_weight_strategy=args.class_weight_strategy,
        save_predictions=args.save_predictions,
    )
    preset = args.preset.lower().strip()
    return CNVXGBoostSweepConfig(
        base_config=base_config,
        output_dir=args.output_dir,
        preset=preset,
        seeds=resolve_sweep_grid(preset, args.seed_list, int, "seeds"),
        max_depth_grid=resolve_sweep_grid(preset, args.max_depth_grid, int, "max_depth_grid"),
        min_child_weight_grid=resolve_sweep_grid(preset, args.min_child_weight_grid, float, "min_child_weight_grid"),
        subsample_grid=resolve_sweep_grid(preset, args.subsample_grid, float, "subsample_grid"),
        colsample_bytree_grid=resolve_sweep_grid(preset, args.colsample_bytree_grid, float, "colsample_bytree_grid"),
        learning_rate_grid=resolve_sweep_grid(preset, args.learning_rate_grid, float, "learning_rate_grid"),
        reg_lambda_grid=resolve_sweep_grid(preset, args.reg_lambda_grid, float, "reg_lambda_grid"),
        gamma_grid=resolve_sweep_grid(preset, args.gamma_grid, float, "gamma_grid"),
        max_runs=args.max_runs,
        save_predictions=args.save_predictions,
    )


def main() -> None:
    config = parse_args()
    run_sweep(config)


if __name__ == "__main__":
    main()
