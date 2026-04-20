from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.batch_experiment_runner import (
    load_batch_runner_config,
    run_batch_experiments,
    write_template_config,
)


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_write_template_config_smoke(tmp_path: Path):
    out = write_template_config(tmp_path / "batch_template.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["experiments"]
    assert data["experiments"][0]["name"]


def test_batch_runner_report_only_generates_reports(tmp_path: Path):
    run_a = tmp_path / "outputs" / "ct_binary"
    run_b = tmp_path / "outputs" / "ct_multiclass"

    _write_json(
        run_a / "metrics.json",
        {
            "family": "ct_only",
            "best_epoch": 3,
            "selection_metric": "auroc",
            "split_source": "predefined",
            "cohort_stats": {"num_total": 12, "num_train": 8, "num_val": 2, "num_test": 2},
            "best_val_metrics": {"auroc": 0.92, "balanced_accuracy": 0.85, "f1": 0.86},
            "test_metrics": {
                "auroc": 0.91,
                "balanced_accuracy": 0.84,
                "f1": 0.85,
                "accuracy": 0.83,
                "auprc": 0.9,
            },
            "class_names": {"0": "normal", "1": "malignant"},
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 0.6,
                    "val_loss": 0.4,
                    "val_auroc": 0.8,
                    "val_balanced_accuracy": 0.7,
                    "val_f1": 0.72,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.4,
                    "val_loss": 0.3,
                    "val_auroc": 0.92,
                    "val_balanced_accuracy": 0.85,
                    "val_f1": 0.86,
                },
            ],
            "config": {
                "model": "resnet3d18",
                "class_mode": "binary",
                "binary_task": "malignant_vs_normal",
                "use_3d_input": True,
            },
        },
    )
    pd.DataFrame(
        [
            {"label": 0, "prediction": 0, "prob_normal": 0.9, "prob_malignant": 0.1},
            {"label": 1, "prediction": 1, "prob_normal": 0.2, "prob_malignant": 0.8},
            {"label": 1, "prediction": 1, "prob_normal": 0.1, "prob_malignant": 0.9},
            {"label": 0, "prediction": 1, "prob_normal": 0.3, "prob_malignant": 0.7},
        ]
    ).to_csv(run_a / "test_predictions.csv", index=False)

    _write_json(
        run_b / "metrics.json",
        {
            "family": "ct_only",
            "best_epoch": 4,
            "selection_metric": "balanced_accuracy",
            "split_source": "predefined",
            "cohort_stats": {"num_total": 18, "num_train": 12, "num_val": 3, "num_test": 3},
            "best_val_metrics": {"auroc": 0.75, "balanced_accuracy": 0.63, "f1": 0.51},
            "test_metrics": {"auroc": 0.73, "balanced_accuracy": 0.61, "f1": 0.49, "accuracy": 0.67},
            "class_names": {"0": "normal", "1": "benign", "2": "malignant"},
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 0.9,
                    "val_loss": 0.8,
                    "val_auroc": 0.7,
                    "val_balanced_accuracy": 0.55,
                    "val_f1": 0.42,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.8,
                    "val_loss": 0.7,
                    "val_auroc": 0.75,
                    "val_balanced_accuracy": 0.63,
                    "val_f1": 0.51,
                },
            ],
            "config": {
                "model": "swin3d_tiny",
                "class_mode": "multiclass",
                "use_3d_input": True,
            },
        },
    )

    config_path = tmp_path / "batch_config.json"
    config_path.write_text(
        json.dumps(
            {
                "report_dir": "reports",
                "export_table": True,
                "export_plots": True,
                "experiments": [
                    {
                        "name": "ct_binary",
                        "entrypoint": "train.py",
                        "args": ["--output-dir", str(run_a)],
                        "run_dir": str(run_a),
                        "plot_split": "test",
                    },
                    {
                        "name": "ct_multiclass",
                        "entrypoint": "train.py",
                        "args": ["--output-dir", str(run_b)],
                        "run_dir": str(run_b),
                        "plot_split": "test",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = load_batch_runner_config(config_path)
    summary = run_batch_experiments(
        config=config,
        config_base_dir=config_path.parent,
        project_root=ROOT,
        report_only=True,
    )

    report_dir = tmp_path / "reports"
    assert (report_dir / "batch_summary.json").exists()
    assert (report_dir / "batch_results.csv").exists()
    assert (report_dir / "table" / "experiment_table.csv").exists()
    assert (report_dir / "batch_test_metrics.png").exists()
    assert summary["num_runs"] == 2
    table_df = pd.read_csv(report_dir / "table" / "experiment_table.csv")
    assert set(table_df["experiment"]) == {"ct_binary", "ct_multiclass"}
    if "plot_warning" not in summary:
        assert (report_dir / "plots" / "ct_binary" / "history_curves.png").exists()
