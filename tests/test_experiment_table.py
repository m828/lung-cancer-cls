from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.experiment_table import export_experiment_table


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_export_experiment_table_smoke(tmp_path: Path):
    ct_dir = tmp_path / "ct_only_run"
    mm_dir = tmp_path / "ct_cnv_run"

    _write_json(
        ct_dir / "metrics.json",
        {
            "dataset_type": "INTRANET_CT",
            "best_epoch": 7,
            "selection_metric": "auroc",
            "best_val_metrics": {"auroc": 0.91, "balanced_accuracy": 0.81, "f1": 0.85, "sensitivity": 0.84, "specificity": 0.78},
            "test_metrics": {"auroc": 0.90, "balanced_accuracy": 0.80, "f1": 0.83, "sensitivity": 0.82, "specificity": 0.77},
            "config": {"model": "resnet3d18", "class_mode": "binary", "binary_task": "malignant_vs_normal", "use_3d_input": True},
            "class_names": {"0": "normal", "1": "malignant"},
        },
    )
    _write_json(
        mm_dir / "metrics.json",
        {
            "best_epoch": 5,
            "selection_metric": "auroc",
            "split_source": "predefined",
            "feature_dim": 128,
            "cohort_stats": {"num_total": 100, "num_train": 70, "num_val": 15, "num_test": 15},
            "best_val_metrics": {"auroc": 0.94, "balanced_accuracy": 0.86, "f1": 0.88, "sensitivity": 0.89, "specificity": 0.83},
            "test_metrics": {"auroc": 0.93, "balanced_accuracy": 0.84, "f1": 0.87, "sensitivity": 0.88, "specificity": 0.82},
            "config": {"ct_model": "resnet3d18", "class_mode": "binary", "binary_task": "malignant_vs_normal", "use_3d_input": True, "gene_hidden_dim": 256},
        },
    )

    summary = export_experiment_table(
        run_specs=[f"ct_only={ct_dir}", f"teacher={mm_dir}"],
        run_dirs=[],
        output_dir=tmp_path / "table_out",
    )

    assert summary["num_runs"] == 2
    assert (tmp_path / "table_out" / "experiment_table.csv").exists()
    assert (tmp_path / "table_out" / "experiment_table.md").exists()
