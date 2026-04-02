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
    text_dir = tmp_path / "text_only_run"
    student_dir = tmp_path / "student_kd_run"
    external_dir = tmp_path / "bundle_eval_run"

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
            "family": "ct_cnv_text",
            "best_epoch": 5,
            "selection_metric": "auroc",
            "split_source": "predefined",
            "modality_feature_dims": {"ct": 128, "text": 256, "cnv": 128},
            "cohort_stats": {"num_total": 100, "num_train": 70, "num_val": 15, "num_test": 15},
            "best_val_metrics": {"auroc": 0.94, "balanced_accuracy": 0.86, "f1": 0.88, "sensitivity": 0.89, "specificity": 0.83},
            "test_metrics": {"auroc": 0.93, "balanced_accuracy": 0.84, "f1": 0.87, "sensitivity": 0.88, "specificity": 0.82},
            "modalities": ["ct", "cnv", "text"],
            "config": {"ct_model": "resnet3d18", "class_mode": "binary", "binary_task": "malignant_vs_normal", "use_3d_input": True, "gene_hidden_dim": 256, "modalities": ["ct", "cnv", "text"]},
        },
    )
    _write_json(
        text_dir / "metrics.json",
        {
            "family": "text_only",
            "best_epoch": 4,
            "selection_metric": "auroc",
            "cohort_stats": {"num_total": 80, "num_train": 56, "num_val": 12, "num_test": 12},
            "best_val_metrics": {"auroc": 0.88, "balanced_accuracy": 0.78, "f1": 0.79},
            "test_metrics": {"auroc": 0.86, "balanced_accuracy": 0.77, "f1": 0.76},
            "modalities": ["text"],
            "modality_feature_dims": {"text": 256},
            "config": {"modalities": ["text"], "class_mode": "binary", "binary_task": "malignant_vs_normal"},
        },
    )
    _write_json(
        student_dir / "metrics.json",
        {
            "family": "student_kd",
            "best_epoch": 6,
            "selection_metric": "auroc",
            "cohort_stats": {"num_total": 100, "num_train": 70, "num_val": 15, "num_test": 15},
            "best_val_metrics": {"auroc": 0.92, "balanced_accuracy": 0.84, "f1": 0.86},
            "test_metrics": {"auroc": 0.91, "balanced_accuracy": 0.83, "f1": 0.85},
            "modalities": ["ct"],
            "teacher_modalities": ["ct", "cnv", "text"],
            "teacher_run_dir": str(mm_dir),
            "modality_feature_dims": {"ct": 128},
            "config": {"ct_model": "resnet3d18", "modalities": ["ct"], "class_mode": "binary", "binary_task": "malignant_vs_normal", "use_3d_input": True},
        },
    )
    _write_json(
        external_dir / "external_bundle_metrics.json",
        {
            "source_run_dir": str(student_dir),
            "source_family": "student_kd",
            "external_dataset": "intranet_bundle",
            "num_samples": 120,
            "metrics": {"auroc": 0.89, "balanced_accuracy": 0.81, "f1": 0.83, "sensitivity": 0.82, "specificity": 0.80},
            "config": {"run_dir": str(student_dir)},
        },
    )

    summary = export_experiment_table(
        run_specs=[f"ct_only={ct_dir}", f"teacher={mm_dir}", f"text_only={text_dir}", f"student={student_dir}", f"student_bundle={external_dir / 'external_bundle_metrics.json'}"],
        run_dirs=[],
        output_dir=tmp_path / "table_out",
    )

    assert summary["num_runs"] == 5
    assert set(summary["families"]) >= {"ct_cnv_text", "student_kd", "text_only", "student_kd_external"}
    assert (tmp_path / "table_out" / "experiment_table.csv").exists()
    assert (tmp_path / "table_out" / "experiment_table.md").exists()
