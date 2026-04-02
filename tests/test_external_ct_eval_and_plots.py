from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.evaluate_external_ct import ExternalCTEvalConfig, evaluate_external_ct_bundle
from lung_cancer_cls.multimodal_teacher_student import MultiModalTrainConfig, train_multimodal_model
from lung_cancer_cls.visualize_experiment import export_experiment_plots


def test_external_ct_eval_and_plots_smoke(tmp_path: Path):
    ct_root = tmp_path / "ct"
    ct_root.mkdir(parents=True)
    meta_csv = tmp_path / "meta.csv"

    meta_rows = []
    labels = ["健康对照", "健康对照", "肺癌", "肺癌", "健康对照", "肺癌"]
    splits = ["train", "train", "train", "train", "val", "val"]
    for idx, (label_name, split) in enumerate(zip(labels, splits)):
        rel_path = f"S{idx}.npy"
        np.save(ct_root / rel_path, np.random.randn(6, 12, 12).astype("float32"))
        meta_rows.append(
            {
                "SampleID": f"S{idx}",
                "record_id": f"R{idx}",
                "样本类型": label_name,
                "CT_numpy_cloud路径": rel_path,
                "CT_train_val_split": split,
            }
        )
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    run_dir = tmp_path / "ct_only_run"
    metrics = train_multimodal_model(
        MultiModalTrainConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=run_dir,
            modalities=("ct",),
            ct_root=ct_root,
            use_predefined_split=True,
            split_mode="train_val",
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            ct_model="attention3d_cnn",
            depth_size=8,
            volume_hw=32,
            ct_feature_dim=16,
            fusion_hidden_dim=8,
            save_predictions=True,
        )
    )
    assert metrics["family"] == "ct_only"

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    np.save(processed / "NM_all.npy", np.random.randn(2, 6, 12, 12).astype("float32"))
    np.save(processed / "BN_all.npy", np.random.randn(1, 6, 12, 12).astype("float32"))
    np.save(processed / "MT_all.npy", np.random.randn(2, 6, 12, 12).astype("float32"))

    external = evaluate_external_ct_bundle(
        ExternalCTEvalConfig(
            run_dir=run_dir,
            output_dir=tmp_path / "bundle_eval",
            data_root=tmp_path,
            bundle_nm_path=processed / "NM_all.npy",
            bundle_bn_path=processed / "BN_all.npy",
            bundle_mt_path=processed / "MT_all.npy",
            class_mode="binary",
            binary_task="malignant_vs_normal",
            batch_size=2,
            num_workers=0,
            cpu=True,
        )
    )
    assert external["external_dataset"] == "intranet_bundle"
    assert (tmp_path / "bundle_eval" / "external_bundle_metrics.json").exists()
    assert (tmp_path / "bundle_eval" / "external_bundle_predictions.csv").exists()

    plot_out = export_experiment_plots(run_dir, tmp_path / "plots", split="val")
    assert "history_curves.png" in plot_out["created_files"]
    assert (tmp_path / "plots" / "plot_summary.json").exists()
