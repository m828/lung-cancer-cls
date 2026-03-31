from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.multimodal_ct_cnv import CTCNVTrainConfig, train_ct_cnv_model


def test_ct_cnv_multimodal_train_smoke(tmp_path: Path):
    ct_root = tmp_path / "ct"
    ct_root.mkdir(parents=True)
    gene_tsv = tmp_path / "gene.tsv"
    meta_csv = tmp_path / "meta.csv"

    gene_rows = []
    meta_rows = []
    split_plan = ["train", "train", "train", "train", "val", "val", "test", "test"]
    labels = ["健康对照", "健康对照", "肺癌", "肺癌", "健康对照", "肺癌", "健康对照", "肺癌"]

    for idx, (split, label_name) in enumerate(zip(split_plan, labels)):
        sample_id = f"S{idx}"
        rel_path = f"{sample_id}.npy"
        np.save(ct_root / rel_path, np.random.randn(6, 12, 12).astype("float32"))
        meta_rows.append(
            {
                "SampleID": sample_id,
                "样本类型": label_name,
                "CT_numpy_cloud路径": rel_path,
                "CT_train_val_split": split,
            }
        )
        gene_rows.append(
            {
                "gene_id": sample_id,
                "gene_label": label_name,
                "g1": idx * 0.1,
                "g2": (idx % 2) * 0.2,
                "g3": (idx % 3) * 0.3,
                "g4": (idx % 4) * 0.4,
            }
        )

    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    pd.DataFrame(gene_rows).to_csv(gene_tsv, sep="\t", index=False)

    cfg = CTCNVTrainConfig(
        data_root=tmp_path,
        metadata_csv=meta_csv,
        gene_tsv=gene_tsv,
        output_dir=tmp_path / "out",
        ct_root=ct_root,
        use_predefined_split=True,
        split_mode="train_val_test",
        epochs=1,
        batch_size=2,
        num_workers=0,
        cpu=True,
        ct_model="attention3d_cnn",
        depth_size=8,
        volume_hw=32,
        ct_feature_dim=16,
        gene_hidden_dim=8,
        fusion_hidden_dim=8,
        save_predictions=False,
    )
    metrics = train_ct_cnv_model(cfg)

    assert metrics["cohort_stats"]["num_total"] == 8
    assert metrics["cohort_stats"]["num_train"] == 4
    assert metrics["cohort_stats"]["num_val"] == 2
    assert metrics["cohort_stats"]["num_test"] == 2
    assert metrics["class_names"] == {0: "normal", 1: "malignant"}
    assert (tmp_path / "out" / "best_model.pt").exists()
