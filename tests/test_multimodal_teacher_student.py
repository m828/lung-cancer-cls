from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.multimodal_teacher_student import (
    MultiModalTrainConfig,
    StudentKDConfig,
    normalize_distill_methods,
    parse_distillation_method_weights,
    train_multimodal_model,
    train_student_kd,
)


def _write_tiny_multimodal_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ct_root = tmp_path / "ct"
    ct_root.mkdir(parents=True)
    meta_csv = tmp_path / "meta.csv"
    gene_tsv = tmp_path / "gene.tsv"
    text_tsv = tmp_path / "text.tsv"

    split_plan = ["train", "train", "train", "train", "val", "val", "test", "test"]
    labels = ["健康对照", "健康对照", "肺癌", "肺癌", "健康对照", "肺癌", "健康对照", "肺癌"]

    meta_rows = []
    gene_rows = []
    text_rows = []
    for idx, (split, label_name) in enumerate(zip(split_plan, labels)):
        sample_id = f"S{idx}"
        record_id = f"R{idx}"
        rel_path = f"{sample_id}.npy"
        ct_array = np.random.randn(6, 12, 12).astype("float32") + (1.0 if label_name == "肺癌" else 0.0)
        np.save(ct_root / rel_path, ct_array)
        meta_rows.append(
            {
                "SampleID": sample_id,
                "record_id": record_id,
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
                "g2": (idx % 2) * 0.2 + (0.8 if label_name == "肺癌" else 0.0),
                "g3": (idx % 3) * 0.3,
                "g4": (idx % 4) * 0.4,
            }
        )
        text_rows.append(
            {
                "record_id": record_id,
                "num__age": 40 + idx,
                "num__marker": 1.0 if label_name == "肺癌" else 0.0,
                "bert_0000": 0.1 * idx,
                "bert_0001": 1.0 if label_name == "肺癌" else -1.0,
                "bert_0002": -0.2 * idx,
                "bert_0003": 0.3 * (idx % 3),
            }
        )

    for offset, label_name in enumerate(["健康对照", "肺癌"], start=len(split_plan)):
        sample_id = f"S{offset}"
        record_id = f"R{offset}"
        rel_path = f"{sample_id}.npy"
        ct_array = np.random.randn(6, 12, 12).astype("float32") + (1.0 if label_name == "肺癌" else 0.0)
        np.save(ct_root / rel_path, ct_array)
        meta_rows.append(
            {
                "SampleID": sample_id,
                "record_id": record_id,
                "样本类型": label_name,
                "CT_numpy_cloud路径": rel_path,
                "CT_train_val_split": "train" if label_name == "健康对照" else "val",
            }
        )
        text_rows.append(
            {
                "record_id": record_id,
                "num__age": 60 + offset,
                "num__marker": 1.0 if label_name == "肺癌" else 0.0,
                "bert_0000": 0.1 * offset,
                "bert_0001": 1.0 if label_name == "肺癌" else -1.0,
                "bert_0002": -0.2 * offset,
                "bert_0003": 0.3 * (offset % 3),
            }
        )

    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    pd.DataFrame(gene_rows).to_csv(gene_tsv, sep="\t", index=False)
    pd.DataFrame(text_rows).to_csv(text_tsv, sep="\t", index=False)
    text_tsv.with_suffix(text_tsv.suffix + ".meta.json").write_text(
        json.dumps({"record_id_col": "record_id", "embedding_backend": "hash"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ct_root, meta_csv, gene_tsv, text_tsv


def test_multimodal_teacher_student_smoke(tmp_path: Path):
    ct_root, meta_csv, gene_tsv, text_tsv = _write_tiny_multimodal_fixture(tmp_path)

    text_metrics = train_multimodal_model(
        MultiModalTrainConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=tmp_path / "text_only",
            modalities=("text",),
            text_feature_tsv=text_tsv,
            use_predefined_split=True,
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            text_feature_dim=8,
            fusion_hidden_dim=8,
            save_predictions=False,
        )
    )
    assert text_metrics["family"] == "text_only"
    assert text_metrics["cohort_stats"]["num_total"] == 10
    assert (tmp_path / "text_only" / "best_model.pt").exists()

    teacher_metrics = train_multimodal_model(
        MultiModalTrainConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=tmp_path / "teacher",
            modalities=("ct", "cnv", "text"),
            ct_root=ct_root,
            gene_tsv=gene_tsv,
            text_feature_tsv=text_tsv,
            use_predefined_split=True,
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            ct_model="attention3d_cnn",
            depth_size=8,
            volume_hw=32,
            ct_feature_dim=16,
            text_feature_dim=8,
            gene_hidden_dim=8,
            fusion_hidden_dim=8,
            save_predictions=False,
        )
    )
    assert teacher_metrics["family"] == "ct_cnv_text"
    assert teacher_metrics["cohort_stats"]["num_total"] == 8
    assert (tmp_path / "teacher" / "best_model.pt").exists()
    assert (tmp_path / "teacher" / "split_manifest.csv").exists()

    ct_text_metrics = train_multimodal_model(
        MultiModalTrainConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=tmp_path / "ct_text_aligned",
            modalities=("ct", "text"),
            ct_root=ct_root,
            text_feature_tsv=text_tsv,
            reference_manifest=tmp_path / "teacher" / "split_manifest.csv",
            use_predefined_split=True,
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            ct_model="attention3d_cnn",
            depth_size=8,
            volume_hw=32,
            ct_feature_dim=16,
            text_feature_dim=8,
            fusion_hidden_dim=8,
            save_predictions=False,
        )
    )
    assert ct_text_metrics["family"] == "ct_text"
    assert ct_text_metrics["split_source"] == "reference_manifest"
    assert ct_text_metrics["cohort_stats"]["num_total"] == 8

    student_metrics = train_student_kd(
        StudentKDConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=tmp_path / "student_ct",
            modalities=("ct",),
            ct_root=ct_root,
            gene_tsv=gene_tsv,
            text_feature_tsv=text_tsv,
            teacher_run_dir=tmp_path / "teacher",
            use_predefined_split=True,
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            ct_model="attention3d_cnn",
            depth_size=8,
            volume_hw=32,
            ct_feature_dim=16,
            text_feature_dim=8,
            gene_hidden_dim=8,
            fusion_hidden_dim=8,
            save_predictions=False,
        )
    )
    assert student_metrics["family"] == "student_kd"
    assert student_metrics["teacher_modalities"] == ["ct", "cnv", "text"]
    assert student_metrics["split_source"] == "teacher_manifest"
    assert student_metrics["config"]["distill_methods"] == ["logits"]
    assert "train_kd_logits_loss" in student_metrics["history"][0]
    assert (tmp_path / "student_ct" / "best_model.pt").exists()

    student_ct_text_metrics = train_student_kd(
        StudentKDConfig(
            data_root=tmp_path,
            metadata_csv=meta_csv,
            output_dir=tmp_path / "student_ct_text_combo",
            modalities=("ct", "text"),
            ct_root=ct_root,
            gene_tsv=gene_tsv,
            text_feature_tsv=text_tsv,
            teacher_run_dir=tmp_path / "teacher",
            use_predefined_split=True,
            epochs=1,
            batch_size=2,
            num_workers=0,
            cpu=True,
            ct_model="attention3d_cnn",
            depth_size=8,
            volume_hw=32,
            ct_feature_dim=16,
            text_feature_dim=8,
            gene_hidden_dim=8,
            fusion_hidden_dim=8,
            distill_methods=("logits", "fused", "hint", "relation", "attention", "ct", "text"),
            distillation_method_weights={
                "logits": 1.0,
                "fused": 0.5,
                "hint": 0.5,
                "relation": 0.25,
                "attention": 0.25,
                "ct": 0.5,
                "text": 0.25,
            },
            distillation_feature_loss="cosine",
            distillation_normalize_features=True,
            save_predictions=False,
        )
    )
    assert student_ct_text_metrics["config"]["distill_methods"] == [
        "logits",
        "fused",
        "hint",
        "relation",
        "attention",
        "ct",
        "text",
    ]
    assert set(student_ct_text_metrics["config"]["distillation_method_weights"]) == {
        "logits",
        "fused",
        "hint",
        "relation",
        "attention",
        "ct",
        "text",
    }
    history_row = student_ct_text_metrics["history"][0]
    assert "train_kd_fused_loss" in history_row
    assert "train_kd_hint_loss" in history_row
    assert "train_kd_relation_loss" in history_row
    assert "train_kd_attention_loss" in history_row
    assert "train_kd_ct_loss" in history_row
    assert "train_kd_text_loss" in history_row


def test_distillation_method_helpers():
    assert normalize_distill_methods("logits,feature,ct_feature,text,rkd,at,hint") == (
        "logits",
        "fused",
        "ct",
        "text",
        "relation",
        "attention",
        "hint",
    )
    assert parse_distillation_method_weights("logits=2,feature=1,ct_feature=0.5,rkd=0.25,at=0.25,hint=0.75") == {
        "logits": 2.0,
        "fused": 1.0,
        "ct": 0.5,
        "relation": 0.25,
        "attention": 0.25,
        "hint": 0.75,
    }
