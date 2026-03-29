from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.cnv_xgboost import (
    CNVXGBoostConfig,
    build_cohort_table,
    deduplicate_gene_rows,
    load_gene_feature_table,
    split_cohort,
)


def test_gene_feature_loading_and_nan_fill(tmp_path: Path):
    gene_tsv = tmp_path / "gene.tsv"
    pd.DataFrame(
        [
            {"gene_id": "S1", "gene_label": "malignant", "f1": 1.0, "f2": np.nan},
            {"gene_id": "S2", "gene_label": "normal", "f1": np.nan, "f2": np.nan},
        ]
    ).to_csv(gene_tsv, sep="\t", index=False)

    gene_df, gene_id_col, feature_names, inferred_label_col = load_gene_feature_table(gene_tsv)

    assert gene_id_col == "gene_id"
    assert inferred_label_col == "gene_label"
    assert feature_names == ["f1", "f2"]
    assert np.isfinite(gene_df[feature_names].to_numpy()).all()


def test_build_cohort_and_deduplicate_conflicts(tmp_path: Path):
    gene_tsv = tmp_path / "gene.tsv"
    pd.DataFrame(
        [
            {"gene_id": "S1", "gene_label": "malignant", "f1": 1.0, "f2": 0.1},
            {"gene_id": "S2", "gene_label": "normal", "f1": 0.2, "f2": 0.3},
            {"gene_id": "S3", "gene_label": "malignant", "f1": 0.8, "f2": 0.4},
        ]
    ).to_csv(gene_tsv, sep="\t", index=False)

    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"SampleID": "S1", "样本类型": "肺癌", "CT_train_val_split": "train"},
            {"SampleID": "S1", "样本类型": "肺癌", "CT_train_val_split": "train"},
            {"SampleID": "S2", "样本类型": "健康对照", "CT_train_val_split": "test"},
            {"SampleID": "S3", "样本类型": "肺癌", "CT_train_val_split": "train"},
            {"SampleID": "S3", "样本类型": "肺癌", "CT_train_val_split": "test"},
        ]
    ).to_csv(meta_csv, index=False)

    cfg = CNVXGBoostConfig(
        metadata_csv=meta_csv,
        gene_tsv=gene_tsv,
        output_dir=tmp_path / "out",
        use_predefined_split=True,
    )

    gene_df, _, _, _ = load_gene_feature_table(gene_tsv)
    cohort, class_names, stats = build_cohort_table(meta_csv, gene_df, cfg)
    deduped, dedup_stats = deduplicate_gene_rows(cohort)

    assert class_names == {0: "normal", 1: "malignant"}
    assert stats["metadata_rows"] == 5
    assert dedup_stats["duplicate_rows_collapsed"] == 2
    assert dedup_stats["duplicate_gene_split_conflicts_dropped"] == 1
    assert len(deduped) == 2
    assert sorted(deduped["gene_id"].tolist()) == ["S1", "S2"]


def test_predefined_split_creates_val_from_train(tmp_path: Path):
    gene_tsv = tmp_path / "gene.tsv"
    gene_rows = []
    meta_rows = []
    for idx in range(10):
        sample_id = f"S{idx}"
        label_name = "肺癌" if idx % 2 else "健康对照"
        gene_rows.append({"gene_id": sample_id, "gene_label": label_name, "f1": idx, "f2": idx + 1})
        meta_rows.append({"SampleID": sample_id, "样本类型": label_name, "CT_train_val_split": "train" if idx < 8 else "test"})

    pd.DataFrame(gene_rows).to_csv(gene_tsv, sep="\t", index=False)
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    cfg = CNVXGBoostConfig(
        metadata_csv=meta_csv,
        gene_tsv=gene_tsv,
        output_dir=tmp_path / "out",
        use_predefined_split=True,
        split_mode="train_val_test",
        val_ratio=0.25,
    )

    gene_df, _, _, _ = load_gene_feature_table(gene_tsv)
    cohort, _, _ = build_cohort_table(meta_csv, gene_df, cfg)
    deduped, _ = deduplicate_gene_rows(cohort)
    train_idx, val_idx, test_idx, split_source = split_cohort(deduped, cfg)

    assert split_source == "predefined"
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) == 2
