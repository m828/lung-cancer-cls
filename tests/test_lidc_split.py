from pathlib import Path
import json

import pandas as pd

from lung_cancer_cls.lidc_split import (
    LIDCSplitConfig,
    build_and_write_lidc_splits,
    build_manifest_from_xml,
)


def test_build_lidc_splits_from_metadata(tmp_path: Path):
    patient_root = tmp_path / "LIDC-IDRI" / "LIDC-IDRI"
    for patient_id in ["LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003", "LIDC-IDRI-0004"]:
        (patient_root / patient_id).mkdir(parents=True, exist_ok=True)

    metadata_csv = tmp_path / "LIDC-IDRI" / "metadata.csv"
    pd.DataFrame(
        [
            {"Subject ID": "LIDC-IDRI-0001", "Annotation ID": "N1", "malignancy_1": 1, "malignancy_2": 2},
            {"Subject ID": "LIDC-IDRI-0002", "Annotation ID": "N2", "malignancy_1": 4, "malignancy_2": 5},
            {"Subject ID": "LIDC-IDRI-0003", "Annotation ID": "N3", "malignancy_1": 2, "malignancy_2": 2},
            {"Subject ID": "LIDC-IDRI-0004", "Annotation ID": "N4", "malignancy_1": 5, "malignancy_2": 4},
            {"Subject ID": "LIDC-IDRI-0004", "Annotation ID": "N5", "malignancy_1": 3, "malignancy_2": 3},
        ]
    ).to_csv(metadata_csv, index=False)

    config = LIDCSplitConfig(
        input_root=tmp_path / "LIDC-IDRI",
        output_dir=tmp_path / "out",
        metadata_csv=metadata_csv,
        metadata_source="csv",
        split_scheme="patient_kfold",
        n_splits=2,
        val_ratio=0.25,
        seed=7,
    )

    outputs = build_and_write_lidc_splits(config)
    manifest = pd.read_csv(outputs["manifest_csv"])
    split_manifest = pd.read_csv(outputs["split_manifest_csv"])
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))

    assert set(manifest["class_name"].tolist()) == {"benign", "malignant"}
    assert "LIDC-IDRI-0004__N5" not in set(manifest["sample_id"].tolist())
    assert manifest["patient_dir"].str.contains("LIDC-IDRI-000").all()

    for fold, fold_df in split_manifest.groupby("fold"):
        patient_splits = fold_df.groupby("patient_id")["split"].nunique()
        assert patient_splits.max() == 1
        assert set(fold_df["split"].tolist()) <= {"train", "val", "test"}

    assert summary["num_samples"] == 4


def test_build_lidc_manifest_from_xml_fallback(tmp_path: Path):
    raw_root = tmp_path / "LIDC-IDRI" / "LIDC-IDRI"
    xml_1 = raw_root / "LIDC-IDRI-0001" / "study1" / "series1" / "annot.xml"
    xml_2 = raw_root / "LIDC-IDRI-0002" / "study1" / "series1" / "annot.xml"
    xml_1.parent.mkdir(parents=True, exist_ok=True)
    xml_2.parent.mkdir(parents=True, exist_ok=True)

    xml_template = """<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage xmlns="http://www.nih.gov">
  <readingSession>
    <unblindedReadNodule>
      <noduleID>{nodule_id}</noduleID>
      <characteristics>
        <malignancy>{score}</malignancy>
      </characteristics>
      <roi>
        <imageZposition>1.0</imageZposition>
      </roi>
    </unblindedReadNodule>
  </readingSession>
</LidcReadMessage>
"""
    xml_1.write_text(xml_template.format(nodule_id="N1", score="1"), encoding="utf-8")
    xml_2.write_text(xml_template.format(nodule_id="N2", score="5"), encoding="utf-8")

    config = LIDCSplitConfig(
        input_root=tmp_path / "LIDC-IDRI",
        output_dir=tmp_path / "out_xml",
        metadata_source="xml",
        split_scheme="patient_holdout",
        train_ratio=0.5,
        val_ratio=0.25,
        seed=3,
    )
    manifest = build_manifest_from_xml(config)

    assert len(manifest) == 2
    assert set(manifest["class_name"].tolist()) == {"benign", "malignant"}
    assert manifest["xml_path"].str.endswith("annot.xml").all()
