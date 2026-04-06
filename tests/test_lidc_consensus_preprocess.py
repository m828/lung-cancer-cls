from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lung_cancer_cls.dataset import LIDCIDRIDataset
from lung_cancer_cls.lidc_consensus_preprocess import (
    LIDCCropConfig,
    compute_crop_bounds,
    load_split_lookup,
    write_processed_manifests,
)


def test_load_split_lookup_requires_fold_for_multi_fold_manifest(tmp_path: Path):
    split_csv = tmp_path / "split_manifest.csv"
    pd.DataFrame(
        [
            {"sample_id": "sample_a", "fold": 0, "split": "train"},
            {"sample_id": "sample_a", "fold": 1, "split": "test"},
        ]
    ).to_csv(split_csv, index=False)

    with pytest.raises(RuntimeError, match="split_fold"):
        load_split_lookup(split_csv, split_fold=None)


def test_compute_crop_bounds_respects_minimum_context():
    row = {
        "x_min": 20,
        "x_max": 24,
        "y_min": 30,
        "y_max": 35,
        "z_min": 5.0,
        "z_max": 7.0,
        "z_center": 6.0,
    }
    z_coords = np.arange(0.0, 20.0, 1.0, dtype=np.float32)
    config = LIDCCropConfig(min_size_xy=32, min_size_z=8, context_scale=1.0)
    z_slice, y_slice, x_slice = compute_crop_bounds(row, (20, 128, 128), z_coords, config)

    assert (x_slice.stop - x_slice.start) >= 32
    assert (y_slice.stop - y_slice.start) >= 32
    assert (z_slice.stop - z_slice.start) >= 8


def test_processed_split_manifest_can_drive_lidc_discovery(tmp_path: Path):
    root = tmp_path / "lidc_3d"
    (root / "benign").mkdir(parents=True)
    (root / "malignant").mkdir(parents=True)
    np.save(root / "benign" / "sample_b.npy", np.zeros((8, 16, 16), dtype=np.float32))
    np.save(root / "malignant" / "sample_m.npy", np.ones((8, 16, 16), dtype=np.float32))

    split_lookup = {
        "sample_b": {"sample_id": "sample_b", "split": "train", "fold": 0, "patient_id": "LIDC-IDRI-0001"},
        "sample_m": {"sample_id": "sample_m", "split": "test", "fold": 0, "patient_id": "LIDC-IDRI-0002"},
    }
    processed_rows = [
        {
            "sample_id": "sample_b",
            "output_stem": "sample_b",
            "patient_id": "LIDC-IDRI-0001",
            "nodule_id": "n1",
            "class_name": "benign",
            "label": 1,
            "output_path": str(root / "benign" / "sample_b.npy"),
            "relative_path": "benign/sample_b.npy",
        },
        {
            "sample_id": "sample_m",
            "output_stem": "sample_m",
            "patient_id": "LIDC-IDRI-0002",
            "nodule_id": "n2",
            "class_name": "malignant",
            "label": 2,
            "output_path": str(root / "malignant" / "sample_m.npy"),
            "relative_path": "malignant/sample_m.npy",
        },
    ]
    outputs = write_processed_manifests(tmp_path, processed_rows, split_lookup)

    dataset = LIDCIDRIDataset.discover(
        root,
        use_3d=True,
        split_manifest_csv=outputs["processed_split_manifest_csv"],
        split_fold=0,
    )
    samples = dataset.get_samples()

    assert len(samples) == 2
    assert {sample.metadata["split"] for sample in samples if sample.metadata} == {"train", "test"}
    assert {sample.metadata["patient_id"] for sample in samples if sample.metadata} == {
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
    }
