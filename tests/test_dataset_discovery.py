from pathlib import Path

import pytest
import pandas as pd
from PIL import Image

from lung_cancer_cls.dataset import DatasetType, create_dataset
from lung_cancer_cls.train import TrainConfig, train_model


def _write_img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=128).save(path)


def test_iqothnccd_discovery(tmp_path: Path):
    """测试 IQ-OTH/NCCD 数据集发现"""
    _write_img(tmp_path / "Normal" / "a.png")
    _write_img(tmp_path / "Benign" / "b.jpg")
    _write_img(tmp_path / "Malignant" / "c.png")

    dataset = create_dataset(DatasetType.IQ_OTHNCCD, tmp_path)
    samples = dataset.get_samples()
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]


def test_iqothnccd_directory_structure(tmp_path: Path):
    """测试不同的目录结构"""
    nested_dir = tmp_path / "subdir" / "data"
    _write_img(nested_dir / "Normal cases" / "x.png")
    _write_img(nested_dir / "Benign cases" / "y.jpg")
    _write_img(nested_dir / "Malignant cases" / "z.png")

    dataset = create_dataset(DatasetType.IQ_OTHNCCD, nested_dir)
    samples = dataset.get_samples()
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]


def test_luna16_extracted_discovery(tmp_path: Path):
    """测试 LUNA16 提取切片目录发现"""
    _write_img(tmp_path / "normal" / "n1.png")
    _write_img(tmp_path / "benign" / "b1.png")
    _write_img(tmp_path / "malignant" / "m1.png")

    dataset = create_dataset(DatasetType.LUNA16, tmp_path)
    samples = dataset.get_samples()
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]


def test_luna16_raw_dir_without_extracted_slices_raises(tmp_path: Path):
    """测试原始 LUNA16 目录但缺少 extracted_slices 时抛错"""
    (tmp_path / "annotations.csv").write_text("seriesuid,coordX,coordY,coordZ,diameter_mm\n")
    (tmp_path / "subset0").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="No extracted slices found"):
        create_dataset(DatasetType.LUNA16, tmp_path)


def test_lidc_idri_discovery(tmp_path: Path):
    """测试 LIDC-IDRI 目录发现。"""
    _write_img(tmp_path / "normal" / "n1.png")
    _write_img(tmp_path / "benign" / "b1.png")
    _write_img(tmp_path / "malignant" / "m1.png")

    dataset = create_dataset(DatasetType.LIDC_IDRI, tmp_path)
    samples = dataset.get_samples()
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]


def test_lidc_idri_3d_discovery_and_loading(tmp_path: Path):
    import numpy as np

    for cls_name in ["normal", "benign", "malignant"]:
        d = tmp_path / cls_name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{cls_name}.npy", np.random.randn(8, 16, 16).astype("float32"))

    dataset = create_dataset(DatasetType.LIDC_IDRI, tmp_path, use_3d=True, depth_size=12)
    x, y = dataset[0]
    assert x.shape == (1, 12, 128, 128)
    assert y in [0, 1, 2]


def test_intranet_ct_discovery(tmp_path: Path):
    ct_root = tmp_path / "ct_root"
    ct_root.mkdir()
    npy_path = ct_root / "a.npy"
    import numpy as np
    np.save(npy_path, np.random.randn(16, 16).astype("float32"))

    metadata_csv = tmp_path / "meta.csv"
    pd.DataFrame([
        {"CT_numpy_cloud路径": "a.npy", "样本类型": "肺癌", "CT_train_val_split": "train"},
    ]).to_csv(metadata_csv, index=False)

    dataset = create_dataset(
        DatasetType.INTRANET_CT,
        tmp_path,
        metadata_csv=metadata_csv,
        ct_root=ct_root,
    )
    samples = dataset.get_samples()
    assert len(samples) == 1
    assert samples[0].label == 2


def test_intranet_ct_predefined_split_fallback(tmp_path: Path):
    ct_root = tmp_path / "ct_root"
    ct_root.mkdir()
    import numpy as np
    for idx, cls in enumerate(["健康对照", "良性结节", "肺癌"]):
        np.save(ct_root / f"{idx}.npy", np.random.randn(16, 16).astype("float32"))

    metadata_csv = tmp_path / "meta.csv"
    pd.DataFrame([
        {"CT_numpy_cloud路径": "0.npy", "样本类型": "健康对照", "CT_train_val_split": ""},
        {"CT_numpy_cloud路径": "1.npy", "样本类型": "良性结节", "CT_train_val_split": ""},
        {"CT_numpy_cloud路径": "2.npy", "样本类型": "肺癌", "CT_train_val_split": ""},
    ]).to_csv(metadata_csv, index=False)

    cfg = TrainConfig(
        dataset_type=DatasetType.INTRANET_CT,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        epochs=1,
        batch_size=1,
        num_workers=0,
        cpu=True,
        metadata_csv=metadata_csv,
        ct_root=ct_root,
        use_predefined_split=True,
    )
    metrics = train_model(cfg)
    assert "test_acc" in metrics


def test_intranet_ct_bundle_discovery(tmp_path: Path):
    import numpy as np

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    np.save(processed / "NM_all.npy", np.random.randn(3, 16, 16).astype("float32"))
    np.save(processed / "BN_all.npy", np.random.randn(2, 16, 16).astype("float32"))
    np.save(processed / "MT_all.npy", np.random.randn(4, 16, 16).astype("float32"))

    ds_bundle = create_dataset(DatasetType.INTRANET_CT, tmp_path, intranet_source="bundle")
    assert len(ds_bundle.get_samples()) == 9

    ct_root = tmp_path / "ct_root"
    ct_root.mkdir(parents=True)
    np.save(ct_root / "a.npy", np.random.randn(16, 16).astype("float32"))
    meta = tmp_path / "meta.csv"
    pd.DataFrame([
        {"CT_numpy_cloud路径": "a.npy", "样本类型": "肺癌", "CT_train_val_split": "train"},
    ]).to_csv(meta, index=False)

    ds_both = create_dataset(
        DatasetType.INTRANET_CT,
        tmp_path,
        intranet_source="both",
        metadata_csv=meta,
        ct_root=ct_root,
    )
    assert len(ds_both.get_samples()) == 10


def test_intranet_ct_3d_volume_hw(tmp_path: Path):
    import numpy as np

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    np.save(processed / "NM_all.npy", np.random.randn(2, 8, 16, 16).astype("float32"))
    np.save(processed / "BN_all.npy", np.random.randn(2, 8, 16, 16).astype("float32"))
    np.save(processed / "MT_all.npy", np.random.randn(2, 8, 16, 16).astype("float32"))

    dataset = create_dataset(
        DatasetType.INTRANET_CT,
        tmp_path,
        intranet_source="bundle",
        use_3d=True,
        depth_size=10,
        volume_hw=64,
    )
    x, y = dataset[0]
    assert x.shape == (1, 10, 64, 64)
    assert y in [0, 1, 2]


def test_intranet_ct_bundle_directory_discovery(tmp_path: Path):
    import numpy as np

    raw = tmp_path / "raw"
    nm_dir = raw / "nm" / "npy_128_256_256_fix"
    bn_dir = raw / "bn" / "npy_128_256_256"
    mt_dir = raw / "mt" / "npy_128_256_256"
    nm_dir.mkdir(parents=True)
    bn_dir.mkdir(parents=True)
    mt_dir.mkdir(parents=True)

    np.save(nm_dir / "nm_0.npy", np.random.randn(8, 16, 16).astype("float32"))
    np.save(nm_dir / "nm_1.npy", np.random.randn(8, 16, 16).astype("float32"))
    np.save(bn_dir / "bn_0.npy", np.random.randn(8, 16, 16).astype("float32"))
    np.save(mt_dir / "mt_0.npy", np.random.randn(8, 16, 16).astype("float32"))
    np.save(mt_dir / "mt_1.npy", np.random.randn(8, 16, 16).astype("float32"))
    np.save(mt_dir / "mt_2.npy", np.random.randn(8, 16, 16).astype("float32"))

    ds = create_dataset(
        DatasetType.INTRANET_CT,
        tmp_path,
        intranet_source="bundle",
        bundle_nm_path=nm_dir,
        bundle_bn_path=bn_dir,
        bundle_mt_path=mt_dir,
    )
    samples = ds.get_samples()
    assert len(samples) == 6
    labels = sorted(s.label for s in samples)
    assert labels == [0, 0, 1, 2, 2, 2]

    x, y = ds[0]
    assert y in [0, 1, 2]
    assert tuple(x.shape[-2:]) == (224, 224)


def test_binary_class_mode_train_smoke(tmp_path: Path):
    import numpy as np

    for cls_name in ["normal", "benign", "malignant"]:
        d = tmp_path / cls_name
        d.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            np.save(d / f"{cls_name}_{idx}.npy", np.random.randn(8, 16, 16).astype("float32"))

    cfg = TrainConfig(
        dataset_type=DatasetType.LIDC_IDRI,
        data_root=tmp_path,
        output_dir=tmp_path / "out_binary",
        epochs=1,
        batch_size=2,
        num_workers=0,
        cpu=True,
        use_3d_input=True,
        depth_size=8,
        volume_hw=32,
        model="attention3d_cnn",
        class_mode="binary",
        binary_task="malignant_vs_rest",
        split_mode="train_val",
    )
    metrics = train_model(cfg)
    assert metrics["config"]["class_mode"] == "binary"
    assert metrics["config"]["binary_task"] == "malignant_vs_rest"
    assert metrics["class_names"] == {0: "non_malignant", 1: "malignant"}
