from pathlib import Path

from PIL import Image

from lung_cancer_cls.dataset import DatasetType, create_dataset


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
    # 支持更深的目录结构
    nested_dir = tmp_path / "subdir" / "data"
    _write_img(nested_dir / "Normal cases" / "x.png")
    _write_img(nested_dir / "Benign cases" / "y.jpg")
    _write_img(nested_dir / "Malignant cases" / "z.png")

    dataset = create_dataset(DatasetType.IQ_OTHNCCD, nested_dir)
    samples = dataset.get_samples()
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]
