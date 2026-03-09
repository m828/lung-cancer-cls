from pathlib import Path

from PIL import Image

from lung_cancer_cls.dataset import discover_iqothnccd_samples


def _write_img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=128).save(path)


def test_discovery(tmp_path: Path):
    _write_img(tmp_path / "Normal" / "a.png")
    _write_img(tmp_path / "Benign" / "b.jpg")
    _write_img(tmp_path / "Malignant" / "c.png")

    samples = discover_iqothnccd_samples(tmp_path)
    labels = sorted([s.label for s in samples])
    assert labels == [0, 1, 2]
