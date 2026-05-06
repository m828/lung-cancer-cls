import torch

from lung_cancer_cls.model import build_model


def test_build_densenet3d_forward_smoke():
    model = build_model("densenet3d", num_classes=4, pretrained=False)
    x = torch.randn(2, 1, 32, 64, 64)
    y = model(x)
    assert y.shape == (2, 4)
