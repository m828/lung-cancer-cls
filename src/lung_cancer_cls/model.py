from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models

try:
    import monai
except ImportError:  # pragma: no cover - optional dependency
    monai = None


class SEModule(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """空间注意力模块。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    """CBAM（通道 + 空间）注意力模块。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = SEModule(channels=channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        return self.spatial(x)


class SEModule3D(nn.Module):
    """3D Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1, 1)
        return x * weights


class Attention3DCNNClassifier(nn.Module):
    """轻量 Attention 3D CNN（DeepLung 风格模块化替代）。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEModule3D(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            SEModule3D(128),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def _adapt_video_stem_to_single_channel(net: nn.Module, pretrained: bool) -> nn.Module:
    """将 torchvision.video 网络 stem 第一层改为单通道。"""
    original_conv = net.stem[0]
    net.stem[0] = nn.Conv3d(
        1,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            net.stem[0].weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
    return net


def _adapt_swin3d_patch_embed_to_single_channel(net: nn.Module, pretrained: bool) -> nn.Module:
    """将 Swin3D patch embedding 改为单通道。"""
    original_conv = net.patch_embed.proj
    net.patch_embed.proj = nn.Conv3d(
        1,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            net.patch_embed.proj.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
    return net


class SimpleCTClassifier(nn.Module):
    """A lightweight 2D CNN baseline for CT slice classification."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class Backbone2DClassifier(nn.Module):
    """通用 2D backbone 分类器，支持可选注意力模块。"""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: int = 3,
        attention: Literal["none", "se", "cbam"] = "none",
    ):
        super().__init__()
        self.backbone = backbone
        if attention == "se":
            self.attention = SEModule(in_features)
        elif attention == "cbam":
            self.attention = CBAMBlock(in_features)
        else:
            self.attention = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.attention(feat)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


class ResNet18CTClassifier(nn.Module):
    """ResNet18 adapted for single-channel CT slice classification."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = False,
        attention: Literal["none", "se", "cbam"] = "none",
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                resnet.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))

        backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = Backbone2DClassifier(backbone, in_features=512, num_classes=num_classes, attention=attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class EfficientNetB0CTClassifier(nn.Module):
    """EfficientNet-B0，适合小样本医学图像微调。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        original_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                effnet.features[0][0].weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        in_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = effnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ConvNeXtTinyCTClassifier(nn.Module):
    """ConvNeXt-Tiny，近两年常见强基线之一。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        original_conv = convnext.features[0][0]
        convnext.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                convnext.features[0][0].weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        in_features = convnext.classifier[2].in_features
        convnext.classifier[2] = nn.Linear(in_features, num_classes)
        self.backbone = convnext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet3D18CTClassifier(nn.Module):
    """3D ResNet18（r3d_18）用于体数据输入 [B, 1, D, H, W]。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        net = models.video.r3d_18(weights=weights)

        net = _adapt_video_stem_to_single_channel(net, pretrained=pretrained)

        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MC3_18CTClassifier(nn.Module):
    """MC3-18：3D ResNet 变体。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.video.MC3_18_Weights.KINETICS400_V1 if pretrained else None
        net = models.video.mc3_18(weights=weights)
        net = _adapt_video_stem_to_single_channel(net, pretrained=pretrained)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class R2Plus1D18CTClassifier(nn.Module):
    """R(2+1)D-18：时空解耦卷积，常用于医学 3D 迁移。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        net = models.video.r2plus1d_18(weights=weights)
        net = _adapt_video_stem_to_single_channel(net, pretrained=pretrained)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Swin3DTinyCTClassifier(nn.Module):
    """Swin3D-Tiny（Video Swin Transformer）适配 CT 体数据。"""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.video.Swin3D_T_Weights.KINETICS400_IMAGENET22K_V1 if pretrained else None
        net = models.video.swin3d_t(weights=weights)
        net = _adapt_swin3d_patch_embed_to_single_channel(net, pretrained=pretrained)
        in_features = net.head.in_features
        net.head = nn.Linear(in_features, num_classes)
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet3DCTClassifier(nn.Module):
    """轻量 3D DenseNet（适合医学 CT 体数据的参数高效基线）。"""

    def __init__(self, num_classes: int = 3, growth_rate: int = 16):
        super().__init__()

        class _DenseLayer(nn.Module):
            def __init__(self, in_ch: int, growth: int):
                super().__init__()
                self.norm = nn.BatchNorm3d(in_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv = nn.Conv3d(in_ch, growth, kernel_size=3, padding=1, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                new_feat = self.conv(self.relu(self.norm(x)))
                return torch.cat([x, new_feat], dim=1)

        class _DenseBlock(nn.Module):
            def __init__(self, in_ch: int, layers: int, growth: int):
                super().__init__()
                dense_layers = []
                ch = in_ch
                for _ in range(layers):
                    dense_layers.append(_DenseLayer(ch, growth))
                    ch += growth
                self.layers = nn.ModuleList(dense_layers)
                self.out_channels = ch

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x

        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = _DenseBlock(32, 3, growth_rate)
        ch1 = self.block1.out_channels
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(ch1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch1, 64, kernel_size=1, bias=False),
            nn.AvgPool3d(2),
        )
        self.block2 = _DenseBlock(64, 3, growth_rate)
        ch2 = self.block2.out_channels
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(ch2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch2, 128, kernel_size=1, bias=False),
            nn.AvgPool3d(2),
        )
        self.block3 = _DenseBlock(128, 3, growth_rate)
        ch3 = self.block3.out_channels
        self.head = nn.Sequential(
            nn.BatchNorm3d(ch3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(ch3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.head(x).flatten(1)
        return self.classifier(x)



def _normalize_model_name(model_name: str) -> str:
    """Normalize public backbone aliases to the internal model factory names."""
    name = model_name.lower().strip()
    aliases = {
        "densenet121": "densenet3d_121",
        "densenet3d121": "densenet3d_121",
        "densenet-121": "densenet3d_121",
        "densenet3d-121": "densenet3d_121",
    }
    return aliases.get(name, name)


class DenseNet3D_121(nn.Module):
    """Monai DenseNet121 3D用于CT体数据分类"""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        if monai is None:
            raise RuntimeError(
                "MONAI is required for densenet3d_121. Install it with `pip install monai` "
                "or switch to another CT backbone."
            )
        self.backbone = monai.networks.nets.DenseNet121(
            spatial_dims=3, n_input_channels=1, out_channels=num_classes
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class HierarchicalMultiTaskClassifier(nn.Module):
    """层次化多任务分类器：共享 backbone + 三个分类 head。

    受 Subregion-Unet 思想启发，将类别分组作为"mask"并行训练：
      - Head A: 0 vs [1,2]  (健康 vs 非健康)
      - Head B: 0, 1, 2     (标准三分类，主输出)
      - Head C: 1 vs 2      (良性 vs 恶性，仅对非健康样本有效)

    训练时返回三个 head 的 logits；推理时仅返回主 head (B) 的输出。
    """

    def __init__(
        self,
        backbone_name: str = "resnet3d18",
        num_classes: int = 3,
        pretrained: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        backbone_name = _normalize_model_name(backbone_name)
        # 共享 backbone（去掉原始分类头）
        base_model = build_model(backbone_name, num_classes=3, pretrained=pretrained)
        if hasattr(base_model, "backbone"):
            self.backbone = base_model.backbone
        else:
            self.backbone = base_model

        # 探测 backbone 输出维度
        feat_dim = self._infer_feature_dim(backbone_name)

        # Remove classifier heads so backbone returns features.
        self._strip_classifier_heads(self.backbone, expected_feature_dim=feat_dim)


        self.backbone_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1) if self._is_3d(backbone_name) else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Head A: 健康 vs 非健康 (binary)
        self.head_abnormal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Head B: 标准三分类 (主输出)
        self.head_main = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Head C: 良性 vs 恶性 (binary, 仅对非健康样本)
        self.head_benign_malignant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        self._hidden_dim = hidden_dim

    @staticmethod
    def _strip_classifier_heads(backbone: nn.Module, expected_feature_dim: int) -> None:
        """Remove final classifier layers while preserving backbone feature pooling.

        MONAI DenseNet stores ``relu -> pool -> flatten -> out`` in ``class_layers``.
        Replacing the whole ``class_layers`` module makes the network return a 5D
        feature map, but some MONAI versions still return the final 3-class logits
        if the head is not stripped correctly.  Replacing only the final Linear
        layer preserves the 1024-D pooled feature vector expected by the
        hierarchical projection layer.
        """
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Module):
            backbone.fc = nn.Identity()
        if hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Module):
            backbone.classifier = nn.Identity()

        class_layers = getattr(backbone, "class_layers", None)
        if isinstance(class_layers, nn.Sequential):
            children = list(class_layers.named_children())
            for child_name, child_module in reversed(children):
                if isinstance(child_module, nn.Linear):
                    if child_module.in_features != expected_feature_dim:
                        raise ValueError(
                            "Unexpected MONAI DenseNet classifier input dimension: "
                            f"got {child_module.in_features}, expected {expected_feature_dim}"
                        )
                    class_layers._modules[child_name] = nn.Identity()
                    return
        elif isinstance(class_layers, nn.Linear):
            if class_layers.in_features != expected_feature_dim:
                raise ValueError(
                    "Unexpected classifier input dimension: "
                    f"got {class_layers.in_features}, expected {expected_feature_dim}"
                )
            backbone.class_layers = nn.Identity()

    @staticmethod
    def _is_3d(backbone_name: str) -> bool:
        backbone_name = _normalize_model_name(backbone_name)
        return backbone_name in {
            "resnet3d18", "mc3_18", "r2plus1d_18", "swin3d_tiny",
            "densenet3d", "densenet3d_121", "attention3d_cnn",
        }

    @staticmethod
    def _infer_feature_dim(backbone_name: str) -> int:
        backbone_name = _normalize_model_name(backbone_name)
        dim_map = {
            "simple": 128, "resnet18": 512, "resnet18_se": 512,
            "resnet18_cbam": 512, "efficientnet_b0": 1280,
            "convnext_tiny": 768, "resnet3d18": 512, "mc3_18": 512,
            "r2plus1d_18": 512, "swin3d_tiny": 768, "densenet3d": 128,
            "densenet3d_121": 1024, "attention3d_cnn": 128,
        }
        if backbone_name not in dim_map:
            raise ValueError(f"Unknown backbone for feature dim: {backbone_name}")
        return dim_map[backbone_name]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.dim() == 2:
            # Some backbones already return pooled [N, C] vectors.
            expected_dim = self.backbone_proj[2].in_features
            if feat.size(1) != expected_dim:
                raise RuntimeError(
                    "Backbone returned a 2D tensor with an unexpected feature dimension: "
                    f"got {feat.size(1)}, expected {expected_dim}. "
                    "This usually means the backbone classifier head was not removed correctly."
                )
            return self.backbone_proj[2:](feat)
        return self.backbone_proj(feat)

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        logits_abnormal = self.head_abnormal(feat)
        logits_main = self.head_main(feat)
        logits_bm = self.head_benign_malignant(feat)

        if self.training:
            return logits_main, logits_abnormal, logits_bm
        return logits_main


def build_model(model_name: str, num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """模型工厂，便于脚本化组合和消融。"""
    name = _normalize_model_name(model_name)

    if name == "simple":
        return SimpleCTClassifier(num_classes=num_classes)
    if name == "resnet18":
        return ResNet18CTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet18_se":
        return ResNet18CTClassifier(num_classes=num_classes, pretrained=pretrained, attention="se")
    if name == "resnet18_cbam":
        return ResNet18CTClassifier(num_classes=num_classes, pretrained=pretrained, attention="cbam")
    if name == "efficientnet_b0":
        return EfficientNetB0CTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return ConvNeXtTinyCTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet3d18":
        return ResNet3D18CTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "mc3_18":
        return MC3_18CTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "r2plus1d_18":
        return R2Plus1D18CTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "swin3d_tiny":
        return Swin3DTinyCTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "densenet3d":
        return DenseNet3DCTClassifier(num_classes=num_classes)
    if name == "densenet3d_121":
        return DenseNet3D_121(num_classes=num_classes)
    if name == "attention3d_cnn":
        return Attention3DCNNClassifier(num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")
