from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


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

        def dense_block(in_ch: int, layers: int):
            blocks = []
            ch = in_ch
            for _ in range(layers):
                blocks.extend([
                    nn.BatchNorm3d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(ch, growth_rate, kernel_size=3, padding=1, bias=False),
                ])
                ch += growth_rate
            return nn.Sequential(*blocks), ch

        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.block1, ch1 = dense_block(32, 3)
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(ch1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch1, 64, kernel_size=1, bias=False),
            nn.AvgPool3d(2),
        )
        self.block2, ch2 = dense_block(64, 3)
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(ch2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch2, 128, kernel_size=1, bias=False),
            nn.AvgPool3d(2),
        )
        self.block3, ch3 = dense_block(128, 3)
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


def build_model(model_name: str, num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """模型工厂，便于脚本化组合和消融。"""
    name = model_name.lower().strip()

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
    if name == "attention3d_cnn":
        return Attention3DCNNClassifier(num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")
