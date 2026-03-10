from __future__ import annotations

import torch
from torch import nn
from torchvision import models


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


class ResNet18CTClassifier(nn.Module):
    """ResNet18 adapted for single-channel CT slice classification."""

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        # Load ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Modify first conv layer to accept 1 channel instead of 3
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize the new conv1 layer by averaging the original weights (if pretrained)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))

        # Replace final fc layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
