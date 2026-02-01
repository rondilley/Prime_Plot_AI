"""Neural network models for prime pattern recognition.

Implements U-Net with ResNet encoder as described in arXiv:2509.18103,
adapted for prime/composite classification in spiral visualizations.
"""

from __future__ import annotations

from typing import Literal, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    up: Union[nn.Upsample, nn.ConvTranspose2d]
    conv: ConvBlock

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    """Simple U-Net architecture for prime detection.

    A lightweight U-Net suitable for smaller datasets and quick training.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        out_channels: Number of output channels (1 for binary classification).
        features: List of feature dimensions for each encoder level.
        bilinear: Use bilinear upsampling instead of transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        bilinear: bool = True,
    ):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))

        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.decoder_blocks.append(UpBlock(prev_channels + feature, feature, bilinear))
            prev_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx, decoder in enumerate(self.decoder_blocks):
            x = decoder(x, skip_connections[idx])

        return self.final_conv(x)


class PrimeUNet(nn.Module):
    """U-Net with ResNet encoder for prime pattern recognition.

    Based on the architecture from arXiv:2509.18103, using a pre-trained
    ResNet-34 encoder with U-Net decoder for pixel-wise classification.

    Args:
        encoder_name: ResNet variant ("resnet18", "resnet34", "resnet50").
        pretrained: Use ImageNet pre-trained weights.
        in_channels: Number of input channels.
        out_channels: Number of output classes.
    """

    def __init__(
        self,
        encoder_name: Literal["resnet18", "resnet34", "resnet50"] = "resnet34",
        pretrained: bool = True,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()

        from torchvision import models

        if encoder_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        self.input_conv: Union[nn.Conv2d, nn.Identity]
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_conv = nn.Identity()

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = UpBlock(encoder_channels[4] + encoder_channels[3], encoder_channels[3])
        self.decoder3 = UpBlock(encoder_channels[3] + encoder_channels[2], encoder_channels[2])
        self.decoder2 = UpBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.decoder1 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0])

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            ConvBlock(encoder_channels[0], 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        out = self.final_up(d1)

        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return self.final_conv(out)


class PrimeClassifier(nn.Module):
    """Simple CNN classifier for prime density regions.

    Classifies image patches as high/low prime density areas,
    useful for identifying promising polynomial regions.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_model(
    model_type: str = "unet",
    **kwargs,
) -> nn.Module:
    """Factory function to create models.

    Args:
        model_type: One of "unet", "prime_unet", "classifier", "simple_unet".
        **kwargs: Arguments passed to the model constructor.

    Returns:
        Initialized model.
    """
    models = {
        "simple_unet": SimpleUNet,
        "unet": SimpleUNet,
        "prime_unet": PrimeUNet,
        "resnet_unet": PrimeUNet,
        "classifier": PrimeClassifier,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)
