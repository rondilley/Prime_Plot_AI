"""3D U-Net model for prime pattern detection in 3D spirals."""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Two 3D convolutions with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(
            x1,
            [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
                diff_z // 2, diff_z - diff_z // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet3D(nn.Module):
    """Simple 3D U-Net for volumetric segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super().__init__()

        # Encoder
        self.inc = DoubleConv3D(in_channels, base_features)
        self.down1 = Down3D(base_features, base_features * 2)
        self.down2 = Down3D(base_features * 2, base_features * 4)
        self.down3 = Down3D(base_features * 4, base_features * 8)

        # Decoder
        self.up1 = Up3D(base_features * 8, base_features * 4)
        self.up2 = Up3D(base_features * 4, base_features * 2)
        self.up3 = Up3D(base_features * 2, base_features)

        # Output
        self.outc = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)


if __name__ == "__main__":
    # Test model
    model = SimpleUNet3D(in_channels=13, out_channels=1, base_features=16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 13, 32, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
