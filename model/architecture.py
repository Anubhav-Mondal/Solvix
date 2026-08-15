"""Model architecture for the KLA image restoration hackathon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Five-convolution densely connected residual block."""

    def __init__(self, in_ch: int, growth_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, growth_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch + growth_ch, growth_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_ch + 2 * growth_ch, growth_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_ch + 3 * growth_ch, growth_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_ch + 4 * growth_ch, in_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, in_ch: int, growth_ch: int = 32) -> None:
        super().__init__()
        self.db1 = DenseBlock(in_ch, growth_ch)
        self.db2 = DenseBlock(in_ch, growth_ch)
        self.db3 = DenseBlock(in_ch, growth_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + 0.2 * out


class RestorationNet(nn.Module):
    """Joint denoising + 2x super-resolution network.

    Default configuration:
        input  : (B, 1, H, W), roughly [0, 1]
        output : (B, 1, 2H, 2W), clamped to [0, 1]
        feat_ch=48, num_blocks=8, growth_ch=32, scale=2
    """

    def __init__(
        self,
        in_ch: int = 1,
        feat_ch: int = 48,
        num_blocks: int = 8,
        growth_ch: int = 32,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("RestorationNet supports scale=2 or scale=4")

        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, feat_ch, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(feat_ch, growth_ch) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)

        # One PixelShuffle stage for 2x, two stages for 4x.
        upsample_layers = []
        n_upsample_stages = 1 if scale == 2 else 2
        for _ in range(n_upsample_stages):
            upsample_layers += [
                nn.Conv2d(feat_ch, feat_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_hr = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)
        self.conv_last = nn.Conv2d(feat_ch, in_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.upsample(feat)
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)

        # Bicubic skip connection gives the network an easy low-frequency path.
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        out = out + base
        return torch.clamp(out, 0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def icnr_init(conv: nn.Conv2d, scale: int = 2) -> None:
    """ICNR initialization for PixelShuffle convolution layers."""
    if scale < 2 or conv.out_channels % (scale ** 2) != 0:
        raise ValueError("Invalid ICNR scale or convolution output channels")

    out_ch, in_ch, kh, kw = conv.weight.shape
    sub_ch = out_ch // (scale ** 2)
    k = torch.empty(sub_ch, in_ch, kh, kw, device=conv.weight.device, dtype=conv.weight.dtype)
    nn.init.kaiming_normal_(k)
    k = k.repeat_interleave(scale ** 2, dim=0)

    with torch.no_grad():
        conv.weight.copy_(k)
        if conv.bias is not None:
            conv.bias.zero_()
