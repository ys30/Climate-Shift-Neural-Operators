"""Compact non-operator U-Net for matched single-channel E1 comparisons."""
from __future__ import annotations
import torch
from torch import nn

def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(1, out_ch), nn.GELU(),
                         nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(1, out_ch), nn.GELU())

class UNet2d(nn.Module):
    """One-downsample U-Net; accepts arbitrary grids >=4x4."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 16):
        super().__init__()
        if min(in_channels, out_channels, width) <= 0:
            raise ValueError("Channels/width must be positive")
        self.in_channels = in_channels
        self.encoder = _block(in_channels + 2, width)
        self.bottom = _block(width, 2 * width)
        self.decoder = _block(3 * width, width)
        self.head = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 4:
            raise ValueError("Expected [batch,channels,lat,lon] with grid >=4x4")
        b, _, h, w = x.shape
        lat = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        lon = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        a = self.encoder(torch.cat([x, lat, lon], dim=1))
        z = self.bottom(nn.functional.avg_pool2d(a, kernel_size=2))
        z = nn.functional.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(self.decoder(torch.cat([z, a], dim=1)))
