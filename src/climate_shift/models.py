"""Minimal reproducible 2-D Fourier neural operator baseline (not production SFNO)."""
from __future__ import annotations
import torch
from torch import nn

class SpectralConv2d(nn.Module):
    """Truncated Fourier convolution with positive and negative first-axis modes."""
    def __init__(self, in_channels: int, out_channels: int, modes_y: int = 8, modes_x: int = 8):
        super().__init__()
        if min(in_channels, out_channels, modes_y, modes_x) <= 0:
            raise ValueError("Channels and Fourier modes must be positive")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes_y, self.modes_x = modes_y, modes_x
        scale = 1 / (in_channels * out_channels)
        self.weight_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected tensor of shape (batch, input_channels, lat, lon)")
        b, _, h, w = x.shape
        if h < 2 * self.modes_y or (w // 2 + 1) < self.modes_x:
            raise ValueError("Input grid too small for configured Fourier modes")
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros((b, self.out_channels, h, w // 2 + 1), device=x.device, dtype=x_ft.dtype)
        out_ft[:, :, :self.modes_y, :self.modes_x] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes_y, :self.modes_x], self.weight_pos
        )
        out_ft[:, :, -self.modes_y:, :self.modes_x] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes_y:, :self.modes_x], self.weight_neg
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

class FNO2d(nn.Module):
    """Simple FNO field-to-field regression; adds position channels, not climate forcing."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 16,
                 modes_y: int = 8, modes_x: int = 8, depth: int = 3):
        super().__init__()
        if min(in_channels, out_channels, width, depth) <= 0:
            raise ValueError("Dimensions must be positive")
        self.in_channels = in_channels
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes_y, modes_x) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.output = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected (batch, input_channels, lat, lon)")
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        z = self.lift(torch.cat([x, yy, xx], dim=1))
        for k, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            z = spectral(z) + local(z)
            if k < len(self.spectral) - 1:
                z = torch.nn.functional.gelu(z)
        return self.output(z)
