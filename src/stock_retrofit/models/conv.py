"""Convolutional family — upstream `deep-learning/` notebooks 17 and 18.

Two notebooks, one class with a `dilated` switch. Both stack causal 1-D
convolutions over the feature window; the dilated variant widens the receptive
field geometrically (1, 2, 4, 8, ...) so a short stack can still see the whole
window, which is the point of notebook 18.

Convolutions are left-padded and the right-hand overhang trimmed, so position
`t` never sees `t+1`. That is the same causality guarantee the recurrent family
gets for free.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import register
from .torch_base import TorchForecaster


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _ConvNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        channels: int,
        n_layers: int,
        kernel: int,
        dropout: float,
        dilated: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        blocks = []
        in_ch = n_features
        for i in range(n_layers):
            dilation = 2**i if dilated else 1
            blocks.append(
                nn.ModuleDict(
                    {
                        "conv": _CausalConv1d(in_ch, channels, kernel, dilation),
                        "norm": nn.BatchNorm1d(channels),
                        "skip": (
                            nn.Conv1d(in_ch, channels, 1) if in_ch != channels else nn.Identity()
                        ),
                    }
                )
            )
            in_ch = channels
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(channels, max(channels // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(channels // 2, 8), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, F, T)
        for block in self.blocks:
            out = torch.relu(block["norm"](block["conv"](h)))
            h = out + block["skip"](h) if self.residual else out
            h = self.drop(h)
        return self.head(h[:, :, -1])  # last timestep only


@register("conv")
class ConvForecaster(TorchForecaster):
    """Covers upstream notebooks 17 (`dilated=False`) and 18 (`dilated=True`)."""

    def __init__(
        self,
        *,
        channels: int = 64,
        n_layers: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
        dilated: bool = False,
        residual: bool = True,
        **params,
    ) -> None:
        super().__init__(
            channels=channels,
            n_layers=n_layers,
            kernel=kernel,
            dropout=dropout,
            dilated=dilated,
            residual=residual,
            **params,
        )
        self.channels = int(channels)
        self.n_layers = int(n_layers)
        self.kernel = int(kernel)
        self.dropout = float(dropout)
        self.dilated = bool(dilated)
        self.residual = bool(residual)

    def build_module(self, n_features: int, timestep: int) -> nn.Module:
        return _ConvNet(
            n_features,
            channels=self.channels,
            n_layers=self.n_layers,
            kernel=self.kernel,
            dropout=self.dropout,
            dilated=self.dilated,
            residual=self.residual,
        )
