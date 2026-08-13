"""Attention family — upstream `deep-learning/16.attention-is-all-you-need.ipynb`.

A transformer encoder over the feature window with sinusoidal positional
encoding, mean-pooled and read out by a linear head. Causality is structural
here rather than enforced by a mask: the window ends at bar `t` and the label is
the return at `t+1`, so every position the model attends to was already
observable at prediction time.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import register
from .torch_base import TorchForecaster


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _AttentionNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pooling = pooling
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.project(x)))
        pooled = h.mean(dim=1) if self.pooling == "mean" else h[:, -1, :]
        return self.head(pooled)


@register("attention")
class AttentionForecaster(TorchForecaster):
    """Covers upstream notebook 16."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        pooling: str = "mean",
        **params,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pooling=pooling,
            **params,
        )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.pooling = str(pooling)

    def build_module(self, n_features: int, timestep: int) -> nn.Module:
        d_model = self.d_model
        if d_model % self.n_heads:
            d_model = self.n_heads * max(1, d_model // self.n_heads)
        return _AttentionNet(
            n_features,
            d_model=d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            pooling=self.pooling,
        )
