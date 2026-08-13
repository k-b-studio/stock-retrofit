"""The recurrent family — upstream `deep-learning/` notebooks 1-9.

Nine notebooks, one class. The upstream set is the cross product

    {cell: lstm | gru | rnn} x {bidirectional: false | true} x {paths: 1 | 2}

and nothing else differs between them but hyperparameters, which now live in
YAML. `paths=2` reproduces the "2-path" notebooks: two independent recurrent
stacks read the same window and their final states are concatenated before the
head.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import register
from .torch_base import TorchForecaster, last_output, make_cell


class _RecurrentNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        cell: str,
        hidden: int,
        layers: int,
        bidirectional: bool,
        paths: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.paths = nn.ModuleList(
            [
                make_cell(cell, n_features, hidden, layers, bidirectional, dropout)
                for _ in range(paths)
            ]
        )
        width = hidden * (2 if bidirectional else 1) * paths
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width, max(width // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(width // 2, 8), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [last_output(path(x)[0]) for path in self.paths]
        return self.head(torch.cat(states, dim=-1))


@register("recurrent")
class RecurrentForecaster(TorchForecaster):
    """Covers upstream notebooks 1-9 through four parameters."""

    def __init__(
        self,
        *,
        cell: str = "lstm",
        hidden: int = 64,
        layers: int = 1,
        bidirectional: bool = False,
        paths: int = 1,
        dropout: float = 0.1,
        **params,
    ) -> None:
        super().__init__(
            cell=cell,
            hidden=hidden,
            layers=layers,
            bidirectional=bidirectional,
            paths=paths,
            dropout=dropout,
            **params,
        )
        self.cell = str(cell)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.bidirectional = bool(bidirectional)
        self.n_paths = int(paths)
        self.dropout = float(dropout)

    def build_module(self, n_features: int, timestep: int) -> nn.Module:
        return _RecurrentNet(
            n_features,
            cell=self.cell,
            hidden=self.hidden,
            layers=self.layers,
            bidirectional=self.bidirectional,
            paths=self.n_paths,
            dropout=self.dropout,
        )
