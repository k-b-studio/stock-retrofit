"""Shared PyTorch training loop.

One loop, many architectures. Each family supplies a `nn.Module` that maps
`(batch, timestep, features) -> (batch,)` and inherits everything else: Adam,
MSE on the scaled target, early stopping on a tail slice of the training block,
and deterministic seeding.

The validation slice is the **last** rows of the training block, never anything
from the test block. That ordering is the whole point of this project.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from torch import nn

from ..eval.preprocessing import FoldArrays
from .base import ForecastModel


def device() -> torch.device:
    return torch.device("cpu")  # these models are small; CPU is the spec's assumption


class TorchForecaster(ForecastModel):
    """A neural forecaster. Subclasses build the module; this runs it."""

    def __init__(
        self,
        *,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.15,
        patience: int = 8,
        grad_clip: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            val_fraction=val_fraction,
            patience=patience,
            **params,
        )
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.module: nn.Module | None = None
        self.history: list[dict] = []

    @abstractmethod
    def build_module(self, n_features: int, timestep: int) -> nn.Module:
        """Return a module mapping (B, T, F) -> (B,)."""

    def extra_loss(self, module: nn.Module) -> torch.Tensor | float:
        """Additional loss term (the VAE family uses this for its KL divergence)."""
        return 0.0

    def reset(self) -> None:
        self.module = None
        self.history = []

    def fit(self, fold: FoldArrays) -> None:
        x = torch.from_numpy(np.ascontiguousarray(fold.x_train)).float()
        y = torch.from_numpy(np.ascontiguousarray(fold.y_train)).float()

        n_val = max(1, int(len(x) * self.val_fraction)) if len(x) > 20 else 0
        if n_val:
            # Validation is the TAIL of training — the most recent rows the model
            # is allowed to see. Never a random split (that would shuffle future
            # rows into training) and never anything from the test block.
            x_tr, y_tr, x_va, y_va = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]
        else:
            x_tr, y_tr, x_va, y_va = x, y, x[:0], y[:0]

        self.module = self.build_module(fold.n_features, x.shape[1]).to(device())
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
        best_val = float("inf")
        stale = 0
        n = len(x_tr)

        for epoch in range(self.epochs):
            self.module.train()
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                pred = self.module(x_tr[idx]).squeeze(-1)
                loss = loss_fn(pred, y_tr[idx]) + self.extra_loss(self.module)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                opt.step()
                total += float(loss.detach()) * len(idx)

            train_loss = total / max(n, 1)
            if len(x_va):
                self.module.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self.module(x_va).squeeze(-1), y_va))
            else:
                val_loss = train_loss

            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.module.load_state_dict(best_state)

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self.module is None:
            raise RuntimeError(f"{self.name}: predict called before fit")
        self.module.eval()
        x = torch.from_numpy(np.ascontiguousarray(fold.x_test)).float()
        with torch.no_grad():
            scaled = self.module(x).squeeze(-1).cpu().numpy()
        return fold.unscale(scaled)


def make_cell(
    kind: str, input_size: int, hidden: int, layers: int, bidirectional: bool, dropout: float
) -> nn.Module:
    """Build an RNN stack. `kind` is lstm | gru | rnn (upstream's 'vanilla')."""
    kinds = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
    if kind not in kinds:
        raise ValueError(f"unknown cell {kind!r}; expected one of {sorted(kinds)}")
    kwargs = dict(
        input_size=input_size,
        hidden_size=hidden,
        num_layers=layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=dropout if layers > 1 else 0.0,
    )
    if kind == "rnn":
        kwargs["nonlinearity"] = "tanh"
    return kinds[kind](**kwargs)


def last_output(out: torch.Tensor) -> torch.Tensor:
    """Final timestep of a (B, T, H) sequence."""
    return out[:, -1, :]
