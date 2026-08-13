"""Encoder-decoder family — upstream `deep-learning/` notebooks 10-15.

Six notebooks, one class with two switches:

    {cell: lstm | gru} x {bidirectional} x {vae: false | true}

The encoder reads the window into a context vector; the decoder unrolls from it
and the head reads the final decoder state. With `vae=True` the context is a
sampled latent with a KL penalty (notebooks 12 and 15), which makes the encoder
stochastic during training and deterministic at inference.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import register
from .torch_base import TorchForecaster, last_output, make_cell


class _Seq2SeqNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        cell: str,
        hidden: int,
        layers: int,
        bidirectional: bool,
        dropout: float,
        vae: bool,
        latent: int,
        decode_steps: int,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.decode_steps = decode_steps
        self.encoder = make_cell(cell, n_features, hidden, layers, bidirectional, dropout)
        enc_width = hidden * (2 if bidirectional else 1)

        if vae:
            self.to_mu = nn.Linear(enc_width, latent)
            self.to_logvar = nn.Linear(enc_width, latent)
            self.from_latent = nn.Linear(latent, hidden)
        else:
            self.from_latent = nn.Linear(enc_width, hidden)

        self.decoder = make_cell(cell, hidden, hidden, 1, False, 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 8), 1),
        )
        self._kl = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = last_output(self.encoder(x)[0])

        if self.vae:
            mu = self.to_mu(context)
            logvar = self.to_logvar(context).clamp(-8, 8)
            if self.training:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                z = mu  # deterministic at inference
            self._kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            context = z

        seed = torch.tanh(self.from_latent(context))
        # Unroll the decoder from the context, feeding its own state forward.
        seq = seed.unsqueeze(1).repeat(1, self.decode_steps, 1)
        out, _ = self.decoder(seq)
        return self.head(last_output(out))

    @property
    def kl(self) -> torch.Tensor:
        return self._kl


@register("seq2seq")
class Seq2SeqForecaster(TorchForecaster):
    """Covers upstream notebooks 10-15."""

    def __init__(
        self,
        *,
        cell: str = "lstm",
        hidden: int = 64,
        layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        vae: bool = False,
        latent: int = 16,
        decode_steps: int = 4,
        kl_weight: float = 1e-3,
        **params,
    ) -> None:
        super().__init__(
            cell=cell,
            hidden=hidden,
            layers=layers,
            bidirectional=bidirectional,
            dropout=dropout,
            vae=vae,
            latent=latent,
            decode_steps=decode_steps,
            kl_weight=kl_weight,
            **params,
        )
        self.cell = str(cell)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)
        self.vae = bool(vae)
        self.latent = int(latent)
        self.decode_steps = int(decode_steps)
        self.kl_weight = float(kl_weight)

    def build_module(self, n_features: int, timestep: int) -> nn.Module:
        return _Seq2SeqNet(
            n_features,
            cell=self.cell,
            hidden=self.hidden,
            layers=self.layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            vae=self.vae,
            latent=self.latent,
            decode_steps=self.decode_steps,
        )

    def extra_loss(self, module: nn.Module):
        if self.vae and isinstance(module, _Seq2SeqNet):
            return self.kl_weight * module.kl
        return 0.0
