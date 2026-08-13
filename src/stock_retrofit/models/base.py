"""Forecaster interface and registry.

Every model — including the naive baseline — implements the same two methods
and is constructed from a YAML config, so "compare all 18 on KBANK" is one
command rather than eighteen notebooks (the review's central point).

Contract: `fit` sees only the fold's training arrays; `predict` returns
**raw next-day simple returns**, one per test row, unscaled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..eval.preprocessing import FoldArrays


class ForecastModel(ABC):
    """A next-day return forecaster."""

    #: Set by the registry from the config's `name`.
    name: str = "unnamed"
    #: Human-readable pointer back to the upstream notebook this covers.
    upstream: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def fit(self, fold: FoldArrays) -> None:
        """Fit on `fold.x_train` / `fold.y_train` only."""

    @abstractmethod
    def predict(self, fold: FoldArrays) -> np.ndarray:
        """Return raw next-day returns for `fold.x_test`, shape (n_test,)."""

    def reset(self) -> None:
        """Drop fitted state. Called before every fold so folds stay independent."""

    def describe(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({bits})"


_REGISTRY: dict[str, Callable[..., ForecastModel]] = {}


def register(kind: str):
    """Register a model family under a config-visible `kind`."""

    def wrap(cls):
        if kind in _REGISTRY:
            raise KeyError(f"model kind {kind!r} is already registered")
        _REGISTRY[kind] = cls
        return cls

    return wrap


def build(kind: str, *, name: str = "", upstream: str = "", **params: Any) -> ForecastModel:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown model kind {kind!r}; registered: {sorted(_REGISTRY)}")
    model = _REGISTRY[kind](**params)
    model.name = name or kind
    model.upstream = upstream
    return model


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)
