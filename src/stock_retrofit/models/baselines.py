"""Baselines. `NaiveLag` is a first-class model and appears on every table (R8).

This is the single most important file in the models package. The upstream repo
had no baseline, which is why it could not tell that its models were worse than
"tomorrow's price equals today's". Here the baseline is registered like anything
else, runs through the identical harness, and the report renderer pins it to the
top of every table automatically.
"""

from __future__ import annotations

import numpy as np

from ..eval.preprocessing import FoldArrays
from .base import ForecastModel, register


@register("naive_lag")
class NaiveLag(ForecastModel):
    """Tomorrow's price equals today's — i.e. a forecast of zero return.

    Makes no directional call, ever. Its directional accuracy is therefore
    undefined (reported NaN with 0% coverage) rather than 0%, and its MASE is
    1.0 by construction, which is what every other model is measured against.
    """

    def fit(self, fold: FoldArrays) -> None:  # noqa: D102 - nothing to learn
        return None

    def predict(self, fold: FoldArrays) -> np.ndarray:
        return np.zeros(len(fold.x_test), dtype=float)


@register("drift")
class Drift(ForecastModel):
    """Constant forecast equal to the training block's mean return.

    The second-cheapest thing that could work, and a useful check on whether a
    model has learned anything beyond the period's average drift.
    """

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._mu = 0.0

    def reset(self) -> None:
        self._mu = 0.0

    def fit(self, fold: FoldArrays) -> None:
        self._mu = float(np.mean(fold.unscale(fold.y_train)))

    def predict(self, fold: FoldArrays) -> np.ndarray:
        return np.full(len(fold.x_test), self._mu, dtype=float)


@register("momentum")
class Momentum(ForecastModel):
    """Yesterday's return repeated — the other classical straw man."""

    def __init__(self, *, feature: str = "log_return", **params) -> None:
        super().__init__(feature=feature, **params)
        self.feature = feature
        self._idx = 0
        self._scale = 1.0
        self._center = 0.0

    def fit(self, fold: FoldArrays) -> None:
        self._idx = fold.feature_names.index(self.feature)
        # Map the scaled feature back to a return-sized number.
        train_last = fold.x_train[:, -1, self._idx]
        y = fold.unscale(fold.y_train)
        sd_x = float(np.std(train_last)) or 1.0
        self._scale = float(np.std(y)) / sd_x
        self._center = float(np.mean(y))

    def predict(self, fold: FoldArrays) -> np.ndarray:
        return fold.x_test[:, -1, self._idx] * self._scale + self._center
