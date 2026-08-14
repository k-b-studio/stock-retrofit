"""The walk-forward evaluation loop.

Every model goes through this and only this. The leakage guard is armed for the
duration of each fold, so a preprocessing step that reaches into the test block
raises rather than quietly improving the score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .leakage import leakage_guard
from .metrics import MetricSet, evaluate
from .preprocessing import WindowSpec, prepare_fold
from .splits import Fold, WalkForward, assert_no_overlap

if TYPE_CHECKING:
    # Type-only, to keep the dependency one-directional: models import the
    # harness, never the other way round. A runtime import here closes the
    # cycle eval -> models -> eval.preprocessing.
    from ..models.base import ForecastModel


def set_seed(seed: int) -> None:
    """One seed for everything (R17)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # some RNN kernels have no deterministic path
    except ImportError:
        pass


@dataclass
class FoldResult:
    fold: Fold
    metrics: MetricSet
    y_true: np.ndarray
    y_pred: np.ndarray
    dates: pd.Series


@dataclass
class EvalResult:
    model_name: str
    symbol: str
    upstream: str = ""
    folds: list[FoldResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.folds)

    def pooled(self, *, cost_per_turn: float = 0.0, allow_short: bool = False) -> MetricSet:
        """Metrics over all test blocks concatenated.

        Pooling rather than averaging per-fold numbers, because Sharpe and
        directional accuracy over a short block are noisy, and the concatenated
        series is the honest out-of-sample record.
        """
        y_true = np.concatenate([f.y_true for f in self.folds])
        y_pred = np.concatenate([f.y_pred for f in self.folds])
        return evaluate(y_true, y_pred, cost_per_turn=cost_per_turn, allow_short=allow_short)

    def predictions_frame(self) -> pd.DataFrame:
        frames = [
            pd.DataFrame(
                {
                    "fold": f.fold.index,
                    "date": f.dates.values,
                    "y_true": f.y_true,
                    "y_pred": f.y_pred,
                }
            )
            for f in self.folds
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_walk_forward(
    model: ForecastModel,
    df: pd.DataFrame,
    *,
    splitter: WalkForward,
    window: WindowSpec | None = None,
    features=None,
    symbol: str = "?",
    seed: int = 42,
    cost_per_turn: float = 0.0,
    allow_short: bool = False,
    guard: bool = True,
    verbose: bool = False,
) -> EvalResult:
    """Fit and score `model` fold by fold. Returns a result even on failure."""
    from .preprocessing import DEFAULT_FEATURES

    features = features or DEFAULT_FEATURES
    folds = splitter.split_frame(df)
    assert_no_overlap(folds)

    result = EvalResult(model_name=model.name, symbol=symbol, upstream=model.upstream)

    for fold in folds:
        set_seed(seed + fold.index)
        try:
            model.reset()
            # Arm the guard around preprocessing, fit *and* predict. The upstream
            # bug lived in preprocessing, so leaving that outside would miss the
            # case this harness exists to catch — and predict is inside too
            # because a model is handed `arrays.y_test` and nothing but the guard
            # stands between a rolling update and a refit on the answers. (ARIMA
            # does consume `y_test` there, legitimately: it appends each realised
            # value only after forecasting it, and estimates no parameters.)
            with leakage_guard(
                np.arange(fold.test_start, fold.test_end), fold_index=fold.index, active=guard
            ):
                arrays = prepare_fold(df, fold, window=window, features=features)
                model.fit(arrays)
                y_pred = np.asarray(model.predict(arrays), dtype=float).ravel()
            if y_pred.shape != arrays.y_test.shape:
                raise ValueError(
                    f"{model.name} predicted {y_pred.shape}, expected {arrays.y_test.shape}"
                )
            metrics = evaluate(
                arrays.y_test, y_pred, cost_per_turn=cost_per_turn, allow_short=allow_short
            )
            result.folds.append(
                FoldResult(
                    fold=fold,
                    metrics=metrics,
                    y_true=arrays.y_test,
                    y_pred=y_pred,
                    dates=arrays.test_dates,
                )
            )
            if verbose:
                print(
                    f"  {fold.describe(df['date'])} -> IC {metrics.ic:+.3f} "
                    f"MASE {metrics.mase:.3f} DA {metrics.directional_accuracy:.1%}"
                )
        except Exception as exc:  # a broken model must not abort the whole table
            result.error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  fold {fold.index} FAILED — {result.error}")
            break

    return result
