"""Walk-forward splits. There is no single fixed tail split anywhere (spec R6).

Upstream took the last 30 rows as "test" once, after scaling the whole series.
This module replaces that with rolling origin evaluation: a model is refit on
each training window and scored on the block of unseen days that follows it, so
a reported number is an average over several genuinely out-of-sample periods
rather than one lucky month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start

    def describe(self, dates: pd.Series | None = None) -> str:
        if dates is None:
            return (
                f"fold {self.index}: train[{self.train_start}:{self.train_end}] "
                f"test[{self.test_start}:{self.test_end}]"
            )
        return (
            f"fold {self.index}: train {dates.iloc[self.train_start].date()}"
            f"..{dates.iloc[self.train_end - 1].date()} "
            f"({self.n_train}) | test {dates.iloc[self.test_start].date()}"
            f"..{dates.iloc[self.test_end - 1].date()} ({self.n_test})"
        )


@dataclass(frozen=True)
class WalkForward:
    """Rolling-origin splitter.

    `train_window` is the number of rows fit on. `expanding=True` grows the
    window from the start of the series instead of sliding it.
    """

    train_window: int = 750
    test_window: int = 60
    step: int = 60
    expanding: bool = False
    max_folds: int | None = None

    def split(self, n: int) -> list[Fold]:
        if self.train_window <= 0 or self.test_window <= 0 or self.step <= 0:
            raise ValueError("train_window, test_window and step must all be positive")

        folds: list[Fold] = []
        train_end = self.train_window
        while train_end + self.test_window <= n:
            train_start = 0 if self.expanding else max(0, train_end - self.train_window)
            folds.append(
                Fold(
                    index=len(folds),
                    train_start=train_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=train_end + self.test_window,
                )
            )
            train_end += self.step

        if not folds:
            raise ValueError(
                f"no folds: series has {n} rows but train_window={self.train_window} + "
                f"test_window={self.test_window} needs at least "
                f"{self.train_window + self.test_window}"
            )
        if self.max_folds is not None and len(folds) > self.max_folds:
            # Keep the most recent folds — the nearest regime is the relevant one.
            folds = folds[-self.max_folds :]
            folds = [
                Fold(i, f.train_start, f.train_end, f.test_start, f.test_end)
                for i, f in enumerate(folds)
            ]
        return folds

    def split_frame(self, df: pd.DataFrame) -> list[Fold]:
        return self.split(len(df))


def assert_no_overlap(folds: list[Fold]) -> None:
    """Every test block must start at or after its own training block ends."""
    for f in folds:
        if f.test_start < f.train_end:
            raise AssertionError(f"{f} — test block starts inside the training block")
        if f.train_start >= f.train_end:
            raise AssertionError(f"{f} — empty training block")


def collect_test_indices(folds: list[Fold]) -> np.ndarray:
    """All row indices that are ever used for testing, deduplicated."""
    if not folds:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([np.arange(f.test_start, f.test_end) for f in folds]))
