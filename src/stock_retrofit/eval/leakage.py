"""Runtime leakage detection.

The upstream repo's defining defect is one line of ordering:
`MinMaxScaler().fit(whole_series)` before the train/test split, which bakes the
test window's min and max into the training representation. Nothing about that
bug is visible in the output — the numbers just come out better than they should.

So it is not enough to *intend* to fit inside folds. This module makes the
intention checkable at runtime: every fit of every statistic registers the row
indices it saw, and the guard raises if any of them belongs to the fold's test
block. Turn the bug back on and the guard fires — which is exactly how
`tests/test_no_leakage.py` proves it works.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


class LeakageError(AssertionError):
    """A statistic was fit on rows belonging to the test block."""


@dataclass
class _GuardState:
    fold_index: int
    test_indices: frozenset[int]
    observed: list[tuple[str, frozenset[int]]] = field(default_factory=list)


_local = threading.local()


def _state() -> _GuardState | None:
    return getattr(_local, "state", None)


@contextmanager
def leakage_guard(test_indices, *, fold_index: int = -1, active: bool = True):
    """Arm the guard for one fold.

    Any `register_fit` inside this block whose row indices intersect
    `test_indices` raises `LeakageError` immediately, naming the offender.
    """
    if not active:
        yield None
        return

    previous = _state()
    state = _GuardState(fold_index=fold_index, test_indices=frozenset(int(i) for i in test_indices))
    _local.state = state
    try:
        yield state
    finally:
        _local.state = previous


def register_fit(name: str, indices) -> None:
    """Declare that `name` was fit on `indices` (positions in the full series).

    Every preprocessing step that learns anything from data must call this. A
    step that does not call it is invisible to the guard — so if you add one,
    add the call too.
    """
    state = _state()
    if state is None:
        return
    idx = frozenset(int(i) for i in np.asarray(indices).ravel())
    state.observed.append((name, idx))
    offending = idx & state.test_indices
    if offending:
        sample = sorted(offending)[:5]
        raise LeakageError(
            f"fold {state.fold_index}: '{name}' was fit on {len(offending)} test-block "
            f"row(s) (positions {sample}{'...' if len(offending) > 5 else ''}). "
            "Fit every statistic on the training block only."
        )


def guard_is_active() -> bool:
    return _state() is not None
