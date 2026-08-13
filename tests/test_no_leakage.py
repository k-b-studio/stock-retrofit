"""Required by spec R7 / acceptance criterion 1.

The bar this file has to clear: deliberately fitting a scaler on the full series
must make it **fail**. `test_upstream_bug_is_detected` does exactly that — it
reproduces the upstream ordering (`MinMaxScaler().fit(everything)` before the
split) and asserts the guard raises. If the guard is ever weakened, that test
goes green and this comment is your clue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from stock_retrofit.eval.leakage import LeakageError, leakage_guard, register_fit
from stock_retrofit.eval.preprocessing import WindowSpec, prepare_fold
from stock_retrofit.eval.runner import run_walk_forward
from stock_retrofit.eval.splits import Fold, WalkForward
from stock_retrofit.models import build


def synthetic_prices(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-01", periods=n),
            "open": close,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.integers(1e5, 1e7, n).astype(float),
            "symbol": "TEST",
        }
    )


# --------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------


def test_guard_raises_when_a_fit_touches_a_test_row():
    with pytest.raises(LeakageError, match="test-block"):
        with leakage_guard(test_indices=range(90, 100), fold_index=0):
            register_fit("scaler", np.arange(0, 95))  # reaches into 90..94


def test_guard_allows_a_fit_confined_to_training_rows():
    with leakage_guard(test_indices=range(90, 100), fold_index=0):
        register_fit("scaler", np.arange(0, 90))  # stops exactly at the boundary


def test_guard_is_inert_when_not_armed():
    register_fit("scaler", np.arange(0, 1000))  # no guard active, must not raise


# --------------------------------------------------------------------------
# The real pipeline
# --------------------------------------------------------------------------


def test_prepare_fold_never_fits_on_test_rows():
    df = synthetic_prices()
    fold = Fold(index=0, train_start=0, train_end=900, test_start=900, test_end=960)
    with leakage_guard(test_indices=range(fold.test_start, fold.test_end), fold_index=0) as state:
        prepare_fold(df, fold, window=WindowSpec(20))
    names = [n for n, _ in state.observed]
    assert names, "prepare_fold fit something without registering it with the guard"
    assert "FoldPreprocessor.features" in names
    for name, idx in state.observed:
        assert max(idx) < fold.train_end, f"{name} fit on row {max(idx)} >= train_end"


def test_walk_forward_runs_clean_with_the_guard_armed():
    df = synthetic_prices()
    result = run_walk_forward(
        build("naive_lag", name="naive_lag"),
        df,
        splitter=WalkForward(train_window=600, test_window=60, step=60, max_folds=3),
        window=WindowSpec(20),
        symbol="TEST",
        guard=True,
    )
    assert result.ok, result.error
    assert len(result.folds) == 3


def test_training_samples_never_use_a_label_from_the_test_block():
    """A training row's label is the *next* day's return, so the last usable
    training position must be strictly before `train_end - 1`."""
    df = synthetic_prices()
    fold = Fold(index=0, train_start=0, train_end=900, test_start=900, test_end=960)
    arrays = prepare_fold(df, fold, window=WindowSpec(20))
    assert arrays.train_positions.max() < fold.train_end - 1
    assert arrays.test_positions.min() >= fold.test_start


# --------------------------------------------------------------------------
# The upstream bug, reintroduced on purpose
# --------------------------------------------------------------------------


def test_upstream_bug_is_detected():
    """Reproduce `deep-learning/1.lstm.ipynb`: fit the scaler, then split.

    This is the exact ordering that makes every upstream forecasting result
    unusable. The guard must catch it. If this test ever passes without raising,
    the harness has stopped protecting anything.
    """
    df = synthetic_prices()
    fold = Fold(index=0, train_start=0, train_end=900, test_start=900, test_end=960)

    def leaky_prepare(frame: pd.DataFrame) -> np.ndarray:
        # ---- the upstream ordering, verbatim in spirit ----
        scaler = MinMaxScaler()
        scaler.fit(frame[["close"]].to_numpy())  # fit on ALL rows, before any split
        register_fit("MinMaxScaler(full series)", np.arange(len(frame)))
        return scaler.transform(frame[["close"]].to_numpy())

    with pytest.raises(LeakageError, match="MinMaxScaler"):
        with leakage_guard(test_indices=range(fold.test_start, fold.test_end), fold_index=0):
            leaky_prepare(df)


def test_fold_local_scaling_changes_the_numbers():
    """Sanity check that fold-local scaling is not accidentally global.

    If the scaler were fit on everything, two different folds would produce the
    same standardisation. They must not.
    """
    df = synthetic_prices()
    a = prepare_fold(df, Fold(0, 0, 600, 600, 660), window=WindowSpec(20))
    b = prepare_fold(df, Fold(1, 300, 900, 900, 960), window=WindowSpec(20))
    assert not np.isclose(a.target_scale, b.target_scale), "folds share a scaler"
