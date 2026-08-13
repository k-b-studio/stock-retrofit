"""Walk-forward splitting and the metric suite."""

from __future__ import annotations

import numpy as np
import pytest

from stock_retrofit.eval.metrics import (
    directional_accuracy,
    evaluate,
    mase,
    sharpe,
    strategy_returns,
    upstream_accuracy_do_not_use,
)
from stock_retrofit.eval.splits import WalkForward, assert_no_overlap, collect_test_indices

# ------------------------------------------------------------------ splits


def test_folds_never_overlap_train_and_test():
    folds = WalkForward(train_window=100, test_window=20, step=20).split(400)
    assert_no_overlap(folds)
    for f in folds:
        assert f.test_start >= f.train_end


def test_test_blocks_are_disjoint_when_step_equals_test_window():
    folds = WalkForward(train_window=100, test_window=20, step=20).split(400)
    idx = collect_test_indices(folds)
    assert len(idx) == sum(f.n_test for f in folds), "test blocks overlap"


def test_sliding_window_has_constant_train_size():
    folds = WalkForward(train_window=100, test_window=20, step=20, expanding=False).split(400)
    assert {f.n_train for f in folds} == {100}


def test_expanding_window_grows():
    folds = WalkForward(train_window=100, test_window=20, step=20, expanding=True).split(400)
    sizes = [f.n_train for f in folds]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]


def test_max_folds_keeps_the_most_recent():
    everything = WalkForward(train_window=100, test_window=20, step=20).split(400)
    limited = WalkForward(train_window=100, test_window=20, step=20, max_folds=3).split(400)
    assert len(limited) == 3
    assert limited[-1].test_end == everything[-1].test_end


def test_too_short_a_series_is_an_error_not_an_empty_list():
    with pytest.raises(ValueError, match="no folds"):
        WalkForward(train_window=1000, test_window=60).split(200)


# ----------------------------------------------------------------- metrics


def test_mase_is_exactly_one_for_the_naive_forecast():
    y = np.array([0.01, -0.02, 0.005, -0.001])
    assert mase(y, np.zeros_like(y)) == pytest.approx(1.0)


def test_mase_below_one_means_better_than_naive():
    y = np.array([0.01, -0.02, 0.005, -0.001])
    assert mase(y, y * 0.5) < 1.0
    assert mase(y, -y) > 1.0


def test_a_zero_forecast_makes_no_directional_call():
    """The naive baseline must not be scored as 0% accurate — it abstains."""
    y = np.array([0.01, -0.02, 0.005])
    acc, coverage = directional_accuracy(y, np.zeros_like(y))
    assert np.isnan(acc)
    assert coverage == 0.0


def test_directional_accuracy_counts_only_calls_made():
    y = np.array([0.01, -0.02, 0.005, -0.001])
    pred = np.array([0.01, 0.0, 0.005, 0.0])  # two calls, both right
    acc, coverage = directional_accuracy(y, pred)
    assert acc == pytest.approx(1.0)
    assert coverage == pytest.approx(0.5)


def test_costs_reduce_strategy_returns():
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0])  # flips every day
    gross = strategy_returns(y, pred, cost_per_turn=0.0).sum()
    net = strategy_returns(y, pred, cost_per_turn=0.005).sum()
    assert net < gross


def test_a_constant_position_pays_cost_once():
    y = np.full(10, 0.01)
    pred = np.ones(10)
    charged = strategy_returns(y, pred, cost_per_turn=0.01).sum()
    free = strategy_returns(y, pred, cost_per_turn=0.0).sum()
    assert free - charged == pytest.approx(0.01)


def test_sharpe_of_a_flat_series_is_zero_not_nan():
    assert sharpe(np.zeros(50)) == 0.0


def test_evaluate_returns_the_full_suite():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 200)
    m = evaluate(y, y * 0.3, cost_per_turn=0.003)
    assert m.n == 200
    assert m.mase < 1.0
    assert m.sharpe_after_costs <= m.sharpe_frictionless


def test_upstream_metric_flatters_a_naive_lag():
    """Demonstrates why upstream's headline number is not evidence of skill.

    A pure lag — 'tomorrow's price is today's' — scores in the high nineties on
    the metric the upstream README reports.
    """
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 500)))
    real, lag = prices[1:], prices[:-1]
    assert upstream_accuracy_do_not_use(real, lag) > 0.95
