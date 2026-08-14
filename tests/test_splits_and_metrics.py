"""Walk-forward splitting and the metric suite."""

from __future__ import annotations

import numpy as np
import pytest

from stock_retrofit.eval.metrics import (
    directional_accuracy,
    evaluate,
    information_coefficient,
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


def test_flat_days_are_not_scored_as_directional_misses():
    """A day that did not move cannot be called, and must not count as a miss.

    This is the defect that made the catalogue read as "worse than chance": on
    KBANK 20% of out-of-sample days close exactly unchanged, so scoring them
    against a 50% coin flip capped the attainable accuracy at 80%.
    """
    y = np.array([0.01, 0.0, 0.0, -0.02])  # two of four days did not move
    pred = np.array([0.01, 0.01, -0.01, -0.01])  # both real days called right
    acc, coverage = directional_accuracy(y, pred)
    assert acc == pytest.approx(1.0), "flat days must be excluded, not counted wrong"
    assert coverage == pytest.approx(1.0), "coverage is over days that moved"


def test_flat_share_reports_what_was_set_aside():
    y = np.array([0.01, 0.0, 0.0, -0.02])
    m = evaluate(y, np.full(4, 0.001))
    assert m.flat_share == pytest.approx(0.5)


def test_a_coin_flip_scores_about_half_on_days_that_moved():
    """The reference line is 50% again once flat days leave the denominator."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 4000)
    y[rng.random(4000) < 0.25] = 0.0  # a quarter of days do not move
    acc, _ = directional_accuracy(y, rng.choice([-1.0, 1.0], 4000))
    assert 0.46 < acc < 0.54


def test_information_coefficient_detects_an_edge_that_mase_misses():
    """The reason IC replaced the `beats_naive` column.

    A forecaster with a real, tradeable edge can score *above* 1.00 on MASE and
    be reported as "does not beat the naive lag", while its IC is unambiguous.
    The returns here are zero-inflated like the real thing — a quarter of days
    close unchanged, as BAY's do — because that is what costs MASE its power:
    a flat day contributes nothing to MAE(naive) and pure error to MAE(model).
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.013, 450)
    y[rng.random(450) < 0.25] = 0.0
    z = (y - y.mean()) / y.std()
    edge = 0.10  # a strong daily-equity signal
    pred = edge * y.std() * (edge * z + np.sqrt(1 - edge**2) * rng.standard_normal(450)) + y.mean()

    ic, t = information_coefficient(y, pred)
    assert ic > 0.05, "a genuine edge must show up in the IC"
    assert t > 1.96, "and be significant"
    assert mase(y, pred) > 1.0, "while MASE still calls it a failure — which is the point"


def test_information_coefficient_is_nan_for_a_constant_forecast():
    y = np.array([0.01, -0.02, 0.005, -0.001])
    ic, t = information_coefficient(y, np.zeros_like(y))
    assert np.isnan(ic) and np.isnan(t)


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
