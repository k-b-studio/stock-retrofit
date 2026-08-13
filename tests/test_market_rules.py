"""Required by spec R10. Property tests over the friction layer.

The four properties the spec names: no order off-tick, no order off-lot, fees
always non-negative, no fill outside the day's high/low. Plus the rules that
make this SET rather than a generic exchange — board lot, ceiling/floor, no
naked shorting, participation cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_retrofit.market import (
    FeeSchedule,
    MarketConfig,
    SETMarket,
    is_on_lot,
    is_on_tick,
    price_limits,
    round_to_lot,
    snap_to_tick,
    tick_size,
)


def bars(n: int = 300, seed: int = 7, start: float = 120.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    close = np.array([snap_to_tick(c) for c in close])
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(50_000, 5_000_000, n).astype(float),
            "symbol": "TEST",
        }
    )


# ---------------------------------------------------------------- tick table


@pytest.mark.parametrize(
    "price,expected",
    [
        (1.50, 0.01),
        (2.00, 0.02),
        (4.98, 0.02),
        (5.00, 0.05),
        (9.95, 0.05),
        (10.00, 0.10),
        (24.9, 0.10),
        (25.00, 0.25),
        (99.75, 0.25),
        (100.00, 0.50),
        (199.5, 0.50),
        (200.00, 1.00),
        (399.0, 1.00),
        (400.00, 2.00),
        (1000.0, 2.00),
    ],
)
def test_tick_size_bands(price, expected):
    assert tick_size(price) == expected


@pytest.mark.parametrize("price", [0.37, 3.111, 7.77, 23.02, 88.88, 150.4, 275.3, 512.7])
@pytest.mark.parametrize("mode", ["nearest", "up", "down"])
def test_snapped_prices_are_always_on_tick(price, mode):
    assert is_on_tick(snap_to_tick(price, mode=mode))


def test_snap_directions_bracket_the_input():
    for price in (3.111, 23.02, 88.88, 150.4):
        assert snap_to_tick(price, mode="down") <= price <= snap_to_tick(price, mode="up")


def test_snap_is_idempotent():
    for price in (0.37, 7.77, 88.88, 275.3):
        once = snap_to_tick(price)
        assert snap_to_tick(once) == once


# ----------------------------------------------------------------- board lot


@pytest.mark.parametrize("shares,expected", [(0, 0), (99, 0), (100, 100), (150, 100), (1999, 1900)])
def test_round_to_lot_never_rounds_up(shares, expected):
    assert round_to_lot(shares) == expected
    assert round_to_lot(shares) <= shares


def test_is_on_lot():
    assert is_on_lot(0) and is_on_lot(100) and is_on_lot(2500)
    assert not is_on_lot(50) and not is_on_lot(150)


# ---------------------------------------------------------------------- fees


@pytest.mark.parametrize("turnover", [0.0, 1.0, 1e3, 1e6, 1e9])
def test_commission_is_never_negative(turnover):
    assert FeeSchedule().commission(turnover) >= 0.0


def test_commission_includes_vat():
    fees = FeeSchedule(commission_rate=0.001, vat_rate=0.07)
    assert fees.commission(1_000_000) == pytest.approx(1_000 * 1.07)


def test_zero_turnover_costs_nothing():
    assert FeeSchedule().commission(0) == 0.0


# ------------------------------------------------------------ price limits


def test_price_limits_are_thirty_percent_and_on_tick():
    floor, ceiling = price_limits(100.0)
    assert is_on_tick(floor) and is_on_tick(ceiling)
    assert 69.0 <= floor <= 70.5
    assert 129.5 <= ceiling <= 130.5


# ------------------------------------------------------- market invariants


def test_every_fill_is_on_tick_on_lot_and_inside_the_days_range():
    df = bars()
    market = SETMarket(df, MarketConfig(initial_cash=1_000_000))
    rng = np.random.default_rng(1)
    for t in range(1, len(df)):
        if rng.random() < 0.3:
            market.buy(t, fraction=0.5)
        elif rng.random() < 0.3:
            market.sell(t)
    market.assert_invariants()
    assert any(f.ok for f in market.fills), "test exercised no successful fills"


def test_shorting_is_refused_without_sbl():
    df = bars()
    market = SETMarket(df, MarketConfig(allow_short=False))
    fill = market.sell(5, shares=1000)  # nothing held
    assert not fill.ok
    assert fill.rejected in {"no_position", "short_selling_disabled"}
    assert market.shares == 0


def test_cash_never_goes_negative():
    df = bars()
    market = SETMarket(df, MarketConfig(initial_cash=50_000))
    for t in range(1, 50):
        market.buy(t, fraction=1.0)
    assert market.cash >= -1e-6


def test_participation_cap_limits_order_size():
    df = bars()
    df.loc[:, "volume"] = 10_000.0  # thin session
    capped = SETMarket(df, MarketConfig(initial_cash=10_000_000, participation_cap=0.05))
    fill = capped.buy(10, fraction=1.0)
    assert fill.filled <= 0.05 * 10_000 + 1
    uncapped = SETMarket(df, MarketConfig(initial_cash=10_000_000, participation_cap=None))
    assert uncapped.buy(10, fraction=1.0).filled > fill.filled


def test_frictionless_twin_charges_nothing():
    df = bars()
    cfg = MarketConfig(initial_cash=100_000)
    friction = SETMarket(df, cfg)
    free = SETMarket(df, cfg.frictionless_twin())
    for t in range(1, 100):
        friction.buy(t, fraction=0.2)
        free.buy(t, fraction=0.2)
    assert sum(f.commission for f in free.fills) == 0.0
    assert sum(f.commission for f in friction.fills) > 0.0


def test_frictionless_beats_friction_on_an_identical_trade_sequence():
    """Like for like: the same shares on the same days, costs the only difference.

    Comparing equity across the two runs is only meaningful when the trades
    match. Sized by `fraction`, the frictionless run buys *more* (no lot
    rounding down, no commission reserve), so its equity can legitimately end
    lower on a falling path — that is position sizing, not friction.
    """
    df = bars()
    cfg = MarketConfig(initial_cash=1_000_000)
    friction = SETMarket(df, cfg)
    free = SETMarket(df, cfg.frictionless_twin())
    for t in range(1, 100, 5):
        friction.buy(t, shares=100)
        free.buy(t, shares=100)
        friction.sell(t + 1, shares=100)
        free.sell(t + 1, shares=100)
    assert friction.shares == free.shares == 0
    assert free.cash > friction.cash, "frictionless must end with more cash after equal trades"
    # The gap is exactly the commission bill — closes in `bars()` are already
    # on-tick, so snapping is a no-op and nothing else can differ.
    total_commission = sum(f.commission for f in friction.fills)
    assert free.cash - friction.cash == pytest.approx(total_commission, rel=1e-9)


def test_fills_never_price_outside_the_bar():
    df = bars()
    market = SETMarket(df, MarketConfig())
    for t in range(1, 200):
        market.buy(t, fraction=0.1)
        market.sell(t, fraction=0.5)
    for f in market.fills:
        if f.ok:
            assert df["low"].iloc[f.t] - 1e-9 <= f.price <= df["high"].iloc[f.t] + 1e-9


def test_a_capped_order_that_fills_counts_as_a_trade():
    """Regression: `Fill.ok` must not be keyed off `rejected`.

    A participation cap trims an order without refusing it. Treating the trimmed
    fill as a non-event undercounts trades and costs — and, worse, makes
    `assert_invariants` skip it. Only BAY carries a cap by default, so this
    would have been invisible on KBANK and SCB and wrong on the one ticker
    where capping actually happens.
    """
    df = bars()
    df.loc[:, "volume"] = 100_000.0
    market = SETMarket(df, MarketConfig(initial_cash=50_000_000, participation_cap=0.05))

    fill = market.buy(10, fraction=1.0)
    assert fill.filled > 0, "the cap should trim this order, not refuse it"
    assert fill.rejected == "participation_cap"
    assert fill.ok, "a capped order that filled is still a trade"
    assert fill.reduced, "and it should be flagged as reduced"
    assert market.shares == fill.filled

    # It must be visible in the accounting and covered by the invariants.
    result_trades = sum(1 for f in market.fills if f.ok)
    assert result_trades == 1
    assert sum(f.commission for f in market.fills if f.ok) > 0
    market.assert_invariants()


def test_capped_fills_are_reductions_not_rejections():
    df = bars()
    df.loc[:, "volume"] = 100_000.0
    market = SETMarket(df, MarketConfig(initial_cash=50_000_000, participation_cap=0.05))
    market.buy(10, fraction=1.0)  # trimmed by the cap, but fills
    market.buy(12, shares=50)  # refused outright: below one board lot

    from stock_retrofit.market.set_market import BacktestResult

    result = BacktestResult(
        symbol="TEST",
        label="t",
        frictionless=False,
        dates=df["date"],
        equity=np.full(len(df), 1.0),
        fills=market.fills,
        initial_cash=1.0,
    )
    # A trimmed order is a trade with a reason; a refused order is neither.
    assert result.reductions == {"participation_cap": 1}
    assert result.rejections == {"below_board_lot": 1}
    assert result.n_trades == 1


def test_shorting_disabled_clamps_an_oversized_sell_to_the_position():
    """Selling more than you hold is not a refusal — it is a full exit.

    Only a sell with *no* position to clamp to gets refused.
    """
    df = bars()
    market = SETMarket(df, MarketConfig(initial_cash=1_000_000))
    market.buy(5, fraction=0.5)
    held = market.shares
    assert held > 0

    fill = market.sell(6, shares=999_999_999)
    assert fill.ok and fill.filled == held
    assert market.shares == 0

    refused = market.sell(7, shares=100)
    assert not refused.ok and refused.rejected == "no_position"
