"""SET market microstructure: rules and the simulator that enforces them.

Nothing in here places a real order. `SETMarket` is a backtest simulator, and
the Settrade credentials the data layer uses are for market data only.
"""

from .rules import (
    BOARD_LOT,
    TICK_TABLE,
    FeeSchedule,
    is_on_lot,
    is_on_tick,
    price_limits,
    round_to_lot,
    snap_to_tick,
    tick_size,
    within_limits,
)
from .set_market import BacktestResult, Fill, MarketConfig, SETMarket

__all__ = [
    "BOARD_LOT",
    "TICK_TABLE",
    "BacktestResult",
    "FeeSchedule",
    "Fill",
    "MarketConfig",
    "SETMarket",
    "is_on_lot",
    "is_on_tick",
    "price_limits",
    "round_to_lot",
    "snap_to_tick",
    "tick_size",
    "within_limits",
]
