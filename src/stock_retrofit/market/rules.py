"""SET trading rules: tick table, board lot, fees, price limits.

    !! RECONSTRUCTED, NOT VERIFIED !!

Every number in this module was reconstructed from general market knowledge, as
the spec's R13 says it was. None of it has been checked against SET's current
published rulebook or a broker's live commission schedule. The tick table's
shape is stable and widely reproduced; the commission rate in particular varies
by broker, by channel, and by turnover tier, and the default here is a plausible
retail internet rate rather than a quoted one.

Treat backtest costs computed from these values as an order-of-magnitude
estimate. Verify before trusting any specific number, and edit
`configs/market.yaml` rather than this file when you do.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

#: (upper price bound, tick). The last entry's bound is infinite.
#: Reconstructed from SET's published tick-size table for common shares.
TICK_TABLE: tuple[tuple[float, float], ...] = (
    (2.0, 0.01),
    (5.0, 0.02),
    (10.0, 0.05),
    (25.0, 0.10),
    (100.0, 0.25),
    (200.0, 0.50),
    (400.0, 1.00),
    (float("inf"), 2.00),
)

_TICK_BOUNDS = [b for b, _ in TICK_TABLE]
_TICK_VALUES = [t for _, t in TICK_TABLE]

#: Standard board lot for common shares on the main board.
BOARD_LOT = 100

#: Retail internet commission, fraction of turnover. RECONSTRUCTED.
DEFAULT_COMMISSION_RATE = 0.00157
#: Thai VAT, charged on the commission (not on turnover). RECONSTRUCTED.
DEFAULT_VAT_RATE = 0.07
#: Ceiling/floor band on the previous close.
DEFAULT_PRICE_LIMIT = 0.30


def tick_size(price: float) -> float:
    """Minimum price increment at `price`.

    Bands are half-open on the upper bound: a price of exactly 25.00 sits in the
    25-100 band and ticks at 0.25.
    """
    if price <= 0:
        return _TICK_VALUES[0]
    idx = bisect_right(_TICK_BOUNDS, price)
    return _TICK_VALUES[min(idx, len(_TICK_VALUES) - 1)]


def snap_to_tick(price: float, *, mode: str = "nearest") -> float:
    """Snap `price` onto a valid tick.

    `mode` is "nearest", "down" (conservative for a sell) or "up"
    (conservative for a buy). Iterated, because rounding can carry a price
    across a band boundary where a different tick size applies.
    """
    if price <= 0:
        return 0.0
    for _ in range(4):
        tick = tick_size(price)
        n = price / tick
        if mode == "down":
            snapped = math.floor(n + 1e-9) * tick
        elif mode == "up":
            snapped = math.ceil(n - 1e-9) * tick
        else:
            snapped = round(n) * tick
        snapped = round(snapped, 6)
        if abs(tick_size(snapped) - tick) < 1e-12:
            return snapped
        price = snapped
    return round(price, 6)


def is_on_tick(price: float, *, atol: float = 1e-6) -> bool:
    if price <= 0:
        return False
    tick = tick_size(price)
    n = price / tick
    return abs(n - round(n)) * tick < atol


def round_to_lot(shares: float, *, lot: int = BOARD_LOT) -> int:
    """Round a share count down to a whole board lot. Never rounds up —
    rounding up could exceed available cash."""
    if shares <= 0:
        return 0
    return int(shares // lot) * lot


def is_on_lot(shares: float, *, lot: int = BOARD_LOT) -> bool:
    return shares >= 0 and float(shares).is_integer() and int(shares) % lot == 0


@dataclass(frozen=True)
class FeeSchedule:
    """Commission plus VAT, charged on both legs.

    RECONSTRUCTED — see the module docstring.
    """

    commission_rate: float = DEFAULT_COMMISSION_RATE
    vat_rate: float = DEFAULT_VAT_RATE
    minimum_commission: float = 0.0

    def commission(self, turnover: float) -> float:
        """Fee on one leg of `turnover` baht. Always >= 0."""
        turnover = abs(float(turnover))
        if turnover == 0:
            return 0.0
        base = max(turnover * self.commission_rate, self.minimum_commission)
        return base * (1.0 + self.vat_rate)

    @property
    def round_trip_rate(self) -> float:
        """Approximate all-in cost of a round trip, as a fraction of turnover."""
        return 2 * self.commission_rate * (1 + self.vat_rate)


def price_limits(
    previous_close: float, *, band: float = DEFAULT_PRICE_LIMIT
) -> tuple[float, float]:
    """Ceiling and floor for the session, snapped to valid ticks."""
    if previous_close <= 0:
        return (0.0, 0.0)
    floor = snap_to_tick(previous_close * (1 - band), mode="up")
    ceiling = snap_to_tick(previous_close * (1 + band), mode="down")
    return (floor, ceiling)


def within_limits(
    price: float, previous_close: float, *, band: float = DEFAULT_PRICE_LIMIT
) -> bool:
    floor, ceiling = price_limits(previous_close, band=band)
    return floor - 1e-9 <= price <= ceiling + 1e-9
