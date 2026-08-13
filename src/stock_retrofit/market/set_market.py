"""`SETMarket` — the friction layer every agent trades through (spec R10-R12).

Upstream's agents buy and sell **one share** at zero cost with unlimited
shorting. None of those three things is true on SET, and all three flatter the
result. This simulator enforces:

* a 100-share board lot,
* tick-size snapping,
* commission + VAT on **both** legs,
* the +/-30% ceiling/floor against the previous close,
* no short selling unless an explicit SBL flag is set,
* an optional cap on participation in the session's volume.

Set `frictionless=True` and every one of those is switched off, which is how
R11's paired run works: the same agent, the same bars, the two numbers, and the
gap between them reported as a headline rather than a footnote.

No code path here places a real order. It is a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rules import (
    BOARD_LOT,
    DEFAULT_PRICE_LIMIT,
    FeeSchedule,
    is_on_lot,
    is_on_tick,
    price_limits,
    round_to_lot,
    snap_to_tick,
)


@dataclass
class MarketConfig:
    initial_cash: float = 1_000_000.0
    board_lot: int = BOARD_LOT
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    price_limit: float = DEFAULT_PRICE_LIMIT
    allow_short: bool = False
    #: Fraction of the session's volume an order may take, or None for no cap.
    participation_cap: float | None = None
    #: Adverse slippage in ticks. Default 0 so the friction gap is attributable
    #: purely to the rules above rather than to a modelling choice of ours.
    slippage_ticks: float = 0.0
    frictionless: bool = False

    def frictionless_twin(self) -> MarketConfig:
        """The same configuration with every friction switched off."""
        return MarketConfig(
            initial_cash=self.initial_cash,
            board_lot=1,
            fees=FeeSchedule(commission_rate=0.0, vat_rate=0.0),
            price_limit=float("inf"),
            allow_short=self.allow_short,
            participation_cap=None,
            slippage_ticks=0.0,
            frictionless=True,
        )


@dataclass
class Fill:
    t: int
    date: pd.Timestamp
    side: str  # "buy" | "sell"
    requested: float
    filled: int
    price: float
    gross: float
    commission: float
    cash_delta: float
    #: Why the order was refused, or *why it was reduced* — a participation cap
    #: trims an order without refusing it, so this can be set on a fill that
    #: went through.
    rejected: str | None = None

    @property
    def ok(self) -> bool:
        """Did shares change hands?

        Deliberately independent of `rejected`. A participation-capped order
        fills a smaller size and records the cap as its reason, so keying `ok`
        off `rejected` would classify a real trade as a non-event — silently
        undercounting trades and costs, and skipping it in `assert_invariants`.
        That failure mode is invisible on KBANK and SCB (no cap) and hits BAY,
        the one ticker where the cap is on by default.
        """
        return self.filled > 0

    @property
    def reduced(self) -> bool:
        """Filled, but for less than was asked."""
        return self.filled > 0 and self.rejected is not None


@dataclass
class BacktestResult:
    symbol: str
    label: str
    frictionless: bool
    dates: pd.Series
    equity: np.ndarray
    fills: list[Fill]
    initial_cash: float

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1]) if len(self.equity) else self.initial_cash

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_cash - 1.0

    @property
    def n_trades(self) -> int:
        return sum(1 for f in self.fills if f.ok)

    @property
    def total_costs(self) -> float:
        return float(sum(f.commission for f in self.fills if f.ok))

    @property
    def rejections(self) -> dict[str, int]:
        """Orders refused outright, by reason. Excludes orders merely trimmed."""
        out: dict[str, int] = {}
        for f in self.fills:
            if f.rejected and not f.ok:
                out[f.rejected] = out.get(f.rejected, 0) + 1
        return out

    @property
    def reductions(self) -> dict[str, int]:
        """Orders that filled for less than requested, by reason (the cap)."""
        out: dict[str, int] = {}
        for f in self.fills:
            if f.reduced:
                out[f.rejected] = out.get(f.rejected, 0) + 1
        return out

    def daily_returns(self) -> np.ndarray:
        if len(self.equity) < 2:
            return np.array([])
        return np.diff(self.equity) / self.equity[:-1]

    def sharpe(self) -> float:
        from ..eval.metrics import sharpe as _sharpe

        return _sharpe(self.daily_returns())

    def max_drawdown(self) -> float:
        if not len(self.equity):
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        return float((self.equity / peak - 1.0).min())

    def summary(self) -> dict:
        return {
            "label": self.label,
            "symbol": self.symbol,
            "frictionless": self.frictionless,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "sharpe": self.sharpe(),
            "max_drawdown": self.max_drawdown(),
            "n_trades": self.n_trades,
            "total_costs": self.total_costs,
            "rejections": self.rejections,
        }


class SETMarket:
    """A single-symbol, daily-bar SET simulator.

    Fills happen at the session close, snapped to a valid tick and clamped into
    the day's range — an agent cannot transact at a price the market never
    printed.
    """

    def __init__(self, bars: pd.DataFrame, config: MarketConfig | None = None) -> None:
        self.bars = bars.reset_index(drop=True)
        self.config = config or MarketConfig()
        self.close = self.bars["close"].to_numpy(dtype=float)
        self.high = self.bars["high"].to_numpy(dtype=float)
        self.low = self.bars["low"].to_numpy(dtype=float)
        self.volume = self.bars["volume"].to_numpy(dtype=float)
        self.dates = self.bars["date"]
        self.reset()

    # ---- state ------------------------------------------------------------

    def reset(self) -> None:
        self.cash = float(self.config.initial_cash)
        self.shares = 0
        self.fills: list[Fill] = []

    def equity(self, t: int) -> float:
        return self.cash + self.shares * float(self.close[t])

    # ---- pricing ----------------------------------------------------------

    def fill_price(self, t: int, side: str) -> float:
        """Executable price at bar `t`, or 0.0 if the order cannot be priced."""
        price = float(self.close[t])
        cfg = self.config
        if cfg.frictionless:
            return price

        if cfg.slippage_ticks:
            from .rules import tick_size

            adverse = tick_size(price) * cfg.slippage_ticks
            price = price + adverse if side == "buy" else price - adverse

        price = snap_to_tick(price, mode="up" if side == "buy" else "down")
        # Never fill outside the range the market actually traded in.
        price = min(max(price, float(self.low[t])), float(self.high[t]))
        price = snap_to_tick(price, mode="down" if side == "buy" else "up")
        return max(price, 0.0)

    def _limit_ok(self, t: int, price: float) -> bool:
        cfg = self.config
        if cfg.frictionless or t == 0 or not np.isfinite(cfg.price_limit):
            return True
        floor, ceiling = price_limits(float(self.close[t - 1]), band=cfg.price_limit)
        return floor - 1e-9 <= price <= ceiling + 1e-9

    def _cap_shares(self, t: int, shares: int) -> tuple[int, str | None]:
        cfg = self.config
        if cfg.frictionless or cfg.participation_cap is None:
            return shares, None
        allowed = int(self.volume[t] * cfg.participation_cap)
        if allowed < shares:
            capped = round_to_lot(allowed, lot=self._lot)
            return capped, ("participation_cap" if capped < shares else None)
        return shares, None

    @property
    def _lot(self) -> int:
        return 1 if self.config.frictionless else self.config.board_lot

    # ---- orders -----------------------------------------------------------

    def buy(self, t: int, *, shares: int | None = None, fraction: float | None = None) -> Fill:
        """Buy `shares`, or as much as `fraction` of equity affords."""
        cfg = self.config
        price = self.fill_price(t, "buy")
        record = dict(t=t, date=self.dates.iloc[t], side="buy")

        if price <= 0:
            return self._reject(record, shares or 0, "unpriceable")
        if not self._limit_ok(t, price):
            return self._reject(record, shares or 0, "outside_price_limit")

        if shares is None:
            budget = self.cash * (1.0 if fraction is None else float(fraction))
            # Leave room for the commission so the order cannot overdraw.
            unit_cost = price * (1 + cfg.fees.round_trip_rate / 2)
            shares = int(budget / unit_cost) if unit_cost > 0 else 0

        shares = round_to_lot(shares, lot=self._lot)
        shares, cap_reason = self._cap_shares(t, shares)
        if shares <= 0:
            return self._reject(record, shares, cap_reason or "below_board_lot")

        gross = shares * price
        commission = 0.0 if cfg.frictionless else cfg.fees.commission(gross)
        if gross + commission > self.cash + 1e-9:
            # Step down to what the cash actually covers.
            affordable = round_to_lot(
                self.cash / (price * (1 + cfg.fees.commission_rate * (1 + cfg.fees.vat_rate))),
                lot=self._lot,
            )
            if affordable <= 0:
                return self._reject(record, shares, "insufficient_cash")
            shares = affordable
            gross = shares * price
            commission = 0.0 if cfg.frictionless else cfg.fees.commission(gross)

        self.cash -= gross + commission
        self.shares += shares
        fill = Fill(
            **record,
            requested=shares,
            filled=shares,
            price=price,
            gross=gross,
            commission=commission,
            cash_delta=-(gross + commission),
            rejected=cap_reason,
        )
        self.fills.append(fill)
        return fill

    def sell(self, t: int, *, shares: int | None = None, fraction: float | None = None) -> Fill:
        """Sell `shares` (default: the whole position)."""
        cfg = self.config
        price = self.fill_price(t, "sell")
        record = dict(t=t, date=self.dates.iloc[t], side="sell")

        if price <= 0:
            return self._reject(record, shares or 0, "unpriceable")
        if not self._limit_ok(t, price):
            return self._reject(record, shares or 0, "outside_price_limit")

        if shares is None:
            shares = self.shares if fraction is None else int(self.shares * float(fraction))

        shares = round_to_lot(shares, lot=self._lot)
        if not cfg.allow_short and shares > self.shares:
            shares = round_to_lot(self.shares, lot=self._lot)
        shares, cap_reason = self._cap_shares(t, shares)

        if shares <= 0:
            reason = cap_reason or ("no_position" if self.shares <= 0 else "below_board_lot")
            return self._reject(record, shares, reason)
        if not cfg.allow_short and shares > self.shares:
            return self._reject(record, shares, "short_selling_disabled")

        gross = shares * price
        commission = 0.0 if cfg.frictionless else cfg.fees.commission(gross)
        self.cash += gross - commission
        self.shares -= shares
        fill = Fill(
            **record,
            requested=shares,
            filled=shares,
            price=price,
            gross=gross,
            commission=commission,
            cash_delta=gross - commission,
            rejected=cap_reason,
        )
        self.fills.append(fill)
        return fill

    def _reject(self, record: dict, shares, reason: str) -> Fill:
        fill = Fill(
            **record,
            requested=float(shares or 0),
            filled=0,
            price=0.0,
            gross=0.0,
            commission=0.0,
            cash_delta=0.0,
            rejected=reason,
        )
        self.fills.append(fill)
        return fill

    # ---- validation --------------------------------------------------------

    def assert_invariants(self) -> None:
        """Property checks the market must satisfy at all times (spec R10 tests)."""
        cfg = self.config
        for f in self.fills:
            if not f.ok:
                continue
            assert f.commission >= 0, f"negative commission on {f.date}"
            assert self.low[f.t] - 1e-9 <= f.price <= self.high[f.t] + 1e-9, (
                f"fill at {f.price} on {f.date} is outside the day's "
                f"[{self.low[f.t]}, {self.high[f.t]}]"
            )
            if not cfg.frictionless:
                assert is_on_tick(f.price), f"fill at {f.price} on {f.date} is off-tick"
                assert is_on_lot(
                    f.filled, lot=cfg.board_lot
                ), f"fill of {f.filled} shares on {f.date} is not a whole board lot"
        if not cfg.allow_short:
            assert self.shares >= 0, "short position held with shorting disabled"
