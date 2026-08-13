"""Rule-based agents — upstream `agent/` notebooks 1, 2, 3, 23, plus a baseline.

These come first because they are cheap and they validate the market layer: if
buy-and-hold does not behave sensibly through `SETMarket`, nothing built on top
of it will. They need no training pass, so `trainable = False`.
"""

from __future__ import annotations

import numpy as np

from .base import BUY, HOLD, SELL, Agent, Observation, register


@register("buy_and_hold")
class BuyAndHold(Agent):
    """Buy once on the first bar, hold to the end. The agent baseline.

    Every agent is measured against this the way every forecaster is measured
    against `NaiveLag`. It is pinned to the top of the agent results table.
    """

    trainable = False

    def reset(self) -> None:
        self._bought = False

    def act(self, obs: Observation) -> int:
        if not self._bought and obs.cash > obs.close * 100:
            self._bought = True
            return BUY
        return HOLD


@register("turtle")
class Turtle(Agent):
    """Upstream `1.turtle-agent.ipynb` — breakout of a rolling high/low channel."""

    trainable = False

    def __init__(self, *, count: int = 20, **params) -> None:
        super().__init__(count=count, **params)
        self.count = int(count)
        self._close: list[float] = []

    def reset(self) -> None:
        self._close = []

    def act(self, obs: Observation) -> int:
        self._close.append(obs.close)
        if len(self._close) <= self.count:
            return HOLD
        window = self._close[-(self.count + 1) : -1]
        if obs.close >= max(window):
            return BUY
        if obs.close <= min(window):
            return SELL
        return HOLD


@register("moving_average")
class MovingAverage(Agent):
    """Upstream `2.moving-average-agent.ipynb` — fast/slow crossover."""

    trainable = False

    def __init__(self, *, short_window: int = 5, long_window: int = 20, **params) -> None:
        super().__init__(short_window=short_window, long_window=long_window, **params)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self._close: list[float] = []
        self._above = False

    def reset(self) -> None:
        self._close = []
        self._above = False

    def act(self, obs: Observation) -> int:
        self._close.append(obs.close)
        if len(self._close) < self.long_window:
            return HOLD
        fast = float(np.mean(self._close[-self.short_window :]))
        slow = float(np.mean(self._close[-self.long_window :]))
        above = fast > slow
        crossed_up = above and not self._above
        crossed_down = (not above) and self._above
        self._above = above
        if crossed_up:
            return BUY
        if crossed_down:
            return SELL
        return HOLD


@register("signal_rolling")
class SignalRolling(Agent):
    """Upstream `3.signal-rolling-agent.ipynb` — act on a rolling momentum signal."""

    trainable = False

    def __init__(self, *, delay: int = 4, threshold: float = 0.0, **params) -> None:
        super().__init__(delay=delay, threshold=threshold, **params)
        self.delay = int(delay)
        self.threshold = float(threshold)
        self._close: list[float] = []

    def reset(self) -> None:
        self._close = []

    def act(self, obs: Observation) -> int:
        self._close.append(obs.close)
        if len(self._close) <= self.delay:
            return HOLD
        past = self._close[-(self.delay + 1)]
        if past <= 0:
            return HOLD
        change = obs.close / past - 1.0
        if change > self.threshold:
            return BUY
        if change < -self.threshold:
            return SELL
        return HOLD


@register("abcd")
class ABCD(Agent):
    """Upstream `23.abcd-strategy-agent.ipynb` — the ABCD retracement pattern.

    A-B is an impulse leg, B-C a retracement of it, and the entry is taken when
    price turns up from C, targeting D. The upstream notebook hard-codes the
    ratio bounds; they are parameters here.
    """

    trainable = False

    def __init__(
        self,
        *,
        lookback: int = 20,
        min_retracement: float = 0.382,
        max_retracement: float = 0.886,
        **params,
    ) -> None:
        super().__init__(
            lookback=lookback,
            min_retracement=min_retracement,
            max_retracement=max_retracement,
            **params,
        )
        self.lookback = int(lookback)
        self.min_retracement = float(min_retracement)
        self.max_retracement = float(max_retracement)
        self._close: list[float] = []

    def reset(self) -> None:
        self._close = []

    def act(self, obs: Observation) -> int:
        self._close.append(obs.close)
        if len(self._close) < self.lookback + 2:
            return HOLD

        window = np.array(self._close[-self.lookback :])
        a_idx = int(np.argmin(window))
        if a_idx >= len(window) - 2:
            return HOLD
        b_idx = a_idx + 1 + int(np.argmax(window[a_idx + 1 :]))
        if b_idx >= len(window) - 1:
            return HOLD

        a, b = float(window[a_idx]), float(window[b_idx])
        leg = b - a
        if leg <= 0:
            return HOLD

        c = float(np.min(window[b_idx:]))
        retracement = (b - c) / leg
        if not (self.min_retracement <= retracement <= self.max_retracement):
            return HOLD

        # C is in place; enter when price turns back up from it.
        if obs.close > c and self._close[-2] <= c * 1.001:
            return BUY
        if obs.close >= b:  # target reached
            return SELL
        return HOLD


@register("mean_reversion")
class MeanReversion(Agent):
    """Not an upstream notebook — a z-score reversion rule, kept as a sanity check.

    Included because the upstream rule set is entirely trend-following, and a
    counter-trend rule is a useful control when reading the results table.
    """

    trainable = False

    def __init__(
        self, *, window: int = 20, entry_z: float = -1.5, exit_z: float = 0.0, **params
    ) -> None:
        super().__init__(window=window, entry_z=entry_z, exit_z=exit_z, **params)
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self._close: list[float] = []

    def reset(self) -> None:
        self._close = []

    def act(self, obs: Observation) -> int:
        self._close.append(obs.close)
        if len(self._close) < self.window:
            return HOLD
        w = np.array(self._close[-self.window :])
        sd = w.std()
        if sd < 1e-9:
            return HOLD
        z = (obs.close - w.mean()) / sd
        if z <= self.entry_z:
            return BUY
        if z >= self.exit_z and obs.shares > 0:
            return SELL
        return HOLD
