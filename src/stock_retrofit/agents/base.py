"""Agent interface and the walk-forward backtest loop.

Two things here are corrections to upstream, not translations of it.

**Agents get a real holdout.** In `agent/6.evolution-strategy-agent.ipynb`,
`get_reward()` (the training objective) and `buy()` (the reported result) both
iterate the *same* `self.trend`. Every published agent return in that repo is
in-sample. Here `fit` sees the training block and `run` executes on the test
block, fold by fold, and never the two together.

**Every agent trades through `SETMarket`.** Upstream transacts one share at zero
cost. Each agent is run twice — frictionless and with SET frictions — and the
gap between the two is reported as a headline (R11).

Note on the upstream bug: `starting_money -= close[t]` in that notebook reads a
module-level global rather than `self.trend[t]`. It is silent only because the
two happen to hold the same list, and it detonates the moment two tickers share
a process. Nothing here reads a global — position and cash live on `SETMarket`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..market import MarketConfig, SETMarket
from ..market.set_market import BacktestResult

HOLD, BUY, SELL = 0, 1, 2
ACTION_NAMES = {HOLD: "hold", BUY: "buy", SELL: "sell"}


@dataclass
class Observation:
    """What an agent sees on one bar. Strictly causal — nothing after `t`."""

    t: int
    #: (timestep, n_features) scaled window ending at `t`.
    window: np.ndarray
    close: float
    cash: float
    shares: int
    equity: float
    initial_cash: float

    @property
    def flat(self) -> np.ndarray:
        return self.window.reshape(-1)

    @property
    def invested(self) -> float:
        """Fraction of equity currently in the stock — the inventory signal."""
        return 0.0 if self.equity <= 0 else (self.shares * self.close) / self.equity

    def state(self) -> np.ndarray:
        """Flat state vector: the price window plus portfolio context."""
        return np.concatenate([self.flat, [self.invested, self.equity / self.initial_cash - 1.0]])


class Agent(ABC):
    """A trading agent. Trains on one block of bars, acts on another."""

    name: str = "unnamed"
    upstream: str = ""
    #: Set False for agents that need no training pass (pure rule-based ones).
    trainable: bool = True

    def __init__(self, **params: Any) -> None:
        self.params = params

    def reset(self) -> None:
        """Clear per-episode state. Called before every fit and every run."""

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        """Train on the training block only. Default: nothing to train."""

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """Return HOLD, BUY or SELL for bar `obs.t`."""

    def describe(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({bits})"


_REGISTRY: dict[str, Callable[..., Agent]] = {}


def register(kind: str):
    def wrap(cls):
        if kind in _REGISTRY:
            raise KeyError(f"agent kind {kind!r} is already registered")
        _REGISTRY[kind] = cls
        return cls

    return wrap


def build(kind: str, *, name: str = "", upstream: str = "", **params: Any) -> Agent:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown agent kind {kind!r}; registered: {sorted(_REGISTRY)}")
    agent = _REGISTRY[kind](**params)
    agent.name = name or kind
    agent.upstream = upstream
    return agent


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_episode(
    agent: Agent,
    bars: pd.DataFrame,
    features: np.ndarray,
    config: MarketConfig,
    *,
    timestep: int = 20,
    start: int = 0,
    label: str = "",
    symbol: str = "?",
) -> BacktestResult:
    """Execute `agent` over `bars` through a `SETMarket`. No training happens here."""
    market = SETMarket(bars, config)
    agent.reset()

    first = max(start, timestep - 1)
    equity = np.full(len(bars), config.initial_cash, dtype=float)

    for t in range(first, len(bars)):
        obs = Observation(
            t=t,
            window=features[t - timestep + 1 : t + 1],
            close=float(bars["close"].iloc[t]),
            cash=market.cash,
            shares=market.shares,
            equity=market.equity(t),
            initial_cash=config.initial_cash,
        )
        action = agent.act(obs)
        if action == BUY:
            market.buy(t, fraction=1.0)
        elif action == SELL:
            market.sell(t)
        equity[t] = market.equity(t)

    if first > 0:
        equity[:first] = config.initial_cash
    market.assert_invariants()

    return BacktestResult(
        symbol=symbol,
        label=label or agent.name,
        frictionless=config.frictionless,
        dates=bars["date"].reset_index(drop=True),
        equity=equity,
        fills=market.fills,
        initial_cash=config.initial_cash,
    )


@dataclass
class AgentFoldResult:
    fold_index: int
    with_frictions: BacktestResult
    frictionless: BacktestResult

    @property
    def friction_gap(self) -> float:
        """Frictionless return minus friction return — the R11 headline."""
        return self.frictionless.total_return - self.with_frictions.total_return


@dataclass
class AgentResult:
    agent_name: str
    symbol: str
    upstream: str = ""
    folds: list[AgentFoldResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.folds)

    def _pooled(self, attr: str) -> dict:
        runs = [getattr(f, attr) for f in self.folds]
        rets = np.array([r.total_return for r in runs], dtype=float)
        daily = np.concatenate([r.daily_returns() for r in runs]) if runs else np.array([])
        from ..eval.metrics import sharpe as _sharpe

        return {
            "mean_fold_return": float(rets.mean()) if len(rets) else float("nan"),
            "compounded_return": float(np.prod(1 + rets) - 1) if len(rets) else float("nan"),
            "sharpe": _sharpe(daily),
            "max_drawdown": float(min(r.max_drawdown() for r in runs)) if runs else float("nan"),
            "n_trades": int(sum(r.n_trades for r in runs)),
            "total_costs": float(sum(r.total_costs for r in runs)),
        }

    def pooled_friction(self) -> dict:
        return self._pooled("with_frictions")

    def pooled_frictionless(self) -> dict:
        return self._pooled("frictionless")

    @property
    def friction_gap(self) -> float:
        if not self.folds:
            return float("nan")
        return (
            self.pooled_frictionless()["mean_fold_return"]
            - self.pooled_friction()["mean_fold_return"]
        )
