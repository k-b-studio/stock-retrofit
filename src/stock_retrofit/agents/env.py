"""A fast training environment.

Learning agents need tens of thousands of simulated steps per fold. Driving
`SETMarket` for all of them would be honest but unusably slow, so training runs
against this lightweight numpy environment instead: all-in / all-out positions
with a proportional round-trip cost, which captures the economics that matter
for learning (churn is expensive) without the per-order bookkeeping.

**Evaluation never uses this.** Every reported number comes from replaying the
trained policy through `SETMarket` on held-out folds, with board lots, tick
snapping, commission + VAT, price limits and the participation cap all enforced.
Training may use any objective it likes; the claim is only ever as good as the
evaluation, and the evaluation is the real thing.
"""

from __future__ import annotations

import numpy as np

from .base import BUY, SELL


class TrainEnv:
    """All-in/all-out single-symbol episode over a fixed bar series."""

    def __init__(
        self,
        close: np.ndarray,
        features: np.ndarray,
        *,
        timestep: int = 20,
        cost_per_turn: float = 0.00336,
        initial_cash: float = 1_000_000.0,
    ) -> None:
        self.close = np.asarray(close, dtype=float)
        self.features = np.asarray(features, dtype=np.float32)
        self.timestep = int(timestep)
        self.cost = float(cost_per_turn)
        self.initial_cash = float(initial_cash)
        self.n = len(self.close)
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.timestep * self.features.shape[1] + 2

    @property
    def n_actions(self) -> int:
        return 3

    def reset(self) -> np.ndarray:
        self.t = self.timestep - 1
        self.cash = self.initial_cash
        self.shares = 0.0
        self.equity = self.initial_cash
        return self.state()

    def state(self) -> np.ndarray:
        window = self.features[self.t - self.timestep + 1 : self.t + 1].reshape(-1)
        invested = 0.0 if self.equity <= 0 else self.shares * self.close[self.t] / self.equity
        return np.concatenate([window, [invested, self.equity / self.initial_cash - 1.0]]).astype(
            np.float32
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        price = self.close[self.t]
        if action == BUY and self.cash > 0:
            traded = self.cash
            self.shares += traded / price
            self.cash = 0.0
            self.cash -= traded * self.cost / 2
        elif action == SELL and self.shares > 0:
            traded = self.shares * price
            self.cash += traded * (1 - self.cost / 2)
            self.shares = 0.0

        previous_equity = self.equity
        self.t += 1
        done = self.t >= self.n - 1
        self.equity = self.cash + self.shares * self.close[self.t]
        # Reward is the log growth of the portfolio — scale-free, and it
        # penalises drawdowns more than a plain difference would.
        reward = float(np.log(max(self.equity, 1e-9) / max(previous_equity, 1e-9)))
        return self.state(), reward, done


def portfolio_curve(
    close: np.ndarray,
    actions: np.ndarray,
    *,
    cost_per_turn: float = 0.00336,
    initial_cash: float = 1.0,
) -> np.ndarray:
    """Vectorised all-in/all-out equity curve for a whole action sequence.

    Used by the evolution strategies, whose training objective must evaluate a
    full episode per candidate per generation — a Python loop there costs
    minutes per fold.
    """
    close = np.asarray(close, dtype=float)
    actions = np.asarray(actions, dtype=int)
    position = np.zeros(len(close))
    held = 0.0
    for i, a in enumerate(actions):
        if a == BUY:
            held = 1.0
        elif a == SELL:
            held = 0.0
        position[i] = held

    returns = np.diff(close) / close[:-1]
    pos = position[:-1]
    turns = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * returns - turns * cost_per_turn / 2
    return initial_cash * np.cumprod(1 + net)
