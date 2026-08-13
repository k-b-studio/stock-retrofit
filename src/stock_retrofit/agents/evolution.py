"""Evolution strategies — upstream `agent/` notebooks 6, 21, 22.

    6.  evolution-strategy               -> EvolutionStrategyAgent
    21. neuro-evolution                  -> NeuroEvolutionAgent
    22. neuro-evolution-novelty-search   -> NeuroEvolutionAgent(novelty=True)

**Two corrections to upstream, both load-bearing.**

*The holdout.* `Agent.get_reward()` (the training objective) and `Agent.buy()`
(the reported equity curve) in notebook 6 iterate the same `self.trend`. Every
published return from those notebooks is in-sample. Here the fitness function
sees the training block and the reported result comes from replaying the evolved
weights through `SETMarket` on held-out folds.

*The global.* Notebook 6's buy branch reads `starting_money -= close[t]`, a
module-level global, where the sell branch reads `self.trend[t]`. Fixed by
construction: the fitness function is a pure function of the arrays passed to it
and holds no module state.

**On training speed.** Population x generations x bars is millions of steps, so
fitness is evaluated with a vectorised numpy equity curve rather than a
per-step simulator. Evaluation still runs through `SETMarket` with every
friction applied — training may use any objective; only the evaluation is a
claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..market import MarketConfig
from .base import HOLD, Agent, Observation, register
from .env import portfolio_curve


class _LinearPolicy:
    """A single hidden-layer network flattened into one weight vector.

    Kept deliberately small: evolution searches without gradients, so parameter
    count is the binding constraint on how far a modest population can get.
    """

    def __init__(self, n_inputs: int, hidden: int, n_actions: int = 3) -> None:
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.n_actions = n_actions
        self.size = n_inputs * hidden + hidden + hidden * n_actions + n_actions

    def unpack(self, weights: np.ndarray) -> tuple[np.ndarray, ...]:
        i = 0
        w1 = weights[i : i + self.n_inputs * self.hidden].reshape(self.n_inputs, self.hidden)
        i += self.n_inputs * self.hidden
        b1 = weights[i : i + self.hidden]
        i += self.hidden
        w2 = weights[i : i + self.hidden * self.n_actions].reshape(self.hidden, self.n_actions)
        i += self.hidden * self.n_actions
        b2 = weights[i : i + self.n_actions]
        return w1, b1, w2, b2

    def actions(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Vectorised forward pass over every state at once."""
        w1, b1, w2, b2 = self.unpack(weights)
        h = np.tanh(states @ w1 + b1)
        return np.argmax(h @ w2 + b2, axis=1)

    def action(self, state: np.ndarray, weights: np.ndarray) -> int:
        w1, b1, w2, b2 = self.unpack(weights)
        h = np.tanh(state @ w1 + b1)
        return int(np.argmax(h @ w2 + b2))


def _state_matrix(features: np.ndarray, timestep: int) -> np.ndarray:
    """Every window in the block, stacked. Portfolio context is held at zero here
    because the fitness function is all-in/all-out and path-independent."""
    n = len(features)
    positions = np.arange(timestep - 1, n)
    idx = positions[:, None] - np.arange(timestep - 1, -1, -1)[None, :]
    windows = features[idx].reshape(len(positions), -1)
    return np.hstack([windows, np.zeros((len(positions), 2), dtype=np.float32)])


class _EvolutionBase(Agent):
    def __init__(
        self,
        *,
        hidden: int = 16,
        population: int = 20,
        generations: int = 40,
        sigma: float = 0.1,
        learning_rate: float = 0.05,
        **params,
    ) -> None:
        super().__init__(
            hidden=hidden,
            population=population,
            generations=generations,
            sigma=sigma,
            learning_rate=learning_rate,
            **params,
        )
        self.hidden = int(hidden)
        self.population = int(population)
        self.generations = int(generations)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.policy: _LinearPolicy | None = None
        self.weights: np.ndarray | None = None
        self._timestep = 20

    def reset(self) -> None:
        pass

    def _setup(self, features: np.ndarray, timestep: int) -> tuple[np.ndarray, np.ndarray]:
        self._timestep = timestep
        n_inputs = timestep * features.shape[1] + 2
        self.policy = _LinearPolicy(n_inputs, self.hidden)
        rng = np.random.default_rng(0)
        self.weights = rng.normal(0, 0.1, self.policy.size)
        return self.weights, rng

    def _fitness(
        self, weights: np.ndarray, states: np.ndarray, close: np.ndarray, cost: float
    ) -> float:
        actions = self.policy.actions(states, weights)
        curve = portfolio_curve(close[self._timestep - 1 :], actions, cost_per_turn=cost)
        return float(curve[-1] - 1.0) if len(curve) else 0.0

    def act(self, obs: Observation) -> int:
        if self.policy is None or self.weights is None:
            return HOLD
        return int(self.policy.action(obs.state(), self.weights))


@register("evolution_strategy")
class EvolutionStrategyAgent(_EvolutionBase):
    """OpenAI-style ES: perturb, weight by standardised fitness, step. Notebook 6."""

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        timestep = self.params.get("timestep", 20)
        weights, rng = self._setup(features, timestep)
        states = _state_matrix(features, timestep)
        close = bars["close"].to_numpy(dtype=float)
        cost = config.fees.round_trip_rate

        for _ in range(self.generations):
            noise = rng.normal(0, 1, (self.population, self.policy.size))
            rewards = np.array(
                [
                    self._fitness(weights + self.sigma * noise[i], states, close, cost)
                    for i in range(self.population)
                ]
            )
            sd = rewards.std()
            if sd < 1e-12:
                continue
            standardised = (rewards - rewards.mean()) / sd
            weights = weights + (self.learning_rate / (self.population * self.sigma)) * (
                noise.T @ standardised
            )

        self.weights = weights


@register("neuro_evolution")
class NeuroEvolutionAgent(_EvolutionBase):
    """Genetic search over network weights. Notebooks 21 and 22.

    With `novelty=True` (notebook 22) fitness is blended with a novelty score —
    the mean distance from a candidate's action sequence to the archive of
    behaviours already seen. Novelty search rewards doing something *different*
    rather than something profitable, which is the point: it escapes the local
    optimum of "hold forever" that plain fitness falls into on a rising market.
    """

    def __init__(
        self,
        *,
        novelty: bool = False,
        novelty_weight: float = 0.5,
        archive_size: int = 30,
        elite_fraction: float = 0.25,
        mutation_rate: float = 0.15,
        **params,
    ) -> None:
        super().__init__(
            novelty=novelty,
            novelty_weight=novelty_weight,
            archive_size=archive_size,
            elite_fraction=elite_fraction,
            mutation_rate=mutation_rate,
            **params,
        )
        self.novelty = bool(novelty)
        self.novelty_weight = float(novelty_weight)
        self.archive_size = int(archive_size)
        self.elite_fraction = float(elite_fraction)
        self.mutation_rate = float(mutation_rate)

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        timestep = self.params.get("timestep", 20)
        _, rng = self._setup(features, timestep)
        states = _state_matrix(features, timestep)
        close = bars["close"].to_numpy(dtype=float)
        cost = config.fees.round_trip_rate

        pop = rng.normal(0, 0.1, (self.population, self.policy.size))
        archive: list[np.ndarray] = []
        n_elite = max(2, int(self.population * self.elite_fraction))
        best_weights, best_fitness = pop[0], -np.inf

        for _ in range(self.generations):
            behaviours = [self.policy.actions(states, w) for w in pop]
            fitness = np.array(
                [self._fitness(pop[i], states, close, cost) for i in range(self.population)]
            )

            top = int(np.argmax(fitness))
            if fitness[top] > best_fitness:
                best_fitness, best_weights = float(fitness[top]), pop[top].copy()

            score = fitness.copy()
            if self.novelty and archive:
                novelty = np.array(
                    [np.mean([np.mean(b != a) for a in archive]) for b in behaviours]
                )
                fs, ns = fitness.std(), novelty.std()
                fz = (fitness - fitness.mean()) / fs if fs > 1e-12 else np.zeros_like(fitness)
                nz = (novelty - novelty.mean()) / ns if ns > 1e-12 else np.zeros_like(novelty)
                score = (1 - self.novelty_weight) * fz + self.novelty_weight * nz
            if self.novelty:
                archive.append(behaviours[top])
                archive[:] = archive[-self.archive_size :]

            elite_idx = np.argsort(score)[-n_elite:]
            elite = pop[elite_idx]
            children = [elite[i % len(elite)].copy() for i in range(self.population - n_elite)]
            for child in children:
                mask = rng.random(len(child)) < self.mutation_rate
                child[mask] += rng.normal(0, self.sigma, int(mask.sum()))
            pop = np.vstack([elite, np.array(children)])

        # Report the best *fitness*, not the most novel — novelty is a search
        # device, not an objective.
        self.weights = best_weights
