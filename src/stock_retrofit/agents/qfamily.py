"""The Q-learning family — upstream `agent/` notebooks 5, 7-13, 18-20.

Eleven notebooks, one skeleton, four boolean flags:

    {double, duel, recurrent, curiosity}

    5.  q-learning                          -> all false
    7.  double-q-learning                   -> double
    8.  recurrent-q-learning                -> recurrent
    9.  double-recurrent-q-learning         -> double + recurrent
    10. duel-q-learning                     -> duel
    11. double-duel-q-learning              -> double + duel
    12. duel-recurrent-q-learning           -> duel + recurrent
    13. double-duel-recurrent-q-learning    -> double + duel + recurrent
    18. curiosity-q-learning                -> curiosity
    19. recurrent-curiosity-q-learning      -> recurrent + curiosity
    20. duel-curiosity-q-learning           -> duel + curiosity

What each flag means here:

* **double** — the online network picks the next action, the target network
  scores it, which removes the max-operator's optimism bias.
* **duel** — the head splits into a state-value stream and an advantage stream
  that recombine as `V + (A - mean A)`.
* **recurrent** — an LSTM reads the window as a sequence instead of an MLP
  reading it flattened.
* **curiosity** — a forward model predicts the next state's embedding, and its
  error is added to the reward as an intrinsic exploration bonus.

All four compose, which is why eleven notebooks reduce to one class.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..market import MarketConfig
from .base import Agent, Observation, register
from .env import TrainEnv


class _Encoder(nn.Module):
    """Flattened-MLP or LSTM encoder, depending on the `recurrent` flag."""

    def __init__(self, *, timestep: int, n_features: int, hidden: int, recurrent: bool) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.timestep = timestep
        self.n_features = n_features
        self.extra = 2  # invested fraction, equity drift
        if recurrent:
            self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
            self.merge = nn.Linear(hidden + self.extra, hidden)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(timestep * n_features + self.extra, hidden), nn.ReLU()
            )
        self.out_dim = hidden

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not self.recurrent:
            return self.mlp(state)
        window = state[:, : self.timestep * self.n_features].view(
            -1, self.timestep, self.n_features
        )
        tail = state[:, self.timestep * self.n_features :]
        h = self.rnn(window)[0][:, -1, :]
        return torch.relu(self.merge(torch.cat([h, tail], dim=-1)))


class _QNet(nn.Module):
    def __init__(self, encoder: _Encoder, n_actions: int, *, duel: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.duel = duel
        h = encoder.out_dim
        if duel:
            self.value = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
            self.advantage = nn.Sequential(
                nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, n_actions)
            )
        else:
            self.q = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, n_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(state)
        if not self.duel:
            return self.q(z)
        v = self.value(z)
        a = self.advantage(z)
        return v + (a - a.mean(dim=1, keepdim=True))


class _ForwardModel(nn.Module):
    """Curiosity: predict the next state embedding from (embedding, action)."""

    def __init__(self, dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + n_actions, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.n_actions = n_actions

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        onehot = nn.functional.one_hot(action, self.n_actions).float()
        return self.net(torch.cat([z, onehot], dim=-1))


@register("q_learning")
class QLearningAgent(Agent):
    """Deep Q-learning with double / duel / recurrent / curiosity switches."""

    def __init__(
        self,
        *,
        double: bool = False,
        duel: bool = False,
        recurrent: bool = False,
        curiosity: bool = False,
        hidden: int = 64,
        episodes: int = 20,
        gamma: float = 0.95,
        lr: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        target_sync: int = 100,
        train_every: int = 4,
        curiosity_weight: float = 0.05,
        **params,
    ) -> None:
        super().__init__(
            double=double,
            duel=duel,
            recurrent=recurrent,
            curiosity=curiosity,
            hidden=hidden,
            episodes=episodes,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            buffer_size=buffer_size,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            target_sync=target_sync,
            train_every=train_every,
            curiosity_weight=curiosity_weight,
            **params,
        )
        self.double = bool(double)
        self.duel = bool(duel)
        self.recurrent = bool(recurrent)
        self.curiosity = bool(curiosity)
        self.hidden = int(hidden)
        self.episodes = int(episodes)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.target_sync = int(target_sync)
        self.train_every = int(train_every)
        self.curiosity_weight = float(curiosity_weight)

        self.online: _QNet | None = None
        self.target: _QNet | None = None
        self.forward_model: _ForwardModel | None = None
        self._timestep = 0
        self._n_features = 0

    def reset(self) -> None:
        pass  # weights persist between fit and act; per-fold reset happens in fit

    def _build(self, timestep: int, n_features: int) -> None:
        self._timestep, self._n_features = timestep, n_features

        def encoder() -> _Encoder:
            return _Encoder(
                timestep=timestep,
                n_features=n_features,
                hidden=self.hidden,
                recurrent=self.recurrent,
            )

        self.online = _QNet(encoder(), 3, duel=self.duel)
        self.target = _QNet(encoder(), 3, duel=self.duel)
        self.target.load_state_dict(self.online.state_dict())
        self.forward_model = _ForwardModel(self.hidden, 3) if self.curiosity else None

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        timestep = self.params.get("timestep", 20)
        env = TrainEnv(
            bars["close"].to_numpy(),
            features,
            timestep=timestep,
            cost_per_turn=config.fees.round_trip_rate,
            initial_cash=config.initial_cash,
        )
        self._build(timestep, features.shape[1])

        params = list(self.online.parameters())
        if self.forward_model is not None:
            params += list(self.forward_model.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        buffer: deque = deque(maxlen=self.buffer_size)
        loss_fn = nn.SmoothL1Loss()
        steps = 0

        for episode in range(self.episodes):
            state = env.reset()
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start
                - (self.epsilon_start - self.epsilon_end) * episode / max(self.episodes - 1, 1),
            )
            done = False
            while not done:
                if random.random() < epsilon:
                    action = random.randrange(3)
                else:
                    with torch.no_grad():
                        action = int(
                            self.online(torch.from_numpy(state).float().unsqueeze(0)).argmax()
                        )
                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, float(done)))
                state = next_state
                steps += 1

                if len(buffer) >= self.batch_size and steps % self.train_every == 0:
                    self._learn(buffer, opt, loss_fn)
                if steps % self.target_sync == 0:
                    self.target.load_state_dict(self.online.state_dict())

    def _learn(self, buffer, opt, loss_fn) -> None:
        batch = random.sample(buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch, strict=True)
        s = torch.from_numpy(np.array(s)).float()
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        s2 = torch.from_numpy(np.array(s2)).float()
        d = torch.tensor(d, dtype=torch.float)

        intrinsic_loss = torch.tensor(0.0)
        if self.forward_model is not None:
            with torch.no_grad():
                z2 = self.online.encoder(s2)
            z = self.online.encoder(s)
            predicted = self.forward_model(z, a)
            per_sample = ((predicted - z2) ** 2).mean(dim=1)
            # Prediction error doubles as an exploration bonus.
            r = r + self.curiosity_weight * per_sample.detach()
            intrinsic_loss = per_sample.mean()

        with torch.no_grad():
            if self.double:
                next_action = self.online(s2).argmax(dim=1, keepdim=True)
                next_q = self.target(s2).gather(1, next_action).squeeze(1)
            else:
                next_q = self.target(s2).max(dim=1).values
            target = r + self.gamma * next_q * (1 - d)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = loss_fn(q, target) + intrinsic_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        opt.step()

    def act(self, obs: Observation) -> int:
        if self.online is None:
            return 0
        state = torch.from_numpy(obs.state()).float().unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            return int(self.online(state).argmax())
