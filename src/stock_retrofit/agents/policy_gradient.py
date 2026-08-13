"""Policy-gradient and actor-critic — upstream `agent/` notebooks 4 and 14-17.

    4.  policy-gradient                 -> PolicyGradientAgent
    14. actor-critic                    -> ActorCriticAgent()
    15. actor-critic-duel               -> ActorCriticAgent(duel=True)
    16. actor-critic-recurrent          -> ActorCriticAgent(recurrent=True)
    17. actor-critic-duel-recurrent     -> ActorCriticAgent(duel=True, recurrent=True)

**The upstream bug, fixed rather than ported.** In
`agent/4.policy-gradient-agent.ipynb` (and notebook 6) the buy branch reads
`starting_money -= close[t]` — a module-level global — while the sell branch
correctly reads `self.trend[t]`. It looks harmless only because the two happen
to hold the same list, and it breaks the moment two tickers share a process.
Nothing here reads a global: cash and inventory live on the environment during
training and on `SETMarket` during evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..market import MarketConfig
from .base import Agent, Observation, register
from .env import TrainEnv
from .qfamily import _Encoder


class _PolicyNet(nn.Module):
    def __init__(self, encoder: _Encoder, n_actions: int) -> None:
        super().__init__()
        self.encoder = encoder
        h = encoder.out_dim
        self.head = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, n_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


class _ActorCriticNet(nn.Module):
    def __init__(self, encoder: _Encoder, n_actions: int, *, duel: bool) -> None:
        super().__init__()
        self.encoder = encoder
        h = encoder.out_dim
        self.actor = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, n_actions))
        if duel:
            # Value from a state stream plus a mean-centred advantage stream,
            # mirroring the duelling head the upstream 'duel' notebooks use.
            self.value_stream = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
            self.adv_stream = nn.Sequential(
                nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, n_actions)
            )
            self.critic = None
        else:
            self.value_stream = None
            self.adv_stream = None
            self.critic = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(state)
        logits = self.actor(z)
        if self.critic is not None:
            value = self.critic(z).squeeze(-1)
        else:
            a = self.adv_stream(z)
            value = (
                self.value_stream(z) + (a - a.mean(dim=1, keepdim=True)).mean(dim=1, keepdim=True)
            ).squeeze(-1)
        return logits, value


def _discounted(rewards: list[float], gamma: float) -> torch.Tensor:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    tensor = torch.from_numpy(out)
    if len(tensor) > 1 and float(tensor.std()) > 1e-8:
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    return tensor


@register("policy_gradient")
class PolicyGradientAgent(Agent):
    """REINFORCE with a discounted, standardised return. Upstream notebook 4."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        episodes: int = 25,
        gamma: float = 0.95,
        lr: float = 1e-3,
        recurrent: bool = False,
        entropy_weight: float = 0.01,
        **params,
    ) -> None:
        super().__init__(
            hidden=hidden,
            episodes=episodes,
            gamma=gamma,
            lr=lr,
            recurrent=recurrent,
            entropy_weight=entropy_weight,
            **params,
        )
        self.hidden = int(hidden)
        self.episodes = int(episodes)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.recurrent = bool(recurrent)
        self.entropy_weight = float(entropy_weight)
        self.policy: _PolicyNet | None = None

    def reset(self) -> None:
        pass

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        timestep = self.params.get("timestep", 20)
        env = TrainEnv(
            bars["close"].to_numpy(),
            features,
            timestep=timestep,
            cost_per_turn=config.fees.round_trip_rate,
            initial_cash=config.initial_cash,
        )
        encoder = _Encoder(
            timestep=timestep,
            n_features=features.shape[1],
            hidden=self.hidden,
            recurrent=self.recurrent,
        )
        self.policy = _PolicyNet(encoder, 3)
        opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        for _ in range(self.episodes):
            state = env.reset()
            log_probs, entropies, rewards = [], [], []
            done = False
            while not done:
                logits = self.policy(torch.from_numpy(state).float().unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                state, reward, done = env.step(int(action))
                rewards.append(reward)

            if not rewards:
                continue
            returns = _discounted(rewards, self.gamma)
            loss = -(torch.cat(log_probs) * returns).mean()
            loss = loss - self.entropy_weight * torch.cat(entropies).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            opt.step()

    def act(self, obs: Observation) -> int:
        if self.policy is None:
            return 0
        self.policy.eval()
        with torch.no_grad():
            return int(self.policy(torch.from_numpy(obs.state()).float().unsqueeze(0)).argmax())


@register("actor_critic")
class ActorCriticAgent(Agent):
    """Advantage actor-critic. Upstream notebooks 14-17 via {duel, recurrent}."""

    def __init__(
        self,
        *,
        duel: bool = False,
        recurrent: bool = False,
        hidden: int = 64,
        episodes: int = 25,
        gamma: float = 0.95,
        lr: float = 1e-3,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        **params,
    ) -> None:
        super().__init__(
            duel=duel,
            recurrent=recurrent,
            hidden=hidden,
            episodes=episodes,
            gamma=gamma,
            lr=lr,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
            **params,
        )
        self.duel = bool(duel)
        self.recurrent = bool(recurrent)
        self.hidden = int(hidden)
        self.episodes = int(episodes)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.value_weight = float(value_weight)
        self.entropy_weight = float(entropy_weight)
        self.net: _ActorCriticNet | None = None

    def reset(self) -> None:
        pass

    def fit(self, bars: pd.DataFrame, features: np.ndarray, config: MarketConfig) -> None:
        timestep = self.params.get("timestep", 20)
        env = TrainEnv(
            bars["close"].to_numpy(),
            features,
            timestep=timestep,
            cost_per_turn=config.fees.round_trip_rate,
            initial_cash=config.initial_cash,
        )
        encoder = _Encoder(
            timestep=timestep,
            n_features=features.shape[1],
            hidden=self.hidden,
            recurrent=self.recurrent,
        )
        self.net = _ActorCriticNet(encoder, 3, duel=self.duel)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for _ in range(self.episodes):
            state = env.reset()
            log_probs, entropies, values, rewards = [], [], [], []
            done = False
            while not done:
                logits, value = self.net(torch.from_numpy(state).float().unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                values.append(value)
                state, reward, done = env.step(int(action))
                rewards.append(reward)

            if not rewards:
                continue
            returns = _discounted(rewards, self.gamma)
            value_tensor = torch.cat(values).squeeze(-1)
            advantage = returns - value_tensor.detach()

            actor_loss = -(torch.cat(log_probs) * advantage).mean()
            critic_loss = nn.functional.mse_loss(value_tensor, returns)
            entropy = torch.cat(entropies).mean()
            loss = actor_loss + self.value_weight * critic_loss - self.entropy_weight * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()

    def act(self, obs: Observation) -> int:
        if self.net is None:
            return 0
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs.state()).float().unsqueeze(0))
            return int(logits.argmax())
