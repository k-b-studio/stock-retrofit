"""YAML configuration loading.

Configs are plain YAML and experiment tracking is plain JSON manifests, per the
spec's stated assumptions. There is no experiment-tracking service to stand up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import AGENT_CONFIG_DIR, CONFIG_DIR, MODEL_CONFIG_DIR


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


@dataclass
class EvalConfig:
    train_window: int = 750
    test_window: int = 60
    step: int = 60
    expanding: bool = False
    max_folds: int | None = 8
    timestep: int = 20
    seed: int = 42
    features: list[str] = field(
        default_factory=lambda: ["log_return", "range_pct", "close_ma_ratio", "volume_z", "rsi"]
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> EvalConfig:
        data = load_yaml(path or CONFIG_DIR / "eval.yaml")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def splitter(self):
        from .eval.splits import WalkForward

        return WalkForward(
            train_window=self.train_window,
            test_window=self.test_window,
            step=self.step,
            expanding=self.expanding,
            max_folds=self.max_folds,
        )

    def window(self):
        from .eval.preprocessing import WindowSpec

        return WindowSpec(timestep=self.timestep)


@dataclass
class MarketConfigSpec:
    initial_cash: float = 1_000_000.0
    board_lot: int = 100
    commission_rate: float = 0.00157
    vat_rate: float = 0.07
    minimum_commission: float = 0.0
    price_limit: float = 0.30
    allow_short: bool = False
    participation_cap: float | None = None
    slippage_ticks: float = 0.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> MarketConfigSpec:
        data = load_yaml(path or CONFIG_DIR / "market.yaml")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def build(self, *, symbol: str | None = None):
        """Materialise a `MarketConfig`, applying the symbol's participation cap.

        BAY carries a cap by default because its float is thin (registry R15);
        an explicit cap in the config overrides the registry.
        """
        from .data.corporate_actions import participation_cap_for
        from .market import FeeSchedule, MarketConfig

        cap = self.participation_cap
        if cap is None and symbol:
            cap = participation_cap_for(symbol)

        return MarketConfig(
            initial_cash=self.initial_cash,
            board_lot=self.board_lot,
            fees=FeeSchedule(
                commission_rate=self.commission_rate,
                vat_rate=self.vat_rate,
                minimum_commission=self.minimum_commission,
            ),
            price_limit=self.price_limit,
            allow_short=self.allow_short,
            participation_cap=cap,
            slippage_ticks=self.slippage_ticks,
        )

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.commission_rate * (1 + self.vat_rate)


@dataclass
class DataConfig:
    symbols: list[str] = field(default_factory=lambda: ["KBANK", "SCB", "BAY"])
    source: str = "yfinance"
    start: str = "2000-01-01"
    scb_history: str = "truncate_at_break"
    repair_policy: str = "widen_bar_to_close"

    @classmethod
    def load(cls, path: str | Path | None = None) -> DataConfig:
        data = load_yaml(path or CONFIG_DIR / "data.yaml")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelSpec:
    name: str
    kind: str
    upstream: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> ModelSpec:
        data = load_yaml(path)
        return cls(
            name=data.get("name") or Path(path).stem,
            kind=data["kind"],
            upstream=data.get("upstream", ""),
            params=data.get("params", {}) or {},
        )

    def build(self):
        from .models import build

        return build(self.kind, name=self.name, upstream=self.upstream, **self.params)


@dataclass
class AgentSpec:
    name: str
    kind: str
    upstream: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> AgentSpec:
        data = load_yaml(path)
        return cls(
            name=data.get("name") or Path(path).stem,
            kind=data["kind"],
            upstream=data.get("upstream", ""),
            params=data.get("params", {}) or {},
        )

    def build(self):
        from .agents import build

        return build(self.kind, name=self.name, upstream=self.upstream, **self.params)


def all_model_specs(directory: Path | None = None) -> list[ModelSpec]:
    directory = directory or MODEL_CONFIG_DIR
    return [ModelSpec.load(p) for p in sorted(directory.glob("*.yaml"))]


def all_agent_specs(directory: Path | None = None) -> list[AgentSpec]:
    directory = directory or AGENT_CONFIG_DIR
    return [AgentSpec.load(p) for p in sorted(directory.glob("*.yaml"))]
