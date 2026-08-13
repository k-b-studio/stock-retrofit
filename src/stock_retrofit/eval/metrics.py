"""Metrics that can tell a model from a lag (spec R9).

Upstream reported `1 - sqrt(mean(((real-predict)/real)**2))` on **price levels**.
On a near-random-walk series, "tomorrow equals today" scores in the high
nineties on that measure, so it cannot distinguish skill from persistence. It is
not implemented here, deliberately — see `upstream_accuracy_do_not_use` at the
bottom for why, kept as documentation only.

Everything below scores **returns**:

* ``directional_accuracy`` — share of non-zero calls whose sign is right.
* ``mase`` — MAE(model) / MAE(naive), where the naive forecast is a zero return
  (equivalently, "tomorrow's price equals today's"). **< 1.0 means the model
  beats the naive lag**; >= 1.0 means it does not. This is a relative-MAE form
  of MASE with the denominator taken on the same out-of-sample block as the
  numerator, so both are measured on identical data.
* ``rmse`` — on returns, not levels.
* ``sharpe_after_costs`` — annualised Sharpe of the simplest strategy the
  forecast implies, charged real round-trip costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class MetricSet:
    n: int
    directional_accuracy: float
    coverage: float
    mase: float
    rmse: float
    mae: float
    sharpe_after_costs: float
    sharpe_frictionless: float
    turnover: float
    hit_rate_long: float

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def directional_accuracy(y_true, y_pred) -> tuple[float, float]:
    """(accuracy over calls actually made, share of rows where a call was made).

    A forecast of exactly zero is *not* a directional call. That matters: the
    naive baseline predicts zero everywhere, so reporting it as "0% directional
    accuracy" would be wrong — it makes no calls at all, and its coverage is 0.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan"), 0.0
    called = np.sign(y_pred) != 0
    coverage = float(called.mean())
    if not called.any():
        return float("nan"), coverage
    correct = np.sign(y_pred[called]) == np.sign(y_true[called])
    return float(correct.mean()), coverage


def mase(y_true, y_pred) -> float:
    """MAE(model) / MAE(naive-lag). < 1.0 beats the naive lag."""
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    naive_mae = float(np.mean(np.abs(y_true)))  # naive predicts a zero return
    if naive_mae < 1e-15:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def strategy_returns(
    y_true,
    y_pred,
    *,
    cost_per_turn: float = 0.0,
    allow_short: bool = False,
    threshold: float = 0.0,
) -> np.ndarray:
    """Daily P&L of the simplest strategy the forecast implies.

    Long when the forecast exceeds `threshold`; flat otherwise (short instead of
    flat when `allow_short`). Costs are charged on the change in position, so a
    forecast that flips daily pays for it — which is the whole point of costing
    a signal rather than scoring it in the abstract.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return np.array([])
    if allow_short:
        position = np.where(y_pred > threshold, 1.0, np.where(y_pred < -threshold, -1.0, 0.0))
    else:
        position = np.where(y_pred > threshold, 1.0, 0.0)
    traded = np.abs(np.diff(position, prepend=0.0))
    return position * y_true - traded * cost_per_turn


def sharpe(returns, *, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def evaluate(
    y_true,
    y_pred,
    *,
    cost_per_turn: float = 0.0,
    allow_short: bool = False,
) -> MetricSet:
    """The full metric suite for one block of out-of-sample predictions."""
    y_true, y_pred = _clean(y_true, y_pred)
    da, coverage = directional_accuracy(y_true, y_pred)

    net = strategy_returns(y_true, y_pred, cost_per_turn=cost_per_turn, allow_short=allow_short)
    gross = strategy_returns(y_true, y_pred, cost_per_turn=0.0, allow_short=allow_short)

    if allow_short:
        position = np.where(y_pred > 0, 1.0, np.where(y_pred < 0, -1.0, 0.0))
    else:
        position = np.where(y_pred > 0, 1.0, 0.0)
    turnover = float(np.abs(np.diff(position, prepend=0.0)).mean()) if len(position) else 0.0

    long_days = position > 0
    hit_long = float((y_true[long_days] > 0).mean()) if long_days.any() else float("nan")

    return MetricSet(
        n=int(len(y_true)),
        directional_accuracy=da,
        coverage=coverage,
        mase=mase(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        sharpe_after_costs=sharpe(net),
        sharpe_frictionless=sharpe(gross),
        turnover=turnover,
        hit_rate_long=hit_long,
    )


def upstream_accuracy_do_not_use(real, predict) -> float:
    """Upstream's `calculate_accuracy`, reproduced only so the README can cite it.

    ``1 - sqrt(mean(((real - predict) / real) ** 2))`` on price levels. Never
    call this as a headline metric: a naive lag scores in the high nineties on a
    bank share, which is why the upstream results tables look impressive and
    mean nothing. Kept here to make that demonstrable, not to be used.
    """
    real = np.asarray(real, dtype=float)
    predict = np.asarray(predict, dtype=float)
    return float(1 - np.sqrt(np.mean(((real - predict) / real) ** 2)))
