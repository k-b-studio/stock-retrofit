"""Metrics that can tell a model from a lag (spec R9).

Upstream reported `1 - sqrt(mean(((real-predict)/real)**2))` on **price levels**.
On a near-random-walk series, "tomorrow equals today" scores in the high
nineties on that measure, so it cannot distinguish skill from persistence. It is
not implemented here, deliberately — see `upstream_accuracy_do_not_use` at the
bottom for why, kept as documentation only.

Everything below scores **returns**:

* ``ic`` — the out-of-sample correlation between forecast and realised return.
  **This is the skill measure.** It is the only statistic here that answers
  "does this model know anything", and unlike MASE it can say yes.
* ``directional_accuracy`` — share of calls whose sign is right, over days the
  price **actually moved**. See the note in `directional_accuracy` for why the
  denominator is not simply every day.
* ``mase`` — MAE(model) / MAE(naive), where the naive forecast is a zero return
  (equivalently, "tomorrow's price equals today's"). Useful as a *ranking*; not
  useful as a pass/fail test, and deliberately no longer reported as one. Two
  things drain its power. MAE is minimised by the conditional median, which on
  daily equity returns is ≈ 0, so the zero forecast is already near-optimal and
  any forecast that moves off it pays. And these series are zero-inflated —
  13-24% of out-of-sample days close exactly unchanged — where a flat day
  contributes nothing to the denominator and pure error to the numerator.
  Simulated against a synthetic forecaster with a *known* edge on this data, an
  IC of 0.10 — a strong daily signal — crosses MASE 1.00 in 27% of draws on
  KBANK, 34% on SCB and **0%** on BAY. The threshold is close to a coin flip at
  best and unreachable at worst, so a column of MASE >= 1.00 is a fact about the
  metric before it is a fact about the models.
* ``rmse`` — on returns, not levels.
* ``sharpe_after_costs`` — annualised Sharpe of the simplest strategy the
  forecast implies, charged real round-trip costs. Read it against the
  ``always_long`` row, not against zero: on these tickers simply holding the
  share scored +1.62, +0.86 and +1.54, which beat most of the catalogue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class MetricSet:
    n: int
    ic: float
    ic_t: float
    directional_accuracy: float
    coverage: float
    flat_share: float
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
    """(accuracy over calls made on days that moved, share of those days called).

    Two exclusions, for two different reasons.

    **A forecast of exactly zero is not a directional call.** The naive baseline
    predicts zero everywhere, so reporting it as "0% directional accuracy" would
    be wrong — it makes no calls at all, and its coverage is 0.

    **A day the price did not move cannot be called.** ``sign(0)`` matches no
    forecast, so a flat close is a guaranteed miss for every model however good.
    That is not a small correction here: 20% of KBANK's out-of-sample days and
    29% of BAY's close exactly unchanged, because a THB-priced bank share on the
    SET tick grid frequently does. Scoring those days against a 50% coin flip
    put the *attainable* maximum at 80% and 71%, and made a catalogue of models
    sitting slightly above chance read as "worse than chance". They are excluded
    from the denominator instead, which restores 50% as the honest reference —
    and ``flat_share`` reports how many days were set aside, so the exclusion is
    visible rather than silent.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan"), 0.0
    moved = np.sign(y_true) != 0
    called = (np.sign(y_pred) != 0) & moved
    coverage = float(called.sum() / moved.sum()) if moved.any() else 0.0
    if not called.any():
        return float("nan"), coverage
    correct = np.sign(y_pred[called]) == np.sign(y_true[called])
    return float(correct.mean()), coverage


def information_coefficient(y_true, y_pred) -> tuple[float, float]:
    """(correlation of forecast with realised return, its t-statistic).

    The measure that actually detects skill. A daily equity IC of 0.02-0.05 is a
    real, tradeable signal; 0.10 is excellent. All of those are invisible to a
    MASE threshold — which is why this is the column to read.

    The t-statistic is the usual `r * sqrt(n - 2) / sqrt(1 - r^2)`, and it is a
    *per-model* figure. Judge a table of them accordingly: over 22 models one
    or two will clear |t| > 1.96 by chance, and models sharing a feature set do
    not produce independent draws.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 3 or np.std(y_pred) < 1e-15 or np.std(y_true) < 1e-15:
        return float("nan"), float("nan")
    r = float(np.corrcoef(y_pred, y_true)[0, 1])
    if not np.isfinite(r) or abs(r) >= 1.0:
        return r, float("nan")
    return r, float(r * np.sqrt(len(y_true) - 2) / np.sqrt(1 - r**2))


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
    ic, ic_t = information_coefficient(y_true, y_pred)

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
        ic=ic,
        ic_t=ic_t,
        directional_accuracy=da,
        coverage=coverage,
        flat_share=float(np.mean(np.sign(y_true) == 0)) if len(y_true) else float("nan"),
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
