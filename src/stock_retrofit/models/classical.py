"""Non-neural models: ARIMA, gradient boosting, linear baselines.

These keep their natural libraries per spec R16 — `statsmodels` for ARIMA,
`xgboost` for boosting, `scikit-learn` for the linear and forest models. They
exist both as models in their own right and as components of the two stacking
ensembles.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..eval.preprocessing import FoldArrays
from .base import ForecastModel, register


@register("arima")
class ArimaForecaster(ForecastModel):
    """ARIMA on the return series, rolling one step ahead.

    The order is fixed by config rather than searched, because a per-fold search
    is slow and the search itself would need to be confined to the training
    block to stay honest.

    **On conditioning:** the model's *parameters* are estimated on the training
    block only (inside the leakage guard). Predicting the test block then walks
    forward one day at a time, appending each realised return **after** it has
    been predicted. That is the same information set the window models get — at
    bar `t` everything up to `t` is known — and it is not leakage. Forecasting
    60 days ahead from the fold boundary instead would be a different and much
    worse task than the one every other model here is being scored on.
    """

    def __init__(self, *, p: int = 2, d: int = 0, q: int = 2, trend: str = "c", **params) -> None:
        super().__init__(p=p, d=d, q=q, trend=trend, **params)
        self.order = (int(p), int(d), int(q))
        self.trend = trend
        self._fitted = None
        self._train_tail: np.ndarray | None = None

    def reset(self) -> None:
        self._fitted = None
        self._train_tail = None

    def fit(self, fold: FoldArrays) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        y = fold.unscale(fold.y_train).astype(float)
        self._train_tail = y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=self.order, trend=self.trend)
            self._fitted = model.fit()

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("arima: predict called before fit")
        preds = np.zeros(len(fold.y_test), dtype=float)
        state = self._fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, actual in enumerate(fold.y_test):
                preds[i] = float(np.asarray(state.forecast(steps=1)).ravel()[0])
                # Append the realised value only AFTER predicting it.
                state = state.append([float(actual)], refit=False)
        return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


@register("xgboost")
class XGBoostForecaster(ForecastModel):
    """Gradient boosting on the flattened feature window."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            **params,
        )
        self._model = None

    def reset(self) -> None:
        self._model = None

    def fit(self, fold: FoldArrays) -> None:
        from xgboost import XGBRegressor

        self._model = XGBRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            reg_lambda=self.params["reg_lambda"],
            objective="reg:squarederror",
            n_jobs=2,
            random_state=0,
            verbosity=0,
        )
        self._model.fit(fold.flat_train(), fold.y_train)

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("xgboost: predict called before fit")
        return fold.unscale(self._model.predict(fold.flat_test()))


@register("linear")
class LinearForecaster(ForecastModel):
    """Ridge on the flattened window — the cheapest learned model in the set."""

    def __init__(self, *, alpha: float = 1.0, **params) -> None:
        super().__init__(alpha=alpha, **params)
        self.alpha = float(alpha)
        self._model = None

    def reset(self) -> None:
        self._model = None

    def fit(self, fold: FoldArrays) -> None:
        from sklearn.linear_model import Ridge

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(fold.flat_train(), fold.y_train)

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("linear: predict called before fit")
        return fold.unscale(self._model.predict(fold.flat_test()))


@register("random_forest")
class RandomForestForecaster(ForecastModel):
    """Random forest on the flattened window — a component of the classical stack."""

    def __init__(self, *, n_estimators: int = 200, max_depth: int | None = 8, **params) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, **params)
        self._model = None

    def reset(self) -> None:
        self._model = None

    def fit(self, fold: FoldArrays) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            n_jobs=2,
            random_state=0,
        )
        self._model.fit(fold.flat_train(), fold.y_train)

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("random_forest: predict called before fit")
        return fold.unscale(self._model.predict(fold.flat_test()))
