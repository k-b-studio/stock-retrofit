"""Stacking ensembles — upstream `stacking/` notebooks.

* `stack-rnn-arima-xgb.ipynb`   -> `stack_rnn_arima_xgb`
* `stack-encoder-ensemble-xgb.ipynb` -> `stack_encoder_ensemble_xgb`

**Where leakage would sneak back in.** A stack has a second fitting step — the
meta-learner — and it consumes *model predictions* rather than raw rows, so the
row-level guard cannot see it. Fit the meta-learner on predictions the base
models made about rows they were trained on, and it learns to trust whichever
base overfits hardest; the stack then looks excellent in training and falls over
out-of-sample.

So the meta-learner here is fit strictly on **out-of-fold predictions generated
by an inner time-series split of the training block**. The inner split is
announced to the leakage guard as well, so the fit is visible in the audit trail
rather than invisible to it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..eval.leakage import register_fit
from ..eval.preprocessing import FoldArrays
from .base import ForecastModel, build, register


def _subset(fold: FoldArrays, train_idx: np.ndarray, val_idx: np.ndarray) -> FoldArrays:
    """Carve an inner train/validation pair out of a fold's training block.

    Everything here is training data — the scaling statistics were already fit
    on the outer training block, so no new information crosses any boundary.
    """
    return FoldArrays(
        x_train=fold.x_train[train_idx],
        y_train=fold.y_train[train_idx],
        x_test=fold.x_train[val_idx],
        y_test=fold.unscale(fold.y_train[val_idx]),
        train_positions=fold.train_positions[train_idx],
        test_positions=fold.train_positions[val_idx],
        feature_names=fold.feature_names,
        target_scale=fold.target_scale,
        target_center=fold.target_center,
        test_close=np.ones(len(val_idx)),
        test_dates=fold.test_dates.iloc[:0],
    )


def _inner_splits(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-origin inner splits — the same shape as the outer harness."""
    k = max(2, min(k, 5))
    block = n // (k + 1)
    if block < 10:
        cut = int(n * 0.75)
        return [(np.arange(cut), np.arange(cut, n))]
    return [
        (np.arange(0, block * (i + 1)), np.arange(block * (i + 1), block * (i + 2)))
        for i in range(k)
    ]


class _BaseStack(ForecastModel):
    """Shared machinery: fit bases out-of-fold, fit meta, refit bases on all training data."""

    base_specs: list[dict] = []

    def __init__(
        self, *, meta: str = "ridge", meta_alpha: float = 1.0, inner_folds: int = 3, **params
    ) -> None:
        super().__init__(meta=meta, meta_alpha=meta_alpha, inner_folds=inner_folds, **params)
        self.meta_kind = str(meta)
        self.meta_alpha = float(meta_alpha)
        self.inner_folds = int(inner_folds)
        self._bases: list[ForecastModel] = []
        self._meta = None

    def _make_bases(self) -> list[ForecastModel]:
        return [
            build(spec["kind"], name=spec.get("name", spec["kind"]), **spec.get("params", {}))
            for spec in self.base_specs
        ]

    def _make_meta(self):
        if self.meta_kind == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                objective="reg:squarederror",
                n_jobs=2,
                random_state=0,
                verbosity=0,
            )
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.meta_alpha)

    def reset(self) -> None:
        self._bases = []
        self._meta = None

    def transform_fold(self, fold: FoldArrays) -> FoldArrays:
        """Hook for stacks that reshape the inputs (the encoder stack overrides it)."""
        return fold

    def fit(self, fold: FoldArrays) -> None:
        fold = self.transform_fold(fold)
        n = len(fold.x_train)
        splits = _inner_splits(n, self.inner_folds)

        oof_pred: list[np.ndarray] = []
        oof_true: list[np.ndarray] = []
        for inner_train, inner_val in splits:
            # Visible in the audit trail: these are the rows the meta-learner's
            # training signal is derived from, and they are all inside the
            # outer training block by construction.
            register_fit("Stack.inner_fold", fold.train_positions[inner_train])
            sub = _subset(fold, inner_train, inner_val)
            preds = []
            for base in self._make_bases():
                base.reset()
                base.fit(sub)
                preds.append(base.predict(sub))
            oof_pred.append(np.column_stack(preds))
            oof_true.append(sub.y_test)

        meta_x = np.vstack(oof_pred)
        meta_y = np.concatenate(oof_true)
        self._meta = self._make_meta()
        self._meta.fit(np.nan_to_num(meta_x), np.nan_to_num(meta_y))

        # Now refit the bases on the whole training block for use at predict time.
        self._bases = self._make_bases()
        for base in self._bases:
            base.reset()
            base.fit(fold)

    def predict(self, fold: FoldArrays) -> np.ndarray:
        if self._meta is None or not self._bases:
            raise RuntimeError(f"{self.name}: predict called before fit")
        fold = self.transform_fold(fold)
        preds = np.column_stack([b.predict(fold) for b in self._bases])
        return np.asarray(self._meta.predict(np.nan_to_num(preds)), dtype=float).ravel()


@register("stack_rnn_arima_xgb")
class StackRnnArimaXgb(_BaseStack):
    """Upstream `stacking/stack-rnn-arima-xgb.ipynb`.

    An LSTM, an ARIMA and a boosted tree, combined by a ridge meta-learner.
    The three see the same data and fail in different ways, which is the only
    reason stacking them is worth anything.
    """

    base_specs = [
        {
            "kind": "recurrent",
            "name": "lstm",
            "params": {"cell": "lstm", "hidden": 32, "epochs": 25},
        },
        {"kind": "arima", "name": "arima", "params": {"p": 2, "d": 0, "q": 2}},
        {"kind": "xgboost", "name": "xgb", "params": {"n_estimators": 150, "max_depth": 3}},
    ]


class _AutoEncoder(nn.Module):
    def __init__(self, n_inputs: int, code: int) -> None:
        super().__init__()
        hidden = max(code * 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, code)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, n_inputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register("stack_encoder_ensemble_xgb")
class StackEncoderEnsembleXgb(_BaseStack):
    """Upstream `stacking/stack-encoder-ensemble-xgb.ipynb`.

    An autoencoder compresses the flattened window; a ridge / forest / boosted
    ensemble reads the codes; a boosted meta-learner combines them. The
    autoencoder is fit on training rows only, like every other statistic here.
    """

    base_specs = [
        {"kind": "linear", "name": "ridge", "params": {"alpha": 1.0}},
        {"kind": "random_forest", "name": "rf", "params": {"n_estimators": 150, "max_depth": 6}},
        {"kind": "xgboost", "name": "xgb", "params": {"n_estimators": 150, "max_depth": 3}},
    ]

    def __init__(
        self, *, code_size: int = 12, ae_epochs: int = 40, meta: str = "xgboost", **params
    ) -> None:
        super().__init__(meta=meta, code_size=code_size, ae_epochs=ae_epochs, **params)
        self.code_size = int(code_size)
        self.ae_epochs = int(ae_epochs)
        self._ae: _AutoEncoder | None = None

    def reset(self) -> None:
        super().reset()
        self._ae = None

    def _fit_autoencoder(self, flat_train: np.ndarray) -> None:
        torch.manual_seed(0)
        self._ae = _AutoEncoder(flat_train.shape[1], self.code_size)
        opt = torch.optim.Adam(self._ae.parameters(), lr=1e-3)
        x = torch.from_numpy(np.ascontiguousarray(flat_train)).float()
        loss_fn = nn.MSELoss()
        self._ae.train()
        for _ in range(self.ae_epochs):
            order = torch.randperm(len(x))
            for start in range(0, len(x), 64):
                idx = order[start : start + 64]
                opt.zero_grad()
                loss = loss_fn(self._ae(x[idx]), x[idx])
                loss.backward()
                opt.step()
        self._ae.eval()

    def _encode(self, flat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            code = self._ae.encoder(torch.from_numpy(np.ascontiguousarray(flat)).float())
        return code.numpy()

    def transform_fold(self, fold: FoldArrays) -> FoldArrays:
        if self._ae is None:
            register_fit("StackEncoder.autoencoder", fold.train_positions)
            self._fit_autoencoder(fold.flat_train())
        train_code = self._encode(fold.flat_train())[:, None, :]
        test_code = self._encode(fold.flat_test())[:, None, :]
        return FoldArrays(
            x_train=train_code,
            y_train=fold.y_train,
            x_test=test_code,
            y_test=fold.y_test,
            train_positions=fold.train_positions,
            test_positions=fold.test_positions,
            feature_names=[f"code_{i}" for i in range(train_code.shape[-1])],
            target_scale=fold.target_scale,
            target_center=fold.target_center,
            test_close=fold.test_close,
            test_dates=fold.test_dates,
        )
