"""Fold-local feature construction and scaling (spec R7).

Every statistic here is fit on training rows only and then applied to the test
rows, and every fit announces itself to `leakage.register_fit`. Features are
strictly causal: the row at position `t` uses bars up to and including `t`, and
its label is the return realised at `t+1`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .leakage import register_fit

#: Feature blocks available to any model. Kept small and causal on purpose.
DEFAULT_FEATURES = ("log_return", "range_pct", "close_ma_ratio", "volume_z", "rsi")

#: Longest rolling lookback used below. Rows before this are warm-up, not data.
WARMUP = 60


def build_features(df: pd.DataFrame, features=DEFAULT_FEATURES) -> pd.DataFrame:
    """Causal feature frame aligned to `df`'s index. No forward-looking windows.

    Degenerate bars are given their correct value rather than left as NaN. On
    SET this matters more than it would on a US large cap: 3-5% of sessions in
    these three names print zero volume or zero range, and treating those as
    *missing* would discard roughly half of all 20-day windows. A zero-range day
    is information (the close sat at both extremes, so `close_loc` is 0.5), not
    an absence of it.

    The only rows left NaN are the first `WARMUP`, where the rolling windows
    genuinely have nothing to average.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    logret = np.log(close).diff()

    if "log_return" in features:
        out["log_return"] = logret
        out["log_return_5"] = logret.rolling(5).mean()
        out["log_return_vol_20"] = logret.rolling(20).std()
    if "range_pct" in features:
        out["range_pct"] = (high - low) / close
        span = high - low
        # No range: the close is trivially at both the high and the low -> mid.
        out["close_loc"] = ((close - low) / span.where(span > 0)).clip(0, 1).fillna(0.5)
    if "close_ma_ratio" in features:
        for w in (5, 20, 60):
            out[f"close_ma_{w}"] = close / close.rolling(w).mean() - 1.0
    if "volume_z" in features:
        vol_ma = volume.rolling(20).mean()
        vol_sd = volume.rolling(20).std()
        # Flat volume over the window -> no deviation to speak of.
        z = (volume - vol_ma) / vol_sd.where(vol_sd > 0)
        out["volume_z"] = z.clip(-5, 5).where(vol_sd.notna(), np.nan).fillna(0.0)
        out.loc[out.index[:20], "volume_z"] = np.nan  # keep the warm-up honest
    if "rsi" in features:
        gain = logret.clip(lower=0).rolling(14).mean()
        loss = (-logret.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.where(loss > 0)
        rsi = 100 - 100 / (1 + rs)
        # No down moves in the window -> RSI 100; no moves at all -> RSI 50.
        rsi = rsi.where(loss > 0, np.where(gain > 0, 100.0, 50.0))
        out["rsi"] = rsi / 100.0 - 0.5
        out.loc[out.index[:14], "rsi"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)

    tail = out.iloc[WARMUP:]
    if tail.isna().any().any():
        offenders = tail.columns[tail.isna().any()].tolist()
        raise ValueError(
            f"features {offenders} are NaN after the {WARMUP}-row warm-up. "
            "Every degenerate case must be given an explicit value in build_features "
            "— silently dropping those rows would discard most of the series."
        )
    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    """Next-day simple return — the thing every forecaster in this repo predicts.

    Returns, not price levels. Upstream scored price levels, which is why a lag
    scored in the high nineties there. The last row's target is NaN by
    construction and is dropped by the windower.
    """
    close = df["close"].astype(float)
    return (close.shift(-1) / close - 1.0).rename("target")


@dataclass
class WindowSpec:
    """How many trailing bars each sample sees."""

    timestep: int = 20

    def __post_init__(self) -> None:
        if self.timestep < 1:
            raise ValueError("timestep must be >= 1")


@dataclass
class FoldArrays:
    """Everything a model needs for one fold, already split and scaled."""

    x_train: np.ndarray  # (n_train, timestep, n_features)
    y_train: np.ndarray  # (n_train,) scaled target
    x_test: np.ndarray
    y_test: np.ndarray  # (n_test,) RAW target — never scaled, it is the truth
    train_positions: np.ndarray
    test_positions: np.ndarray
    feature_names: list[str]
    target_scale: float
    target_center: float
    #: Close price on the last bar of each test sample's window (fill reference).
    test_close: np.ndarray
    test_dates: pd.Series

    @property
    def n_features(self) -> int:
        return self.x_train.shape[-1]

    def flat_train(self) -> np.ndarray:
        return self.x_train.reshape(len(self.x_train), -1)

    def flat_test(self) -> np.ndarray:
        return self.x_test.reshape(len(self.x_test), -1)

    def unscale(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled, dtype=float) * self.target_scale + self.target_center


@dataclass
class FoldPreprocessor:
    """Standardises features and the target using training rows only.

    `fit` is the only method that learns anything, and it registers the exact
    rows it learned from with the leakage guard.
    """

    window: WindowSpec = field(default_factory=WindowSpec)
    scale_target: bool = True
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _std: np.ndarray | None = field(default=None, init=False, repr=False)
    _t_center: float = field(default=0.0, init=False, repr=False)
    _t_scale: float = field(default=1.0, init=False, repr=False)

    def fit(
        self, feats: pd.DataFrame, target: pd.Series, positions: np.ndarray
    ) -> FoldPreprocessor:
        register_fit("FoldPreprocessor.features", positions)
        block = feats.iloc[positions].to_numpy(dtype=float)
        self._mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        self._std = np.where(std < 1e-12, 1.0, std)

        if self.scale_target:
            register_fit("FoldPreprocessor.target", positions)
            t = target.iloc[positions].to_numpy(dtype=float)
            self._t_center = float(np.nanmean(t))
            scale = float(np.nanstd(t))
            self._t_scale = scale if scale > 1e-12 else 1.0
        return self

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("FoldPreprocessor.transform called before fit")
        z = (feats.to_numpy(dtype=float) - self._mean) / self._std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def target_center(self) -> float:
        return self._t_center

    @property
    def target_scale(self) -> float:
        return self._t_scale


def _windows(matrix: np.ndarray, positions: np.ndarray, timestep: int) -> np.ndarray:
    """Stack the `timestep` rows ending at each position."""
    if len(positions) == 0:
        return np.zeros((0, timestep, matrix.shape[1]), dtype=np.float32)
    idx = positions[:, None] - np.arange(timestep - 1, -1, -1)[None, :]
    return matrix[idx].astype(np.float32)


def prepare_fold(
    df: pd.DataFrame,
    fold,
    *,
    window: WindowSpec | None = None,
    features=DEFAULT_FEATURES,
    scale_target: bool = True,
) -> FoldArrays:
    """Turn one fold of a price frame into model-ready arrays, leak-free.

    The train block contributes only positions whose full lookback window and
    whose label both lie inside it, so no training sample can peek across the
    boundary. Test windows may reach back into training bars — that is a
    legitimate use of history that was available at prediction time, not
    leakage, which runs the other way.
    """
    window = window or WindowSpec()
    feats = build_features(df, features)
    target = build_target(df)

    valid = feats.notna().all(axis=1).to_numpy() & target.notna().to_numpy()
    ts = window.timestep

    def usable(lo: int, hi: int, *, allow_history: bool) -> np.ndarray:
        start = lo if allow_history else lo + ts - 1
        pos = np.arange(max(start, ts - 1), hi)
        if len(pos) == 0:
            return pos
        # Every bar in the lookback window must be a usable feature row.
        win_ok = np.array([valid[p - ts + 1 : p + 1].all() for p in pos])
        return pos[win_ok & valid[pos]]

    # `train_end - 1` is excluded on purpose: its label is the return realised on
    # `train_end`, which is the first *test* bar. Including it would leak one day
    # of the test block into training through the target — subtle, and exactly
    # the class of mistake this harness exists to prevent.
    train_pos = usable(fold.train_start, fold.train_end - 1, allow_history=False)
    test_pos = usable(fold.test_start, fold.test_end, allow_history=True)

    if len(train_pos) == 0 or len(test_pos) == 0:
        raise ValueError(
            f"fold {fold.index}: not enough usable rows "
            f"(train={len(train_pos)}, test={len(test_pos)}). "
            "Lower eval.timestep or raise eval.train_window."
        )

    pre = FoldPreprocessor(window=window, scale_target=scale_target).fit(feats, target, train_pos)
    matrix = pre.transform(feats)

    y_train_raw = target.iloc[train_pos].to_numpy(dtype=float)
    y_train = (y_train_raw - pre.target_center) / pre.target_scale

    return FoldArrays(
        x_train=_windows(matrix, train_pos, ts),
        y_train=y_train.astype(np.float32),
        x_test=_windows(matrix, test_pos, ts),
        y_test=target.iloc[test_pos].to_numpy(dtype=float),
        train_positions=train_pos,
        test_positions=test_pos,
        feature_names=list(feats.columns),
        target_scale=pre.target_scale,
        target_center=pre.target_center,
        test_close=df["close"].iloc[test_pos].to_numpy(dtype=float),
        test_dates=df["date"].iloc[test_pos].reset_index(drop=True),
    )
