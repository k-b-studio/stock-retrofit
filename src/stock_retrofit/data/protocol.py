"""The canonical price frame and the protocol every price source must satisfy.

One shape, enforced in one place. `test_schema_contract.py` runs the identical
assertion set against every source, so a new source cannot quietly widen it.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: Exactly these columns, in exactly this order (spec R3).
CANONICAL_COLUMNS: list[str] = ["date", "open", "high", "low", "close", "volume", "symbol"]

PRICE_COLUMNS: list[str] = ["open", "high", "low", "close"]


class SchemaViolation(ValueError):
    """A frame does not satisfy the canonical contract."""


def is_session(df: pd.DataFrame) -> np.ndarray:
    """True where the bar is a session the market actually held.

    yfinance pads SET holidays with a bar that repeats the previous close on
    zero volume and zero range. About 5% of bars in each of KBANK, SCB and BAY
    are such padding — it is present on the liquid names as heavily as on thin
    BAY, which is what identifies it as calendar padding rather than an absence
    of trading interest.

    These are not sessions, and treating them as such corrupts two things:

    * **A label.** The return "realised" on a padded bar is zero by
      construction. A row whose target lands on one is not a forecast — it is a
      free win for a zero baseline and a guaranteed loss for anything that calls
      a direction. `build_target` drops those rows.
    * **A fill.** An order cannot execute on a day the market was shut, and
      charging commission for one is a cost the strategy never paid. `SETMarket`
      refuses to trade on them.

    A genuine flat close on real volume is a *session*, and stays: on the SET
    tick grid an unchanged close is ordinary information, not padding.
    """
    volume = df["volume"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    return ~((volume <= 0) & (high <= low))


@runtime_checkable
class PriceSource(Protocol):
    """Fetches daily OHLCV bars for one symbol."""

    name: str

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return the canonical frame for `symbol` over [start, end]."""
        ...


def validate_canonical(df: pd.DataFrame, *, source: str = "?") -> pd.DataFrame:
    """Assert the canonical contract and return the frame unchanged.

    Raises rather than warns: a malformed frame that reaches the cache is the
    failure mode this whole layer exists to prevent.
    """
    if list(df.columns) != CANONICAL_COLUMNS:
        raise SchemaViolation(
            f"[{source}] columns {list(df.columns)} != canonical {CANONICAL_COLUMNS}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise SchemaViolation(f"[{source}] 'date' must be datetime64, got {df['date'].dtype}")
    if getattr(df["date"].dtype, "tz", None) is not None:
        raise SchemaViolation(
            f"[{source}] 'date' must be timezone-naive Asia/Bangkok trading dates"
        )
    for col in PRICE_COLUMNS + ["volume"]:
        if not pd.api.types.is_float_dtype(df[col]):
            raise SchemaViolation(f"[{source}] '{col}' must be float64, got {df[col].dtype}")
    if not df["date"].is_monotonic_increasing:
        raise SchemaViolation(f"[{source}] 'date' must be sorted ascending")
    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].tolist()[:5]
        raise SchemaViolation(f"[{source}] duplicate dates: {dupes}")
    if df["symbol"].nunique(dropna=False) > 1:
        raise SchemaViolation(f"[{source}] frame mixes symbols: {df['symbol'].unique().tolist()}")
    return df


def empty_canonical() -> pd.DataFrame:
    """A zero-row frame with the canonical dtypes — the correct 'no data' value."""
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "symbol": pd.Series([], dtype="object"),
        }
    )


def coerce_canonical(df: pd.DataFrame, symbol: str, *, source: str = "?") -> pd.DataFrame:
    """Best-effort normalisation into the canonical frame, then validate."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "date" not in out.columns:
        out = out.reset_index()
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    dates = pd.to_datetime(out["date"], errors="coerce")
    if getattr(dates.dtype, "tz", None) is not None:
        # Bars are stamped in exchange time; the trading *date* is what we key on.
        dates = dates.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    out["date"] = dates.dt.normalize()
    out["symbol"] = symbol
    for col in PRICE_COLUMNS + ["volume"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").astype("float64")
    out = out[CANONICAL_COLUMNS]
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return validate_canonical(out, source=source)
