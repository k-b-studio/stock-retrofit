"""Price sources.

`YFinanceSource` is **primary** in this build. That is a demotion of the spec's
intent (R1 wanted Settrade primary) forced by an environment fact, not a
preference: Settrade Open API is credential-gated behind a broker relationship
and no credentials exist here. See `docs/settrade-api-notes.md` for what was and
was not verifiable, and the README for the consequences.

`SettradeSource` is written against the SDK's documented surface but has never
been executed against the live API. It is wired in and selectable; it raises a
precise, actionable error when credentials or the SDK are absent rather than
silently degrading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .protocol import coerce_canonical, empty_canonical

#: SET symbol -> Yahoo Finance ticker (spec R2).
YAHOO_SUFFIX_MAP: dict[str, str] = {"SCB": "SCB.BK", "KBANK": "KBANK.BK", "BAY": "BAY.BK"}


def yahoo_ticker(symbol: str) -> str:
    return YAHOO_SUFFIX_MAP.get(symbol.upper(), f"{symbol.upper()}.BK")


@dataclass
class CorporateActionRecord:
    """Dividends and splits as reported by the source, kept beside the bars.

    Not part of the canonical frame — it exists so the quality gate can tell an
    explained price jump (a par split) from a structural violation.
    """

    dividends: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["date", "amount"])
    )
    splits: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "ratio"]))

    def to_json(self) -> dict:
        return {
            "dividends": [
                {"date": str(pd.Timestamp(r.date).date()), "amount": float(r.amount)}
                for r in self.dividends.itertuples()
            ],
            "splits": [
                {"date": str(pd.Timestamp(r.date).date()), "ratio": float(r.ratio)}
                for r in self.splits.itertuples()
            ],
        }

    @classmethod
    def from_json(cls, payload: dict | None) -> CorporateActionRecord:
        payload = payload or {}
        div = pd.DataFrame(payload.get("dividends", []) or [], columns=["date", "amount"])
        spl = pd.DataFrame(payload.get("splits", []) or [], columns=["date", "ratio"])
        for frame in (div, spl):
            if len(frame):
                frame["date"] = pd.to_datetime(frame["date"])
        return cls(dividends=div, splits=spl)


class YFinanceSource:
    """Yahoo Finance daily bars via `yfinance`.

    Prices are fetched **unadjusted** (`auto_adjust=False`). A backtest fills at
    the price that was actually quoted, so adjusted series are the wrong input;
    the dividend/split record is carried separately for the quality gate.
    """

    name = "yfinance"

    def __init__(self, *, auto_adjust: bool = False) -> None:
        self.auto_adjust = auto_adjust
        self._last_actions: CorporateActionRecord = CorporateActionRecord()

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(yahoo_ticker(symbol)).history(
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            # yfinance's `end` is exclusive; add a day so `end` itself is included.
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=self.auto_adjust,
            actions=True,
        )
        if raw is None or raw.empty:
            self._last_actions = CorporateActionRecord()
            return empty_canonical()

        raw = raw.reset_index().rename(columns={"Datetime": "Date"})
        self._last_actions = self._extract_actions(raw)
        return coerce_canonical(
            raw[["Date", "Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"Date": "date"}
            ),
            symbol,
            source=self.name,
        )

    @staticmethod
    def _extract_actions(raw: pd.DataFrame) -> CorporateActionRecord:
        dates = pd.to_datetime(raw["Date"])
        if getattr(dates.dtype, "tz", None) is not None:
            dates = dates.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
        dates = dates.dt.normalize()

        div = pd.DataFrame(columns=["date", "amount"])
        if "Dividends" in raw:
            mask = pd.to_numeric(raw["Dividends"], errors="coerce").fillna(0.0) > 0
            div = pd.DataFrame(
                {"date": dates[mask], "amount": raw.loc[mask, "Dividends"].astype(float)}
            )

        spl = pd.DataFrame(columns=["date", "ratio"])
        if "Stock Splits" in raw:
            mask = pd.to_numeric(raw["Stock Splits"], errors="coerce").fillna(0.0) > 0
            spl = pd.DataFrame(
                {"date": dates[mask], "ratio": raw.loc[mask, "Stock Splits"].astype(float)}
            )
        return CorporateActionRecord(
            dividends=div.reset_index(drop=True), splits=spl.reset_index(drop=True)
        )

    def last_actions(self) -> CorporateActionRecord:
        return self._last_actions


class SettradeCredentialsMissing(RuntimeError):
    """Raised when SettradeSource is selected without usable credentials."""


class SettradeSource:
    """Settrade Open API daily bars via the `settrade-v2` SDK.

    **Never executed against the live API in this build.** Credentials are
    broker-issued and none exist in this environment, so the request shape below
    follows the published SDK reference and should be treated as unverified.
    The four R2 questions it depends on are recorded as open in
    `docs/settrade-api-notes.md`.
    """

    name = "settrade"

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        broker_id: str | None = None,
        environment: str | None = None,
    ) -> None:
        self.app_id = app_id or os.getenv("SETTRADE_APP_ID") or ""
        self.app_secret = app_secret or os.getenv("SETTRADE_APP_SECRET") or ""
        self.broker_id = broker_id or os.getenv("SETTRADE_BROKER_ID") or ""
        self.environment = environment or os.getenv("SETTRADE_ENVIRONMENT") or "sandbox"

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.app_secret and self.broker_id)

    def _client(self):
        if not self.configured:
            raise SettradeCredentialsMissing(
                "Settrade needs SETTRADE_APP_ID / SETTRADE_APP_SECRET / SETTRADE_BROKER_ID "
                "in .env (copy .env.example). None are set. yfinance is primary in this "
                "build — see docs/settrade-api-notes.md."
            )
        try:
            from settrade_v2 import Investor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise SettradeCredentialsMissing(
                "settrade-v2 is not installed. `pip install -e '.[settrade]'`. "
                "yfinance is primary in this build."
            ) from exc
        return Investor(
            app_id=self.app_id,
            app_secret=self.app_secret,
            broker_id=self.broker_id,
            app_code="ALGO",
            is_auto_queue=False,
        )

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        market = self._client().MarketData()
        payload = market.get_candlestick(
            symbol=symbol.upper(),
            interval="1d",
            start=pd.Timestamp(start).strftime("%d/%m/%Y"),
            end=pd.Timestamp(end).strftime("%d/%m/%Y"),
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            return empty_canonical()
        # SDK returns epoch seconds under `time`, OHLCV under single letters.
        frame = frame.rename(
            columns={
                "time": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        if pd.api.types.is_numeric_dtype(frame["date"]):
            frame["date"] = pd.to_datetime(frame["date"], unit="s", utc=True).dt.tz_convert(
                "Asia/Bangkok"
            )
        return coerce_canonical(frame, symbol, source=self.name)

    def last_actions(self) -> CorporateActionRecord:
        return CorporateActionRecord()


def get_source(name: str) -> YFinanceSource | SettradeSource:
    sources = {"yfinance": YFinanceSource, "settrade": SettradeSource}
    if name not in sources:
        raise KeyError(f"unknown source {name!r}; available: {sorted(sources)}")
    return sources[name]()
