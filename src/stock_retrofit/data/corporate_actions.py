"""Declarative registry of instrument discontinuities and liquidity caveats.

The point of this module is that the SCB caveat lives in code rather than in
someone's memory. A caller must never receive a series spanning a registered
break without a signal that it did (spec R13-R15, R5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class StructuralBreak:
    """A date on which the instrument behind a ticker changed."""

    symbol: str
    date: date
    kind: str
    description: str
    source: str


@dataclass(frozen=True)
class InstrumentMeta:
    symbol: str
    name: str
    liquidity_note: str
    #: Cap orders at this fraction of the session's volume, or None for no cap.
    default_participation_cap: float | None = None
    breaks: tuple[StructuralBreak, ...] = field(default_factory=tuple)


SCB_RESTRUCTURE = StructuralBreak(
    symbol="SCB",
    date=date(2022, 4, 22),
    kind="issuer_substitution",
    description=(
        "SCB delisted and SCB X PCL listed 1:1 in its place, retaining the SCB ticker. "
        "A change of issuer (bank -> holding company), not merely of name. Pre-2022-04-22 "
        "'SCB' bars belong to a different legal entity."
    ),
    source="SCB/SCBX first-party announcements, March-April 2022",
)


REGISTRY: dict[str, InstrumentMeta] = {
    "SCB": InstrumentMeta(
        symbol="SCB",
        name="SCB X PCL (formerly The Siam Commercial Bank PCL)",
        liquidity_note="Large cap, liquid. Series carries an issuer substitution at 2022-04-22.",
        default_participation_cap=None,
        breaks=(SCB_RESTRUCTURE,),
    ),
    "KBANK": InstrumentMeta(
        symbol="KBANK",
        name="Kasikornbank PCL",
        liquidity_note="Large cap, liquid, no known discontinuity. The clean case.",
        default_participation_cap=None,
    ),
    "BAY": InstrumentMeta(
        symbol="BAY",
        name="Bank of Ayudhya PCL (Krungsri)",
        liquidity_note=(
            "Thin float: ~72-76% held by MUFG since the 2013 acquisition. Daily turnover is "
            "small relative to SCB/KBANK. Treat as the liquidity stress case and cap "
            "participation in any backtest."
        ),
        default_participation_cap=0.05,
    ),
}

VALID_POLICIES = ("truncate_at_break", "full_with_changepoint")


def meta_for(symbol: str) -> InstrumentMeta:
    key = symbol.upper()
    if key not in REGISTRY:
        return InstrumentMeta(
            symbol=key, name=key, liquidity_note="No registry entry; caveats unknown."
        )
    return REGISTRY[key]


def breaks_for(symbol: str) -> tuple[StructuralBreak, ...]:
    return meta_for(symbol).breaks


def participation_cap_for(symbol: str) -> float | None:
    return meta_for(symbol).default_participation_cap


def apply_policy(df: pd.DataFrame, symbol: str, policy: str = "truncate_at_break") -> pd.DataFrame:
    """Apply the corporate-action policy to a canonical frame.

    `truncate_at_break` (default) keeps only bars at or after the latest break.
    `full_with_changepoint` keeps everything and adds an `is_changepoint`
    boolean column, true on exactly the break dates — so a consumer can exclude
    the boundary from evaluation windows.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {VALID_POLICIES}")

    events = breaks_for(symbol)
    out = df.copy()

    if not events:
        if policy == "full_with_changepoint":
            out["is_changepoint"] = False
        return out.reset_index(drop=True)

    if policy == "truncate_at_break":
        cutoff = pd.Timestamp(max(b.date for b in events))
        return out.loc[out["date"] >= cutoff].reset_index(drop=True)

    break_dates = {pd.Timestamp(b.date) for b in events}
    out["is_changepoint"] = out["date"].isin(break_dates)
    return out.reset_index(drop=True)


def describe(symbol: str) -> str:
    m = meta_for(symbol)
    lines = [f"{m.symbol} — {m.name}", f"  liquidity: {m.liquidity_note}"]
    if m.default_participation_cap is not None:
        lines.append(f"  default participation cap: {m.default_participation_cap:.1%} of volume")
    for b in m.breaks:
        lines.append(f"  break {b.date} [{b.kind}]: {b.description}")
        lines.append(f"    source: {b.source}")
    return "\n".join(lines)
