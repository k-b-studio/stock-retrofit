"""Quality gates. Fail loud on structural violations, report on the rest.

Two tiers, deliberately:

* **Structural violations raise.** `high < low`, a close outside `[low, high]`,
  duplicate dates, a non-positive price, or an *unexplained* single-day move
  beyond SET's +/-30% ceiling/floor. None of these are possible in correct data,
  so encountering one means the data is wrong, and silent bad data is the
  failure mode this layer exists to prevent (spec R12).
* **Advisories are reported.** Missing sessions, zero-volume days, null runs.
  These are real facts about a thin market, not corruption.

The +/-30% check consults the source's split record and the corporate-action
registry before raising: a jump explained by a recorded action is explained, not
a violation (R4 asks for zero *unexplained* gaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .corporate_actions import breaks_for
from .sources import CorporateActionRecord

#: SET ceiling/floor band on the previous close.
DAILY_LIMIT = 0.30
#: Tolerance so a move of exactly 30.0% does not trip on float noise.
LIMIT_TOLERANCE = 0.005


@dataclass
class QualityReport:
    symbol: str
    rows: int
    start: str | None
    end: str | None
    violations: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def render(self) -> str:
        head = f"Data quality — {self.symbol}: {self.rows} bars, {self.start} .. {self.end}"
        lines = [head, "=" * len(head)]
        if self.stats:
            for k, v in self.stats.items():
                lines.append(f"  {k:.<34} {v}")
        if self.violations:
            lines.append("  STRUCTURAL VIOLATIONS (fatal):")
            lines += [f"    ! {v}" for v in self.violations]
        else:
            lines.append("  structural violations................ none")
        if self.advisories:
            lines.append("  advisories:")
            lines += [f"    - {a}" for a in self.advisories]
        return "\n".join(lines)


class DataQualityError(ValueError):
    """A structural violation was found in cached market data."""


def trading_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Derive SET trading days as the union of dates observed across symbols.

    The spec's stated fallback (assumption list): swap in an authoritative
    holiday calendar if this proves too noisy. With three large-cap banks the
    union is a good proxy — a day on which none of them traded was a holiday.
    """
    if not frames:
        return pd.DatetimeIndex([])
    union = pd.DatetimeIndex([])
    for df in frames.values():
        union = union.union(pd.DatetimeIndex(df["date"]))
    return union.sort_values()


def check(
    df: pd.DataFrame,
    symbol: str,
    *,
    actions: CorporateActionRecord | None = None,
    calendar: pd.DatetimeIndex | None = None,
    repairs: dict | None = None,
    raise_on_violation: bool = True,
) -> QualityReport:
    """Run every gate over one symbol's canonical frame."""
    report = QualityReport(
        symbol=symbol.upper(),
        rows=int(len(df)),
        start=str(df["date"].min().date()) if len(df) else None,
        end=str(df["date"].max().date()) if len(df) else None,
    )
    if not len(df):
        report.violations.append("frame is empty")
        if raise_on_violation:
            raise DataQualityError(f"{symbol}: no rows")
        return report

    # --- structural: bar internal consistency -------------------------------
    bad_hl = df.loc[df["high"] < df["low"]]
    if len(bad_hl):
        report.violations.append(
            f"{len(bad_hl)} bars with high < low, first {bad_hl['date'].iloc[0].date()}"
        )

    for col in ("close", "open"):
        outside = df.loc[(df[col] < df["low"]) | (df[col] > df["high"])]
        if len(outside):
            report.violations.append(
                f"{len(outside)} bars with {col} outside [low, high], "
                f"first {outside['date'].iloc[0].date()}"
            )

    nonpositive = df.loc[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if len(nonpositive):
        report.violations.append(
            f"{len(nonpositive)} bars with a non-positive price, "
            f"first {nonpositive['date'].iloc[0].date()}"
        )

    if df["date"].duplicated().any():
        report.violations.append(f"{int(df['date'].duplicated().sum())} duplicate dates")

    if not df["date"].is_monotonic_increasing:
        report.violations.append("dates are not sorted ascending")

    # --- structural: moves beyond the ceiling/floor band ---------------------
    excessive = _excessive_moves(df, symbol, actions)
    if len(excessive):
        shown = ", ".join(
            f"{r.date.date()} {r.pct_change:+.1%}" for r in excessive.head(5).itertuples()
        )
        report.violations.append(
            f"{len(excessive)} unexplained single-day moves beyond "
            f"+/-{DAILY_LIMIT:.0%} (impossible under SET limits): {shown}"
        )

    # --- advisories ----------------------------------------------------------
    nulls = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
    if nulls:
        report.advisories.append(f"{nulls} null price/volume cells")

    zero_vol = df.loc[df["volume"] <= 0]
    if len(zero_vol):
        report.advisories.append(
            f"{len(zero_vol)} zero-volume sessions "
            f"({len(zero_vol) / len(df):.2%} of bars) — no tradeable liquidity on those days"
        )

    if calendar is not None and len(calendar):
        window = calendar[(calendar >= df["date"].min()) & (calendar <= df["date"].max())]
        missing = window.difference(pd.DatetimeIndex(df["date"]))
        if len(missing):
            report.advisories.append(
                f"{len(missing)} sessions missing vs. the derived SET calendar "
                f"(first {missing[0].date()}, last {missing[-1].date()})"
            )
        report.stats["sessions expected"] = len(window)

    if repairs and repairs.get("count"):
        dates = sorted({r["date"] for r in repairs.get("records", [])})
        report.advisories.append(
            f"{repairs['count']} vendor bar defect(s) repaired under policy "
            f"'{repairs['policy']}' on {len(dates)} date(s): {', '.join(dates[:6])}"
            + (" ..." if len(dates) > 6 else "")
            + " — see the .meta.json sidecar for the full audit trail"
        )

    ret = df["close"].pct_change()
    report.stats["mean daily volume"] = f"{df['volume'].mean():,.0f}"
    report.stats["median daily volume"] = f"{df['volume'].median():,.0f}"
    report.stats["daily return sd"] = f"{ret.std():.4%}"
    report.stats["largest 1-day move"] = f"{ret.abs().max():.2%}"
    report.stats["registered breaks"] = len(breaks_for(symbol)) or "none"

    if report.violations and raise_on_violation:
        raise DataQualityError(f"{symbol}: " + "; ".join(report.violations))
    return report


def _excessive_moves(
    df: pd.DataFrame, symbol: str, actions: CorporateActionRecord | None
) -> pd.DataFrame:
    """Close-to-close moves beyond the band that no recorded action explains.

    A move is *explained* when the interval it spans contains a recorded split
    or a registered structural break. Spanning matters rather than same-day
    matching, because a break is usually accompanied by a trading halt: SCB's
    issuer substitution is dated 2022-04-22 but the price discontinuity lands on
    the first session after it reopened, 2022-04-27.
    """
    moves = pd.DataFrame(
        {
            "prev_date": df["date"].shift(1),
            "date": df["date"],
            "pct_change": df["close"].pct_change(),
        }
    ).dropna()
    flagged = moves.loc[moves["pct_change"].abs() > DAILY_LIMIT + LIMIT_TOLERANCE]
    if not len(flagged):
        return flagged

    event_dates: list[pd.Timestamp] = [pd.Timestamp(b.date) for b in breaks_for(symbol)]
    if actions is not None and len(actions.splits):
        event_dates += list(pd.DatetimeIndex(actions.splits["date"]))
    if not event_dates:
        return flagged

    spans_event = flagged.apply(
        lambda r: any(r["prev_date"] < e <= r["date"] for e in event_dates), axis=1
    )
    return flagged.loc[~spans_event]


def reconcile(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    tick_of: callable = None,
    primary_name: str = "primary",
    secondary_name: str = "secondary",
) -> pd.DataFrame:
    """Per-date close comparison between two sources (spec R10).

    Never averages and never silently prefers one — disagreement is a finding to
    surface. Returns one row per overlapping date with the absolute difference
    and whether it exceeds one tick at that price level.
    """
    merged = primary[["date", "close"]].merge(
        secondary[["date", "close"]], on="date", how="inner", suffixes=("_a", "_b")
    )
    if not len(merged):
        return merged.assign(diff=[], abs_diff=[], tick=[], exceeds_one_tick=[])

    merged["diff"] = merged["close_a"] - merged["close_b"]
    merged["abs_diff"] = merged["diff"].abs()
    if tick_of is None:
        from ..market.rules import tick_size as tick_of
    merged["tick"] = merged["close_a"].map(tick_of)
    # One tick of float noise is not a disagreement; strictly more than one is.
    merged["exceeds_one_tick"] = merged["abs_diff"] > merged["tick"] * 1.0001
    return merged.rename(
        columns={"close_a": f"close_{primary_name}", "close_b": f"close_{secondary_name}"}
    )
