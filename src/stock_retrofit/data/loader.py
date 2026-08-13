"""The public read path: `load(symbol)` returns cache-backed, policy-applied bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import RAW_DIR
from . import cache
from .corporate_actions import apply_policy
from .quality import QualityReport, check, trading_calendar
from .sources import CorporateActionRecord, get_source

DEFAULT_START = date(2000, 1, 1)


def load(
    symbol: str,
    *,
    policy: str = "truncate_at_break",
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Read cached bars for `symbol`, applying the corporate-action policy.

    Never touches the network. Under the default `truncate_at_break` policy a
    registered break truncates the history; under `full_with_changepoint` the
    frame carries an `is_changepoint` column. There is no third option that
    returns a break-spanning series with no signal.
    """
    df = cache.read_cache(symbol, root)
    df = apply_policy(df, symbol, policy)
    if start is not None:
        df = df.loc[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def fetch(
    symbols: list[str],
    *,
    source: str = "yfinance",
    start: date = DEFAULT_START,
    end: date | None = None,
    force_refresh: bool = False,
    repair_policy: str = "widen_bar_to_close",
    root: Path | None = None,
) -> dict[str, cache.CacheMeta]:
    """Fetch and cache bars for each symbol. The only network path in the package."""
    src = get_source(source)
    end = end or date.today()
    out: dict[str, cache.CacheMeta] = {}
    for symbol in symbols:
        _, meta = cache.fetch_symbol(
            symbol.upper(),
            src,
            start=start,
            end=end,
            force_refresh=force_refresh,
            repair_policy=repair_policy,
            root=root or RAW_DIR,
        )
        out[symbol.upper()] = meta
    return out


def quality_report(
    symbol: str,
    *,
    root: Path | None = None,
    calendar_symbols: list[str] | None = None,
    raise_on_violation: bool = True,
) -> QualityReport:
    """Run the quality gates over one cached symbol."""
    df = cache.read_cache(symbol, root)
    actions: CorporateActionRecord = cache.read_actions(symbol, root)
    meta = cache.read_meta(symbol, root)

    calendar = None
    peers = calendar_symbols or cache.cached_symbols(root)
    if len(peers) > 1:
        frames = {s: cache.read_cache(s, root) for s in peers}
        calendar = trading_calendar(frames)

    return check(
        df,
        symbol,
        actions=actions,
        calendar=calendar,
        repairs=meta.repairs if meta else None,
        raise_on_violation=raise_on_violation,
    )


def reconcile(
    symbol: str,
    *,
    against: str = "yfinance",
    root: Path | None = None,
) -> pd.DataFrame:
    """Compare cached bars against a live pull from `against` (spec R4/R10).

    Surfaces per-date disagreement; never averages, never silently prefers one.
    When the cache was itself built from `against`, this degenerates into a
    self-check that the cache still matches the vendor — which is worth having,
    and is labelled as such by the caller.
    """
    from .quality import reconcile as _reconcile

    cached = cache.read_cache(symbol, root)
    meta = cache.read_meta(symbol, root)
    other = get_source(against).fetch(
        symbol, cached["date"].min().date(), cached["date"].max().date()
    )
    return _reconcile(
        cached,
        other,
        primary_name=(meta.source if meta else "cache"),
        secondary_name=against,
    )
