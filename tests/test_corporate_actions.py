"""SCB must never be handed back spanning its 2022-04-22 break unsignalled (R5/R14)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_retrofit.data.corporate_actions import (
    REGISTRY,
    apply_policy,
    breaks_for,
    describe,
    participation_cap_for,
)

BREAK = pd.Timestamp("2022-04-22")


def spanning_frame() -> pd.DataFrame:
    """A frame that straddles the break — the dangerous case."""
    dates = pd.bdate_range("2022-04-01", "2022-05-31")
    n = len(dates)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
            "symbol": "SCB",
        }
    )


def test_registry_records_the_scb_substitution():
    breaks = breaks_for("SCB")
    assert len(breaks) == 1
    assert breaks[0].date == date(2022, 4, 22)
    assert breaks[0].kind == "issuer_substitution"
    assert "SCB X" in breaks[0].description or "SCBX" in breaks[0].description


def test_default_policy_truncates_at_the_break():
    out = apply_policy(spanning_frame(), "SCB")
    assert out["date"].min() >= BREAK


def test_full_policy_keeps_everything_and_flags_the_boundary():
    df = spanning_frame()
    out = apply_policy(df, "SCB", policy="full_with_changepoint")
    assert len(out) == len(df)
    assert "is_changepoint" in out.columns
    assert out["is_changepoint"].sum() == 1
    assert out.loc[out["is_changepoint"], "date"].iloc[0] == BREAK


def test_there_is_no_policy_that_spans_the_break_silently():
    """The two policies are the whole option space, and both signal the break —
    one by removing the pre-break rows, one by marking the boundary."""
    df = spanning_frame()
    truncated = apply_policy(df, "SCB", policy="truncate_at_break")
    flagged = apply_policy(df, "SCB", policy="full_with_changepoint")
    assert (truncated["date"] >= BREAK).all()
    assert flagged["is_changepoint"].any()
    with pytest.raises(ValueError, match="unknown policy"):
        apply_policy(df, "SCB", policy="just_give_me_everything")


def test_symbols_without_a_break_are_untouched():
    df = spanning_frame().assign(symbol="KBANK")
    assert len(apply_policy(df, "KBANK")) == len(df)
    flagged = apply_policy(df, "KBANK", policy="full_with_changepoint")
    assert not flagged["is_changepoint"].any()


def test_bay_carries_a_participation_cap_and_kbank_does_not():
    assert participation_cap_for("BAY") == pytest.approx(0.05)
    assert participation_cap_for("KBANK") is None
    assert "MUFG" in REGISTRY["BAY"].liquidity_note


def test_describe_mentions_the_break_and_its_source():
    text = describe("SCB")
    assert "2022-04-22" in text and "source:" in text


def test_unknown_symbol_degrades_safely():
    assert breaks_for("NOPE") == ()
    assert participation_cap_for("NOPE") is None


@pytest.mark.parametrize("symbol", ["KBANK", "SCB", "BAY"])
def test_cached_scb_never_predates_the_break_under_the_default(symbol):
    """Against the real cache, if it exists. Skipped when nothing is cached."""
    from stock_retrofit.data import load

    try:
        df = load(symbol)
    except FileNotFoundError:
        pytest.skip(f"{symbol} not cached; run `cli fetch` first")
    assert len(df) > 0
    if symbol == "SCB":
        assert df["date"].min() >= BREAK
