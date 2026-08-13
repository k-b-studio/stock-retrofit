"""Quality gates must raise on structural violations, not warn (spec R12)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stock_retrofit.data.cache import read_cache, write_cache
from stock_retrofit.data.protocol import (
    CANONICAL_COLUMNS,
    SchemaViolation,
    coerce_canonical,
    validate_canonical,
)
from stock_retrofit.data.quality import DataQualityError, check, reconcile, trading_calendar
from stock_retrofit.data.repair import repair_bars
from stock_retrofit.data.sources import CorporateActionRecord, YFinanceSource, yahoo_ticker


def good_frame(n: int = 200, symbol: str = "TEST") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
            "symbol": symbol,
        }
    )


# ------------------------------------------------------------------ schema


def test_canonical_frame_passes():
    validate_canonical(good_frame(), source="test")


def test_wrong_column_order_is_a_violation():
    df = good_frame()[["close", "date", "open", "high", "low", "volume", "symbol"]]
    with pytest.raises(SchemaViolation, match="canonical"):
        validate_canonical(df, source="test")


def test_timezone_aware_dates_are_rejected():
    df = good_frame()
    df["date"] = df["date"].dt.tz_localize("Asia/Bangkok")
    with pytest.raises(SchemaViolation, match="timezone-naive"):
        validate_canonical(df, source="test")


def test_duplicate_dates_are_rejected():
    df = pd.concat([good_frame(10), good_frame(10)]).sort_values("date").reset_index(drop=True)
    with pytest.raises(SchemaViolation, match="duplicate"):
        validate_canonical(df, source="test")


def test_coerce_normalises_a_messy_frame():
    raw = good_frame().rename(columns={"date": "Date", "open": "Open", "close": "Close"})
    raw["Date"] = raw["Date"].dt.tz_localize("UTC")
    out = coerce_canonical(raw, "TEST", source="test")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["date"].dt.tz is None


# ---------------------------------------------------------------- gates


def test_close_outside_high_low_raises():
    """Acceptance criterion 7 of the data-layer spec, verbatim."""
    df = good_frame()
    df.loc[50, "close"] = df.loc[50, "high"] * 1.5
    with pytest.raises(DataQualityError, match="close outside"):
        check(df, "TEST")


def test_high_below_low_raises():
    df = good_frame()
    df.loc[10, "high"] = df.loc[10, "low"] - 1.0
    with pytest.raises(DataQualityError):
        check(df, "TEST")


def test_non_positive_price_raises():
    df = good_frame()
    df.loc[7, "low"] = -1.0
    with pytest.raises(DataQualityError):
        check(df, "TEST")


def test_move_beyond_thirty_percent_raises_when_unexplained():
    df = good_frame()
    df.loc[100:, ["open", "high", "low", "close"]] *= 2.0
    with pytest.raises(DataQualityError, match="beyond"):
        check(df, "TEST")


def test_move_beyond_thirty_percent_is_allowed_when_a_split_explains_it():
    df = good_frame()
    df.loc[100:, ["open", "high", "low", "close"]] *= 2.0
    actions = CorporateActionRecord(
        splits=pd.DataFrame({"date": [df["date"].iloc[100]], "ratio": [2.0]})
    )
    report = check(df, "TEST", actions=actions)
    assert report.ok


def test_scb_break_explains_the_relisting_jump():
    """The real case: SCB's price discontinuity lands on the first session after
    the halt, not on the break date itself."""
    df = good_frame(60, symbol="SCB")
    df["date"] = pd.to_datetime(
        ["2022-04-20"] + [str(d.date()) for d in pd.bdate_range("2022-04-27", periods=59)]
    )
    df.loc[1:, ["open", "high", "low", "close"]] *= 1.4
    report = check(df, "SCB")
    assert report.ok, report.violations


def test_zero_volume_is_an_advisory_not_a_violation():
    df = good_frame()
    df.loc[5:15, "volume"] = 0.0
    report = check(df, "TEST")
    assert report.ok
    assert any("zero-volume" in a for a in report.advisories)


def test_report_can_collect_without_raising():
    df = good_frame()
    df.loc[50, "close"] = df.loc[50, "high"] * 1.5
    report = check(df, "TEST", raise_on_violation=False)
    assert not report.ok and report.violations


# ---------------------------------------------------------------- repair


def test_repair_widens_the_bar_and_records_what_it_did():
    df = good_frame()
    original = float(df.loc[20, "high"])
    df.loc[20, "close"] = original * 1.02
    fixed, result = repair_bars(df, policy="widen_bar_to_close")
    assert result.count == 1
    assert result.records[0].field == "high"
    assert result.records[0].old == pytest.approx(original)
    check(fixed, "TEST")  # now passes the gate


def test_repair_policy_none_is_a_no_op():
    df = good_frame()
    df.loc[20, "close"] = df.loc[20, "high"] * 1.02
    fixed, result = repair_bars(df, policy="none")
    assert result.count == 0
    assert fixed.equals(df)


def test_unknown_repair_policy_raises():
    with pytest.raises(ValueError, match="unknown repair policy"):
        repair_bars(good_frame(), policy="make_it_up")


# ---------------------------------------------------------------- cache


def test_cache_roundtrip_writes_meta_sidecar(tmp_path):
    df = good_frame()
    meta = write_cache(df, "TEST", "unit", root=tmp_path)
    assert (tmp_path / "TEST.parquet").exists()
    assert (tmp_path / "TEST.meta.json").exists()
    payload = json.loads((tmp_path / "TEST.meta.json").read_text())
    assert payload["rows"] == len(df)
    assert payload["content_hash"] == meta.content_hash
    assert read_cache("TEST", tmp_path).equals(df)


def test_missing_cache_names_the_fix():
    with pytest.raises(FileNotFoundError, match="cli fetch"):
        read_cache("NOPE")


# ---------------------------------------------------------------- misc


def test_trading_calendar_is_the_union_of_observed_dates():
    a = good_frame(10, "A")
    b = good_frame(10, "B").iloc[2:]
    cal = trading_calendar({"A": a, "B": b})
    assert len(cal) == 10


def test_reconcile_flags_disagreement_beyond_one_tick():
    a = good_frame()
    b = a.copy()
    b.loc[30, "close"] += 5.0  # far more than a tick at ~100 baht
    table = reconcile(a, b)
    assert bool(table.loc[table["date"] == a["date"].iloc[30], "exceeds_one_tick"].iloc[0])
    assert int(table["exceeds_one_tick"].sum()) == 1


def test_reconcile_tolerates_sub_tick_differences():
    a = good_frame()
    b = a.copy()
    b["close"] += 1e-9
    assert int(reconcile(a, b)["exceeds_one_tick"].sum()) == 0


def test_yahoo_ticker_mapping():
    assert yahoo_ticker("SCB") == "SCB.BK"
    assert yahoo_ticker("KBANK") == "KBANK.BK"
    assert yahoo_ticker("BAY") == "BAY.BK"


def test_yfinance_source_satisfies_the_protocol():
    from stock_retrofit.data.protocol import PriceSource

    assert isinstance(YFinanceSource(), PriceSource)
