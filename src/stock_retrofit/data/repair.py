"""Explicit, audited repair of vendor bar defects.

Repair is deliberately a **separate stage** from the quality gate. The gate
still raises on any violation it is handed (spec R12's fail-loud requirement is
not softened); repair runs before it, under a named policy, and writes every
change it made into the cache metadata and the quality report. A repaired bar is
therefore always visible in the audit trail — the thing the gate exists to
prevent is a bad bar passing *silently*, not a bad bar being fixed on the record.

Why this is needed at all: Yahoo's SET bars contain a small number of days where
`close` falls outside `[low, high]` — 3 in KBANK, 1 in SCB, 6 in BAY over 26
years, by one to three ticks. The defect is in the vendor's intraday extremes,
not in the close (the close is the reference price and is corroborated by the
next day's open). `widen_bar_to_close` therefore trusts open/close and widens
the range to contain them, which is the minimal edit that makes the bar
internally consistent. No price is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

POLICIES = ("none", "widen_bar_to_close")


@dataclass
class RepairRecord:
    date: str
    field: str
    old: float
    new: float
    reason: str

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RepairResult:
    policy: str
    records: list[RepairRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_json(self) -> dict:
        return {
            "policy": self.policy,
            "count": self.count,
            "records": [r.to_json() for r in self.records],
        }

    def render(self) -> str:
        if not self.records:
            return f"repair policy '{self.policy}': nothing to repair"
        lines = [f"repair policy '{self.policy}': {self.count} field(s) adjusted"]
        for r in self.records[:20]:
            lines.append(f"    {r.date} {r.field}: {r.old:g} -> {r.new:g}  ({r.reason})")
        if self.count > 20:
            lines.append(f"    ... and {self.count - 20} more")
        return "\n".join(lines)


def repair_bars(
    df: pd.DataFrame, *, policy: str = "widen_bar_to_close"
) -> tuple[pd.DataFrame, RepairResult]:
    """Apply `policy` to a canonical frame. Returns the frame and an audit trail."""
    if policy not in POLICIES:
        raise ValueError(f"unknown repair policy {policy!r}; expected one of {POLICIES}")

    result = RepairResult(policy=policy)
    if policy == "none" or not len(df):
        return df, result

    out = df.copy()
    wanted_high = out[["high", "open", "close"]].max(axis=1)
    wanted_low = out[["low", "open", "close"]].min(axis=1)

    high_fixes = out.index[wanted_high > out["high"] + 1e-9]
    low_fixes = out.index[wanted_low < out["low"] - 1e-9]

    for idx in high_fixes:
        result.records.append(
            RepairRecord(
                date=str(out.at[idx, "date"].date()),
                field="high",
                old=float(out.at[idx, "high"]),
                new=float(wanted_high.at[idx]),
                reason="open/close exceeded the reported high",
            )
        )
    for idx in low_fixes:
        result.records.append(
            RepairRecord(
                date=str(out.at[idx, "date"].date()),
                field="low",
                old=float(out.at[idx, "low"]),
                new=float(wanted_low.at[idx]),
                reason="open/close fell below the reported low",
            )
        )

    out["high"] = wanted_high
    out["low"] = wanted_low
    result.records.sort(key=lambda r: (r.date, r.field))
    return out, result
