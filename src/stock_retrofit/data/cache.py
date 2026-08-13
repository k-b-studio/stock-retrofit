"""Parquet cache with metadata sidecars.

Only `fetch` touches the network (R3/R7). Every write lands a
`{symbol}.meta.json` recording source, timestamp, rows, range and a content
hash, so any downstream result traces back to the exact bytes that produced it
(R9). Corporate actions from the source go to `{symbol}.actions.json`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ..paths import RAW_DIR
from .protocol import CANONICAL_COLUMNS, validate_canonical
from .repair import repair_bars
from .sources import CorporateActionRecord


def content_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 over the canonical frame's values."""
    payload = df[CANONICAL_COLUMNS].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CacheMeta:
    symbol: str
    source: str
    fetched_at: str
    rows: int
    start: str | None
    end: str | None
    content_hash: str
    #: Audit trail of vendor-defect repairs applied before caching (see repair.py).
    repairs: dict = field(default_factory=lambda: {"policy": "none", "count": 0, "records": []})

    def to_json(self) -> dict:
        return self.__dict__.copy()


def parquet_path(symbol: str, root: Path | None = None) -> Path:
    return (root or RAW_DIR) / f"{symbol.upper()}.parquet"


def meta_path(symbol: str, root: Path | None = None) -> Path:
    return (root or RAW_DIR) / f"{symbol.upper()}.meta.json"


def actions_path(symbol: str, root: Path | None = None) -> Path:
    return (root or RAW_DIR) / f"{symbol.upper()}.actions.json"


def write_cache(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    *,
    actions: CorporateActionRecord | None = None,
    repairs: dict | None = None,
    root: Path | None = None,
) -> CacheMeta:
    root = root or RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    validate_canonical(df, source=source)

    df.to_parquet(parquet_path(symbol, root), index=False)
    meta = CacheMeta(
        symbol=symbol.upper(),
        source=source,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        rows=int(len(df)),
        start=str(df["date"].min().date()) if len(df) else None,
        end=str(df["date"].max().date()) if len(df) else None,
        content_hash=content_hash(df),
        repairs=repairs or {"policy": "none", "count": 0, "records": []},
    )
    meta_path(symbol, root).write_text(json.dumps(meta.to_json(), indent=2) + "\n")
    if actions is not None:
        actions_path(symbol, root).write_text(json.dumps(actions.to_json(), indent=2) + "\n")
    return meta


def read_cache(symbol: str, root: Path | None = None) -> pd.DataFrame:
    path = parquet_path(symbol, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached bars for {symbol} at {path}. "
            f"Run: python -m stock_retrofit.cli fetch --symbols {symbol}"
        )
    return validate_canonical(pd.read_parquet(path), source="cache")


def read_meta(symbol: str, root: Path | None = None) -> CacheMeta | None:
    path = meta_path(symbol, root)
    if not path.exists():
        return None
    return CacheMeta(**json.loads(path.read_text()))


def read_actions(symbol: str, root: Path | None = None) -> CorporateActionRecord:
    path = actions_path(symbol, root)
    if not path.exists():
        return CorporateActionRecord()
    return CorporateActionRecord.from_json(json.loads(path.read_text()))


def cached_symbols(root: Path | None = None) -> list[str]:
    root = root or RAW_DIR
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.parquet"))


def fetch_symbol(
    symbol: str,
    source,
    *,
    start: date,
    end: date,
    force_refresh: bool = False,
    repair_policy: str = "widen_bar_to_close",
    root: Path | None = None,
) -> tuple[pd.DataFrame, CacheMeta]:
    """Fetch and cache one symbol, incrementally unless `force_refresh` (R8).

    Incremental fetch re-requests the last cached day so a bar that was still
    provisional when first seen gets corrected rather than frozen. Vendor bar
    defects are repaired under `repair_policy` and the edits are recorded in the
    metadata sidecar — never applied silently.
    """
    root = root or RAW_DIR
    existing = None
    fetch_start = start

    if not force_refresh and parquet_path(symbol, root).exists():
        existing = read_cache(symbol, root)
        if len(existing):
            last = existing["date"].max()
            if last.date() >= end:
                return existing, read_meta(symbol, root)  # type: ignore[return-value]
            fetch_start = last.date()

    fresh = source.fetch(symbol, fetch_start, end)
    actions = source.last_actions() if hasattr(source, "last_actions") else None

    if existing is not None and len(existing):
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = (
            combined.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")  # fresh wins on overlap
            .reset_index(drop=True)
        )
        if actions is not None and fetch_start > start:
            # Incremental window only carries recent actions; merge with what we had.
            actions = _merge_actions(read_actions(symbol, root), actions)
    else:
        combined = fresh.reset_index(drop=True)

    combined, repair_result = repair_bars(combined, policy=repair_policy)
    meta = write_cache(
        combined,
        symbol,
        source.name,
        actions=actions,
        repairs=repair_result.to_json(),
        root=root,
    )
    return combined, meta


def _merge_actions(old: CorporateActionRecord, new: CorporateActionRecord) -> CorporateActionRecord:
    def _merge(a: pd.DataFrame, b: pd.DataFrame, value_col: str) -> pd.DataFrame:
        frames = [f for f in (a, b) if len(f)]
        if not frames:
            return pd.DataFrame(columns=["date", value_col])
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

    return CorporateActionRecord(
        dividends=_merge(old.dividends, new.dividends, "amount"),
        splits=_merge(old.splits, new.splits, "ratio"),
    )
