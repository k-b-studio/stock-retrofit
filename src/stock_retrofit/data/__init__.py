"""Price data: sources, cache, quality gates, instrument semantics.

This is the price axis of the `thai-market-data` spec, living inside
stock-retrofit rather than in a sibling package. See the README for why, and for
what was deferred (the fundamentals axis, in full).
"""

from .cache import cached_symbols, read_actions, read_cache, read_meta
from .corporate_actions import REGISTRY, apply_policy, breaks_for, describe, participation_cap_for
from .loader import fetch, load, quality_report, reconcile
from .protocol import (
    CANONICAL_COLUMNS,
    PriceSource,
    SchemaViolation,
    is_session,
    validate_canonical,
)
from .quality import DataQualityError, QualityReport
from .sources import SettradeSource, YFinanceSource, get_source

__all__ = [
    "CANONICAL_COLUMNS",
    "REGISTRY",
    "DataQualityError",
    "PriceSource",
    "QualityReport",
    "SchemaViolation",
    "SettradeSource",
    "YFinanceSource",
    "apply_policy",
    "breaks_for",
    "cached_symbols",
    "describe",
    "fetch",
    "get_source",
    "is_session",
    "load",
    "participation_cap_for",
    "quality_report",
    "read_actions",
    "read_cache",
    "read_meta",
    "reconcile",
    "validate_canonical",
]
