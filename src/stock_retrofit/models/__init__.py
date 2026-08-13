"""Forecasting model families.

Coverage of the upstream `deep-learning/` catalogue without its duplication:
18 notebooks collapse into a handful of parameterised families plus one YAML
config per upstream file. See `docs/upstream-mapping.md` for the 1:1 trace.
"""

# Importing each family registers it. Order is irrelevant; completeness is not.
from . import attention, baselines, classical, conv, recurrent, seq2seq, stacking  # noqa: F401
from .base import ForecastModel, build, register, registered_kinds

__all__ = ["ForecastModel", "build", "register", "registered_kinds"]
