"""stock-retrofit — the Stock-Prediction-Models catalogue, rebuilt for Thai SET.

The upstream repo supplies architectures. It supplies nothing trustworthy about
methodology, so none of its evaluation code is reproduced here: scalers are fit
inside folds, splits are walk-forward, agents get a real holdout, and a naive
lag baseline sits on every results table.
"""

__version__ = "0.1.0"

DEFAULT_UNIVERSE = ("KBANK", "SCB", "BAY")
