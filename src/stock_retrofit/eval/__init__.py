"""The evaluation harness — this project's spine.

Built before any model, as the review insisted, because a faithful port of a
leaking pipeline is just a faster leaking pipeline.
"""

from .leakage import LeakageError, leakage_guard, register_fit
from .manifest import RunManifest, data_fingerprint, git_sha
from .metrics import MetricSet, evaluate, mase, sharpe
from .preprocessing import FoldArrays, WindowSpec, prepare_fold
from .report import render_table, results_table, summarise_beats
from .runner import EvalResult, run_walk_forward, set_seed
from .splits import Fold, WalkForward

__all__ = [
    "EvalResult",
    "Fold",
    "FoldArrays",
    "LeakageError",
    "MetricSet",
    "RunManifest",
    "WalkForward",
    "WindowSpec",
    "data_fingerprint",
    "evaluate",
    "git_sha",
    "leakage_guard",
    "mase",
    "prepare_fold",
    "register_fit",
    "render_table",
    "results_table",
    "run_walk_forward",
    "set_seed",
    "sharpe",
    "summarise_beats",
]
