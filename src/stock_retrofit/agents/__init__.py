"""Trading agents.

23 upstream notebooks, collapsed into families. The Q-learning set alone
(`5,7,8,9,10,11,12,13,18,19,20`) is one skeleton with four boolean flags —
`{double, duel, recurrent, curiosity}` — rather than eleven near-identical
files. See `docs/upstream-mapping.md` for the 1:1 trace.
"""

# Importing each family registers it.
from . import evolution, policy_gradient, qfamily, rule_based  # noqa: F401
from .base import (
    BUY,
    HOLD,
    SELL,
    Agent,
    AgentResult,
    Observation,
    build,
    register,
    registered_kinds,
)
from .runner import agent_results_table, render_agent_table, run_agent_walk_forward

__all__ = [
    "BUY",
    "HOLD",
    "SELL",
    "Agent",
    "AgentResult",
    "Observation",
    "agent_results_table",
    "build",
    "register",
    "registered_kinds",
    "render_agent_table",
    "run_agent_walk_forward",
]
