"""Walk-forward agent backtesting, run twice: frictionless and with SET costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.leakage import leakage_guard
from ..eval.preprocessing import (
    DEFAULT_FEATURES,
    FoldPreprocessor,
    WindowSpec,
    build_features,
    build_target,
)
from ..eval.runner import set_seed
from ..eval.splits import WalkForward, assert_no_overlap
from ..market import MarketConfig
from .base import Agent, AgentFoldResult, AgentResult, run_episode


def _fold_features(df: pd.DataFrame, fold, features=DEFAULT_FEATURES) -> np.ndarray:
    """Scale features with statistics from the fold's training block only."""
    feats = build_features(df, features)
    target = build_target(df)
    train_positions = np.arange(fold.train_start, fold.train_end - 1)
    pre = FoldPreprocessor(scale_target=False).fit(feats, target, train_positions)
    return pre.transform(feats)


def run_agent_walk_forward(
    agent: Agent,
    df: pd.DataFrame,
    *,
    splitter: WalkForward,
    config: MarketConfig,
    window: WindowSpec | None = None,
    features=DEFAULT_FEATURES,
    symbol: str = "?",
    seed: int = 42,
    guard: bool = True,
    verbose: bool = False,
) -> AgentResult:
    """Fit on each training block, execute on the test block that follows it.

    The agent never sees the test block during training — the correction to
    upstream's in-sample agent results. Each test block is then replayed twice
    through `SETMarket`, once frictionless and once with frictions.
    """
    window = window or WindowSpec()
    folds = splitter.split_frame(df)
    assert_no_overlap(folds)

    result = AgentResult(agent_name=agent.name, symbol=symbol, upstream=agent.upstream)
    free_config = config.frictionless_twin()

    for fold in folds:
        set_seed(seed + fold.index)
        try:
            with leakage_guard(
                np.arange(fold.test_start, fold.test_end), fold_index=fold.index, active=guard
            ):
                scaled = _fold_features(df, fold, features)
                train_bars = df.iloc[fold.train_start : fold.train_end].reset_index(drop=True)
                train_feats = scaled[fold.train_start : fold.train_end]
                agent.reset()
                agent.fit(train_bars, train_feats, config)

            # Test block, with enough leading history to fill the first window.
            lead = window.timestep - 1
            lo = max(0, fold.test_start - lead)
            test_bars = df.iloc[lo : fold.test_end].reset_index(drop=True)
            test_feats = scaled[lo : fold.test_end]
            offset = fold.test_start - lo

            with_frictions = run_episode(
                agent,
                test_bars,
                test_feats,
                config,
                timestep=window.timestep,
                start=offset,
                label=agent.name,
                symbol=symbol,
            )
            frictionless = run_episode(
                agent,
                test_bars,
                test_feats,
                free_config,
                timestep=window.timestep,
                start=offset,
                label=agent.name,
                symbol=symbol,
            )
            result.folds.append(
                AgentFoldResult(
                    fold_index=fold.index,
                    with_frictions=with_frictions,
                    frictionless=frictionless,
                )
            )
            if verbose:
                print(
                    f"  fold {fold.index}: friction {with_frictions.total_return:+.2%} "
                    f"| free {frictionless.total_return:+.2%} "
                    f"| {with_frictions.n_trades} trades"
                )
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  fold {fold.index} FAILED — {result.error}")
            break

    return result


def agent_results_table(results: list[AgentResult]) -> pd.DataFrame:
    """One row per agent, both runs side by side, gap computed (R11)."""
    rows = []
    for r in results:
        if not r.ok:
            rows.append(
                {
                    "agent": r.agent_name,
                    "symbol": r.symbol,
                    "folds": len(r.folds),
                    "ret_friction": float("nan"),
                    "ret_frictionless": float("nan"),
                    "friction_gap": float("nan"),
                    "sharpe_friction": float("nan"),
                    "sharpe_frictionless": float("nan"),
                    "trades": 0,
                    "costs": float("nan"),
                    "max_dd": float("nan"),
                    "status": r.error or "no folds",
                }
            )
            continue
        f = r.pooled_friction()
        g = r.pooled_frictionless()
        rows.append(
            {
                "agent": r.agent_name,
                "symbol": r.symbol,
                "folds": len(r.folds),
                "ret_friction": f["mean_fold_return"],
                "ret_frictionless": g["mean_fold_return"],
                "friction_gap": g["mean_fold_return"] - f["mean_fold_return"],
                "sharpe_friction": f["sharpe"],
                "sharpe_frictionless": g["sharpe"],
                "trades": f["n_trades"],
                "costs": f["total_costs"],
                "max_dd": f["max_drawdown"],
                "status": "ok",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    is_bh = table["agent"].str.contains("buy_and_hold", case=False, regex=False)
    return pd.concat(
        [
            table[is_bh],
            table[~is_bh].sort_values("ret_friction", ascending=False, na_position="last"),
        ],
        ignore_index=True,
    )


def render_agent_table(table: pd.DataFrame, *, title: str = "") -> str:
    if table.empty:
        return (title + "\n" if title else "") + "(no results)"
    d = table.copy()
    for col in ("ret_friction", "ret_frictionless", "friction_gap", "max_dd"):
        if col in d:
            d[col] = d[col].map(lambda v: "—" if pd.isna(v) else f"{v:+.2%}")
    for col in ("sharpe_friction", "sharpe_frictionless"):
        if col in d:
            d[col] = d[col].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    if "costs" in d:
        d["costs"] = d["costs"].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
    if (d["status"] == "ok").all():
        d = d.drop(columns=["status"])

    out = []
    if title:
        out += [title, "=" * (len(title) + 6)]
    out.append(d.to_string(index=False))

    ok = table[table["status"] == "ok"]
    if len(ok):
        profitable_free = int((ok["ret_frictionless"] > 0).sum())
        profitable_net = int((ok["ret_friction"] > 0).sum())
        out.append("")
        out.append(
            f"{profitable_free} of {len(ok)} agents are profitable frictionless; "
            f"{profitable_net} survive SET frictions. "
            f"Mean cost of frictions: {ok['friction_gap'].mean():+.2%} per fold."
        )
    failed = table[table["status"] != "ok"]
    if len(failed):
        out.append(f"{len(failed)} agent(s) failed to run: {', '.join(failed['agent'])}")
    return "\n".join(out)
