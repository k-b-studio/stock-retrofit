"""The final comparison report — models x agents x tickers.

Acceptance criterion 9: the report must state plainly how many models beat
`NaiveLag` out-of-sample after costs, **including if the answer is zero**. That
sentence is generated from the numbers, not written by hand, so it cannot drift
away from what the tables actually say.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .config import DataConfig, EvalConfig, MarketConfigSpec, all_agent_specs, all_model_specs
from .paths import RESULTS_DIR


def _policy_for(symbol: str, data_cfg: DataConfig) -> str:
    return data_cfg.scb_history if symbol.upper() == "SCB" else "truncate_at_break"


def evaluate_symbol(symbol: str, *, verbose: bool = True) -> pd.DataFrame:
    from .data import load
    from .eval import render_table, results_table, run_walk_forward

    eval_cfg, market_cfg, data_cfg = EvalConfig.load(), MarketConfigSpec.load(), DataConfig.load()
    df = load(symbol, policy=_policy_for(symbol, data_cfg))
    results = []
    for spec in all_model_specs():
        if verbose:
            print(f"    {spec.name} ...", end=" ", flush=True)
        r = run_walk_forward(
            spec.build(),
            df,
            splitter=eval_cfg.splitter(),
            window=eval_cfg.window(),
            features=eval_cfg.features,
            symbol=symbol,
            seed=eval_cfg.seed,
            cost_per_turn=market_cfg.round_trip_cost,
            allow_short=market_cfg.allow_short,
        )
        if verbose:
            print("ok" if r.ok else "FAILED")
        results.append(r)
    table = results_table(
        results, cost_per_turn=market_cfg.round_trip_cost, allow_short=market_cfg.allow_short
    )
    table.to_csv(RESULTS_DIR / f"evaluate-{symbol}.csv", index=False)
    if verbose:
        print(render_table(table, title=f"{symbol} — models"))
    return table


def backtest_symbol(symbol: str, *, verbose: bool = True) -> pd.DataFrame:
    from .agents import agent_results_table, render_agent_table, run_agent_walk_forward
    from .data import load

    eval_cfg, market_spec, data_cfg = EvalConfig.load(), MarketConfigSpec.load(), DataConfig.load()
    df = load(symbol, policy=_policy_for(symbol, data_cfg))
    config = market_spec.build(symbol=symbol)
    results = []
    for spec in all_agent_specs():
        if verbose:
            print(f"    {spec.name} ...", end=" ", flush=True)
        r = run_agent_walk_forward(
            spec.build(),
            df,
            splitter=eval_cfg.splitter(),
            config=config,
            window=eval_cfg.window(),
            features=eval_cfg.features,
            symbol=symbol,
            seed=eval_cfg.seed,
        )
        if verbose:
            print("ok" if r.ok else "FAILED")
        results.append(r)
    table = agent_results_table(results)
    table.to_csv(RESULTS_DIR / f"backtest-{symbol}.csv", index=False)
    if verbose:
        print(render_agent_table(table, title=f"{symbol} — agents"))
    return table


def _load_or_run(symbol: str, kind: str, run_missing: bool) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{kind}-{symbol}.csv"
    if path.exists():
        return pd.read_csv(path)
    if not run_missing:
        return None
    print(f"  {kind} {symbol} (no cached result, running)")
    return evaluate_symbol(symbol) if kind == "evaluate" else backtest_symbol(symbol)


def _md_table(df: pd.DataFrame, columns: dict[str, str], formats: dict) -> str:
    view = df[[c for c in columns if c in df.columns]].copy()
    for col, spec in formats.items():
        if col in view:
            view[col] = view[col].map(lambda v, s=spec: "—" if pd.isna(v) else s.format(v))
    view = view.rename(columns=columns)
    header = "| " + " | ".join(view.columns) + " |"
    rule = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _buy_and_hold_verdict(ok: pd.DataFrame, symbol: str) -> list[str]:
    """How many active agents beat simply holding the stock, net of costs.

    The counterpart to `NaiveLag` on the model tables. An agent that trades hard
    and lands below buy-and-hold has bought turnover, not alpha — and upstream
    had no way to notice, because it ran no baseline and charged no costs.
    """
    is_bh = ok["agent"].str.contains("buy_and_hold", case=False, regex=False)
    if not is_bh.any():
        return []
    baseline = float(ok.loc[is_bh, "ret_friction"].iloc[0])
    active = ok[~is_bh]
    beat = active[active["ret_friction"] > baseline]

    lines = [
        f"Buy-and-hold returned **{baseline:+.2%}** per fold after costs on {symbol}. "
        f"**{len(beat)} of {len(active)} active agents beat it.**"
        + (f" Namely: {', '.join(beat['agent'])}." if len(beat) else ""),
        "",
    ]
    flipped = active[(active["ret_frictionless"] > 0) & (active["ret_friction"] <= 0)]
    if len(flipped):
        lines += [
            f"{len(flipped)} agent(s) are profitable frictionless and lose money once SET "
            f"costs are charged: {', '.join(flipped['agent'])}. That sign change is the "
            "single clearest argument for the friction layer existing.",
            "",
        ]
    return lines


def build_report(symbols: list[str], *, run_missing: bool = True) -> str:
    from .data import describe, read_meta
    from .eval import git_sha, summarise_beats

    eval_cfg, market_spec = EvalConfig.load(), MarketConfigSpec.load()
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    out: list[str] = [
        "# stock-retrofit — results",
        "",
        f"Generated {now} · git `{git_sha()}` · seed {eval_cfg.seed}",
        "",
        "Walk-forward evaluation of the "
        "[huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) "
        "catalogue on Thai SET bank shares, on a harness that does not leak and a "
        "backtest that charges SET trading costs.",
        "",
        "## How to read this",
        "",
        "- **MASE** is MAE(model) / MAE(naive lag) on next-day returns. **Below 1.00 beats the "
        "naive lag; at or above 1.00 it does not.** The naive lag is on every table by construction.",
        "- **dir_acc** counts only rows where a model made a directional call. The naive lag "
        "abstains everywhere, so its accuracy is undefined rather than zero.",
        f"- **sharpe_net** is annualised, after a round-trip cost of "
        f"{market_spec.round_trip_cost:.3%}. **sharpe_gross** charges nothing.",
        f"- Splits: {eval_cfg.train_window} training bars, {eval_cfg.test_window}-bar test blocks, "
        f"step {eval_cfg.step}, most recent {eval_cfg.max_folds} folds. Every scaler is fit inside "
        "its own fold.",
        "",
        "> **Cost figures are reconstructed, not verified** against SET's rulebook or a broker "
        "> schedule (spec R13). Treat them as order-of-magnitude.",
        "",
        "## Universe",
        "",
    ]

    for symbol in symbols:
        meta = read_meta(symbol)
        out.append(
            f"**{symbol}** — " + describe(symbol).split("\n", 1)[1].strip().replace("\n", "; ")
        )
        if meta:
            out.append(
                f"  · {meta.rows} bars {meta.start} → {meta.end}, source `{meta.source}`, "
                f"hash `{meta.content_hash[:12]}`, {meta.repairs.get('count', 0)} repaired field(s)"
            )
        out.append("")

    totals = {"ran": 0, "beat": 0}
    agent_totals = {"ran": 0, "profit_free": 0, "profit_net": 0}

    for symbol in symbols:
        out += [f"## {symbol}", ""]

        models = _load_or_run(symbol, "evaluate", run_missing)
        if models is not None and len(models):
            summary = summarise_beats(models)
            totals["ran"] += summary["ran"]
            totals["beat"] += summary["beat_naive"]
            out += [
                "### Forecasting models",
                "",
                _md_table(
                    models,
                    {
                        "model": "model",
                        "MASE": "MASE",
                        "beats_naive": "beats naive",
                        "dir_acc": "dir acc",
                        "RMSE_ret": "RMSE(ret)",
                        "sharpe_net": "Sharpe net",
                        "sharpe_gross": "Sharpe gross",
                    },
                    {
                        "MASE": "{:.4f}",
                        "dir_acc": "{:.1%}",
                        "RMSE_ret": "{:.5f}",
                        "sharpe_net": "{:+.2f}",
                        "sharpe_gross": "{:+.2f}",
                    },
                ),
                "",
                f"**{summary['beat_naive']} of {summary['ran']} models beat the naive lag on "
                f"{symbol}.**"
                + (f" Winners: {', '.join(summary['winners'])}." if summary["winners"] else ""),
                "",
            ]

        agents = _load_or_run(symbol, "backtest", run_missing)
        if agents is not None and len(agents):
            ok = agents[agents["status"] == "ok"] if "status" in agents else agents
            agent_totals["ran"] += len(ok)
            agent_totals["profit_free"] += int((ok["ret_frictionless"] > 0).sum())
            agent_totals["profit_net"] += int((ok["ret_friction"] > 0).sum())
            out += [
                "### Agents — frictionless vs. SET frictions",
                "",
                _md_table(
                    agents,
                    {
                        "agent": "agent",
                        "ret_frictionless": "return (free)",
                        "ret_friction": "return (frictions)",
                        "friction_gap": "cost of frictions",
                        "sharpe_friction": "Sharpe net",
                        "trades": "trades",
                        "max_dd": "max DD",
                    },
                    {
                        "ret_frictionless": "{:+.2%}",
                        "ret_friction": "{:+.2%}",
                        "friction_gap": "{:+.2%}",
                        "sharpe_friction": "{:+.2f}",
                        "max_dd": "{:+.2%}",
                    },
                ),
                "",
                f"Mean return per fold is shown. **{int((ok['ret_frictionless'] > 0).sum())} of "
                f"{len(ok)} agents make money frictionless; "
                f"{int((ok['ret_friction'] > 0).sum())} still do after SET costs.** "
                f"Frictions cost {ok['friction_gap'].mean():+.2%} per fold on average.",
                "",
                *_buy_and_hold_verdict(ok, symbol),
            ]

    scope = symbols[0] if len(symbols) == 1 else f"{len(symbols)} tickers ({', '.join(symbols)})"
    out += [
        "## Headline",
        "",
        f"**{totals['beat']} of {totals['ran']} model runs beat the naive lag out-of-sample "
        f"across {scope}.**",
        "",
    ]
    if totals["beat"] == 0:
        out += [
            "That number is zero, and it is reported as zero. It is the result the spec "
            "anticipated as legitimate and likely, and it is what a non-leaking, "
            "cost-charging harness produces from this catalogue on this universe.",
            "",
            "The upstream repository reports accuracies in the high nineties for the same "
            "architectures. Both things are true at once, and the reason is methodological, "
            "not architectural: upstream fits its scaler on the full series before splitting, "
            "scores price levels rather than returns, and shows no baseline. On price levels a "
            "naive lag also scores in the high nineties — `metrics.upstream_accuracy_do_not_use` "
            "and its test demonstrate this. Those numbers never measured skill.",
            "",
        ]
    else:
        out += [
            "Treat any winner with suspicion proportional to its margin: 8 folds of 60 days is "
            "480 observations, and a MASE of 0.99 over that sample is not a discovery.",
            "",
        ]

    if agent_totals["ran"]:
        out += [
            f"**{agent_totals['profit_free']} of {agent_totals['ran']} agent runs are profitable "
            f"frictionless; {agent_totals['profit_net']} survive SET frictions.** The difference "
            "between those two numbers is the cost of assuming a market without board lots, "
            "tick sizes, commission or VAT — the assumption every upstream agent notebook makes.",
            "",
        ]

    out += [
        "## What would change these numbers",
        "",
        "- **A better data source.** yfinance is a redistributor; Settrade is the exchange's own "
        "feed and is unreachable here without broker credentials. See `docs/settrade-api-notes.md`.",
        "- **Verified cost parameters.** The commission rate drives the friction gap directly and "
        "is currently a plausible guess (spec R13).",
        "- **Features.** Every model reads the same five causal feature blocks. The catalogue was "
        "designed around a single close-price series; giving it a richer, well-motivated feature "
        "set is a more promising direction than any architecture on these tables.",
        "- **More folds.** Configurable in `configs/eval.yaml`; `max_folds: null` uses every "
        "available fold rather than the most recent 8.",
        "",
        "## Provenance",
        "",
        "Every run in `results/` has a `*.manifest.json` beside it recording the config, the git "
        "SHA, and the content hash of the bars consumed.",
        "",
    ]
    return "\n".join(out)
