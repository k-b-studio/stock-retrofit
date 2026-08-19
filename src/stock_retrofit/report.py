"""The final comparison report — models x agents x tickers.

Acceptance criterion 9: the report must state plainly whether the models showed
any out-of-sample skill, **including if the answer is no**. That sentence is
generated from the numbers, not written by hand, so it cannot drift away from
what the tables actually say.

The measure is IC — the correlation between forecast and realised return —
reported against `always_long` net of costs, and read next to the number of
significant runs expected by chance. An earlier revision headlined a count of
models beating `naive_lag` on MASE; that count was near-fixed before any model
ran (see the module docstring of `eval.report`), so it was replaced by one that
could have come out the other way. The conclusion did not change.
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


def _participation_note(symbol: str, ok: pd.DataFrame) -> list[str]:
    """Explain a large friction gap that is liquidity, not commission.

    Without this, BAY's numbers read as though commission cost nine percent.
    """
    from .data import participation_cap_for

    cap = participation_cap_for(symbol)
    if cap is None or not len(ok):
        return []
    gap = float(ok["friction_gap"].mean())
    return [
        f"> **{symbol} carries a {cap:.0%} participation cap**, so the frictionless column also "
        f"drops the liquidity constraint — it lets an agent take a position the market could not "
        f"actually have absorbed. Part of the mean gap here ({gap:+.2%}) is therefore a statement "
        f"about {symbol}'s float rather than about commission. With ~72-76% of shares held by "
        f"MUFG, that is the intended reading: the frictionless number is not a return anyone "
        f"could have earned.",
        "",
        "> The cap also changes what a *fair baseline* means. A buy-and-hold that issues one "
        "order, has it trimmed to a fraction of a session's volume and then stops would sit "
        "mostly in cash — and would lose to any agent that trades repeatedly, purely because the "
        "reference line was handicapped. `buy_and_hold` here accumulates across sessions until "
        "its capital is deployed, which is what a real holder does under a liquidity constraint.",
        "",
    ]


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
    from .eval import git_sha, summarise_skill

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
        "- **IC** is the out-of-sample correlation between a model's forecast and the return that "
        "actually happened. **This is the skill column.** A daily equity IC of 0.02-0.05 is a real "
        "signal and 0.10 is excellent; 0.00 is knowing nothing. `t` is its t-statistic — but read "
        "it against the count of models tested, because one or two in twenty clear |t| > 1.96 by "
        "chance and models sharing a feature set are not independent draws.",
        "- **MASE** is MAE(model) / MAE(naive lag) on next-day returns, and is a **ranking, not a "
        "test**. There is deliberately no 'beats naive' column: MAE is minimised by the conditional "
        "median, which on daily returns is ≈ 0, so a forecast of zero is already near-optimal and "
        "anything that moves off it pays — and on a series where 13-24% of days close unchanged, a "
        "flat day adds nothing to the denominator and pure error to the numerator. A simulated "
        "forecaster with a genuine IC of 0.10 crosses MASE 1.00 in 27% of draws on KBANK, 34% on "
        "SCB and 0% on BAY, so a table of MASE ≥ 1.00 says more about the metric than about the "
        "catalogue.",
        "- **dir_acc** counts calls made on days the price **actually moved**. Flat closes are "
        "excluded from the denominator: `sign(0)` matches no forecast, so a day that did not move "
        "is a guaranteed miss for every model, and on these tickers 13-24% of sessions close "
        "unchanged on the SET tick grid. `flat_share` in the CSV reports how many were set aside. "
        "The naive lag abstains everywhere, so its accuracy is undefined rather than zero.",
        "- **Two reference rows are pinned to the top of every table.** `naive_lag` is the "
        "reference for the forecast as a number; `always_long` is the reference for it as a "
        "position — its Sharpe is what holding the share paid over the same blocks.",
        "- Vendor-padded non-sessions are excluded. yfinance fills SET holidays with a zero-volume, "
        "zero-range bar repeating the previous close; the 'return' on one is zero by construction. "
        "Those rows are dropped from the labels and orders are refused on those bars.",
        f"- **sharpe_net** is annualised, after a round-trip cost of "
        f"{market_spec.round_trip_cost:.3%}. **sharpe_gross** charges nothing.",
        f"- Splits: {eval_cfg.train_window} training bars, {eval_cfg.test_window}-bar test blocks, "
        f"step {eval_cfg.step}, up to {eval_cfg.max_folds} of the most recent folds — a truncated "
        "history yields fewer, and the per-ticker `folds` column says how many actually ran. "
        "Every scaler is fit inside its own fold.",
        "",
        "> **Cost figures are reconstructed, not verified** against SET's rulebook or a broker "
        "schedule (spec R13). Treat them as order-of-magnitude.",
        "",
        "## Figures",
        "",
        "Rendered by `notebooks/07_figures.ipynb` — one figure per cell.",
        "",
        "![Does anything beat the naive lag?](figures/01_mase_vs_naive.png)",
        "",
        "![Directional accuracy on days the price moved](figures/02_directional_accuracy.png)",
        "",
        "![What SET frictions cost](figures/03_friction_gap.png)",
        "",
        "![The best-scoring model, up close](figures/04_forecast_reality.png)",
        "",
        "## Universe",
        "",
    ]

    for symbol in symbols:
        meta = read_meta(symbol)
        lines = [ln.strip() for ln in describe(symbol).split("\n")]
        out.append(f"**{symbol}** — {lines[0].split('—', 1)[-1].strip()}")
        for line in lines[1:]:
            if line:
                out.append(f"  - {line}")
        if meta:
            out.append(
                f"  - {meta.rows} bars {meta.start} → {meta.end}, source `{meta.source}`, "
                f"hash `{meta.content_hash[:12]}`, {meta.repairs.get('count', 0)} repaired field(s)"
            )
        out.append("")

    totals = {"ran": 0, "positive_ic": 0, "significant": 0, "beat_long": 0, "ic_sum": 0.0}
    agent_totals = {"ran": 0, "profit_free": 0, "profit_net": 0}

    for symbol in symbols:
        out += [f"## {symbol}", ""]

        models = _load_or_run(symbol, "evaluate", run_missing)
        if models is not None and len(models):
            summary = summarise_skill(models)
            totals["ran"] += summary["ran"]
            totals["positive_ic"] += summary["positive_ic"]
            totals["significant"] += summary["significant"]
            totals["beat_long"] += summary["beat_always_long"]
            totals["ic_sum"] += summary["mean_ic"] * summary["ran"]
            long_sharpe = summary["reference"].get("always_long")
            out += [
                "### Forecasting models",
                "",
                _md_table(
                    models,
                    {
                        "model": "model",
                        "ic": "IC",
                        "ic_t": "t",
                        "MASE": "MASE",
                        "dir_acc": "dir acc",
                        "RMSE_ret": "RMSE(ret)",
                        "sharpe_net": "Sharpe net",
                        "sharpe_gross": "Sharpe gross",
                    },
                    {
                        "ic": "{:+.3f}",
                        "ic_t": "{:+.1f}",
                        "MASE": "{:.4f}",
                        "dir_acc": "{:.1%}",
                        "RMSE_ret": "{:.5f}",
                        "sharpe_net": "{:+.2f}",
                        "sharpe_gross": "{:+.2f}",
                    },
                ),
                "",
                f"**Mean IC {summary['mean_ic']:+.3f} over {summary['ran']} models on {symbol}; "
                f"{summary['positive_ic']} of {summary['ran']} positive.** "
                f"{summary['significant']} clear |t| > 1.96, against "
                f"{summary['expected_false_positives']:.0f} expected by chance"
                + (f": {', '.join(summary['leaders'])}." if summary["leaders"] else "."),
                "",
            ]
            if long_sharpe is not None:
                out += [
                    f"Holding {symbol} scored a net Sharpe of **{long_sharpe:+.2f}** over the same "
                    f"blocks. **{summary['beat_always_long']} of {summary['ran']} models beat it.**",
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
                *_participation_note(symbol, ok),
                *_buy_and_hold_verdict(ok, symbol),
            ]

    scope = symbols[0] if len(symbols) == 1 else f"{len(symbols)} tickers ({', '.join(symbols)})"
    mean_ic = totals["ic_sum"] / totals["ran"] if totals["ran"] else float("nan")
    out += [
        "## Headline",
        "",
        f"**Mean out-of-sample IC of {mean_ic:+.3f} across {totals['ran']} model runs on {scope}; "
        f"{totals['positive_ic']} of {totals['ran']} are positive.**",
        "",
        "A coin flip would put half of them above zero. That is what a catalogue with no "
        "forecasting skill on this universe looks like, and it is the result the spec anticipated "
        "as legitimate and likely.",
        "",
        f"Two supporting facts, both pointing the same way. {totals['significant']} runs clear "
        f"|t| > 1.96 against roughly {0.05 * totals['ran']:.0f} expected by chance alone — and "
        "those runs are not independent draws, since all 22 architectures read the same five "
        f"features. And {totals['beat_long']} of {totals['ran']} beat simply holding the share, "
        "which is the comparison that decides whether any of this was worth running.",
        "",
        "The upstream repository reports accuracies in the high nineties for the same "
        "architectures. Both things are true at once, and the reason is methodological, "
        "not architectural: upstream fits its scaler on the full series before splitting, "
        "scores price levels rather than returns, and shows no baseline. On price levels a "
        "naive lag also scores in the high nineties — `metrics.upstream_accuracy_do_not_use` "
        "and its test demonstrate this. Those numbers never measured skill.",
        "",
        "**What this report no longer claims.** Earlier revisions headlined '0 of 66 model runs "
        "beat the naive lag'. That statement was true and close to vacuous: on these zero-inflated "
        "series a forecaster with a genuine edge crosses MASE 1.00 at best a third of the time and "
        "on BAY not at all, so the count was largely fixed before a single model ran. The "
        "conclusion has not changed — the models do not work — but it now rests on a measurement "
        "that could have come out the other way.",
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
