"""Command line: fetch | quality | reconcile | evaluate | backtest | report.

    python -m stock_retrofit.cli fetch --symbols SCB,KBANK,BAY
    python -m stock_retrofit.cli evaluate --config configs/models/01_lstm.yaml --symbol KBANK
    python -m stock_retrofit.cli evaluate --all --symbol KBANK
    python -m stock_retrofit.cli backtest --all --symbol KBANK
    python -m stock_retrofit.cli report --symbols KBANK,SCB,BAY

Every run that produces numbers also writes a manifest next to them recording
the config, the git SHA and the content hash of the data consumed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .config import (
    AgentSpec,
    DataConfig,
    EvalConfig,
    MarketConfigSpec,
    ModelSpec,
    all_agent_specs,
    all_model_specs,
)
from .paths import AGENT_CONFIG_DIR, MODEL_CONFIG_DIR, RESULTS_DIR, ensure_dirs


def _symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _policy_for(symbol: str, data_cfg: DataConfig) -> str:
    return data_cfg.scb_history if symbol.upper() == "SCB" else "truncate_at_break"


# ---------------------------------------------------------------- fetch


def cmd_fetch(args) -> int:
    from .data import fetch, quality_report

    ensure_dirs()
    data_cfg = DataConfig.load()
    symbols = _symbols(args.symbols) if args.symbols else data_cfg.symbols
    start = pd.Timestamp(args.start or data_cfg.start).date()

    print(f"Fetching {', '.join(symbols)} from {data_cfg.source} since {start}\n")
    metas = fetch(
        symbols,
        source=args.source or data_cfg.source,
        start=start,
        end=date.today(),
        force_refresh=args.force_refresh,
        repair_policy=data_cfg.repair_policy,
    )
    for sym, meta in metas.items():
        print(
            f"  {sym:6s} {meta.rows:5d} bars  {meta.start} .. {meta.end}  "
            f"hash {meta.content_hash[:12]}  repairs {meta.repairs.get('count', 0)}"
        )

    print()
    failed = 0
    for sym in symbols:
        report = quality_report(sym, raise_on_violation=False)
        print(report.render())
        print()
        if not report.ok:
            failed += 1
    if failed:
        print(f"{failed} symbol(s) have structural violations.", file=sys.stderr)
    return 1 if failed else 0


def cmd_quality(args) -> int:
    from .data import quality_report

    failed = 0
    for sym in _symbols(args.symbols):
        report = quality_report(sym, raise_on_violation=False)
        print(report.render())
        print()
        failed += 0 if report.ok else 1
    return 1 if failed else 0


def cmd_reconcile(args) -> int:
    from .data import reconcile

    for sym in _symbols(args.symbols):
        table = reconcile(sym, against=args.against)
        n_bad = int(table["exceeds_one_tick"].sum())
        print(f"{sym}: {len(table)} overlapping dates, {n_bad} exceed one tick of disagreement")
        if n_bad:
            worst = table.loc[table["exceeds_one_tick"]].nlargest(10, "abs_diff")
            print(worst.to_string(index=False))
        print()
    return 0


# ---------------------------------------------------------------- evaluate


def cmd_evaluate(args) -> int:
    from .data import load
    from .eval import RunManifest, data_fingerprint, render_table, results_table, run_walk_forward

    ensure_dirs()
    eval_cfg = EvalConfig.load()
    market_cfg = MarketConfigSpec.load()
    data_cfg = DataConfig.load()

    specs = _resolve_model_specs(args)
    symbol = args.symbol.upper()
    df = load(symbol, policy=_policy_for(symbol, data_cfg))
    print(f"{symbol}: {len(df)} bars, {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"{len(specs)} model config(s), cost per turn {market_cfg.round_trip_cost:.4%}\n")

    results = []
    for spec in specs:
        print(f"  running {spec.name} ({spec.kind}) ...", end=" ", flush=True)
        result = run_walk_forward(
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
        print("ok" if result.ok else f"FAILED ({result.error})")
        results.append(result)

    table = results_table(
        results, cost_per_turn=market_cfg.round_trip_cost, allow_short=market_cfg.allow_short
    )
    # Report the folds actually run, not the configured maximum: SCB's truncated
    # history yields fewer than `max_folds`, and a title that says otherwise is
    # a small lie in exactly the place this project is trying to be careful.
    actual_folds = max((len(r.folds) for r in results if r.ok), default=0)
    print()
    print(render_table(table, title=f"{symbol} — walk-forward, {actual_folds} folds"))

    manifest = RunManifest.create(
        "evaluate",
        {
            "symbol": symbol,
            "models": [s.name for s in specs],
            "eval": eval_cfg.__dict__,
            "market": market_cfg.__dict__,
        },
        data=data_fingerprint([symbol]),
        seed=eval_cfg.seed,
    )
    # A single-config run must not clobber a full sweep's table — the report
    # reads `evaluate-{symbol}.csv` and would silently shrink to one model.
    out = (
        RESULTS_DIR / f"evaluate-{symbol}.csv"
        if args.all
        else RESULTS_DIR / f"evaluate-{symbol}-{Path(args.config).stem}.csv"
    )
    table.to_csv(out, index=False)
    path = manifest.write()
    print(f"\nwrote {out}\nwrote {path}")
    return 0


def _resolve_model_specs(args) -> list[ModelSpec]:
    if args.all:
        specs = all_model_specs()
    elif args.config:
        specs = [ModelSpec.load(args.config)]
    else:
        raise SystemExit("give --config <path> or --all")
    return _with_naive(specs)


def _with_naive(specs: list[ModelSpec]) -> list[ModelSpec]:
    """R8: the naive baseline is on every results table, without being asked for.

    Not a convenience. Without it a reader has no way to tell whether a model
    did anything, which is precisely how the upstream repo shipped numbers that
    look like skill and are not.
    """
    if any(s.kind == "naive_lag" for s in specs):
        return specs
    baseline = MODEL_CONFIG_DIR / "00_naive_lag.yaml"
    naive = (
        ModelSpec.load(baseline)
        if baseline.exists()
        else ModelSpec(name="naive_lag", kind="naive_lag")
    )
    return [naive, *specs]


# ---------------------------------------------------------------- backtest


def cmd_backtest(args) -> int:
    from .agents import agent_results_table, render_agent_table, run_agent_walk_forward
    from .data import load
    from .eval import RunManifest, data_fingerprint

    ensure_dirs()
    eval_cfg = EvalConfig.load()
    market_spec = MarketConfigSpec.load()
    data_cfg = DataConfig.load()

    specs = _resolve_agent_specs(args)
    symbol = args.symbol.upper()
    df = load(symbol, policy=_policy_for(symbol, data_cfg))
    market_cfg = market_spec.build(symbol=symbol)

    folds = args.folds if args.folds else eval_cfg.max_folds
    splitter = eval_cfg.splitter()
    splitter = type(splitter)(
        train_window=splitter.train_window,
        test_window=splitter.test_window,
        step=splitter.step,
        expanding=splitter.expanding,
        max_folds=folds,
    )

    cap = market_cfg.participation_cap
    # The configured maximum is an upper bound; a truncated history yields fewer.
    available = len(splitter.split_frame(df))
    print(f"{symbol}: {len(df)} bars, {available} folds")
    print(
        f"frictions: lot {market_cfg.board_lot}, commission {market_spec.commission_rate:.3%}"
        f" + {market_spec.vat_rate:.0%} VAT, limit +/-{market_cfg.price_limit:.0%}, "
        f"short {'on' if market_cfg.allow_short else 'off'}, "
        f"participation cap {'none' if cap is None else f'{cap:.0%}'}\n"
    )

    results = []
    for spec in specs:
        print(f"  running {spec.name} ({spec.kind}) ...", end=" ", flush=True)
        result = run_agent_walk_forward(
            spec.build(),
            df,
            splitter=splitter,
            config=market_cfg,
            window=eval_cfg.window(),
            features=eval_cfg.features,
            symbol=symbol,
            seed=eval_cfg.seed,
        )
        print("ok" if result.ok else f"FAILED ({result.error})")
        results.append(result)

    table = agent_results_table(results)
    print()
    print(render_agent_table(table, title=f"{symbol} — agents, frictionless vs SET frictions"))

    manifest = RunManifest.create(
        "backtest",
        {
            "symbol": symbol,
            "agents": [s.name for s in specs],
            "eval": eval_cfg.__dict__,
            "market": market_spec.__dict__,
            "max_folds": folds,
            "folds_run": available,
        },
        data=data_fingerprint([symbol]),
        seed=eval_cfg.seed,
    )
    out = RESULTS_DIR / f"backtest-{symbol}.csv"
    table.to_csv(out, index=False)
    path = manifest.write()
    print(f"\nwrote {out}\nwrote {path}")
    return 0


def _resolve_agent_specs(args) -> list[AgentSpec]:
    if args.all:
        return all_agent_specs()
    if args.config:
        return [AgentSpec.load(args.config)]
    raise SystemExit("give --config <path> or --all")


# ---------------------------------------------------------------- report


def cmd_report(args) -> int:
    from .report import build_report

    ensure_dirs()
    symbols = _symbols(args.symbols)
    text = build_report(symbols, run_missing=not args.from_cache)
    out = RESULTS_DIR / "final-report.md"
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}")
    return 0


def cmd_status(args) -> int:
    from .agents import registered_kinds as agent_kinds
    from .data import cached_symbols, describe, read_meta
    from .models import registered_kinds as model_kinds

    print("cached symbols:")
    for sym in cached_symbols() or ["(none — run `fetch`)"]:
        meta = read_meta(sym)
        if meta:
            print(f"  {sym:6s} {meta.rows:5d} bars  {meta.start} .. {meta.end}  via {meta.source}")
            print("    " + describe(sym).replace("\n", "\n    "))
        else:
            print(f"  {sym}")
    print(f"\nmodel configs: {len(list(MODEL_CONFIG_DIR.glob('*.yaml')))}")
    print(f"agent configs: {len(list(AGENT_CONFIG_DIR.glob('*.yaml')))}")
    print(f"model kinds:   {', '.join(model_kinds())}")
    print(f"agent kinds:   {', '.join(agent_kinds())}")
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-retrofit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="fetch and cache daily bars, then quality-check them")
    p.add_argument("--symbols", default="")
    p.add_argument("--source", default="")
    p.add_argument("--start", default="")
    p.add_argument("--force-refresh", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("quality", help="quality report for cached symbols")
    p.add_argument("--symbols", required=True)
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("reconcile", help="compare the cache against a second source")
    p.add_argument("--symbols", required=True)
    p.add_argument("--against", default="yfinance")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("evaluate", help="walk-forward evaluation of forecasting models")
    p.add_argument("--config", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--symbol", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "backtest", help="walk-forward agent backtest, frictionless and with frictions"
    )
    p.add_argument("--config", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--symbol", required=True)
    p.add_argument("--folds", type=int, default=0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="comparison report across models x agents x tickers")
    p.add_argument("--symbols", default="KBANK,SCB,BAY")
    p.add_argument(
        "--from-cache", action="store_true", help="use results/*.csv instead of rerunning"
    )
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("status", help="what is cached and what is registered")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
