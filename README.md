# stock-retrofit

The [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models)
catalogue — 18 forecasting architectures and 23 trading agents — rebuilt in PyTorch for three Thai
SET bank shares (**KBANK**, **SCB**, **BAY**), on an evaluation harness that does not leak and a
backtest that charges SET trading costs.

The upstream repository supplies *architectures*. It supplies nothing trustworthy about
*methodology*, and none of its evaluation code is reproduced here.

---

## Read this first

### 1. yfinance is the primary data source, and that is a compromise

Spec R1 wants Settrade Open API as primary. Settrade is credential-gated behind a participating
brokerage, no credentials exist in this environment, and none can be self-issued — so **the four
R2 questions (historical depth, bars per request, rate limits, sandbox vs. live key) are
unanswered, and acceptance criterion 3 is not met.** Spec R2 provides for exactly this fallback
and it has been taken. `docs/settrade-api-notes.md` records what was tried and what remains open.

What it costs: Yahoo is a redistributor rather than the exchange, and reconciliation (R4/R10)
degrades from "two independent sources" to "cache vs. a fresh vendor pull" — which catches
staleness and revisions but cannot catch an error Yahoo makes consistently. `SettradeSource` is
written and wired in; promoting it is a one-line config change once credentials exist.

### 2. Trading cost parameters are reconstructed, not verified

Spec R13 requires checking the tick table, board-lot exceptions and commission schedule against
SET's published rules before trusting any backtest number. **That has not been done.** The values
in `configs/market.yaml` are flagged as reconstructed at every point they appear. The friction
gap is directionally right; its magnitude is an estimate.

### 3. The fundamentals axis is deferred, not forgotten

`specs/thai-market-data-layer.md` specifies a second axis — quarterly/yearly fundamentals via
`thaifin`, behind a point-in-time gate. It is **not built**. That spec calls the axis "strictly
optional to the parent project" and says it "must never block" the price axis, and the modelling
work here is price-and-technical throughout. The price axis of that spec *is* implemented, inside
`src/stock_retrofit/data/` rather than as a sibling package: `PriceSource` protocol and canonical
frame, Parquet cache with metadata sidecars, fail-loud quality gates, `reconcile()`, the
corporate-actions registry, and BAY's liquidity metadata.

If you build the fundamentals axis later, R20–R25 of that spec are the ones that matter — a
look-ahead through fundamentals yields *plausible* outperformance, which makes it far harder to
spot than the scaler bug this project exists to eliminate.

### 4. Logic lives in the package; notebooks are outputs

`notebooks/01_data.ipynb` … `06_report.ipynb` are the runnable deliverable for each phase, and
each is a thin call into `src/stock_retrofit/`. The package stays `.py` because the acceptance
criteria require `pytest` and `python -m stock_retrofit.cli`; the per-phase deliverables are
notebooks.

---

## Figures

```bash
python scripts/make_figures.py     # -> results/figures/*.png
```

| | |
|---|---|
| `01_mase_vs_naive.png` | every model against the naive-lag line, one panel per ticker |
| `02_directional_accuracy.png` | directional accuracy against the 50% coin flip |
| `03_friction_gap.png` | dumbbell: frictionless return → return after SET costs, per agent |
| `04_forecast_reality.png` | the best-scoring model up close — forecasts vs what actually happened |

## The headline result

**No model in the catalogue beats a naive lag out-of-sample on these tickers, after costs.**

The upstream README reports accuracies in the high nineties for the same architectures. Both are
true at once, and the reason is methodological:

| | upstream | here |
|---|---|---|
| scaler | `MinMaxScaler().fit()` on the full series, *then* split | fit inside each fold, guarded at runtime |
| split | one fixed 30-day tail | walk-forward, 8 folds |
| target | price levels | next-day returns |
| headline metric | `1 − RMSPE` on levels | MASE vs. naive, directional accuracy, Sharpe after costs |
| baseline | none | `NaiveLag` on every table, automatically |
| agent evaluation | in-sample — `get_reward()` and `buy()` share `self.trend` | held-out folds only |
| trading costs | none; 1 share per transaction | board lot, tick table, commission + VAT, ±30% limits, no naked shorting |

A naive lag scores in the high nineties on upstream's metric too — `metrics.upstream_accuracy_do_not_use`
and its test demonstrate this. Those numbers never measured skill, so failing to reproduce them is
the correct outcome, not a shortfall.

---

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env                     # optional; only needed for Settrade

python -m stock_retrofit.cli fetch --symbols KBANK,SCB,BAY
python -m stock_retrofit.cli evaluate --config configs/models/01_lstm.yaml --symbol KBANK
python -m stock_retrofit.cli evaluate --all --symbol KBANK      # ~2 min
python -m stock_retrofit.cli backtest --all --symbol KBANK      # ~20 min
python -m stock_retrofit.cli report --symbols KBANK,SCB,BAY
pytest
```

`fetch` is the only command that touches the network. Everything else reads the Parquet cache.

---

## How it is put together

```
src/stock_retrofit/
  data/      sources · cache · quality · repair · corporate_actions · loader
  eval/      splits · preprocessing · leakage · metrics · runner · report · manifest
  market/    rules (tick/lot/fees/limits) · set_market
  models/    baselines · recurrent · seq2seq · attention · conv · classical · stacking
  agents/    rule_based · qfamily · policy_gradient · evolution · env · runner
  cli.py     fetch | quality | reconcile | evaluate | backtest | report | status
configs/     data · eval · market · models/*.yaml (23) · agents/*.yaml (24)
notebooks/   01_data … 06_report — one runnable deliverable per phase
docs/        settrade-api-notes.md · upstream-mapping.md
```

**62 upstream notebooks → 13 model kinds and 11 agent kinds, driven by 47 configs.** The upstream
`deep-learning/` set is one train loop with `{cell} × {bidirectional} × {paths} × {decoder}`, so
its 18 notebooks become 4 registered kinds and 18 YAML files; the 11 Q-learning agents become one
skeleton with `{double, duel, recurrent, curiosity}`. Same coverage, a fraction of the code, and a
fair cross-model comparison becomes one command. `docs/upstream-mapping.md` maps all 62 notebooks,
including an explicit reason for each of the 21 not ported.

Run `python -m stock_retrofit.cli status` to list what is cached and what is registered.

### The leakage guard

Intending to fit inside folds is not enough — the upstream bug is invisible in the output. So
every statistic that learns anything registers the rows it saw, and the guard raises if any
belongs to the test block:

```python
with leakage_guard(test_indices=range(900, 960)):
    register_fit("MinMaxScaler(full series)", np.arange(0, 960))
# LeakageError: 'MinMaxScaler(full series)' was fit on 60 test-block row(s)
```

`tests/test_no_leakage.py` asserts this, and the guard was verified by reintroducing the upstream
bug into `prepare_fold` and confirming the suite goes red. It also caught a genuine one-day label
leak in this project's own code during development: the last training row's target is the return
realised on the first test bar.

### Metrics

- **MASE** — MAE(model) / MAE(naive lag) on returns. Below 1.00 beats the lag. Computed on the
  same out-of-sample block for both, so numerator and denominator see identical data.
- **Directional accuracy** — over rows where a call was actually made. The naive lag abstains, so
  its accuracy is undefined rather than 0%.
- **Sharpe after costs** — annualised, charged on position changes.

### Data quality

Structural violations **raise**; advisories are reported. Repair is a separate, audited stage — the
gate still raises on anything unrepaired, and every edit lands in `data/raw/{symbol}.meta.json`.

Real findings on this data: 10 vendor bars across the three names have a `close` outside
`[low, high]` by one to three ticks; SCB shows a 4-session halt around 2022-04-21…26 and a +39.9%
move on the first session back, which the gate recognises as spanning a registered break rather
than breaching the ±30% limit.

**`SCB.BK` on Yahoo begins 2022-04-20** — the vendor already carries SCBX only. So
`truncate_at_break` is satisfied trivially and `full_with_changepoint` cannot be populated from
this source. Both policies are implemented and tested anyway; the caveat belongs in code.

---

## Decisions taken on Kim's open questions

The spec leaves six decisions open. Each had a stated default and the default was taken; all are
one config edit to reverse.

| Question | Decision | Where |
|---|---|---|
| Settrade depth short → promote yfinance or scope down? | **yfinance promoted**, full depth kept. Access was absent rather than shallow. | `configs/data.yaml` |
| SCB history | **`truncate_at_break`** (~4 years). Moot — the source has no earlier data. | `configs/data.yaml` |
| Short selling | **Disabled** (no SBL). Long-only agents. | `configs/market.yaml` |
| Effort — stop after Phase 4? | **Full scope built**, all 6 phases. Compute was not the constraint. | — |
| Backtest framework | **Own `SETMarket`.** `vectorbt`/`backtesting.py` model neither the board lot nor the tick table natively. | `market/` |
| Reporting lag (fundamentals) | **Not reached** — axis deferred. | — |

---

## Limits worth stating

- **8 folds × 60 days = 480 out-of-sample observations per ticker.** Enough to reject a model, not
  enough to crown one. Raise `max_folds` in `configs/eval.yaml` to use the full history.
- **Fills are at the close.** Intraday execution is out of scope, and `slippage_ticks` defaults to
  0 so the friction gap reflects the specified rules rather than a modelling choice of ours.
- **Agents train against a fast numpy environment**, then evaluate through `SETMarket`. Training
  may use any objective; only evaluation is a claim.
- **Features are deliberately plain** — five causal blocks shared by every model, so the comparison
  is between architectures rather than between feature sets.
- **BAY's participation cap is on by default** (5% of session volume) and its results should be
  read as the liquidity stress case, not as one third of an equally-weighted table. Under a cap
  the frictionless column also drops the liquidity constraint, so part of BAY's friction gap is a
  statement about its float rather than about commission. The cap is also why `buy_and_hold`
  accumulates across sessions rather than issuing a single order — a baseline that stops after one
  trimmed fill sits mostly in cash and loses to anything that trades, which measures the handicap
  rather than the strategy.

## Source material

- `specs/thai-set-retrofit.md`, `specs/thai-market-data-layer.md` — the specifications
- `research/thai-set-retrofit-feasibility-2026-08-09.md` — data-source findings
- `research/google-finance-pipeline-feasibility-2026-08-09.md` — why Google Sheets was rejected
- `reviews/scrutinize-thai-set-retrofit.md` — the defects found in the upstream code
- `requirements-legacy-tf1.txt` — the upstream stack, for reference only; do not install it
