# Thai SET Retrofit of Stock-Prediction-Models (SCB, KBANK, BAY)

> **Standalone spec.** You have no other context from the conversation that produced this. Everything needed is here.
> Companion documents in this repo, worth reading before you start:
> `research/thai-set-retrofit-feasibility-2026-08-09.md` (data-source findings) and
> `reviews/scrutinize-thai-set-retrofit.md` (defects found in the upstream code).

## Goal

Rebuild the model and agent catalogue from `huseinzol05/Stock-Prediction-Models` as a modern PyTorch project that forecasts and backtests three Thai bank stocks — **SCB**, **KBANK**, **BAY** — using Stock Exchange of Thailand data, on an evaluation harness that does not leak and a backtest that charges real SET trading costs. The upstream repo supplies *architectures*; it supplies nothing trustworthy about *methodology*, and none of its evaluation code should be reproduced.

## Context: the two source repos

Both already exist on disk. Read them; do not modify them.

- **`../../external-repo/Stock-Prediction-Models-master/`** — the upstream repo. 62 notebooks: `deep-learning/` (18 sequence models + 2 bonus), `agent/` (23 trading agents), `stacking/` (2 ensembles), `simulation/` (Monte Carlo), `misc/` (exploratory), `realtime-agent/`, `free-agent/`, `stock-forecasting-js/` (TF.js port — **out of scope**), `dataset/` (17 US/FX CSVs). Written for TensorFlow 1.x: 41 of 62 notebooks use `tf.Session` / `tf.placeholder` / `tf.InteractiveSession`. No `requirements.txt` shipped; a reconstructed one is at `requirements-legacy-tf1.txt` in this project for reference only.
- **`../../external-repo/ThaiStock-main/`** — a small SET scraper. **Its endpoint is dead** (see Known-bad below). Use it only as documentation of SET's published column schema.

## Known-bad — verified, do not rediscover

1. **`ThaiStock` does not work.** `thaistock/stock.py:36` hardcodes `https://classic.settrade.com/`; that host returned an empty body and then timed out. It also caps history at 6 months, which is too short to train anything here. Do not build on it.
2. **Test-set leakage in every `deep-learning/` notebook.** `MinMaxScaler().fit()` is called on the full series *before* `df_train`/`df_test` are split (`deep-learning/1.lstm.ipynb`). Never reproduce this.
3. **The upstream accuracy metric is meaningless.** `1 - sqrt(mean(((real-predict)/real)**2))` on price levels scores a naive lag in the high nineties. Do not port it as a headline metric.
4. **Agents have no holdout.** `agent/6.evolution-strategy-agent.ipynb` — `get_reward()` (training) and `buy()` (reported result) both iterate the full `self.trend`. All upstream agent returns are in-sample.
5. **Live bug to fix, not port.** Same file, `get_reward()` buy branch: `starting_money -= close[t]` reads a module-level global instead of `self.trend[t]`. Confirmed in **2 of 23** agent notebooks — `agent/4.policy-gradient-agent.ipynb` and `agent/6.evolution-strategy-agent.ipynb`. Fix, do not port.
6. **`SCB` is two companies.** SCB delisted and SCBX listed 1:1 in April 2022, retaining the ticker. A continuous `SCB` series has a structural break at **2022-04-22**.
7. **`BAY` is ~72–76% MUFG-held**, so float and turnover are thin. Treat as the liquidity stress case.

## Requirements

**Data**

- R1. Fetch daily OHLCV for `SCB`, `KBANK`, `BAY` from **Settrade Open API** via the `settrade-v2` Python SDK, authenticating with `app_id` / `app_secret` / `broker_id` loaded from `.env` (never committed).
- R2. Before writing the adapter, confirm from the developer portal and **record the answers in `docs/settrade-api-notes.md`**: historical depth for daily bars, max bars per request, rate limits, and whether historical data is reachable with a sandbox key or requires a live broker key. These were not verifiable by plain HTTP fetch — the reference is a client-rendered SPA. If depth turns out to be insufficient, fall back to `yfinance` as primary and say so loudly in the README.
- R3. Cache every fetch to Parquet under `data/raw/{symbol}.parquet`; all downstream code reads the cache, never the network.
- R4. Reconcile against `yfinance` (`SCB.BK`, `KBANK.BK`, `BAY.BK`) and emit a data-quality report: missing sessions, zero-volume days, closes differing from the cross-check by more than one tick, and any single-day move beyond ±30% (which should be impossible under SET's ceiling/floor and therefore indicates bad data).
- R5. Handle the SCB break explicitly. Config flag `scb_history: truncate_2022 | full_with_changepoint`. Default `truncate_2022`. Never silently span the boundary.

**Evaluation harness — build this before any model**

- R6. Walk-forward splits only. Config: train window, test window, step. No single fixed tail split.
- R7. All fitting — scalers, feature statistics, everything — happens **inside** each fold on training data only, then transforms the test fold. Add a unit test that fails if a scaler sees test-fold data.
- R8. A `NaiveLag` baseline model is registered like any other model and appears in **every** results table automatically.
- R9. Metrics: directional accuracy, MASE vs. naive, RMSE on returns (not levels), and out-of-sample Sharpe after costs. If a model does not beat `NaiveLag` on MASE, the results table marks it as such in the output — no manual interpretation required.

**Market model**

- R10. A `SETMarket` friction layer that every agent trades through, enforcing: 100-share board lot, tick-size table snapping, commission + VAT charged on both legs, ±30% ceiling/floor, and no short selling without an explicit SBL flag.
- R11. Every agent backtest runs twice — frictionless and with frictions — and reports both. The gap is a headline result, not a footnote.
- R12. Optional turnover-participation cap (e.g. order ≤ X% of that day's volume), on by default for BAY.
- R13. Verify the tick table, board-lot exceptions, and current commission schedule against SET's published rules / your broker before trusting any number. The values in this spec are reconstructed, not verified.

**Models — parameterised families, not 62 files**

- R14. Cover all 18 upstream forecasting architectures and all 23 agents, but implement them as **parameterised families with one config per variant**, not one file per notebook. The upstream `deep-learning/` set is one train loop with `{cell: lstm|gru|rnn} × {bidirectional} × {paths: 1|2} × {decoder: none|seq2seq|vae|attention}`; the Q-learning agents (`5,7,8,9,10,11,12,13`) are one skeleton with `{double, duel, recurrent, curiosity}` flags. Reproduce the coverage, not the duplication.
- R15. Everything in PyTorch ≥2.3. No TensorFlow anywhere in the final tree.
- R16. Non-neural components keep their natural libraries: ARIMA via `statsmodels`, boosting via `xgboost`, classical ensembles via `scikit-learn`.
- R17. Seeded and reproducible: one seed config, deterministic dataloaders, run manifest (config + git SHA + data hash) written alongside every result.

## Proposed approach

Six phases. **Each phase ends with something runnable.** Do not start a phase before the previous one passes its checks.

**Phase 0 — scaffold.** `pyproject.toml` (or requirements.txt as provided), package layout below, `.env.example`, pre-commit with ruff + black, pytest wired up.

**Phase 1 — data layer.** Answer R2 first and write up the findings. Then `SettradeSource` and `YFinanceSource` behind one `PriceSource` protocol returning a canonical frame: `date, open, high, low, close, volume, symbol`. Parquet cache. Data-quality report. SCB break handling. **Ends with:** three cached Parquet files and a printed quality report for each.

**Phase 2 — evaluation harness.** Walk-forward splitter, fold-local preprocessing, metric suite, `NaiveLag`, results table renderer, the leakage unit test. **Ends with:** `NaiveLag` scored on all three tickers, walk-forward, with a real results table. This is the project's spine — everything later plugs into it.

**Phase 3 — SET market model.** `SETMarket` with lot/tick/fee/limit/short rules and a participation cap. Property tests: no order off-tick, no order off-lot, fees always non-negative, no fill outside the day's high/low. **Ends with:** a buy-and-hold "agent" backtested through the friction layer, with the frictionless-vs-friction gap printed.

**Phase 4 — forecasting families.** `RecurrentForecaster` covering the cell/direction/path variants, then the seq2seq/VAE/attention/CNN decoders, then ARIMA and the two stacking ensembles. One YAML config per upstream notebook, named after it (`configs/models/01_lstm.yaml` … `18_dilated_cnn_seq2seq.yaml`) so the mapping to upstream is traceable. **Ends with:** all 18 configs scored against `NaiveLag` on KBANK in one command.

**Phase 5 — agents.** Rule-based (turtle, moving-average, signal-rolling, ABCD) first — they are cheap and validate the market layer. Then the Q-family skeleton with its four flags, then policy-gradient/actor-critic, then evolution-strategy and neuro-evolution. All train on the training folds only and are reported on held-out folds only. **Ends with:** all 23 agents backtested on KBANK, frictionless and with frictions.

**Phase 6 — full universe and report.** Add SCB, then BAY with the participation cap. Produce a comparison report across models × agents × tickers, with the naive baseline pinned to the top of every table.

**Ticker order is deliberate:** KBANK (clean) → SCB (structural break) → BAY (thin float). Do not run all three from the start; each adds one distinct failure mode and you want them isolated.

**Explicit assumptions — override any of these if you disagree:**

- Python 3.11+, PyTorch ≥2.3, no GPU assumed (these models are small; CPU is fine).
- Daily bars only. Intraday is available from Settrade but is out of scope here.
- Configs are YAML; experiment tracking is plain JSON run manifests, not MLflow/W&B.
- No off-the-shelf backtest framework — `SETMarket` is small and SET-specific. If you find `vectorbt` or `backtesting.py` genuinely cleaner for R10–R12 after looking, that substitution is fine; note it in the README.
- Notebooks are outputs, not sources. All logic lives in the package; notebooks only call it.

## Files / modules affected

Everything is new; nothing in the two `external-repo/` folders is modified.

```
stock-retrofit/
  pyproject.toml | requirements.txt
  .env.example                      # SETTRADE_APP_ID, SETTRADE_APP_SECRET, SETTRADE_BROKER_ID
  docs/
    settrade-api-notes.md           # R2 findings — write this in Phase 1
    upstream-mapping.md             # 62 upstream notebooks -> config file, 1:1 traceability
  src/stock_retrofit/
    data/  sources.py cache.py quality.py corporate_actions.py
    eval/  splits.py preprocessing.py metrics.py baselines.py report.py
    market/ set_market.py rules.py          # lot, tick table, fees, ceiling/floor, SBL
    models/ base.py recurrent.py seq2seq.py attention.py conv.py classical.py stacking.py
    agents/ base.py rule_based.py qfamily.py policy_gradient.py actor_critic.py evolution.py
    cli.py                                   # fetch | evaluate | backtest | report
  configs/
    data.yaml  eval.yaml  market.yaml
    models/01_lstm.yaml ... 18_dilated_cnn_seq2seq.yaml
    agents/01_turtle.yaml ... 23_abcd.yaml
  tests/
    test_no_leakage.py              # required by R7
    test_market_rules.py            # required by R10
    test_data_quality.py
  data/raw/ data/processed/ results/
```

Speculative: `models/stacking.py` may end up thin if the two upstream stacking notebooks reduce to composing existing pieces. `agents/evolution.py` may need its own runner — evolution strategies do not fit the gradient-based training loop and may warrant a separate entry point rather than being forced into `agents/base.py`.

## Acceptance criteria

1. `pytest` green, including `test_no_leakage.py` — which must be written so that deliberately fitting a scaler on the full series makes it **fail**. Verify that by temporarily introducing the bug.
2. `python -m stock_retrofit.cli fetch --symbols SCB,KBANK,BAY` produces three Parquet files and a quality report with zero unexplained gaps.
3. `docs/settrade-api-notes.md` answers all four R2 questions with a source link or a portal screenshot reference.
4. `python -m stock_retrofit.cli evaluate --config configs/models/01_lstm.yaml --symbol KBANK` prints a walk-forward table with `NaiveLag` on the same table.
5. All 18 model configs and all 23 agent configs run end-to-end on KBANK without manual edits.
6. `docs/upstream-mapping.md` maps each of the 62 upstream notebooks to its config or records an explicit "not ported, because —".
7. Every agent result appears twice, frictionless and with SET frictions, from `SETMarket`.
8. `grep -ri "tensorflow" src/ configs/` returns nothing.
9. The final report states plainly how many models beat `NaiveLag` out-of-sample after costs — **including if the answer is zero.** That is a legitimate and likely result, and the report must be able to say it.

## Non-goals

- `stock-forecasting-js/` — the TensorFlow.js browser port. Not in scope.
- Live order execution. `SETMarket` is a simulator; no code path places a real order. The Settlement credentials are for market data only.
- Intraday / tick data.
- Sentiment models (`deep-learning/sentiment-consensus.ipynb`, `dataset/BTC-sentiment.csv`) — no Thai-language sentiment source is specified, so skip rather than fake it.
- Reproducing upstream's reported numbers. They are in-sample and leaked; matching them would be a failure, not a success.

## Open decision points for Kim

1. **Settrade historical depth (R2).** If daily history turns out to be short, do you want `yfinance` promoted to primary, or the study scoped to the shorter window?
2. **SCB history.** Default is truncate at 2022-04-22 (~4 years). Say if you would rather model the full series with a flagged changepoint.
3. **Short selling.** Assumed disabled by default (no SBL). Several upstream agents assume symmetric long/short; with shorting off they become long-only and their character changes. Confirm that is what you want.
4. **Effort.** Phases 4–5 are the bulk. If the goal is a working comparison rather than exhaustive coverage, stopping after Phase 4 plus the rule-based agents gives most of the insight for a fraction of the work.
