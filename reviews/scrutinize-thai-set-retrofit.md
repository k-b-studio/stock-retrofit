# Scrutinize — retrofit plan for Thai SET (SCB, KBANK, BAY)

Reviewed cold, as an outsider. Artifact under review: the stated plan to retrofit `huseinzol05/Stock-Prediction-Models` for SCB/KBANK/BAY, sourcing data via Settrade (`ThaiStock-main`), porting **all 62 notebooks** to **PyTorch**.

---

## 1. Intent

**Goal in one sentence:** run the model and agent catalogue from a 2018-era TensorFlow-1 stock-prediction repo against three Thai bank stocks, on a modern PyTorch stack, using Thai market data.

That goal is coherent. The plan for reaching it is not, for three reasons, and one of them is load-bearing enough to change the shape of the project.

### Simpler alternative — take it seriously before writing code

**Do not port 62 notebooks. Port one pipeline and 62 configs.**

The 62 notebooks are not 62 ideas. `deep-learning/` holds 18 files that are the same train loop with a different cell type and direction flag — LSTM, LSTM-Bidirectional, LSTM-2-Path, GRU, GRU-Bidirectional, GRU-2-Path, Vanilla, and so on. `agent/` holds 23 files where the Q-learning family (`5,7,8,9,10,11,12,13`) shares one replay-buffer skeleton with a swapped head. Ported one-for-one, you get 62 near-identical PyTorch files, 62 places for the same bug, and no ability to compare models fairly because each notebook has its own hyperparameters baked in.

The equivalent-outcome version:

- one `models/` package where architecture = `{cell: lstm|gru|rnn, bidirectional: bool, paths: 1|2, decoder: none|seq2seq|vae|attention}`,
- one `agents/` package where the Q-family is `{double: bool, duel: bool, recurrent: bool, curiosity: bool}`,
- one training entry point, one evaluation harness, 62 YAML configs.

Same coverage. Roughly a fifth of the code. And it is the only version where "compare all 62 on KBANK" is a single command rather than 62 manual runs. **If you keep one thing from this review, keep this.**

**Second alternative worth one breath:** doing almost nothing. If the actual objective is "forecast SCB/KBANK/BAY well," the honest prior is that a naive-lag baseline plus a well-specified gradient-boosted model on engineered features beats most of this catalogue out-of-sample, at a fraction of the effort. The catalogue is worth building as a *comparison set* — it is not worth building as *the answer*. Budget accordingly.

---

## 2. Trace — where the plan meets reality

### 2.1 Data layer — the plan's entry point does not exist

`ThaiStock-main/thaistock/stock.py:36` sets `self.base_url = 'https://classic.settrade.com/'`. Every method — `historical()` (line 40), `current()` (line 163) — builds a URL from it and parses with BeautifulSoup against `table.table.table-info.table-hover` (line 47).

Traced against reality this session:

- `GET classic.settrade.com/C04_02_stock_historical_p1.jsp?txtSymbol=SCB&...` → empty body.
- `GET classic.settrade.com/` → timeout after 180 s.

The plan's first step therefore fails at step one. Note also that `historical()` wraps its per-row parse in `try/except: pass` (lines 61–62) — if the endpoint ever returns *changed* HTML rather than nothing, this returns an empty list rather than an error, and the failure surfaces later as "the model trained on zero rows."

Second, structural: the ThaiStock README caps history at **6 months**. `deep-learning/1.lstm.ipynb` trains 300 epochs with a 5-step window and a 30-day holdout. ~120 daily bars cannot support that. Even a working ThaiStock would not have been the right dependency.

### 2.2 The evaluation harness you would be porting is broken

**Forecasting side.** In `deep-learning/1.lstm.ipynb` the cell order is: fit scaler on all rows → transform all rows → *then* `df_train = df_log.iloc[:-test_size]`. The test window's min and max are inside the training representation. Every deep-learning notebook in this repo follows that order.

**Metric side.** `calculate_accuracy(real, predict) = 1 - sqrt(mean(((real-predict)/real)**2))` is 1−RMSPE on *price levels*. On a near-random-walk series, "predict today's price for tomorrow" scores in the high nineties. This is why the README's accuracy figures look good. They are not measuring skill.

**Agent side.** In `agent/6.evolution-strategy-agent.ipynb`, `Agent.get_reward()` — the training objective — loops over `self.trend`, the complete series. `Agent.buy()` — the function whose output becomes the README's equity curve — loops over the *same* `self.trend`. There is no holdout anywhere in the agent notebooks. Every published agent return is in-sample.

**Surprise found while tracing, worth fixing during the port:** inside `get_reward`, the buy branch is `starting_money -= close[t]` while the sell branch is `starting_money += self.trend[t]`. The buy path reads a module-level global; the sell path reads the instance attribute. It is invisible today only because the global and the attribute happen to hold the same list. Reproduce that faithfully in PyTorch and you will inherit a latent bug that detonates the first time you run two tickers in one process.

### 2.3 The universe assumes three comparable instruments; it has two and a half

- **SCB** retained its ticker through the April 2022 SCBX restructuring, but the issuer changed from a bank to a fintech holding company. A 2015–2026 `SCB` series is two companies glued at 2022-04-22. Any model trained across that boundary is learning a merger artifact.
- **BAY** is ~72–76% held by MUFG. Frictionless backtests on a thin float are the most flattering backtests there are, because the assumption they violate — that you can transact at the close in any size — is exactly the assumption that fails when float is small.
- **KBANK** is clean.

### 2.4 Zero market frictions

`README.md:118` — "This agent only able to buy or sell 1 unit per transaction." On SET the board lot is 100 shares, prices must sit on a tick-table increment, commission plus VAT applies both ways, moves are bounded by ±30% ceiling/floor, and retail shorting needs SBL. A one-share, zero-cost, freely-shortable agent is not a model of SET. Porting the agents without a friction layer produces numbers that cannot be acted on and, worse, look like they can.

---

## 3. Findings

### Blocker — the specified data source is non-functional

**Why it matters:** every downstream task is blocked; discovering this after the PyTorch port would waste the entire port.
**Evidence:** `stock.py:36` hardcodes `classic.settrade.com`; direct requests returned empty body and then a 180 s timeout.
**Change:** build the data layer against **Settrade Open API** (`settrade-v2` SDK, `app_id`/`app_secret`/`broker_id`). Before writing the adapter, confirm in the developer portal: historical depth for daily bars, max bars per request, rate limits, and whether historical data is available with a sandbox key or only a live broker key — the API reference is a client-rendered SPA and none of this was verifiable by fetch. Keep `yfinance` (`SCB.BK`, `KBANK.BK`, `BAY.BK`) as an independent reconciliation series, not as the primary.

### Blocker — porting the evaluation harness reproduces the leakage

**Why it matters:** a faithful PyTorch port of a leaking pipeline is a faster leaking pipeline. Every number it produces is unusable, and it will look successful.
**Evidence:** scaler fit precedes the train/test split in `deep-learning/1.lstm.ipynb`; `get_reward`/`buy` share `self.trend` in `agent/6.evolution-strategy-agent.ipynb`.
**Change:** write the evaluation harness **first**, before porting any model. Fit scalers inside each training fold only. Use walk-forward splits, not a single 30-day tail. Give agents a genuine holdout. Then port models into that harness. Treat the notebooks as *specifications of architecture*, never as specifications of methodology.

### Blocker — the reported metric cannot distinguish a model from a lag

**Why it matters:** without a baseline you will ship a model that is worse than `y_hat = y_lag1` and not know it.
**Evidence:** `calculate_accuracy` is 1−RMSPE on levels; the notebooks report it as the headline number.
**Change:** make the naive-lag baseline a first-class model in the harness and print it on every results table. Report directional accuracy, MASE vs. naive, and out-of-sample Sharpe after costs. A model that does not beat naive gets marked as such in the output, automatically.

### Major — 62 one-for-one ports is the wrong unit of work

**Why it matters:** ~5× the code, 62 copies of every bug, and no fair cross-model comparison because hyperparameters are baked per-notebook.
**Evidence:** `deep-learning/` is 18 variants of one loop; `agent/5,7,8,9,10,11,12,13` are one Q-skeleton with swapped heads.
**Change:** one parameterised implementation per family + a config per variant. Coverage is preserved; the count of *files* drops, not the count of *models*.

### Major — no SET friction model

**Why it matters:** frictionless returns on a 100-share-lot market with real commission are not merely optimistic, they invert sign at realistic turnover.
**Evidence:** `README.md:118`, one-unit transactions; no fee/slippage term anywhere in the agent code.
**Change:** a `SETMarket` layer enforcing board lot, tick snapping, commission+VAT both ways, ceiling/floor, and no-short-without-SBL. Every agent trades through it. Run each agent frictionless *and* with frictions and report both — the gap is itself the finding.

### Major — SCB is a spliced series

**Why it matters:** a structural break at 2022-04-22 will be learned as signal.
**Evidence:** SCB delisted and SCBX listed 1:1 in April 2022, ticker retained (SCBX/SCB first-party announcements).
**Change:** decide explicitly — truncate at 2022-04-22, or model the full series with the break flagged as a known changepoint and excluded from evaluation windows. Record the decision in the data layer, not in someone's memory.

### Minor — BAY is the wrong stock to validate on

**Why it matters:** ~72–76% MUFG ownership means thin float; it is the ticker most likely to produce a backtest that cannot be executed.
**Change:** bring KBANK up end-to-end first. Add SCB. Add BAY last, labelled as the liquidity stress case, with a turnover-participation cap in the backtest.

### Nit — `starting_money -= close[t]` reads a global

**Why it matters:** silent today, wrong the moment two tickers share a process.
**Evidence:** `agent/6.evolution-strategy-agent.ipynb`, `Agent.get_reward()`, buy branch vs. sell branch. Grepped the full `agent/` directory: the same pattern appears in **2 of 23** notebooks — `agent/4.policy-gradient-agent.ipynb` and `agent/6.evolution-strategy-agent.ipynb`.
**Change:** `self.trend[t]` in both.

---

## Verdict

**Rework** — the goal is sound, the plan is not yet executable.

Single biggest reason: **the plan's data source is dead and its evaluation harness leaks**, so the two things the whole project rests on are both broken before a line of PyTorch is written. Fix the harness and the data layer first, restructure the port from 62 files into ~6 parameterised families, and this becomes a good project. Port first and you will have spent the effort producing numbers you cannot trust.

---

*Reviewed: data layer (`ThaiStock-main/thaistock/stock.py` in full), `deep-learning/1.lstm.ipynb` (scaling → split → train → forecast path), `agent/6.evolution-strategy-agent.ipynb` (`get_reward` / `buy`), `README.md` results claims, import surface across all 62 notebooks. Not reviewed: `stock-forecasting-js/`, `realtime-agent/`, `stacking/` internals, `simulation/` internals — the findings above may or may not extend to them.*
