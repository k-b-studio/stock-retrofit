# Retrofitting huseinzol05/Stock-Prediction-Models for Thai SET equities (SCB, KBANK, BAY)

**Research question:** What has to be true — in data access, market microstructure, and code — for the huseinzol05 Stock-Prediction-Models repo to produce trustworthy forecasts and backtests on SCB, KBANK, and BAY?

**Mode:** problem-solving / feasibility investigation
**Date:** 2026-08-09
**Citation style:** APA 7

---

## Summary of findings

1. **The chosen data source is broken.** `ThaiStock-main` scrapes `classic.settrade.com`, which no longer serves content. A replacement data layer is required before anything else in this project can run.
2. **Settrade Open API is the right official replacement**, but it is credential-gated (broker `app_id`/`app_secret`) and its sandbox does not remove the broker relationship for live use.
3. **SCB is not a continuous series.** The ticker survived the 2022 SCBX restructuring, but the issuer behind it changed. Any pre-April-2022 SCB history is a different legal entity.
4. **BAY is structurally illiquid** (MUFG holds ~72–76%), which makes it the weakest of the three for any trading-agent evaluation.
5. **The repo's headline results do not survive inspection** — the forecasting notebooks leak the test window into the scaler, and the agent notebooks have no train/test split at all. Porting them faithfully would port the flaws.

---

## 1. Data access for SET equities

### 1.1 The `ThaiStock` path is dead

`ThaiStock-main/thaistock/stock.py` builds every request against `self.base_url = 'https://classic.settrade.com/'` (line 36) and parses the response with BeautifulSoup against a table selector `table.table.table-info.table-hover` (line 47).

Direct verification this session:

- `GET https://classic.settrade.com/C04_02_stock_historical_p1.jsp?txtSymbol=SCB&selectPage=2&max=30&offset=0` → **empty response body**.
- `GET https://classic.settrade.com/` → **request timed out** (180 s).

This is consistent with community reports that the classic Settrade and classic SET portals were retired (Pantip, n.d. — *tier 3, forum post*; used only as corroboration, the primary evidence is the two failed requests above).

Two further limits would disqualify the library even if the endpoint returned: the README states history is capped at **6 months** (ThaiStock, n.d.), and the parser is HTML-shape-dependent with a bare `except: pass` (stock.py, lines 61–62) that silently discards malformed rows. Six months of daily bars is ~120 observations — not enough to train any of the sequence models in the upstream repo, which use a 5-step window and 300 epochs on multi-year series.

**Conclusion:** treat `ThaiStock-main` as a schema reference (its Thai column ordering is useful documentation of what SET publishes), not as a runtime dependency.

### 1.2 Settrade Open API — the official route

Settrade Open API is SET's sanctioned programmatic interface, covering both equities and derivatives, with real-time and historical market data and an OHLCV/candlestick endpoint supporting intervals down to one minute (Settrade, n.d.-a; Settrade, n.d.-b).

Access mechanics that shape the build:

- Authentication is via `app_id` / `app_secret` plus a `broker_id`, instantiated through `settrade.openapi.Investor(...)` (Settrade, n.d.-c).
- A **sandbox** exists for algorithm testing, registered from the developer portal; but executing against the live market still requires an account with a participating brokerage (Chawannakul, 2021 — *tier 2, practitioner writeup*; corroborated by Pi Securities, n.d.).
- An official SDK example repository is published by Settrade covering Python, Excel/VB, and Amibroker (Settrade, n.d.-d).

**Coverage gap — flagged honestly:** the API reference at `developer.settrade.com/open-api/api-reference/` is a client-rendered single-page app and returned only metadata to a plain fetch. **Exact historical-depth limits, rate limits, and per-call bar caps for the candlestick endpoint were not verified this session.** Before committing the data layer, confirm directly in the portal: (a) how far back daily bars go, (b) max bars per request, (c) request rate limits, (d) whether historical data is available in sandbox or only with a live broker key. This is the single largest open risk in the plan.

### 1.3 Cross-check source

`yfinance` exposes `SCB.BK`, `KBANK.BK`, and `BAY.BK`, and Yahoo Finance maintains a quote page for Bank of Ayudhya under `BAY.BK` (Yahoo Finance, n.d.). It is not authoritative for SET corporate actions but is valuable as an **independent reconciliation series** — if Settlement-sourced and Yahoo-sourced closes diverge by more than a tick on a given day, that is a data-quality alarm worth raising rather than silently averaging.

---

## 2. The three tickers are not equivalent instruments

### 2.1 SCB — ticker continuity without entity continuity

Between 2 March and 18 April 2022, SCB shareholders were offered a 1:1 swap of SCB ordinary/preferred shares for newly issued SCBX shares; over 99% accepted, SCB was delisted, and SCB X Public Company Limited listed in its place — **retaining the `SCB` ticker** (SCBX, 2022; SCB, 2022). SCBX registered its new share capital on 22 April 2022 and became the group holding company.

Consequence for modelling: a price series labelled `SCB` spanning 2015–2026 silently concatenates *Siam Commercial Bank* (a bank) with *SCB X* (a fintech-oriented holding company). The fundamental drivers differ, and the April 2022 boundary is a structural break, not a regime the model should be asked to learn through. **Either** truncate SCB history at 2022-04-22 and accept ~4 years of data, **or** model the full series and treat the break as a known changepoint — but do not do it accidentally.

### 2.2 BAY — thin float

MUFG acquired a majority stake of roughly 72% in 2013 and Bank of Ayudhya operates as an MUFG Bank subsidiary (Wikipedia, n.d., citing the 2013 transaction — *tertiary source, used for orientation*; corroborated by Krungsri's own group overview, Krungsri, n.d.). BAY remains listed on SET under `BAY` (SET, n.d.).

Consequence: with the overwhelming majority of shares held by a strategic parent, free float and daily turnover are small relative to SCB and KBANK. For a **forecasting** task this mainly means noisier prints. For a **trading-agent** task it is more serious — a backtest that assumes the agent can transact at the observed close is assuming liquidity that may not be there. BAY should be carried as a deliberately hard case, not as one third of an equally-weighted result table.

**I did not retrieve current free-float percentages or average daily turnover for the three names this session.** Pull those from SET's company pages before finalising universe selection.

### 2.3 KBANK

No structural discontinuity or float issue identified. KBANK is the cleanest of the three and is the sensible first ticker to bring end-to-end.

---

## 3. Thai market frictions the upstream repo does not model

The upstream backtests are US-equity-shaped and frictionless. SET has rules that materially change agent economics and must be encoded in any honest backtest:

- **Board lot of 100 shares** for most listings — the upstream agents transact **1 unit per transaction** (README.md, line 118), which is not a tradeable quantity on SET.
- **Tick-size table** — the minimum price increment is a step function of price level, so fills must be snapped to a valid tick.
- **Commission plus VAT** — real round-trip cost is on the order of tens of basis points; the upstream agents charge zero.
- **Ceiling/floor limits (±30%)** and intraday halts.
- **No retail short selling without SBL** — several agents assume symmetric long/short freedom.

None of these are exotic; all of them turn a nominally profitable frictionless agent into a losing one at realistic turnover. **These parameters were reconstructed from general market knowledge and were not verified against SET's current published rulebook this session** — verify tick table, board lot exceptions, and current commission schedule against SET/your broker before trusting any backtest number.

---

## 4. What the upstream repo is actually worth porting

Direct inspection of the code, not the README.

### 4.1 Confirmed defect — test-set leakage in the forecasting models

In `deep-learning/1.lstm.ipynb` the scaler is fit on the entire series **before** the split:

```python
minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))   # fit on ALL rows
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))
...
test_size = 30
df_train = df_log.iloc[:-test_size]     # split happens AFTER scaling
df_test  = df_log.iloc[-test_size:]
```

The min and max of the test window are therefore baked into the training representation. This pattern repeats across the `deep-learning/` notebooks.

### 4.2 Confirmed defect — the accuracy metric flatters a random walk

The notebooks report `calculate_accuracy` = `1 - sqrt(mean(((real - predict)/real)^2))`, i.e. one minus RMSPE **on price levels**. On a series that is close to a random walk, predicting "tomorrow ≈ today" scores in the high nineties. The headline accuracy figures are therefore not evidence of predictive skill. A retrofit must report **directional accuracy, MASE against a naive-lag baseline, and out-of-sample Sharpe** instead — and must show the naive baseline alongside every model.

### 4.3 Confirmed defect — the agents have no out-of-sample evaluation

In `agent/6.evolution-strategy-agent.ipynb`, `Agent.get_reward()` (the training objective) iterates `range(0, len(self.trend) - 1, self.skip)` over the full series, and `Agent.buy()` (the reported result) iterates the **same** `self.trend`. There is no held-out period anywhere. Every agent equity curve in the README is in-sample.

The same file contains a real bug: inside `get_reward`, the buy branch does `starting_money -= close[t]` — referencing a module-level global `close` rather than `self.trend[t]`, which the sell branch correctly uses. It only appears to work because the global happens to hold the same series. A grep across `agent/` finds this pattern in 2 of the 23 notebooks (`agent/4.policy-gradient-agent.ipynb`, `agent/6.evolution-strategy-agent.ipynb`).

### 4.4 What does transfer

- The **model catalogue** — 18 sequence architectures and 23 agent strategies — is a genuinely useful menu of things to try, independent of the quality of its implementations.
- The **stacking ideas** (autoencoder → RNN → ARIMA → XGBoost; classical ensemble stack) are sound in principle.
- The **Monte Carlo simulation** notebooks are self-contained and least affected by the leakage problem.
- The **notebook-per-model layout** is a liability at 62 files; a retrofit should collapse the near-duplicate variants into one parameterised implementation.

---

## 5. Where sources disagree / what remains unverified

- **Settrade historical depth and rate limits** — unverified (client-rendered docs). Highest-priority gap.
- **Current free float and turnover for SCB / KBANK / BAY** — unverified.
- **Current SET commission, tick table, board-lot exceptions** — unverified against the official rulebook.
- **Whether classic.settrade.com is permanently retired or transiently down** — two failed requests plus one forum thread. Strong signal, not proof. Either way the 6-month history cap makes it unusable.
- No source disagreement was found on the SCBX share swap; SCB's and SCBX's own statements agree on the 1:1 ratio, the >99% acceptance, and the ticker retention.

---

## Annotated source list

| Source | Tier | Note | Takeaway |
|---|---|---|---|
| `ThaiStock-main/thaistock/stock.py` (local) | 1 — primary | The actual code under evaluation | Hardcoded to classic.settrade.com; 6-month cap; fragile HTML parsing |
| Direct HTTP requests to classic.settrade.com | 1 — primary observation | Made this session | Empty body, then timeout — endpoint not serving |
| `Stock-Prediction-Models-master/` notebooks (local) | 1 — primary | The code being retrofitted | Leakage, flattering metric, in-sample agents, `close[t]` bug |
| Settrade Open API portal | 1 — primary/official | Vendor documentation | Official SET programmatic access; equity + derivatives; sandbox exists |
| Settrade SDK example repo (GitHub) | 1 — primary/official | Vendor-published | Python/Excel/Amibroker SDKs confirmed |
| Settrade Python SDK reference (`Investor(app_id, app_secret, broker_id, ...)`) | 1 — primary/official | Vendor docs | Confirms credential + broker_id auth model |
| SCBX / SCB corporate announcements | 1 — primary | First-party issuer statements | 1:1 swap, >99% acceptance, SCB delisted, SCBX listed, ticker retained |
| Pi Securities help centre — Settrade Open API setup | 2 — credible secondary | Broker documentation | Corroborates broker-mediated credential issuance |
| Chawannakul (Medium) — trading Thai market via Settrade Open API | 2 — practitioner | Named practitioner writeup | Sandbox tests algorithms; live execution still needs a broker |
| Krungsri group overview | 1 — primary (first-party) | Company's own site | MUFG subsidiary status |
| Yahoo Finance BAY.BK quote page | 2 — credible secondary | Data aggregator | BAY.BK exists as a cross-check series |
| SET BAY price/historical-trading pages | 1 — primary/official | Exchange | BAY currently listed |
| Pantip thread on classic portal retirement | **3 — use with caution** | Anonymous forum | Corroboration only; not load-bearing |
| Wikipedia — Bank of Ayudhya | Tertiary | Orientation only | Pointed to the 2013 MUFG transaction; verified against Krungsri's own site |

---

## References

Chawannakul, T. (2021). *Trading Thai stock market using Settrade Open API*. Medium. https://theerapatcha.medium.com/trading-thai-stock-market-using-settrade-open-api-58e4b3cebb81

Krungsri. (n.d.). *Company overview*. Bank of Ayudhya PCL. Retrieved August 9, 2026, from https://www.krungsri.com/en/about-krungsri/about-us/overview/overview

Pantip. (n.d.). *ลาก่อน Classic.Set.or.th & Classic.Settrade.com เราจะคิดถึงคุณ* [Forum thread]. Retrieved August 9, 2026, from https://pantip.com/topic/41850045

Pi Securities. (n.d.). *How to set up Settrade Open API*. Retrieved August 9, 2026, from https://support.pi.financial/hc/en-us/articles/6333789684121-How-to-set-up-Settrade-Open-API

SCB. (2022). *SCBX Mothership to list on the SET*. The Siam Commercial Bank PCL. https://www.scb.co.th/en/about-us/news/mar-2022/scbx-share-swap.html

SCBX. (2022). *"SCBX Group" achieved more than 99%*. https://www.scb.co.th/en/about-us/news/apr-2022/scbx-completed-share-swap.html

Settrade. (n.d.-a). *Settrade Open API*. Retrieved August 9, 2026, from https://developer.settrade.com/open-api/

Settrade. (n.d.-b). *Settrade Open API — API reference*. Retrieved August 9, 2026, from https://developer.settrade.com/open-api/api-reference/

Settrade. (n.d.-c). *settrade.openapi.Investor — Python SDK getting started*. Retrieved August 9, 2026, from https://developer.settrade.com/open-api/api-reference/reference/sdk/python/investor-derivatives/gettingStart

Settrade. (n.d.-d). *stt-open-api-sdk-example* [Source code]. GitHub. https://github.com/settrade/stt-open-api-sdk-example

Stock Exchange of Thailand. (n.d.). *BAY — Price*. Retrieved August 9, 2026, from https://www.set.or.th/en/market/product/stock/quote/bay/price

ThaiStock. (n.d.). *thaistock* [Source code and README]. Retrieved August 9, 2026, from https://pypi.org/project/thaistock/

Yahoo Finance. (n.d.). *Bank of Ayudhya Public Company Limited (BAY.BK)*. Retrieved August 9, 2026, from https://finance.yahoo.com/quote/BAY.BK/

Zolkepli, H. (n.d.). *Stock-Prediction-Models* [Source code]. GitHub. https://github.com/huseinzol05/Stock-Prediction-Models
