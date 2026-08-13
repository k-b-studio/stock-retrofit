# thai-market-data — a standalone Thai equity data package

> **Standalone spec.** You have no other context from the conversation that produced this.
> Background reading in this repo, in priority order:
> `research/google-finance-pipeline-feasibility-2026-08-09.md` — why Google Sheets was rejected as a source.
> `research/thai-set-retrofit-feasibility-2026-08-09.md` — why the `ThaiStock` scraper was rejected, and the SCB/BAY instrument caveats.
> `specs/thai-set-retrofit.md` — the modelling project that will be this package's first consumer.

## Goal

Build `thai-market-data`: a small, installable Python package that fetches, reconciles, caches, and quality-gates Stock Exchange of Thailand market data along two axes — **daily OHLCV prices** and **quarterly/yearly fundamentals** — with point-in-time correctness enforced on the latter. It is the single source of truth for market data across projects: the `stock-retrofit` modelling work imports it rather than reimplementing fetching, and any future project does the same. It knows nothing about models, features, or backtests.

The price axis is the critical path. The fundamentals axis is additive and must never block it.

## Why a separate package

The modelling project needs price data; so will the next project. Data acquisition has a completely different change cadence from modelling — vendor endpoints move, credentials rotate, corporate actions land — and a different testing style (contract tests against a live API vs. numerical tests on tensors). Keeping them apart means a Settrade outage or schema change is one package's problem, and the caveats encoded here (SCB's 2022 break, BAY's thin float) get applied consistently instead of being re-remembered per project.

## Requirements

**Sources**

- R1. `SettradeSource` — **primary**. Settrade Open API via the `settrade-v2` SDK. Auth by `app_id` / `app_secret` / `broker_id`, read from environment (`.env`, never committed). Daily OHLCV.
- R2. `YFinanceSource` — **reconciliation only, never primary**. Maps `SCB → SCB.BK`, `KBANK → KBANK.BK`, `BAY → BAY.BK`.
- R3. Both implement one `PriceSource` protocol: `fetch(symbol: str, start: date, end: date) -> pd.DataFrame` returning exactly `[date, open, high, low, close, volume, symbol]`, `date` as timezone-naive Asia/Bangkok trading date, sorted ascending, no duplicate dates.
- R4. **No Google Finance source.** `GOOGLEFINANCE` historical data is inaccessible to the Sheets API and Apps Script by design, its ToS prohibits storing and reprocessing the data, and its noon-UTC date handling shifts dates for exchanges closing before noon UTC — which includes SET. See the feasibility note. Do not add one later without re-reading it.
- R5. **No scraping fallback.** `classic.settrade.com` is dead and scraping SET's current site is not a supported path. If Settrade is unavailable, the package fails loudly rather than degrading to an unreliable source.

- R4b. **`thaifin` is a fundamentals source, not a price source.** It returns quarterly/yearly financial statements, not daily bars. It is specified below under Fundamentals and must not be used to satisfy R1–R3.

**Unknowns to resolve first**

- R6. Before writing `SettradeSource`, determine and record in `docs/settrade-api-notes.md`: (a) how far back daily bars are available, (b) max bars per request, (c) rate limits, (d) whether historical data works with a sandbox key or requires a live broker key, (e) whether returned prices are corporate-action adjusted. The API reference is a client-rendered SPA and none of this was answerable by plain HTTP fetch. **If depth proves insufficient for multi-year training, stop and escalate** — that finding changes the parent project's scope, and the right response is a decision, not a workaround.

**Cache**

- R7. Parquet under `data/raw/{symbol}.parquet`, one file per symbol. Consumers read the cache; only an explicit `fetch` touches the network.
- R8. Incremental: fetch only the gap between the cache's last date and today. `--force-refresh` refetches from scratch.
- R9. Every cache write is accompanied by `data/raw/{symbol}.meta.json` recording source, fetch timestamp, row count, date range, and a content hash — so any downstream result can be traced to the exact data that produced it.

**Reconciliation and quality gates**

- R10. `reconcile(symbol)` pulls both sources over the overlapping window and reports per-date close differences. Flag any date where the two disagree by more than one tick. **Never average them, never silently prefer one** — disagreement is a finding to surface, not a value to smooth.
- R11. Quality report per symbol, covering: missing trading sessions (against a SET trading calendar), zero-volume days, `high < low` or `close` outside `[low, high]`, null runs, duplicate dates, and any single-day move beyond ±30% (impossible under SET's ceiling/floor rules, so it indicates bad data rather than a real move).
- R12. Quality checks are **fail-loud**: a structural violation (R11's third and sixth items) raises rather than warns. Silent bad data is the failure mode this whole package exists to prevent.

**Corporate actions and instrument identity**

- R13. `corporate_actions.py` holds a small declarative registry of known discontinuities. Seed it with the SCB entry: on **2022-04-22**, SCB was delisted and SCB X Public Company Limited listed 1:1 in its place, retaining the `SCB` ticker — a change of issuer, not merely of name.
- R14. `load(symbol, policy=...)` where policy is `truncate_at_break` (default) or `full_with_changepoint`. Under the latter, the returned frame carries an `is_changepoint` boolean column so consumers can exclude the boundary from evaluation windows. **A caller must never receive a series spanning a registered break without a signal that it did.**
- R15. Instrument metadata per symbol — at minimum a `liquidity_note`. Seed BAY with its thin-float caveat (~72–76% MUFG-held), so a consumer building a backtest can programmatically know to apply a participation cap.

**Fundamentals — second axis, additive**

Source: [`thaifin`](https://github.com/ninyawee/thaifin), ISC licence, Python ≥3.11. **A local clone is at `../../external-repo/thaifin-master/` — read it before writing this module.** The requirements below were derived by reading that source, not the README, and several of them contradict what the README implies.

What it actually is: a wrapper around two undocumented third-party HTTP endpoints.

- **Financials** — `https://www.finnomena.com/market-info/api/public/stock/summary/{security_id}` (`thaifin/sources/finnomena/api.py`). An undocumented public web endpoint of Finnomena, not a contracted API. It can change shape or close without notice, and its terms of use have **not** been reviewed.
- **Listings / sector / market metadata** — `https://raw.githubusercontent.com/lumduan/thai-securities-data/main/...` (`thaifin/sources/thai_securities_data/api.py`). This is a personal GitHub repository, fetched from the **mutable `main` branch**, so the peer/sector data can change under you between runs with no version to pin.

Both are wrapped in a 24-hour in-memory `TTLCache` and a `tenacity` retry with exponential backoff — reasonable engineering, but it does not change the provenance. Treat this whole axis as tier-3 sourcing: usable for feature research, never as evidence for a claim about a company's financial position.

- R19. `ThaifinSource` exposes `fetch_fundamentals(symbol, period="quarter"|"year") -> pd.DataFrame`. **This is not a price source** — it does not implement `PriceSource` and must not be registered as one.
- R19a. **Drop the `close` column on ingest, unconditionally.** In `thaifin/sources/finnomena/model.py` the `THAI_FIELD_MAPPING` renders `close` as `ราคาล่าสุด (บาท)` — *latest price*, not *quarter-end close*. Whether the API returns a quarter-end value or the price at call time is unverified, and if it is the latter then the column is the same number repeated down every row, dated to quarters that predate it — pure look-ahead wearing a plausible label. Dropping it costs nothing (the price axis already supplies closes) and removes the ambiguity permanently. `mkt_cap`, `price_earning_ratio`, `price_book_value`, and `dividend_yield` are derived from a price and inherit the same doubt — carry them only if `docs/fundamentals-notes.md` resolves which price they use.
- R20. **Point-in-time correctness is a hard requirement, not an option.** Verified against `QuarterFinancialSheetDatum` in `thaifin/sources/finnomena/model.py`: the model carries `security_id`, `fiscal`, `quarter`, and `end_of_year_date` — and **no announcement or publication date field**. The date a figure became public is simply not in the data. SET allows listed companies 45 days after quarter end to submit reviewed quarterly statements. Every fundamentals row must therefore be assigned an `available_from` date, computed as `quarter_end + reporting_lag_days`, and **no consumer may ever receive a row before its `available_from`**.
- R20a. Deriving `quarter_end` takes work: `thaifin`'s `Stock.quarter_dataframe` indexes on a **string** like `"2009Q1"` (built in `thaifin/stock.py` as `fiscal.astype(str) + "Q" + quarter.astype(str)`), not a datetime. Parse it to a period, map to a calendar quarter end, then add the lag. Do not assume the fiscal year aligns to the calendar year for every issuer — `end_of_year_date` exists precisely because it does not always.
- R20b. **`quarter == 9` is a sentinel meaning "annual", not a ninth quarter.** `Stock.quarter_dataframe` filters `!= 9` and `Stock.yearly_dataframe` filters `== 9`. Handle it explicitly and assert on it, so a future upstream change surfaces as a test failure rather than as twelve months of phantom quarters.
- R21. `reporting_lag_days` is configurable, defaults to **60** (conservative), with 45 as the documented regulatory floor. The annual audited deadline was **not verified** — confirm it against SET's rulebook and, if it is longer than 60 days, raise the default for `period="year"` accordingly. Record the finding in `docs/fundamentals-notes.md`.
- R22. `load_fundamentals(symbol, as_of: date)` returns only rows where `available_from <= as_of`. There is **no parameter that disables this filter.** If a caller wants unfiltered data they must reach for a separately named `_load_fundamentals_unsafe()` that logs a warning on every call — the friction is deliberate. Look-ahead through fundamentals is the single most likely way this project reintroduces the leakage it exists to eliminate, and it is far harder to spot than the scaler bug in the upstream repo because it yields plausible outperformance rather than obvious nonsense.
- R23. thaifin types **every** financial field as `Optional[str]` in its Pydantic model, and its `quarter_dataframe` / `yearly_dataframe` properties never call `pd.to_numeric` — so the frames arrive with `object` dtype throughout. Coerce to `float` on ingest with an explicit failure path: a value that will not parse is recorded as null **and** counted in the quality report, never silently dropped or zero-filled.
- R23a. Pin `language="en"` on every thaifin call. The `language` parameter switches the DataFrame's **column names** between English and Thai (`thaifin/stock.py` selects `'ไตรมาส'` vs `'quarter'`), so any code that reads columns by name breaks silently under the wrong setting.
- R24. Extend the corporate-actions registry (R13) to fundamentals. SCB's series spans a bank and then a holding company; the fundamentals are likely to be *less* comparable across that boundary than the prices, since the reporting entity and its consolidation scope both changed. The same `truncate_at_break` default applies.
- R25. `peers(symbol)` wraps `Stocks.filter_by_sector(sector, language="en") -> List[str]` to return the sector cohort (Banking, for all three target symbols). This exists to enable cross-sectional features — relative valuation, sector-relative momentum — which the upstream modelling repo never had. Returns symbols only; the consumer decides what to do with them.
- R25a. **Snapshot the peer list; do not resolve it live.** `filter_by_sector` is ultimately backed by JSON on the `main` branch of a personal GitHub repo, so a live call makes your sector cohort a function of when you ran it. Fetch once, write to `data/fundamentals/sector_membership.json` with a fetch date, and have `peers()` read the snapshot. Refreshing is an explicit command, never a side effect of a backtest.
- R25b. Sector membership is **current**, not historical — the snapshot says who is in Banking today, with no record of who was in it in 2015. Any cross-sectional feature built from it inherits survivorship bias. That is acceptable for exploratory work and must be stated in the results, not discovered later.

**Interface**

- R26. Python API: `from thai_market_data import load, fetch, reconcile, quality_report, load_fundamentals, peers`.
- R27. CLI: `thai-market-data fetch --symbols SCB,KBANK,BAY`, `... reconcile --symbol SCB`, `... quality --symbol BAY`, `... fundamentals --symbol KBANK --as-of 2024-06-30`, `... status`.
- R28. Installable and versioned so `stock-retrofit` can pin it (`pip install -e ../thai-market-data` during development).

## Proposed approach

Four phases; each ends with something runnable.

**Phase 0 — scaffold and unknowns.** Package skeleton, `pyproject.toml`, `.env.example`, pytest, ruff/black. Then answer R6 **before** writing any source code, and write up `docs/settrade-api-notes.md`. This phase is deliberately gated on a research answer — resist the urge to start coding around an unknown that a portal visit would settle.

**Phase 1 — yfinance first.** Implement `PriceSource`, the canonical frame, the Parquet cache, and `YFinanceSource`. yfinance needs no credentials, so the entire cache/quality/CLI surface can be built and tested before touching Settrade auth. **Ends with:** three cached Parquet files and a quality report per symbol, from a source you can hit without any setup.

**Phase 2 — Settrade primary.** Implement `SettradeSource` against the same protocol, promote it to primary, demote yfinance to reconciliation. Add `reconcile()`. **Ends with:** a reconciliation report for each symbol showing where the two sources disagree.

**Phase 3 — instrument semantics.** Corporate-action registry, `load()` policies, metadata, changepoint column. **Ends with:** `load("SCB")` returning a truncated series by default, and `load("SCB", policy="full_with_changepoint")` returning the full series with the 2022-04-22 boundary flagged.

**Phase 4 — fundamentals.** `ThaifinSource`, string→float coercion, the `available_from` computation, `load_fundamentals(as_of=...)`, `peers()`. Build the point-in-time machinery **before** the fetching — write `test_point_in_time.py` first and let it drive the design. **Ends with:** `load_fundamentals("KBANK", as_of=date(2024,5,1))` demonstrably excluding 2024Q1 (whose `available_from` falls later), and including it at `as_of=date(2024,7,1)`.

**This phase is strictly optional to the parent project.** If Phases 0–3 run late, ship without it. The modelling work in `stock-retrofit` is price-and-technical throughout and has no dependency on fundamentals; this axis unlocks new features rather than unblocking existing ones.

**Assumptions — override any of these:**

- Python 3.11+, pandas 2.x, Parquet via pyarrow.
- Daily bars only. Settrade offers intraday; out of scope here.
- No database. Parquet on disk is right at three symbols and stays right well past thirty.
- The SET trading calendar is derived from observed trading dates in the data initially; swap in an authoritative holiday calendar if R11's missing-session check proves too noisy.
- No async. Three symbols daily does not justify it.

## Files / modules affected

New standalone package, a sibling of `stock-retrofit`:

```
thai-market-data/
  pyproject.toml
  .env.example                  # SETTRADE_APP_ID, SETTRADE_APP_SECRET, SETTRADE_BROKER_ID
  docs/settrade-api-notes.md    # R6 answers — write in Phase 0
  docs/fundamentals-notes.md    # R21 answers — reporting deadlines, verified
  src/thai_market_data/
    __init__.py                 # load, fetch, reconcile, quality_report, load_fundamentals, peers
    protocol.py                 # PriceSource, canonical schema
    sources/settrade.py yfinance.py thaifin.py
    cache.py                    # Parquet + sidecar metadata
    reconcile.py
    quality.py
    corporate_actions.py        # SCB 2022-04-22 registry, instrument metadata
    calendar.py                 # SET trading days
    fundamentals.py             # ingest, float coercion, peers
    point_in_time.py            # available_from computation, as_of filtering
    cli.py
  tests/
    test_schema_contract.py     # both price sources satisfy the canonical frame
    test_cache_incremental.py
    test_quality_gates.py       # malformed frames raise, not warn
    test_corporate_actions.py   # SCB never spans the break unsignalled
    test_point_in_time.py       # no row ever returned before available_from
  data/raw/  data/fundamentals/
```

Change in the consumer: `stock-retrofit`'s Phase 1 data layer is replaced by a dependency on this package. Its `src/stock_retrofit/data/` shrinks to a thin adapter, or disappears.

Speculative: `calendar.py` may collapse into `quality.py` if deriving trading days from the data proves sufficient.

## Acceptance criteria

1. `docs/settrade-api-notes.md` answers all five R6 questions, each with a source link or portal reference.
2. `thai-market-data fetch --symbols SCB,KBANK,BAY` produces three Parquet files plus three `.meta.json` sidecars.
3. `thai-market-data quality --symbol BAY` runs and reports; any structural violation exits non-zero.
4. `thai-market-data reconcile --symbol KBANK` prints a per-date comparison and a count of dates exceeding one tick of disagreement.
5. `load("SCB")` returns data starting no earlier than 2022-04-22 under the default policy; `load("SCB", policy="full_with_changepoint")` returns the full series with `is_changepoint` true on exactly that date. Both asserted in `test_corporate_actions.py`.
6. `test_schema_contract.py` runs the identical assertion set against both sources — same columns, same dtypes, same ordering guarantees.
7. `test_quality_gates.py` feeds a deliberately malformed frame (`close` outside `[low, high]`) and asserts it **raises**.
8. `grep -ri "googlefinance\|classic.settrade" src/` returns nothing.
9. `stock-retrofit` imports this package and its Phase 1 acceptance criteria still pass.
10. `test_point_in_time.py` asserts, over the full history of all three symbols, that **no returned row has `available_from > as_of`** — and includes a case that fails if the lag is set to zero. Verify it fails by temporarily setting `reporting_lag_days = 0`.
11. `load_fundamentals("KBANK", as_of=<a date inside a quarter's reporting lag>)` excludes that quarter; the same call after the lag window includes it. Both asserted, with the dates written out explicitly rather than computed in the test.
12. `docs/fundamentals-notes.md` states the verified SET submission deadlines for reviewed quarterly and audited annual statements, with a source link, and states which default `reporting_lag_days` each drives.
13. Fundamentals frames contain no `close` column (R19a), and every remaining financial field has dtype `float64`, not `object`.
14. `peers("KBANK")` returns a list containing `SCB` and `BAY`, read from the committed snapshot — asserted with the network unavailable, proving no live call (R25a).
15. A test asserts that no row with `quarter == 9` appears in a quarterly frame and that annual frames contain only such rows (R20b).
16. `docs/fundamentals-notes.md` records the pinned thaifin commit SHA and states plainly that both upstream endpoints are undocumented third-party services.

## Non-goals

- Any Google Finance / Google Sheets integration (R4). A live-quote Sheet for human monitoring is fine as a separate artifact, but it does not live in this package and its values never reach disk.
- Intraday, tick, or order-book data.
- News, sentiment, and analyst data. (Fundamentals **are** in scope as of R19–R25 — but only via thaifin, only quarterly/yearly, and only behind the point-in-time gate.)
- Order execution. This package reads market data and nothing else — no code path places an order.
- Feature engineering, indicators, resampling. Those belong to the consumer; this package returns raw OHLCV.
- Universe expansion beyond SCB/KBANK/BAY for now, though nothing in the design should prevent it.

## Open decision points for Kim

1. **If Settrade daily history turns out to be short** — promote yfinance to primary and accept the weaker provenance, or scope the modelling study to the shorter window? This is the one answer that could change the whole plan, which is why R6 is gated first.
2. **Sandbox vs. live broker key.** If historical data needs a live key, are you willing to open the brokerage account now, or should Phase 2 be deferred and yfinance carry the project meanwhile?
3. **SCB default policy.** Spec assumes `truncate_at_break` (~4 years of data). Say if you would rather default to the full series with the changepoint flagged.
4. **Adjusted vs. unadjusted prices.** If Settrade returns unadjusted, do you want the package to apply its own dividend/split adjustment, or to serve raw prices and let consumers decide? Raw-plus-metadata is the safer default but pushes work downstream.
5. **Reporting lag default.** Spec assumes 60 days for both quarterly and annual, against a verified 45-day regulatory floor for reviewed quarterly statements. Going to 45 buys ~15 days of extra usable history per quarter at the cost of assuming every company files at the deadline — which they do not. Say if you want the tighter number.
6. **Whether to vendor thaifin.** The case for vendoring got stronger after reading the source. Its `pyproject.toml` says `version = "1.0.0"` while the tree already contains the `Stocks` class the README lists as a v1.1 feature — so the version string does not identify the code, and `thaifin>=1.1` in a requirements file may not resolve to what you tested against. It is also a single-maintainer project depending on two undocumented third-party endpoints, one of them a mutable GitHub branch. ISC licence permits vendoring. Options: **pin by commit SHA** (recommended default — cheap, reversible), or vendor the ~3 modules actually used (`finnomena/api.py`, `finnomena/model.py`, `stock.py`) and own them. Do not pin by version number.
