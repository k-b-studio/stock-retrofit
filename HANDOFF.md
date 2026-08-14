# Handoff — 2026-08-14

State at the end of the session that **regenerated every result** against the code carrying
the fixes from `reviews/forecasting-models-2026-08-14.md`.

**Code and published results are now in sync.** The previous handoff's three "do this next"
items are all done. What remains is one genuine open decision, listed at the bottom.

---

## Current status

| | |
|---|---|
| Test suite | **155 passed** (`PYTHONPATH=src python3 -m pytest -q`) |
| Lint | `ruff check src tests` **clean**. Bare `ruff check` reports **42 pre-existing findings, all in `notebooks/`** (`import sys, pathlib`, E402 imports below code) across all seven — not introduced here, and unchanged by this session's edits. The earlier handoff's "ruff check clean" was scoped to `src`/`tests`. |
| `results/` | **regenerated end to end** — 3 evaluate CSVs, 3 backtest CSVs, `final-report.md`, 4 PNGs |
| Agent runs | 72 of 72 clean (24 agents × 3 tickers), no failures |
| Outstanding errors | **none** |

⚠️ `pytest` fails to collect without `PYTHONPATH=src` — the package is not pip-installed
in this environment. Either run `pip install -e ".[dev]"` once, or prefix every command
with `PYTHONPATH=src`. This is environment setup, not a defect.

## What this session did

### 1. Regenerated every result

```bash
export PYTHONPATH=src
for S in KBANK SCB BAY; do python3 -m stock_retrofit.cli evaluate --all --symbol $S; done
for S in KBANK SCB BAY; do python3 -m stock_retrofit.cli backtest --all --symbol $S; done
python3 -m stock_retrofit.cli report --symbols KBANK,SCB,BAY
# then: jupyter nbconvert --to notebook --execute --inplace 07_figures.ipynb
```

**Measured runtimes, replacing the previous estimate of "~20 min each":** KBANK backtest
**30.5 min**, BAY **~30 min**, SCB **~15 min**. SCB runs at exactly half the per-agent cost
because it has 4 folds, not 8 — its Yahoo history begins 2022-04-20. Total ≈ 77 min, not 60.
The README quickstart now says ~30 min.

**The numbers moved as predicted, and the conclusion held.**

| | KBANK | SCB | BAY |
|---|---|---|---|
| test rows (was 480/240/480) | 439 | 237 | 450 |
| folds | 8 | 4 | 8 |
| flat share | 13.0% | 15.2% | 24.2% |
| mean IC | +0.041 | −0.038 | +0.025 |
| models beating always-long | 0/22 | 0/22 | 0/22 |

Pooled headline: **mean out-of-sample IC +0.009 across 66 runs, 36 positive, 5 clearing
|t| > 1.96 against ~3 expected, and 0 of 66 beating buy-and-hold.** No model reaches
MASE < 1.00 on any ticker — closest is `14_bidirectional_gru_seq2seq` on KBANK at 1.0017.

### 2. Rewrote `README.md` onto IC

Four stale places, one more than the previous handoff listed:

- **Headline** — was "no model beats a naive lag out-of-sample"; now the IC result, with the
  explicit note that MASE is a ranking rather than a test. Matches `report.py`.
- **Comparison table** — headline-metric row now says IC; baseline row names `AlwaysLong`;
  a new trading-calendar row records the padded-bar fix.
- **Metrics section** — rewritten around `ic` / `ic_t`, plus `flat_share`, the two pinned
  reference rows, and a paragraph on padded non-sessions. That paragraph also settles the
  previous handoff's open question 2: padded bars stay in feature windows as history and
  carry no label, and the README now says so.
- **Limits** — "8 folds × 60 days = 480 per ticker" → the real 439 / 237 / 450. That line was
  *already* wrong for SCB before the padded-bar fix, since SCB runs 4 folds.

### 3. Fixed two figures the review's list did not cover

Regenerating exposed two figures asserting things the report now retracts:

- **Figure 1** headlined *"No model beats a naive lag out-of-sample"* — the exact claim §3
  retires as near-vacuous. Retitled to "MASE ranks the catalogue — it does not test it";
  the shaded band is relabelled as orientation rather than "where skill would appear".
  It also ranked **`always_long` among the models**, where its degenerate MASE of ~1.00 put
  it above every real model on KBANK and SCB — it is a position, not a forecast, so it is
  now excluded from the dots exactly as figure 2 already excludes it.
- **Figure 4** pinned `config = "13_gru_seq2seq"` under a comment claiming it was the
  lowest-MASE model on KBANK. After regeneration that model sits at **rank 4**. The cell now
  reads the ranking from the CSV, and its subtitle quotes the model's IC rather than
  repeating "it still loses to the naive lag".

### 4. Closed the review

`reviews/forecasting-models-2026-08-14.md` now carries a status block recording all five
items plus the market fix, the regenerated numbers, and the two extra figure fixes. The
"What to change" list is kept — the reasoning is the useful part.

## Unfinished — the one real decision left

**Should `always_long` report an IC at all?** It is inconsistent across tickers today, which
is worse than either answer:

| | KBANK | SCB | BAY |
|---|---|---|---|
| `always_long` IC | +0.047 | −0.094 | **NaN** |

Its forecast is constant *within* a fold but refits per fold, so pooled across folds it
varies slightly and correlates with realised returns by chance — except on BAY, where it
comes out exactly constant and the correlation is undefined. Either suppress it to `—`
everywhere (as `dir_acc` already does for the naive lag) or keep it and document why a
reference row carries a skill number. Decide and make it explicit; do not leave it varying
by ticker.

Also still true from the previous handoff:

- The MASE power figures quoted in `metrics.py` and `report.py` (IC 0.10 crosses 1.00 in
  27% / 34% / 0% of draws on KBANK / SCB / BAY) were measured on this evaluation set. If
  `configs/eval.yaml` changes, re-measure rather than carrying them forward.

## Errors hit, and their resolutions

None outstanding — recorded so they are not re-discovered.

1. **`ModuleNotFoundError: stock_retrofit` under pytest** — package not installed. Use
   `PYTHONPATH=src`.
2. **A near-miss that would have silently wrecked the harness.** `prepare_fold` derived
   one `valid` mask from *both* "features are present" and "target is present". Once
   `build_target` began dropping padded labels, requiring all 20 bars of a lookback
   window to have valid targets would have discarded ~64% of samples instead of the ~5%
   at issue. Split into `feature_ok` (window membership) and `label_ok` (sample
   eligibility). If you touch that function, keep them separate.
3. **A test asserted something false.** The first version of
   `test_information_coefficient_detects_an_edge_that_mase_misses` used clean Gaussian
   returns, where MASE ≈ sqrt(1 − IC²) — so an IC of 0.20 scores 0.98 and *does* beat
   1.00. MASE's failure here depends on **zero-inflation**, not on shrinkage alone. The
   test now uses 25%-flat returns, matching BAY.
4. **pandas `FutureWarning`** on `.fillna` object downcasting in `build_target` — replaced
   with a numpy shift.
5. `.pytest_cache/v/cache/lastfailed` lists three node IDs that no longer exist (tests
   renamed at some earlier point). Stale cache, not failures. Harmless.
