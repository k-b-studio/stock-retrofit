# Review — do the forecasting models work?

Independent check of the claims in `results/final-report.md`, 2026-08-14, against
the committed artefacts and a re-run of the sweep.

**Verdict: no, the models do not work — and that conclusion is correct. But three of
the pieces of evidence offered for it do not support it, and one of them is wrong.**

---

## 1. The harness is sound

Everything the README stakes its credibility on holds up.

| check | result |
|---|---|
| `PYTHONPATH=src pytest` | **149 passed**, incl. `test_no_leakage.py`. (`.pytest_cache/v/cache/lastfailed` lists 3 entries, but all three node IDs are gone — the tests were renamed since that cache was written. Stale, not failing.) |
| Determinism | Re-ran `04_gru`, `19_stack_rnn_arima_xgb`, `00_naive_lag` on KBANK: MASE, dir_acc and Sharpe reproduce the committed CSV **exactly**, to every printed digit. |
| Leakage guard | Real, not decorative. Fold-scoped, raises on any registered fit touching a test row. |
| Splits | `WalkForward` is correct; `assert_no_overlap` holds; excluding `train_end - 1` genuinely closes the one-day label leak the README describes. |
| Models touching test data | Audited all `x_test`/`y_test` uses. ARIMA's `predict` walks forward and appends each realised value *after* forecasting it — legitimate. The stacking autoencoder fits on train rows only and registers. No model cheats. |

Two soft spots, neither of which affects the published numbers:

- `register_fit` is **opt-in**. A future preprocessing step that forgets to call it is
  invisible to the guard. Nothing currently forgets.
- The guard is armed around `prepare_fold` + `model.fit`, but `model.predict` runs
  **outside** it (`eval/runner.py:124`). ARIMA legitimately consumes `fold.y_test`
  there; a model that cheated at predict time would not be caught.

## 2. The models have no usable skill — confirmed independently

The report's own metric (MASE) can't actually establish this — see §3 — so I re-ran all
22 models × 3 tickers keeping the predictions the CSVs discard, and measured the
out-of-sample **information coefficient**, `corr(forecast, realised next-day return)`.

| ticker | n | mean IC | IC > 0 | individually significant (\|t\| > 1.96) |
|---|---|---|---|---|
| KBANK | 480 | **+0.051** | 18/22 | `04_gru` +0.091, `13_gru_seq2seq` +0.112, `15_gru_seq2seq_vae` +0.098 |
| BAY | 480 | **+0.025** | 15/22 | `07_vanilla` +0.101, `08_bidirectional_vanilla` +0.123, `09_vanilla_2path` +0.107, `19_stack_rnn_arima_xgb` +0.129 |
| SCB | 240 | **−0.044** | 2/22 | none |
| **pooled** | | **+0.011** | **35/66** | 7 of 66, where ~3 are expected by chance |

35 of 66 positive is a coin flip. The sign is not consistent across tickers, the
per-ticker ICs cluster tightly (all 22 models read the same five features, so the 66
runs are nowhere near independent — the effective sample is closer to 3 than 66), and
SCB runs negative throughout. **No evidence of real forecasting skill.**

A second, blunter check: model-implied strategies against simply holding the share.

| ticker | always-long net Sharpe | models beating it |
|---|---|---|
| KBANK | +1.62 | 4 / 22 |
| SCB | +0.86 | 2 / 22 |
| BAY | +1.54 | 0 / 22 |

**6 of 66.** Owning the stock beats almost the entire catalogue.

## 3. "0 of 66 beat the naive lag" was true before any model ran

MASE here is `MAE(model) / MAE(zero forecast)`. MAE is minimised by the conditional
*median*, which for daily equity returns is ≈ 0 — so the zero forecast is already
near-optimal under this loss, and any forecast that moves off zero pays for it.

Simulated on KBANK's actual test block, using a synthetic forecaster with a **known**
edge and MSE-optimal shrinkage:

| true IC | mean MASE | 5–95% |
|---|---|---|
| 0.00 | 1.0213 | — |
| 0.05 | 1.0193 | 1.014–1.025 |
| 0.10 | 1.0171 | 1.006–1.029 |
| **0.20** | **1.0140** | 0.995–1.034 |
| 0.40 | 0.9800 | 0.949–1.015 |

An IC of 0.20 would be an outstanding daily equity signal. It scores MASE **1.014** and
is reported as *"does not beat the naive lag."* Even IC 0.40 fails a fifth of the time.
The pass/fail test cannot be passed by any realistic forecaster on this data, so "0 of
66" is a property of the metric, not a finding about the catalogue.

MASE is still fine as a *ranking*: on KBANK and BAY the lowest-MASE model is also the
highest-IC model. It is the **`beats_naive` boolean and the headline built on it** that
should go — replace with the IC table, or keep MASE and drop the 1.00 threshold.

## 4. "The forecasts call direction worse than chance" is wrong

`notebooks/07_figures.ipynb` cell 5 draws a hard line at 50%, colours every dot "below a
coin flip", legends `at or above (none)`, and titles the figure *"The forecasts call
direction worse than chance."*

50% is the wrong line. `directional_accuracy` scores `sign(y_pred) == sign(y_true)`, and
a large share of test days have **exactly zero** return — unwinnable by construction:

| ticker | zero-return test days | max attainable dir_acc | true chance line |
|---|---|---|---|
| KBANK | 20.4% | 79.6% | **39.8%** |
| SCB | 16.2% | 83.8% | **41.9%** |
| BAY | 29.0% | 71.0% | **35.5%** |

Against the correct line, **55 of 66 runs are at or above chance** (KBANK 22/22, BAY
19/22, SCB 14/22) — not 0 of 66. The models still have no *skill* (§2); they are simply
not "worse than chance," and the figure's headline is an artifact of the reference line.

Fix: either draw the per-ticker chance line, or exclude zero-return days from the
denominator, or score direction only on days that actually moved.

## 5. Some test rows are days the stock never traded

~5% of bars in each series have `volume == 0` **and** `high == low` — vendor-padded SET
holidays, not sessions. `data/quality.py:149` flags these as an *advisory* and keeps them,
so they land in the evaluation set:

| ticker | test rows whose target falls on a padded bar |
|---|---|
| KBANK | 41 / 480 = **8.5%** |
| BAY | 30 / 480 = **6.2%** |
| SCB | 3 / 240 = 1.2% |

On those rows the target is exactly 0 by construction: a free win for the naive baseline
and a guaranteed loss for anything that makes a call. They account for 42% of KBANK's
zero-return days. Drop them from the target, or exclude them from scoring.

## 6. The Sharpe columns have no reference row

The model tables print `sharpe_net` up to +2.10 with nothing to compare against. Holding
the share returned +1.62 (KBANK) / +0.86 (SCB) / +1.54 (BAY) over the same blocks. The
agent tables get a `buy_and_hold` baseline row; the model tables should get the same.

---

## What to change

> **Status: closed 2026-08-14.** All five applied in code, plus the market-side twin of §5
> (`SETMarket` now refuses orders on padded bars). `results/` has since been regenerated
> end to end against that code — evaluations, backtests, `final-report.md` and all four
> PNGs — and `README.md` re-framed onto IC. The list is kept rather than deleted because
> the reasoning is the useful part.
>
> What the regeneration showed: test rows fall to 439 / 237 / 450 on KBANK / SCB / BAY,
> mean out-of-sample IC is **+0.009 across 66 runs** with 36 positive and **0 of 66**
> beating buy-and-hold. The verdict below is unchanged; it now rests on a measurement
> with power. Two figures needed the same treatment as §4 and got it: figure 1 no longer
> headlines "no model beats a naive lag" (the claim §3 retires) and no longer ranks
> `always_long` among the models, and figure 4 picks its subject from the data rather
> than by a pinned name — `13_gru_seq2seq` had fallen to rank 4.


1. Drop the `beats_naive` boolean and the "0 of 66" headline — the test has no power.
   Report the IC table instead. *(§3)*
2. Fix figure 2's reference line and title. *(§4)*
3. Exclude zero-volume padded bars from the evaluation set. *(§5)*
4. Add an always-long reference row to the model tables. *(§6)*
5. Optional: arm the leakage guard around `model.predict` too. *(§1)*

None of this changes the bottom line. **The catalogue does not forecast these three
shares, the harness that says so is trustworthy, and the report is right for reasons
partly different from the ones it gives.**
