# Handoff — 2026-08-14

State at the end of the session that **regenerated every result** against the code carrying
the fixes from `reviews/forecasting-models-2026-08-14.md`.

**Code and published results are now in sync.** The previous handoff's three "do this next"
items are all done, and committed as `56adb41`.

What remains, all of it below: **one design decision** (should `always_long` report an IC?)
and **four cleanup items** — the largest being that notebooks 01–06 were never re-executed,
so five of them still display a deleted column and one embeds a two-commit-stale report.
Nothing outstanding touches `src/`, the test suite, or the numbers in `results/`.

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

## The one design decision left

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

---

## Issues to fix next session

Found by auditing the repo *after* commit `56adb41`. Ranked by how visible the damage is.
None of these affect `src/`, the test suite, or the numbers in `results/` — they are stale
**deliverables** and housekeeping.

### 1. Notebooks 01–06 still show pre-fix outputs — **fix this first**

Only `07_figures.ipynb` was re-executed. The other six carry stored outputs from before the
IC change, so as deliverables they display a column that no longer exists and a conclusion
the report now retracts. Two need a **prose edit before re-running** — re-execution alone
will not correct them:

| file | cell | what is wrong | source or output? |
|---|---|---|---|
| `04_models.ipynb` | 5 | "The `beats_naive` column is computed, not interpreted: **MASE < 1.00 beats the naive lag.**" — that column was deleted | **source** (edit) |
| `04_models.ipynb` | 5 | "These models sit around 40–45% directional accuracy — *below* a coin flip." — the flat-days artefact §4 removed. Post-fix the real range is **44–58%, mean 51.7 / 51.0 / 54.4%** | **source** (edit) |
| `04_models.ipynb` | 4 | two result tables printing a `beats_naive` column | output (re-run) |
| `02_harness.ipynb` | 10 | result table printing a `beats_naive` column | output (re-run) |
| `06_report.ipynb` | 5 | embeds a **fully rendered report generated 2026-08-13 from git `13ebaad`** — two commits stale, carrying the retracted "beats the naive lag" headline | output (re-run) |

Order: edit `04_models` cell 5 prose → re-execute 01–06 in sequence → confirm no
`beats_naive` survives anywhere:

```bash
grep -l "beats_naive" notebooks/*.ipynb     # should return nothing
```

### 2. Decide whether `results/*.csv` should be tracked

`.gitignore:18` is a blanket `results/*`; `final-report.md` and the four PNGs are force-added
as deliverables, so the six result CSVs **and every `*.manifest.json`** are untracked. The
~77 minutes of compute is therefore not reproducible from a clone, and the provenance
manifests the README advertises ("every run has a manifest recording config, git SHA and
data hash") are not actually in the repo. Consistent with earlier commits, so it looks
deliberate — but it is worth an explicit decision rather than an inherited default.

### 3. Ruff: 42 findings, all in `notebooks/`

`ruff check src tests` is clean; bare `ruff check` is not. All seven notebooks trip
`E401` (`import sys, pathlib`), `I001` (unsorted imports) and `E402` (imports below code —
partly unavoidable, since the `sys.path` bootstrap must run before the package imports).
Pre-existing and unchanged by this session. `ruff check --fix` clears 27 of the 42; the
`E402`s need either a per-file ignore or a `# noqa: E402` on the bootstrap cells.

### 4. Figure 3's buy-and-hold reference assumes exactly one baseline row

`07_figures.ipynb` cell 7 computes `n_beat` against `d.loc[is_bh, "ret_friction"].iloc[0]`.
With one `buy_and_hold` agent per ticker that is correct today, but it will silently compare
against whichever row sorts first if a second baseline agent is ever registered. Low
priority; noted so it is not mistaken for intent.

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
