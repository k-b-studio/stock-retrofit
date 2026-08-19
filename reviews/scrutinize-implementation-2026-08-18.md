# Scrutinize — the implementation, 2026-08-18

Third review of this project, and the first of the **built system** rather than a
plan or the model layer.

- `reviews/scrutinize-thai-set-retrofit.md` (2026-08-09) reviewed the *plan*.
  Verdict: rework. All six of its findings were acted on.
- `reviews/forecasting-models-2026-08-14.md` reviewed the *model and metric*
  layer. Verdict: the conclusion is right, three pieces of evidence for it are
  not. All five items closed, plus the optional sixth.

Neither looked hard at the **agent / `SETMarket` path** or at whether the
tracked deliverable still reproduces. That is where this review goes. Everything
below was recomputed from the committed artefacts and the code at `38f49af`;
`PYTHONPATH=src pytest` is **155 passed**.

**Verdict: fix-then-ship.** The harness deserves the trust the README places in
it, and the bottom line — no forecasting skill, no agent that beats holding the
share — survives everything here. But the agent half of the report is measured
on a different basis from the model half, for a reason that is a genuine defect,
and the tracked report has a reproducibility cliff about six weeks out.

---

## 1. Intent, and the simpler-alternative pass

**Goal in one sentence:** run a 2018-era TF1 catalogue of 22 forecasters and 24
agents against three Thai bank shares, on a harness honest enough that a
negative result is believable.

The 2026-08-09 review's central demand — *"do not port 62 notebooks, port one
pipeline and 62 configs"* — was taken, and it worked. 22 model configs over ~8
model modules, 24 agent configs over 6 agent modules, one `evaluate` command.
That restructuring is the reason this project could answer its own question at
all, and it should be recorded as a success rather than re-litigated.

The mandatory simpler-alternative pass, applied to what exists now:

- **The remaining complexity is load-bearing.** `SETMarket` is 414 lines to
  enforce five rules; the tick-snapping and float32 handling in `fill_price`
  look like over-engineering until you notice they exist because BAY's vendor
  high arrives as `24.299999237060547`. That is the correct amount of care.
- **One thing is now over-built relative to its yield: the agent catalogue.**
  24 agents × 3 tickers × 2 friction modes × 8 folds costs ~77 minutes per full
  regeneration, and the finding it produces is one sentence ("frictions cost
  ~1% per fold; almost nothing beats buy-and-hold"). That sentence was already
  visible after the Q-family. This is not worth undoing — it is built and it
  runs — but it argues against adding agents, and it makes Finding 1 more
  urgent, because a 77-minute regeneration is one nobody re-runs casually.
- **Doing less is no longer available.** The project's value is now the
  *negative result plus the harness that makes it credible*. Both are complete.

No restructuring is recommended. The findings below are defects, not scope.

---

## 2. Findings

### BLOCKER — The two halves of the report measure "hold the share" differently, and it changes who wins

**Outsider.** `results/final-report.md` prints, for KBANK, a model table whose
pinned reference row reads:

| model | … | Sharpe net |
|---|---|---|
| `00_always_long` | … | **+1.69** |

and 30 lines later an agent table whose pinned reference row reads:

| agent | … | Sharpe net |
|---|---|---|
| `00_buy_and_hold` | … | **+1.41** |

Same strategy — own the share across the same test blocks. Same column name.
Two numbers, 20% apart. Across all three tickers:

| ticker | model table `always_long` | agent table `buy_and_hold` | gap |
|---|---|---|---|
| KBANK | +1.6905 | +1.4095 | **17%** |
| SCB | +0.9435 | +0.7215 | **24%** |
| BAY | +1.5942 | +1.4175 | **11%** |

They come from different code paths that never meet: the model number is
`eval/metrics.py:sharpe` over `strategy_returns` on the concatenated test rows;
the agent number is `BacktestResult.sharpe()` over a `SETMarket` equity curve.

This is not cosmetic. Scoring the 23 active KBANK agents against each bar:

| bar used | agents that beat holding the share |
|---|---|
| agent table's +1.41 | **4 of 23** |
| model table's +1.69 | **0 of 23** |

The four are `08_recurrent_q_learning` (+1.585),
`09_double_recurrent_q_learning` (+1.578), `16_actor_critic_recurrent` (+1.515)
and `13_double_duel_recurrent_q_learning` (+1.509). **Every KBANK agent that
appears to beat buy-and-hold on a risk-adjusted basis sits inside the gap
between the project's two measurements of buy-and-hold.** On SCB, 2 of the 5
agents that clear the agent bar fall inside the gap.

**Insider.** "No published sentence is wrong. The agent verdicts in the report
are stated on **return**, not Sharpe — line 114 says *'Buy-and-hold returned
+7.82% per fold after costs. 0 of 23 active agents beat it.'* The Sharpe column
is supporting detail, and the two paths genuinely measure different things: one
is a costless position series, the other is a cash-and-shares portfolio with
board lots."

**Resolution — the insider is right that no stated claim is false, and that is
the most fragile kind of correct.** I checked every headline sentence; none
compares the two Sharpes. But:

- The report's own summary (line 268) says *"0 of 66 beat simply holding the
  share, which is the comparison that decides whether any of this was worth
  running"* — elevating "beats holding the share" to **the** criterion, while
  the document contains two different values for it under one column name.
- A reader asking the obvious follow-up — *"did any agent beat holding on a
  risk-adjusted basis?"* — gets **yes, four** from one table and **no** from
  the other, with nothing in the document to adjudicate.
- The gap has a real cause, and it is a bug. See the next finding.

**Suggested change.** Pick one measurement of "hold the share" and use it in
both tables, or rename the columns so they cannot be read as the same quantity
(`sharpe_position` vs `sharpe_portfolio`) and state the difference where the two
tables meet. Then fix the underlying defect, which is worth fixing regardless.

---

### BLOCKER — `run_episode` pads every fold's return series with structural zeros, deflating every agent Sharpe

**Outsider.** `agents/base.py:138`:

```python
equity = np.full(len(bars), config.initial_cash, dtype=float)
first = max(start, timestep - 1)
for t in range(first, len(bars)):
    ...
    equity[t] = market.equity(t)
if first > 0:
    equity[:first] = config.initial_cash
```

`bars` deliberately includes a lead-in of `timestep - 1` rows so the first
observation window can be filled (`agents/runner.py:70`). The agent does not
trade on those bars, so equity holds flat at `initial_cash` across them. Then:

```python
def daily_returns(self):
    return np.diff(self.equity) / self.equity[:-1]
```

diffs the **whole** array — lead-in included. Every fold therefore contributes
`timestep - 1 = 19` exact zeros to the return series before a single decision is
made, and `sharpe()` divides by the standard deviation of that padded series.

Measured directly, an always-long probe over 80 synthetic bars:

```
daily_returns length            : 79
leading exact-zero returns      : 18   (23% of the series)
reported sharpe (as shipped)    : +3.0920
sharpe over the traded block    : +3.5322
understated by                  : 12.5%
```

With ~55 test rows per fold on KBANK, 19 padded zeros is ~26% of each fold's
series. The deflation is `≈ sqrt(n / (n + k))`, which is the dominant term in
the 11–24% gaps in Finding 1; board-lot rounding and per-fold re-entry account
for the remainder.

**Insider.** "It is conservative — it understates agent performance, so no
agent is being flattered. And it hits `buy_and_hold` identically, so the agent
table's internal ranking and its 'beats buy-and-hold' verdict are unaffected."

**Resolution — both true, and it still has to be fixed.** Internal rankings
survive; that is why the defect went unnoticed through two prior reviews and 155
tests. What does not survive is any comparison *across* the two tables, which is
Finding 1, and any statement about an agent's absolute risk-adjusted return. A
Sharpe reported to two decimals that is systematically 12–24% low is a wrong
number even when it is conservatively wrong.

It is also the only quantity in this project with **no test at all**.
`grep -rn "daily_returns\|sharpe" tests/` returns nothing touching
`BacktestResult`; the two `run_episode` tests in `test_market_rules.py` assert
on fills, not on the equity curve. In a repo whose defining feature is
regression-locking its own findings, this is the conspicuous gap.

**Suggested change.** Carry `first` onto `BacktestResult` and slice the equity
curve before differencing (`equity[first:]`), or return returns from `first`
onward. Add a test that a flat lead-in contributes no returns — e.g. assert
`len(result.daily_returns()) == len(bars) - first`. Regenerating `results/` is
~77 minutes and every agent Sharpe will rise slightly; the report's stated
verdicts do not change.

---

### MAJOR — `final-report.md` is tracked, reproduces exactly today, and will stop in ~33 trading days

**Outsider.** `.gitignore:21` tracks the report with this reasoning:

> *"…the final report is tracked. It is NOT reproducible from this repo: the
> cached bars are untracked and the vendor's series moves daily, so a
> regenerated report would not match this one."*

The data has indeed moved since the report was committed. Comparing the
manifests against the cache now on disk:

| ticker | run manifest (2026-08-14) | on disk (2026-08-18) | content hash |
|---|---|---|---|
| KBANK | 6,594 rows → 2026-08-13 | 6,597 rows → 2026-08-18 | **differs** |
| SCB | 1,047 rows | 1,050 rows | **differs** |
| BAY | 6,594 rows | 6,597 rows | **differs** |

**Insider.** "Which is exactly what the comment says. The report is a
point-in-time deliverable, stamped with a git SHA and a data fingerprint. That
is the honest way to ship an irreproducible artefact."

**Resolution — the comment is wrong, and being wrong is why there is no guard.**
I regenerated KBANK against today's data and compared every row to the committed
report:

```
model                          IC now  IC rpt  MASE now  MASE rpt  ShNet now  ShNet rpt
00_always_long                 +0.047  +0.047    1.0010    1.0010      +1.69      +1.69
14_bidirectional_gru_seq2seq   +0.092  +0.092    1.0017    1.0017      +0.95      +0.95
13_gru_seq2seq                 +0.131  +0.131    1.0088    1.0088      +0.83      +0.83
...
rows that no longer match: 0 of 8
```

**It reproduces exactly** — to every printed digit, four days and three bars
later. The report is far more reproducible than the repo claims.

The reason is fold arithmetic, and it is a cliff rather than a slope.
`configs/eval.yaml` sets `train_window=750, test_window=60, step=60,
max_folds=8`. Folds are generated at `train_end = 750 + 60k` while
`train_end + 60 <= n`, then the most recent 8 are kept. So:

| n (rows) | folds | last test block |
|---|---|---|
| 6,594 *(report)* | 8 | `[6510:6570]` |
| **6,597** *(today)* | 8 | `[6510:6570]` |
| 6,620 | 8 | `[6510:6570]` |
| **6,630** | 8 | `[6570:6630]` ← window slides |

Appending rows changes **nothing** until `n` reaches 6,630, at which point a new
fold is created, `max_folds=8` drops the earliest one, and every pooled number
in the report changes at once. Today `n = 6,597`. **That is 33 trading days —
roughly six to seven weeks.**

So the risk is not the daily drift the comment describes. It is a single
discrete event, on a knowable date, after which the tracked deliverable
silently stops matching its own code. And because the comment asserts the
report is already irreproducible, nobody built the check that would catch it.

**Suggested change.** Add a `verify` command (or a test) that recomputes
`data_fingerprint` and the fold layout, and compares them against the manifest
alongside the committed report — failing loudly when the fold boundaries move.
The fingerprint is already recorded by `eval/manifest.py`; nothing ever reads it
back. Then correct the `.gitignore` comment to describe the real mechanism, and
consider pinning the three parquet files with `git-lfs` or recording the exact
row count the report was built on.

---

### MODERATE — The recorded data fingerprint is never checked against anything

Every run writes `content_hash`, `rows`, `range` and a `git_sha` into
`results/*.manifest.json`, and `report.py:254` prints the hash into the report.
That is good practice, and it is write-only:

```
grep -rn "content_hash|fingerprint" src/ tests/ --exclude manifest.py
  → cli.py (writes), report.py (prints), cache.py (computes)
```

Nothing compares a fingerprint to a previous one. The infrastructure for
detecting exactly the drift in the previous finding exists and is inert. This is
the cheapest item on the list and it closes the previous one.

**Suggested change.** One test: load the newest manifest for each kind, recompute
`data_fingerprint`, and assert the fold layout is unchanged — skipping (not
failing) when the cache is absent, so a fresh clone stays green.

---

### MINOR — Acceptance criterion 3 is not met, and the docs say so

`specs/thai-set-retrofit.md` AC3 requires `docs/settrade-api-notes.md` to answer
all four R2 questions "with a source link or a portal screenshot reference".
That document answers **none** of them — all five rows read **OPEN**, blocked on
an authenticated portal session.

This is disclosed rather than hidden, the fallback (yfinance primary, Settrade
adapter written but unconfigured) is implemented and documented, and the
2026-08-09 review's data blocker is genuinely resolved. But the README's status
section should not imply full acceptance while an AC stands unmet; say
"7 of 9 met, AC3 blocked on portal access, AC2 met via the fallback source."

---

## 3. What I traced and found solid

Stated so the coverage of this review is auditable.

- **The leakage guard now wraps `predict` as well as `fit`.** The 2026-08-14
  review listed this as optional item 5; `eval/runner.py:123` shows it applied,
  with a comment explaining why ARIMA's legitimate `y_test` consumption is safe.
  The remediation went further than the closure note claims.
- **Walk-forward splits are correct.** `assert_no_overlap` holds, test blocks are
  non-overlapping at `step == test_window`, and `max_folds` keeps the recent
  folds and re-indexes them.
- **`SETMarket` is careful in the places that matter.** Fills clamp into the
  bar's traded range; the inward re-snap exists because of a real float32
  artefact in BAY's high; an order that cannot land on a valid tick inside the
  bar is **refused** rather than invented. `Fill.ok` is deliberately independent
  of `Fill.rejected` so a participation-capped partial fill is still counted as
  a trade — with a docstring naming BAY as the ticker where that matters.
- **Agents get a genuine holdout.** `run_agent_walk_forward` fits on the training
  block inside the guard and executes on the test block outside it; `run_episode`
  calls `agent.reset()` so the frictionless twin does not inherit state from the
  friction run. The upstream in-sample defect is fully corrected.
- **The frictionless twin is honest.** `frictionless_twin()` switches off lot,
  fees, limits and cap — but keeps the session mask, so the twin cannot trade on
  padded SET holidays either. That is the right call and it is commented as such.
- **`market/rules.py` is labelled `!! RECONSTRUCTED, NOT VERIFIED !!`** in a
  module docstring that tells the reader which number is least trustworthy
  (commission) and where to change it (`configs/market.yaml`, not the module).
- **No TensorFlow** (AC8): `grep -ri tensorflow src/ configs/` → 0.
- **155 tests pass**, `ruff check src tests` clean.
- **The metric layer's self-criticism is unusual and correct.** `metrics.py`
  carries the simulation showing MASE 1.00 is unreachable, and
  `upstream_accuracy_do_not_use` is retained as executable documentation of the
  measure that made the upstream repo look successful.

---

## 4. Improvements, in priority order

| # | Change | Effort | Why here |
|---|---|---|---|
| 1 | Slice the lead-in out of `BacktestResult.daily_returns`; add the missing test | ~1 hour + 77 min regen | Fixes a wrong number and removes the cause of #2 |
| 2 | One measurement of "hold the share" across both tables, or distinct column names | ~2 hours | Four KBANK agents currently win or lose depending on which table you read |
| 3 | `verify` command / test comparing fingerprint + fold layout to the manifest | ~2 hours | The cliff is ~33 trading days out and nothing will announce it |
| 4 | Correct the `.gitignore` reasoning to describe the fold-boundary mechanism | 10 minutes | The current comment is why #3 was never built |
| 5 | README status: "7 of 9 ACs met, AC3 blocked on portal access" | 15 minutes | Honest non-compliance beats implied compliance |

Items 1 and 2 are one piece of work: fix the defect, then decide how the two
tables talk to each other. Item 3 is the one with a deadline.

None of this changes the conclusion. **The catalogue does not forecast these
three shares, almost nothing beats owning them, and the harness that says so is
trustworthy.** That verdict has now survived three independent reviews, and the
defects found this time are in how the answer is *measured and presented*, not
in the answer.

---

## 5. The one structural observation

Each review of this project has found the same shape of problem one layer
further out. The first found a leaking pipeline. The second found that the
metric declaring the models failures had no power. This one finds that the two
halves of the report measure their shared reference differently.

In every case the *conclusion* was right and the *instrument* was not, and in
every case the instrument was one the project had stopped examining because it
had been promoted from "thing under test" to "thing we test with". `always_long`
and `buy_and_hold` are the current example: introduced as the fix for the
missing-baseline finding, then trusted without anyone checking that the two
implementations agreed.

The generalisable move is the one this repo already applies to models and does
not yet apply to references: **pin the baseline the way the findings are
pinned.** A single test asserting that the model-path and agent-path
buy-and-hold Sharpes agree to within a tolerance would have caught Finding 1 the
day it appeared, and would catch the next instrument that drifts.
