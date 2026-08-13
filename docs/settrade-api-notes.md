# Settrade Open API — R2/R6 findings

**Status: the four required questions could NOT be answered in this environment.**
**Consequence: `yfinance` is the primary price source in this build.**

This document exists because spec R2 (and R6 of the data-layer spec) gate the
data layer on four facts about Settrade Open API. Acceptance criterion 3 asks
for each to be answered "with a source link or a portal screenshot reference".
That criterion **is not met**, and it cannot be met from here. What follows is
what was attempted, what was established, and what remains open — written this
way rather than filled in with plausible-looking numbers, because a fabricated
answer to a data-provenance question is worse than an open one.

## The four questions and their status

| # | Question (R2) | Status | Why |
|---|---|---|---|
| a | How far back do daily bars go? | **OPEN** | Requires an authenticated portal session |
| b | Max bars per request? | **OPEN** | Same |
| c | Rate limits? | **OPEN** | Same |
| d | Sandbox key sufficient, or live broker key required? | **OPEN** (strong indication: live key needed for production market data) | Secondary sources agree a broker relationship is required to execute; whether *historical data* is gated the same way is unconfirmed |
| e | Are returned prices corporate-action adjusted? | **OPEN** | Same |

## What was tried

1. **The credentials do not exist.** `SETTRADE_APP_ID`, `SETTRADE_APP_SECRET`
   and `SETTRADE_BROKER_ID` are issued through a participating brokerage. None
   are present in this environment and none can be self-issued. Without them the
   SDK cannot be exercised at all, so no empirical answer to (a)-(e) is
   obtainable here.
2. **The API reference is a client-rendered SPA.** As the feasibility research
   already recorded (`research/thai-set-retrofit-feasibility-2026-08-09.md`,
   §1.2), `developer.settrade.com/open-api/api-reference/` returns only shell
   metadata to a plain HTTP fetch. The endpoint documentation is rendered
   client-side and is not reachable without a browser session.
3. **`classic.settrade.com` remains dead** and is not a fallback. Confirmed in
   the prior research session: empty body, then a 180-second timeout. The
   `ThaiStock` scraper built on it is unusable regardless, since it caps history
   at six months.

## The decision, and its licence in the spec

Spec R2 provides for exactly this case:

> If depth turns out to be insufficient, fall back to `yfinance` as primary and
> say so loudly in the README.

The situation here is stronger than "insufficient depth" — it is *no access at
all* — so the fallback applies a fortiori. `yfinance` is primary; this is stated
in the README, in `configs/data.yaml`, and in the docstring of
`src/stock_retrofit/data/sources.py`.

## What this costs the project

Being explicit, because it is a real limitation and not a formality:

- **Provenance is weaker.** Yahoo is a redistributor, not the exchange. For a
  universe whose two named hazards are a corporate action (SCB→SCBX) and thin
  float (BAY), a third-party aggregation layer is exactly the wrong place to be.
- **Reconciliation is degraded, not absent.** R4/R10 want two independent
  sources cross-checked. With only one reachable source, `reconcile` compares
  the cache against a fresh vendor pull — which catches cache staleness and
  vendor revisions, but *cannot* catch an error Yahoo itself makes consistently.
  The CLI labels the comparison with the actual source names so this is visible
  in the output rather than implied.
- **Adjustment policy is inferred, not documented.** Bars are fetched with
  `auto_adjust=False` and the dividend/split record is cached alongside them, so
  the ±30% quality gate can distinguish a real limit-breach from a repricing.
  Whether Yahoo's SET closes are split-adjusted in every historical case has not
  been verified.

## Empirical findings about the data that *is* reachable

These were measured this session and are worth recording regardless of source:

- **`SCB.BK` on Yahoo begins 2022-04-20** — 1,047 bars, not the ~6,600 the other
  two names carry. The vendor's series is **already SCBX-only**. Two
  consequences: the `truncate_at_break` default (R5) is satisfied trivially, and
  `full_with_changepoint` **cannot be populated from this source** — there is no
  pre-break history to keep. The policy flag and the registry entry are
  implemented and tested regardless, because the caveat must live in code, and
  because a future Settrade fetch could supply the missing history.
- **`KBANK.BK` and `BAY.BK` both return 6,594 bars from 2000-01-04.** Depth is
  not a constraint for those two.
- **SCB shows a 4-session gap, 2022-04-21 to 2022-04-26** — the halt around the
  issuer substitution — and a +39.9% move on 2022-04-27, the first session after
  it. The quality gate treats a move that *spans* a registered break as
  explained rather than as a limit violation.
- **Vendor bar defects exist and are not rare enough to ignore**: 3 bars in
  KBANK, 1 in SCB, 6 in BAY have a `close` outside `[low, high]` by one to three
  ticks. These are repaired under an audited policy — see `data/raw/*.meta.json`
  for the per-bar trail.

## To close this out later

1. Open a brokerage account with a Settrade-participating broker; obtain
   `app_id` / `app_secret` / `broker_id`.
2. Copy `.env.example` to `.env` and fill them in.
3. Answer (a)-(e) from the authenticated portal and replace the table above.
4. Set `source: settrade` in `configs/data.yaml` and run
   `python -m stock_retrofit.cli fetch --symbols SCB,KBANK,BAY --force-refresh`.
5. Run `python -m stock_retrofit.cli reconcile --symbols SCB,KBANK,BAY --against yfinance`.
   That is the first run where reconciliation does what R10 actually intends.

`SettradeSource` is written against the SDK's documented surface and is wired
in, so step 4 is a config change rather than a development task. It has **never
been executed against the live API** — treat its request shape as unverified.

## Open decision this raises for Kim

Spec open-decision 1 asked whether, if Settrade history proved short, `yfinance`
should be promoted or the study scoped down. The situation is different from the
one anticipated — access is absent rather than shallow — and the default has
been taken: **yfinance promoted, full depth retained**. The alternative worth
weighing is whether the provenance concern above is severe enough to justify
opening a brokerage account before trusting any of these results.
