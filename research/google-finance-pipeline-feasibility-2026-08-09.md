# Is a Google Sheets + GOOGLEFINANCE + Apps Script + Docker pipeline viable for SET data?

**Research question:** Can SCB/KBANK/BAY historical price data be collected into the stock-retrofit project via Google Sheets' `GOOGLEFINANCE` function, orchestrated by Apps Script and consumed from a Docker container?

**Mode:** problem-solving / feasibility
**Date:** 2026-08-09
**Verdict:** **No — not for historical training data.** Viable only as a manual live-quote dashboard, separate from the data pipeline.

---

## Summary

| Question | Answer | Confidence |
|---|---|---|
| Does Google Finance cover the Thai exchange? | Yes — `BKK`, 15-minute delay | High — Google's own exchange table |
| Does the `GOOGLEFINANCE` *sheets function* return BKK data? | Probably, but **Google's two pages contradict each other** | Low — needs a 30-second empirical test |
| Can Apps Script / Sheets API read GOOGLEFINANCE **historical** data? | **No — returns `#N/A` by design** | High — stated explicitly in Google's docs |
| Can Apps Script read **real-time** single-value attributes? | Yes | Medium |
| Does the ToS permit caching the data for model training? | **No** — storage and reprocessing are explicitly prohibited | High — verbatim ToS language |
| Can Apps Script run inside Docker? | No — architectural mismatch | High |
| Is there a Thailand-specific correctness hazard? | **Yes — dates may shift by one day** | High — documented, and SET's close falls in the affected window |

---

## 1. The blocking finding

Google's own GOOGLEFINANCE reference states, verbatim:

> Historical data cannot be downloaded or accessed via the Sheets API or Apps Script. If you attempt to do so, you'll see a `#N/A` error in place of the values in the corresponding cells of your spreadsheet. (Google, n.d.-a)

This is the proposed architecture described and prohibited in one sentence. It is a deliberate product restriction, not a bug or a quota, so it will not be fixed and there is no supported flag to disable it.

**What remains readable:** real-time single-value attributes (`price`, `volume`, `high`, `low`, `closeyest`, …) return "as a value within a single cell" and are accessible to script. Historical requests "even for a single day … will be returned as an expanded array with column headers" — and it is that array form which is script-inaccessible (Google, n.d.-a).

**Practical consequence.** A script could snapshot one closing value per day from today forward, accumulating history at one row per day. It cannot retrieve the multi-year back-history the models in this project need. Building a pipeline that starts from zero rows and grows at one row per trading day is not a viable path to training data.

**Workarounds — noted and not recommended.** Community reports describe manually copying the historical array and pasting as values before reading it with script. This is *tier-3 sourcing*, brittle (it breaks the automation that motivated the design), and it is a deliberate circumvention of a stated product restriction — which also runs into §3 below. Not recommended.

## 2. Thailand-specific correctness hazard

From the same reference:

> Google treats dates passed into `GOOGLEFINANCE` as noon UTC time. Exchanges that close before that time may be shifted by a day. (Google, n.d.-a)

SET's regular session closes at 16:30 ICT, which is **09:30 UTC — before noon UTC**. Thai equities therefore sit squarely inside the class of exchanges this warning describes.

This matters more than it first appears. An off-by-one on the date index does not raise an error, does not look wrong in a spreadsheet, and does not fail any obvious test. It silently misaligns every feature with its target — which in a walk-forward forecasting setup either destroys the signal or, worse, introduces a one-day look-ahead that manufactures fake skill. Given that leakage is already the headline defect in the upstream repo (see `thai-set-retrofit-feasibility-2026-08-09.md`, §4.1), adopting a data source with a documented date-shift hazard would be adding a second, subtler leak on top of the one being fixed.

## 3. Terms of service

Google Finance's disclaimer states:

> You agree not to copy, modify, reformat, download, store, reproduce, reprocess, transmit or redistribute any data or information found herein or use any such data or information in a commercial enterprise without obtaining prior written consent. Google or its third party data or content providers have exclusive proprietary rights in the data and information provided. (Google, n.d.-b)

And the function reference adds:

> The data is not for financial industry professional use or use by other professionals at non-financial firms (including government entities). Professional use may be subject to additional licensing fees from a third-party data provider. (Google, n.d.-a)

A pipeline whose purpose is to **download** quotes, **store** them as Parquet, and **reprocess** them into model features engages three of the prohibited verbs by design. Personal, non-commercial research is a gray area in practice, but the prohibition on storage is explicit rather than inferred, and it is worth knowing that before building infrastructure on it.

*This is a plain reading of the published terms, not legal advice — I'm not a lawyer. If the project is ever likely to become commercial, get an actual opinion.*

## 4. Data provenance and quality

The exchange table shows BKK at a **15-minute delay**, with end-of-day prices provided by **Morningstar** and corporate actions and company metadata by **Refinitiv** (Google, n.d.-b).

Two implications for this project:

- Google Finance is a **redistributor**, not the exchange. For a market where the specific concerns are corporate actions (the SCB→SCBX restructuring) and thin-float print quality (BAY), a third-party aggregation layer is strictly worse than the exchange-sanctioned feed.
- Google explicitly "does not verify any data and disclaims any obligation to do so" (Google, n.d.-b). There is no quality guarantee to rely on and no recourse when a print is wrong.

## 5. The architecture doesn't compose

Apps Script executes on Google's managed infrastructure. It cannot be containerised, so "Docker with Google Apps Script" has no literal implementation. The nearest real design is:

```
Sheets (GOOGLEFINANCE)  →  Apps Script (on Google)  →  writes values to a sheet/Drive
                                                    ↓
                         Docker container  →  Sheets API  →  reads those values
```

Every arrow in that chain works **except** the one that matters: the historical array never becomes readable values without manual intervention (§1). The container adds deployment complexity without solving the constraint that blocks the design.

## 6. Source disagreement — flagged, not resolved

The two Google pages conflict on international coverage:

- The **function reference** says: "`GOOGLEFINANCE` is only available in English and does not support most international exchanges" (Google, n.d.-a).
- The **Finance Data Listing** explicitly includes `BKK — Thailand Stock Exchange — 15` under Asia, and `INDEXBKK — Thailand Stock Exchange Indexes — 15` (Google, n.d.-b).

The most likely reconciliation is that the disclaimer table describes coverage of the **Google Finance product**, while the function reference warns about the narrower coverage of the **spreadsheet function** — but that is inference, not something either page states. I did not test the function directly, because doing so requires an authenticated Google account.

**A 30-second test settles it.** In any sheet:

```
=GOOGLEFINANCE("BKK:KBANK", "price")
=GOOGLEFINANCE("BKK:KBANK", "close", TODAY()-30, TODAY(), "DAILY")
```

Row 1 returning a number and row 2 returning an array confirms function-level BKK support. Then read the same range from Apps Script with `getValues()` and confirm §1 — expect `#N/A` on the historical range.

## 7. Recommendation

**Do not build the pipeline as designed.** Split the intent in two:

- **Training data** → Settrade Open API as primary, `yfinance` as an independent reconciliation series. Both permit programmatic historical access; neither has the date-shift hazard, and Settrade is exchange-sanctioned. Specified in `specs/thai-market-data-layer.md`.
- **Live monitoring** → a Google Sheet with `GOOGLEFINANCE("BKK:SCB")`-style real-time cells, read by a human, never written to disk. This is squarely within intended use, costs an afternoon, and is genuinely useful for eyeballing whether a model's signal matches what the market is doing today. Keep it strictly outside the data path.

The instinct behind the question was sound — a second, independent price source is exactly the right thing to want. `yfinance` fills that role without any of these constraints.

---

## Coverage and limits of this investigation

- **Not tested empirically:** whether the `GOOGLEFINANCE` function returns data for `BKK:` tickers, and whether Apps Script `getValues()` yields `#N/A` on a BKK historical range specifically. Both findings above are from Google's documentation, which is authoritative on the restriction but was not verified against a live sheet this session (no authenticated account). §6 gives the test.
- **Not investigated:** Apps Script execution quotas and trigger limits. Moot given §1, but relevant if the live-dashboard option is pursued at scale.
- **Not investigated:** whether a paid Google Workspace or Google Cloud market-data offering lifts the Apps Script restriction.
- **Legal reading is non-professional** — see §3.

## Annotated source list

| Source | Tier | Note | Takeaway |
|---|---|---|---|
| Google — GOOGLEFINANCE function reference | 1 — primary/official | Vendor documentation for the exact function proposed | Historical data inaccessible to Apps Script/Sheets API; noon-UTC date shift; professional-use restriction |
| Google — Google Finance disclaimer & Finance Data Listing | 1 — primary/official | Vendor ToS + authoritative exchange coverage table | BKK covered at 15-min delay via Morningstar/Refinitiv; storage and reprocessing prohibited; no data verification |
| Stock Exchange of Thailand session hours | — | Used as the general fact that SET closes 16:30 ICT | Places SET before noon UTC, inside the documented date-shift window. **Not re-verified this session** — confirm against SET's published trading hours before relying on the §2 conclusion |

## References

Google. (n.d.-a). *GOOGLEFINANCE — Google Docs Editors Help*. Retrieved August 9, 2026, from https://support.google.com/docs/answer/3093281?hl=en

Google. (n.d.-b). *Disclaimer — Google Finance*. Retrieved August 9, 2026, from https://www.google.com/googlefinance/disclaimer/
