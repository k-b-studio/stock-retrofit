"""Results tables.

Both reference lines are pinned to the top of every table: `naive_lag` (the
reference for the forecast as a *number*) and `always_long` (the reference for
the forecast as a *position*). Requirement R9 asks that "did this model actually
do anything" need no interpretation, and that needs both.

There is deliberately **no `beats_naive` column**. A MASE threshold at 1.00
cannot be crossed by any realistic forecaster on daily returns — see the module
docstring of `metrics` — so a column of `False` measured the metric rather than
the models, and a headline built on it was true before any model ran. The `ic`
column replaces it: the out-of-sample correlation between forecast and realised
return, which can actually distinguish skill from noise.
"""

from __future__ import annotations

import pandas as pd

from .runner import EvalResult

NAIVE_NAME = "naive_lag"
LONG_NAME = "always_long"
#: Reference rows, in the order they are pinned to the top of a table.
BASELINE_NAMES = (NAIVE_NAME, LONG_NAME)

_EMPTY = {
    "n": 0,
    "ic": float("nan"),
    "ic_t": float("nan"),
    "MASE": float("nan"),
    "dir_acc": float("nan"),
    "coverage": float("nan"),
    "flat_share": float("nan"),
    "RMSE_ret": float("nan"),
    "sharpe_net": float("nan"),
    "sharpe_gross": float("nan"),
    "turnover": float("nan"),
}


def is_baseline(names: pd.Series) -> pd.Series:
    """Rows that are reference lines rather than competitors."""
    pattern = "|".join(BASELINE_NAMES)
    return names.str.contains(pattern, case=False, regex=True)


def results_table(
    results: list[EvalResult],
    *,
    cost_per_turn: float = 0.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """One row per model, baselines pinned first, sorted by MASE thereafter."""
    rows = []
    for r in results:
        head = {
            "model": r.model_name,
            "symbol": r.symbol,
            "upstream": r.upstream,
            "folds": len(r.folds),
        }
        if not r.ok:
            rows.append({**head, **_EMPTY, "status": r.error or "no folds"})
            continue
        m = r.pooled(cost_per_turn=cost_per_turn, allow_short=allow_short)
        rows.append(
            {
                **head,
                "n": m.n,
                "ic": m.ic,
                "ic_t": m.ic_t,
                "MASE": m.mase,
                "dir_acc": m.directional_accuracy,
                "coverage": m.coverage,
                "flat_share": m.flat_share,
                "RMSE_ret": m.rmse,
                "sharpe_net": m.sharpe_after_costs,
                "sharpe_gross": m.sharpe_frictionless,
                "turnover": m.turnover,
                "status": "ok",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # The baselines are not just more rows — they are the reference lines, and
    # sorting them into the pack by MASE would bury exactly what a reader needs
    # to compare against.
    baseline = is_baseline(table["model"])
    pinned = pd.concat(
        [
            table[table["model"].str.contains(name, case=False, regex=False)]
            for name in BASELINE_NAMES
        ]
    )
    return pd.concat(
        [pinned, table[~baseline].sort_values("MASE", na_position="last")],
        ignore_index=True,
    )


def render_table(table: pd.DataFrame, *, title: str = "") -> str:
    """Fixed-width rendering for a terminal or a notebook cell."""
    if table.empty:
        return (title + "\n" if title else "") + "(no results)"

    display = table.copy()
    fmt = {
        "ic": "{:+.3f}",
        "ic_t": "{:+.1f}",
        "MASE": "{:.4f}",
        "dir_acc": "{:.1%}",
        "coverage": "{:.0%}",
        "flat_share": "{:.0%}",
        "RMSE_ret": "{:.5f}",
        "sharpe_net": "{:+.2f}",
        "sharpe_gross": "{:+.2f}",
        "turnover": "{:.2f}",
    }
    for col, spec in fmt.items():
        if col in display:
            display[col] = display[col].map(lambda v, s=spec: "—" if pd.isna(v) else s.format(v))
    for col in ("upstream", "flat_share", "coverage"):
        if col in display:
            display = display.drop(columns=[col])
    if (display["status"] == "ok").all():
        display = display.drop(columns=["status"])

    body = display.to_string(index=False)
    width = max(len(line) for line in body.splitlines())
    out = []
    if title:
        out += [title, "=" * min(width, len(title) + 20)]
    out.append(body)

    summary = summarise_skill(table)
    out.append("")
    out.append(
        f"IC (forecast vs realised return): mean {summary['mean_ic']:+.3f} over "
        f"{summary['ran']} models, {summary['positive_ic']} positive, "
        f"{summary['significant']} with |t| > 1.96 "
        f"(~{summary['expected_false_positives']:.0f} expected by chance)."
    )
    if summary["reference"]:
        out.append(
            "Reference lines: "
            + " · ".join(f"{k} Sharpe {v:+.2f}" for k, v in summary["reference"].items())
            + f" — {summary['beat_always_long']} of {summary['ran']} models beat holding the share."
        )
    failed = table[table["status"] != "ok"] if "status" in table else table.iloc[0:0]
    if len(failed):
        out.append(f"{len(failed)} model(s) failed to run: {', '.join(failed['model'])}")
    return "\n".join(out)


def summarise_skill(table: pd.DataFrame) -> dict:
    """Machine-readable version of the headline claim (acceptance criterion 9).

    Reports skill as measured by IC, and how the field did against the two
    reference lines. Note `significant` against `expected_false_positives`
    before reading anything into it: 22 models over one ticker throw up a
    2-sigma result about one time in one, and models built on a shared feature
    set are not independent draws.
    """
    empty = {
        "models": 0,
        "ran": 0,
        "mean_ic": float("nan"),
        "positive_ic": 0,
        "significant": 0,
        "expected_false_positives": 0.0,
        "beat_always_long": 0,
        "reference": {},
        "leaders": [],
    }
    if table.empty:
        return empty
    ok = table[table["status"] == "ok"] if "status" in table else table
    field = ok[~is_baseline(ok["model"])]
    if field.empty:
        return empty

    reference = {}
    for name in BASELINE_NAMES:
        row = ok[ok["model"].str.contains(name, case=False, regex=False)]
        if len(row):
            reference[name] = float(row["sharpe_net"].iloc[0])

    long_sharpe = reference.get(LONG_NAME)
    ic = field["ic"].dropna()
    significant = field[field["ic_t"].abs() > 1.96]
    return {
        "models": int(len(table[~is_baseline(table["model"])])),
        "ran": int(len(field)),
        "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
        "positive_ic": int((ic > 0).sum()),
        "significant": int(len(significant)),
        "expected_false_positives": 0.05 * len(field),
        "beat_always_long": (
            int((field["sharpe_net"] > long_sharpe).sum()) if long_sharpe is not None else 0
        ),
        "reference": reference,
        "leaders": significant.sort_values("ic", ascending=False)["model"].tolist(),
    }
