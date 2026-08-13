"""Results tables.

`NaiveLag` is pinned to the top of every table and a `beats_naive` column is
computed for you, so "did this model actually do anything" needs no
interpretation — which was requirement R9's whole point.
"""

from __future__ import annotations

import pandas as pd

from .runner import EvalResult

NAIVE_NAME = "naive_lag"


def results_table(
    results: list[EvalResult],
    *,
    cost_per_turn: float = 0.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """One row per model, naive pinned first, sorted by MASE thereafter."""
    rows = []
    for r in results:
        if not r.ok:
            rows.append(
                {
                    "model": r.model_name,
                    "symbol": r.symbol,
                    "upstream": r.upstream,
                    "folds": len(r.folds),
                    "n": 0,
                    "MASE": float("nan"),
                    "beats_naive": False,
                    "dir_acc": float("nan"),
                    "coverage": float("nan"),
                    "RMSE_ret": float("nan"),
                    "sharpe_net": float("nan"),
                    "sharpe_gross": float("nan"),
                    "turnover": float("nan"),
                    "status": r.error or "no folds",
                }
            )
            continue
        m = r.pooled(cost_per_turn=cost_per_turn, allow_short=allow_short)
        rows.append(
            {
                "model": r.model_name,
                "symbol": r.symbol,
                "upstream": r.upstream,
                "folds": len(r.folds),
                "n": m.n,
                "MASE": m.mase,
                "beats_naive": bool(m.mase < 1.0),
                "dir_acc": m.directional_accuracy,
                "coverage": m.coverage,
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

    # The baseline is not just another row — it is the reference line.
    is_naive = table["model"].str.contains(NAIVE_NAME, case=False, regex=False)
    table = pd.concat(
        [table[is_naive], table[~is_naive].sort_values("MASE", na_position="last")],
        ignore_index=True,
    )
    return table


def render_table(table: pd.DataFrame, *, title: str = "") -> str:
    """Fixed-width rendering for a terminal or a notebook cell."""
    if table.empty:
        return (title + "\n" if title else "") + "(no results)"

    display = table.copy()
    fmt = {
        "MASE": "{:.4f}",
        "dir_acc": "{:.1%}",
        "coverage": "{:.0%}",
        "RMSE_ret": "{:.5f}",
        "sharpe_net": "{:+.2f}",
        "sharpe_gross": "{:+.2f}",
        "turnover": "{:.2f}",
    }
    for col, spec in fmt.items():
        if col in display:
            display[col] = display[col].map(lambda v, s=spec: "—" if pd.isna(v) else s.format(v))
    if "beats_naive" in display:
        display["beats_naive"] = display.apply(
            lambda r: "—" if r["status"] != "ok" else ("YES" if r["beats_naive"] else "no"), axis=1
        )
    if "upstream" in display:
        display = display.drop(columns=["upstream"])
    if (display["status"] == "ok").all():
        display = display.drop(columns=["status"])

    body = display.to_string(index=False)
    width = max(len(line) for line in body.splitlines())
    out = []
    if title:
        out += [title, "=" * min(width, len(title) + 20)]
    out.append(body)

    ok = table[table["status"] == "ok"] if "status" in table else table
    winners = ok[ok["beats_naive"]] if "beats_naive" in ok else ok.iloc[0:0]
    non_naive = ok[~ok["model"].str.contains(NAIVE_NAME, case=False, regex=False)]
    winners = winners[~winners["model"].str.contains(NAIVE_NAME, case=False, regex=False)]
    out.append("")
    out.append(
        f"{len(winners)} of {len(non_naive)} models beat NaiveLag on MASE out-of-sample"
        + (f": {', '.join(winners['model'])}" if len(winners) else " — none did.")
    )
    failed = table[table["status"] != "ok"] if "status" in table else table.iloc[0:0]
    if len(failed):
        out.append(f"{len(failed)} model(s) failed to run: {', '.join(failed['model'])}")
    return "\n".join(out)


def summarise_beats(table: pd.DataFrame) -> dict:
    """Machine-readable version of the headline claim (acceptance criterion 9)."""
    if table.empty:
        return {"models": 0, "ran": 0, "beat_naive": 0, "winners": []}
    ok = table[table["status"] == "ok"] if "status" in table else table
    non_naive = ok[~ok["model"].str.contains(NAIVE_NAME, case=False, regex=False)]
    winners = non_naive[non_naive["beats_naive"]]
    return {
        "models": int(
            len(table[~table["model"].str.contains(NAIVE_NAME, case=False, regex=False)])
        ),
        "ran": int(len(non_naive)),
        "beat_naive": int(len(winners)),
        "winners": winners["model"].tolist(),
    }
