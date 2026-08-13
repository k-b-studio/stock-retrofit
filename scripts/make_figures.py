"""Render the results figures into `results/figures/`.

    python scripts/make_figures.py

Four figures, each answering one question a reader actually has:

  1. does anything beat the naive lag?          -> MASE, small multiples
  2. do the forecasts call direction?           -> directional accuracy vs 50%
  3. what do SET frictions cost?                -> dumbbell, free -> after costs
  4. what does "no skill" look like up close?   -> forecast vs realised returns

Design rules followed here, from the data-viz method:

* **Form before color.** 1 and 2 are magnitude-against-a-reference, so they use
  one hue plus a de-emphasis gray rather than a categorical rainbow. 3 is
  before-after per item, which is a dumbbell. 4 is a distribution/relationship.
* **The palette is validated, not eyeballed.** Blue `#2a78d6` and orange
  `#eb6834` are slots 1-2 of the reference categorical theme; the pair clears
  the all-pairs CVD and normal-vision gates on this surface.
* **One axis, never two.** No dual-scale plots anywhere.
* **Identity is never color-alone** — every series is direct-labelled or named
  in the axis, and figure 3 encodes before/after by shape as well as hue.
* **Recessive grid, thin marks, selective labels.** No number on every point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SYMBOLS = ["KBANK", "SCB", "BAY"]

# --- palette (validated: see references/palette.md, slots 1-2) --------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
RULE = "#b8b6ae"


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.dpi": 200,
        }
    )


def load(kind: str, symbol: str) -> pd.DataFrame | None:
    path = RESULTS / f"{kind}-{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def short(name: str) -> str:
    """Strip the leading index so labels read as names, not filenames."""
    return name.split("_", 1)[1].replace("_", " ") if "_" in name else name


def caption(fig, text: str) -> None:
    fig.text(0.011, 0.012, text, ha="left", va="bottom", fontsize=7.4, color=MUTED)


# ---------------------------------------------------------------- figure 1


def fig_mase() -> Path:
    """MASE per model, one panel per ticker, against the naive-lag line at 1.00.

    Emphasis form: the reference line is the subject, so the bars are one hue
    and anything beating the line would flip to orange. Nothing does — which is
    the finding, and the chart should show that plainly rather than hide it.
    """
    frames = {s: load("evaluate", s) for s in SYMBOLS}
    frames = {s: d for s, d in frames.items() if d is not None and len(d)}
    if not frames:
        raise SystemExit("no evaluate-*.csv found — run `cli evaluate --all` first")

    fig, axes = plt.subplots(
        1, len(frames), figsize=(4.6 * len(frames), 6.6), sharex=False
    )
    axes = np.atleast_1d(axes)

    for ax, (symbol, df) in zip(axes, frames.items(), strict=True):
        d = df[~df["model"].str.contains("naive_lag")].sort_values("MASE", ascending=False)
        y = np.arange(len(d))
        beats = d["MASE"] < 1.0
        colors = [ORANGE if b else BLUE for b in beats]

        # Lollipop: thin stem from the reference line to the value, round cap.
        for yi, (val, col) in enumerate(zip(d["MASE"], colors, strict=True)):
            ax.plot([1.0, val], [yi, yi], color=col, lw=1.6, alpha=0.45, zorder=2,
                    solid_capstyle="round")
        ax.scatter(d["MASE"], y, s=42, color=colors, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)

        ax.axvline(1.0, color=INK, lw=1.4, zorder=4)
        ax.text(1.0, len(d) - 0.2, "  naive lag = 1.00", ha="left", va="top",
                fontsize=8, color=INK, fontweight="600")

        ax.set_yticks(y)
        ax.set_yticklabels([short(m) for m in d["model"]], fontsize=7.6)
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.set_xlabel("MASE  (model MAE ÷ naive-lag MAE)  —  lower is better")
        n_folds = int(d["folds"].iloc[0])
        ax.set_title(f"{symbol}   ·   {n_folds} folds, {int(d['n'].iloc[0])} out-of-sample days",
                     loc="left", pad=10)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

        lo = min(0.99, float(d["MASE"].min()) - 0.01)
        ax.set_xlim(lo, float(d["MASE"].max()) + 0.02)

        # Shade the region that would represent skill, so its emptiness reads.
        # No label inside it: the band is ~1% of the axis and any text overflows
        # into the marks. The xlabel and the annotated line carry the meaning.
        ax.axvspan(lo, 1.0, color=ORANGE, alpha=0.07, zorder=0)

    fig.suptitle(
        "No model beats a naive lag out-of-sample",
        x=0.011, ha="left", fontsize=15, fontweight="700", y=0.985,
    )
    fig.text(
        0.011, 0.945,
        "Lower is better. MASE below 1.00 means the model predicts next-day returns more accurately "
        "than “tomorrow’s price equals today’s”.\nEvery model on every ticker sits to the right of the "
        "line. The shaded band is where skill would appear; it is empty.",
        ha="left", va="top", fontsize=9, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.028, 1, 0.925])
    caption(fig, "stock-retrofit · walk-forward, fold-local scaling · returns not price levels")

    out = FIGURES / "01_mase_vs_naive.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 2


def fig_direction() -> Path:
    """Directional accuracy against the 50% coin flip.

    Diverging job — above/below a baseline — so the reference is the midpoint
    and the two sides get the warm/cool poles. Almost everything lands below.
    """
    rows = []
    for symbol in SYMBOLS:
        d = load("evaluate", symbol)
        if d is None:
            continue
        d = d[~d["model"].str.contains("naive_lag")].dropna(subset=["dir_acc"])
        for _, r in d.iterrows():
            rows.append({"symbol": symbol, "model": short(r["model"]), "acc": r["dir_acc"]})
    if not rows:
        raise SystemExit("no evaluate-*.csv found")
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    # Reverse the list rather than inverting the axis: with y=0 at the bottom
    # this puts KBANK on top (matching every other figure) while the "+0.37"
    # label offsets still render *above* their rules.
    order = SYMBOLS[::-1]
    rng = np.random.default_rng(0)

    for i, symbol in enumerate(order):
        sub = df[df["symbol"] == symbol]
        jitter = rng.uniform(-0.13, 0.13, len(sub))
        above = sub["acc"] >= 0.5
        ax.scatter(sub["acc"], np.full(len(sub), i) + jitter,
                   s=58, color=[ORANGE if a else BLUE for a in above],
                   edgecolor=SURFACE, linewidth=1.1, zorder=3, alpha=0.9)
        mean = sub["acc"].mean()
        ax.plot([mean, mean], [i - 0.3, i + 0.3], color=INK, lw=2.0, zorder=4)
        ax.text(mean, i + 0.37, f"mean {mean:.1%}", ha="center", va="bottom",
                fontsize=8, color=INK, fontweight="600")

    ax.axvline(0.5, color=INK, lw=1.4, zorder=5)
    ax.text(0.5, len(order) - 0.42, "  coin flip = 50%", ha="left", va="center",
            fontsize=8.4, color=INK, fontweight="600")
    ax.axvspan(0.5, 0.55, color=ORANGE, alpha=0.06, zorder=0)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    ax.set_ylim(-0.6, len(order) - 0.35)
    ax.set_xlim(0.315, 0.55)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("directional accuracy — share of days the sign of the forecast was right")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="below a coin flip"),
        # Kept in the legend though it has no members: it documents the encoding,
        # and "(none)" is the finding rather than an omission.
        Line2D([], [], marker="o", ls="", color=ORANGE, markersize=8,
               label="at or above  (none)"),
        Line2D([], [], color=INK, lw=2, label="ticker mean"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.4, ncol=3,
              bbox_to_anchor=(1.0, -0.30))

    fig.suptitle("The forecasts call direction worse than chance",
                 x=0.011, ha="left", fontsize=15, fontweight="700", y=0.99)
    fig.text(0.011, 0.925,
             "One dot per model. A model trained to minimise squared error on a near-random-walk "
             "return series has no reason to\ncall direction well — and does not. The naive lag is "
             "absent because it abstains: it never makes a directional call.",
             ha="left", va="top", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    caption(fig, "stock-retrofit · 22 models × 3 tickers, out-of-sample folds only")

    out = FIGURES / "02_directional_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 3


def fig_friction() -> Path:
    """Dumbbell: frictionless return -> return after SET frictions, per agent.

    Before-after per item, which is the dumbbell's exact job. One hue in two
    roles plus shape, so the two ends are distinguishable without color.
    """
    panels = [(s, load("backtest", s)) for s in SYMBOLS]
    panels = [(s, d) for s, d in panels if d is not None and len(d)]
    if not panels:
        raise SystemExit("no backtest-*.csv found — run `cli backtest --all` first")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 7.0))
    axes = np.atleast_1d(axes)

    for ax, (symbol, d) in zip(axes, panels, strict=True):
        d = d.sort_values("ret_friction")
        y = np.arange(len(d))
        is_bh = d["agent"].str.contains("buy_and_hold").to_numpy()

        for yi, (free, net) in enumerate(zip(d["ret_frictionless"], d["ret_friction"], strict=True)):
            ax.plot([net, free], [yi, yi], color=MUTED, lw=1.5, alpha=0.55, zorder=2,
                    solid_capstyle="round")
        ax.scatter(d["ret_frictionless"], y, s=46, marker="o", color=ORANGE,
                   edgecolor=SURFACE, linewidth=1.1, zorder=3, label="frictionless")
        ax.scatter(d["ret_friction"], y, s=52, marker="D", color=BLUE,
                   edgecolor=SURFACE, linewidth=1.1, zorder=4, label="after SET frictions")

        ax.axvline(0, color=INK, lw=1.2, zorder=5)

        labels = []
        for name, bh in zip(d["agent"], is_bh, strict=True):
            labels.append(f"{short(name)}  (baseline)" if bh else short(name))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.6)
        for tick, bh in zip(ax.get_yticklabels(), is_bh, strict=True):
            if bh:
                tick.set_fontweight("700")
                tick.set_color(INK)

        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("mean return per fold")
        n_beat = int((d.loc[~is_bh, "ret_friction"] > d.loc[is_bh, "ret_friction"].iloc[0]).sum())
        ax.set_title(f"{symbol}   ·   {n_beat} of {int((~is_bh).sum())} agents beat buy-and-hold",
                     loc="left", pad=10)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

    # Figure-level legend at bottom-right: the caption owns bottom-left, the
    # title owns the top, and every panel's interior is occupied by marks.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.995, 0.004),
               ncol=2, fontsize=9, handletextpad=0.5, columnspacing=2.0)

    fig.suptitle("What SET frictions cost — and why buy-and-hold is hard to beat",
                 x=0.011, ha="left", fontsize=15, fontweight="700", y=0.985)
    fig.text(0.011, 0.945,
             "Each line is one agent: the circle is its return in a market with no board lot, no tick "
             "table, no commission and no VAT;\nthe diamond is the same strategy charged real SET "
             "costs. The length of the line is the cost of that fantasy.",
             ha="left", va="top", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])
    caption(fig, "stock-retrofit · agents trained on training folds only, scored on held-out folds "
                 "· cost parameters reconstructed, not verified")

    out = FIGURES / "03_friction_gap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 4


def fig_forecast_reality(symbol: str = "KBANK", config: str = "13_gru_seq2seq") -> Path:
    """What "no skill" looks like up close, for the single best-MASE model.

    Left: forecast vs realised next-day return, with the 1:1 line a skilful
    model would sit on. Right: the two distributions, which is where the
    behaviour actually shows — the forecasts are an order of magnitude smaller
    than the moves they are predicting.
    """
    from stock_retrofit.config import EvalConfig, ModelSpec
    from stock_retrofit.data import load as load_bars
    from stock_retrofit.eval import run_walk_forward

    cfg_path = ROOT / "configs" / "models" / f"{config}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    eval_cfg = EvalConfig.load()
    bars = load_bars(symbol)
    result = run_walk_forward(
        ModelSpec.load(cfg_path).build(), bars,
        splitter=eval_cfg.splitter(), window=eval_cfg.window(),
        features=eval_cfg.features, symbol=symbol, seed=eval_cfg.seed,
    )
    if not result.ok:
        raise SystemExit(f"{config} failed: {result.error}")
    preds = result.predictions_frame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    lim = float(np.abs(np.concatenate([preds["y_true"], preds["y_pred"]])).max()) * 1.06
    ax1.plot([-lim, lim], [-lim, lim], color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax1.text(lim * 0.62, -lim * 0.30, "a perfect forecast\nwould lie on this line",
             ha="left", va="center", fontsize=7.8, color=INK_2, style="italic",
             bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="none", alpha=0.92))
    ax1.axhline(0, color=RULE, lw=0.9, zorder=1)
    ax1.axvline(0, color=RULE, lw=0.9, zorder=1)
    ax1.scatter(preds["y_true"], preds["y_pred"], s=17, color=BLUE, alpha=0.42,
                edgecolor="none", zorder=3)

    corr = float(np.corrcoef(preds["y_true"], preds["y_pred"])[0, 1])
    ax1.text(-lim * 0.96, lim * 0.9,
             f"correlation  r = {corr:+.3f}", ha="left", va="top",
             fontsize=9.5, color=INK, fontweight="700")

    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax1.set_xlabel("realised next-day return")
    ax1.set_ylabel("forecast next-day return")
    ax1.set_title("Forecast vs reality", loc="left", pad=8)
    ax1.set_axisbelow(True)

    bins = np.linspace(-lim, lim, 61)
    ax2.hist(preds["y_true"], bins=bins, color=MUTED, alpha=0.55, label="realised returns")
    ax2.hist(preds["y_pred"], bins=bins, color=BLUE, alpha=0.9, label="forecasts")
    ax2.axvline(0, color=RULE, lw=0.9)
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("next-day return")
    ax2.set_ylabel("days")
    ax2.set_title("The forecasts barely move", loc="left", pad=8)
    ax2.legend(fontsize=8.6, loc="upper right")
    ax2.set_axisbelow(True)

    sd_true, sd_pred = preds["y_true"].std(), preds["y_pred"].std()
    ax2.text(0.02, 0.72,
             f"realised sd  {sd_true:.2%}\nforecast sd  {sd_pred:.2%}\n"
             f"{sd_true / max(sd_pred, 1e-12):.0f}x narrower",
             transform=ax2.transAxes, fontsize=8.8, color=INK_2, va="top", linespacing=1.5)

    fig.suptitle(f"Up close: the best-scoring model on {symbol}",
                 x=0.011, ha="left", fontsize=15, fontweight="700", y=0.99)
    fig.text(0.011, 0.918,
             f"“{short(config)}” had the lowest MASE of all 22 models — and it still loses to the naive lag. "
             "This is what that looks like:\nforecasts uncorrelated with what happens next, and hedged so far "
             "toward zero they can barely be wrong.",
             ha="left", va="top", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.035, 1, 0.885])
    caption(fig, f"stock-retrofit · {symbol} · {len(preds)} out-of-sample days across "
                 f"{preds['fold'].nunique()} walk-forward folds")

    out = FIGURES / "04_forecast_reality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (fig_mase, fig_direction, fig_friction, fig_forecast_reality):
        path = fn()
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
