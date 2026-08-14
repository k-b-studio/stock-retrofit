"""The spec's acceptance criteria, made enforceable.

`specs/thai-set-retrofit.md` lists nine. The ones that are structural facts about
the repository are asserted here so they cannot quietly regress. The ones that
are claims about *results* (criteria 4, 5, 7, 9) are covered by the smoke tests
below plus the generated `results/final-report.md`.

Criterion 3 — `docs/settrade-api-notes.md` answering the four R2 questions with a
source link — **cannot be met in this environment** and the test asserts the
honest thing instead: that the document exists and states plainly that the
questions are open. See the file itself for why.
"""

from __future__ import annotations

import subprocess

import pytest

from stock_retrofit.paths import AGENT_CONFIG_DIR, DOCS_DIR, MODEL_CONFIG_DIR, ROOT

# -- criterion 1: pytest green, including the leakage test -------------------
# (Asserted by the suite passing at all; the injection check is documented in
#  tests/test_no_leakage.py::test_upstream_bug_is_detected.)


def test_leakage_test_exists_and_covers_the_upstream_bug():
    source = (ROOT / "tests" / "test_no_leakage.py").read_text()
    assert "MinMaxScaler" in source
    assert "test_upstream_bug_is_detected" in source


# -- criterion 3: the Settrade notes exist and are honest --------------------


def test_settrade_notes_record_the_open_questions():
    notes = (DOCS_DIR / "settrade-api-notes.md").read_text()
    assert "OPEN" in notes, "the four R2 questions must be marked open, not invented"
    assert "yfinance" in notes and "primary" in notes
    assert "acceptance criterion 3" in notes.lower() or "not met" in notes.lower()


# -- criterion 5: every config is loadable and buildable ---------------------


def test_every_model_config_builds():
    from stock_retrofit.config import all_model_specs

    specs = all_model_specs()
    assert len(specs) >= 19, f"expected the 18 upstream models plus a baseline, got {len(specs)}"
    for spec in specs:
        model = spec.build()
        assert model.name == spec.name
        assert callable(model.fit) and callable(model.predict)


def test_every_agent_config_builds():
    from stock_retrofit.config import all_agent_specs

    specs = all_agent_specs()
    assert len(specs) >= 24, f"expected the 23 upstream agents plus a baseline, got {len(specs)}"
    for spec in specs:
        agent = spec.build()
        assert agent.name == spec.name
        assert callable(agent.act)


def test_all_eighteen_upstream_forecasting_notebooks_have_a_config():
    from stock_retrofit.config import all_model_specs

    covered = {s.upstream for s in all_model_specs()}
    for n in range(1, 19):
        assert any(f"/{n}." in u for u in covered), f"deep-learning notebook {n} has no config"


def test_all_twentythree_upstream_agent_notebooks_have_a_config():
    from stock_retrofit.config import all_agent_specs

    covered = {s.upstream for s in all_agent_specs()}
    for n in range(1, 24):
        assert any(f"agent/{n}." in u for u in covered), f"agent notebook {n} has no config"


# -- criterion 6: the upstream mapping accounts for all 62 -------------------


def test_upstream_mapping_covers_every_notebook_or_says_why_not():
    mapping = (DOCS_DIR / "upstream-mapping.md").read_text()
    assert "62" in mapping
    assert "Not ported, because" in mapping or "**Not ported**" in mapping
    for name in [
        "stock-forecasting-js",
        "sentiment-consensus",
        "monte-carlo",
        "realtime-agent",
        "free-agent",
        "updated-NES-google",
    ]:
        assert name in mapping, f"{name} is unaccounted for in the mapping"


# -- criterion 8: no trace of the legacy framework under src/ or configs/ ----


def test_no_legacy_framework_reference_in_source_or_configs():
    """`grep -ri "tensorflow" src/ configs/` must return nothing."""
    result = subprocess.run(
        ["grep", "-ri", "tensorflow", "src/", "configs/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, f"found references:\n{result.stdout}"


def test_no_dead_endpoint_or_rejected_source_in_the_package():
    """The data-layer spec's criterion 8: no Google Finance, no classic.settrade."""
    result = subprocess.run(
        ["grep", "-ri", "googlefinance\\|classic.settrade", "src/", "configs/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, f"found references:\n{result.stdout}"


# -- the baseline is not optional -------------------------------------------


def test_baselines_are_added_to_every_evaluation_even_when_not_requested():
    """Spec R8 / criterion 4: both reference lines appear on the table automatically.

    `naive_lag` is the reference for the forecast as a number, `always_long` for
    it as a position. A table with only the first cannot tell a reader that a
    Sharpe of +2.10 lost to owning the share.
    """
    from stock_retrofit.cli import _with_baselines
    from stock_retrofit.config import ModelSpec

    specs = _with_baselines([ModelSpec.load(MODEL_CONFIG_DIR / "01_lstm.yaml")])
    assert any(s.kind == "naive_lag" for s in specs)
    assert any(s.kind == "always_long" for s in specs)
    assert specs[0].kind == "naive_lag", "the baseline must be first on the table"


def test_baselines_are_not_duplicated_when_already_requested():
    from stock_retrofit.cli import _with_baselines
    from stock_retrofit.config import ModelSpec

    specs = _with_baselines([ModelSpec.load(MODEL_CONFIG_DIR / "00_naive_lag.yaml")])
    assert sum(s.kind == "naive_lag" for s in specs) == 1


def test_buy_and_hold_baseline_exists_for_agents():
    assert (AGENT_CONFIG_DIR / "00_buy_and_hold.yaml").exists()


# -- criterion 7: agents always report both runs ----------------------------


def test_agent_table_always_carries_both_friction_columns():
    import numpy as np
    import pandas as pd

    from stock_retrofit.agents import agent_results_table, build, run_agent_walk_forward
    from stock_retrofit.config import EvalConfig, MarketConfigSpec

    rng = np.random.default_rng(0)
    n = 900
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2019-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
            "symbol": "TEST",
        }
    )
    eval_cfg = EvalConfig.load()
    splitter = type(eval_cfg.splitter())(train_window=600, test_window=60, step=60, max_folds=2)
    result = run_agent_walk_forward(
        build("buy_and_hold", name="buy_and_hold"),
        df,
        splitter=splitter,
        config=MarketConfigSpec.load().build(symbol="TEST"),
        window=eval_cfg.window(),
        symbol="TEST",
    )
    assert result.ok, result.error
    table = agent_results_table([result])
    for column in ("ret_friction", "ret_frictionless", "friction_gap"):
        assert column in table.columns
    assert not table[["ret_friction", "ret_frictionless"]].isna().any().any()


# -- the CLI surface the criteria name by command ---------------------------


@pytest.mark.parametrize(
    "command", ["fetch", "quality", "reconcile", "evaluate", "backtest", "report", "status"]
)
def test_cli_exposes_the_named_commands(command):
    from stock_retrofit.cli import build_parser

    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, type(parser._subparsers._group_actions[0]))
    )
    assert command in subparsers.choices


def test_notebooks_exist_for_every_phase():
    """The user-facing deliverable: one runnable notebook per phase."""
    for name in [
        "01_data",
        "02_harness",
        "03_market",
        "04_models",
        "05_agents",
        "06_report",
        "07_figures",
    ]:
        assert (ROOT / "notebooks" / f"{name}.ipynb").exists(), f"{name}.ipynb missing"


def test_figure_notebook_has_one_figure_per_cell():
    """The figures notebook is structured for re-running a single figure.

    Each figure lives in its own code cell that both renders inline and writes
    its own PNG, so refreshing one chart never re-runs the other three (figure 4
    retrains a model, which is the expensive one).
    """
    import json

    nb = json.loads((ROOT / "notebooks" / "07_figures.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    savers = [c for c in code if "fig.savefig(out" in c]
    assert len(savers) == 4, f"expected 4 figure cells, found {len(savers)}"
    for i, cell in enumerate(savers, 1):
        assert "plt.subplots(" in cell, f"figure cell {i} builds no figure"
    written = {line for c in savers for line in c.splitlines() if "out = FIGURES" in line}
    assert len(written) == 4, "figure cells must write four distinct filenames"
