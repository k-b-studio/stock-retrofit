# Upstream mapping — all 62 notebooks

Spec acceptance criterion 6: every notebook in
`external-repo/Stock-Prediction-Models-master/` maps to a config here, or
carries an explicit "not ported, because —".

The count: **62 notebooks**. 41 are ported into 9 parameterised families with
one config each; 21 are not ported, each with a reason below.

**What "ported" means here.** The upstream notebooks are treated as
specifications of *architecture* and nothing else. None of their methodology
survives: no scaler is fit before a split, no result is reported in-sample, and
the accuracy metric they report is not implemented as a headline anywhere. A
config that maps to a notebook reproduces its model, not its numbers — and
reproducing its numbers would be a failure, since those numbers are leaked.

---

## `deep-learning/` — 18 forecasting notebooks, all ported

Collapsed into 4 families. The 18 are the cross product
`{cell} x {bidirectional} x {paths}` plus decoder variants — 18 files, one train
loop, differing hyperparameters.

| # | Upstream notebook | Config | Family | Parameters |
|---|---|---|---|---|
| 1 | `1.lstm.ipynb` | `models/01_lstm.yaml` | `recurrent` | cell=lstm |
| 2 | `2.bidirectional-lstm.ipynb` | `models/02_bidirectional_lstm.yaml` | `recurrent` | cell=lstm, bidirectional |
| 3 | `3.lstm-2path.ipynb` | `models/03_lstm_2path.yaml` | `recurrent` | cell=lstm, paths=2 |
| 4 | `4.gru.ipynb` | `models/04_gru.yaml` | `recurrent` | cell=gru |
| 5 | `5.bidirectional-gru.ipynb` | `models/05_bidirectional_gru.yaml` | `recurrent` | cell=gru, bidirectional |
| 6 | `6.gru-2path.ipynb` | `models/06_gru_2path.yaml` | `recurrent` | cell=gru, paths=2 |
| 7 | `7.vanilla.ipynb` | `models/07_vanilla.yaml` | `recurrent` | cell=rnn |
| 8 | `8.bidirectional-vanilla.ipynb` | `models/08_bidirectional_vanilla.yaml` | `recurrent` | cell=rnn, bidirectional |
| 9 | `9.vanilla-2path.ipynb` | `models/09_vanilla_2path.yaml` | `recurrent` | cell=rnn, paths=2 |
| 10 | `10.lstm-seq2seq.ipynb` | `models/10_lstm_seq2seq.yaml` | `seq2seq` | cell=lstm |
| 11 | `11.bidirectional-lstm-seq2seq.ipynb` | `models/11_bidirectional_lstm_seq2seq.yaml` | `seq2seq` | cell=lstm, bidirectional |
| 12 | `12.lstm-seq2seq-vae.ipynb` | `models/12_lstm_seq2seq_vae.yaml` | `seq2seq` | cell=lstm, vae |
| 13 | `13.gru-seq2seq.ipynb` | `models/13_gru_seq2seq.yaml` | `seq2seq` | cell=gru |
| 14 | `14.bidirectional-gru-seq2seq.ipynb` | `models/14_bidirectional_gru_seq2seq.yaml` | `seq2seq` | cell=gru, bidirectional |
| 15 | `15.gru-seq2seq-vae.ipynb` | `models/15_gru_seq2seq_vae.yaml` | `seq2seq` | cell=gru, vae |
| 16 | `16.attention-is-all-you-need.ipynb` | `models/16_attention_is_all_you_need.yaml` | `attention` | transformer encoder |
| 17 | `17.cnn-seq2seq.ipynb` | `models/17_cnn_seq2seq.yaml` | `conv` | dilated=false |
| 18 | `18.dilated-cnn-seq2seq.ipynb` | `models/18_dilated_cnn_seq2seq.yaml` | `conv` | dilated=true |

### `deep-learning/` — supporting files and the two bonus notebooks

| Upstream file | Status |
|---|---|
| `how-to-forecast.ipynb` | **Not ported** — a tutorial walkthrough of notebook 1's pipeline, including the scaler-before-split ordering. It documents the defect rather than adding a model. |
| `sentiment-consensus.ipynb` | **Not ported** — needs a sentiment source. No Thai-language sentiment feed is specified and inventing one would be worse than omitting it. Explicit non-goal in the spec. |
| `autoencoder.py` | Ported as a component — the autoencoder inside `models/stacking.py`. |
| `dnc.py`, `access.py`, `addressing.py` | **Not ported** — Differentiable Neural Computer support code, imported by no notebook in the catalogue the spec asks for. |
| `util.py` | **Not ported** — plotting and CSV helpers, superseded by `eval/report.py`. |

---

## `agent/` — 23 trading notebooks, all ported

Collapsed into 6 families. The Q-learning set alone is 11 notebooks sharing one
replay-buffer skeleton with a swapped head.

| # | Upstream notebook | Config | Family | Parameters |
|---|---|---|---|---|
| 1 | `1.turtle-agent.ipynb` | `agents/01_turtle.yaml` | `turtle` | count=20 |
| 2 | `2.moving-average-agent.ipynb` | `agents/02_moving_average.yaml` | `moving_average` | 5/20 |
| 3 | `3.signal-rolling-agent.ipynb` | `agents/03_signal_rolling.yaml` | `signal_rolling` | delay=4 |
| 4 | `4.policy-gradient-agent.ipynb` | `agents/04_policy_gradient.yaml` | `policy_gradient` | REINFORCE |
| 5 | `5.q-learning-agent.ipynb` | `agents/05_q_learning.yaml` | `q_learning` | (no flags) |
| 6 | `6.evolution-strategy-agent.ipynb` | `agents/06_evolution_strategy.yaml` | `evolution_strategy` | pop=20, gen=40 |
| 7 | `7.double-q-learning-agent.ipynb` | `agents/07_double_q_learning.yaml` | `q_learning` | double |
| 8 | `8.recurrent-q-learning-agent.ipynb` | `agents/08_recurrent_q_learning.yaml` | `q_learning` | recurrent |
| 9 | `9.double-recurrent-q-learning-agent.ipynb` | `agents/09_double_recurrent_q_learning.yaml` | `q_learning` | double+recurrent |
| 10 | `10.duel-q-learning-agent.ipynb` | `agents/10_duel_q_learning.yaml` | `q_learning` | duel |
| 11 | `11.double-duel-q-learning-agent.ipynb` | `agents/11_double_duel_q_learning.yaml` | `q_learning` | double+duel |
| 12 | `12.duel-recurrent-q-learning-agent.ipynb` | `agents/12_duel_recurrent_q_learning.yaml` | `q_learning` | duel+recurrent |
| 13 | `13.double-duel-recurrent-q-learning-agent.ipynb` | `agents/13_double_duel_recurrent_q_learning.yaml` | `q_learning` | double+duel+recurrent |
| 14 | `14.actor-critic-agent.ipynb` | `agents/14_actor_critic.yaml` | `actor_critic` | (no flags) |
| 15 | `15.actor-critic-duel-agent.ipynb` | `agents/15_actor_critic_duel.yaml` | `actor_critic` | duel |
| 16 | `16.actor-critic-recurrent-agent.ipynb` | `agents/16_actor_critic_recurrent.yaml` | `actor_critic` | recurrent |
| 17 | `17.actor-critic-duel-recurrent-agent.ipynb` | `agents/17_actor_critic_duel_recurrent.yaml` | `actor_critic` | duel+recurrent |
| 18 | `18.curiosity-q-learning-agent.ipynb` | `agents/18_curiosity_q_learning.yaml` | `q_learning` | curiosity |
| 19 | `19.recurrent-curiosity-q-learning-agent.ipynb` | `agents/19_recurrent_curiosity_q_learning.yaml` | `q_learning` | recurrent+curiosity |
| 20 | `20.duel-curiosity-q-learning-agent.ipynb` | `agents/20_duel_curiosity_q_learning.yaml` | `q_learning` | duel+curiosity |
| 21 | `21.neuro-evolution-agent.ipynb` | `agents/21_neuro_evolution.yaml` | `neuro_evolution` | novelty=false |
| 22 | `22.neuro-evolution-novelty-search-agent.ipynb` | `agents/22_neuro_evolution_novelty.yaml` | `neuro_evolution` | novelty=true |
| 23 | `23.abcd-strategy-agent.ipynb` | `agents/23_abcd.yaml` | `abcd` | 0.382-0.886 |

| Upstream file | Status |
|---|---|
| `updated-NES-google.ipynb` | **Not ported** — a re-run of notebook 6 on a different US ticker. Same code, different CSV; the config covers it. |

---

## `stacking/` — 2 notebooks, both ported

| Upstream notebook | Config | Family |
|---|---|---|
| `stack-rnn-arima-xgb.ipynb` | `models/19_stack_rnn_arima_xgb.yaml` | `stack_rnn_arima_xgb` |
| `stack-encoder-ensemble-xgb.ipynb` | `models/20_stack_encoder_ensemble_xgb.yaml` | `stack_encoder_ensemble_xgb` |
| `model.py`, `autoencoder.py` | Ported as components inside `models/stacking.py`. |

Their ARIMA and XGBoost components are also exposed standalone
(`models/21_arima.yaml`, `models/22_xgboost.yaml`) so the stack can be compared
against its own parts — which the upstream notebooks never did.

---

## Not ported — 21 notebooks, with reasons

### `simulation/` — 5 notebooks

| Notebook | Reason |
|---|---|
| `monte-carlo-drift.ipynb` | **Not ported, because** it is a *simulation* of price paths, not a forecaster or an agent. It produces no prediction to score against `NaiveLag` and no trades to run through `SETMarket`, so it has no place on either results table. Self-contained and least affected by the leakage defect — worth reading, nothing to retrofit. |
| `monte-carlo-dynamic-volatility.ipynb` | Same. |
| `monte-carlo-simple.ipynb` | Same. |
| `multivariate-drift-monte-carlo.ipynb` | Same, plus it needs a multi-asset panel that the three-symbol universe does not motivate. |
| `portfolio-optimization.ipynb` | **Not ported, because** portfolio construction across a universe is a different problem from the single-symbol forecasting and agent work specified here. Would need a universe wider than three correlated bank stocks to mean anything. |

### `misc/` — 6 notebooks

| Notebook | Reason |
|---|---|
| `bitcoin-analysis-lstm.ipynb` | **Not ported, because** it is notebook 1's architecture on a crypto series. Out of universe; `01_lstm.yaml` covers the model. |
| `fashion-forecasting.ipynb` | **Not ported, because** it forecasts retail demand, not equities. Not in scope. |
| `kijang-emas-bank-negara.ipynb` | **Not ported, because** it is a Malaysian gold-price study. Out of universe. |
| `tesla-study.ipynb` | **Not ported, because** it is exploratory single-name analysis of a US stock, and it depends on the unmaintained `mpl_finance`. |
| `outliers.ipynb` | **Not ported as a notebook**, but its intent is superseded: outlier detection now lives in `data/quality.py` as an enforced gate rather than an exploratory chart. |
| `overbought-oversold.ipynb` | **Not ported as a notebook** — the RSI logic it explores is a feature in `eval/preprocessing.py` and a rule in the `mean_reversion` agent. |
| `which-stock.ipynb` | **Not ported, because** it is a universe-selection exploration over US tickers. The universe here is fixed by the spec. |

### `free-agent/` — 2 notebooks

| Notebook | Reason |
|---|---|
| `evolution-strategy-agent.ipynb` | **Not ported, because** it duplicates `agent/6` with a different data loader. Covered by `agents/06_evolution_strategy.yaml`. |
| `evolution-strategy-bayesian-agent.ipynb` | **Not ported, because** it adds a Bayesian hyperparameter search (`bayes_opt`) around `agent/6`. Hyperparameter search belongs outside the model catalogue, and doing it honestly would require an inner validation split per fold — worth adding later, but it is a harness feature, not a 24th agent. |

### `realtime-agent/` — 2 notebooks + `app.py`

| File | Reason |
|---|---|
| `realtime-evolution-strategy.ipynb` | **Not ported, because** it serves a live agent over Flask. The spec's non-goals exclude live execution, and `SETMarket` is explicitly a simulator with no order path. |
| `request.ipynb` | **Not ported** — a client for the above. |
| `app.py`, `model.pkl` | **Not ported** — Flask server and a pickled model for the above. |

### `stock-forecasting-js/`

**Not ported, because** it is a browser port of the catalogue in JavaScript.
Explicit non-goal in the spec.

### `dataset/` — 17 CSVs

**Not ported, because** they are US and FX series (AMD, FB, GOOG, TSLA, TWTR,
oil, usd-myr, …). The universe here is SCB, KBANK and BAY, sourced live and
cached to Parquet. `BTC-sentiment.csv` additionally belongs to the sentiment
work that is an explicit non-goal.

---

## Summary

| Category | Notebooks | Ported |
|---|---|---|
| `deep-learning/` forecasting models | 18 | 18 |
| `deep-learning/` bonus (how-to, sentiment) | 2 | 0 |
| `agent/` | 23 | 23 |
| `agent/updated-NES-google` | 1 | 0 |
| `stacking/` | 2 | 2 |
| `simulation/` | 5 | 0 |
| `misc/` | 7 | 0 |
| `free-agent/` | 2 | 0 |
| `realtime-agent/` | 2 | 0 |
| **Total** | **62** | **43** |

43 upstream notebooks map to 45 configs (43 plus the two required baselines,
`00_naive_lag` and `00_buy_and_hold`, which have no upstream counterpart because
upstream had no baselines — which is exactly why its results looked good).
