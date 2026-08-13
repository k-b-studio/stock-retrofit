# stock-retrofit — results

Generated 2026-08-13 11:10 UTC · git `b748feedf7107b02727d4d1f10195a651c243471-dirty` · seed 42

Walk-forward evaluation of the [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) catalogue on Thai SET bank shares, on a harness that does not leak and a backtest that charges SET trading costs.

## How to read this

- **MASE** is MAE(model) / MAE(naive lag) on next-day returns. **Below 1.00 beats the naive lag; at or above 1.00 it does not.** The naive lag is on every table by construction.
- **dir_acc** counts only rows where a model made a directional call. The naive lag abstains everywhere, so its accuracy is undefined rather than zero.
- **sharpe_net** is annualised, after a round-trip cost of 0.336%. **sharpe_gross** charges nothing.
- Splits: 750 training bars, 60-bar test blocks, step 60, up to 8 of the most recent folds — a truncated history yields fewer, and the per-ticker `folds` column says how many actually ran. Every scaler is fit inside its own fold.

> **Cost figures are reconstructed, not verified** against SET's rulebook or a broker schedule (spec R13). Treat them as order-of-magnitude.

## Universe

**KBANK** — Kasikornbank PCL
  - liquidity: Large cap, liquid, no known discontinuity. The clean case.
  - 6594 bars 2000-01-04 → 2026-08-13, source `yfinance`, hash `7f890f001da6`, 3 repaired field(s)

**SCB** — SCB X PCL (formerly The Siam Commercial Bank PCL)
  - liquidity: Large cap, liquid. Series carries an issuer substitution at 2022-04-22.
  - break 2022-04-22 [issuer_substitution]: SCB delisted and SCB X PCL listed 1:1 in its place, retaining the SCB ticker. A change of issuer (bank -> holding company), not merely of name. Pre-2022-04-22 'SCB' bars belong to a different legal entity.
  - source: SCB/SCBX first-party announcements, March-April 2022
  - 1047 bars 2022-04-20 → 2026-08-13, source `yfinance`, hash `009611b3e1c1`, 1 repaired field(s)

**BAY** — Bank of Ayudhya PCL (Krungsri)
  - liquidity: Thin float: ~72-76% held by MUFG since the 2013 acquisition. Daily turnover is small relative to SCB/KBANK. Treat as the liquidity stress case and cap participation in any backtest.
  - default participation cap: 5.0% of volume
  - 6594 bars 2000-01-04 → 2026-08-13, source `yfinance`, hash `4ccb5ee8ae79`, 6 repaired field(s)

## KBANK

### Forecasting models

| model | MASE | beats naive | dir acc | RMSE(ret) | Sharpe net | Sharpe gross |
|---|---|---|---|---|---|---|
| 00_naive_lag | 1.0000 | False | — | 0.01288 | +0.00 | +0.00 |
| 13_gru_seq2seq | 1.0046 | False | 43.1% | 0.01282 | +1.17 | +1.92 |
| 19_stack_rnn_arima_xgb | 1.0054 | False | 42.5% | 0.01286 | +1.27 | +1.28 |
| 21_arima | 1.0097 | False | 42.7% | 0.01288 | -0.39 | +1.48 |
| 12_lstm_seq2seq_vae | 1.0097 | False | 41.0% | 0.01285 | +0.95 | +1.18 |
| 15_gru_seq2seq_vae | 1.0116 | False | 41.5% | 0.01285 | +1.50 | +2.04 |
| 14_bidirectional_gru_seq2seq | 1.0121 | False | 43.8% | 0.01288 | +1.61 | +2.48 |
| 03_lstm_2path | 1.0158 | False | 41.0% | 0.01293 | +0.51 | +1.36 |
| 02_bidirectional_lstm | 1.0166 | False | 40.4% | 0.01294 | +0.03 | +1.03 |
| 04_gru | 1.0169 | False | 45.4% | 0.01289 | +2.10 | +2.89 |
| 18_dilated_cnn_seq2seq | 1.0196 | False | 42.5% | 0.01291 | -0.67 | +1.78 |
| 01_lstm | 1.0206 | False | 43.8% | 0.01294 | +2.10 | +2.56 |
| 06_gru_2path | 1.0208 | False | 41.2% | 0.01293 | +0.52 | +1.72 |
| 11_bidirectional_lstm_seq2seq | 1.0222 | False | 44.6% | 0.01295 | +1.78 | +2.22 |
| 08_bidirectional_vanilla | 1.0229 | False | 40.0% | 0.01306 | -1.02 | +0.93 |
| 05_bidirectional_gru | 1.0230 | False | 40.4% | 0.01292 | +0.26 | +1.44 |
| 10_lstm_seq2seq | 1.0233 | False | 45.4% | 0.01291 | +2.03 | +2.48 |
| 16_attention_is_all_you_need | 1.0283 | False | 41.9% | 0.01302 | +1.02 | +1.42 |
| 17_cnn_seq2seq | 1.0286 | False | 42.7% | 0.01297 | -0.57 | +1.71 |
| 09_vanilla_2path | 1.0311 | False | 40.2% | 0.01304 | -0.43 | +1.29 |
| 07_vanilla | 1.0318 | False | 40.6% | 0.01307 | -0.84 | +0.99 |
| 20_stack_encoder_ensemble_xgb | 1.0667 | False | 40.6% | 0.01326 | -1.09 | +1.30 |
| 22_xgboost | 1.1064 | False | 40.8% | 0.01366 | -1.27 | +0.89 |

**0 of 22 models beat the naive lag on KBANK.**

### Agents — frictionless vs. SET frictions

| agent | return (free) | return (frictions) | cost of frictions | Sharpe net | trades | max DD |
|---|---|---|---|---|---|---|
| 00_buy_and_hold | +8.09% | +7.82% | +0.27% | +1.41 | 8 | -11.33% |
| 04_policy_gradient | +6.63% | +5.75% | +0.88% | +1.38 | 37 | -16.42% |
| 13_double_duel_recurrent_q_learning | +6.27% | +5.44% | +0.83% | +1.52 | 36 | -4.97% |
| 09_double_recurrent_q_learning | +6.46% | +5.33% | +1.13% | +1.59 | 49 | -7.00% |
| 16_actor_critic_recurrent | +5.20% | +4.98% | +0.22% | +1.47 | 10 | -5.06% |
| 08_recurrent_q_learning | +5.73% | +4.44% | +1.29% | +1.40 | 57 | -5.81% |
| 19_recurrent_curiosity_q_learning | +5.15% | +4.35% | +0.80% | +1.22 | 34 | -6.07% |
| 21_neuro_evolution | +5.39% | +3.93% | +1.46% | +1.05 | 59 | -12.26% |
| 17_actor_critic_duel_recurrent | +4.03% | +3.91% | +0.12% | +1.10 | 4 | -10.78% |
| 07_double_q_learning | +6.28% | +3.75% | +2.52% | +1.17 | 113 | -5.23% |
| 12_duel_recurrent_q_learning | +4.35% | +3.37% | +0.98% | +0.89 | 42 | -11.20% |
| 03_signal_rolling | +5.07% | +3.14% | +1.94% | +0.78 | 90 | -12.23% |
| 20_duel_curiosity_q_learning | +3.68% | +3.11% | +0.57% | +0.85 | 26 | -14.81% |
| 22_neuro_evolution_novelty | +4.78% | +3.09% | +1.69% | +0.85 | 77 | -11.09% |
| 11_double_duel_q_learning | +4.90% | +2.54% | +2.35% | +0.60 | 108 | -10.65% |
| 18_curiosity_q_learning | +3.98% | +2.32% | +1.66% | +0.57 | 77 | -12.74% |
| 23_abcd | +2.37% | +2.17% | +0.20% | +0.80 | 8 | -8.16% |
| 14_actor_critic | +2.33% | +1.59% | +0.75% | +0.48 | 35 | -15.76% |
| 02_moving_average | +1.14% | +0.61% | +0.53% | +0.14 | 26 | -15.42% |
| 15_actor_critic_duel | +0.80% | +0.35% | +0.45% | +0.23 | 21 | -6.59% |
| 01_turtle | +0.16% | -0.17% | +0.32% | -0.12 | 17 | -18.85% |
| 10_duel_q_learning | +2.27% | -0.40% | +2.67% | -0.10 | 126 | -15.51% |
| 06_evolution_strategy | +1.23% | -0.43% | +1.66% | -0.10 | 79 | -11.27% |
| 05_q_learning | +2.01% | -0.53% | +2.54% | -0.17 | 122 | -11.62% |

Mean return per fold is shown. **24 of 24 agents make money frictionless; 20 still do after SET costs.** Frictions cost +1.16% per fold on average.

Buy-and-hold returned **+7.82%** per fold after costs on KBANK. **0 of 23 active agents beat it.**

4 agent(s) are profitable frictionless and lose money once SET costs are charged: 01_turtle, 10_duel_q_learning, 06_evolution_strategy, 05_q_learning. That sign change is the single clearest argument for the friction layer existing.

## SCB

### Forecasting models

| model | MASE | beats naive | dir acc | RMSE(ret) | Sharpe net | Sharpe gross |
|---|---|---|---|---|---|---|
| 00_naive_lag | 1.0000 | False | — | 0.01005 | +0.00 | +0.00 |
| 12_lstm_seq2seq_vae | 1.0059 | False | 45.8% | 0.01003 | +0.98 | +1.09 |
| 15_gru_seq2seq_vae | 1.0068 | False | 42.9% | 0.01006 | -0.52 | +0.28 |
| 11_bidirectional_lstm_seq2seq | 1.0085 | False | 43.3% | 0.01005 | +0.69 | +0.81 |
| 19_stack_rnn_arima_xgb | 1.0096 | False | 45.0% | 0.01006 | +0.94 | +0.96 |
| 21_arima | 1.0097 | False | 44.2% | 0.01007 | -1.01 | +0.82 |
| 08_bidirectional_vanilla | 1.0135 | False | 41.7% | 0.01011 | -1.96 | +0.25 |
| 02_bidirectional_lstm | 1.0138 | False | 44.2% | 0.01007 | +0.07 | +0.94 |
| 14_bidirectional_gru_seq2seq | 1.0142 | False | 42.5% | 0.01009 | -0.68 | +0.50 |
| 07_vanilla | 1.0148 | False | 42.9% | 0.01009 | -0.74 | +0.72 |
| 10_lstm_seq2seq | 1.0155 | False | 40.4% | 0.01008 | -0.10 | +0.17 |
| 16_attention_is_all_you_need | 1.0222 | False | 45.8% | 0.01013 | +0.78 | +1.21 |
| 13_gru_seq2seq | 1.0235 | False | 42.5% | 0.01014 | -0.55 | +0.35 |
| 05_bidirectional_gru | 1.0251 | False | 41.2% | 0.01015 | -0.49 | +0.39 |
| 03_lstm_2path | 1.0258 | False | 42.5% | 0.01017 | -0.47 | +0.45 |
| 04_gru | 1.0266 | False | 41.2% | 0.01017 | -0.51 | +0.47 |
| 09_vanilla_2path | 1.0268 | False | 42.5% | 0.01021 | -1.43 | +0.48 |
| 17_cnn_seq2seq | 1.0279 | False | 42.5% | 0.01030 | -2.57 | -0.05 |
| 06_gru_2path | 1.0291 | False | 40.4% | 0.01020 | -0.93 | +0.21 |
| 01_lstm | 1.0298 | False | 40.4% | 0.01019 | -0.67 | +0.10 |
| 18_dilated_cnn_seq2seq | 1.0339 | False | 41.7% | 0.01026 | -2.70 | -0.16 |
| 20_stack_encoder_ensemble_xgb | 1.0888 | False | 45.0% | 0.01060 | -1.30 | +1.32 |
| 22_xgboost | 1.1297 | False | 39.2% | 0.01114 | -2.73 | +0.10 |

**0 of 22 models beat the naive lag on SCB.**

### Agents — frictionless vs. SET frictions

| agent | return (free) | return (frictions) | cost of frictions | Sharpe net | trades | max DD |
|---|---|---|---|---|---|---|
| 00_buy_and_hold | +3.30% | +3.12% | +0.18% | +0.72 | 5 | -12.65% |
| 08_recurrent_q_learning | +5.70% | +4.89% | +0.81% | +1.37 | 18 | -5.84% |
| 23_abcd | +4.40% | +4.12% | +0.29% | +1.64 | 6 | -4.12% |
| 18_curiosity_q_learning | +4.89% | +3.22% | +1.67% | +0.96 | 38 | -5.66% |
| 16_actor_critic_recurrent | +3.06% | +2.95% | +0.11% | +1.63 | 2 | -3.39% |
| 14_actor_critic | +2.53% | +2.33% | +0.20% | +0.87 | 4 | -7.24% |
| 21_neuro_evolution | +4.65% | +2.00% | +2.66% | +0.53 | 43 | -12.75% |
| 13_double_duel_recurrent_q_learning | +2.90% | +1.87% | +1.03% | +0.49 | 24 | -12.11% |
| 02_moving_average | +2.34% | +1.78% | +0.56% | +0.56 | 13 | -7.96% |
| 06_evolution_strategy | +2.48% | +1.39% | +1.09% | +0.44 | 25 | -11.20% |
| 22_neuro_evolution_novelty | +2.96% | +1.28% | +1.68% | +0.38 | 39 | -9.10% |
| 09_double_recurrent_q_learning | +2.11% | +1.13% | +0.98% | +0.32 | 23 | -11.17% |
| 12_duel_recurrent_q_learning | +1.50% | +0.49% | +1.02% | +0.14 | 23 | -11.39% |
| 01_turtle | +0.41% | +0.04% | +0.37% | +0.02 | 9 | -7.96% |
| 03_signal_rolling | +1.94% | -0.12% | +2.07% | -0.04 | 49 | -6.17% |
| 19_recurrent_curiosity_q_learning | -0.39% | -0.48% | +0.09% | -0.14 | 22 | -11.20% |
| 11_double_duel_q_learning | +2.40% | -0.56% | +2.96% | -0.15 | 70 | -10.44% |
| 15_actor_critic_duel | -0.32% | -0.68% | +0.35% | -0.40 | 9 | -7.50% |
| 07_double_q_learning | +2.27% | -0.89% | +3.15% | -0.30 | 66 | -9.61% |
| 17_actor_critic_duel_recurrent | -1.48% | -1.51% | +0.03% | -0.64 | 1 | -11.21% |
| 04_policy_gradient | -2.47% | -2.73% | +0.26% | -1.01 | 7 | -13.61% |
| 05_q_learning | -0.71% | -3.22% | +2.51% | -1.19 | 62 | -7.43% |
| 10_duel_q_learning | -0.69% | -3.28% | +2.59% | -1.03 | 65 | -12.37% |
| 20_duel_curiosity_q_learning | -2.76% | -3.93% | +1.17% | -1.52 | 29 | -11.72% |

Mean return per fold is shown. **17 of 24 agents make money frictionless; 14 still do after SET costs.** Frictions cost +1.16% per fold on average.

Buy-and-hold returned **+3.12%** per fold after costs on SCB. **3 of 23 active agents beat it.** Namely: 08_recurrent_q_learning, 23_abcd, 18_curiosity_q_learning.

3 agent(s) are profitable frictionless and lose money once SET costs are charged: 03_signal_rolling, 11_double_duel_q_learning, 07_double_q_learning. That sign change is the single clearest argument for the friction layer existing.

## BAY

### Forecasting models

| model | MASE | beats naive | dir acc | RMSE(ret) | Sharpe net | Sharpe gross |
|---|---|---|---|---|---|---|
| 00_naive_lag | 1.0000 | False | — | 0.01516 | +0.00 | +0.00 |
| 19_stack_rnn_arima_xgb | 1.0091 | False | 37.1% | 0.01518 | +0.00 | +0.00 |
| 21_arima | 1.0205 | False | 40.0% | 0.01534 | -1.89 | +0.50 |
| 07_vanilla | 1.0280 | False | 39.8% | 0.01523 | -0.28 | +1.49 |
| 09_vanilla_2path | 1.0366 | False | 40.6% | 0.01525 | +0.17 | +1.92 |
| 08_bidirectional_vanilla | 1.0373 | False | 41.9% | 0.01520 | +0.58 | +2.32 |
| 14_bidirectional_gru_seq2seq | 1.0392 | False | 40.0% | 0.01541 | -0.19 | +1.49 |
| 13_gru_seq2seq | 1.0417 | False | 39.2% | 0.01536 | +0.20 | +1.40 |
| 02_bidirectional_lstm | 1.0440 | False | 41.7% | 0.01532 | +0.86 | +2.32 |
| 15_gru_seq2seq_vae | 1.0474 | False | 36.9% | 0.01538 | +0.42 | +1.22 |
| 04_gru | 1.0583 | False | 38.3% | 0.01553 | +0.21 | +1.39 |
| 11_bidirectional_lstm_seq2seq | 1.0586 | False | 37.3% | 0.01562 | +0.08 | +1.17 |
| 05_bidirectional_gru | 1.0589 | False | 40.2% | 0.01568 | -0.04 | +1.67 |
| 12_lstm_seq2seq_vae | 1.0590 | False | 35.6% | 0.01547 | -0.31 | +0.48 |
| 10_lstm_seq2seq | 1.0615 | False | 35.4% | 0.01551 | -0.42 | +0.43 |
| 18_dilated_cnn_seq2seq | 1.0630 | False | 40.0% | 0.01556 | -0.69 | +1.36 |
| 17_cnn_seq2seq | 1.0646 | False | 40.0% | 0.01551 | +0.11 | +1.79 |
| 03_lstm_2path | 1.0663 | False | 39.2% | 0.01574 | +0.23 | +1.48 |
| 01_lstm | 1.0738 | False | 38.5% | 0.01594 | +0.00 | +1.05 |
| 06_gru_2path | 1.0799 | False | 40.6% | 0.01606 | +0.04 | +1.74 |
| 20_stack_encoder_ensemble_xgb | 1.1001 | False | 34.4% | 0.01572 | -1.51 | +0.66 |
| 22_xgboost | 1.1154 | False | 40.6% | 0.01603 | -0.21 | +1.85 |
| 16_attention_is_all_you_need | 1.1301 | False | 35.2% | 0.01615 | +0.04 | +0.52 |

**0 of 22 models beat the naive lag on BAY.**

### Agents — frictionless vs. SET frictions

| agent | return (free) | return (frictions) | cost of frictions | Sharpe net | trades | max DD |
|---|---|---|---|---|---|---|
| 00_buy_and_hold | +11.56% | +10.56% | +1.00% | +1.42 | 36 | -12.52% |
| 11_double_duel_q_learning | +9.79% | +9.27% | +0.52% | +1.67 | 191 | -8.85% |
| 22_neuro_evolution_novelty | +7.88% | +7.85% | +0.03% | +1.31 | 114 | -9.03% |
| 18_curiosity_q_learning | +10.18% | +7.48% | +2.70% | +1.29 | 126 | -10.32% |
| 01_turtle | +7.56% | +7.17% | +0.39% | +1.29 | 32 | -11.33% |
| 02_moving_average | +8.26% | +6.83% | +1.43% | +1.62 | 21 | -7.16% |
| 03_signal_rolling | +6.18% | +6.32% | -0.14% | +1.20 | 206 | -6.74% |
| 19_recurrent_curiosity_q_learning | +7.19% | +5.76% | +1.44% | +1.29 | 70 | -7.07% |
| 20_duel_curiosity_q_learning | +10.08% | +5.44% | +4.64% | +1.09 | 101 | -11.32% |
| 12_duel_recurrent_q_learning | +3.63% | +3.31% | +0.32% | +0.82 | 131 | -10.02% |
| 09_double_recurrent_q_learning | +3.15% | +2.90% | +0.26% | +0.84 | 120 | -7.65% |
| 06_evolution_strategy | +4.60% | +2.52% | +2.08% | +0.80 | 109 | -6.63% |
| 08_recurrent_q_learning | +4.83% | +2.07% | +2.76% | +0.76 | 109 | -4.91% |
| 13_double_duel_recurrent_q_learning | +1.67% | +1.98% | -0.31% | +0.59 | 100 | -8.10% |
| 10_duel_q_learning | +3.22% | +1.95% | +1.28% | +0.60 | 194 | -9.68% |
| 07_double_q_learning | +2.04% | +1.67% | +0.37% | +0.47 | 121 | -10.07% |
| 16_actor_critic_recurrent | +1.27% | +0.98% | +0.29% | +0.61 | 4 | -3.77% |
| 05_q_learning | +4.28% | +0.76% | +3.52% | +0.24 | 176 | -6.94% |
| 21_neuro_evolution | +0.57% | +0.64% | -0.08% | +0.40 | 53 | -4.16% |
| 15_actor_critic_duel | +0.00% | +0.18% | -0.18% | +0.43 | 6 | -1.14% |
| 23_abcd | +0.71% | -0.08% | +0.79% | -0.02 | 22 | -7.66% |
| 17_actor_critic_duel_recurrent | +1.36% | -0.08% | +1.44% | -0.04 | 6 | -10.33% |
| 04_policy_gradient | -0.17% | -0.25% | +0.08% | -0.12 | 37 | -5.84% |
| 14_actor_critic | -1.39% | -0.84% | -0.55% | -0.90 | 19 | -4.45% |

Mean return per fold is shown. **21 of 24 agents make money frictionless; 20 still do after SET costs.** Frictions cost +1.00% per fold on average.

> **BAY carries a 5% participation cap**, so the frictionless column also drops the liquidity constraint — it lets an agent take a position the market could not actually have absorbed. Part of the mean gap here (+1.00%) is therefore a statement about BAY's float rather than about commission. With ~72-76% of shares held by MUFG, that is the intended reading: the frictionless number is not a return anyone could have earned.

> The cap also changes what a *fair baseline* means. A buy-and-hold that issues one order, has it trimmed to a fraction of a session's volume and then stops would sit mostly in cash — and would lose to any agent that trades repeatedly, purely because the reference line was handicapped. `buy_and_hold` here accumulates across sessions until its capital is deployed, which is what a real holder does under a liquidity constraint.

Buy-and-hold returned **+10.56%** per fold after costs on BAY. **0 of 23 active agents beat it.**

2 agent(s) are profitable frictionless and lose money once SET costs are charged: 23_abcd, 17_actor_critic_duel_recurrent. That sign change is the single clearest argument for the friction layer existing.

## Headline

**0 of 66 model runs beat the naive lag out-of-sample across 3 tickers (KBANK, SCB, BAY).**

That number is zero, and it is reported as zero. It is the result the spec anticipated as legitimate and likely, and it is what a non-leaking, cost-charging harness produces from this catalogue on this universe.

The upstream repository reports accuracies in the high nineties for the same architectures. Both things are true at once, and the reason is methodological, not architectural: upstream fits its scaler on the full series before splitting, scores price levels rather than returns, and shows no baseline. On price levels a naive lag also scores in the high nineties — `metrics.upstream_accuracy_do_not_use` and its test demonstrate this. Those numbers never measured skill.

**62 of 72 agent runs are profitable frictionless; 54 survive SET frictions.** The difference between those two numbers is the cost of assuming a market without board lots, tick sizes, commission or VAT — the assumption every upstream agent notebook makes.

## What would change these numbers

- **A better data source.** yfinance is a redistributor; Settrade is the exchange's own feed and is unreachable here without broker credentials. See `docs/settrade-api-notes.md`.
- **Verified cost parameters.** The commission rate drives the friction gap directly and is currently a plausible guess (spec R13).
- **Features.** Every model reads the same five causal feature blocks. The catalogue was designed around a single close-price series; giving it a richer, well-motivated feature set is a more promising direction than any architecture on these tables.
- **More folds.** Configurable in `configs/eval.yaml`; `max_folds: null` uses every available fold rather than the most recent 8.

## Provenance

Every run in `results/` has a `*.manifest.json` beside it recording the config, the git SHA, and the content hash of the bars consumed.
