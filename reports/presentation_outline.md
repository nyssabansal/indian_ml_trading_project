# Presentation outline (7 slides + live demo)

## Slide 1 — Title and motivation
Forecasting Indian Stock Returns from Supplier Stocks with ML. Team: Nyssa, Yathindra. Course: UE24CS352A.
Motivation: carmakers depend on their suppliers. Do suppliers' stock moves help predict the carmaker's stock? Based on a Stanford CS229 report about Volkswagen.

## Slide 2 — The reference paper in one slide
- VW plus 30 suppliers × 28 days of lagged returns, plus macro data
- Models: Elastic Net, Decision Tree, XGBoost, LightGBM; Elastic Net was best
- Strategy: +21.49% vs +7.98% buy-and-hold
- Gaps we address:
  - No baseline (its own footnote says predict-0 is almost as good)
  - Missing values filled with 0
  - No transaction costs and no risk measures
  - The best result was picked from 96 strategy variants

## Slide 3 — Problem and data
- Target: Maruti Suzuki (MARUTI.NS), forward 5-day return
- Peers: Motherson, Bosch, Uno Minda, Endurance, MRF, Apollo Tyres, Exide, Tata Steel, JSW Steel, plus the NIFTY 50
- Yahoo Finance, 10 years: **2,415 rows, Dec 2016 → Oct 2026**
- All peers had at least 90% coverage. Short holiday gaps forward-filled. No zero-filling.

## Slide 4 — Features and models
- 61 features:
  - 11 own: momentum, moving-average gaps, volatility, volume, intraday range
  - 50 peer: 10 peers × last 5 daily returns
- Models: Elastic Net, Decision Tree, Random Forest, Gradient Boosting
- Baselines: predict 0, historical mean

## Slide 5 — Experimental design
- Time-ordered split:
  - Train Dec 2016–Oct 2022 (1,444 rows)
  - Validation Nov 2022–Oct 2024 (478)
  - Test Oct 2024–Oct 2026 (483)
  - 5-row embargo before each boundary
- Tune on validation, test once
- Ablation (with vs without peers), 5-period walk-forward, 1/5/20-day horizons

## Slide 6 — Results (+ live demo)
| Model | RMSE own only | RMSE + peers |
|---|---|---|
| Predict 0 (baseline) | 3.375% | 3.375% |
| Random Forest | 3.432% | **3.356%** ✓ |
| Gradient Boosting | 3.420% | 3.385% |
| Elastic Net | 3.387% | 3.387% |
| Decision Tree (picked on validation) | 3.470% | 3.470% |

- Only Random Forest with peers beats guessing 0, by about 1%.
- Peers help Random Forest and Gradient Boosting. Elastic Net and Decision Tree ignore them.
- At 20 days, both ensembles with peers beat predict-0 (7.31% / 7.36% vs 7.54%).
- Walk-forward: no model beats the historical-mean baseline on average.
- Strategy: +6.7% vs −3.3% buy-and-hold; drawdown −23% vs −33%; 32 trades. Show `equity_curve.png`.

## Slide 7 — Conclusion, limitations, contributions
- Suppliers carry a small signal, usable only by the ensemble models. It's not enough to reliably beat "no change", which matches the paper's own footnote.
- The strategy's win mostly comes from sitting in cash during a fall. It's fragile: −3% to +25% depending on the threshold. Backtests can mislead, and the paper's 21% has the same problem.
- Limitations: hand-picked suppliers, one validation period, no slippage or taxes, one stock.
- Future work: verified supply-chain data, macro variables, other automakers.
- Contributions:
  - Nyssa: data, features, backtest, dashboard, write-up
  - Yathindra: models and tuning, evaluation, tests, experiments

## Live demo script (~2 min)
1. App already running. Select Yahoo Finance and MARUTI.NS (or Upload CSV with `reports/results/MARUTI.NS/data_target.csv` + `data_peers.csv`).
2. Data overview: 2,415 rows, all 10 peers used.
3. Click Train. Point out the green box: the selected model does not beat predict-0.
4. Show the "Do supplier/peer stocks help?" table. Random Forest improves with peers.
5. Strategy: move the threshold from 0.25% to 1% and watch the result change. Then set the cost to 0.2%.
6. Click walk-forward and show that results vary a lot between periods.

## Likely questions about the results
- **"Your model doesn't beat guessing zero. Is the project a failure?"** No. That's the honest finding, and the paper's footnote says the same. Our contribution is measuring it properly, with baselines, costs and walk-forward testing.
- **"Why did Decision Tree get selected if it's worst on test?"** Validation scores were nearly tied (2.702 vs 2.706). With noisy data, the validation winner is partly luck. That's why we also ran walk-forward.
- **"Why do peers help Random Forest but not Elastic Net?"** Any supplier effect is weak and probably non-linear, appearing only in certain conditions. Elastic Net's penalty shrinks weak linear effects to zero, while trees can pick up conditional patterns.
- **"Why are the strategy returns the same at 0%, 0.25% and 0.5% thresholds?"** The depth-3 tree only produces a few distinct prediction values on the test period, so those thresholds give the same buy/sell decisions.
- **"So did the strategy make money?"** +6.7% over two years, while the stock fell 3.3%. But it mostly came from being out of the market during the drop, and the forecasts themselves don't beat zero, so we don't claim an edge.
