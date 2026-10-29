# Forecasting Indian Stock Returns from Supplier Stocks with Machine Learning
**Course:** UE24CS352A — Machine Learning Mini-Project  
**Team members:** Nyssa, Yathindra

## 1. Problem statement
Can the past returns of Maruti Suzuki's suppliers, together with Maruti's own price and volume history, predict its return over the next five trading days better than simply guessing "no change"? And does a trading rule built on those forecasts beat buy-and-hold once transaction costs are included?

We adapt René M. Glawion's Stanford CS229 report *Quantitative Trading with Machine Learning* (2021). That report predicted Volkswagen's returns from 30 suppliers' lagged returns, found Elastic Net best across horizons, and reported a strategy return of 21.49% vs 7.98% for buy-and-hold. We apply the idea to an Indian automaker and address four limitations of the report: no baseline, missing prices filled with zero, no transaction costs, and no risk measures.

## 2. Dataset
- **Source:** daily split- and dividend-adjusted prices from Yahoo Finance via `yfinance`, covering the 10 years to 5 Oct 2026.
- **Target:** `MARUTI.NS`.
- **Peers:** auto components (`MOTHERSON.NS`, `BOSCHLTD.NS`, `UNOMINDA.NS`, `ENDURANCE.NS`), tyres/batteries (`MRF.NS`, `APOLLOTYRE.NS`, `EXIDEIND.NS`), steel (`TATASTEEL.NS`, `JSWSTEEL.NS`) and the NIFTY 50 (`^NSEI`).
- **Size:** 2,471 trading days, giving **2,415 usable rows (23 Dec 2016 – 5 Oct 2026)**. The first 50 days are dropped because the moving averages need history, and the last 5 because there is no future price to compute the target.
- **Missing data:** all 10 peers covered at least 90% of trading days, so none were dropped. Holiday gaps were forward-filled for at most 5 days using past prices only. Nothing was filled with zero.

## 3. Approach
**Features at day t** use only data up to day t, 61 in total:
- 11 own features: 1/5/20-day returns; price relative to its 5/20/50-day moving averages; 10/20-day volatility; volume change; intraday range; and close location within the day's range.
- For each of the 10 peers, its daily return on each of the last 5 days.

**Target:** `Close[t+5]/Close[t] − 1`. We also tested horizons of 1 and 20 days.

**Models:** Elastic Net, Decision Tree, Random Forest and Gradient Boosting, plus two baselines: predict 0 and historical mean.

**Evaluation:**
- **Split:** chronological. Train Dec 2016–Oct 2022 (1,444 rows), validation Nov 2022–Oct 2024 (478 rows), test Oct 2024–Oct 2026 (483 rows). The 5 rows before each boundary are dropped so that no training target overlaps a later period.
- **Tuning and selection:** each model is tuned on validation RMSE, and the best model is chosen on validation. All models are then refit on train+validation and scored once on test.
- **Metrics:** RMSE, MAE, direction accuracy, and skill vs predict-0 (1 − SSE_model / SSE_zero).
- **Robustness:**
  - An ablation retrains each model without peer features.
  - A 5-period walk-forward evaluation retrains on all earlier data for each period.
  - A horizon comparison repeats the test at 1, 5 and 20 days.

**Backtest:** long/flat. Hold Maruti on day t if the forecast made at close t−1 exceeds 0.25%; otherwise hold cash. Each buy or sell costs 0.05%.

## 4. Implementation
- `trading_ml.py`: data cleaning, peer alignment, features, splitting, tuning, metrics, ablation, walk-forward and backtest.
- `app.py`: a Streamlit dashboard for the live demo, using Yahoo Finance, CSV upload or simulated data.
- `run_experiment.py`: reproduces every table and chart and saves them to `reports/results/`.
- `tests/test_pipeline.py`: 11 tests covering look-ahead leakage, the target definition, the embargo, the backtest timing and the ablation.

## 5. Results
**Test set, 5-day horizon** (lower RMSE is better):

| Model | RMSE own only (%) | RMSE own + peers (%) | MAE (%) | Direction acc. (%) | Skill vs predict-0 (%) |
|---|---|---|---|---|---|
| Baseline: predict 0 | 3.375 | 3.375 | 2.493 | — | 0.0 |
| Random Forest | 3.432 | **3.356** | 2.511 | 53.6 | **+1.1** |
| Gradient Boosting | 3.420 | 3.385 | 2.517 | 52.0 | −0.6 |
| Elastic Net | 3.387 | 3.387 | 2.518 | 48.7 | −0.7 |
| Baseline: historical mean | 3.387 | 3.387 | 2.518 | 48.7 | −0.7 |
| Decision Tree (selected) | 3.470 | 3.470 | 2.578 | 51.6 | −5.7 |

**Forecast accuracy:**
- Only Random Forest with peers beat predict-0 on test, and only narrowly.
- Validation RMSEs were nearly tied (2.702–2.762%). The model picked on validation (Decision Tree) was the worst on test.
- In the walk-forward evaluation, the historical-mean baseline had the lowest average RMSE (3.335%). Random Forest and Gradient Boosting beat predict-0 in 2 of 5 periods, Elastic Net in 1, and Decision Tree in 0.

**Do peers help?**
- Peer features lowered RMSE for Random Forest (−0.076 pts) and Gradient Boosting (−0.036 pts). Elastic Net and Decision Tree effectively ignored them.
- The Decision Tree's only peer split was on Motherson's previous-day return, inside a high-volatility branch that the test period never reached.

**Horizons:**
- Error grows with the horizon: predict-0 RMSE is 1.44% at 1 day, 3.38% at 5 days and 7.54% at 20 days.
- At 20 days, Gradient Boosting (7.31%) and Random Forest (7.36%) with peers beat predict-0. Without peers, only Gradient Boosting did (7.51%), and only marginally.

**Strategy** (Decision Tree, test period):
- **+6.7% vs −3.3%** for buy-and-hold.
- Max drawdown −23.1% vs −33.5%; Sharpe ratio 0.28 vs 0.04; 32 trades; 71% of days invested.
- The gap comes mainly from holding cash during the late-2026 decline.
- With own features only, the strategy's return ranged from −3.2% to +24.7% depending on the threshold.

## 6. Conclusion
Supplier returns carry a small amount of information about Maruti's future returns, and only the non-linear ensembles could use it. That information is not enough to reliably beat simply guessing "no change": only one model did so on the test set, none did on average in walk-forward testing, and the gains were a fraction of a percentage point. This agrees with the reference paper's own observation that predicting zero gives almost the same RMSE.

The strategy beat buy-and-hold over the test period, but the forecasts behind it show no reliable edge, and the result swings widely with the threshold. We therefore treat it as an illustration of how backtests can mislead, not as evidence of a profitable signal. The same caution applies to the paper's headline result, which was the best of 96 configurations and ignored costs.

**Limitations:**
- A hand-picked supplier list, not verified supply contracts
- Tuning used one validation period, with small grids
- No slippage or taxes modelled
- A single target stock
- Correlation, not causation

**Future work:**
- Verified supply-chain data
- Macro variables (bond yields, INR/USD)
- Re-tuning within each walk-forward period
- Repeating the study for other Indian automakers
