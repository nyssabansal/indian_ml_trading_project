# What was added in this version (plain-language summary)

| # | What was missing | What we added | Where |
|---|---|---|---|
| 1 | No supply-chain idea from the paper | Supplier/peer stocks (default: Maruti + 9 suppliers + NIFTY 50). Their past daily returns become features. | `make_features`, `align_peers`, app sidebar |
| 2 | No "is this better than guessing?" check | **Predict-0** and **historical-mean** baselines in every table, plus a "skill vs predict-0" score | `BASELINES`, `metric_row` |
| 3 | Settings were fixed by hand | **Hyperparameter tuning** on the validation set for every model | `MODEL_SPACE`, `train_and_evaluate` |
| 4 | Only 3 models, no boosting | Added **Gradient Boosting** (similar to the paper's LightGBM) | `MODEL_SPACE` |
| 5 | Only a 5-day horizon | Choose **1, 5 or 20 days**, plus a horizon comparison table like the paper's Table 1 | `compare_horizons` |
| 6 | Only one test period | **Walk-forward evaluation** over 5 periods, like the paper's rolling window | `walk_forward` |
| 7 | No answer to "do suppliers help?" | **Ablation**: every model trained with vs without peer features, plus importance grouped by source | `own_vs_peer_ablation`, `group_importance` |
| 8 | Only return and drawdown | Annual return, volatility, **Sharpe ratio**, number of trades, time in market | `risk_stats`, `backtest` |
| 9 | No threshold comparison | **Threshold table** (0 / 0.25 / 0.5 / 1%), like the paper's Table 2 | `threshold_table` |
| 10 | Sliders did nothing until you re-trained | The strategy now updates **instantly** when the sliders move. There's a warning if model settings changed. | `app.py` |
| 11 | "Synthetic data" banner showed on real data too | The banner only shows for the demo data | `app.py` |
| 12 | Demo could fail if Yahoo was down | **Download buttons** to save the data, then reuse it with Upload CSV | `app.py` |
| 13 | No way to reproduce the results for the write-up | `run_experiment.py` saves every table, chart and data file | `run_experiment.py` |
| 14 | No tests | 10 automated tests (no future leakage, correct target, embargo, lagged signal, …) | `tests/` |
| 15 | Embargo was 10 rows, README said 5 | The embargo is now exactly h rows, and the docs match | `chronological_split` |
| 16 | The best model was refit 3 times for no reason | Each model is fitted once | `train_and_evaluate` |
| 17 | No team contributions section | Added to the README | `README.md` |

Note: `reports/two_page_writeup_template.pdf` and `reports/presentation_template.pptx` are from the earlier version. Use the updated `.md` files as the text source, and fill in real numbers from `reports/results/<ticker>/summary.md`.

# Version 3: stronger evaluation (plain-language summary)

| # | What was missing | What we added | Where |
|---|---|---|---|
| 18 | No way to tell whether a model's small edge is real or luck | **Diebold–Mariano significance test** against predict-0 for every model (p-value column in every results table) | `diebold_mariano`, `train_and_evaluate` |
| 19 | Models and settings were chosen on one validation period, so a lucky period could pick the wrong model | **Time-series cross-validation**: each setting is scored on 4 expanding folds inside train+validation, with a 5-row embargo per fold | `time_series_folds`, `train_and_evaluate(cv_folds=4)` |
| 20 | Market-wide moves make the target noisy | Optional **excess-return target**: the stock's return minus NIFTY 50's over the same days | `make_features(target_mode="excess")`, `--target excess`, app sidebar |
| 21 | App still called "SentinelQuant" | Renamed to match the project | `app.py` |
| 22 | — | 4 new tests (significance test, CV folds, excess target, selection report): 15 in total | `tests/` |
| 23 | Plain default dashboard on one long page | **Redesigned dashboard**: themed header, headline findings strip, six tabs, interactive Plotly charts (error vs predict-0, with/without suppliers, equity curve, walk-forward, actual vs predicted), split periods shaded on the price chart, horizon table with cells that beat predict-0 highlighted | `app.py`, `.streamlit/config.toml` |
| 24 | Dashboard looked generic | **Topic-specific design**: headline names the company ("Can supplier stocks predict Maruti Suzuki?"), a market strip with each supplier's latest price, 1-month change and role, a supply-chain map (line thickness = how closely each supplier moves with Maruti), a lead-lag heatmap testing whether supplier moves come *before* Maruti's, and company names instead of tickers | `app.py` |
| 25 | Light theme only | **Dark theme** for the whole dashboard and every chart | `.streamlit/config.toml`, `app.py` |
