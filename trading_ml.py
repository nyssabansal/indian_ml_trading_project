"""Leakage-aware pipeline for forecasting Indian stock returns (educational).

Pipeline:
    load OHLCV (+ optional supplier/peer prices)
    -> make_features()          features at day t use data up to day t only
    -> chronological_split()    train / validation / test, with an h-row embargo
    -> train_and_evaluate()     tune + select with time-series cross-validation, test once
    -> diebold_mariano()        is a model's error significantly different from predict-0?
    -> backtest()               long/flat strategy with transaction costs
    -> walk_forward()           robustness check over several time periods
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DEFAULT_HORIZON = 5
TRADING_DAYS = 252

# Default target: an automaker. Default peers: its listed suppliers + the market index.
DEFAULT_TARGET = "MARUTI.NS"
DEFAULT_PEERS = [
    "MOTHERSON.NS", "BOSCHLTD.NS", "UNOMINDA.NS", "ENDURANCE.NS", "MRF.NS",
    "APOLLOTYRE.NS", "EXIDEIND.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "^NSEI",
]

# Features computed from the target stock's own price/volume history.
OWN_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "ma_ratio_5", "ma_ratio_20", "ma_ratio_50",
    "volatility_10", "volatility_20", "volume_change",
    "range_pct", "close_location",
]


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------
def _simple_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change without forward-filling gaps (works on pandas 2 and 3)."""
    return series / series.shift(periods) - 1


def normalize_ohlcv(frame: pd.DataFrame, min_rows: int = 180) -> pd.DataFrame:
    """Normalize common CSV/provider columns to Date/Open/High/Low/Close/Volume."""
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "date" not in lower and "datetime" in lower:
        df = df.rename(columns={lower["datetime"]: "Date"})
        lower = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    df = df.rename(columns={lower[k]: k.title() for k in required})
    df["Date"] = _to_naive_dates(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} valid daily rows; found {len(df)}.")
    if (df["Close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def normalize_peers(frame: pd.DataFrame) -> pd.DataFrame:
    """Peer CSV: a Date column plus one closing-price column per peer ticker."""
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if date_col is None:
        raise ValueError("Peer CSV needs a Date column.")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = _to_naive_dates(df["Date"])
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")
    df = df.loc[:, df.notna().any()]
    df = df.where(df > 0)
    if df.shape[1] == 0:
        raise ValueError("Peer CSV has no usable price columns.")
    return df


def _to_naive_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)
    return dates.dt.normalize()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def align_peers(dates: pd.Series, peers: pd.DataFrame, min_coverage: float = 0.9,
                ffill_limit: int = 5):
    """Align peer prices to the target's trading dates.

    Peers can trade on different holidays (e.g. INR=X). We forward-fill short
    gaps (at most `ffill_limit` days, using only past prices) and drop peers that
    cover less than `min_coverage` of the target's dates, e.g. late listings.
    """
    union = peers.index.union(pd.DatetimeIndex(dates))
    aligned = peers.reindex(union).ffill(limit=ffill_limit).reindex(pd.DatetimeIndex(dates))
    coverage = aligned.notna().mean()
    kept = coverage[coverage >= min_coverage].index.tolist()
    dropped = {c: round(float(coverage[c]) * 100, 1) for c in aligned.columns if c not in kept}
    return aligned[kept], dropped


def make_features(frame: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                  peers: pd.DataFrame | None = None, peer_lags: int = 5,
                  min_peer_coverage: float = 0.9, target_mode: str = "raw",
                  market: str = "^NSEI"):
    """Build the modelling table.

    Returns (data, feature_columns, info). A feature row at date t only uses
    information available at the close of day t.

    target_mode="raw":    forward h-day return, Close[t+h] / Close[t] - 1
    target_mode="excess": that return minus the market's (default NIFTY 50)
                          return over the same h days, i.e. how much the stock
                          beats or lags the market. This strips out market-wide
                          moves, which are noise for a supplier-based signal.
    """
    df = normalize_ohlcv(frame)
    close = df["Close"]
    out = df.copy()
    returns = _simple_returns(close)
    out["ret_1d"] = returns
    out["ret_5d"] = _simple_returns(close, 5)
    out["ret_20d"] = _simple_returns(close, 20)
    for window in [5, 20, 50]:
        out[f"ma_ratio_{window}"] = close / close.rolling(window).mean() - 1
    out["volatility_10"] = returns.rolling(10).std()
    out["volatility_20"] = returns.rolling(20).std()
    out["volume_change"] = _simple_returns(df["Volume"])
    out["range_pct"] = (df["High"] - df["Low"]) / close
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    out["close_location"] = ((close - df["Low"]) / denom).fillna(0.5)
    features = list(OWN_FEATURES)

    info = {"peers_used": [], "peers_dropped": {}, "peer_lags": 0}
    if peers is not None and peers.shape[1] > 0:
        aligned, dropped = align_peers(df["Date"], peers, min_peer_coverage)
        info["peers_dropped"] = dropped
        info["peers_used"] = aligned.columns.tolist()
        info["peer_lags"] = peer_lags
        peer_cols = {}
        for ticker in aligned.columns:
            peer_ret = _simple_returns(aligned[ticker]).to_numpy()
            peer_ret = pd.Series(peer_ret, index=out.index)
            for lag in range(peer_lags):
                peer_cols[f"{ticker}_lag{lag}"] = peer_ret.shift(lag)
        if peer_cols:
            out = pd.concat([out, pd.DataFrame(peer_cols, index=out.index)], axis=1)
            features += list(peer_cols)

    out["target"] = close.shift(-horizon) / close - 1
    if target_mode == "excess":
        if peers is None or market not in peers.columns:
            raise ValueError(f"The excess-return target needs {market} in the peer data.")
        mkt, _ = align_peers(df["Date"], peers[[market]], min_coverage=0.0)
        mkt = pd.Series(mkt[market].to_numpy(), index=out.index)
        out["target"] = out["target"] - (mkt.shift(-horizon) / mkt - 1)
    elif target_mode != "raw":
        raise ValueError("target_mode must be 'raw' or 'excess'.")
    info["target_mode"] = target_mode
    out = out.replace([np.inf, -np.inf], np.nan)
    rows_before = len(out)
    out = out.dropna(subset=features + ["target"]).reset_index(drop=True)
    info["rows_raw"] = rows_before
    info["rows_used"] = len(out)
    info["horizon"] = horizon
    return out, features, info


def feature_group(name: str) -> str:
    """'MOTHERSON.NS_lag3' -> 'MOTHERSON.NS'; own features -> 'Own price/volume'."""
    if name in OWN_FEATURES:
        return "Own price/volume"
    return name.rsplit("_lag", 1)[0]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def chronological_split(data: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                        train_frac: float = 0.60, valid_frac: float = 0.20):
    """Time-ordered split with an h-row embargo before each boundary.

    The target of row i looks h days ahead, so the last h rows before a
    boundary would "see" prices inside the next block. Those rows are dropped.
    """
    n = len(data)
    a, b = int(n * train_frac), int(n * (train_frac + valid_frac))
    train = data.iloc[:max(0, a - horizon)].copy()
    valid = data.iloc[a:max(a, b - horizon)].copy()
    test = data.iloc[b:].copy()
    if min(len(train), len(valid), len(test)) < 20:
        raise ValueError("Not enough rows after the time split. Use a longer history.")
    return train, valid, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*space.values())]


# name -> (builder, hyperparameter grid). The first grid entry is the default
# used when tuning is switched off and in walk-forward evaluation.
MODEL_SPACE = {
    "Elastic Net": (
        lambda p: make_pipeline(StandardScaler(),
                                ElasticNet(max_iter=20000, random_state=42, **p)),
        # l1_ratio mirrors the paper's alpha grid {0, .5, 1}; 0.1 is used instead
        # of 0 because scikit-learn's solver is unreliable at a pure-ridge mix.
        {"alpha": [1e-3, 1e-4, 1e-2], "l1_ratio": [0.5, 0.1, 1.0]},
    ),
    "Decision Tree": (
        lambda p: DecisionTreeRegressor(random_state=42, **p),
        {"max_depth": [5, 3, 8], "min_samples_leaf": [10, 30]},
    ),
    "Random Forest": (
        lambda p: RandomForestRegressor(n_estimators=150, max_features=0.5,
                                        random_state=42, n_jobs=-1, **p),
        {"max_depth": [8, 4], "min_samples_leaf": [5, 20]},
    ),
    "Gradient Boosting": (
        lambda p: HistGradientBoostingRegressor(max_iter=200, early_stopping=False,
                                                random_state=42, **p),
        {"learning_rate": [0.05, 0.1], "max_depth": [3, 2]},
    ),
}

BASELINES = {
    "Baseline: predict 0": lambda: DummyRegressor(strategy="constant", constant=0.0),
    "Baseline: historical mean": lambda: DummyRegressor(strategy="mean"),
}


def build_model(name: str, params: dict | None = None):
    if name in BASELINES:
        return BASELINES[name]()
    builder, space = MODEL_SPACE[name]
    return builder(params if params is not None else _grid(space)[0])


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return model.fit(X, y)


def _rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def metric_row(name: str, actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sse_zero = float(np.sum(actual ** 2))
    sse = float(np.sum((actual - predicted) ** 2))
    if np.allclose(predicted, 0):
        direction = np.nan  # "predict 0" never says up or down
    else:
        direction = float(np.mean(np.sign(actual) == np.sign(predicted)) * 100)
    return {
        "Model": name,
        "RMSE (%)": _rmse(actual, predicted) * 100,
        "MAE (%)": float(mean_absolute_error(actual, predicted)) * 100,
        "Direction accuracy (%)": direction,
        # > 0 means the model's squared error is lower than always guessing 0.
        "Skill vs predict-0 (%)": (1 - sse / sse_zero) * 100 if sse_zero > 0 else np.nan,
    }


def diebold_mariano(actual, pred_a, pred_b, horizon: int = 1):
    """Diebold-Mariano test: do models A and B have different squared errors?

    Uses the loss difference d_t = e_a^2 - e_b^2. Overlapping h-day targets make
    d_t autocorrelated, so the variance includes autocovariances up to lag h-1.
    The Harvey-Leybourne-Newbold small-sample correction is applied and the
    statistic is compared with a t distribution (n-1 degrees of freedom).

    Returns (statistic, two-sided p-value). A negative statistic means A has the
    lower error; p < 0.05 means the difference is unlikely to be chance.
    """
    actual = np.asarray(actual, dtype=float)
    d = (actual - np.asarray(pred_a, float)) ** 2 - (actual - np.asarray(pred_b, float)) ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for k in range(1, horizon):
        var += 2 * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = d.mean() / np.sqrt(var / n) * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def time_series_folds(n: int, horizon: int, n_folds: int = 4, start_frac: float = 0.5):
    """Expanding-window folds over n rows: [(train_end, valid_start, valid_end), ...].

    The second half of the rows is cut into n_folds consecutive validation
    blocks; each block is predicted by a model trained on every earlier row,
    minus an h-row embargo so no training target overlaps the block.
    """
    edges = np.linspace(int(n * start_frac), n, n_folds + 1).astype(int)
    return [(max(0, edges[k] - horizon), edges[k], edges[k + 1]) for k in range(n_folds)]


def train_and_evaluate(data: pd.DataFrame, features: list[str],
                       horizon: int = DEFAULT_HORIZON, tune: bool = True,
                       cv_folds: int = 4):
    """Tune and select models, then score every model once on the test set.

    cv_folds > 0: each setting is scored by its average RMSE over `cv_folds`
    time-series folds inside train+validation, so the choice does not hinge on
    one validation period. cv_folds = 0: the single validation period is used.
    """
    train, valid, test = chronological_split(data, horizon)
    Xtr, ytr = train[features], train["target"]
    Xva, yva = valid[features], valid["target"]
    dev = pd.concat([train, valid], ignore_index=True)
    Xdev, ydev, Xte, yte = dev[features], dev["target"], test[features], test["target"]

    if cv_folds:
        folds = time_series_folds(len(dev), horizon, cv_folds)
        selection = f"time-series cross-validation ({cv_folds} folds inside train+validation)"

        def score(name, params):
            errs = []
            for tr_end, va_start, va_end in folds:
                tr, va = dev.iloc[:tr_end], dev.iloc[va_start:va_end]
                model = _fit(build_model(name, params), tr[features], tr["target"])
                errs.append(_rmse(va["target"], model.predict(va[features])))
            return float(np.mean(errs))
    else:
        selection = "single validation period"

        def score(name, params):
            return _rmse(yva, _fit(build_model(name, params), Xtr, ytr).predict(Xva))

    tuning_rows, best_params, valid_rmse = [], {}, {}
    for name, (_, space) in MODEL_SPACE.items():
        grid = _grid(space) if tune else _grid(space)[:1]
        for params in grid:
            s = score(name, params)
            tuning_rows.append({"Model": name, "Params": _fmt_params(params),
                                "Selection RMSE (%)": s * 100})
            if name not in valid_rmse or s < valid_rmse[name]:
                valid_rmse[name], best_params[name] = s, params
    for name in BASELINES:
        valid_rmse[name] = score(name, None)

    best_name = min(MODEL_SPACE, key=lambda m: valid_rmse[m])

    # Refit each model's chosen settings on train+validation; test is used once.
    predictions = test[["Date", "Close", "target"]].reset_index(drop=True)
    metrics, fitted = [], {}
    for name in list(MODEL_SPACE) + list(BASELINES):
        model = _fit(build_model(name, best_params.get(name)), Xdev, ydev)
        fitted[name] = model
        pred = model.predict(Xte)
        predictions[name] = pred
        row = metric_row(name, yte, pred)
        row["DM p-value vs predict-0"] = (np.nan if name == "Baseline: predict 0" else
                                          diebold_mariano(yte, pred, np.zeros(len(pred)), horizon)[1])
        row["Selection RMSE (%)"] = valid_rmse[name] * 100
        metrics.append(row)

    metrics = pd.DataFrame(metrics)
    importance = feature_importances(fitted[best_name], best_name, features, Xte, yte)
    return {
        "best_name": best_name,
        "best_params": best_params,
        "metrics": metrics.sort_values("RMSE (%)").reset_index(drop=True),
        "tuning": pd.DataFrame(tuning_rows).sort_values(["Model", "Selection RMSE (%)"]),
        "selection": selection,
        "predictions": predictions,
        "model": fitted[best_name],
        "feature_importance": importance,
        "group_importance": group_importance(importance),
        "n_train": len(train), "n_validation": len(valid), "n_test": len(test),
        "dates": {
            "train": (train["Date"].min(), train["Date"].max()),
            "validation": (valid["Date"].min(), valid["Date"].max()),
            "test": (test["Date"].min(), test["Date"].max()),
        },
        "features": features,
        "horizon": horizon,
    }


def own_vs_peer_ablation(data: pd.DataFrame, features: list[str], best_params: dict,
                         horizon: int = DEFAULT_HORIZON):
    """Do supplier/peer features help? Every model, with and without them.

    Each model keeps the settings chosen on validation; only the feature set
    changes. Scored on the same test period. Lower RMSE is better.
    """
    own = [f for f in features if f in OWN_FEATURES]
    if len(own) == len(features):
        return None
    train, valid, test = chronological_split(data, horizon)
    dev = pd.concat([train, valid], ignore_index=True)
    zero = _rmse(test["target"], np.zeros(len(test))) * 100
    rows = []
    for name in MODEL_SPACE:
        rmse = {}
        for label, cols in [("own", own), ("peers", features)]:
            model = _fit(build_model(name, best_params.get(name)), dev[cols], dev["target"])
            rmse[label] = _rmse(test["target"], model.predict(test[cols])) * 100
        rows.append({
            "Model": name,
            "RMSE own only (%)": rmse["own"],
            "RMSE own + peers (%)": rmse["peers"],
            "Change (%)": rmse["peers"] - rmse["own"],
            "Peers help?": "yes" if rmse["peers"] < rmse["own"] else "no",
            "Beats predict-0 with peers?": "yes" if rmse["peers"] < zero else "no",
        })
    return pd.DataFrame(rows)


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def feature_importances(model, name, features, X_eval=None, y_eval=None) -> pd.DataFrame:
    if name in ("Random Forest", "Decision Tree"):
        values, method = model.feature_importances_, "impurity"
    elif name == "Elastic Net":
        values, method = np.abs(model[-1].coef_), "|coefficient| (scaled)"
    elif X_eval is not None:
        values = permutation_importance(model, X_eval, y_eval, n_repeats=5,
                                        random_state=42).importances_mean
        values, method = np.clip(values, 0, None), "permutation"
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Method"])
    return pd.DataFrame({"Feature": features, "Importance": values, "Method": method}
                        ).sort_values("Importance", ascending=False).reset_index(drop=True)


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return importance
    total = importance["Importance"].sum()
    g = importance.assign(Source=importance["Feature"].map(feature_group))
    g = g.groupby("Source", as_index=False)["Importance"].sum()
    g["Share (%)"] = g["Importance"] / total * 100 if total > 0 else 0.0
    return g.sort_values("Share (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward (rolling) evaluation
# ---------------------------------------------------------------------------
def walk_forward(data: pd.DataFrame, features: list[str], horizon: int = DEFAULT_HORIZON,
                 n_folds: int = 5, start_frac: float = 0.5):
    """Expanding-window evaluation, closer to the paper's rolling procedure.

    The second half of the history is cut into `n_folds` consecutive test
    blocks. For each block, every model is trained on all rows before it
    (minus an h-row embargo) with its default settings, so no hyperparameter
    was chosen using these test blocks.
    """
    n = len(data)
    edges = np.linspace(int(n * start_frac), n, n_folds + 1).astype(int)
    rows = []
    for k in range(n_folds):
        start, end = edges[k], edges[k + 1]
        train = data.iloc[:max(0, start - horizon)]
        test = data.iloc[start:end]
        if len(train) < 50 or len(test) < 10:
            continue
        for name in list(MODEL_SPACE) + list(BASELINES):
            pred = _fit(build_model(name), train[features], train["target"]).predict(test[features])
            r = metric_row(name, test["target"], pred)
            rows.append({"Fold": k + 1,
                         "Test period": f"{test['Date'].min().date()} → {test['Date'].max().date()}",
                         "Model": name, "RMSE (%)": r["RMSE (%)"],
                         "Direction accuracy (%)": r["Direction accuracy (%)"]})
    folds = pd.DataFrame(rows)
    if folds.empty:
        raise ValueError("Not enough data for walk-forward evaluation.")
    summary = (folds.groupby("Model")
               .agg(**{"Mean RMSE (%)": ("RMSE (%)", "mean"),
                       "Worst fold RMSE (%)": ("RMSE (%)", "max"),
                       "Mean direction acc. (%)": ("Direction accuracy (%)", "mean")})
               .sort_values("Mean RMSE (%)").reset_index())
    zero = folds[folds["Model"] == "Baseline: predict 0"].set_index("Fold")["RMSE (%)"]
    beats = (folds.assign(zero=folds["Fold"].map(zero))
             .assign(beat=lambda d: d["RMSE (%)"] < d["zero"])
             .groupby("Model")["beat"].sum().astype(int))
    summary["Folds beating predict-0"] = [
        "—" if m == "Baseline: predict 0" else f"{beats[m]} / {folds['Fold'].nunique()}"
        for m in summary["Model"]
    ]
    return folds, summary


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(predictions: pd.DataFrame, model_col: str, threshold_pct: float = 0.25,
             transaction_cost_pct: float = 0.05):
    """Long/flat strategy on the test period.

    A forecast made at the close of day t-1 decides whether we hold the stock
    during day t (signal is lagged by one day, so no look-ahead). Each change
    of position (buy or sell) pays `transaction_cost_pct`. Held one day at a
    time so overlapping h-day forecasts are not double-counted.
    """
    p = predictions[["Date", "Close", model_col]].sort_values("Date").reset_index(drop=True)
    p = p.rename(columns={model_col: "predicted_return"})
    p["daily_return"] = _simple_returns(p["Close"])
    p["signal"] = (p["predicted_return"].shift(1) > threshold_pct / 100).astype(int)
    p["turnover"] = p["signal"].diff().abs().fillna(p["signal"].abs())
    p["strategy_return"] = (p["signal"] * p["daily_return"]
                            - p["turnover"] * transaction_cost_pct / 100)
    p["buy_hold_return"] = p["daily_return"]
    p = p.dropna(subset=["daily_return"]).reset_index(drop=True)
    if p.empty:
        raise ValueError("Not enough test observations for backtest.")
    p["strategy_equity"] = (1 + p["strategy_return"]).cumprod()
    p["buy_hold_equity"] = (1 + p["buy_hold_return"]).cumprod()
    summary = pd.DataFrame([
        {"Strategy": f"ML long/flat ({model_col})",
         **risk_stats(p["strategy_return"]),
         "Trades": int(p["turnover"].sum()),
         "Time in market (%)": float(p["signal"].mean() * 100)},
        {"Strategy": "Buy and hold", **risk_stats(p["buy_hold_return"]),
         "Trades": 1, "Time in market (%)": 100.0},
    ])
    return p, summary


def risk_stats(daily: pd.Series) -> dict:
    equity = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    total = equity.iloc[-1] - 1
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    return {
        "Total return (%)": float(total * 100),
        "Annual return (%)": float(((1 + total) ** (1 / years) - 1) * 100) if years > 0 else np.nan,
        "Annual volatility (%)": float(vol * 100),
        # Sharpe with a 0% risk-free rate, for simplicity.
        "Sharpe ratio": float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else np.nan,
        "Max drawdown (%)": float((equity / equity.cummax() - 1).min() * 100),
    }


def threshold_table(predictions: pd.DataFrame, model_col: str,
                    thresholds=(0.0, 0.25, 0.5, 1.0), transaction_cost_pct: float = 0.05):
    """Like Table 2 of the paper: profit and number of trades per threshold."""
    rows = []
    for c in thresholds:
        _, s = backtest(predictions, model_col, c, transaction_cost_pct)
        r = s.iloc[0]
        rows.append({"Rule": f"ML, buy if forecast > {c:g}%", "Total return (%)": r["Total return (%)"],
                     "Sharpe ratio": r["Sharpe ratio"], "Max drawdown (%)": r["Max drawdown (%)"],
                     "Trades": r["Trades"], "Time in market (%)": r["Time in market (%)"]})
    bh = s.iloc[1]
    rows.append({"Rule": "Buy and hold", "Total return (%)": bh["Total return (%)"],
                 "Sharpe ratio": bh["Sharpe ratio"], "Max drawdown (%)": bh["Max drawdown (%)"],
                 "Trades": 1, "Time in market (%)": 100.0})
    return pd.DataFrame(rows)


def compare_horizons(frame: pd.DataFrame, peers: pd.DataFrame | None, horizons=(1, 5, 20),
                     peer_lags: int = 5, tune: bool = True, cv_folds: int = 4,
                     target_mode: str = "raw", market: str = "^NSEI") -> pd.DataFrame:
    """Like Table 1 of the paper: test RMSE for every model at each horizon."""
    tables = []
    for h in horizons:
        data, feats, _ = make_features(frame, h, peers, peer_lags, target_mode=target_mode, market=market)
        m = train_and_evaluate(data, feats, h, tune, cv_folds)["metrics"][["Model", "RMSE (%)"]]
        tables.append(m.set_index("Model").rename(columns={"RMSE (%)": f"h={h}"}))
    return pd.concat(tables, axis=1).sort_values(f"h={horizons[0]}").reset_index()


# ---------------------------------------------------------------------------
# Yahoo Finance download
# ---------------------------------------------------------------------------
def download_yfinance(ticker: str, period: str = "10y") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker, internet, "
                         "and Yahoo Finance availability.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    if "Date" not in raw.columns and "Datetime" in raw.columns:
        raw = raw.rename(columns={"Datetime": "Date"})
    return normalize_ohlcv(raw)


def download_peer_closes(tickers: list[str], period: str = "10y"):
    """Closing prices for several tickers. Returns (DataFrame with Date column, failed tickers)."""
    import yfinance as yf
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        return None, []
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return None, tickers
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if list(close.columns) == ["Close"]:
        close.columns = tickers[:1]
    failed = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    close = close[[t for t in tickers if t not in failed]]
    if close.empty:
        return None, failed
    close = close.reset_index().rename(columns={"Datetime": "Date"})
    close.columns.name = None
    return close, failed
