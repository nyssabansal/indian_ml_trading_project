"""Sanity checks. Run with:  python -m pytest -q"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trading_ml as tm  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA / "synthetic_demo.csv")


@pytest.fixture(scope="module")
def peers():
    return tm.normalize_peers(pd.read_csv(DATA / "synthetic_peers_demo.csv"))


def test_features_do_not_use_future_prices(frame):
    """Changing prices after day t must not change the features of day t."""
    data, feats, _ = tm.make_features(frame)
    cut = 600
    changed = frame.copy()
    later = pd.to_datetime(changed["Date"]) > pd.to_datetime(data.loc[cut, "Date"])
    changed.loc[later, ["Open", "High", "Low", "Close"]] *= 1.5
    data2, _, _ = tm.make_features(changed)
    pd.testing.assert_frame_equal(data.loc[:cut, feats], data2.loc[:cut, feats])


def test_target_is_forward_return(frame):
    data, _, _ = tm.make_features(frame, horizon=5)
    raw = tm.normalize_ohlcv(frame).set_index("Date")["Close"]
    row = data.iloc[100]
    i = raw.index.get_loc(row["Date"])
    assert row["target"] == pytest.approx(raw.iloc[i + 5] / raw.iloc[i] - 1)


@pytest.mark.parametrize("h", [1, 5, 20])
def test_split_is_ordered_with_embargo(frame, h):
    data, _, _ = tm.make_features(frame, horizon=h)
    train, valid, test = tm.chronological_split(data, h)
    assert train["Date"].max() < valid["Date"].min() < valid["Date"].max() < test["Date"].min()
    # at least h rows between blocks, so no training target overlaps the next block
    assert valid.index.min() - train.index.max() > h
    assert test.index.min() - valid.index.max() > h


def test_peer_features_are_lagged_returns(frame, peers):
    data, feats, info = tm.make_features(frame, peers=peers, peer_lags=3)
    assert info["peers_used"] and len(feats) == len(tm.OWN_FEATURES) + 3 * len(info["peers_used"])
    p = info["peers_used"][0]
    assert np.allclose(data[f"{p}_lag1"].iloc[1:].to_numpy(), data[f"{p}_lag0"].iloc[:-1].to_numpy())


def test_late_listed_peer_is_dropped(frame, peers):
    late = peers.copy()
    late["LATE"] = late.iloc[:, 0]
    late.iloc[:500, late.columns.get_loc("LATE")] = np.nan
    _, _, info = tm.make_features(frame, peers=late)
    assert "LATE" in info["peers_dropped"]


def test_predict_zero_baseline_and_skill(frame):
    data, feats, _ = tm.make_features(frame)
    res = tm.train_and_evaluate(data, feats, tune=False)
    m = res["metrics"].set_index("Model")
    zero = m.loc["Baseline: predict 0"]
    y = res["predictions"]["target"]
    assert zero["RMSE (%)"] == pytest.approx(np.sqrt((y ** 2).mean()) * 100)
    assert zero["Skill vs predict-0 (%)"] == pytest.approx(0)
    assert np.isnan(zero["Direction accuracy (%)"])


def test_backtest_signal_is_lagged_and_costs_apply():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    preds = pd.DataFrame({"Date": dates, "Close": [100, 101, 102, 101, 103, 104.0],
                          "target": 0.0, "m": [0.01, 0.01, -0.01, 0.01, 0.01, 0.01]})
    bt, _ = tm.backtest(preds, "m", threshold_pct=0.0, transaction_cost_pct=0.0)
    # forecast on day 0 decides position on day 1; day-2 forecast (-1%) means flat on day 3
    assert bt["signal"].tolist() == [1, 1, 0, 1, 1]
    bt_cost, _ = tm.backtest(preds, "m", threshold_pct=0.0, transaction_cost_pct=1.0)
    assert bt_cost["strategy_equity"].iloc[-1] < bt["strategy_equity"].iloc[-1]


def test_walk_forward_runs(frame):
    data, feats, _ = tm.make_features(frame)
    folds, summary = tm.walk_forward(data, feats, n_folds=3)
    assert folds["Fold"].nunique() == 3
    assert "Baseline: predict 0" in summary["Model"].tolist()


def test_ablation_covers_every_model(frame, peers):
    data, feats, _ = tm.make_features(frame, peers=peers)
    res = tm.train_and_evaluate(data, feats, tune=False)
    ab = tm.own_vs_peer_ablation(data, feats, res["best_params"])
    assert set(ab["Model"]) == set(tm.MODEL_SPACE)
    assert tm.own_vs_peer_ablation(data, tm.OWN_FEATURES, res["best_params"]) is None


def test_diebold_mariano_detects_real_differences():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.02, 400)
    good = y + rng.normal(0, 0.002, 400)          # nearly perfect forecasts
    stat, p = tm.diebold_mariano(y, good, np.zeros(400), horizon=5)
    assert stat < 0 and p < 0.01                   # clearly better than predict-0
    stat, p = tm.diebold_mariano(y, np.zeros(400), np.zeros(400))
    assert np.isnan(p)                             # identical forecasts: no test


def test_time_series_folds_are_ordered_with_embargo():
    folds = tm.time_series_folds(1000, horizon=5, n_folds=4)
    assert len(folds) == 4 and folds[-1][2] == 1000
    for tr_end, va_start, va_end in folds:
        assert va_start - tr_end == 5 and va_start < va_end
    assert all(folds[k][2] == folds[k + 1][1] for k in range(3))


def test_excess_target_subtracts_market_return(frame, peers):
    data, _, _ = tm.make_features(frame, horizon=5, peers=peers, target_mode="excess", market="SYN_INDEX")
    raw = tm.normalize_ohlcv(frame).set_index("Date")["Close"]
    mkt = peers["SYN_INDEX"]
    row = data.iloc[100]
    i = raw.index.get_loc(row["Date"])
    expected = (raw.iloc[i + 5] / raw.iloc[i] - 1) - (mkt.loc[raw.index[i + 5]] / mkt.loc[raw.index[i]] - 1)
    assert row["target"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        tm.make_features(frame, peers=None, target_mode="excess")


def test_cv_selection_reports_method(frame):
    data, feats, _ = tm.make_features(frame)
    res = tm.train_and_evaluate(data, feats, tune=False, cv_folds=3)
    assert "cross-validation" in res["selection"]
    assert "DM p-value vs predict-0" in res["metrics"].columns
