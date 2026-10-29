"""Run the full experiment from the command line and save every table + chart.

Examples
--------
Real data (needs internet):
    python run_experiment.py --ticker MARUTI.NS
    python run_experiment.py --ticker MARUTI.NS --no-peers
Offline / saved CSVs:
    python run_experiment.py --csv data/synthetic_demo.csv --peers-csv data/synthetic_peers_demo.csv

Outputs go to reports/results/<name>/ : metrics, tuning, ablation, thresholds,
walk-forward, horizons, equity_curve.png, data CSVs and summary.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trading_ml import (
    DEFAULT_PEERS, DEFAULT_TARGET, backtest, compare_horizons, download_peer_closes,
    download_yfinance, make_features, normalize_peers, own_vs_peer_ablation,
    threshold_table, train_and_evaluate, walk_forward,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=DEFAULT_TARGET)
    ap.add_argument("--peers", nargs="*", default=DEFAULT_PEERS)
    ap.add_argument("--no-peers", action="store_true", help="Own price/volume features only")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--csv", help="Target OHLCV CSV instead of downloading")
    ap.add_argument("--peers-csv", help="Peer closes CSV instead of downloading")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--peer-lags", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.25, help="percent")
    ap.add_argument("--cost", type=float, default=0.05, help="percent per buy/sell")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--target", choices=["raw", "excess"], default="raw",
                    help="raw = stock return; excess = stock return minus NIFTY 50 return")
    ap.add_argument("--cv-folds", type=int, default=4,
                    help="time-series CV folds for model selection (0 = single validation period)")
    ap.add_argument("--name", help="output folder name (default: ticker or CSV name)")
    ap.add_argument("--out", default="reports/results")
    args = ap.parse_args()

    # ---- data
    if args.csv:
        frame = pd.read_csv(args.csv)
        peers_raw = pd.read_csv(args.peers_csv) if args.peers_csv and not args.no_peers else None
        name = Path(args.csv).stem
        source = f"CSV file {args.csv}"
    else:
        frame = download_yfinance(args.ticker, args.period)
        peers_raw, failed = (None, []) if args.no_peers else download_peer_closes(args.peers, args.period)
        if failed:
            print("No data for (skipped):", ", ".join(failed))
        name = args.ticker.replace("^", "")
        source = f"Yahoo Finance via yfinance, period={args.period}"
    name += "_own_only" if args.no_peers else ""
    name += "_excess" if args.target == "excess" else ""
    name = args.name or name
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "data_target.csv", index=False)
    if peers_raw is not None:
        peers_raw.to_csv(out / "data_peers.csv", index=False)

    peers = normalize_peers(peers_raw) if peers_raw is not None else None
    data, features, info = make_features(frame, args.horizon, peers, args.peer_lags,
                                         target_mode=args.target)

    # ---- models
    res = train_and_evaluate(data, features, args.horizon, not args.no_tune, args.cv_folds)
    best = res["best_name"]
    res["metrics"].to_csv(out / "metrics.csv", index=False)
    res["tuning"].to_csv(out / "tuning.csv", index=False)
    res["feature_importance"].to_csv(out / "feature_importance.csv", index=False)
    res["group_importance"].to_csv(out / "importance_by_source.csv", index=False)
    res["predictions"].to_csv(out / "test_predictions.csv", index=False)
    ablation = own_vs_peer_ablation(data, features, res["best_params"], args.horizon)
    if ablation is not None:
        ablation.to_csv(out / "ablation_peers.csv", index=False)

    # ---- strategy
    bt, summary = backtest(res["predictions"], best, args.threshold, args.cost)
    summary.to_csv(out / "strategy_summary.csv", index=False)
    thresholds = threshold_table(res["predictions"], best, transaction_cost_pct=args.cost)
    thresholds.to_csv(out / "thresholds.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bt["Date"], bt["strategy_equity"], label=f"ML long/flat ({best})")
    ax.plot(bt["Date"], bt["buy_hold_equity"], label="Buy and hold")
    ax.set_title(f"{name}: test-period equity (threshold {args.threshold}%, cost {args.cost}%)")
    ax.set_ylabel("Growth of 1 rupee")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=150)

    # ---- robustness
    folds, wf = walk_forward(data, features, args.horizon)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    wf.to_csv(out / "walk_forward_summary.csv", index=False)
    horizons = compare_horizons(frame, peers, (1, 5, 20), args.peer_lags, not args.no_tune,
                                args.cv_folds, args.target)
    horizons.to_csv(out / "horizons.csv", index=False)

    # ---- summary for the write-up
    def md(df):
        r = df.round(3)
        return r.astype(object).where(r.notna(), "—").to_markdown(index=False)

    zero = res["metrics"].set_index("Model").loc["Baseline: predict 0", "RMSE (%)"]
    best_rmse = res["metrics"].set_index("Model").loc[best, "RMSE (%)"]
    lines = [
        f"# Results: {name}",
        "",
        f"- Source: {source}",
        f"- Rows used: {info['rows_used']} (from {info['rows_raw']}); "
        f"{data['Date'].min().date()} → {data['Date'].max().date()}",
        f"- Target: forward {args.horizon}-day return" + (" minus NIFTY 50's (excess)" if args.target == "excess" else "")
        + f"; features: {len(features)}",
        f"- Peers used: {', '.join(info['peers_used']) or 'none'}"
        + (f"; dropped (coverage %): {info['peers_dropped']}" if info["peers_dropped"] else ""),
        "- Split: " + "; ".join(f"{k} {v[0].date()} → {v[1].date()}" for k, v in res["dates"].items()),
        f"- Model selection: {res['selection']}",
        f"- Selected model: **{best}** ({res['best_params'][best]})",
        f"- Test RMSE {best_rmse:.3f}% vs predict-0 baseline {zero:.3f}% → "
        + ("beats baseline" if best_rmse < zero else "does NOT beat baseline")
        + f"; Diebold–Mariano p = {res['metrics'].set_index('Model').loc[best, 'DM p-value vs predict-0']:.3f}",
        "",
        "## Test metrics", md(res["metrics"]), "",
    ]
    if ablation is not None:
        lines += ["## Do peers help? (each model, with vs without)", md(ablation), ""]
    lines += [
        "## Importance by source", md(res["group_importance"]), "",
        f"## Strategy (threshold {args.threshold}%, cost {args.cost}%)", md(summary), "",
        "## Threshold sensitivity", md(thresholds), "",
        "## Walk-forward (5 periods)", md(wf), "",
        "## Test RMSE (%) by horizon", md(horizons), "",
        "![equity](equity_curve.png)",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\nSaved everything to {out}")


if __name__ == "__main__":
    main()
