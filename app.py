from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_ml import (
    DEFAULT_PEERS, DEFAULT_TARGET, align_peers, backtest, compare_horizons, download_peer_closes,
    download_yfinance, make_features, normalize_ohlcv, normalize_peers, own_vs_peer_ablation,
    threshold_table, train_and_evaluate, walk_forward,
)

ROOT = Path(__file__).parent
DEMO_PATH = ROOT / "data" / "synthetic_demo.csv"
DEMO_PEERS_PATH = ROOT / "data" / "synthetic_peers_demo.csv"
ZERO = "Baseline: predict 0"

# Friendly names and supply-chain roles for the default tickers (and the simulated demo ones)
NAMES = {
    "MARUTI.NS": "Maruti Suzuki", "MOTHERSON.NS": "Motherson", "BOSCHLTD.NS": "Bosch",
    "UNOMINDA.NS": "Uno Minda", "ENDURANCE.NS": "Endurance", "MRF.NS": "MRF",
    "APOLLOTYRE.NS": "Apollo Tyres", "EXIDEIND.NS": "Exide", "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel", "^NSEI": "NIFTY 50", "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra", "BAJAJ-AUTO.NS": "Bajaj Auto", "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS": "Eicher Motors", "SYN_PARTS_A": "Parts maker A", "SYN_PARTS_B": "Parts maker B",
    "SYN_TYRES": "Tyre maker", "SYN_STEEL": "Steel maker", "SYN_INDEX": "Market index",
}
ROLES = {
    "MOTHERSON.NS": "Auto components", "BOSCHLTD.NS": "Auto components", "UNOMINDA.NS": "Auto components",
    "ENDURANCE.NS": "Auto components", "MRF.NS": "Tyres & batteries", "APOLLOTYRE.NS": "Tyres & batteries",
    "EXIDEIND.NS": "Tyres & batteries", "TATASTEEL.NS": "Steel", "JSWSTEEL.NS": "Steel",
    "^NSEI": "Market index", "SYN_PARTS_A": "Auto components", "SYN_PARTS_B": "Auto components",
    "SYN_TYRES": "Tyres & batteries", "SYN_STEEL": "Steel", "SYN_INDEX": "Market index",
}


def nice(t: str) -> str:
    return NAMES.get(t, t.replace(".NS", ""))


def role(t: str) -> str:
    return ROLES.get(t, "Other peers")

# Palette shared with the slides
# Dark theme (matches .streamlit/config.toml)
BG, PANEL, BORDER, TEXT, MUTED, GRID = "#0B1220", "#131C2E", "#22304A", "#E6EDF7", "#9AA8BC", "#1E2A40"
INK, TEAL, AMBER, GREY, CORAL, SLATE, LIGHT = (
    "#14213D", "#2A9D8F", "#FCA311", "#8D99AE", "#E76F51", "#5B8BD0", PANEL)
ROLE_COLORS = {"Auto components": "#5B8BD0", "Tyres & batteries": "#2A9D8F", "Steel": "#8D99AE",
               "Market index": "#E76F51", "Other peers": "#98C1D9"}

st.set_page_config(page_title="Indian Stock Return Forecasting", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
h1, h2, h3 {{ font-family: Cambria, Georgia, serif !important; color: {TEXT}; letter-spacing: -0.2px; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
.hero {{ background: linear-gradient(120deg, #16264A 0%, #0F1A33 60%, #13243F 100%); border: 1px solid {BORDER}; color: white; border-radius: 16px; padding: 26px 32px; margin-bottom: 18px; }}
.hero h1 {{ color: white !important; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0; }}
.hero p {{ color: #C9D6E8; margin: 0; font-size: 1.02rem; max-width: calc(100% - 150px); }}
.hero h1 {{ max-width: calc(100% - 150px); }}
.hero .tag {{ display: inline-block; background: {AMBER}; color: {INK}; font-weight: 700;
             font-size: 0.75rem; padding: 3px 10px; border-radius: 999px; margin-bottom: 10px; }}
.kpi {{ background: {LIGHT}; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.kpi {{ border: 1px solid {BORDER}; }}
.kpi .label {{ font-size: 0.8rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.04em; }}
.kpi .value {{ font-family: Cambria, Georgia, serif; font-size: 1.85rem; font-weight: 700;
              color: {TEXT}; line-height: 1.2; margin: 4px 0 2px 0; }}
.kpi .sub {{ font-size: 0.85rem; color: {MUTED}; }}
.kpi.good .value {{ color: {TEAL}; }}
.kpi.bad .value {{ color: {CORAL}; }}
.hero {{ position: relative; overflow: hidden; }}
.hero .art {{ position: absolute; right: 28px; top: 50%; transform: translateY(-50%); font-size: 84px; opacity: 0.95; }}
.hero .chain {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
.hero .chip {{ background: rgba(255,255,255,0.10); color: #E6EDF7; border-radius: 999px; padding: 3px 12px; font-size: 0.82rem; }}
.tape {{ display: flex; gap: 10px; overflow-x: auto; padding: 2px 0 10px 0; margin-bottom: 8px; }}
.tick {{ flex: 0 0 auto; min-width: 128px; border: 1px solid {BORDER}; border-radius: 12px; padding: 8px 12px; background: {PANEL}; }}
.tick.main {{ background: #1A2B4D; border-color: {AMBER}; }}
.tick .n {{ font-size: 0.78rem; color: {MUTED}; white-space: nowrap; }}
.tick.main .n {{ color: #C9D6E8; }}
.tick .p {{ font-family: Cambria, Georgia, serif; font-weight: 700; font-size: 1.05rem; color: {TEXT}; }}
.tick.main .p {{ color: white; }}
.tick .c {{ font-size: 0.8rem; font-weight: 600; }}
.up {{ color: #3CCB9F; }} .down {{ color: {CORAL}; }}
.tick .r {{ font-size: 0.68rem; color: #6F7E95; white-space: nowrap; }}
.note {{ margin-top: 12px; background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; font-size: 0.92rem; color: {TEXT}; }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def style(fig, title=None, height=380, ytitle=None, xtitle=None):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=20),
        title=dict(text=title, font=dict(size=16, color=TEXT, family="Cambria, Georgia, serif")) if title else None,
        font=dict(family="Calibri, Segoe UI, sans-serif", size=13, color=TEXT),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0),
        hoverlabel=dict(font_size=13), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, title=xtitle, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, title=ytitle, tickfont=dict(color=MUTED), zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def table(df, decimals: dict):
    """Show a dataframe with fixed decimals per column and '—' for missing values."""
    shown = df.copy()
    for c, d in decimals.items():
        if c in shown.columns:
            shown[c] = [("—" if pd.isna(v) else f"{v:.{d}f}") for v in shown[c]]
    st.dataframe(shown, hide_index=True, width="stretch")


def pct_cols(df, cols, fmt="%.3f"):
    return {c: st.column_config.NumberColumn(c, format=fmt) for c in cols if c in df.columns}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 1 · Data")
    source = st.radio("Source", ["Bundled synthetic demo", "Upload CSV", "Download from Yahoo Finance"])
    uploaded = uploaded_peers = None
    ticker, peer_text, period, use_demo_peers = DEFAULT_TARGET, ", ".join(DEFAULT_PEERS), "10y", True
    if source == "Bundled synthetic demo":
        use_demo_peers = st.checkbox("Include synthetic peer stocks", True)
    elif source == "Upload CSV":
        ticker = st.text_input("Ticker of the uploaded stock (for labels)", DEFAULT_TARGET)
        uploaded = st.file_uploader("Target stock OHLCV CSV", type=["csv"])
        uploaded_peers = st.file_uploader("Optional: supplier/peer closes CSV", type=["csv"],
                                          help="Columns: Date, then one closing-price column per ticker.")
    else:
        ticker = st.text_input("Target ticker", DEFAULT_TARGET, help="e.g. MARUTI.NS, RELIANCE.NS, TCS.NS")
        peer_text = st.text_area("Supplier / peer tickers (comma separated, optional)", peer_text,
                                 help="Their past returns become extra features, as in the Stanford paper.")
        period = st.selectbox("History", ["5y", "10y", "max"], index=1)

    st.markdown("### 2 · Model")
    horizon = st.selectbox("Forecast horizon (trading days)", [1, 5, 20], index=1)
    target_label = st.radio("What to predict", ["Stock return", "Return vs NIFTY 50 (excess)"],
                            help="Excess = the stock's return minus the market's over the same days. "
                                 "Needs ^NSEI in the peer list.")
    target_mode = "excess" if target_label.startswith("Return vs") else "raw"
    use_cv = st.checkbox("Choose models with time-series cross-validation (4 folds)", True,
                         help="Off = choose on one validation period only.")
    peer_lags = st.slider("Peer return lags (days)", 1, 28, 5,
                          help="The paper used 28 days of lagged supplier returns.")
    tune = st.checkbox("Tune hyperparameters", True)

    st.markdown("### 3 · Strategy")
    threshold = st.slider("Buy threshold: predicted return above (%)", 0.0, 2.0, 0.25, 0.05)
    costs = st.slider("Transaction cost per buy/sell (%)", 0.0, 0.5, 0.05, 0.01)


# ---------------------------------------------------------------- header
company = "a simulated automaker" if source == "Bundled synthetic demo" else nice(ticker.strip())
st.markdown(f"""
<div class="hero">
  <div class="art">🚗</div>
  <div class="tag">Indian Stock Return Forecasting · UE24CS352A Machine Learning</div>
  <h1>Can supplier stocks predict {company}?</h1>
  <p>A carmaker depends on the companies that make its parts, tyres and steel. We test whether
  their share-price moves on the NSE carry a signal about the carmaker's next move.</p>
  <div class="chain"><span class="chip">🔩 Auto components</span><span class="chip">🛞 Tyres &amp; batteries</span>
  <span class="chip">🏗️ Steel</span><span class="chip">📈 NIFTY 50</span><span class="chip">→ 🚗 {company}</span></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_demo(with_peers: bool):
    peers = pd.read_csv(DEMO_PEERS_PATH) if with_peers else None
    return pd.read_csv(DEMO_PATH), peers


@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo(tkr: str, peers_csv: str, per: str):
    frame = download_yfinance(tkr, per)
    peer_list = [p.strip() for p in peers_csv.split(",") if p.strip()]
    peers, failed = download_peer_closes(peer_list, per) if peer_list else (None, [])
    return frame, peers, failed


frame = peers_raw = None
failed_peers = []
try:
    if source == "Bundled synthetic demo":
        frame, peers_raw = load_demo(use_demo_peers)
        st.info("You are using **simulated** data for an offline demo. Its results say nothing about real markets.")
    elif source == "Upload CSV":
        if uploaded is None:
            st.warning("Upload a CSV with Date, Open, High, Low, Close, Volume columns.")
        else:
            frame = pd.read_csv(uploaded)
            peers_raw = pd.read_csv(uploaded_peers) if uploaded_peers is not None else None
    else:
        with st.spinner(f"Downloading {ticker} and peers…"):
            frame, peers_raw, failed_peers = load_yahoo(ticker.strip(), peer_text, period)
except Exception as exc:
    st.error(f"Could not load data: {exc}")

if frame is None:
    st.stop()
if failed_peers:
    st.warning("No data for: " + ", ".join(failed_peers) + ". They were skipped — check the tickers on finance.yahoo.com.")

try:
    peers = normalize_peers(peers_raw) if peers_raw is not None else None
    market = "SYN_INDEX" if source == "Bundled synthetic demo" else "^NSEI"
    data, features, info = make_features(frame, horizon, peers, peer_lags,
                                         target_mode=target_mode, market=market)
except Exception as exc:
    st.error(f"Data validation / feature engineering failed: {exc}")
    st.stop()

target_name = company[0].upper() + company[1:]
target_px = normalize_ohlcv(frame).set_index("Date")["Close"]
peer_px = (align_peers(target_px.index.to_series(), peers[info["peers_used"]], 0.0)[0]
           if info["peers_used"] else pd.DataFrame(index=target_px.index))
peer_px.index = target_px.index


def tape_item(name, series, sub, main=False, money=True):
    s_ = series.dropna()
    if len(s_) < 22:
        return ""
    last, chg = s_.iloc[-1], (s_.iloc[-1] / s_.iloc[-22] - 1) * 100
    price_txt = f"₹{last:,.0f}" if money else f"{last:,.0f}"
    cls = "up" if chg >= 0 else "down"
    return (f'<div class="tick{" main" if main else ""}"><div class="n">{name}</div>'
            f'<div class="p">{price_txt}</div><div class="c {cls}">{"▲" if chg >= 0 else "▼"} {chg:+.1f}% · 1M</div>'
            f'<div class="r">{sub}</div></div>')


items = [tape_item(target_name, target_px, "target", main=True)]
items += [tape_item(nice(t), peer_px[t], role(t), money=role(t) != "Market index") for t in info["peers_used"]]
st.markdown(f'<div class="tape">{"".join(items)}</div>', unsafe_allow_html=True)
st.caption(f"Closing prices on {target_px.index.max():%d %b %Y} and change over the last month (21 trading days).")
target_desc = f"forward {horizon}-day return" + (" minus the market's" if target_mode == "excess" else "")


# ---------------------------------------------------------------- data strip + train
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Trading days", f"{len(data):,}", f"{data['Date'].min():%b %Y} → {data['Date'].max():%b %Y}")
kpi(c2, "Features", f"{len(features)}", f"{len(features) - len(info['peers_used']) * info['peer_lags']} own + "
                                         f"{len(info['peers_used'])} peers × {info['peer_lags']} lags")
kpi(c3, "Peer stocks", f"{len(info['peers_used'])}",
    f"{len(info['peers_dropped'])} dropped for short history" if info["peers_dropped"] else "all with enough history")
kpi(c4, "Predicting", f"{horizon}-day", target_desc)
st.write("")

config = (source, ticker, peer_text, period, use_demo_peers, horizon, peer_lags, tune, target_mode, use_cv,
          getattr(uploaded, "name", None), getattr(uploaded_peers, "name", None))
left, right = st.columns([1, 3])
train_clicked = left.button("Train models and evaluate", type="primary", width="stretch")
right.caption("Tunes four models with time-series cross-validation, scores them once on the held-out "
              "test period, and compares them with two baselines." if use_cv else
              "Tunes four models on a single validation period and scores them once on the test period.")
if train_clicked:
    with st.spinner("Tuning and evaluating models… (about 20–60 seconds on real data)"):
        result = train_and_evaluate(data, features, horizon, tune, 4 if use_cv else 0)
        result["ablation"] = own_vs_peer_ablation(data, features, result["best_params"], horizon)
        st.session_state.update(result=result, config=config, wf=None, hz=None)

result = st.session_state.get("result")
if result is not None and st.session_state.get("config") != config:
    st.warning("Settings changed since the last training run — click **Train** again to update the results.")


# ---------------------------------------------------------------- headline findings
if result is not None:
    metrics = result["metrics"]
    m = metrics.set_index("Model")
    best = result["best_name"]
    zero_rmse, best_rmse = m.loc[ZERO, "RMSE (%)"], m.loc[best, "RMSE (%)"]
    best_p = m.loc[best, "DM p-value vs predict-0"]
    real = [x for x in metrics["Model"] if not x.startswith("Baseline")]
    top = min(real, key=lambda x: m.loc[x, "RMSE (%)"])
    bt0, summ0 = backtest(result["predictions"], best, threshold, costs)
    strat, bh = summ0.iloc[0]["Total return (%)"], summ0.iloc[1]["Total return (%)"]

    st.markdown("#### Headline findings")
    h1, h2, h3, h4 = st.columns(4)
    kpi(h1, "Selected model", best, "chosen by " + ("cross-validation" if use_cv else "validation"))
    kpi(h2, "Test error vs guessing 0", f"{best_rmse:.3f}%", f"predict-0: {zero_rmse:.3f}%",
        "good" if best_rmse < zero_rmse else "bad")
    sig = best_p < 0.05
    kpi(h3, "Significant?", "Yes" if sig else "No", f"Diebold–Mariano p = {best_p:.2f}", "good" if sig else "bad")
    kpi(h4, "Strategy vs buy & hold", f"{strat:+.1f}%", f"buy & hold: {bh:+.1f}%",
        "good" if strat > bh else "bad")
    beats = [x for x in real if m.loc[x, "RMSE (%)"] < zero_rmse]
    sig_beats = [x for x in beats if m.loc[x, "DM p-value vs predict-0"] < 0.05]
    if sig_beats:
        msg = f"<b>{', '.join(sig_beats)}</b> beat guessing 'no change' with a statistically significant margin."
    elif beats:
        msg = (f"<b>{', '.join(beats)}</b> had lower error than guessing 'no change', but the difference is not "
               "statistically significant — it could be luck.")
    else:
        msg = "No model beat guessing 'no change' on the test period. The returns look close to unpredictable."
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)
    st.write("")


# ---------------------------------------------------------------- tabs
tabs = st.tabs(["🚗 Overview", "📊 Models", "🔗 Supply chain", "💹 Trading", "🧪 Robustness", "🔮 Predictions"])

with tabs[0]:
    if info["peers_used"]:
        corr = {t: data["ret_1d"].corr(data[f"{t}_lag0"]) for t in info["peers_used"]}
        order_t = sorted(info["peers_used"], key=lambda t: (list(ROLE_COLORS).index(role(t)), t))
        n = len(order_t)
        ang = np.pi / 2 - np.arange(n) * 2 * np.pi / n
        xs, ys = np.cos(ang), np.sin(ang)
        fig = go.Figure()
        for t, x, y in zip(order_t, xs, ys):
            c = corr[t]
            fig.add_scatter(x=[0, x], y=[0, y], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(color=ROLE_COLORS[role(t)], width=1 + 14 * abs(c)), opacity=0.55)
        for r_ in ROLE_COLORS:
            pts = [(t, x, y) for t, x, y in zip(order_t, xs, ys) if role(t) == r_]
            if not pts:
                continue
            fig.add_scatter(
                x=[p_[1] for p_ in pts], y=[p_[2] for p_ in pts], mode="markers+text", name=r_,
                marker=dict(size=34, color=ROLE_COLORS[r_], line=dict(color=BG, width=2)),
                text=[f"<b>{nice(p_[0])}</b><br>r = {corr[p_[0]]:.2f}" for p_ in pts],
                textposition=["top center" if p_[2] >= 0 else "bottom center" for p_ in pts],
                textfont=dict(size=12, color=TEXT),
                customdata=[[nice(p_[0]), corr[p_[0]]] for p_ in pts],
                hovertemplate="%{customdata[0]}<br>Same-day correlation with " + target_name
                              + ": %{customdata[1]:.2f}<extra></extra>")
        fig.add_scatter(x=[0], y=[0], mode="markers+text", showlegend=False, hoverinfo="skip",
                        marker=dict(size=74, color=AMBER, line=dict(color=BG, width=3)),
                        text=[f"<b>{target_name.split()[0]}</b>"], textfont=dict(size=13, color=INK))
        fig.update_xaxes(visible=False, range=[-1.55, 1.55])
        fig.update_yaxes(visible=False, range=[-1.45, 1.45], scaleanchor="x")
        show(style(fig, f"{target_name}'s supply chain — how closely each stock moves with it", 500))
        strongest = max(corr, key=lambda t: abs(corr[t]))
        st.caption(f"Line thickness = correlation of daily returns with {target_name} over the whole period "
                   f"(r = 1 would mean they always move together). The strongest link is {nice(strongest)} "
                   f"(r = {corr[strongest]:.2f}). Moving together on the same day is not the same as moving "
                   "first — the 🔗 Supply chain tab tests that.")
    price = data[["Date", "Close"]]
    fig = go.Figure(go.Scatter(x=price["Date"], y=price["Close"], mode="lines", line=dict(color="#C9D6E8", width=1.8),
                               name="Close", hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
    if result is not None:
        colors = {"train": "rgba(91,139,208,0.08)", "validation": "rgba(252,163,17,0.10)", "test": "rgba(42,157,143,0.16)"}
        for k, (a, b) in result["dates"].items():
            fig.add_vrect(x0=a, x1=b, fillcolor=colors[k], line_width=0, annotation_text=k.title(),
                          annotation_position="top left", annotation_font_color=MUTED)
    show(style(fig, f"{target_name} closing price", 380, "Price (₹)"))
    peer_msg = (f"**Peers used:** {', '.join(nice(t) for t in info['peers_used'])} — each with its last {info['peer_lags']} daily returns."
                if info["peers_used"] else "**No peer stocks** — own price/volume features only.")
    if info["peers_dropped"]:
        peer_msg += " **Dropped** (too little history, % coverage): " + ", ".join(
            f"{k} ({v}%)" for k, v in info["peers_dropped"].items())
    st.markdown(peer_msg)
    if result is not None:
        st.caption(f"Split by time — train {result['n_train']:,} days, validation {result['n_validation']:,}, "
                   f"test {result['n_test']:,}. {horizon} days are dropped before each boundary so no training "
                   "target overlaps a later period.")
    if source == "Download from Yahoo Finance":
        with st.expander("Save this data for an offline backup"):
            st.download_button("Download target CSV", frame.to_csv(index=False).encode(), f"{ticker}_ohlcv.csv", "text/csv")
            if peers_raw is not None:
                st.download_button("Download peers CSV", peers_raw.to_csv(index=False).encode(),
                                   f"{ticker}_peers.csv", "text/csv")
            st.caption("If Yahoo fails during a demo, use 'Upload CSV' with these files.")

if result is None:
    for t in tabs[1:]:
        with t:
            st.info("Click **Train models and evaluate** to see results here.")
    st.stop()

with tabs[1]:
    params_str = ", ".join(f"{k}={v}" for k, v in result["best_params"][best].items())
    st.markdown(f"Selected by **{result['selection']}**: **{best}** ({params_str}).")
    order = metrics.sort_values("RMSE (%)", ascending=False)
    bar_colors = [GREY if x.startswith("Baseline") else (TEAL if v < zero_rmse else SLATE)
                  for x, v in zip(order["Model"], order["RMSE (%)"])]
    fig = go.Figure(go.Bar(
        x=order["RMSE (%)"], y=order["Model"], orientation="h", marker_color=bar_colors,
        text=[f"{v:.3f}%" for v in order["RMSE (%)"]], textposition="outside",
        customdata=order["DM p-value vs predict-0"].fillna(-1),
        hovertemplate="%{y}<br>RMSE %{x:.3f}%<br>DM p = %{customdata:.2f}<extra></extra>"))
    fig.add_vline(x=zero_rmse, line_dash="dash", line_color=CORAL,
                  annotation_text="guess 'no change'", annotation_position="top", annotation_font_color=CORAL)
    lo, hi = order["RMSE (%)"].min(), order["RMSE (%)"].max()
    fig.update_xaxes(range=[lo - (hi - lo) * 1.2 - 0.01, hi + (hi - lo) * 0.6 + 0.01])
    show(style(fig, "Test-set error by model (lower is better)", 360, xtitle="RMSE (%)"))
    st.caption("Teal = lower error than guessing 'no change'. The x-axis is zoomed in, so small gaps look larger than they are.")
    table(metrics, {"RMSE (%)": 3, "MAE (%)": 3, "Selection RMSE (%)": 3, "Direction accuracy (%)": 1,
                    "Skill vs predict-0 (%)": 1, "DM p-value vs predict-0": 2})
    with st.expander("What do these columns mean?"):
        st.markdown(
            "- **RMSE / MAE** — average size of the forecast error, in percentage points of return.\n"
            "- **Direction accuracy** — how often up/down was called correctly (50% = coin flip).\n"
            "- **Skill vs predict-0** — how much lower the squared error is than always guessing 0 (positive = better).\n"
            "- **DM p-value** — Diebold–Mariano test of whether the difference from predict-0 could be chance "
            "(below 0.05 = unlikely to be chance).\n"
            "- **Selection RMSE** — the score used to choose settings, before the test set was touched.")
    with st.expander("Hyperparameter tuning — every setting tried"):
        st.dataframe(result["tuning"], hide_index=True, width="stretch",
                     column_config=pct_cols(result["tuning"], ["Selection RMSE (%)"], "%.4f"))

with tabs[2]:
    if info["peers_used"]:
        lags = range(info["peer_lags"])
        order_t = sorted(info["peers_used"], key=lambda t: (list(ROLE_COLORS).index(role(t)), t))
        z = [[data[f"{t}_lag{k}"].corr(data["target"]) for k in lags] for t in order_t]
        noise = 2 / np.sqrt(len(data) / horizon)
        lim = max(noise * 1.5, float(np.nanmax(np.abs(z))))
        fig = go.Figure(go.Heatmap(
            z=z, x=["today" if k == 0 else f"{k} day{'s' if k > 1 else ''} ago" for k in lags],
            y=[f"{nice(t)}  ·  {role(t)}" for t in order_t], zmin=-lim, zmax=lim, zmid=0,
            colorscale=[[0, CORAL], [0.5, PANEL], [1, TEAL]], text=np.round(z, 3), texttemplate="%{text:.2f}",
            hovertemplate="%{y}<br>%{x}<br>correlation %{z:.3f}<extra></extra>", colorbar=dict(title="r")))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, f"Do supplier moves come before {target_name}'s? "
                        f"Correlation with its next {horizon}-day return", 80 + 42 * len(order_t)))
        st.caption(f"Each cell: correlation between a supplier's daily return (today, or a few days ago) and "
                   f"{target_name}'s return over the next {horizon} days, across the whole period. Every column "
                   "is information you would already have when making the forecast. Values within about "
                   f"±{noise:.2f} are indistinguishable from noise for this much data. Teal = supplier up → "
                   f"{target_name} tends to rise afterwards; coral = the opposite.")
    ab = result.get("ablation")
    if ab is None:
        st.info("This run used no peer stocks, so there is nothing to compare. Add supplier tickers to see their effect.")
    else:
        fig = go.Figure()
        fig.add_bar(name="Own features only", x=ab["Model"], y=ab["RMSE own only (%)"], marker_color=GREY,
                    hovertemplate="%{x}<br>%{y:.3f}%<extra>own only</extra>")
        fig.add_bar(name="Own + supplier features", x=ab["Model"], y=ab["RMSE own + peers (%)"], marker_color=TEAL,
                    hovertemplate="%{x}<br>%{y:.3f}%<extra>with suppliers</extra>")
        fig.add_scatter(x=ab["Model"], y=[zero_rmse] * len(ab), mode="lines", name=f"Guess 'no change' ({zero_rmse:.3f}%)",
                        line=dict(color=CORAL, dash="dash", width=2), hoverinfo="skip")
        vals = pd.concat([ab["RMSE own only (%)"], ab["RMSE own + peers (%)"]])
        span = vals.max() - min(vals.min(), zero_rmse)
        fig.update_yaxes(range=[min(vals.min(), zero_rmse) - span * 0.6 - 0.01, vals.max() + span * 0.3 + 0.01])
        fig.update_layout(barmode="group")
        show(style(fig, "Does adding supplier stocks lower the error?", 380, "Test RMSE (%)"))
        helped = ab.loc[ab["Peers help?"] == "yes", "Model"].tolist()
        st.markdown(f"Supplier data lowered the error for **{', '.join(helped) if helped else 'no model'}**. "
                    "Each model keeps the same settings with and without suppliers; only the features change.")
    imp, groups = result["feature_importance"], result["group_importance"]
    st.markdown(f"#### What drove the {best}'s predictions?")
    if imp.empty or imp["Importance"].sum() == 0:
        st.markdown(f'<div class="note">{best} gave every feature zero weight — it predicts the same value '
                    "(the historical average) every day. No feature carried a signal it could rely on.</div>",
                    unsafe_allow_html=True)
    else:
        a, b = st.columns(2)
        with a:
            g = groups[groups["Share (%)"] > 0].sort_values("Share (%)").copy()
            g["Source"] = [s_ if s_ == "Own price/volume" else nice(s_) for s_ in g["Source"]]
            fig = go.Figure(go.Bar(x=g["Share (%)"], y=g["Source"], orientation="h",
                                   marker_color=[AMBER if s == "Own price/volume" else TEAL for s in g["Source"]],
                                   hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            show(style(fig, "Importance by source", 340, xtitle="Share of importance (%)"))
        with b:
            t = imp.head(10).sort_values("Importance")
            fig = go.Figure(go.Bar(x=t["Importance"], y=t["Feature"], orientation="h", marker_color=AMBER,
                                   hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            show(style(fig, f"Top features ({imp['Method'].iloc[0]})", 340))

with tabs[3]:
    model_choices = [x for x in metrics["Model"] if x != ZERO]
    strategy_model = st.selectbox("Model driving the strategy", model_choices, index=model_choices.index(best))
    bt, summary = backtest(result["predictions"], strategy_model, threshold, costs)
    s, h = summary.iloc[0], summary.iloc[1]
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Strategy return", f"{s['Total return (%)']:+.1f}%", f"buy & hold {h['Total return (%)']:+.1f}%",
        "good" if s["Total return (%)"] > h["Total return (%)"] else "bad")
    kpi(k2, "Worst drop", f"{s['Max drawdown (%)']:.1f}%", f"buy & hold {h['Max drawdown (%)']:.1f}%")
    kpi(k3, "Sharpe ratio", f"{s['Sharpe ratio']:.2f}" if pd.notna(s["Sharpe ratio"]) else "—",
        f"buy & hold {h['Sharpe ratio']:.2f}")
    kpi(k4, "Trades", f"{int(s['Trades'])}", f"invested {s['Time in market (%)']:.0f}% of days")
    fig = go.Figure()
    fig.add_scatter(x=bt["Date"], y=bt["buy_hold_equity"], name="Buy and hold", line=dict(color=GREY, width=1.6),
                    hovertemplate="%{x|%d %b %Y}<br>₹%{y:.3f}<extra>buy & hold</extra>")
    fig.add_scatter(x=bt["Date"], y=bt["strategy_equity"], name=f"ML long/flat ({strategy_model})",
                    line=dict(color=TEAL, width=2.4),
                    hovertemplate="%{x|%d %b %Y}<br>₹%{y:.3f}<extra>strategy</extra>")
    fig.add_hline(y=1, line_color=BORDER)
    show(style(fig, "Value of ₹1 invested over the test period", 420, "Value (₹)"))
    st.caption(f"Hold the stock on a day only if yesterday's forecast was above {threshold:.2f}%; otherwise hold cash. "
               f"Every buy or sell costs {costs:.2f}%. Flat stretches of the teal line are days in cash. "
               "Move the sidebar sliders to see the effect instantly.")
    st.markdown("#### Threshold sensitivity")
    tt = threshold_table(result["predictions"], strategy_model, transaction_cost_pct=costs)
    table(tt, {"Total return (%)": 2, "Sharpe ratio": 2, "Max drawdown (%)": 2, "Time in market (%)": 1})
    st.markdown('<div class="note">A strategy that looks good at one threshold and bad at another is fragile. '
                "Treat a backtest win as evidence only if the forecasts behind it also beat guessing.</div>",
                unsafe_allow_html=True)

with tabs[4]:
    st.markdown("One test period can be lucky. These checks repeat the evaluation over several periods and horizons.")
    a, b = st.columns(2)
    if a.button("Run walk-forward evaluation (5 periods)", width="stretch"):
        with st.spinner("Running walk-forward…"):
            st.session_state["wf"] = walk_forward(data, features, horizon)
    if b.button("Compare horizons 1 / 5 / 20 days", width="stretch"):
        with st.spinner("Training at three horizons… (a minute or two on real data)"):
            st.session_state["hz"] = compare_horizons(frame, peers, (1, 5, 20), peer_lags, tune,
                                                      4 if use_cv else 0, target_mode, market)
    if st.session_state.get("wf") is not None:
        folds, wf_summary = st.session_state["wf"]
        fig = go.Figure()
        palette = {ZERO: CORAL, "Baseline: historical mean": GREY, "Elastic Net": SLATE,
                   "Decision Tree": AMBER, "Random Forest": TEAL, "Gradient Boosting": "#C9D6E8"}
        for name, g in folds.groupby("Model", sort=False):
            periods = g["Test period"].str.replace(r"(\d{4})-(\d{2})-\d{2}", r"\2/\1", regex=True)
            fig.add_scatter(x=periods, y=g["RMSE (%)"], name=name, mode="lines+markers",
                            line=dict(color=palette.get(name, GREY), width=3 if name == ZERO else 2,
                                      dash="dash" if name.startswith("Baseline") else "solid"),
                            hovertemplate="%{x}<br>%{y:.3f}%<extra>" + name + "</extra>")
        show(style(fig, "Walk-forward: test error in each period (month/year)", 420, "RMSE (%)"))
        table(wf_summary, {"Mean RMSE (%)": 3, "Worst fold RMSE (%)": 3, "Mean direction acc. (%)": 1})
    if st.session_state.get("hz") is not None:
        hz = st.session_state["hz"].set_index("Model")
        st.markdown("#### Test RMSE (%) by forecast horizon")
        st.caption("Green cells beat guessing 'no change' at that horizon.")
        styled = hz.style.format("{:.3f}").apply(
            lambda col: ["background-color: rgba(42,157,143,0.30); color: #8FE6D6; font-weight: 600"
                         if v < col[ZERO] and i != ZERO else "" for i, v in col.items()], axis=0)
        st.dataframe(styled, width="stretch")

with tabs[5]:
    view = result["predictions"].copy()
    view["Actual forward return (%)"] = view.pop("target") * 100
    model_cols = [c for c in view.columns if c not in ("Date", "Close", "Actual forward return (%)")]
    view = view[["Date", "Close", "Actual forward return (%)"] + model_cols]
    for c in model_cols:
        view[c] = view[c] * 100
    fig = go.Figure()
    fig.add_scatter(x=view["Date"], y=view["Actual forward return (%)"], name="Actual", line=dict(color=GREY, width=1.2))
    fig.add_scatter(x=view["Date"], y=view[best], name=f"Predicted ({best})", line=dict(color=TEAL, width=2.2))
    fig.add_hline(y=0, line_color=BORDER)
    show(style(fig, f"Actual vs predicted {horizon}-day return on the test period", 380, "Return (%)"))
    st.caption("Forecasts that hug zero while actual returns swing widely are what 'close to unpredictable' looks like.")
    st.dataframe(view.tail(100), hide_index=True, width="stretch",
                 column_config={"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                                **pct_cols(view, ["Close"], "%.2f"),
                                **pct_cols(view, ["Actual forward return (%)"] + model_cols, "%.3f")})
    a, b = st.columns(2)
    a.download_button("Download test predictions CSV", view.to_csv(index=False).encode("utf-8"),
                      "test_predictions.csv", "text/csv", width="stretch")
    b.download_button("Download strategy results CSV", bt0.to_csv(index=False).encode("utf-8"),
                      "strategy_results.csv", "text/csv", width="stretch")

st.divider()
st.caption("Research prototype only • No financial advice • Past performance does not guarantee future results")
