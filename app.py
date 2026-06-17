import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

st.set_page_config(
    page_title="SET Dashboard — Day Trade Scanner",
    page_icon="📈",
    layout="wide",
)

# ─── Stocks (top 10 most liquid SET) ─────────────────────────────────────────

SET_WATCHLIST = {
    "PTT.BK":    "PTT",
    "KBANK.BK":  "KBANK",
    "SCB.BK":    "SCB",
    "AOT.BK":    "AOT",
    "CPALL.BK":  "CPALL",
    "ADVANC.BK": "ADVANC",
    "DELTA.BK":  "DELTA",
    "GULF.BK":   "GULF",
    "BBL.BK":    "BBL",
    "KTC.BK":    "KTC",
}

TICKERS = tuple(SET_WATCHLIST.keys())

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def volume_ratio(vol: pd.Series, window: int = 20) -> float:
    if len(vol) < window + 1:
        return 1.0
    avg = vol.iloc[-(window + 1):-1].mean()
    return round(float(vol.iloc[-1]) / avg, 2) if avg > 0 else 1.0


def safe_float(val) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except Exception:
        return np.nan


@st.cache_data(ttl=300)
def fetch_ticker_data(ticker: str, period: str = "5d", interval: str = "15m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = _flatten_cols(df)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_snapshot(tickers: tuple):
    rows = []
    for sym in tickers:
        try:
            df = yf.download(sym, period="60d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 2:
                continue
            df = _flatten_cols(df)
            today = df.iloc[-1]
            prev  = df.iloc[-2]
            close = safe_float(today["Close"])
            prev_close = safe_float(prev["Close"])
            if np.isnan(close) or np.isnan(prev_close) or prev_close == 0:
                continue
            chg = close - prev_close
            pct = chg / prev_close * 100
            vr  = volume_ratio(df["Volume"])
            rsi_s = compute_rsi(df["Close"])
            rsi   = safe_float(rsi_s.iloc[-1]) if len(rsi_s) else np.nan
            rows.append({
                "Ticker":    sym,
                "Name":      SET_WATCHLIST.get(sym, sym),
                "Price":     round(close, 2),
                "Chg":       round(chg, 2),
                "%Chg":      round(pct, 2),
                "Volume":    int(safe_float(today["Volume"]) or 0),
                "Vol Ratio": vr,
                "RSI":       round(rsi, 1) if not np.isnan(rsi) else np.nan,
                "Open":      round(safe_float(today["Open"]), 2),
                "High":      round(safe_float(today["High"]), 2),
                "Low":       round(safe_float(today["Low"]),  2),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def fetch_set_index():
    try:
        df = yf.download("^SET", period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        return _flatten_cols(df)
    except Exception:
        return None


# ─── Style helpers ────────────────────────────────────────────────────────────

def colour_pct(val):
    try:
        v = float(val)
        if np.isnan(v): return ""
        if v > 0: return "color: #26a69a; font-weight:bold"
        if v < 0: return "color: #ef5350; font-weight:bold"
    except Exception:
        pass
    return ""


def colour_rsi(val):
    try:
        v = float(val)
        if np.isnan(v): return ""
        if v >= 70: return "color: #ef5350; font-weight:bold"
        if v <= 30: return "color: #26a69a; font-weight:bold"
    except Exception:
        pass
    return ""


def colour_vr(val):
    try:
        if float(val) >= 2.0:
            return "color: #ff9800; font-weight:bold"
    except Exception:
        pass
    return ""


def _fmt_rsi(x):
    try:
        v = float(x)
        return "—" if np.isnan(v) else f"{v:.1f}"
    except Exception:
        return "—"


# ─── Layout ───────────────────────────────────────────────────────────────────

st.title("📈 SET Day-Trade Dashboard")
st.caption(f"Thailand Stock Exchange  •  Data via Yahoo Finance  •  {datetime.now().strftime('%H:%M:%S ICT')}")

col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
with col_ctrl1:
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
with col_ctrl2:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# ─── SET Index ────────────────────────────────────────────────────────────────

set_df = fetch_set_index()
if set_df is not None and len(set_df) >= 2:
    latest  = set_df.iloc[-1]
    prev_r  = set_df.iloc[-2]
    idx_chg = safe_float(latest["Close"]) - safe_float(prev_r["Close"])
    idx_pct = idx_chg / safe_float(prev_r["Close"]) * 100
    arrow   = "▲" if idx_chg >= 0 else "▼"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SET Index",   f"{safe_float(latest['Close']):,.2f}", f"{arrow} {idx_chg:+.2f} ({idx_pct:+.2f}%)")
    m2.metric("Today High",  f"{safe_float(latest['High']):,.2f}")
    m3.metric("Today Low",   f"{safe_float(latest['Low']):,.2f}")
    m4.metric("Volume (M)",  f"{safe_float(latest['Volume'])/1e6:,.1f}")

st.divider()

# ─── Fetch data ───────────────────────────────────────────────────────────────

with st.spinner("Loading market data…"):
    snap = fetch_snapshot(TICKERS)

if snap.empty:
    st.error("Could not fetch data. Check internet connection.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview", "🚀 Opportunity Scanner", "🔥 Volume Alert", "📉 Chart Viewer"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Top 10 SET Stocks — Today")

    styled = (
        snap[["Name", "Price", "Chg", "%Chg", "Open", "High", "Low", "Volume", "Vol Ratio", "RSI"]]
        .sort_values("%Chg", ascending=False)
        .style
        .map(colour_pct, subset=["%Chg", "Chg"])
        .map(colour_rsi, subset=["RSI"])
        .map(colour_vr,  subset=["Vol Ratio"])
        .format({
            "Price": "{:.2f}", "Chg": "{:+.2f}", "%Chg": "{:+.2f}%",
            "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
            "Volume": "{:,.0f}", "Vol Ratio": "{:.2f}x", "RSI": _fmt_rsi,
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        g = snap.nlargest(5, "%Chg")[["Name", "%Chg"]]
        fig = px.bar(g, x="Name", y="%Chg", color="%Chg",
                     color_continuous_scale=["#ef5350", "#26a69a"],
                     title="Top 5 Gainers (%)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        l = snap.nsmallest(5, "%Chg")[["Name", "%Chg"]]
        fig2 = px.bar(l, x="Name", y="%Chg", color="%Chg",
                      color_continuous_scale=["#ef5350", "#26a69a"],
                      title="Top 5 Losers (%)", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🚀 Day Trade Opportunity Scanner")
    st.caption("Scores each stock by RSI + Volume surge + Momentum")

    df_opp = snap.copy()
    df_opp["Signal"] = "Neutral"
    df_opp["Score"]  = 0

    for i, row in df_opp.iterrows():
        score, signals = 0, []
        rsi = row["RSI"]
        vr  = row["Vol Ratio"]
        pct = row["%Chg"]

        rsi_ok = isinstance(rsi, (int, float)) and not np.isnan(float(rsi))
        if rsi_ok:
            if rsi <= 35:
                score += 3; signals.append("RSI oversold")
            elif rsi >= 65:
                score += 2; signals.append("RSI overbought")
        if vr >= 2.0:
            score += 3; signals.append(f"Vol {vr:.1f}x")
        elif vr >= 1.5:
            score += 1; signals.append(f"Vol {vr:.1f}x")
        if pct >= 2.0:
            score += 2; signals.append("Strong up")
        elif pct <= -2.0:
            score += 2; signals.append("Strong down")

        df_opp.at[i, "Score"]  = score
        df_opp.at[i, "Signal"] = " | ".join(signals) if signals else "Neutral"

    df_opp = df_opp[df_opp["Score"] > 0].sort_values("Score", ascending=False)

    if df_opp.empty:
        st.info("No strong signals right now. Market may be calm.")
    else:
        st.dataframe(
            df_opp[["Name", "Price", "%Chg", "RSI", "Vol Ratio", "Score", "Signal"]]
            .style
            .map(colour_pct, subset=["%Chg"])
            .map(colour_rsi, subset=["RSI"])
            .background_gradient(subset=["Score"], cmap="YlOrRd")
            .format({
                "Price": "{:.2f}", "%Chg": "{:+.2f}%",
                "RSI": _fmt_rsi, "Vol Ratio": "{:.2f}x",
            }, na_rep="—"),
            use_container_width=True,
        )

    st.subheader("RSI vs % Change Map")
    fig_sc = px.scatter(
        snap.dropna(subset=["RSI"]),
        x="RSI", y="%Chg", text="Name", color="%Chg",
        color_continuous_scale="RdYlGn", size="Volume", size_max=30,
        template="plotly_dark",
        labels={"RSI": "RSI (14)", "%Chg": "Daily % Change"},
    )
    fig_sc.add_vline(x=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
    fig_sc.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_sc.add_hline(y=0,  line_dash="dot",  line_color="gray")
    fig_sc.update_traces(textposition="top center", textfont_size=10)
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔥 Volume Surge Alert")
    st.caption("Stocks trading above their 20-day average volume")

    vol_df = snap[snap["Vol Ratio"] >= 1.3].sort_values("Vol Ratio", ascending=False)

    if vol_df.empty:
        st.info("No unusual volume detected today.")
    else:
        fig_vol = px.bar(
            vol_df, x="Name", y="Vol Ratio",
            color="Vol Ratio", color_continuous_scale="Oranges",
            text="Vol Ratio", title="Volume vs 20-Day Average",
            template="plotly_dark",
        )
        fig_vol.update_traces(texttemplate="%{text:.1f}x", textposition="outside")
        fig_vol.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="2× surge")
        st.plotly_chart(fig_vol, use_container_width=True)

        st.dataframe(
            vol_df[["Name", "Price", "%Chg", "Volume", "Vol Ratio", "RSI"]]
            .style
            .map(colour_pct, subset=["%Chg"])
            .map(colour_vr,  subset=["Vol Ratio"])
            .format({
                "Price": "{:.2f}", "%Chg": "{:+.2f}%",
                "Volume": "{:,.0f}", "Vol Ratio": "{:.2f}x", "RSI": _fmt_rsi,
            }, na_rep="—"),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📉 Intraday Chart Viewer")

    col_a, _ = st.columns([1, 2])
    with col_a:
        sel_name = st.selectbox("Select Stock", options=list(SET_WATCHLIST.values()))
        sel_tick = [k for k, v in SET_WATCHLIST.items() if v == sel_name][0]
        interval = st.radio("Interval", ["15m", "30m", "1h", "1d"], horizontal=True)
        period   = "5d" if interval in ["15m", "30m", "1h"] else "3mo"

    with st.spinner(f"Loading {sel_tick}…"):
        df_chart = fetch_ticker_data(sel_tick, period=period, interval=interval)

    if df_chart is None or df_chart.empty:
        st.error(f"No data for {sel_tick}")
    else:
        df_chart["RSI"]         = compute_rsi(df_chart["Close"])
        df_chart["MACD"], df_chart["Signal_line"] = compute_macd(df_chart["Close"])
        df_chart["EMA9"]        = df_chart["Close"].ewm(span=9,  adjust=False).mean()
        df_chart["EMA21"]       = df_chart["Close"].ewm(span=21, adjust=False).mean()

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=[f"{sel_name} — Candle + EMA", "RSI (14)", "MACD"],
        )

        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"],   close=df_chart["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["EMA9"],
                                  name="EMA9",  line=dict(color="#f9a825", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["EMA21"],
                                  name="EMA21", line=dict(color="#ab47bc", width=1.2)), row=1, col=1)

        bar_colours = ["#26a69a" if c >= o else "#ef5350"
                       for c, o in zip(df_chart["Close"], df_chart["Open"])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart["Volume"],
                              marker_color=bar_colours, opacity=0.35, name="Volume"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["RSI"],
                                  name="RSI", line=dict(color="#29b6f6", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        hist = df_chart["MACD"] - df_chart["Signal_line"]
        fig.add_trace(go.Bar(x=df_chart.index, y=hist,
                              marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in hist],
                              opacity=0.7, name="Histogram"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["MACD"],
                                  name="MACD",   line=dict(color="#f06292", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["Signal_line"],
                                  name="Signal", line=dict(color="#ffcc02", width=1.5)), row=3, col=1)

        fig.update_layout(
            height=800, xaxis_rangeslider_visible=False,
            template="plotly_dark",
            legend=dict(orientation="h", y=1.02),
            margin=dict(t=40, b=10),
        )
        fig.update_yaxes(title_text="THB",  row=1, col=1)
        fig.update_yaxes(title_text="RSI",  row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        last = df_chart.iloc[-1]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Last Close", f"{safe_float(last['Close']):.2f}")
        r2.metric("RSI",  _fmt_rsi(last["RSI"]))
        r3.metric("EMA9",  f"{safe_float(last['EMA9']):.2f}")
        r4.metric("EMA21", f"{safe_float(last['EMA21']):.2f}")

# ─── Auto-refresh ─────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
