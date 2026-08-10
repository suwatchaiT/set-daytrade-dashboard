from __future__ import annotations

from datetime import datetime, time as clock_time
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="SET Intraday Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BANGKOK = ZoneInfo("Asia/Bangkok")
WATCHLIST = {
    "PTT.BK": "PTT",
    "KBANK.BK": "KBANK",
    "SCB.BK": "SCB",
    "AOT.BK": "AOT",
    "CPALL.BK": "CPALL",
    "ADVANC.BK": "ADVANC",
    "DELTA.BK": "DELTA",
    "GULF.BK": "GULF",
    "BBL.BK": "BBL",
    "KTC.BK": "KTC",
}
TICKERS = tuple(WATCHLIST)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#171d2b; border:1px solid #2b3548; padding:12px; border-radius:10px;}
    [data-testid="stMetricLabel"] {font-size:.78rem;}
    .signal-long {color:#33d17a; font-weight:700;}
    .signal-short {color:#ff6b6b; font-weight:700;}
    .signal-wait {color:#f7c948; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def safe_float(value, default=np.nan) -> float:
    try:
        value = float(value)
        return default if np.isnan(value) else value
    except (TypeError, ValueError):
        return default


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    out["EMA9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["RSI"] = rsi(out["Close"])
    out["ATR"] = atr(out)
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    session = pd.Series(out.index.date, index=out.index)
    cumulative_value = (typical * out["Volume"]).groupby(session).cumsum()
    cumulative_volume = out["Volume"].groupby(session).cumsum().replace(0, np.nan)
    out["VWAP"] = cumulative_value / cumulative_volume
    out["Prior20High"] = out["High"].shift(1).rolling(20).max()
    out["Prior20Low"] = out["Low"].shift(1).rolling(20).min()
    return out


def relative_volume(df: pd.DataFrame) -> float:
    """Latest 15-minute bar volume versus the previous 20 bars."""
    if len(df) < 6:
        return np.nan
    baseline = df["Volume"].iloc[-21:-1].replace(0, np.nan).median()
    if pd.isna(baseline) or baseline <= 0:
        return np.nan
    return safe_float(df["Volume"].iloc[-1] / baseline)


@st.cache_data(ttl=120, show_spinner=False)
def download_intraday(ticker: str, interval: str = "15m", period: str = "5d"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=12,
        )
        if data.empty:
            return None
        data = flatten_columns(data)
        data.index = pd.to_datetime(data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        data.index = data.index.tz_convert(BANGKOK)
        return data
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def download_set_index():
    # Yahoo's Thailand SET Composite symbol is ^SET.BK. Do not accept ^SET,
    # which can resolve to a different index.
    for symbol in ("^SET.BK", "SET.BK"):
        try:
            data = yf.download(
                symbol,
                period="1mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=12,
            )
            if not data.empty:
                return flatten_columns(data)
        except Exception:
            continue
    return None


def market_state(now: datetime) -> Tuple[str, str]:
    if now.weekday() >= 5:
        return "Closed", "#ff6b6b"
    current = now.time()
    morning = clock_time(10, 0) <= current <= clock_time(12, 30)
    afternoon = clock_time(14, 0) <= current <= clock_time(16, 30)
    preopen_1 = clock_time(9, 30) <= current < clock_time(10, 0)
    intermission = clock_time(12, 30) < current < clock_time(13, 30)
    preopen_2 = clock_time(13, 30) <= current < clock_time(14, 0)
    preclose = clock_time(16, 30) < current <= clock_time(16, 40)
    if morning or afternoon:
        return "Open", "#33d17a"
    if preopen_1 or preopen_2:
        return "Pre-open", "#f7c948"
    if intermission:
        return "Intermission", "#f7c948"
    if preclose:
        return "Pre-close", "#f7c948"
    return "Closed", "#ff6b6b"


def score_setup(symbol: str, raw: pd.DataFrame) -> Optional[Dict]:
    data = add_indicators(raw)
    if len(data) < 25:
        return None

    last = data.iloc[-1]
    previous = data.iloc[-2]
    price = safe_float(last["Close"])
    ema9 = safe_float(last["EMA9"])
    ema21 = safe_float(last["EMA21"])
    vwap = safe_float(last["VWAP"])
    rsi_value = safe_float(last["RSI"])
    atr_value = safe_float(last["ATR"], price * 0.01)
    rel_vol = relative_volume(data)
    prior_high = safe_float(last["Prior20High"])
    prior_low = safe_float(last["Prior20Low"])

    long_points = 0
    short_points = 0
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    if price > vwap:
        long_points += 2
        long_reasons.append("Above VWAP")
    elif price < vwap:
        short_points += 2
        short_reasons.append("Below VWAP")

    if ema9 > ema21 and price > ema9:
        long_points += 2
        long_reasons.append("EMA bullish")
    elif ema9 < ema21 and price < ema9:
        short_points += 2
        short_reasons.append("EMA bearish")

    if 52 <= rsi_value <= 68:
        long_points += 1
        long_reasons.append("RSI momentum")
    elif 32 <= rsi_value <= 48:
        short_points += 1
        short_reasons.append("RSI weakness")

    if not np.isnan(rel_vol):
        if rel_vol >= 2:
            long_points += 2
            short_points += 2
        elif rel_vol >= 1.3:
            long_points += 1
            short_points += 1

    breakout_up = not np.isnan(prior_high) and price >= prior_high * 0.998
    breakout_down = not np.isnan(prior_low) and price <= prior_low * 1.002
    if breakout_up:
        long_points += 2
        long_reasons.append("20-bar breakout")
    if breakout_down:
        short_points += 2
        short_reasons.append("20-bar breakdown")

    bar_change = (price / safe_float(previous["Close"]) - 1) * 100
    if bar_change >= 0.35:
        long_points += 1
        long_reasons.append("Fast momentum")
    elif bar_change <= -0.35:
        short_points += 1
        short_reasons.append("Fast momentum")

    if long_points >= short_points + 2 and long_points >= 5:
        direction = "WATCH LONG"
        score = long_points
        reasons = long_reasons
        trigger = max(price, safe_float(data["High"].iloc[-3:].max()))
        risk = max(atr_value * 0.8, price * 0.004)
        stop = trigger - risk
        target = trigger + risk * 1.8
    elif short_points >= long_points + 2 and short_points >= 5:
        direction = "WATCH SHORT"
        score = short_points
        reasons = short_reasons
        trigger = min(price, safe_float(data["Low"].iloc[-3:].min()))
        risk = max(atr_value * 0.8, price * 0.004)
        stop = trigger + risk
        target = trigger - risk * 1.8
    else:
        direction = "WAIT"
        score = max(long_points, short_points)
        reasons = long_reasons if long_points >= short_points else short_reasons
        trigger = np.nan
        stop = np.nan
        target = np.nan

    return {
        "Ticker": symbol,
        "Stock": WATCHLIST[symbol],
        "Direction": direction,
        "Score": min(score, 10),
        "Price": price,
        "15m %": bar_change,
        "Rel Volume": rel_vol,
        "RSI": rsi_value,
        "VWAP": vwap,
        "Trigger": trigger,
        "Stop": stop,
        "Target": target,
        "Setup": " · ".join(reasons[:3]) if reasons else "No confirmation",
        "Updated": data.index[-1],
    }


@st.cache_data(ttl=120, show_spinner=False)
def build_monitor(tickers: tuple[str, ...]):
    rows = []
    histories = {}
    for ticker in tickers:
        data = download_intraday(ticker)
        if data is None:
            continue
        result = score_setup(ticker, data)
        if result:
            rows.append(result)
            histories[ticker] = data
    frame = pd.DataFrame(rows)
    if not frame.empty:
        direction_order = {"WATCH LONG": 0, "WATCH SHORT": 1, "WAIT": 2}
        frame["_direction_order"] = frame["Direction"].map(direction_order)
        frame = frame.sort_values(["_direction_order", "Score", "Rel Volume"], ascending=[True, False, False])
        frame = frame.drop(columns="_direction_order").reset_index(drop=True)
        frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    return frame, histories


def direction_style(value: str) -> str:
    if value == "WATCH LONG":
        return "color:#33d17a;font-weight:700"
    if value == "WATCH SHORT":
        return "color:#ff6b6b;font-weight:700"
    return "color:#f7c948;font-weight:700"


def score_style(value) -> str:
    score = safe_float(value, 0)
    if score >= 7:
        return "background-color:#9f1239;color:white;font-weight:700"
    if score >= 5:
        return "background-color:#c2410c;color:white;font-weight:700"
    return "background-color:#854d0e;color:white"


def fmt_number(value, decimals=2, suffix=""):
    value = safe_float(value)
    return "—" if np.isnan(value) else f"{value:,.{decimals}f}{suffix}"


now = datetime.now(BANGKOK)
state, state_color = market_state(now)
index_data = download_set_index()

index_close = index_high = index_low = index_delta = np.nan
if index_data is not None and not index_data.empty:
    index_close = safe_float(index_data["Close"].iloc[-1])
    index_high = safe_float(index_data["High"].iloc[-1])
    index_low = safe_float(index_data["Low"].iloc[-1])
    if len(index_data) > 1:
        previous_close = safe_float(index_data["Close"].iloc[-2])
        if previous_close:
            index_delta = (index_close / previous_close - 1) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("SET Index", fmt_number(index_close), None if np.isnan(index_delta) else f"{index_delta:+.2f}%")
m2.metric("Today High", fmt_number(index_high))
m3.metric("Today Low", fmt_number(index_low))
m4.markdown(
    f"<div style='padding:8px 0'><small>MARKET STATUS</small><br>"
    f"<span style='font-size:1.8rem;color:{state_color};font-weight:700'>{state}</span></div>",
    unsafe_allow_html=True,
)

header, refresh_col = st.columns([4, 1])
with header:
    st.title("SET Intraday Opportunity Monitor")
    st.caption(f"15-minute signals · Updated {now:%d %b %Y, %H:%M} ICT · Data: Yahoo Finance (may be delayed)")
with refresh_col:
    auto_refresh = st.toggle("Auto-refresh", value=False)
    if st.button("↻ Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    st_autorefresh(interval=120_000, key="market-refresh")

with st.spinner("Scanning 10 liquid SET stocks…"):
    monitor, histories = build_monitor(TICKERS)

if monitor.empty:
    st.error("Market data is temporarily unavailable. Try Refresh again in a minute.")
    st.stop()

actionable = monitor[monitor["Direction"] != "WAIT"]
best = actionable.iloc[0] if not actionable.empty else monitor.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stocks scanned", len(monitor))
k2.metric("Actionable setups", len(actionable))
k3.metric("Highest score", f"{int(best['Score'])}/10", best["Stock"])
k4.metric("Best relative volume", fmt_number(monitor["Rel Volume"].max(), 1, "×"))

st.subheader("Stocks to monitor now")
display_columns = [
    "Rank", "Stock", "Direction", "Score", "Price", "15m %", "Rel Volume",
    "RSI", "Trigger", "Stop", "Target", "Setup",
]
styled = (
    monitor[display_columns]
    .style.map(direction_style, subset=["Direction"])
    .map(score_style, subset=["Score"])
    .format(
        {
            "Price": "{:.2f}",
            "15m %": "{:+.2f}%",
            "Rel Volume": "{:.1f}×",
            "RSI": "{:.0f}",
            "Trigger": "{:.2f}",
            "Stop": "{:.2f}",
            "Target": "{:.2f}",
        },
        na_rep="—",
    )
)
st.dataframe(styled, width="stretch", hide_index=True, height=425)
st.caption("A setup is actionable only after price crosses its Trigger. Stop and Target are volatility-based planning levels, not guaranteed prices.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Signal strength")
    signal_chart = px.bar(
        monitor.sort_values("Score"),
        x="Score",
        y="Stock",
        orientation="h",
        color="Direction",
        color_discrete_map={"WATCH LONG": "#33d17a", "WATCH SHORT": "#ff6b6b", "WAIT": "#f7c948"},
        hover_data=["Setup", "Rel Volume", "RSI"],
        template="plotly_dark",
    )
    signal_chart.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 10])
    st.plotly_chart(signal_chart)

with right:
    st.subheader("Volume versus momentum")
    scatter = px.scatter(
        monitor,
        x="Rel Volume",
        y="15m %",
        text="Stock",
        color="Direction",
        size="Score",
        color_discrete_map={"WATCH LONG": "#33d17a", "WATCH SHORT": "#ff6b6b", "WAIT": "#f7c948"},
        template="plotly_dark",
    )
    scatter.add_vline(x=1.3, line_dash="dash", line_color="#ff9f43")
    scatter.add_hline(y=0, line_dash="dot", line_color="#718096")
    scatter.update_traces(textposition="top center")
    scatter.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(scatter)

st.divider()
st.subheader("Intraday chart")
selected_name = st.selectbox("Stock", monitor["Stock"].tolist(), index=0)
selected_ticker = monitor.loc[monitor["Stock"] == selected_name, "Ticker"].iloc[0]
chart_data = add_indicators(histories[selected_ticker])

figure = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.58, 0.22, 0.20],
    vertical_spacing=0.04,
)
figure.add_trace(
    go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"], high=chart_data["High"],
        low=chart_data["Low"], close=chart_data["Close"],
        increasing_line_color="#33d17a", decreasing_line_color="#ff6b6b", name="Price",
    ),
    row=1, col=1,
)
for column, colour in (("EMA9", "#f7c948"), ("EMA21", "#a78bfa"), ("VWAP", "#38bdf8")):
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], name=column, line=dict(color=colour, width=1.3)), row=1, col=1)
figure.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color="#64748b"), row=2, col=1)
figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["RSI"], name="RSI", line=dict(color="#38bdf8")), row=3, col=1)
figure.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", row=3, col=1)
figure.add_hline(y=30, line_dash="dash", line_color="#33d17a", row=3, col=1)
figure.update_layout(
    height=760,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.03),
    margin=dict(t=35, b=10),
)
st.plotly_chart(figure)

with st.expander("How the monitor ranks stocks"):
    st.markdown(
        """
        The monitor gives separate long and short points. It looks for price relative to VWAP,
        EMA9/EMA21 trend alignment, RSI momentum, unusual 15-minute volume, a 20-bar breakout
        or breakdown, and fast price movement. A stock becomes **WATCH LONG** or **WATCH SHORT**
        only when one direction scores at least 5 and clearly exceeds the opposite direction.

        This dashboard is a screening and monitoring tool. Yahoo Finance data can be delayed and
        should not be used as the sole source for order execution or investment decisions.
        """
    )
