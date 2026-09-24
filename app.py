from __future__ import annotations

from datetime import datetime, time as clock_time
from pathlib import Path
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
    # SET50 constituents for Jul 1–Dec 31, 2026.
    "ADVANC.BK": "ADVANC",
    "AOT.BK": "AOT",
    "AWC.BK": "AWC",
    "BANPU.BK": "BANPU",
    "BBL.BK": "BBL",
    "BCP.BK": "BCP",
    "BDMS.BK": "BDMS",
    "BEM.BK": "BEM",
    "BH.BK": "BH",
    "BJC.BK": "BJC",
    "CCET.BK": "CCET",
    "COM7.BK": "COM7",
    "CPALL.BK": "CPALL",
    "CPF.BK": "CPF",
    "CPN.BK": "CPN",
    "CRC.BK": "CRC",
    "DELTA.BK": "DELTA",
    "EGCO.BK": "EGCO",
    "GPSC.BK": "GPSC",
    "GULF.BK": "GULF",
    "HMPRO.BK": "HMPRO",
    "IVL.BK": "IVL",
    "KBANK.BK": "KBANK",
    "KKP.BK": "KKP",
    "KTB.BK": "KTB",
    "KTC.BK": "KTC",
    "LH.BK": "LH",
    "MINT.BK": "MINT",
    "MRDIYT.BK": "MRDIYT",
    "MTC.BK": "MTC",
    "OR.BK": "OR",
    "OSP.BK": "OSP",
    "PTT.BK": "PTT",
    "PTTEP.BK": "PTTEP",
    "PTTGC.BK": "PTTGC",
    "RATCH.BK": "RATCH",
    "SCB.BK": "SCB",
    "SCC.BK": "SCC",
    "SCGP.BK": "SCGP",
    "TCAP.BK": "TCAP",
    "TFG.BK": "TFG",
    "THAI.BK": "THAI",
    "TIDLOR.BK": "TIDLOR",
    "TISCO.BK": "TISCO",
    "TLI.BK": "TLI",
    "TOP.BK": "TOP",
    "TRUE.BK": "TRUE",
    "TTB.BK": "TTB",
    "TU.BK": "TU",
    "WHA.BK": "WHA",
}
TICKERS = tuple(WATCHLIST)
SIGNAL_LOG_PATH = Path(__file__).with_name("signal_log.csv")
LOG_COLUMNS = [
    "Logged At", "Bar Time", "Stock", "Ticker", "Signal", "Score", "Price",
    "Trigger", "Stop", "Target", "Daily", "1H", "15m Setup", "5m Confirm",
    "Rel Volume", "RSI", "Setup",
]

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


def completed_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Remove the currently forming intraday candle."""
    if df.empty:
        return df
    last_start = pd.Timestamp(df.index[-1])
    now = pd.Timestamp.now(tz=BANGKOK)
    if last_start.tzinfo is None:
        last_start = last_start.tz_localize(BANGKOK)
    if last_start + pd.Timedelta(minutes=minutes) > now:
        return df.iloc[:-1]
    return df


def normalize_price_frame(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None
    data = flatten_columns(data).dropna(how="all")
    if data.empty:
        return None
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert(BANGKOK)
    return data


@st.cache_data(ttl=120, show_spinner=False)
def download_universe(tickers: tuple[str, ...], interval: str, period: str):
    """Fetch the whole SET50 in one Yahoo request for a timeframe."""
    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
            timeout=20,
        )
        if raw.empty:
            return {}
        output = {}
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(raw.columns.get_level_values(0))
            level1 = set(raw.columns.get_level_values(1))
            for ticker in tickers:
                if ticker in level0:
                    frame = raw[ticker].copy()
                elif ticker in level1:
                    frame = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
                frame = normalize_price_frame(frame)
                if frame is not None:
                    output[ticker] = frame
        elif len(tickers) == 1:
            frame = normalize_price_frame(raw)
            if frame is not None:
                output[tickers[0]] = frame
        return output
    except Exception:
        return {}


def timeframe_bias(raw: pd.DataFrame, timeframe: str) -> str:
    source = completed_bars(raw, 60) if timeframe == "1H" else raw
    data = add_indicators(source)
    if len(data) < 22:
        return "NEUTRAL"
    last = data.iloc[-1]
    close = safe_float(last["Close"])
    if timeframe == "1D":
        ema20 = data["Close"].ewm(span=20, adjust=False).mean()
        slope = safe_float(ema20.iloc[-1] - ema20.iloc[-4])
        if close > safe_float(ema20.iloc[-1]) and slope > 0:
            return "BULLISH"
        if close < safe_float(ema20.iloc[-1]) and slope < 0:
            return "BEARISH"
        return "NEUTRAL"
    ema9 = safe_float(last["EMA9"])
    ema21 = safe_float(last["EMA21"])
    rsi_value = safe_float(last["RSI"])
    if close > ema9 > ema21 and rsi_value >= 50:
        return "BULLISH"
    if close < ema9 < ema21 and rsi_value <= 50:
        return "BEARISH"
    return "NEUTRAL"


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
    data = add_indicators(completed_bars(raw, 15))
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


def five_minute_confirmation(raw: pd.DataFrame, direction: str, trigger: float) -> Tuple[str, bool]:
    data = add_indicators(completed_bars(raw, 5))
    if len(data) < 22 or np.isnan(trigger):
        return "WAIT", False
    last, previous = data.iloc[-1], data.iloc[-2]
    close = safe_float(last["Close"])
    previous_close = safe_float(previous["Close"])
    open_price = safe_float(last["Open"])
    rel_vol = relative_volume(data)
    volume_ok = not np.isnan(rel_vol) and rel_vol >= 1.2
    if direction == "WATCH LONG":
        crossed = previous_close < trigger <= close
        if crossed and close > open_price and volume_ok:
            return "BREAKOUT", True
        return ("ABOVE" if close >= trigger else "BELOW"), False
    if direction == "WATCH SHORT":
        crossed = previous_close > trigger >= close
        if crossed and close < open_price and volume_ok:
            return "BREAKDOWN", True
        return ("BELOW" if close <= trigger else "ABOVE"), False
    return "WAIT", False


def final_signal(setup: str, daily: str, hourly: str, confirmed: bool) -> str:
    if setup == "WATCH LONG":
        if daily == "BEARISH" or hourly == "BEARISH":
            return "CONFLICT"
        if daily == hourly == "BULLISH":
            return "BUY NOW" if confirmed else "ARMED LONG"
        return "WATCH LONG"
    if setup == "WATCH SHORT":
        if daily == "BULLISH" or hourly == "BULLISH":
            return "CONFLICT"
        if daily == hourly == "BEARISH":
            return "SELL NOW" if confirmed else "ARMED SHORT"
        return "WATCH SHORT"
    return "WAIT"


@st.cache_data(ttl=120, show_spinner=False)
def build_monitor(tickers: tuple[str, ...]):
    rows, histories = [], {}
    daily_map = download_universe(tickers, "1d", "6mo")
    hourly_map = download_universe(tickers, "1h", "1mo")
    setup_map = download_universe(tickers, "15m", "5d")
    trigger_map = download_universe(tickers, "5m", "5d")
    for ticker in tickers:
        daily = daily_map.get(ticker)
        hourly = hourly_map.get(ticker)
        setup_data = setup_map.get(ticker)
        trigger_data = trigger_map.get(ticker)
        if any(item is None for item in (daily, hourly, setup_data, trigger_data)):
            continue
        result = score_setup(ticker, setup_data)
        if not result:
            continue
        daily_bias = timeframe_bias(daily, "1D")
        hourly_bias = timeframe_bias(hourly, "1H")
        confirm_text, confirmed = five_minute_confirmation(
            trigger_data, result["Direction"], safe_float(result["Trigger"])
        )
        result.update({
            "15m Setup": result["Direction"], "Daily": daily_bias, "1H": hourly_bias,
            "5m Confirm": confirm_text,
            "Signal": final_signal(result["Direction"], daily_bias, hourly_bias, confirmed),
            "Bar Time": completed_bars(trigger_data, 5).index[-1],
        })
        rows.append(result)
        histories[ticker] = setup_data
    frame = pd.DataFrame(rows)
    if not frame.empty:
        order = {
            "BUY NOW": 0, "SELL NOW": 0, "ARMED LONG": 1, "ARMED SHORT": 1,
            "WATCH LONG": 2, "WATCH SHORT": 2, "CONFLICT": 3, "WAIT": 4,
        }
        frame["_order"] = frame["Signal"].map(order)
        frame = frame.sort_values(["_order", "Score", "Rel Volume"], ascending=[True, False, False])
        frame = frame.drop(columns="_order").reset_index(drop=True)
        frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    return frame, histories


def load_signal_log() -> pd.DataFrame:
    if not SIGNAL_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        return pd.read_csv(SIGNAL_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def record_signals(frame: pd.DataFrame, logged_at: datetime) -> pd.DataFrame:
    existing = load_signal_log()
    loggable = frame[frame["Signal"].isin(["ARMED LONG", "ARMED SHORT", "BUY NOW", "SELL NOW"])]
    existing_keys = set()
    if not existing.empty:
        existing_keys = set(existing["Ticker"].astype(str) + "|" + existing["Bar Time"].astype(str) + "|" + existing["Signal"].astype(str))
    records = []
    for _, row in loggable.iterrows():
        bar_time = pd.Timestamp(row["Bar Time"]).isoformat()
        if f"{row['Ticker']}|{bar_time}|{row['Signal']}" in existing_keys:
            continue
        records.append({
            "Logged At": logged_at.isoformat(), "Bar Time": bar_time, "Stock": row["Stock"],
            "Ticker": row["Ticker"], "Signal": row["Signal"], "Score": row["Score"],
            "Price": row["Price"], "Trigger": row["Trigger"], "Stop": row["Stop"],
            "Target": row["Target"], "Daily": row["Daily"], "1H": row["1H"],
            "15m Setup": row["15m Setup"], "5m Confirm": row["5m Confirm"],
            "Rel Volume": row["Rel Volume"], "RSI": row["RSI"], "Setup": row["Setup"],
        })
    if records:
        new_records = pd.DataFrame(records, columns=LOG_COLUMNS)
        existing = new_records if existing.empty else pd.concat([existing, new_records], ignore_index=True)
        try:
            existing.to_csv(SIGNAL_LOG_PATH, index=False)
        except OSError:
            pass
    return existing


def direction_style(value: str) -> str:
    if value in ("BUY NOW", "ARMED LONG", "WATCH LONG", "BULLISH"):
        return "color:#33d17a;font-weight:700"
    if value in ("SELL NOW", "ARMED SHORT", "WATCH SHORT", "BEARISH"):
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
    st.title("SET Multi-Timeframe Opportunity Monitor")
    st.caption(f"1D regime · 1H trend · 15m setup · 5m trigger · Updated {now:%d %b %Y, %H:%M} ICT")
with refresh_col:
    auto_refresh = st.toggle("Auto-refresh", value=False)
    if st.button("↻ Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    st_autorefresh(interval=120_000, key="market-refresh")

with st.spinner("Scanning all 50 SET50 constituents across four timeframes…"):
    monitor, histories = build_monitor(TICKERS)

if monitor.empty:
    st.error("Market data is temporarily unavailable. Try Refresh again in a minute.")
    st.stop()

missing = sorted(set(WATCHLIST.values()) - set(monitor["Stock"]))
if missing:
    st.warning(f"Yahoo data unavailable for {len(missing)} constituent(s): {', '.join(missing)}")

signal_log = record_signals(monitor, now)
actionable = monitor[monitor["Signal"].isin(["BUY NOW", "SELL NOW", "ARMED LONG", "ARMED SHORT"])]
best = actionable.iloc[0] if not actionable.empty else monitor.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stocks scanned", len(monitor))
k2.metric("Armed / triggered", len(actionable))
k3.metric("Highest score", f"{int(best['Score'])}/10", best["Stock"])
k4.metric("Best relative volume", fmt_number(monitor["Rel Volume"].max(), 1, "×"))

st.subheader("Stocks to monitor now")
display_columns = [
    "Rank", "Stock", "Signal", "Daily", "1H", "15m Setup", "5m Confirm",
    "Score", "Price", "Rel Volume", "Trigger", "Stop", "Target", "Setup",
]
styled = (
    monitor[display_columns]
    .style.map(direction_style, subset=["Signal", "Daily", "1H", "15m Setup"])
    .map(score_style, subset=["Score"])
    .format(
        {
            "Price": "{:.2f}",
            "Rel Volume": "{:.1f}×",
            "Trigger": "{:.2f}",
            "Stop": "{:.2f}",
            "Target": "{:.2f}",
        },
        na_rep="—",
    )
)
st.dataframe(styled, width="stretch", hide_index=True, height=700)
st.caption("BUY/SELL NOW requires aligned 1D + 1H direction and a volume-confirmed 5-minute trigger cross. ARMED means aligned but not yet triggered.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Signal strength")
    signal_chart = px.bar(
        monitor.nlargest(20, ["Score", "Rel Volume"]).sort_values("Score"),
        x="Score",
        y="Stock",
        orientation="h",
        color="Signal",
        color_discrete_map={
            "BUY NOW": "#00e676", "ARMED LONG": "#33d17a", "WATCH LONG": "#66bb6a",
            "SELL NOW": "#ff1744", "ARMED SHORT": "#ff6b6b", "WATCH SHORT": "#ef5350",
            "CONFLICT": "#ab47bc", "WAIT": "#f7c948",
        },
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
        color="Signal",
        size="Score",
        color_discrete_map={
            "BUY NOW": "#00e676", "ARMED LONG": "#33d17a", "WATCH LONG": "#66bb6a",
            "SELL NOW": "#ff1744", "ARMED SHORT": "#ff6b6b", "WATCH SHORT": "#ef5350",
            "CONFLICT": "#ab47bc", "WAIT": "#f7c948",
        },
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

st.divider()
st.subheader("Signal journal")
if signal_log.empty:
    st.info("No aligned signals have been logged yet.")
else:
    st.dataframe(signal_log.tail(100).iloc[::-1], width="stretch", hide_index=True, height=320)
    st.download_button(
        "Download signal log (CSV)",
        signal_log.to_csv(index=False).encode("utf-8"),
        file_name="set_signal_log.csv",
        mime="text/csv",
    )
st.caption("Community Cloud storage is temporary: download the CSV regularly. A database is required for permanent history across restarts and redeployments.")

with st.expander("How the monitor ranks stocks"):
    st.markdown(
        """
        **1D** defines the broad regime from price versus EMA20 and its slope. **1H** confirms
        direction using EMA9/EMA21 and RSI. **15m** scores VWAP, trend, RSI, relative volume,
        breakout proximity and momentum. **5m** issues BUY/SELL NOW only after price crosses the
        trigger with a directionally correct candle and at least 1.2× relative volume.

        This dashboard is a screening and monitoring tool. Yahoo Finance data can be delayed and
        should not be used as the sole source for order execution or investment decisions.
        """
    )
