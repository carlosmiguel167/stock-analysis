import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Technical Analysis", page_icon="chart_with_upwards_trend", layout="wide")
st.title("Stock Technical Analysis Dashboard")
st.markdown("Built with Python Â· `yfinance` Â· `ta` Â· `plotly` Â· `streamlit`")
st.divider()

st.sidebar.header("âš™ï¸ Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Time Period", options=["1mo", "3mo", "6mo", "1y", "2y"], index=3)
st.sidebar.divider()
st.sidebar.markdown("**Indicators**")
show_sma    = st.sidebar.checkbox("SMA 20 / 50",     value=True)
show_ema    = st.sidebar.checkbox("EMA 20",          value=True)
show_bb     = st.sidebar.checkbox("Bollinger Bands", value=True)
show_volume = st.sidebar.checkbox("Volume",          value=True)
run = st.sidebar.button("Run Analysis", use_container_width=True)

@st.cache_data
def get_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data

def add_indicators(data):
    close = data["Close"].squeeze()
    data["SMA_20"]         = data["Close"].rolling(window=20).mean()
    data["SMA_50"]         = data["Close"].rolling(window=50).mean()
    data["EMA_20"]         = data["Close"].ewm(span=20, adjust=False).mean()
    data["RSI"]            = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd                   = ta.trend.MACD(close=close)
    data["MACD"]           = macd.macd()
    data["MACD_Signal"]    = macd.macd_signal()
    data["MACD_Histogram"] = macd.macd_diff()
    bb                     = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    data["BB_Upper"]       = bb.bollinger_hband()
    data["BB_Middle"]      = bb.bollinger_mavg()
    data["BB_Lower"]       = bb.bollinger_lband()
    data["BB_Width"]       = bb.bollinger_wband()
    data["Volume_SMA20"]   = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"]   = data["Volume"] / data["Volume_SMA20"]
    data["Volume_Spike"]   = data["Volume_Ratio"] > 1.5
    data.dropna(inplace=True)
    return data

def detect_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals["Close"]  = data["Close"]
    signals["Signal"] = ""
    signals["Type"]   = ""
    golden = (data["SMA_20"] > data["SMA_50"]) & (data["SMA_20"].shift(1) <= data["SMA_50"].shift(1))
    death  = (data["SMA_20"] < data["SMA_50"]) & (data["SMA_20"].shift(1) >= data["SMA_50"].shift(1))
    signals.loc[golden, ["Signal","Type"]] = ["BUY",  "Golden Cross"]
    signals.loc[death,  ["Signal","Type"]] = ["SELL", "Death Cross"]
    signals.loc[data["RSI"] < 30, ["Signal","Type"]] = ["BUY",  "RSI Oversold"]
    signals.loc[data["RSI"] > 70, ["Signal","Type"]] = ["SELL", "RSI Overbought"]
    macd_bull = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
    macd_bear = (data["MACD"] < data["MACD_Signal"]) & (data["MACD"].shift(1) >= data["MACD_Signal"].shift(1))
    signals.loc[macd_bull, ["Signal","Type"]] = ["BUY",  "MACD Crossover"]
    signals.loc[macd_bear, ["Signal","Type"]] = ["SELL", "MACD Crossover"]
    signals.loc[data["Close"].squeeze() <= data["BB_Lower"], ["Signal","Type"]] = ["BUY",  "BB Lower Touch"]
    signals.loc[data["Close"].squeeze() >= data["BB_Upper"], ["Signal","Type"]] = ["SELL", "BB Upper Touch"]
    return signals[signals["Signal"] != ""]

if run:
    with st.spinner(f"Downloading {ticker} data..."):
        data = get_data(ticker, period)
    if data.empty:
        st.error("No data found. Check the ticker symbol.")
        st.stop()
    if len(data) < 60:
        st.warning("Not enough data for this period. Please select 6mo or longer for best results.")
        st.stop()
    with st.spinner("Calculating indicators..."):
        data    = add_indicators(data)
        signals = detect_signals(data)
    latest    = data.iloc[-1]
    close_val = float(latest["Close"])
    sma20_val = float(latest["SMA_20"])
    sma50_val = float(latest["SMA_50"])
    rsi_val   = float(latest["RSI"])
    vol_ratio = float(latest["Volume_Ratio"])
    st.subheader(f"f{ticker} Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close Price",  f"${close_val:.2f}")
    c2.metric("SMA 20",       f"${sma20_val:.2f}")
    c3.metric("SMA 50",       f"${sma50_val:.2f}")
    c4.metric("RSI (14)",     f"{rsi_val:.1f}",
              delta="Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
    c5.metric("Volume Ratio", f"{vol_ratio:.2f}x",
              delta="Spike!" if vol_ratio > 1.5 else "Normal")
    st.divider()
    rows      = 4 if show_volume else 3
    heights   = [0.45, 0.2, 0.2, 0.15] if show_volume else [0.5, 0.25, 0.25]
    subtitles = [f"{ticker} Price", "RSI (14)", "MACD"]
    if show_volume:
        subtitles.append("Volume")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights,
                        subplot_titles=subtitles, vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                 low=data["Low"], close=data["Close"], name="Price",
                                 increasing_line_color="limegreen",
                                 decreasing_line_color="red"), row=1, col=1)
    if show_sma:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color="orange",    width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color="royalblue", width=1.2)), row=1, col=1)
    if show_ema:
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"], name="EMA 20", line=dict(color="violet",    width=1.2)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper",
                                 line=dict(color="rgba(255,165,0,0.5)", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower",
                                 line=dict(color="rgba(255,165,0,0.5)", width=1, dash="dash"),
                                 fill="tonexty", fillcolor="rgba(255,165,0,0.05)"), row=1, col=1)
    buys  = signals[signals["Signal"] == "BUY"]
    sells = signals[signals["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["Close"]  * 0.98, mode="markers", name="Buy",
                             marker=dict(symbol="triangle-up",   size=10, color="limegreen")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"] * 1.02, mode="markers", name="Sell",
                             marker=dict(symbol="triangle-down", size=10, color="red")),       row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="orchid", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"],        name="MACD",   line=dict(color="royalblue", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color="orange",    width=1.5)), row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["MACD_Histogram"], name="Hist",
                         marker_color=["limegreen" if v >= 0 else "red" for v in data["MACD_Histogram"]]), row=3, col=1)
    if show_volume:
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume",
                             marker_color=["red" if s else "steelblue" for s in data["Volume_Spike"]]), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["Volume_SMA20"], name="Avg Vol",
                                 line=dict(color="white", width=1, dash="dot")), row=4, col=1)
    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=-0.05), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Detected Signals")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Buy Signals")
        st.dataframe(buys[["Close","Type"]].tail(10).sort_index(ascending=False), use_container_width=True)
    with col2:
        st.markdown("#### Sell Signals")
        st.dataframe(sells[["Close","Type"]].tail(10).sort_index(ascending=False), use_container_width=True)
    st.divider()
    st.subheader("Export Data")
    c1, c2 = st.columns(2)
    c1.download_button("Download Full Data (CSV)", data=data.to_csv().encode("utf-8"), file_name=f"{ticker}_data.csv", mime="text/csv", use_container_width=True)
    c2.download_button("Download Signals (CSV)", data=signals.to_csv().encode("utf-8"), file_name=f"{ticker}_signals.csv", mime="text/csv", use_container_width=True)
else:
    st.info("Enter a ticker in the sidebar and click Run Analysis to start.")
