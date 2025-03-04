import streamlit as st
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# CoinMarketCap API key (replace with your own API key)
COINMARKETCAP_API_KEY = "e53652fe-973b-40c8-83aa-3610b7a6f0c6"

# Function to fetch market dominance data from CoinMarketCap API
def fetch_market_dominance():
    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        btc_dominance = data['data']['btc_dominance']  # Bitcoin dominance in percentage
        others_dominance = 100 - btc_dominance  # Others dominance in percentage
        return btc_dominance, others_dominance
    else:
        st.error("Failed to fetch market dominance data from CoinMarketCap API.")
        return None, None

# Function to fetch data from Binance API
def fetch_data(symbol, interval, limit=500):  # Default limit set to 500
    url = f"https://api.binance.us/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    else:
        st.error("Failed to fetch data from Binance API.")
        return None

# Function to fetch order book data
def fetch_order_book(symbol, limit=100):
    url = "https://api.binance.us/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch order book data.")
        return None

# Function to calculate liquidity at different price levels
def calculate_liquidity(order_book, depth_pct=0.1):
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

    bids = bids.sort_values(by='price', ascending=False)
    asks = asks.sort_values(by='price', ascending=True)

    # Calculate bid-ask spread
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    spread = best_ask - best_bid

    # Define price impact range
    price_range_bid = best_bid * (1 - depth_pct)  # e.g., 10% below best bid
    price_range_ask = best_ask * (1 + depth_pct)  # e.g., 10% above best ask

    # Cumulative liquidity within price impact range
    downside_liquidity_coin = bids[bids['price'] >= price_range_bid]['quantity'].sum()
    upside_liquidity_coin = asks[asks['price'] <= price_range_ask]['quantity'].sum()

    # Convert BTC liquidity to USD liquidity
    downside_liquidity_usd = (bids[bids['price'] >= price_range_bid]['price'] * bids[bids['price'] >= price_range_bid]['quantity']).sum()
    upside_liquidity_usd = (asks[asks['price'] <= price_range_ask]['price'] * asks[asks['price'] <= price_range_ask]['quantity']).sum()

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "downside_liquidity_coin": downside_liquidity_coin,
        "upside_liquidity_coin": upside_liquidity_coin,
        "downside_liquidity_usd": downside_liquidity_usd,
        "upside_liquidity_usd": upside_liquidity_usd
    }

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci(df):
    # Calculate Fibonacci retracement levels (38.2%, 50%, 61.8%)
    df['fib_382'] = df['close'].rolling(window=20).apply(lambda x: x.max() - (x.max() - x.min()) * 0.382)
    df['fib_50'] = df['close'].rolling(window=20).apply(lambda x: x.max() - (x.max() - x.min()) * 0.5)
    df['fib_618'] = df['close'].rolling(window=20).apply(lambda x: x.max() - (x.max() - x.min()) * 0.618)
    return df

# Function to calculate support and resistance levels
def calculate_support_resistance(df):
    # Calculate resistance level (R1)
    df['R1'] = df['high'].rolling(window=20).max()
    # Calculate support level (S1)
    df['S1'] = df['low'].rolling(window=20).min()
    return df

# Function to determine sentiment based on profitability
def determine_sentiment(profitability):
    if profitability > 10:  # High profitability
        return "Greed", "green"
    elif profitability > -10:  # Moderate profitability
        return "Neutral", "gray"
    else:  # Low profitability
        return "Fear", "red"

# Function to apply trading strategy
def apply_strategy(df):
    # Calculate technical indicators using pandas_ta
    df.ta.ema(length=20, append=True)  # 20-period EMA
    df.ta.ema(length=50, append=True)  # 50-period EMA
    df.ta.ema(length=100, append=True)  # 100-period EMA
    df.ta.macd(fast=12, slow=26, signal=9, append=True)  # MACD

    # Calculate Fibonacci retracement levels
    df = calculate_fibonacci(df)

    # Generate buy/sell signals
    df['signal'] = 0
    # Buy signal: EMA(20) > EMA(50) > EMA(100) and (MACD > MACD Signal or Close > Fibonacci 61.8%)
    df.loc[
        (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_100']) &  # EMA condition
        (
            (df['MACD_12_26_9'] > df['MACDs_12_26_9']) |  # MACD condition
            (df['close'] > df['fib_618'])  # Fibonacci condition
        ),
        'signal'
    ] = 1
    # Sell signal: EMA(20) < EMA(50) < EMA(100) and (MACD < MACD Signal or Close < Fibonacci 38.2%)
    df.loc[
        (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_100']) &  # EMA condition
        (
            (df['MACD_12_26_9'] < df['MACDs_12_26_9']) |  # MACD condition
            (df['close'] < df['fib_382'])  # Fibonacci condition
        ),
        'signal'
    ] = -1

    return df

# Function to evaluate profitability
def evaluate_profitability(df):
    initial_investment = 100  # Fixed initial investment
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    final_value = initial_investment * df['cumulative_returns'].iloc[-1]
    profitability = ((final_value - initial_investment) / initial_investment) * 100
    return profitability

# Function to detect market manipulation
def detect_market_manipulation(df):
    # Simple heuristic: Check for abnormal volume spikes
    volume_mean = df['volume'].mean()
    volume_std = df['volume'].std()
    abnormal_volume = df['volume'] > (volume_mean + 2 * volume_std)
    manipulation_detected = abnormal_volume.any()
    return manipulation_detected

# Function to calculate volatility
def calculate_volatility(df):
    # Calculate daily volatility as the standard deviation of daily returns
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.std() * 100  # Convert to percentage
    return volatility

# Function to interpret volatility
def interpret_volatility(volatility):
    if volatility < 1:
        return "Extreme Low Volatility: Market is very stable. Suitable for conservative trading."
    elif 1 <= volatility <= 5:
        return "Moderate Volatility: Market is balanced. Good for normal trading."
    else:
        return "Extreme High Volatility: Market is highly risky. Trade with caution or avoid."

# Function to evaluate market conditions based on volatility and manipulation
def evaluate_market_conditions(volatility, manipulation_detected):
    if manipulation_detected:
        if volatility < 1:
            return "Market is stable but manipulation detected. **Trade with extreme caution.**"
        elif 1 <= volatility <= 5:
            return "Market is balanced but manipulation detected. **Trade with caution.**"
        else:
            return "Market is highly risky and manipulation detected. **Avoid trading.**"
    else:
        if volatility < 1:
            return "Market is very stable with no manipulation. **Suitable for conservative trading.**"
        elif 1 <= volatility <= 5:
            return "Market is balanced with no manipulation. **Good for normal trading.**"
        else:
            return "Market is highly risky but no manipulation detected. **Trade with caution or avoid.**"

# Function to display live candlestick chart with EMAs
def display_chart(df):
    # Add EMAs to the chart
    apds = [
        mpf.make_addplot(df['EMA_20'], color='yellow', width=1, panel=0),  # EMA 20 (Yellow)
        mpf.make_addplot(df['EMA_50'], color='green', width=1, panel=0),  # EMA 50 (Green)
        mpf.make_addplot(df['EMA_100'], color='blue', width=1, panel=0),  # EMA 100 (Blue)
    ]

    # Create a figure and axis for the candlestick chart
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        volume=True,  # Add volume subplot
        addplot=apds,  # Add EMAs to the chart
        returnfig=True
    )
    st.pyplot(fig)

# Function to format numbers in M/K notation
def format_number(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"  # Convert to millions
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"  # Convert to thousands
    else:
        return f"{value:.2f}"  # Keep normal format

# Main Streamlit app
def main():
    # Set page title and layout
    st.set_page_config(page_title="Crypto Trading Dashboard", layout="wide")

    # Add a title and description
    st.title("🚀 Cryptocurrency Trading Strategy Analyzer")
    st.markdown("""
        This app fetches live cryptocurrency data from Binance, applies a trading strategy, and evaluates profitability.
        **Explore the market dynamics and make informed trading decisions!**
    """)

    # Fetch market dominance data
    btc_dominance, others_dominance = fetch_market_dominance()
    if btc_dominance is not None and others_dominance is not None:
        # Display market dominance in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("**Bitcoin Dominance (BTC.D)**", f"{btc_dominance:.2f}%")
        with col2:
            st.metric("**Others Dominance (OTHER.D)**", f"{others_dominance:.2f}%")

    # Add a divider
    st.markdown("---")

    # User inputs in a sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        symbol = st.selectbox("Select Coin", [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "USUALUSDT", "XLMUSDT", "STXUSDT", "VELODROMEUSDT", 
            "TIAUSDT", "IOTAUSDT", "THETAUSDT", "NEARUSDT", "HBARUSDT", "ADAUSDT", 
            "MKRUSDT", "TRUMPUSDT", "DOGEUSDT", "FLOKIUSDT", "FILUSDT","SOLUSDT","SUIUSDT",
            "QTUMUSDT","AVAXUSDT","DOTUSDT","FETUSDT","GALAUSDT","TRXUSDT","MANAUSDT","SANDUST",
            "ARKMUSDT","POLUSDT","OCEANUSDT","LPTUSDT"
        ])
        interval = st.selectbox("Select Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],index=2)
        limit = st.slider("Select Limit for Data Fetching", min_value=100, max_value=2000, value=500, step=100)

    # Fetch candlestick data
    df = fetch_data(symbol, interval, limit=limit)
    if df is not None:
        # Apply strategy
        df = apply_strategy(df)

        # Calculate support and resistance levels
        df = calculate_support_resistance(df)
        css = """
        <style>
        [data-testid="stMetricValue"] {
            color: gray;
        }
        </style>"""
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("**Current Price**", f"{df['close'].iloc[-1]:.4f}")
        with col2:
            st.metric("**Resistance (R1)**", f"{df['R1'].iloc[-1]:.4f}")
        with col3:
            st.markdown(css, unsafe_allow_html=True)
            st.metric("**Support (S1)**", f"{df['S1'].iloc[-1]:.4f}")

    

        # Display the latest signal
        latest_signal = df['signal'].iloc[-1]
        signal_text = "Buy" if latest_signal == 1 else "Sell" if latest_signal == -1 else "Hold"
        signal_color = "green" if latest_signal == 1 else "red" if latest_signal == -1 else "gray"
        st.markdown(f"**Latest Signal:** :{signal_color}[{signal_text}]")

        # Evaluate profitability
        profitability = evaluate_profitability(df)
        sentiment, sentiment_color = determine_sentiment(profitability)
        st.markdown(f"**Profitability:** {profitability:.2f}%")
        st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment}]")

        # Calculate volatility
        volatility = calculate_volatility(df)
        volatility_interpretation = interpret_volatility(volatility)
        st.markdown(f"**Volatility:** {volatility:.2f}%")
        st.markdown(f"**Volatility Interpretation:** {volatility_interpretation}")

        # Fetch order book data
        order_book = fetch_order_book(symbol, limit=limit)
        if order_book:
            # Calculate highest liquidity on upside and downside
            liquidity_data= calculate_liquidity(order_book)
            st.markdown("**Liquidity Analysis**")
            if liquidity_data:
                # Display results in two columns
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Downside Liquidity Coin", f"{liquidity_data['downside_liquidity_coin']:.2f}")
                    st.metric("Downside Liquidity USD", f"{format_number(liquidity_data['downside_liquidity_usd'])}")

                with col2:
                    st.metric("Upside Liquidity Coin", f"{liquidity_data['upside_liquidity_coin']:.2f}")
                    st.metric("Upside Liquidity USD", f"{format_number(liquidity_data['upside_liquidity_usd'])}")

                

            

        # Display candlestick chart with EMAs
        st.markdown("**Live Candlestick Chart**")
        display_chart(df)

        # Evaluate best timeframe
        st.markdown("**Evaluating Best Timeframe for Trading**")
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        profitability_dict = {}

        for tf in timeframes:
            df_tf = fetch_data(symbol, tf, limit=limit)
            if df_tf is not None:
                df_tf = apply_strategy(df_tf)
                profitability_dict[tf] = evaluate_profitability(df_tf)

        if profitability_dict:
            best_timeframe = max(profitability_dict, key=profitability_dict.get)
            best_profitability = profitability_dict[best_timeframe]
            st.markdown(f"**Best Timeframe for Trading:** {best_timeframe}")
            st.markdown(f"**Profitability for Best Timeframe:** {best_profitability:.2f}%")

# Run the app
if __name__ == "__main__":
    main()
