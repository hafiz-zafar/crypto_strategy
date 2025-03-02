import streamlit as st
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Function to fetch data from Binance API
def fetch_data(symbol, interval, limit=2000):  # Updated limit to 2000
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

# Main Streamlit app
def main():
    st.title("Cryptocurrency Trading Strategy Analyzer")
    st.write("This app fetches live cryptocurrency data from Binance, applies a trading strategy, and evaluates profitability.")

    # User inputs
    symbol = st.selectbox("Select Coin", [
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "USUALUSDT", "XLMUSDT", "STXUSDT", "VELODROMEUSDT", 
        "TIAUSDT", "IOTAUSDT", "THETAUSDT", "NEARUSDT", "HBARUSDT", "ADAUSDT", 
        "MKRUSDT", "TRUMPUSDT", "DOGEUSDT", "FLOKIUSDT", "FILUSDT","SOLUSDT","SUIUSDT",
        "QTUMUSDT","AVAXUSDT","DOTUSDT","FETUSDT","GALAUSDT","TRXUSDT","MANAUSDT","SANDUST",
        "ARKMUSDT","POLUSDT","OCEANUSDT","LPTUSDT"
    ])
    interval = st.selectbox("Select Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])

    # Fetch data
    df = fetch_data(symbol, interval)
    if df is not None:
        # Apply strategy
        df = apply_strategy(df)

        # Calculate support and resistance levels
        df = calculate_support_resistance(df)

        # Display support and resistance levels
        st.write("**Support and Resistance Levels**")
        st.write(f"**Resistance (R1):** {df['R1'].iloc[-1]:.4f}")
        st.write(f"**Support (S1):** {df['S1'].iloc[-1]:.4f}")

        # Display current price
        current_price = df['close'].iloc[-1]
        st.write(f"**Current Price:** {current_price:.4f}")

        # Get the latest signal
        latest_signal = df['signal'].iloc[-1]
        signal_text = "Buy" if latest_signal == 1 else "Sell" if latest_signal == -1 else "Hold"
        signal_color = "green" if latest_signal == 1 else "red" if latest_signal == -1 else "gray"

        # Display the latest signal
        st.write(f"**Latest Signal:** :{signal_color}[{signal_text}]")

        # Evaluate profitability
        profitability = evaluate_profitability(df)
        st.write(f"**Profitability:** {profitability:.2f}%")

        # Determine sentiment
        sentiment, sentiment_color = determine_sentiment(profitability)
        st.write(f"**Sentiment:** :{sentiment_color}[{sentiment}]")

  

        # Calculate volatility
        volatility = calculate_volatility(df)
        st.write(f"**Volatility:** {volatility:.2f}%")

        # Interpret volatility
        volatility_interpretation = interpret_volatility(volatility)
        st.write(f"**Volatility Interpretation:** {volatility_interpretation}")

  

        # Display candlestick chart with EMAs
        st.write("**Live Candlestick Chart**")
        display_chart(df)

        # Evaluate best timeframe
        st.write("**Evaluating Best Timeframe for Trading**")
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        profitability_dict = {}

        for tf in timeframes:
            df_tf = fetch_data(symbol, tf)
            if df_tf is not None:
                df_tf = apply_strategy(df_tf)
                profitability_dict[tf] = evaluate_profitability(df_tf)

        if profitability_dict:
            best_timeframe = max(profitability_dict, key=profitability_dict.get)
            best_profitability = profitability_dict[best_timeframe]
            st.write(f"**Best Timeframe for Trading:** {best_timeframe}")
            st.write(f"**Profitability for Best Timeframe:** {best_profitability:.2f}%")

# Run the app
if __name__ == "__main__":
    main()