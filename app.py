import streamlit as st
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import time
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# CoinMarketCap API key (replace with your own API key)
load_dotenv()
API_KEY = os.getenv("COINMARKETCAP_API_KEY")
COINMARKETCAP_API_KEY = API_KEY

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

def fetch_24h_volume(symbol):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY
    }
    params = {"symbol": symbol.upper()}  # Ensure uppercase symbol
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if symbol.upper() in data["data"]:
            volume_24h = data["data"][symbol.upper()]["quote"]["USD"]["volume_24h"]
            return format_volume(volume_24h)
        else:
            st.warning("Symbol not found in response.")
            return None
    else:
        st.error("Failed to fetch data from CoinMarketCap API.")
        return None

def fetch_fear_greed_index():
    """Fetches the market's overall Fear & Greed index with text and value"""
    url = "https://api.alternative.me/fng/"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        fear_greed_value = data['data'][0]['value']  # Numeric value
        fear_greed_text = data['data'][0]['value_classification']  # Text (e.g., Fear, Greed, Neutral)

        return f"{fear_greed_text} ({fear_greed_value})"
    else:
        st.error("Failed to fetch Fear & Greed Index.")
        return None

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
def fetch_order_book(symbol, limit=500):
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

    # Liquidity Price
    cumulative_bid_liquidity = 0
    downside_price = best_bid
    for _, row in bids.iterrows():
        if row['price'] < price_range_bid:
            break
        cumulative_bid_liquidity += row['quantity'] * row['price']
        downside_price = row['price']
    
    cumulative_ask_liquidity = 0
    upside_price = best_ask
    for _, row in asks.iterrows():
        if row['price'] > price_range_ask:
            break
        cumulative_ask_liquidity += row['quantity'] * row['price']
        upside_price = row['price']

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "downside_liquidity_coin": downside_liquidity_coin,
        "upside_liquidity_coin": upside_liquidity_coin,
        "downside_liquidity_usd": downside_liquidity_usd,
        "upside_liquidity_usd": upside_liquidity_usd,
        "downside_price": downside_price,
        "upside_price": upside_price
    }


# Function to calculate support and resistance levels
def calculate_support_resistance(df):
    # Calculate resistance level (R1)
    df['R1'] = df['high'].rolling(window=20).max()
    # Calculate support level (S1)
    df['S1'] = df['low'].rolling(window=20).min()
    return df

# Function to apply trading strategy

def calculate_dynamic_fibonacci(df):
    """Calculate dynamic Fibonacci levels based on high and low prices."""
    # Calculate high and low over the period (can be adjusted)
    high_price = df['high'].max()
    low_price = df['low'].min()
    
    # Calculate the range
    price_range = high_price - low_price
    
    # Calculate Fibonacci levels
    fib_236 = low_price + 0.236 * price_range  # 23.6% Fibonacci level
    fib_382 = low_price + 0.382 * price_range  # 38.2% Fibonacci level
    fib_50 = low_price + 0.5 * price_range  # 50% Fibonacci level
    fib_618 = low_price + 0.618 * price_range  # 61.8% Fibonacci level
    fib_100 = high_price  # 100% Fibonacci level
    
    return fib_236, fib_382, fib_50, fib_618, fib_100

def get_timeframe_settings(timeframe):
    if timeframe in ['1m','3m','5m', '15m', '30m']:
        # Settings for 1 to 30 minutes
        ema_lengths = [9, 21, 50]
        macd_params = (6, 13, 5)
        rsi_length = 14
        adx_length = 14
        atr_length = 14
        volume_ma_window = 20
    elif timeframe in ['1h', '4h', '1d']:
        # Settings for 1 hour to 1 day
        ema_lengths = [20, 50, 200]
        macd_params = (12, 26, 9)
        rsi_length = 14
        adx_length = 14
        atr_length = 14
        volume_ma_window = 50
    else:
        raise ValueError("Unsupported timeframe selected.")
    
    return ema_lengths, macd_params, rsi_length, adx_length, atr_length, volume_ma_window

def apply_strategy(df, timeframe):
    # Get timeframe-specific settings
    ema_lengths, macd_params, rsi_length, adx_length, atr_length, volume_ma_window = get_timeframe_settings(timeframe)
    
    # Calculate technical indicators
    df.ta.ema(length=ema_lengths[0], append=True)
    df.ta.ema(length=ema_lengths[1], append=True)
    df.ta.ema(length=ema_lengths[2], append=True)
    
    # Calculate MACD and dynamically generate column names
    macd_fast, macd_slow, macd_signal = macd_params
    df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
    macd_column = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"  # Dynamic MACD column name
    macd_signal_column = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"  # Dynamic MACD signal column name
    
    df.ta.rsi(length=rsi_length, append=True)
    df.ta.adx(length=adx_length, append=True)
    df.ta.atr(length=atr_length, append=True)
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_window).mean()

    # Calculate Fibonacci retracement levels
    df = calculate_fibonacci(df)

    # Generate buy/sell signals
    df['signal'] = 0

    # Buy signal: EMA(9) > EMA(21) > EMA(50), MACD > MACD Signal, Close > Fibonacci 50%, RSI > 60, ADX > 25, Volume > MA
    df.loc[
        (df[f'EMA_{ema_lengths[0]}'] > df[f'EMA_{ema_lengths[1]}']) & 
        (df[f'EMA_{ema_lengths[1]}'] > df[f'EMA_{ema_lengths[2]}']) &  # EMA condition
        (df[macd_column] > df[macd_signal_column]) &  # MACD condition (dynamic column name)
        (df['close'] > df['fib_50']) &  # Fibonacci condition
        (df[f'RSI_{rsi_length}'] > 50) &  # RSI condition
        (df[f'ADX_{adx_length}'] > 20) &  # ADX condition
        (df['volume'] > df['volume_ma']),  # Volume condition
        'signal'
    ] = 1

    # Sell signal: EMA(9) < EMA(21) < EMA(50), MACD < MACD Signal, Close < Fibonacci 38.2%, RSI < 40, ADX > 25, Volume > MA
    df.loc[
        (df[f'EMA_{ema_lengths[0]}'] < df[f'EMA_{ema_lengths[1]}']) & 
        (df[f'EMA_{ema_lengths[1]}'] < df[f'EMA_{ema_lengths[2]}']) &  # EMA condition
        (df[macd_column] < df[macd_signal_column]) &  # MACD condition (dynamic column name)
        (df['close'] < df['fib_382']) &  # Fibonacci condition
        (df[f'RSI_{rsi_length}'] < 40) &  # RSI condition
        (df[f'ADX_{adx_length}'] > 20) &  # ADX condition
        (df['volume'] > df['volume_ma']),  # Volume condition
        'signal'
    ] = -1

    # Add confirmation candle
   # df['confirmed_signal'] = df['signal'].shift(1)
    return df




# Function to calculate Fibonacci retracement levels
def calculate_fibonacci(df, window=30):
    """
    Calculate Fibonacci retracement levels (38.2%, 50%, 61.8%) based on rolling high and low.
    """
    try:
        df['high_max'] = df['high'].rolling(window=window).max()
        df['low_min'] = df['low'].rolling(window=window).min()
        df['fib_382'] = df['high_max'] - (df['high_max'] - df['low_min']) * 0.382
        df['fib_50'] = df['high_max'] - (df['high_max'] - df['low_min']) * 0.5
        df['fib_618'] = df['high_max'] - (df['high_max'] - df['low_min']) * 0.618
    except Exception as e:
        print(f"Error calculating Fibonacci levels: {e}")
    return df


# here code starts
# Function to backtest the strategy
def backtest_strategy(df):
    """
    Backtest the strategy and calculate cumulative returns.
    """
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()


    # Calculate strategy returns based on signals
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
   
    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    
    return df


def calculate_metrics(df,investment_amount=1000):
    """
    Calculate performance metrics like win rate, risk-reward ratio, max drawdown,
    long/short trades, positive/negative trades, profit factor, and date range.
    """
    # Total trades

    total_trades = df['signal'].abs().sum()

    # Long trades (buy signals)
    long_trades = df[df['signal'] == 1]['signal'].count()

    # Short trades (sell signals)
    short_trades = df[df['signal'] == -1]['signal'].abs().count()

    # Winning trades (positive returns)
    winning_trades = df[df['strategy_returns'] > 0]['strategy_returns'].count()

    # Losing trades (negative returns)
    losing_trades = df[df['strategy_returns'] < 0]['strategy_returns'].count()

    # Win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Risk-reward ratio
    avg_gain = df[df['strategy_returns'] > 0]['strategy_returns'].mean()
    avg_loss = df[df['strategy_returns'] < 0]['strategy_returns'].mean()
    risk_reward_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0

    # Maximum drawdown
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = df['cumulative_returns'] - df['cumulative_max']
    max_drawdown = df['drawdown'].min()

    # Profit factor
    gross_profit = df[df['strategy_returns'] > 0]['strategy_returns'].sum()
    gross_loss = df[df['strategy_returns'] < 0]['strategy_returns'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # Date range and number of days
    start_date = df.index[0]  # First date in the dataset
    end_date = df.index[-1]  # Last date in the dataset
    num_days = (end_date - start_date).days  # Number of days

     # Calculate profit/loss in USD
    final_cumulative_return = df['cumulative_returns'].iloc[-1]  # Final cumulative return
    profit_loss_usd = investment_amount * final_cumulative_return  # Profit/loss in USD

    
      # Create a list for the metrics to display
    metrics = [
        ("Total Trades", total_trades),
        ("Long Trades", long_trades),
        ("Short Trades", short_trades),
        ("Positive Trades", winning_trades),
        ("Negative Trades", losing_trades),
        ("Win Rate", f"{win_rate:.2%}"),
        ("Risk-Reward Ratio", risk_reward_ratio),
        ("Profit Factor", profit_factor),     
        ("Start Date", start_date.strftime('%Y-%m-%d')),
        ("End Date", end_date.strftime('%Y-%m-%d')),
        ("Maximum Drawdown", f"{max_drawdown:.2%}"),
        ("Number of Days", num_days),
        ("Profit/Loss (USD)", f"${profit_loss_usd:.2f}")  # Add profit/loss in USD
        
    ]
    
# Using markdown with HTML to create a table with vertical and horizontal lines
    st.markdown("### Strategy Performance")
       # Create 6 columns to organize the data better
    col1, col2, col3, col4, col5, col6 = st.columns(6)  # 6 columns

    # Assigning values to columns
    with col1:
        st.write("Total Trades")
        st.write("Long Trades")
        st.write("Short Trades")
        st.write("Positive Trades")
        st.write("Negative Trades")
        
    with col2:
        st.write(total_trades)
        st.write(long_trades)
        st.write(short_trades)
        st.write(winning_trades)
        st.write(losing_trades)

    with col3:
        st.write("Win Rate")
        st.write("Risk-Reward Ratio")
        st.write("Profit Factor")
        st.write("Start Date")
        st.write("End Date")
       
        
    with col4:
        st.write(f"{win_rate:.2%}")
        st.write(risk_reward_ratio)
        st.write(profit_factor)
        st.write(start_date.strftime('%Y-%m-%d'))
        st.write(end_date.strftime('%Y-%m-%d'))       
        

    with col5:        
        st.write("Maximum Drawdown")
        st.write("Number of Days")
        st.write("Profit/Loss")
        
    with col6:
        st.write(f"{max_drawdown:.2%}")
        st.write(num_days)
        st.write(f"{investment_amount} / {profit_loss_usd:.2f}")
# Main function to run the strategy
def run_strategy(df,interval):
    """
    Run the strategy, backtest, and visualize results.
    """
    # Apply strategy
    df = apply_strategy(df,interval)

    # Backtest strategy
    df = backtest_strategy(df)

    # Calculate performance metrics
    calculate_metrics(df)

    
#here code ends

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
def display_chart(df, timeframe):
    # Get timeframe-specific settings
    ema_lengths, macd_params, rsi_length, adx_length, atr_length, volume_ma_window = get_timeframe_settings(timeframe)
    
    # Dynamically generate EMA column names
    ema_columns = [f'EMA_{ema_length}' for ema_length in ema_lengths]
    
    # Dynamically generate MACD column names
    macd_fast, macd_slow, macd_signal = macd_params
    macd_column = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    macd_signal_column = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
    macd_histogram_column = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"

    # Add EMAs to the chart
    apds = [
        mpf.make_addplot(df[ema_columns[0]], color='yellow', width=1, panel=0),  # EMA 1 (Yellow)
        mpf.make_addplot(df[ema_columns[1]], color='green', width=1, panel=0),  # EMA 2 (Green)
        mpf.make_addplot(df[ema_columns[2]], color='blue', width=1, panel=0),  # EMA 3 (Blue)
    ]

    # Add MACD to the chart (optional)
    apds.append(mpf.make_addplot(df[macd_column], color='purple', width=1, panel=2, ylabel='MACD'))  # MACD line
    apds.append(mpf.make_addplot(df[macd_signal_column], color='orange', width=1, panel=2))  # MACD signal line
    apds.append(mpf.make_addplot(df[macd_histogram_column], type='bar', color='gray', width=0.7, panel=2))  # MACD histogram

    # Create a figure and axis for the candlestick chart
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        volume=True,  # Add volume subplot
        addplot=apds,  # Add EMAs and MACD to the chart
        returnfig=True,
        figsize=(12, 8),
        panel_ratios=(4, 1, 1)  # Adjust panel ratios for main chart, volume, and MACD
    )
    
    # Display the chart in Streamlit
    st.pyplot(fig)

# Function to format numbers in M/K notation
def format_number(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"  # Convert to millions
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"  # Convert to thousands
    else:
        return f"{value:.2f}"  # Keep normal format


# GRU MODEL TRAINING STARTS HERE

# ## Prepare Data

def prepare_data(df, time_steps=20):
    if len(df) <= time_steps:
        raise ValueError(f"Dataset too small for time_steps={time_steps}. Needs at least {time_steps + 1} rows.")
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(df_scaled) - time_steps):
        X.append(df_scaled[i:i + time_steps])
        y.append(df_scaled[i + time_steps, 0])  # Select only the 'close' column
    
   
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Reshape to (samples, 1)
    return X, y, scaler

# ## Build GRU Model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        GRU(50, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ## Train and Save Model for Multiple Coins and Intervals

def train_and_save_model(symbols, intervals, epochs=100, batch_size=16, time_steps=20, model_filename="crypto_gru_model.h5"):
    combined_data = []
    
    for symbol in symbols:
        for interval in intervals:
            print(f"Fetching data for {symbol} at {interval} interval...")
            df = fetch_data(symbol, interval,limit=500)
            if df is None:
                continue
            combined_data.append(df)
    
    if not combined_data:
        print("No data fetched for the specified symbols and intervals.")
        return None
    
    # Combine all dataframes into a single one
    combined_df = pd.concat(combined_data, axis=0)
    
    # Prepare the data
    X, y, scaler = prepare_data(combined_df, time_steps)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    model = build_gru_model((X.shape[1], X.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
              callbacks=[early_stopping, model_checkpoint])
    
    # Load the best model after training finishes
    model = load_model(model_filename)
    
    model.save(model_filename)
    scaler_filename = model_filename.replace(".h5", "_scaler.pkl")
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved as {model_filename}, Scaler saved as {scaler_filename}")
    return model_filename, scaler_filename
# GRU MODEL TRAINING ENDS HERE

# ## Predict Price
def predict_price(model_filename, scaler_filename, symbol, interval, time_steps=20):
    print(f"Fetching latest data for {symbol} at {interval} interval...")

    df = fetch_data(symbol, interval, limit=time_steps)
    if df is None or len(df) < time_steps:
        print("Not enough data for prediction.")
        return None

    # Load the model & scaler
    model = load_model(model_filename)
    scaler = joblib.load(scaler_filename)

    # Prepare input data
    df_scaled = scaler.transform(df)  # Ensure shape is (time_steps, num_features)
    
    num_features = df_scaled.shape[1]  # Dynamically get the number of features
    X_input = df_scaled.reshape(1, time_steps, num_features)  # Shape: (1, 20, num_features)

    # Make prediction
    predicted_scaled = model.predict(X_input)  # Ensure shape: (1, num_features) or (1, 1)

    # Ensure correct shape before inverse scaling
    if predicted_scaled.shape[1] != num_features:
        predicted_scaled = np.tile(predicted_scaled, (1, num_features))  # Repeat for missing features

    # Convert back to original scale
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    
    return predicted_price

# Function to format large numbers in K, M, B
def format_volume(volume):
    volume = float(volume)  # Ensure the volume is a float
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.2f} B"  # Billions
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f} M"  # Millions
    elif volume >= 1_000:
        return f"{volume / 1_000:.2f} K"  # Thousands
    else:
        return str(volume)  # No formatting needed if below 1,000





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
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT", 
            "TIAUSDT", "IOTAUSDT", "THETAUSDT", "NEARUSDT", "HBARUSDT", "ADAUSDT", 
            "MKRUSDT", "TRUMPUSDT", "DOGEUSDT", "FLOKIUSDT", "FILUSDT","SOLUSDT","SUIUSDT",
            "QTUMUSDT","AVAXUSDT","DOTUSDT","FETUSDT","GALAUSDT","TRXUSDT","MANAUSDT","SANDUSDT"
            ,"POLUSDT","OCEANUSDT","LPTUSDT"
        ],index=0)
        interval = st.selectbox("Select Timeframe", ["1m","3m","5m", "15m", "30m", "1h", "4h", "1d"],index=3)
        limit = st.slider("Select Limit for Data Fetching", min_value=100, max_value=2000, value=500, step=100)
        
        epochs = st.slider("Select Number of Epochs", min_value=10, max_value=200, value=50)
        batch_size = st.slider("Select Batch Size", min_value=8, max_value=64, value=16)
      # Fetch candlestick data
    df = fetch_data(symbol, interval, limit=limit)
    # 24 hour volume
    volume = fetch_24h_volume(symbol.replace("USDT", "").upper())
    if volume:
        st.success(f"24h Trading Volume for {symbol.replace('USDT', '').upper()}: {volume}")
        
    
    

    if st.button("Predict Price"):
        with st.spinner("Predicting in progress..."):
            model_path, scaler_path = train_and_save_model([symbol], [interval], epochs=epochs, batch_size=batch_size)
            predicted_price = predict_price("crypto_gru_model.h5", "crypto_gru_model_scaler.pkl", [symbol], [interval])
            current_price = df['close'].iloc[-1]  # Get latest closing price
            if predicted_price:
                # Compare and display in appropriate color
              if predicted_price > current_price:
                  st.success(f"BUY Predicted next price for {symbol} ({interval}): **${predicted_price:.4f}**")
              else:
                  st.error(f"SELL Predicted next price for {symbol} ({interval}): **${predicted_price:.4f}**")
            else:
                st.error("Prediction failed. Not enough data.")
           # st.success(f"Model trained and saved as {model_path}")
           # st.success(f"Scaler saved as {scaler_path}")
            st.balloons()
            

  
    if df is not None:
        # Apply strategy
        df = apply_strategy(df,interval)

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
        st.markdown(f"**Profitability:** {profitability:.2f}%")
        fear_greed = fetch_fear_greed_index()
        if fear_greed:
            st.markdown(f"**Market Sentiment:** {fear_greed}")

        
        #st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment}]")

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
                col1, col2,col3 = st.columns(3)

                with col1:
                    st.metric("Upside Liquidity Coin", f"{liquidity_data['upside_liquidity_coin']:.2f}")
                    st.metric("Downside Liquidity Coin", f"{liquidity_data['downside_liquidity_coin']:.2f}")
                    
                    
                with col2: 
                    st.metric("Upside Liquidity USD", f"{format_number(liquidity_data['upside_liquidity_usd'])}")
                    st.metric("Downside Liquidity USD", f"{format_number(liquidity_data['downside_liquidity_usd'])}")
                with col3:
                    st.metric("Upside Price", f"{liquidity_data['upside_price']}")
                    st.metric("Downside Price", f"{liquidity_data['downside_price']}")

        run_strategy(df,interval)
      

            

        # Display candlestick chart with EMAs
        st.markdown("**Live Candlestick Chart**")
        display_chart(df,interval)

        # Evaluate best timeframe
        st.markdown("**Evaluating Best Timeframe for Trading**")
        timeframes = ["1m","3m", "5m", "15m", "30m", "1h", "4h", "1d"]
        profitability_dict = {}

        for tf in timeframes:
            df_tf = fetch_data(symbol, tf, limit=limit)
            if df_tf is not None:
                df_tf = apply_strategy(df_tf,interval)
                profitability_dict[tf] = evaluate_profitability(df_tf)

        if profitability_dict:
            best_timeframe = max(profitability_dict, key=profitability_dict.get)
            best_profitability = profitability_dict[best_timeframe]
            st.markdown(f"**Best Timeframe for Trading:** {best_timeframe}")
            st.markdown(f"**Profitability for Best Timeframe:** {best_profitability:.2f}%")

# Run the app
if __name__ == "__main__":
    main()
    
