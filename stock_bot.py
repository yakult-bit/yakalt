import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from pandas.tseries.offsets import BDay
import requests
import re
from fuzzywuzzy import fuzz

# Finnhub API Key (Replace with your actual API key)
FINNHUB_API_KEY = "cutcql1r01qrsirm2u80cutcql1r01qrsirm2u8g"

# Function to fetch market data from Finnhub
def fetch_finnhub_data(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Could not decode JSON response from Finnhub.")
        return None
    
    if "c" in data:
        return {
            "Current Price": data["c"],
            "High Price": data["h"],
            "Low Price": data["l"],
            "Open Price": data["o"],
            "Previous Close": data["pc"],
            "Volume": data.get("v", "N/A")
        }
    else:
        print("Warning: Failed to fetch data from Finnhub.")
        return None

# Function to predict stock prices
def predict_prices(yahoo_data, target_days=5):
    features = yahoo_data[['SMA_5', 'SMA_20', 'RSI']].values[:-target_days]
    targets_close = [yahoo_data['Close'].iloc[i+1:i+1+target_days].values for i in range(len(yahoo_data) - target_days)]
    targets_open = [yahoo_data['Open'].iloc[i+1:i+1+target_days].values for i in range(len(yahoo_data) - target_days)]
    
    targets_close = np.array(targets_close)
    targets_open = np.array(targets_open)
    
    model_close = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model_close.fit(features, targets_close)
    
    model_open = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model_open.fit(features, targets_open)
    
    last_features = yahoo_data[['SMA_5', 'SMA_20', 'RSI']].iloc[-1].values.reshape(1, -1)
    weekly_prediction_close = model_close.predict(last_features)[0]
    weekly_prediction_open = model_open.predict(last_features)[0]
    
    future_dates = [yahoo_data.index[-1] + BDay(i) for i in range(1, target_days+1)]
    
    predictions_output = "Predicted prices:\nDate       |   Open   |  Close\n--------------------------------\n"
    for d, op, cl in zip(future_dates, weekly_prediction_open, weekly_prediction_close):
        predictions_output += f"{d.date()} | ${op:.2f} | ${cl:.2f}\n"
    
    return predictions_output

# Function for AI-powered stock assistant
def stock_assistant(question, stock_symbol, data, predictions):
    question = question.lower()
    
    if "trend" in question:
        return f"The current trend for {stock_symbol} is {data['trend']} based on recent price movement."
    elif "support" in question or "resistance" in question:
        return (f"Short-Term Support: ${data['short_term_support']:.2f}, Short-Term Resistance: ${data['short_term_resistance']:.2f}\n"
                f"Long-Term Support: ${data['long_term_support']:.2f}, Long-Term Resistance: ${data['long_term_resistance']:.2f}")
    elif "price" in question or "current" in question:
        return (f"Current Price: ${data['current_price']:.2f}\n"
                f"Open Price: ${data['open_price']:.2f}, High: ${data['high_price']:.2f}, Low: ${data['low_price']:.2f}")
    elif "entry" in question:
        entry_reason = ("An ideal entry point is typically near a support level because it represents a price where buying interest has been strong historically. "
                        "However, entering at this support level may not be realistic if the stock is trading significantly higher. Instead, consider a pullback to a key moving average like the 200-day MA "
                        "or a retest of a minor support closer to the current price.")
        return f"Based on support and resistance levels, consider entering near ${data['short_term_support']:.2f} if it's an uptrend.\n\n{entry_reason}"
    elif "exit" in question:
        return f"If you're holding, consider exiting near ${data['short_term_resistance']:.2f} in an uptrend."
    elif "predict" in question or "future price" in question:
        return predictions
    elif "why" in question or "explain" in question:
        return "The suggestion is based on historical price movements, support/resistance levels, and trend direction. If you need further clarification, ask about support, resistance, or price trends."
    else:
        return "Sorry, I can't answer that question. Try asking about trend, support, resistance, entry, exit, or price."

# Prompt for stock ticker
while True:
    stock_symbol = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not stock_symbol.isalpha():
        print("Invalid stock ticker. Please enter a valid stock symbol (e.g., AAPL, NVDA).")
        continue
    
    print(f"\nFetching data for {stock_symbol}...\n")
    finnhub_data = fetch_finnhub_data(stock_symbol)
    yahoo_data = yf.download(stock_symbol, period="2y", interval="1d", auto_adjust=False)
    
    if finnhub_data and not yahoo_data.empty:
        break
    else:
        print("Error: Could not retrieve data. Please check the stock symbol or API limits.")

# --- Yahoo Finance Data Processing ---
yahoo_data.reset_index(inplace=True)
yahoo_data.columns = yahoo_data.columns.get_level_values(0)
yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'])
yahoo_data.set_index('Date', inplace=True)
yahoo_data.dropna(inplace=True)

# --- Predict Prices ---
predictions_output = predict_prices(yahoo_data)

# --- Display stock analysis data ---
print(f"\nStock Analysis for {stock_symbol}:")
print(f"Current Price: ${finnhub_data['Current Price']:.2f}")
print(predictions_output)

# --- AI Assistant Loop ---
while True:
    user_question = input("\nAsk the stock assistant a question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        break
    print(stock_assistant(user_question, stock_symbol, {
        "trend": "Uptrend" if yahoo_data['Close'].iloc[-1] > yahoo_data['Close'].iloc[-2] else "Downtrend",
        "short_term_support": yahoo_data['Low'].rolling(window=20).min().iloc[-1],
        "short_term_resistance": yahoo_data['High'].rolling(window=20).max().iloc[-1],
        "long_term_support": yahoo_data['Low_52wk'].iloc[-1],
        "long_term_resistance": yahoo_data['High_52wk'].iloc[-1],
        "current_price": finnhub_data['Current Price']
    }, predictions_output))
