import pandas as pd
import yfinance as yf
import os

class StockDataLoader:
    def __init__(self):
        pass

    def fetch_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def load_data(self, symbol):
        """Load data from CSV file"""
        filepath = f'data/raw/{symbol}.csv'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        else:
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def save_data(self, df, symbol):
        """Save data to CSV file"""
        os.makedirs('data/raw', exist_ok=True)
        filepath = f'data/raw/{symbol}.csv'
        if df is not None and not df.empty:
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")
        else:
            print(f"No data to save for {symbol}")
