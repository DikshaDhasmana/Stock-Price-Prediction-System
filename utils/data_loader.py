"""
utils/data_loader.py - yfinance-based stock data loader

Provides StockDataLoader with:
- fetch_stock_data(symbol, period='2y')
- load_data(symbol)
- save_data(df, symbol)
"""

import os
import pandas as pd
import yfinance as yf
from typing import Optional

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


class StockDataLoader:
    def __init__(self, raw_dir: str = RAW_DIR):
        self.raw_dir = raw_dir

    def raw_path(self, symbol: str) -> str:
        return f"{self.raw_dir}/{symbol.upper()}.csv"

    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV using yfinance. period examples: '1y','2y','5y','max'
        """
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return None
            # Ensure datetime index and drop NA
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"❌ yfinance fetch failed for {symbol}: {e}")
            return None

    def save_data(self, df: pd.DataFrame, symbol: str):
        path = self.raw_path(symbol)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"✓ Raw data saved to {path}")

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self.raw_path(symbol)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"❌ Failed to load raw data from {path}: {e}")
            return None
