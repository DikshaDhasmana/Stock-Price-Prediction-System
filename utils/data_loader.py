"""
utils/data_loader.py - Improved yfinance loader with:
- Custom User-Agent (prevents 429 rate limit)
- HTTP request status printing
- Retry logic
- Cleaned response handling
"""

import os
import time
import pandas as pd
import yfinance as yf
import requests
from typing import Optional

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


class StockDataLoader:
    def __init__(self, raw_dir: str = RAW_DIR):
        self.raw_dir = raw_dir

        # Create custom session for yfinance (avoids Yahoo 429)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })

    def raw_path(self, symbol: str) -> str:
        return f"{self.raw_dir}/{symbol.upper()}.csv"

    # ---------------------------------------------------------
    # 🔵 HEALTH CHECK (makes a HEAD request to Yahoo endpoint)
    # ---------------------------------------------------------
    def print_request_status(self, symbol: str):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        try:
            res = self.session.get(url, timeout=5)
            print(f"HTTP Status for {symbol}: {res.status_code}")
            return res.status_code
        except Exception as e:
            print(f"Request check failed: {e}")
            return None

    # ---------------------------------------------------------
    def fetch_stock_data(self, symbol: str, period: str = "2y",interval: str = "1d") -> Optional[pd.DataFrame]:

        print(f"\n====== Fetching {symbol} from Yahoo Finance ======")

        for attempt in range(1, 3 + 1):  # Retry 3 times
            print(f"\nAttempt {attempt}/3")

        # Print Yahoo status (your existing function)
            status = self.print_request_status(symbol)

            if status == 429:
                print("❌ Rate limited (429). Waiting 3 sec...")
                time.sleep(3)
                continue

            elif status is not None and status >= 500:
                print("❌ Server error at Yahoo. Retrying in 3 sec...")
                time.sleep(3)
                continue

            try:
            # IMPORTANT: yfinance MUST NOT receive a custom session
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

            # Validate response
                if df is None or df.empty:
                    print("⚠️ Empty response from Yahoo. Retrying in 2s...")
                    time.sleep(2)
                    continue

                df = df.dropna()
                df.index = pd.to_datetime(df.index)

                print("✓ Successfully fetched stock data!")
                return df

            except Exception as e:
                print(f"❌ yfinance fetch failed: {e}")
                print("Retrying in 2 sec...")
                time.sleep(2)

        print("❌ Failed after 3 attempts — Yahoo API issue or rate limit.")
        return None

    # ---------------------------------------------------------
    def save_data(self, df: pd.DataFrame, symbol: str):
        path = self.raw_path(symbol)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"✓ Raw data saved to {path}")

    # ---------------------------------------------------------
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
