import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def engineer_features(self, df, n_lags=5, horizon=1, normalize=True):
        """Engineer features for stock prediction"""
        df = df.copy()

        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()

        # Lag features
        for lag in range(1, n_lags + 1):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)

        # Target variable (future price)
        df['Target'] = df['Close'].shift(-horizon)

        # Drop NaN values
        df.dropna(inplace=True)

        # Normalize if requested
        if normalize:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df

    def save_features(self, df, symbol):
        """Save engineered features to CSV"""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/{symbol}_features.csv'
        df.to_csv(filepath)
        print(f"Features saved to {filepath}")
