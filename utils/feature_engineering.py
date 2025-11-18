import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class FeatureEngineer:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False

    def create_features(self, df, n_lags=5, horizon=1):
        """
        Create features WITHOUT normalization
        This should be called on the entire dataset before splitting
        """
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

        # Target variable (future price) - created BEFORE normalization
        df['Target'] = df['Close'].shift(-horizon)

        # Drop NaN values
        df.dropna(inplace=True)

        return df

    def fit_scalers(self, train_df):
        """
        Fit scalers on TRAINING data only
        This prevents data leakage
        """
        # Get feature columns (exclude target)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['Target']]
        
        # Fit feature scaler
        self.feature_scaler.fit(train_df[feature_cols])
        
        # Fit target scaler
        self.target_scaler.fit(train_df[['Target']])
        
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        print(f"✓ Scalers fitted on {len(train_df)} training samples")
        print(f"  Features: {len(feature_cols)} columns")

    def transform_features(self, df):
        """
        Transform features using fitted scaler
        """
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        df_scaled = df.copy()
        
        # Scale features
        df_scaled[self.feature_cols] = self.feature_scaler.transform(
            df[self.feature_cols]
        )
        
        # Scale target
        df_scaled['Target'] = self.target_scaler.transform(
            df[['Target']]
        )
        
        return df_scaled

    def inverse_transform_target(self, predictions):
        """
        Convert normalized predictions back to original scale
        """
        if not self.is_fitted:
            raise ValueError("Scalers not fitted.")
        
        # Reshape if necessary
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(predictions).flatten()

    def engineer_features(self, df, n_lags=5, horizon=1, normalize=True):
        """
        Legacy method for backward compatibility
        WARNING: This method should NOT be used for train/test splitting
        Use create_features() + fit_scalers() + transform_features() instead
        """
        print("⚠️  WARNING: Using legacy method. Consider using the new pipeline:")
        print("   1. create_features()")
        print("   2. split data")
        print("   3. fit_scalers() on train")
        print("   4. transform_features() on train/val/test")
        
        df = self.create_features(df, n_lags, horizon)
        
        if normalize:
            # This is the OLD buggy way - kept for compatibility
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.feature_scaler.fit_transform(df[numeric_cols])
            self.is_fitted = True
        
        return df

    def save_features(self, df, symbol):
        """Save engineered features to CSV"""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/{symbol}_features.csv'
        df.to_csv(filepath)
        print(f"✓ Features saved to {filepath}")

    def get_feature_names(self):
        """Get list of feature column names"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted.")
        return self.feature_cols