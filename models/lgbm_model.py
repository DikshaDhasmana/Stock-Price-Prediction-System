import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os

class LightGBMPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None

    def prepare_data(self, df):
        """Auto-select lag + technical features"""

        feature_cols = [c for c in df.columns 
                        if ('Lag' in c) or ('MA' in c) or ('RSI' in c)
                        or ('MACD' in c) or ('Volatility' in c)]

        # fallback
        if not feature_cols:
            feature_cols = ['Close']

        self.feature_cols = feature_cols

        X = df[feature_cols]
        y = df['Target']

        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM for time-series"""

        train_data = lgb.Dataset(X_train, y_train)
        valid_data = lgb.Dataset(X_val, y_val)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 127,
            'max_depth': -1,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'verbose': -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,        # much better for time series
            valid_sets=[train_data, valid_data],
            valid_names=['train','valid'],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        }, preds

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
