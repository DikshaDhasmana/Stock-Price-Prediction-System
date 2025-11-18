import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class LinearPredictor:
    def __init__(self, lags=5, mas=[5, 10, 20], add_volatility=True):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.lags = lags
        self.mas = mas
        self.add_volatility = add_volatility

    def engineer_features(self, df):
        """Automatically create lag features, MA indicators, volatility"""
        df = df.copy()

        # Create Lag features
        for lag in range(1, self.lags + 1):
            df[f"Lag_{lag}"] = df["Close"].shift(lag)

        # Moving Averages
        for m in self.mas:
            df[f"MA_{m}"] = df["Close"].rolling(m).mean()

        # Volatility (rolling standard deviation)
        if self.add_volatility:
            df["Volatility"] = df["Close"].rolling(10).std()

        # Target → Next-day Close price
        df["Target"] = df["Close"].shift(-1)

        df.dropna(inplace=True)
        return df

    def prepare_data(self, df):
        """Extract X and y, scale features"""
        df = df.copy()

        # Use engineered features
        feature_cols = [col for col in df.columns if col not in ["Target", "Close"]]
        self.feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["Target"].values.reshape(-1, 1)

        # Scale X (not y)
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train the linear regression model"""
        if tune_hyperparameters:
            param_grid = {'fit_intercept': [True, False]}
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error'
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict using model"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).flatten()

    def evaluate(self, X_test, y_test):
        """Compute regression metrics safely"""
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mask = y_test != 0
        mape = (
            np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
            if mask.sum() > 0 else np.nan
        )

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        }, predictions

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
