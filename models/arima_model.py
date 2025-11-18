import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class ARIMAPredictor:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, train_series):
        """Fit ARIMA model"""
        self.model = ARIMA(train_series, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps):
        """Make predictions"""
        if self.model_fit is None:
            raise ValueError("Model not fitted yet")

        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def evaluate(self, train_series, test_series):
        """Train and evaluate ARIMA model"""
        # Fit model
        self.fit(train_series)

        # Make predictions
        predictions = self.predict(len(test_series))

        # Calculate actual metrics
        mse = mean_squared_error(test_series, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_series, predictions)
        r2 = r2_score(test_series, predictions)
        mape = np.mean(np.abs((test_series - predictions) / test_series)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        return metrics, predictions.values

    def save_model(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_fit, f)
        print(f"ARIMA model saved to {filepath}")
