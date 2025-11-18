import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class ARIMAPredictor:
    def __init__(self):
        self.model = None

    def fit(self, train_series):
        """Fit auto-ARIMA with log transform"""
        # Apply log transform for stationarity
        self.train_mean = train_series.mean()
        log_series = np.log(train_series)

        # Fit auto ARIMA
        self.model = auto_arima(
            log_series,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

    def predict(self, steps, last_train_value):
        """Generate rolling forecast and reverse transform"""
        preds = []

        current_log = np.log(last_train_value)

        for _ in range(steps):
            next_log = self.model.predict(n_periods=1)[0]
            next_val = np.exp(next_log)  # Reverse log
            preds.append(next_val)

            # Update model with new value for next prediction
            self.model.update(next_log)

        return np.array(preds)

    def evaluate(self, train_series, test_series):
        """Train and evaluate ARIMA properly"""
        self.fit(train_series)

        last_training_value = train_series.iloc[-1]

        predictions = self.predict(
            steps=len(test_series),
            last_train_value=last_training_value
        )

        # Compute metrics
        mse = mean_squared_error(test_series, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_series, predictions)
        r2 = r2_score(test_series, predictions)
        mape = np.mean(np.abs((test_series - predictions) / test_series)) * 100

        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        }

        return metrics, predictions

    def save_model(self, filepath):
        """Save ARIMA model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Auto-ARIMA model saved at {filepath}")
