import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class EnsemblePredictor:
    def __init__(self, meta_model_type='ridge'):
        self.meta_model_type = meta_model_type
        self.meta_model = None
        self.base_model_names = []

    def train_meta_learner(self, base_predictions, y_true):
        """Train meta-learner using base model predictions"""
        if self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=0.1)
        else:
            # Default to Ridge
            self.meta_model = Ridge(alpha=0.1)

        # Train meta-learner
        self.meta_model.fit(base_predictions, y_true)

    def predict(self, base_predictions):
        """Make ensemble predictions"""
        return self.meta_model.predict(base_predictions)

    def evaluate(self, base_predictions, y_true):
        """Evaluate ensemble performance"""
        predictions = self.predict(base_predictions)

        # Calculate actual metrics
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        return metrics, predictions

    def save_model(self, filepath):
        """Save ensemble model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.meta_model, f)
        print(f"Ensemble model saved to {filepath}")
