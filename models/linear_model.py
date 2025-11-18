import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class LinearPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols = None

    def prepare_data(self, df):
        """Prepare data for linear regression"""
        # Use lag features and technical indicators as predictors
        feature_cols = [col for col in df.columns if 'Lag' in col or 'MA' in col or 'Volatility' in col]
        if not feature_cols:
            # Fallback to basic features
            feature_cols = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']

        self.feature_cols = feature_cols
        X = df[feature_cols]
        y = df['Target']

        return X, y

    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train linear regression model"""
        if tune_hyperparameters:
            # Simple grid search (though LinearRegression has few params)
            param_grid = {'fit_intercept': [True, False]}
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)

        # Calculate actual metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        return metrics, predictions

    def save_model(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Linear regression model saved to {filepath}")
