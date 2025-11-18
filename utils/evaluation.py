import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

    def compare_models(self):
        """Compare all trained models"""
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)

        print("\nModel Comparison:")
        print(results_df)

        # Sort by RMSE (lower is better)
        best_model = results_df['RMSE'].idxmin()
        print(f"\nBest performing model: {best_model} (RMSE: {results_df.loc[best_model, 'RMSE']:.4f})")

        return results_df
