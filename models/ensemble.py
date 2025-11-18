import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class EnsemblePredictor:
    def __init__(self, meta_model_type='ridge'):
        """
        CORRECTED Ensemble Model
        
        Key improvements:
        1. Handles predictions of different lengths
        2. Uses separate holdout set for meta-learner
        3. Better error handling
        """
        self.meta_model_type = meta_model_type
        self.meta_model = None
        self.base_model_names = []

    def align_predictions(self, predictions_dict, y_true):
        """
        Align predictions from different models to same length
        
        CORRECTION: Handles LSTM predictions that are shorter due to sequences
        """
        # Find minimum length
        lengths = [len(pred) for pred in predictions_dict.values()]
        lengths.append(len(y_true))
        min_length = min(lengths)
        
        # Truncate all to same length
        aligned_predictions = {}
        for name, preds in predictions_dict.items():
            aligned_predictions[name] = preds[-min_length:]  # Take last N predictions
        
        y_true_aligned = y_true[-min_length:]
        
        print(f"✓ Predictions aligned to length: {min_length}")
        
        return aligned_predictions, y_true_aligned

    def prepare_ensemble_data(self, predictions_dict, y_true, 
                             holdout_ratio=0.3, random_state=42):
        """
        Prepare data for ensemble training with proper holdout
        
        CORRECTION: Uses separate holdout set for meta-learner evaluation
        This prevents overfitting to test set
        """
        # Align predictions
        aligned_predictions, y_true_aligned = self.align_predictions(
            predictions_dict, y_true
        )
        
        # Stack predictions
        self.base_model_names = list(aligned_predictions.keys())
        base_predictions = np.column_stack([
            aligned_predictions[name] for name in self.base_model_names
        ])
        
        # Split into train and holdout for meta-learner
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            base_predictions,
            y_true_aligned,
            test_size=holdout_ratio,
            random_state=random_state,
            shuffle=False  # Keep time series order
        )
        
        print(f"✓ Ensemble data prepared:")
        print(f"  Base models: {len(self.base_model_names)}")
        print(f"  Meta-train samples: {len(X_train)}")
        print(f"  Meta-holdout samples: {len(X_holdout)}")
        
        return X_train, X_holdout, y_train, y_holdout

    def train_meta_learner(self, base_predictions, y_true):
        """
        Train meta-learner using base model predictions
        
        CORRECTION: Now supports different meta-learners
        """
        if self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif self.meta_model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
        else:
            # Default to Ridge
            self.meta_model = Ridge(alpha=1.0)
        
        # Train meta-learner
        self.meta_model.fit(base_predictions, y_true)
        
        # Print weights
        if hasattr(self.meta_model, 'coef_'):
            print(f"\n✓ Meta-learner weights:")
            for name, weight in zip(self.base_model_names, self.meta_model.coef_):
                print(f"  {name}: {weight:.4f}")

    def predict(self, base_predictions):
        """Make ensemble predictions"""
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained")
        
        # Ensure correct shape
        if isinstance(base_predictions, dict):
            base_predictions = np.column_stack([
                base_predictions[name] for name in self.base_model_names
            ])
        
        return self.meta_model.predict(base_predictions)

    def evaluate(self, base_predictions, y_true):
        """
        Evaluate ensemble performance
        
        CORRECTION: Better error handling for metrics
        """
        predictions = self.predict(base_predictions)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        # MAPE - handle edge cases
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, predictions

    def train_and_evaluate(self, predictions_dict, y_true, 
                          holdout_ratio=0.3, random_state=42):
        """
        Complete training and evaluation pipeline
        
        This is the recommended way to use the ensemble
        """
        # Prepare data with proper holdout
        X_train, X_holdout, y_train, y_holdout = self.prepare_ensemble_data(
            predictions_dict, y_true, holdout_ratio, random_state
        )
        
        # Train meta-learner
        self.train_meta_learner(X_train, y_train)
        
        # Evaluate on training set
        train_metrics, _ = self.evaluate(X_train, y_train)
        print(f"\n✓ Meta-learner training performance:")
        print(f"  RMSE: {train_metrics['RMSE']:.4f}")
        print(f"  R²: {train_metrics['R2']:.4f}")
        
        # Evaluate on holdout set
        holdout_metrics, holdout_preds = self.evaluate(X_holdout, y_holdout)
        print(f"\n✓ Meta-learner holdout performance:")
        print(f"  RMSE: {holdout_metrics['RMSE']:.4f}")
        print(f"  R²: {holdout_metrics['R2']:.4f}")
        
        # Compare to individual models
        print(f"\n✓ Individual model performance on holdout:")
        for i, name in enumerate(self.base_model_names):
            rmse = np.sqrt(mean_squared_error(y_holdout, X_holdout[:, i]))
            print(f"  {name}: RMSE = {rmse:.4f}")
        
        return holdout_metrics, holdout_preds

    def save_model(self, filepath):
        """Save ensemble model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save meta-model and configuration
        ensemble_data = {
            'meta_model': self.meta_model,
            'base_model_names': self.base_model_names,
            'meta_model_type': self.meta_model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"✓ Ensemble model saved to {filepath}")

    def load_model(self, filepath):
        """Load ensemble model from file"""
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.meta_model = ensemble_data['meta_model']
        self.base_model_names = ensemble_data['base_model_names']
        self.meta_model_type = ensemble_data['meta_model_type']
        
        print(f"✓ Ensemble model loaded from {filepath}")
        print(f"  Base models: {self.base_model_names}")