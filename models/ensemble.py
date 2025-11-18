import numpy as np
import pickle
import os
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EnsemblePredictor:
    """
    Time-series safe stacking ensemble for regression.

    Usage:
      - predictions_dict: dict[name -> 1D np.array] (model predictions)
      - y_true: 1D np.array of true target values (aligned to chronological order)
      - call train_and_evaluate(...) to train meta-learner and get holdout results
    """

    def __init__(self, meta_model_type='ridge', use_timeseries_cv=True, ts_splits=5, random_state=42):
        self.meta_model_type = meta_model_type.lower()
        self.use_timeseries_cv = use_timeseries_cv
        self.ts_splits = ts_splits
        self.random_state = random_state

        self.meta_model = None
        self.base_model_names = []
        self.meta_model_params = None

    # -------------------------
    # Utilities
    # -------------------------
    def _validate_inputs(self, predictions_dict, y_true):
        if not isinstance(predictions_dict, dict) or len(predictions_dict) == 0:
            raise ValueError("predictions_dict must be a non-empty dict of name->1D-array")
        for k, v in predictions_dict.items():
            arr = np.asarray(v)
            if arr.ndim != 1:
                raise ValueError(f"All prediction arrays must be 1D. '{k}' has shape {arr.shape}")
        y_arr = np.asarray(y_true)
        if y_arr.ndim != 1:
            raise ValueError("y_true must be a 1D array")
        return {k: np.asarray(v) for k, v in predictions_dict.items()}, y_arr

    # -------------------------
    # Alignment
    # -------------------------
    def align_predictions(self, predictions_dict, y_true):
        """
        Align predictions from different models to the earliest common start.
        This handles models (e.g. LSTM) that produce shorter prediction arrays
        because they need sequences.
        Returns: (aligned_predictions_dict, y_true_aligned)
        """
        preds, y_true = self._validate_inputs(predictions_dict, y_true)

        # lengths and earliest start index for each pred relative to its end
        lengths = {name: len(arr) for name, arr in preds.items()}
        target_len = len(y_true)

        # Determine min length among all arrays (we can only evaluate where all models + truth exist)
        min_len = min(list(lengths.values()) + [target_len])
        if min_len <= 0:
            raise ValueError("After alignment there is no overlapping data (min length <= 0).")

        # Align by taking the last `min_len` values of each array (keeps chronology)
        aligned = {name: arr[-min_len:] for name, arr in preds.items()}
        y_aligned = y_true[-min_len:]

        self.base_model_names = list(aligned.keys())
        return aligned, y_aligned

    # -------------------------
    # Prepare meta-training / holdout (time-series safe)
    # -------------------------
    def prepare_ensemble_data(self, predictions_dict, y_true, holdout_ratio=0.3):
        """
        Returns: X_meta_train, X_meta_holdout, y_meta_train, y_meta_holdout
        Uses index-based split (no shuffle) to preserve time ordering.
        """
        aligned_preds, y_aligned = self.align_predictions(predictions_dict, y_true)
        base_predictions = np.column_stack([aligned_preds[name] for name in self.base_model_names])

        n = base_predictions.shape[0]
        if not (0.0 < holdout_ratio < 1.0):
            raise ValueError("holdout_ratio must be between 0 and 1")

        holdout_size = int(np.ceil(n * holdout_ratio))
        train_size = n - holdout_size
        if train_size < 1:
            raise ValueError("Not enough data after holdout split. Reduce holdout_ratio or provide more predictions.")

        X_train = base_predictions[:train_size]
        X_holdout = base_predictions[train_size:]
        y_train = y_aligned[:train_size]
        y_holdout = y_aligned[train_size:]

        return X_train, X_holdout, y_train, y_holdout

    # -------------------------
    # Meta learner selection & training
    # -------------------------
    def _init_meta_model(self):
        t = self.meta_model_type
        if t == 'ridge':
            return Ridge()
        elif t == 'linear':
            return LinearRegression()
        elif t == 'lasso':
            return Lasso(max_iter=5000)
        elif t == 'elasticnet':
            return ElasticNet(max_iter=5000)
        else:
            raise ValueError(f"Unsupported meta_model_type: {t}")

    def train_meta_learner(self, X_meta_train, y_meta_train, cv_tune=True):
        """
        Train meta-learner. If cv_tune=True and use_timeseries_cv=True,
        use TimeSeriesSplit-based grid search to find best alpha/lambda (where applicable).
        """
        # choose base estimator
        estimator = self._init_meta_model()

        # default simple parameter grid for regularized models
        param_grid = None
        if isinstance(estimator, Ridge):
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
        elif isinstance(estimator, Lasso):
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0]}
        elif isinstance(estimator, ElasticNet):
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
        else:
            param_grid = None

        if cv_tune and param_grid is not None and self.use_timeseries_cv:
            tscv = TimeSeriesSplit(n_splits=max(2, min(self.ts_splits, max(2, X_meta_train.shape[0] // 10))))
            gsearch = GridSearchCV(estimator, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            gsearch.fit(X_meta_train, y_meta_train)
            self.meta_model = gsearch.best_estimator_
            self.meta_model_params = gsearch.best_params_
        else:
            # fallback: train without CV
            estimator.fit(X_meta_train, y_meta_train)
            self.meta_model = estimator
            self.meta_model_params = getattr(estimator, 'get_params', lambda: {})()

        return self.meta_model

    # -------------------------
    # Predict & evaluate
    # -------------------------
    def predict(self, base_predictions):
        """
        base_predictions: either dict[name->1D] or 2D-array with columns in same
        order as self.base_model_names
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet")

        if isinstance(base_predictions, dict):
            # ensure same order
            X = np.column_stack([base_predictions[name] for name in self.base_model_names])
        else:
            X = np.asarray(base_predictions)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        return self.meta_model.predict(X)

    def evaluate(self, base_predictions, y_true):
        preds = self.predict(base_predictions)
        y = np.asarray(y_true)

        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        mask = y != 0
        mape = (np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100) if mask.sum() > 0 else np.nan

        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        return metrics, preds

    # -------------------------
    # Full pipeline
    # -------------------------
    def train_and_evaluate(self, predictions_dict, y_true, holdout_ratio=0.3, cv_tune=True):
        """
        Full pipeline:
         - align predictions
         - prepare meta training/holdout (time ordered)
         - train meta-learner (optionally CV tune with TimeSeriesSplit)
         - evaluate on holdout and return results & per-base-model metrics
        """
        X_train, X_holdout, y_train, y_holdout = self.prepare_ensemble_data(predictions_dict, y_true, holdout_ratio)
        self.train_meta_learner(X_train, y_train, cv_tune=cv_tune)

        train_metrics, _ = self.evaluate(X_train, y_train)
        holdout_metrics, holdout_preds = self.evaluate(X_holdout, y_holdout)

        # Compute individual base-model holdout metrics for comparison
        per_model_metrics = {}
        for i, name in enumerate(self.base_model_names):
            base_preds_holdout = X_holdout[:, i]
            mse_i = mean_squared_error(y_holdout, base_preds_holdout)
            per_model_metrics[name] = {
                'MSE': mse_i,
                'RMSE': np.sqrt(mse_i),
                'MAE': mean_absolute_error(y_holdout, base_preds_holdout),
                'R2': r2_score(y_holdout, base_preds_holdout)
            }

        return {
            'meta_train_metrics': train_metrics,
            'meta_holdout_metrics': holdout_metrics,
            'holdout_predictions': holdout_preds,
            'per_model_holdout_metrics': per_model_metrics,
            'meta_model': self.meta_model,
            'meta_model_params': self.meta_model_params
        }

    # -------------------------
    # Persistence
    # -------------------------
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'meta_model': self.meta_model,
            'base_model_names': self.base_model_names,
            'meta_model_type': self.meta_model_type,
            'meta_model_params': self.meta_model_params
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"✓ Ensemble saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.meta_model = payload['meta_model']
        self.base_model_names = payload['base_model_names']
        self.meta_model_type = payload.get('meta_model_type', self.meta_model_type)
        self.meta_model_params = payload.get('meta_model_params', None)
        print(f"✓ Ensemble loaded from {filepath}. Base models: {self.base_model_names}")
