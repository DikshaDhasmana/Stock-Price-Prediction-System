"""
MODIFIED train_all_models.py - Now works with ANY stock symbol

Changes made to support any stock:
1. Symbol parameter instead of hardcoded 'AAPL'
2. Dynamic file paths based on symbol
3. Flexible data loading for any ticker
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.data_loader import StockDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.evaluation import ModelEvaluator

# Import models
from models.arima_model import ARIMAPredictor
from models.linear_model import LinearPredictor
from models.lstm_model import LSTMPredictor
from models.lgbm_model import LightGBMPredictor
from models.ensemble import EnsemblePredictor

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/predictions',
        'models/saved_models',
        'models/saved_models/arima_models',
        'results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Directories created")

def load_and_prepare_data(symbol, period='2y'):
    """
    Load data WITHOUT normalization for ANY stock symbol
    """
    print(f"\n{'='*80}")
    print(f"LOADING AND PREPARING DATA FOR {symbol}")
    print(f"{'='*80}")
    
    # Load data
    loader = StockDataLoader()
    
    # Check if data exists, otherwise fetch
    if os.path.exists(f'data/raw/{symbol}.csv'):
        print(f"Loading existing data for {symbol}...")
        try:
            df = loader.load_data(symbol)
            # Check if data is recent (within last 7 days)
            if not df.empty and df.index[-1] < pd.Timestamp.now() - pd.Timedelta(days=7):
                print(f"  ⚠️  Data is outdated, fetching fresh data...")
                df = loader.fetch_stock_data(symbol, period=period)
                if df is not None and not df.empty:
                    loader.save_data(df, symbol)
        except Exception as e:
            print(f"  ⚠️  Error loading cached data: {e}")
            print(f"  Fetching fresh data...")
            df = loader.fetch_stock_data(symbol, period=period)
            if df is not None and not df.empty:
                loader.save_data(df, symbol)
    else:
        print(f"Fetching new data for {symbol}...")
        df = loader.fetch_stock_data(symbol, period=period)
        if df is not None and not df.empty:
            loader.save_data(df, symbol)
        else:
            raise ValueError(
                f"❌ No data available for symbol {symbol}\n"
                f"   Please verify:\n"
                f"   1. Symbol is correct (try: AAPL, MSFT, GOOGL, AMZN)\n"
                f"   2. You have internet connection\n"
                f"   3. Symbol is actively traded\n"
                f"   4. Try a different period (--period 5y or --period max)"
            )
    
    # Validate data
    if df is None or df.empty:
        raise ValueError(f"❌ Failed to load data for {symbol}")
    
    print(f"✓ Loaded {len(df)} data points from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Create features WITHOUT normalization
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df, n_lags=5, horizon=1)
    
    print(f"✓ Features created: {len(features_df)} samples, {len(features_df.columns)} features")
    
    return features_df, engineer

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    print(f"\n✓ Data split (BEFORE normalization):")
    print(f"  Training: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_df)} samples ({val_ratio*100:.0f}%)")
    print(f"  Testing: {len(test_df)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_df, val_df, test_df

def normalize_splits(train_df, val_df, test_df, engineer):
    """Normalize data AFTER splitting"""
    print(f"\n{'='*80}")
    print("NORMALIZING DATA (No Data Leakage)")
    print(f"{'='*80}")
    
    # Fit scalers on training data only
    engineer.fit_scalers(train_df)
    
    # Transform all splits using training scalers
    train_scaled = engineer.transform_features(train_df)
    val_scaled = engineer.transform_features(val_df)
    test_scaled = engineer.transform_features(test_df)
    
    print("✓ All splits normalized using training statistics only")
    
    return train_scaled, val_scaled, test_scaled

def prepare_model_data(train_df, val_df, test_df, feature_cols):
    """Prepare X, y for supervised models"""
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['Target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Target']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_arima(train_df, test_df, engineer, symbol):
    """Train ARIMA model on ORIGINAL scale"""
    print(f"\n{'='*80}")
    print(f"TRAINING ARIMA MODEL")
    print(f"{'='*80}")
    
    train_series = train_df['Target']
    test_series = test_df['Target']
    
    # Train
    arima = ARIMAPredictor()
    metrics, predictions = arima.evaluate(train_series, test_series)
    
    predictions_original = predictions
    
    # Save model with symbol-specific name
    arima.save_model(f'models/saved_models/arima_models/{symbol}_arima.pkl')
    
    print(f"✓ ARIMA trained and evaluated")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return metrics, predictions_original

def train_linear_regression(train_df, val_df, test_df, engineer, symbol):
    """Train Linear Regression model"""
    print(f"\n{'='*80}")
    print(f"TRAINING LINEAR REGRESSION MODEL")
    print(f"{'='*80}")
    
    feature_cols = [col for col in train_df.columns if col != 'Target']
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(
        train_df, val_df, test_df, feature_cols
    )
    
    lr = LinearPredictor()
    lr.feature_cols = feature_cols
    lr.train(X_train, y_train)
    
    metrics_normalized, predictions_normalized = lr.evaluate(X_test, y_test)
    
    predictions_original = engineer.inverse_transform_target(predictions_normalized)
    y_test_original = engineer.inverse_transform_target(y_test.values)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    lr.save_model(f'models/saved_models/lr_model_{symbol}.pkl')
    
    print(f"✓ Linear Regression trained and evaluated")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return metrics, predictions_original, (X_train, y_train, X_val, y_val, X_test, y_test)

def train_lstm(train_df, val_df, test_df, engineer, symbol):
    """Train LSTM model using ALL features"""
    print(f"\n{'='*80}")
    print(f"TRAINING LSTM MODEL")
    print(f"{'='*80}")
    
    feature_cols = [col for col in train_df.columns if col != 'Target']
    
    lstm = LSTMPredictor(
        sequence_length=10, 
        units=50, 
        layers=2,
        dropout=0.2,
        n_features=len(feature_cols)
    )
    
    X_train, y_train = lstm.prepare_sequences(train_df, feature_cols)
    X_val, y_val = lstm.prepare_sequences(val_df, feature_cols)
    X_test, y_test = lstm.prepare_sequences(test_df, feature_cols)
    
    print(f"✓ Sequences prepared:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        model_path=f'models/saved_models/lstm_model_{symbol}.h5'
    )
    
    metrics_normalized, predictions_normalized = lstm.evaluate(X_test, y_test)
    
    predictions_original = engineer.inverse_transform_target(predictions_normalized)
    y_test_original = engineer.inverse_transform_target(y_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"✓ LSTM trained and evaluated")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return metrics, predictions_original

def train_lightgbm(train_df, val_df, test_df, engineer, symbol):
    """Train LightGBM model"""
    print(f"\n{'='*80}")
    print(f"TRAINING LIGHTGBM MODEL")
    print(f"{'='*80}")
    
    feature_cols = [col for col in train_df.columns if col != 'Target']
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(
        train_df, val_df, test_df, feature_cols
    )
    
    lgbm = LightGBMPredictor()
    lgbm.train(X_train, y_train, X_val, y_val, 
               num_boost_round=500, early_stopping_rounds=50)
    
    metrics_normalized, predictions_normalized = lgbm.evaluate(X_test, y_test)
    
    predictions_original = engineer.inverse_transform_target(predictions_normalized)
    y_test_original = engineer.inverse_transform_target(y_test.values)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    lgbm.save_model(f'models/saved_models/lgbm_model_{symbol}.pkl')
    
    print(f"✓ LightGBM trained and evaluated")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return metrics, predictions_original

def create_baseline(test_df, engineer):
    """Create a simple baseline: tomorrow's price = today's price"""
    y_test_original = engineer.inverse_transform_target(test_df['Target'].values)
    
    baseline_preds = test_df['Target'].shift(1).fillna(method='bfill').values
    baseline_preds = engineer.inverse_transform_target(baseline_preds)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test_original, baseline_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, baseline_preds)
    r2 = r2_score(y_test_original, baseline_preds)
    mape = np.mean(np.abs((y_test_original - baseline_preds) / y_test_original)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    return metrics, baseline_preds

def main(symbol='AAPL', period='2y'):
    """
    FLEXIBLE training pipeline for ANY stock symbol
    
    Args:
        symbol: Stock ticker (e.g., 'TSLA', 'GOOGL', 'MSFT', 'AMZN')
        period: Time period ('1y', '2y', '5y', 'max')
    """
    start_time = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"# STOCK PRICE PREDICTION PIPELINE")
    print(f"# Symbol: {symbol}")
    print(f"# Period: {period}")
    print(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    # Create directories
    create_directories()
    
    # Load and prepare data
    features_df, engineer = load_and_prepare_data(symbol, period)
    
    # Split data BEFORE normalization
    train_df_raw, val_df_raw, test_df_raw = split_data(features_df)
    
    # Normalize using training statistics only
    train_df, val_df, test_df = normalize_splits(
        train_df_raw, val_df_raw, test_df_raw, engineer
    )
    
    # Store all predictions and metrics
    all_predictions = {}
    all_metrics = {}
    
    # Create baseline
    print(f"\n{'='*80}")
    print("CREATING BASELINE (Tomorrow = Today)")
    print(f"{'='*80}")
    baseline_metrics, baseline_preds = create_baseline(test_df, engineer)
    all_metrics['Baseline'] = baseline_metrics
    all_predictions['Baseline'] = baseline_preds
    print(f"✓ Baseline RMSE: {baseline_metrics['RMSE']:.4f}")
    
    # Train models
    try:
        arima_metrics, arima_preds = train_arima(
            train_df_raw, test_df_raw, engineer, symbol
        )
        all_metrics['ARIMA'] = arima_metrics
        all_predictions['ARIMA'] = arima_preds
    except Exception as e:
        print(f"✗ ARIMA training failed: {str(e)}")
    
    try:
        lr_metrics, lr_preds, _ = train_linear_regression(
            train_df, val_df, test_df, engineer, symbol
        )
        all_metrics['Linear Regression'] = lr_metrics
        all_predictions['Linear Regression'] = lr_preds
    except Exception as e:
        print(f"✗ Linear Regression training failed: {str(e)}")
    
    try:
        lstm_metrics, lstm_preds = train_lstm(
            train_df, val_df, test_df, engineer, symbol
        )
        all_metrics['LSTM'] = lstm_metrics
        all_predictions['LSTM'] = lstm_preds
    except Exception as e:
        print(f"✗ LSTM training failed: {str(e)}")
    
    try:
        lgbm_metrics, lgbm_preds = train_lightgbm(
            train_df, val_df, test_df, engineer, symbol
        )
        all_metrics['LightGBM'] = lgbm_metrics
        all_predictions['LightGBM'] = lgbm_preds
    except Exception as e:
        print(f"✗ LightGBM training failed: {str(e)}")
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"FINAL MODEL COMPARISON FOR {symbol} (Original Scale)")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)
    
    # Save results with symbol-specific filename
    comparison_df.to_csv(f'results/{symbol}_model_comparison.csv')
    print(f"\n✓ Results saved to results/{symbol}_model_comparison.csv")
    
    # Calculate total time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING COMPLETE FOR {symbol}")
    print(f"# Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"{'#'*80}\n")
    
    return all_metrics, all_predictions, comparison_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train stock prediction models for ANY symbol'
    )
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='AAPL',
        help='Stock symbol (e.g., AAPL, TSLA, GOOGL, MSFT, AMZN)'
    )
    parser.add_argument(
        '--period', 
        type=str, 
        default='2y',
        help='Time period (1y, 2y, 5y, max)'
    )
    
    args = parser.parse_args()
    
    # Run training for ANY stock
    metrics, predictions, comparison = main(
        symbol=args.symbol.upper(), 
        period=args.period
    )