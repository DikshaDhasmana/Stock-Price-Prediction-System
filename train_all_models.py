
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
    Load and prepare data for training
    
    Args:
        symbol: Stock symbol
        period: Time period
        
    Returns:
        Processed DataFrame
    """
    print(f"\n{'='*80}")
    print(f"LOADING AND PREPARING DATA FOR {symbol}")
    print(f"{'='*80}")
    
    # Load data
    loader = StockDataLoader()
    
    # Check if data exists
    if os.path.exists(f'data/raw/{symbol}.csv'):
        print(f"Loading existing data for {symbol}...")
        df = loader.load_data(symbol)
    else:
        print(f"Fetching new data for {symbol}...")
        df = loader.fetch_stock_data(symbol, period=period)
        if df is not None and not df.empty:
            loader.save_data(df, symbol)
        else:
            raise ValueError(f"No data available for symbol {symbol}. Please check the symbol or try a different one.")
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df, n_lags=5, horizon=1, normalize=True)
    
    # Save features
    engineer.save_features(features_df, symbol)
    
    return features_df, engineer

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: DataFrame
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"\n✓ Data split:")
    print(f"  Training: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_df)} samples ({val_ratio*100:.0f}%)")
    print(f"  Testing: {len(test_df)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_df, val_df, test_df

def train_arima(train_df, test_df, symbol):
    """Train ARIMA model"""
    print(f"\n{'='*80}")
    print(f"TRAINING ARIMA MODEL")
    print(f"{'='*80}")
    
    # Get price series
    train_series = train_df['Close']
    test_series = test_df['Close']
    
    # Train
    arima = ARIMAPredictor()
    metrics, predictions = arima.evaluate(train_series, test_series)
    
    # Save model
    arima.save_model(f'models/saved_models/arima_models/{symbol}_arima.pkl')
    
    return metrics, predictions

def train_linear_regression(train_df, val_df, test_df, symbol):
    """Train Linear Regression model"""
    print(f"\n{'='*80}")
    print(f"TRAINING LINEAR REGRESSION MODEL")
    print(f"{'='*80}")
    
    # Initialize
    lr = LinearPredictor()
    
    # Prepare data
    X_train, y_train = lr.prepare_data(train_df)
    X_val, y_val = lr.prepare_data(val_df)
    X_test, y_test = lr.prepare_data(test_df)
    
    # Train
    lr.train(X_train, y_train, tune_hyperparameters=True)
    
    # Evaluate
    metrics, predictions = lr.evaluate(X_test, y_test)
    
    # Save model
    lr.save_model(f'models/saved_models/lr_model_{symbol}.pkl')
    
    return metrics, predictions, (X_train, y_train, X_val, y_val, X_test, y_test)

def train_lstm(train_df, val_df, test_df, symbol):
    """Train LSTM model"""
    print(f"\n{'='*80}")
    print(f"TRAINING LSTM MODEL")
    print(f"{'='*80}")
    
    # Initialize
    lstm = LSTMPredictor(sequence_length=10, units=50, layers=3, dropout=0.2)
    
    # Prepare sequences
    X_train, y_train = lstm.prepare_sequences(train_df)
    X_val, y_val = lstm.prepare_sequences(val_df)
    X_test, y_test = lstm.prepare_sequences(test_df)
    
    print(f"Sequences prepared:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        model_path=f'models/saved_models/lstm_model_{symbol}.h5'
    )
    
    # Evaluate
    metrics, predictions = lstm.evaluate(X_test, y_test)
    
    return metrics, predictions, (X_train, y_train, X_val, y_val, X_test, y_test)

def train_lightgbm(train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, symbol):
    """Train LightGBM model"""
    print(f"\n{'='*80}")
    print(f"TRAINING LIGHTGBM MODEL")
    print(f"{'='*80}")
    
    # Initialize
    lgbm = LightGBMPredictor()
    
    # Train
    lgbm.train(X_train, y_train, X_val, y_val, num_boost_round=1000, early_stopping_rounds=50)
    
    # Evaluate
    metrics, predictions = lgbm.evaluate(X_test, y_test)
    
    # Save model
    lgbm.save_model(f'models/saved_models/lgbm_model_{symbol}.pkl')
    
    return metrics, predictions

def train_ensemble(arima_preds, lr_preds, lstm_preds, lgbm_preds, y_test, symbol):
    """Train Ensemble meta-learner"""
    print(f"\n{'='*80}")
    print(f"TRAINING ENSEMBLE META-LEARNER")
    print(f"{'='*80}")

    # Ensure all predictions have the same length
    min_len = min(len(arima_preds), len(lr_preds), len(lstm_preds), len(lgbm_preds), len(y_test))
    arima_preds = arima_preds[:min_len]
    lr_preds = lr_preds[:min_len]
    lstm_preds = lstm_preds[:min_len]
    lgbm_preds = lgbm_preds[:min_len]
    y_test = y_test[:min_len]

    # Stack base predictions
    base_predictions = np.column_stack([arima_preds, lr_preds, lstm_preds, lgbm_preds])

    # Split for meta-learner training
    train_size = int(len(base_predictions) * 0.7)

    train_preds = base_predictions[:train_size]
    test_preds = base_predictions[train_size:]
    y_train_meta = y_test[:train_size]
    y_test_meta = y_test[train_size:]

    # Initialize ensemble
    ensemble = EnsemblePredictor(meta_model_type='ridge')
    ensemble.base_model_names = ['ARIMA', 'Linear Regression', 'LSTM', 'LightGBM']

    # Train
    ensemble.train_meta_learner(train_preds, y_train_meta)

    # Evaluate
    metrics, ensemble_preds = ensemble.evaluate(test_preds, y_test_meta)

    # Save
    ensemble.save_model(f'models/saved_models/meta_learner_{symbol}.pkl')

    return metrics, ensemble_preds

def save_all_predictions(y_test, predictions_dict, symbol):
    """Save all predictions to CSV"""
    results_df = pd.DataFrame({
        'Actual': y_test,
        'ARIMA': predictions_dict['ARIMA'],
        'Linear_Regression': predictions_dict['Linear Regression'],
        'LSTM': predictions_dict['LSTM'],
        'LightGBM': predictions_dict['LightGBM'],
        'Ensemble': predictions_dict['Ensemble']
    })
    
    filepath = f'data/predictions/{symbol}_predictions.csv'
    results_df.to_csv(filepath, index=False)
    print(f"\n✓ All predictions saved to {filepath}")

def main(symbol='AAPL', period='2y'):
    """
    Main training pipeline
    
    Args:
        symbol: Stock symbol to train on
        period: Time period for data
    """
    start_time = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"# STOCK PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print(f"# Symbol: {symbol}")
    print(f"# Period: {period}")
    print(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    # Create directories
    create_directories()
    
    # Load and prepare data
    features_df, engineer = load_and_prepare_data(symbol, period)
    
    # Split data
    train_df, val_df, test_df = split_data(features_df)
    
    # Store all predictions
    all_predictions = {}
    all_metrics = {}
    
    # 1. Train ARIMA
    try:
        arima_metrics, arima_preds = train_arima(train_df, test_df, symbol)
        all_metrics['ARIMA'] = arima_metrics
        all_predictions['ARIMA'] = arima_preds
    except Exception as e:
        print(f"✗ ARIMA training failed: {str(e)}")
        arima_preds = None
    
    # 2. Train Linear Regression
    try:
        lr_metrics, lr_preds, lr_data = train_linear_regression(train_df, val_df, test_df, symbol)
        X_train, y_train, X_val, y_val, X_test, y_test = lr_data
        all_metrics['Linear Regression'] = lr_metrics
        all_predictions['Linear Regression'] = lr_preds
    except Exception as e:
        print(f"✗ Linear Regression training failed: {str(e)}")
        lr_preds = None
    
    # 3. Train LSTM
    try:
        lstm_metrics, lstm_preds, lstm_data = train_lstm(train_df, val_df, test_df, symbol)
        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = lstm_data
        all_metrics['LSTM'] = lstm_metrics
        all_predictions['LSTM'] = lstm_preds
    except Exception as e:
        print(f"✗ LSTM training failed: {str(e)}")
        lstm_preds = None
        # Set dummy data for ensemble if LSTM fails - use the same length as other models
        y_test_lstm = y_test[:len(y_test)]  # Use the same y_test as linear regression for ensemble
    
    # 4. Train LightGBM (use same data as Linear Regression)
    try:
        lgbm_metrics, lgbm_preds = train_lightgbm(
            train_df, val_df, test_df,
            X_train, y_train, X_val, y_val, X_test, y_test,
            symbol
        )
        all_metrics['LightGBM'] = lgbm_metrics
        all_predictions['LightGBM'] = lgbm_preds
    except Exception as e:
        print(f"✗ LightGBM training failed: {str(e)}")
        lgbm_preds = None
    
    # 5. Train Ensemble (if all base models succeeded)
    if all(pred is not None for pred in [arima_preds, lr_preds, lstm_preds, lgbm_preds]):
        try:
            # Use LSTM test labels (they match in length with predictions)
            ensemble_metrics, ensemble_preds = train_ensemble(
                arima_preds, lr_preds, lstm_preds, lgbm_preds,
                y_test_lstm, symbol
            )
            all_metrics['Ensemble'] = ensemble_metrics
            all_predictions['Ensemble'] = ensemble_preds
        except Exception as e:
            print(f"✗ Ensemble training failed: {str(e)}")
    else:
        print("\n✗ Skipping ensemble training - some base models failed")
    
    # Evaluate and compare all models
    print(f"\n{'='*80}")
    print(f"FINAL MODEL COMPARISON")
    print(f"{'='*80}")
    
    evaluator = ModelEvaluator()
    evaluator.results = all_metrics
    comparison_df = evaluator.compare_models()
    
    # Save comparison
    comparison_df.to_csv(f'results/{symbol}_model_comparison.csv')
    print(f"\n✓ Model comparison saved to results/{symbol}_model_comparison.csv")
    
    # Save all predictions
    if 'Ensemble' in all_predictions:
        # Align all predictions to same length (use shortest)
        min_len = min(len(pred) for pred in all_predictions.values())
        aligned_predictions = {
            name: pred[:min_len] for name, pred in all_predictions.items()
        }
        y_test_aligned = y_test_lstm[:min_len]
        
        save_all_predictions(y_test_aligned, aligned_predictions, symbol)
    
    # Calculate total time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING COMPLETE")
    print(f"# Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"# End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    return all_metrics, all_predictions, comparison_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--period', type=str, default='2y', help='Time period (1y, 2y, 5y, max)')
    
    args = parser.parse_args()
    
    # Run training
    metrics, predictions, comparison = main(symbol=args.symbol, period=args.period)