import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import StockDataLoader
from utils.feature_engineering import FeatureEngineer
from models.ensemble import EnsemblePredictor

def load_processed_data(symbol='AAPL'):
    """Load already processed and split data"""
    print(f"\n{'='*80}")
    print(f"LOADING PROCESSED DATA FOR {symbol}")
    print(f"{'='*80}")
    
    # Load feature data
    features_path = f'data/processed/{symbol}_features.csv'
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"❌ No processed features found at {features_path}\n"
            f"   Please run: python train_all_models.py --symbol {symbol}"
        )
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(features_df)} samples with {len(features_df.columns)} features")
    
    return features_df

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    print(f"\n✓ Data split:")
    print(f"  Training: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_df)} samples ({val_ratio*100:.0f}%)")
    print(f"  Testing: {len(test_df)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_df, val_df, test_df

def generate_base_model_predictions(train_df, val_df, test_df, symbol='AAPL'):
    """
    Generate predictions from all base models
    
    This loads trained models and generates predictions on the test set
    """
    print(f"\n{'='*80}")
    print("GENERATING BASE MODEL PREDICTIONS")
    print(f"{'='*80}")
    
    # Initialize feature engineer to handle inverse transforms
    engineer = FeatureEngineer()
    engineer.fit_scalers(train_df)
    
    # Normalize data
    train_scaled = engineer.transform_features(train_df)
    val_scaled = engineer.transform_features(val_df)
    test_scaled = engineer.transform_features(test_df)
    
    feature_cols = [col for col in test_scaled.columns if col != 'Target']
    X_test = test_scaled[feature_cols]
    y_test = test_scaled['Target'].values
    
    # Store predictions
    predictions_dict = {}
    
    # 1. ARIMA Predictions
    print("\n1. Loading ARIMA predictions...")
    try:
        from models.arima_model import ARIMAPredictor
        import pickle
        
        arima_path = f'models/saved_models/arima_models/{symbol}_arima.pkl'
        if os.path.exists(arima_path):
            with open(arima_path, 'rb') as f:
                arima_model_fit = pickle.load(f)
            
            # Generate predictions
            arima_preds = arima_model_fit.forecast(steps=len(test_df))
            
            # Normalize ARIMA predictions to match other models
            arima_preds_norm = engineer.target_scaler.transform(
                arima_preds.values.reshape(-1, 1)
            ).flatten()
            
            predictions_dict['ARIMA'] = arima_preds_norm
            print(f"  ✓ ARIMA predictions: {len(arima_preds_norm)} samples")
        else:
            print(f"  ⚠️  ARIMA model not found, skipping")
    except Exception as e:
        print(f"  ✗ ARIMA prediction failed: {e}")
    
    # 2. Linear Regression Predictions
    print("\n2. Loading Linear Regression predictions...")
    try:
        import pickle
        lr_path = f'models/saved_models/lr_model_{symbol}.pkl'
        
        if os.path.exists(lr_path):
            with open(lr_path, 'rb') as f:
                lr_model = pickle.load(f)
            
            lr_preds = lr_model.predict(X_test)
            predictions_dict['Linear_Regression'] = lr_preds
            print(f"  ✓ Linear Regression predictions: {len(lr_preds)} samples")
        else:
            print(f"  ⚠️  Linear Regression model not found, skipping")
    except Exception as e:
        print(f"  ✗ Linear Regression prediction failed: {e}")
    
    # 3. LSTM Predictions
    print("\n3. Loading LSTM predictions...")
    try:
        from models.lstm_model import LSTMPredictor
        lstm_path = f'models/saved_models/lstm_model_{symbol}.h5'
        
        if os.path.exists(lstm_path):
            lstm = LSTMPredictor(sequence_length=10, n_features=len(feature_cols))
            lstm.load_model(lstm_path)
            
            # Prepare sequences
            X_test_seq, y_test_seq = lstm.prepare_sequences(test_scaled, feature_cols)
            lstm_preds = lstm.predict(X_test_seq)
            
            predictions_dict['LSTM'] = lstm_preds
            print(f"  ✓ LSTM predictions: {len(lstm_preds)} samples")
        else:
            print(f"  ⚠️  LSTM model not found, skipping")
    except Exception as e:
        print(f"  ✗ LSTM prediction failed: {e}")
    
    # 4. LightGBM Predictions
    print("\n4. Loading LightGBM predictions...")
    try:
        import lightgbm as lgb
        lgbm_path = f'models/saved_models/lgbm_model_{symbol}.pkl'
        
        if os.path.exists(lgbm_path):
            lgbm_model = lgb.Booster(model_file=lgbm_path)
            lgbm_preds = lgbm_model.predict(X_test)
            
            predictions_dict['LightGBM'] = lgbm_preds
            print(f"  ✓ LightGBM predictions: {len(lgbm_preds)} samples")
        else:
            print(f"  ⚠️  LightGBM model not found, skipping")
    except Exception as e:
        print(f"  ✗ LightGBM prediction failed: {e}")
    
    # Print summary
    print(f"\n✓ Generated predictions from {len(predictions_dict)} base models")
    for name, preds in predictions_dict.items():
        print(f"  {name}: {len(preds)} predictions")
    
    return predictions_dict, y_test, engineer

def train_ensemble_model(predictions_dict, y_test, symbol='AAPL'):
    """Train the ensemble meta-learner"""
    print(f"\n{'='*80}")
    print("TRAINING ENSEMBLE META-LEARNER")
    print(f"{'='*80}")
    
    if len(predictions_dict) < 2:
        raise ValueError(
            f"❌ Need at least 2 base models for ensemble. "
            f"Found {len(predictions_dict)}. "
            f"Please train base models first: python train_all_models.py --symbol {symbol}"
        )
    
    # Initialize ensemble
    ensemble = EnsemblePredictor(meta_model_type='ridge')
    
    # Train and evaluate with proper holdout
    holdout_metrics, holdout_preds = ensemble.train_and_evaluate(
        predictions_dict, 
        y_test,
        holdout_ratio=0.3,
        random_state=42
    )
    
    # Save ensemble model
    ensemble.save_model(f'models/saved_models/ensemble_model_{symbol}.pkl')
    
    return ensemble, holdout_metrics, holdout_preds

def compare_all_models(predictions_dict, y_test, ensemble_preds, engineer, symbol='AAPL'):
    """Generate final comparison including ensemble"""
    print(f"\n{'='*80}")
    print("FINAL MODEL COMPARISON (Original Scale)")
    print(f"{'='*80}")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Convert all predictions to original scale
    all_metrics = {}
    
    # Align predictions (take shortest length)
    min_length = min([len(p) for p in predictions_dict.values()] + [len(y_test)])
    y_test_aligned = y_test[-min_length:]
    
    # Individual models
    for name, preds in predictions_dict.items():
        preds_aligned = preds[-min_length:]
        
        # Inverse transform
        preds_original = engineer.inverse_transform_target(preds_aligned)
        y_true_original = engineer.inverse_transform_target(y_test_aligned)
        
        # Calculate metrics
        mse = mean_squared_error(y_true_original, preds_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_original, preds_original)
        r2 = r2_score(y_true_original, preds_original)
        
        mask = y_true_original != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_original[mask] - preds_original[mask]) / y_true_original[mask])) * 100
        else:
            mape = np.nan
        
        all_metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    # Ensemble model
    ensemble_preds_aligned = ensemble_preds[-min_length:]
    ensemble_original = engineer.inverse_transform_target(ensemble_preds_aligned)
    y_true_original = engineer.inverse_transform_target(y_test_aligned)
    
    mse = mean_squared_error(y_true_original, ensemble_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_original, ensemble_original)
    r2 = r2_score(y_true_original, ensemble_original)
    
    mask = y_true_original != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_original[mask] - ensemble_original[mask]) / y_true_original[mask])) * 100
    else:
        mape = np.nan
    
    all_metrics['Ensemble'] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df.round(4)
    
    # Sort by RMSE
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("\n" + "="*80)
    print(comparison_df)
    print("="*80)
    
    # Highlight best model
    best_model = comparison_df['RMSE'].idxmin()
    best_rmse = comparison_df.loc[best_model, 'RMSE']
    best_mape = comparison_df.loc[best_model, 'MAPE']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   RMSE: {best_rmse:.4f}")
    print(f"   MAPE: {best_mape:.2f}%")
    
    # Check if target achieved
    target_mape = 3.5
    if best_mape < target_mape:
        print(f"\n✓ ✓ ✓ TARGET ACHIEVED! MAPE ({best_mape:.2f}%) < {target_mape}% ✓ ✓ ✓")
    else:
        print(f"\n⚠️  Target not met. MAPE ({best_mape:.2f}%) >= {target_mape}%")
        print(f"   Consider: More data, sentiment features, hyperparameter tuning")
    
    # Save results
    comparison_df.to_csv(f'results/{symbol}_ensemble_comparison.csv')
    print(f"\n✓ Results saved to results/{symbol}_ensemble_comparison.csv")
    
    return comparison_df

def main(symbol='AAPL'):
    """Main ensemble training pipeline"""
    start_time = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE MODEL TRAINING")
    print(f"# Symbol: {symbol}")
    print(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    try:
        # Load data
        features_df = load_processed_data(symbol)
        
        # Split data
        train_df, val_df, test_df = split_data(features_df)
        
        # Generate base model predictions
        predictions_dict, y_test, engineer = generate_base_model_predictions(
            train_df, val_df, test_df, symbol
        )
        
        # Train ensemble
        ensemble, metrics, ensemble_preds = train_ensemble_model(
            predictions_dict, y_test, symbol
        )
        
        # Final comparison
        comparison_df = compare_all_models(
            predictions_dict, y_test, ensemble_preds, engineer, symbol
        )
        
        # Calculate total time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'#'*80}")
        print(f"# ENSEMBLE TRAINING COMPLETE")
        print(f"# Total time: {duration:.2f} seconds")
        print(f"{'#'*80}\n")
        
        return ensemble, comparison_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ensemble model with all base models'
    )
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='AAPL',
        help='Stock symbol (e.g., AAPL, TSLA, GOOGL)'
    )
    
    args = parser.parse_args()
    
    ensemble, comparison = main(symbol=args.symbol.upper())