"""
train_with_sentiment.py - Train Models WITH Sentiment Analysis

This script:
1. Loads historical stock data
2. Fetches and analyzes financial news sentiment
3. Combines technical + sentiment features  
4. Trains all models with enhanced features
5. Compares performance with/without sentiment

Usage:
    python train_with_sentiment.py --symbol AAPL --api-key YOUR_NEWS_API_KEY
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import StockDataLoader
from utils.feature_engineering import FeatureEngineer
from sentiment_analyzer import FinancialSentimentAnalyzer

# Import models
from models.linear_model import LinearPredictor
from models.lstm_model import LSTMPredictor
from models.lgbm_model import LightGBMPredictor
from models.ensemble import EnsemblePredictor

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def fetch_and_analyze_sentiment(symbol, start_date, end_date, api_key=None):
    """
    Fetch news and calculate sentiment scores
    """
    print(f"\n{'='*80}")
    print("FETCHING AND ANALYZING FINANCIAL NEWS SENTIMENT")
    print(f"{'='*80}")
    
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer(news_api_key=api_key)
    
    # Fetch news
    articles = analyzer.fetch_news(
        symbol,
        start_date.isoformat(),
        end_date.isoformat(),
        max_articles=100
    )
    
    if not articles:
        print("\n⚠️  WARNING: No news articles fetched")
        print("   Possible reasons:")
        print("   1. Invalid or missing NewsAPI key")
        print("   2. API rate limit reached (free tier: 100 requests/day)")
        print("   3. No news available for date range")
        print("\n   Get free key: https://newsapi.org/")
        print("   Continuing WITHOUT sentiment features...")
        return None
    
    # Calculate daily sentiment
    sentiment_df = analyzer.calculate_daily_sentiment(articles)
    
    if sentiment_df.empty:
        print("⚠️  No sentiment data generated")
        return None
    
    # Save sentiment data
    analyzer.save_sentiment_data(sentiment_df, symbol)
    
    return sentiment_df


def prepare_data_with_sentiment(symbol, api_key=None, period='2y'):
    """
    Prepare features WITH sentiment analysis
    """
    print(f"\n{'='*80}")
    print(f"PREPARING DATA WITH SENTIMENT FOR {symbol}")
    print(f"{'='*80}")
    
    # Load stock data
    loader = StockDataLoader()
    
    # Try to load existing data
    if os.path.exists(f'data/raw/{symbol}.csv'):
        df = loader.load_data(symbol)
        if df.index[-1] < pd.Timestamp.now() - pd.Timedelta(days=7):
            print("  Data outdated, fetching fresh data...")
            df = loader.fetch_stock_data(symbol, period=period)
            if df is not None:
                loader.save_data(df, symbol)
    else:
        df = loader.fetch_stock_data(symbol, period=period)
        if df is not None:
            loader.save_data(df, symbol)
    
    if df is None or df.empty:
        raise ValueError(f"Failed to load data for {symbol}")
    
    print(f"✓ Stock data: {len(df)} samples from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Create technical features
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df, n_lags=5, horizon=1)
    
    print(f"✓ Technical features: {len(features_df.columns)} features")
    
    # Fetch and integrate sentiment
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    
    sentiment_df = fetch_and_analyze_sentiment(symbol, start_date, end_date, api_key)
    
    if sentiment_df is not None:
        # Integrate sentiment with technical features
        analyzer = FinancialSentimentAnalyzer()
        combined_df = analyzer.integrate_with_features(features_df, sentiment_df)
        
        # Save combined features
        os.makedirs('data/processed', exist_ok=True)
        combined_df.to_csv(f'data/processed/{symbol}_features_with_sentiment.csv')
        print(f"✓ Saved enhanced features to data/processed/{symbol}_features_with_sentiment.csv")
        
        has_sentiment = True
    else:
        combined_df = features_df
        has_sentiment = False
    
    return combined_df, engineer, has_sentiment


def split_and_normalize(df, engineer, train_ratio=0.7, val_ratio=0.15):
    """Split data and normalize"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Normalize
    engineer.fit_scalers(train_df)
    train_scaled = engineer.transform_features(train_df)
    val_scaled = engineer.transform_features(val_df)
    test_scaled = engineer.transform_features(test_df)
    
    return train_scaled, val_scaled, test_scaled


def train_all_models_with_sentiment(train_df, val_df, test_df, engineer, symbol):
    """
    Train all models with sentiment-enhanced features
    """
    feature_cols = [col for col in train_df.columns if col != 'Target']
    print(f"\n✓ Training with {len(feature_cols)} features (including sentiment)")
    
    # Check if sentiment features are included
    sentiment_features = [col for col in feature_cols if 'Sentiment' in col or 'Article' in col]
    if sentiment_features:
        print(f"  Sentiment features detected: {sentiment_features}")
    
    all_predictions = {}
    all_metrics = {}
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    X_val = val_df[feature_cols]
    y_val = val_df['Target']
    X_test = test_df[feature_cols]
    y_test = test_df['Target']
    
    # 1. Linear Regression
    print(f"\n{'='*60}")
    print("Training Linear Regression with Sentiment")
    print(f"{'='*60}")
    
    lr = LinearPredictor()
    lr.feature_cols = feature_cols
    lr.train(X_train, y_train)
    
    lr_metrics, lr_preds = evaluate_model_on_original_scale(
        lr, X_test, y_test, engineer, "Linear Regression"
    )
    all_metrics['Linear_Regression'] = lr_metrics
    all_predictions['Linear_Regression'] = lr_preds
    
    lr.save_model(f'models/saved_models/lr_sentiment_{symbol}.pkl')
    
    # 2. LSTM
    print(f"\n{'='*60}")
    print("Training LSTM with Sentiment")
    print(f"{'='*60}")
    
    lstm = LSTMPredictor(
        sequence_length=10,
        units=50,
        layers=2,
        dropout=0.2,
        n_features=len(feature_cols)
    )
    
    X_train_seq, y_train_seq = lstm.prepare_sequences(train_df, feature_cols)
    X_val_seq, y_val_seq = lstm.prepare_sequences(val_df, feature_cols)
    X_test_seq, y_test_seq = lstm.prepare_sequences(test_df, feature_cols)
    
    lstm.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=50,
        batch_size=32,
        model_path=f'models/saved_models/lstm_sentiment_{symbol}.h5'
    )
    
    lstm_metrics, lstm_preds = evaluate_model_on_original_scale(
        lstm, X_test_seq, y_test_seq, engineer, "LSTM"
    )
    all_metrics['LSTM'] = lstm_metrics
    all_predictions['LSTM'] = lstm_preds
    
    # 3. LightGBM
    print(f"\n{'='*60}")
    print("Training LightGBM with Sentiment")
    print(f"{'='*60}")
    
    lgbm = LightGBMPredictor()
    lgbm.train(X_train, y_train, X_val, y_val, 
               num_boost_round=500, early_stopping_rounds=50)
    
    lgbm_metrics, lgbm_preds = evaluate_model_on_original_scale(
        lgbm, X_test, y_test, engineer, "LightGBM"
    )
    all_metrics['LightGBM'] = lgbm_metrics
    all_predictions['LightGBM'] = lgbm_preds
    
    lgbm.save_model(f'models/saved_models/lgbm_sentiment_{symbol}.pkl')
    
    return all_metrics, all_predictions


def evaluate_model_on_original_scale(model, X_test, y_test, engineer, model_name):
    """Helper to evaluate model and convert to original scale"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Get predictions
    if hasattr(model, 'evaluate'):
        _, predictions_norm = model.evaluate(X_test, y_test)
    else:
        predictions_norm = model.predict(X_test)
    
    # Convert to original scale
    predictions_original = engineer.inverse_transform_target(predictions_norm)
    if isinstance(y_test, pd.Series):
        y_test_original = engineer.inverse_transform_target(y_test.values)
    else:
        y_test_original = engineer.inverse_transform_target(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    
    mask = y_test_original != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_original[mask] - predictions_original[mask]) / y_test_original[mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"\n✓ {model_name} Performance:")
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    
    return metrics, predictions_original


def compare_with_without_sentiment(symbol):
    """
    Compare model performance with and without sentiment
    """
    print(f"\n{'='*80}")
    print("COMPARING MODELS: WITH vs WITHOUT SENTIMENT")
    print(f"{'='*80}")
    
    # Load results WITHOUT sentiment
    results_without_path = f'results/{symbol}_model_comparison.csv'
    if not os.path.exists(results_without_path):
        print(f"\n⚠️  No baseline results found at {results_without_path}")
        print("   Run: python train_all_models.py --symbol {symbol} first")
        return None
    
    results_without = pd.read_csv(results_without_path, index_col=0)
    
    # Load results WITH sentiment
    results_with_path = f'results/{symbol}_sentiment_comparison.csv'
    if not os.path.exists(results_with_path):
        print(f"⚠️  No sentiment results found")
        return None
    
    results_with = pd.read_csv(results_with_path, index_col=0)
    
    # Compare
    print("\n" + "="*80)
    print("WITHOUT SENTIMENT:")
    print("="*80)
    print(results_without)
    
    print("\n" + "="*80)
    print("WITH SENTIMENT:")
    print("="*80)
    print(results_with)
    
    # Calculate improvement
    print("\n" + "="*80)
    print("IMPROVEMENT (%) - Negative = Better Performance")
    print("="*80)
    
    improvement = pd.DataFrame()
    for metric in ['RMSE', 'MAE', 'MAPE']:
        if metric in results_with.columns and metric in results_without.columns:
            improvement[metric] = ((results_with[metric] - results_without[metric]) / 
                                  results_without[metric] * 100)
    
    print(improvement)
    
    # Save comparison
    comparison_path = f'results/{symbol}_sentiment_impact.csv'
    improvement.to_csv(comparison_path)
    print(f"\n✓ Comparison saved to {comparison_path}")
    
    return improvement


def main(symbol='AAPL', api_key=None, period='2y'):
    """
    Main training pipeline WITH sentiment analysis
    """
    start_time = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING WITH SENTIMENT ANALYSIS")
    print(f"# Symbol: {symbol}")
    print(f"# Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    try:
        # Prepare data with sentiment
        combined_df, engineer, has_sentiment = prepare_data_with_sentiment(
            symbol, api_key, period
        )
        
        if not has_sentiment:
            print("\n⚠️  WARNING: Training WITHOUT sentiment features")
            print("   To include sentiment, provide a valid NewsAPI key")
        
        # Split and normalize
        train_df, val_df, test_df = split_and_normalize(
            combined_df, engineer
        )
        
        # Train models
        all_metrics, all_predictions = train_all_models_with_sentiment(
            train_df, val_df, test_df, engineer, symbol
        )
        
        # Save results
        results_df = pd.DataFrame(all_metrics).T.round(4)
        results_path = f'results/{symbol}_sentiment_comparison.csv'
        results_df.to_csv(results_path)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS WITH SENTIMENT")
        print(f"{'='*80}")
        print(results_df)
        print(f"\n✓ Results saved to {results_path}")
        
        # Compare with baseline
        if has_sentiment:
            improvement = compare_with_without_sentiment(symbol)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING COMPLETE")
        print(f"# Duration: {duration:.2f}s ({duration/60:.2f}min)")
        print(f"{'#'*80}\n")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train models WITH sentiment analysis'
    )
    parser.add_argument(
        '--symbol', type=str, default='AAPL',
        help='Stock symbol (e.g., AAPL, TSLA, GOOGL)'
    )
    parser.add_argument(
        '--api-key', type=str,
        help='NewsAPI key (get from https://newsapi.org/)'
    )
    parser.add_argument(
        '--period', type=str, default='2y',
        help='Time period (1y, 2y, 5y, max)'
    )
    
    args = parser.parse_args()
    
    # Set API key in environment if provided
    if args.api_key:
        os.environ['NEWS_API_KEY'] = args.api_key
    
    results = main(
        symbol=args.symbol.upper(),
        api_key=args.api_key,
        period=args.period
    )