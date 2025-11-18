"""
train_with_sentiment.py - Train models WITH Alpha Vantage sentiment (Free plan)

- Uses yfinance (via utils.data_loader.StockDataLoader)
- Uses sentiment_analyzer.FinancialSentimentAnalyzer (Alpha Vantage)
- Trains ARIMA, LinearRegression, LSTM, LightGBM
- Aligns sequence lengths: drops initial seq_len rows from non-LSTM models so all predictions match
- Trains EnsemblePredictor on base model predictions
- Saves models to models/saved_models_with_sentiment/
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import StockDataLoader
from utils.feature_engineering import FeatureEngineer
from sentiment_analyzer import FinancialSentimentAnalyzer

# models
from models.linear_model import LinearPredictor
from models.lstm_model import LSTMPredictor
from models.lgbm_model import LightGBMPredictor
from models.arima_model import ARIMAPredictor
from models.ensemble import EnsemblePredictor

# reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# output dirs
SAVED_DIR = "models/saved_models_with_sentiment"
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


def prepare_data_with_sentiment(symbol: str, api_key: str = None, period: str = "2y"):
    loader = StockDataLoader()
    # load or fetch raw
    df = loader.load_data(symbol)
    if df is None or df.empty:
        df = loader.fetch_stock_data(symbol, period=period)
        if df is None or df.empty:
            raise ValueError(f"No price data for {symbol}")
        loader.save_data(df, symbol)

    # generate technical features
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df, n_lags=5, horizon=1)

    # sentiment pipeline (Alpha Vantage)
    start_date = df.index[0].date().isoformat()
    end_date = df.index[-1].date().isoformat()
    analyzer = FinancialSentimentAnalyzer(api_key)
    sentiment_df = analyzer.run_pipeline(symbol, start_date, end_date)

    if sentiment_df is None or sentiment_df.empty:
        print("⚠️ No sentiment produced; continuing without sentiment")
        combined = features_df
        has_sentiment = False
    else:
        combined = analyzer.integrate_with_features(features_df, sentiment_df) if hasattr(analyzer, "integrate_with_features") else features_df.join(sentiment_df, how="left")
        # ensure columns filled
        combined = combined.fillna(method="ffill").fillna(0)
        out_path = f"data/processed/{symbol}_features_with_sentiment.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.to_csv(out_path)
        has_sentiment = True
        print(f"✓ Combined features saved to {out_path}")

    return combined, engineer, has_sentiment


def split_and_scale(combined_df: pd.DataFrame, engineer: FeatureEngineer, train_ratio=0.7, val_ratio=0.15):
    n = len(combined_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = combined_df[:train_end].copy()
    val_df = combined_df[train_end:val_end].copy()
    test_df = combined_df[val_end:].copy()

    # fit scalers on train only
    engineer.fit_scalers(train_df)
    train_scaled = engineer.transform_features(train_df)
    val_scaled = engineer.transform_features(val_df)
    test_scaled = engineer.transform_features(test_df)

    return train_scaled, val_scaled, test_scaled


def train_models(train_df, val_df, test_df, engineer, symbol: str):
    feature_cols = [c for c in train_df.columns if c != "Target"]
    print(f"Training with {len(feature_cols)} features")

    results = {}
    predictions = {}

    # 1. ARIMA on original (unscaled) target series: use raw train/test (inverse transform target from scaled)
    print("\n=== TRAIN ARIMA ===")
    try:
        # arima expects original-scale series; get raw train/test from engineer.inverse (we need train/test before scaling)
        # For safety, reconstruct original target series using inverse transform
        y_train_orig = engineer.inverse_transform_target(train_df["Target"].values)
        y_test_orig = engineer.inverse_transform_target(test_df["Target"].values)

        arima = ARIMAPredictor()
        arima.fit(y_train_orig)
        arima_preds = arima.predict(len(y_test_orig))
        # normalize arima preds to scaled target space
        arima_preds_norm = engineer.target_scaler.transform(arima_preds.reshape(-1, 1)).flatten()
        predictions["ARIMA"] = arima_preds_norm
        # compute metrics on original scale
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test_orig, arima_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, arima_preds)
        r2 = r2_score(y_test_orig, arima_preds)
        results["ARIMA"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": np.mean(np.abs((y_test_orig - arima_preds) / y_test_orig)) * 100}
        # Save arima model
        import pickle
        os.makedirs(os.path.join(SAVED_DIR, "arima_models"), exist_ok=True)
        with open(os.path.join(SAVED_DIR, f"{symbol}_arima.pkl"), "wb") as f:
            pickle.dump(arima.model_fit, f)
        print("✓ ARIMA done")
    except Exception as e:
        print(f"ARIMA failed: {e}")

    # 2. Linear Regression (on scaled features)
    print("\n=== TRAIN LR ===")
    lr = LinearPredictor()
    lr.feature_cols = feature_cols
    lr.train(train_df[feature_cols], train_df["Target"])
    lr_preds_norm = lr.predict(test_df[feature_cols])
    predictions["Linear_Regression"] = lr_preds_norm
    lr_metrics, lr_preds_orig = evaluate_model_on_original_scale(lr, test_df[feature_cols], test_df["Target"], engineer, "Linear Regression")
    results["Linear_Regression"] = lr_metrics
    # save
    import pickle
    with open(os.path.join(SAVED_DIR, f"lr_sentiment_{symbol}.pkl"), "wb") as f:
        pickle.dump(lr.model, f)

    # 3. LSTM (use scaled data sequences)
    print("\n=== TRAIN LSTM ===")
    seq_len = 10
    lstm = LSTMPredictor(sequence_length=seq_len, units=50, layers=2, dropout=0.2, n_features=len(feature_cols))
    X_train_seq, y_train_seq = lstm.prepare_sequences(train_df, feature_cols)
    X_val_seq, y_val_seq = lstm.prepare_sequences(val_df, feature_cols)
    X_test_seq, y_test_seq = lstm.prepare_sequences(test_df, feature_cols)
    lstm.build_model()
    lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, batch_size=32, model_path=os.path.join(SAVED_DIR, f"lstm_sentiment_{symbol}.h5"))
    lstm_metrics, lstm_preds_norm = lstm.evaluate(X_test_seq, y_test_seq)
    # lstm_preds_norm are normalized (scaled) target predictions
    predictions["LSTM"] = lstm_preds_norm
    results["LSTM"] = lstm_metrics

    # 4. LightGBM
    print("\n=== TRAIN LightGBM ===")
    lgbm = LightGBMPredictor()
    lgbm.train(train_df[feature_cols], train_df["Target"], val_df[feature_cols], val_df["Target"], num_boost_round=500, early_stopping_rounds=50)
    lgbm_preds_norm = lgbm.predict(test_df[feature_cols])
    predictions["LightGBM"] = lgbm_preds_norm
    lgbm_metrics, lgbm_preds_orig = evaluate_model_on_original_scale(lgbm, test_df[feature_cols], test_df["Target"], engineer, "LightGBM")
    results["LightGBM"] = lgbm_metrics
    lgbm.save_model(os.path.join(SAVED_DIR, f"lgbm_sentiment_{symbol}.pkl"))

    # --------------------
    # Align predictions lengths:
    # LSTM predictions length = len(test_df) - seq_len
    # For fair comparison, drop first seq_len rows from other model predictions & test target.
    # --------------------
    min_len = None
    for k, preds in predictions.items():
        if min_len is None or len(preds) < min_len:
            min_len = len(preds)
    # min_len should be len(lstm_preds) typically
    print(f"Aligning all model preds to length = {min_len}")

    aligned_preds = {}
    for k, preds in predictions.items():
        aligned_preds[k] = preds[-min_len:]

    # Align y_test (scaled) by trimming the beginning seq_len rows so it matches LSTM
    y_test_scaled = test_df["Target"].values
    y_test_aligned_scaled = y_test_scaled[-min_len:]

    # inverse transform y_test for ensemble training metrics
    y_test_aligned_orig = engineer.inverse_transform_target(y_test_aligned_scaled)

    # Convert all preds to original scale for ensemble training (ensemble expects original-scale targets)
    preds_orig = {}
    for k, preds_scaled in aligned_preds.items():
        # preds may already be in scaled space or original depending on model; our convention:
        # - LR, LGBM, LSTM: outputs are scaled (because trained on scaled Target)
        # - ARIMA: we stored normalized (arima_preds_norm) when adding to predictions dict
        try:
            preds_orig[k] = engineer.inverse_transform_target(preds_scaled)
        except Exception:
            # fallback if already original
            import numpy as _np
            preds_orig[k] = _np.array(preds_scaled)

    # TRAIN ENSEMBLE meta-learner on aligned preds
    print("\n=== TRAIN ENSEMBLE (with sentiment) ===")
    ensemble = EnsemblePredictor(meta_model_type="ridge")
    meta_X = {k: v for k, v in preds_orig.items()}
    # Ensemble class expects dict of base model preds and y_true (original)
    holdout_metrics, holdout_preds = ensemble.train_and_evaluate(meta_X, y_test_aligned_orig, holdout_ratio=0.3, random_state=42)
    ensemble.save_model(os.path.join(SAVED_DIR, f"ensemble_sentiment_{symbol}.pkl"))

    # Compute and store final comparison DataFrame
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    all_metrics = {}
    for name, arr in preds_orig.items():
        y_pred = arr[-min_len:]
        y_true = y_test_aligned_orig
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mask = y_true != 0
        mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else np.nan
        all_metrics[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    # ensemble metrics
    en_preds = holdout_preds
    mse = mean_squared_error(y_test_aligned_orig, en_preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test_aligned_orig, en_preds)
    r2 = r2_score(y_test_aligned_orig, en_preds)
    mask = y_test_aligned_orig != 0
    mape = (np.mean(np.abs((y_test_aligned_orig[mask] - en_preds[mask]) / y_test_aligned_orig[mask])) * 100) if mask.sum() > 0 else np.nan
    all_metrics["Ensemble"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    # Save results
    comp_df = pd.DataFrame(all_metrics).T.round(4).sort_values("RMSE")
    comp_df.to_csv(f"results/{symbol}_sentiment_comparison.csv")
    print(f"✓ Saved comparison to results/{symbol}_sentiment_comparison.csv")

    return results, preds_orig, comp_df


def evaluate_model_on_original_scale(model, X_test, y_test, engineer, model_name):
    """
    Accepts either sklearn/lightgbm model or custom model with evaluate()
    Returns metrics (on original scale) and original-scale predictions.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if hasattr(model, "evaluate"):
        _, preds_scaled = model.evaluate(X_test, y_test)
    else:
        preds_scaled = model.predict(X_test)

    preds_orig = engineer.inverse_transform_target(preds_scaled)
    if isinstance(y_test, pd.Series):
        y_true_orig = engineer.inverse_transform_target(y_test.values)
    else:
        y_true_orig = engineer.inverse_transform_target(y_test)

    mse = mean_squared_error(y_true_orig, preds_orig)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true_orig, preds_orig)
    r2 = r2_score(y_true_orig, preds_orig)
    mask = y_true_orig != 0
    mape = (np.mean(np.abs((y_true_orig[mask] - preds_orig[mask]) / y_true_orig[mask])) * 100) if mask.sum() > 0 else np.nan

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
    print(f"\n✓ {model_name} performance (original scale): RMSE {rmse:.4f}, MAE {mae:.4f}, R2 {r2:.4f}")
    return metrics, preds_orig


def main(symbol="AAPL", api_key=None, period="2y"):
    start = datetime.now()
    combined, engineer, has_sentiment = prepare_data_with_sentiment(symbol, api_key, period)
    train_df, val_df, test_df = split_and_scale(combined, engineer)
    results, preds, comp_df = train_models(train_df, val_df, test_df, engineer, symbol)
    duration = (datetime.now() - start).total_seconds()
    print(f"Total time: {duration:.2f}s")
    return results, preds, comp_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--api-key", type=str, help="Alpha Vantage API key")
    parser.add_argument("--period", type=str, default="2y")
    args = parser.parse_args()
    if args.api_key:
        os.environ["ALPHA_VANTAGE_API_KEY"] = args.api_key
    main(symbol=args.symbol.upper(), api_key=args.api_key, period=args.period)
