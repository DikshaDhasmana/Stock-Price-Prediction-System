import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class LSTMPredictor:
    def __init__(self, sequence_length=10, units=50, layers=2, dropout=0.2, n_features=1):
        """
        CORRECTED LSTM Model
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units per layer
            layers: Number of LSTM layers
            dropout: Dropout rate
            n_features: Number of features (IMPORTANT: now supports multiple features)
        """
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.n_features = n_features
        self.model = None

    def prepare_sequences(self, df, feature_cols=None):
        """
        Prepare sequences for LSTM training using ALL features
        
        CORRECTION: Now uses all engineered features, not just Close price
        """
        if feature_cols is None:
            # If not specified, use all numeric columns except Target
            feature_cols = [col for col in df.columns if col != 'Target']
        
        # Extract feature data
        feature_data = df[feature_cols].values
        target_data = df['Target'].values
        
        X, y = [], []
        for i in range(self.sequence_length, len(feature_data)):
            # Create sequence of features
            X.append(feature_data[i-self.sequence_length:i])
            y.append(target_data[i])
        
        # Ensure we have data
        if len(X) == 0:
            raise ValueError(
                f"Not enough data for sequence length {self.sequence_length}. "
                f"Need at least {self.sequence_length + 1} data points."
            )
        
        # Reshape X for LSTM input: (samples, timesteps, features)
        X = np.array(X)
        y = np.array(y)
        
        # Update n_features if not set
        if self.n_features == 1 and X.shape[2] > 1:
            self.n_features = X.shape[2]
            print(f"  Auto-detected {self.n_features} features")
        
        return X, y

    def build_model(self):
        """
        Build LSTM model
        
        CORRECTION: Last LSTM layer now has return_sequences=False
        """
        model = Sequential()
        
        # First LSTM layer
        if self.layers > 1:
            model.add(LSTM(
                units=self.units, 
                return_sequences=True,  # True because we have more layers
                input_shape=(self.sequence_length, self.n_features)
            ))
            model.add(Dropout(self.dropout))
            
            # Hidden LSTM layers
            for i in range(self.layers - 2):
                model.add(LSTM(units=self.units, return_sequences=True))
                model.add(Dropout(self.dropout))
            
            # Last LSTM layer - CORRECTION: return_sequences=False
            model.add(LSTM(units=self.units, return_sequences=False))
            model.add(Dropout(self.dropout))
        else:
            # Single LSTM layer
            model.add(LSTM(
                units=self.units,
                return_sequences=False,  # False for single output
                input_shape=(self.sequence_length, self.n_features)
            ))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        
        print("✓ LSTM model built:")
        print(f"  Architecture: {self.layers} LSTM layers")
        print(f"  Units per layer: {self.units}")
        print(f"  Input shape: ({self.sequence_length}, {self.n_features})")
        print(f"  Total parameters: {model.count_params():,}")

    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, model_path=None):
        """
        Train LSTM model with early stopping
        
        CORRECTION: Added early stopping to prevent overfitting
        """
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        print(f"\n✓ Training LSTM...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Save model if path provided
        if model_path:
            self.model.save(model_path)
            print(f"✓ Model saved to {model_path}")
        
        return history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        NOTE: Returns metrics on the SAME scale as input data
        Caller should inverse transform if needed
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # MAPE - handle division by zero
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
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

    def load_model(self, model_path):
        """Load a saved model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        print(f"✓ Model loaded from {model_path}")

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            print("Model not built yet")
        else:
            self.model.summary()