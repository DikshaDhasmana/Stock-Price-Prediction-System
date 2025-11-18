import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class LSTMPredictor:
    def __init__(self, sequence_length=10, units=50, layers=3, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.model = None
    def prepare_sequences(self, df):
        """Prepare sequences for LSTM training"""
        # Use Close price for sequences (already scaled by feature engineering)
        data = df['Close'].values

        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])

        # Ensure we have data
        if len(X) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}. Need at least {self.sequence_length + 1} data points.")

        # Reshape X for LSTM input (samples, timesteps, features)
        X = np.array(X).reshape((len(X), self.sequence_length, 1))
        y = np.array(y)

        return X, y

    def build_model(self):
        """Build LSTM model"""
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(self.dropout))

        # Hidden LSTM layers
        for _ in range(self.layers - 1):
            model.add(LSTM(units=self.units, return_sequences=True))
            model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(units=1))

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path=None):
        """Train LSTM model"""
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        if model_path:
            self.model.save(model_path)
            print(f"LSTM model saved to {model_path}")

        return history

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)

        # Calculate metrics (data is already scaled)
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
