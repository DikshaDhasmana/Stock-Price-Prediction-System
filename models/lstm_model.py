import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LSTMPredictor:
    def __init__(self, sequence_length=60, units=64, layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def prepare_data(self, df, feature_cols=["Close"]):
        """
        Automatically prepares:
            - Scaling
            - Sequence creation
            - Target (next-day Close)
        """

        df = df.copy()

        # Create target column
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        # Scale features
        scaled_features = self.scaler.fit_transform(df[feature_cols])
        scaled_target = self.target_scaler.fit_transform(df[["Target"]])

        X, y = [], []
        for i in range(self.sequence_length, len(df)):
            X.append(scaled_features[i - self.sequence_length:i])
            y.append(scaled_target[i])

        X = np.array(X)
        y = np.array(y)

        return X, y

    def build_model(self, n_features):
        model = Sequential()

        # First LSTM Layer
        model.add(LSTM(self.units, return_sequences=(self.layers > 1),
                       input_shape=(self.sequence_length, n_features)))
        model.add(Dropout(self.dropout))

        # Additional LSTM layers
        for _ in range(self.layers - 2):
            model.add(LSTM(self.units, return_sequences=True))
            model.add(Dropout(self.dropout))

        if self.layers > 1:
            model.add(LSTM(self.units, return_sequences=False))
            model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(1))

        model.compile(optimizer=Adam(0.001), loss='mse')
        self.model = model

        print("\n✓ LSTM Model Summary:")
        model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build_model(X_train.shape[2])

        early_stop = EarlyStopping(
            patience=10, monitor="val_loss", restore_best_weights=True, verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        return history

    def predict(self, X):
        preds = self.model.predict(X)
        # inverse scaling to original price
        preds = self.target_scaler.inverse_transform(preds)
        return preds.flatten()

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        actual = self.target_scaler.inverse_transform(y_test)

        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)

        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        }, predictions
      