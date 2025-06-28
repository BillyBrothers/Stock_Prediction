# lstm_forecaster.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(window_size, len(X)):
        Xs.append(X[i - window_size:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def run_lstm_forecast(msft_df, target_col='Close', window_size=35, n_splits=2):
    # Feature/Target split
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    # Scaling
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # Sequence generation
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    actual_values, predicted_values = [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
        print(f"Fold: {i}")
        print("Train dates:", msft_df.index[train_idx + window_size - 1][[0, -1]])
        print("Test dates:", msft_df.index[test_idx + window_size - 1][[0, -1]])

        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50),
            Dropout(0.2),
            BatchNormalization(),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])

        y_pred = model.predict(X_test[:1])
        actual_values.append(y_test[0])
        predicted_values.append(y_pred[0][0])

    mse = mean_squared_error(actual_values, predicted_values)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, predicted_values
