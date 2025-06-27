
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def run_xgb_forecast(msft_df, target_col='Rolling_Log_Avg', n_splits=100):
    X = msft_df.drop([target_col], axis=1)
    y = msft_df[target_col]

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, predicted_values = [], []

    xgb = XGBRegressor()

    for i, (train_idx, test_idx) in enumerate(tscv.split(msft_df)):
        print(f"Fold: {i}")
        print("Train dates:", msft_df.index[train_idx][[0, -1]])
        print("Test dates:", msft_df.index[test_idx][[0, -1]])

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test[[0]])

        actual_values.append(y_test[0])
        predicted_values.append(y_pred[0])

    mse = mean_squared_error(actual_values, predicted_values)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, predicted_values
