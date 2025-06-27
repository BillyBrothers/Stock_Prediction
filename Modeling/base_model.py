import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def run_naive_forecast(msft_df, target_col='Rolling_Log_Avg', n_splits=100):
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    # Scaling
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, naive_predictions = [], []

    for train_idx, test_idx in tscv.split(X_scaled):
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

        last_known = y_train[-1]
        next_actual = y_test[0]

        actual_values.append(last_known)
        naive_predictions.append(next_actual)

    mse = mean_squared_error(actual_values, naive_predictions)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, naive_predictions
