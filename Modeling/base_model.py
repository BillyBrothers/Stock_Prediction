import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def run_naive_forecast(msft_df, target_col='Close', n_splits=100):
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, naive_predictions = [], []

    for train_idx, test_idx in tscv.split(X):
        y_train, y_test = y[train_idx], y[test_idx]

        last_known = y_train[-1]
        next_actual = y_test[0]

        actual_values.append(last_known)
        naive_predictions.append(next_actual)

    mse = mean_squared_error(actual_values, naive_predictions)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, naive_predictions
