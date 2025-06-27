# arima_forecaster.py

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def run_arima_forecast(msft_df, target_col='Close', n_splits=100):
    y = msft_df[target_col]

    # Hyperparameter tuning with auto_arima
    arima_hypertuned = auto_arima(
        y,
        seasonal=False,
        start_p=0,
        start_q=0,
        test='adf',
        max_p=5,
        max_q=5,
        m=7,
        d=None,
        D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False
    )
    best_p, best_d, best_q = arima_hypertuned.order

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, predicted_values = [], []

    for train_idx, test_idx in tscv.split(msft_df):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = ARIMA(y_train, order=(best_p, best_d, best_q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)

        actual_values.append(y_test.iloc[0])
        predicted_values.append(forecast.values[0])

    mse = mean_squared_error(actual_values, predicted_values)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, predicted_values
