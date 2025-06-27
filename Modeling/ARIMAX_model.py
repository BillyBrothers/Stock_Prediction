# arimax_forecaster.py

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def run_arimax_forecast(msft_df, target_col='Rolling_Log_Avg', n_splits=100):
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    # Convert all features to float (e.g., for boolean compatibility)
    X_floats = X.astype(float)

    # Hyperparameter tuning using auto_arima
    arimax_hypertuned = auto_arima(
        y,
        exogenous=X,
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
    best_p, best_d, best_q = arimax_hypertuned.order

    # Walk-forward ARIMAX forecasting
    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, predicted_values = [], []

    for train_idx, test_idx in tscv.split(msft_df):
        X_train, X_test = X_floats.iloc[train_idx], X_floats.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = ARIMA(endog=y_train, exog=X_train, order=(best_p, best_d, best_q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1, exog=X_test.iloc[[0]])

        actual_values.append(y_test.iloc[0])
        predicted_values.append(forecast.values[0])

    mse = mean_squared_error(actual_values, predicted_values)
    results_df = pd.DataFrame({"Actual_vs_Predicted_MSE": [mse]})
    return results_df, actual_values, predicted_values
