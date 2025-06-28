import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings

def run_arimax_forecast(msft_df, target_col='Close', n_splits=100):
    """
    Runs an ARIMAX time series forecast with hyperparameter tuning via auto_arima
    and walk-forward validation.

    Parameters:
        msft_df (pd.DataFrame): DataFrame containing the target_col and exogenous variables.
        target_col (str): The name of the target column (e.g., 'Close').
        n_splits (int): Number of splits for TimeSeriesSplit cross-validation.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the ARIMAX Mean Squared Error and Order.
            - list: List of actual values for the first step of each test split.
            - list: List of predicted values for the first step of each test split.
            - tuple: The (p,d,q) order found by auto_arima.
    """
    # Ensure there are enough rows for splitting
    if len(msft_df) < n_splits + 1: # At least n_splits for training + 1 for test
        raise ValueError(f"Not enough data points ({len(msft_df)}) for {n_splits} splits. Consider reducing n_splits or providing more data.")

    # Separate target (endogenous) and features (exogenous)
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    # Convert all exogenous features to float and handle potential non-numeric
    # It's crucial that X does not contain NaNs or non-numeric types for statsmodels ARIMA
    X_floats = X.select_dtypes(include=np.number).astype(float)

    if X_floats.empty:
        raise ValueError("No numeric features found for exogenous variables (X). ARIMAX requires at least one numeric exogenous variable.")

    # Suppress specific statsmodels warnings that can be verbose during fitting
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Hyperparameter tuning using auto_arima
    # It tunes on the full dataset for efficiency, but note potential data leakage for strict validation.
    arimax_hypertuned = auto_arima(
        y,
        exogenous=X_floats, # Pass exogenous variables here for tuning
        seasonal=False,
        start_p=0,
        start_q=0,
        test='adf',
        max_p=5,
        max_q=5,
        m=7, # Ignored since seasonal=False
        d=None,
        D=None, # Ignored since seasonal=False
        trace=False, # Set to False for cleaner output when integrated with Streamlit
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False # Non-stepwise searches more extensively but is slower
    )
    best_p, best_d, best_q = arimax_hypertuned.order

    # Walk-forward ARIMAX forecasting
    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, predicted_values = [], []

    for train_idx, test_idx in tscv.split(msft_df):
        # Ensure X_train and X_test are derived from X_floats for consistency
        X_train, X_test = X_floats.iloc[train_idx], X_floats.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # For forecasting 1 step ahead, we need the exogenous variable for *that specific future step*
        # which is the first row of X_test in this walk-forward setup.
        exog_forecast = X_test.iloc[[0]]

        model = ARIMA(endog=y_train, exog=X_train, order=(best_p, best_d, best_q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1, exog=exog_forecast)

        actual_values.append(y_test.iloc[0])
        predicted_values.append(forecast.values[0])

    mse = mean_squared_error(actual_values, predicted_values)
    # Consistent return format for results_df, including the ARIMAX order
    results_df = pd.DataFrame({"ARIMAX_MSE": [mse], "ARIMAX_Order": [f"({best_p},{best_d},{best_q})"]})
    
    # Return 4 values: results_df, actual_values, predicted_values, and the order tuple
    return results_df, actual_values, predicted_values, (best_p, best_d, best_q)
