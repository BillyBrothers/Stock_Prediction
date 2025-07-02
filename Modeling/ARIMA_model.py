import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings

def run_arima_forecast(msft_df, target_col='Close', n_splits=3):
    """
    Runs an ARIMA time series forecast with hyperparameter tuning via auto_arima
    and walk-forward validation.

    Parameters:
        msft_df (pd.DataFrame): DataFrame containing the target_col.
        target_col (str): The name of the target column (e.g., 'Close').
        n_splits (int): Number of splits for TimeSeriesSplit cross-validation.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the ARIMA Mean Squared Error and Order.
            - list: List of actual values for the first step of each test split.
            - list: List of predicted values for the first step of each test split.
            - tuple: The (p,d,q) order found by auto_arima.
    """
    y = msft_df[target_col]

    # Suppress specific statsmodels warnings that can be verbose during fitting
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Hyperparameter tuning with auto_arima on the full series
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
        trace=False,
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
    # MODIFIED: Include ARIMA_Order in results_df
    results_df = pd.DataFrame({"ARIMA_MSE": [mse], "ARIMA_Order": [f"({best_p},{best_d},{best_q})"]})

    # MODIFIED: Return the order as the fourth item in the tuple
    return results_df, actual_values, predicted_values, (best_p, best_d, best_q)
