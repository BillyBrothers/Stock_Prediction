import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings

def run_arima_forecast(msft_df, target_col='Close', n_splits=100):
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

    # Suppress specific statsmodels warnings 
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

    for i, (train_idx, test_idx) in enumerate(tscv.split(msft_df)):

        print("Fold", i)
    
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) <= best_d:
            warnings.warn(f"Training data length ({len(y_train)}) is too short for differencing order d={best_d}. Skipping this split.")
            continue

        model = ARIMA(y_train, order=(best_p, best_d, best_q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1) # returns a 1d array of the single predicted value based on the prior training data

        print(f"  Actual value: {y_test.iloc[0]:.8f}")
        print(f"  Predicted level: {forecast.iloc[0]}")

        actual_values.append(y_test.iloc[0])
        predicted_values.append(forecast.values[0]) # forecast[0] would work because its a 1d numpy array, but forecast.values[0] is a robust approach in the event it was a pandas series.

    mse = mean_squared_error(actual_values, predicted_values)
    
    results_df = pd.DataFrame({"Actual_Vs_Predicted": [mse], "ARIMA_Order": [f"({best_p},{best_d},{best_q})"]})

    return results_df, actual_values, predicted_values, (best_p, best_d, best_q)
