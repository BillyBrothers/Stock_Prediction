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
    
    if len(msft_df) < n_splits + 1: 
        raise ValueError(f"Not enough data points ({len(msft_df)}) for {n_splits} splits. Consider reducing n_splits or providing more data.")

    
    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    
    X_floats = X.select_dtypes(include=np.number).astype(float)

    if X_floats.empty:
        raise ValueError("No numeric features found for exogenous variables (X). ARIMAX requires at least one numeric exogenous variable.")

    # Suppress specific statsmodels warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)

    arimax_hypertuned = auto_arima(
        y,
        exogenous=X_floats,
        seasonal=False,
        start_p=0,
        start_q=0,
        test='adf',
        max_p=5,
        max_q=5,
        d=None,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False
    )
    best_p, best_d, best_q = arimax_hypertuned.order 

    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values, predicted_values = [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(msft_df)):

        print("Fold", i)
    
        X_train, X_test = X_floats.iloc[train_idx], X_floats.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) <= best_d:
            warnings.warn(f"Training data length ({len(y_train)}) is too short for differencing order d={best_d}. Skipping this split.")
            continue

        if best_d > 0:
            y_train_differenced = y_train.diff(periods=best_d).dropna()
        else:
            y_train_differenced = y_train

        
        if best_d > 0:
            X_train_differenced = X_train.diff(periods=best_d).dropna()
        else:
            X_train_differenced = X_train

        
        if len(y_train_differenced) != len(X_train_differenced):
            warnings.warn(f"Length mismatch after differencing: y_train_differenced ({len(y_train_differenced)}) vs X_train_differenced ({len(X_train_differenced)})")

       # Ensure X_train_original has enough data for the differencing order for exog_forecast
        if X_train.empty:
            warnings.warn(f"X_train_original is empty for split {i+1}. Skipping this split.")
            continue

        if best_d > 0:
            if len(X_train) < best_d + 1:
                warnings.warn(f"X_train_original is too short ({len(X_train)}) for differencing order d={best_d} for exog_forecast. Skipping this split.")
                continue
            
            # The last (best_d + 1) points of X_train_original are needed to compute the 'd'th difference for X_k-1
            exog_for_diff_calc = X_train.iloc[-(best_d + 1):] # returns a sliced dataframe composed of the last 2 rows of each exogenous variable for each fold based on best_d size.
            exog_forecast = exog_for_diff_calc.diff(periods=best_d).iloc[[-1]] # applies differencing indepedently for each column and then selects the last row because the first row will be a nan due to the nature of differncing with only containing two rows. 

            if exog_forecast.empty: 
                warnings.warn(f"Differenced exog_forecast became empty for split {i+1} despite enough data. Skipping this split.")
                continue

        else: 
            exog_forecast = X_train.iloc[[-1]] 

        if y_train_differenced.empty or X_train_differenced.empty:
            warnings.warn("Differenced training data (y or X) became empty. Skipping this split.")
            continue

        try:
            model = ARIMA(endog=y_train_differenced, exog=X_train_differenced, order=(best_p, 0, best_q)) # we've already manually differenced earlier, so d is set to 0. 
            model_fit = model.fit()

            # using the last known differenced value of X to predict the next differenced value of y. 

            forecast_diff_series = model_fit.forecast(steps=1, exog=exog_forecast)
            forecast_diff_value = forecast_diff_series.iloc[0]

            print(f"   y_train_original.iloc[-1]: {y_train.iloc[-1]:.8f}") 
            print(f"   Forecasted difference (forecast_diff_value): {forecast_diff_value:.8f}") 

        # Taking  the last known values of y, add the predicted change, and reconstruct the next value. 
        # This process is inverse transforming the data back to original scale from differenced to absolute price.
            predicted_level = None
            if best_d > 0:
                last_original_y_values = y_train.iloc[-best_d:]
                temp_series_for_inverse = np.concatenate([
                    last_original_y_values.values, # taking the known values of y 
                    np.array([forecast_diff_value]) # the predicted differenced price 
                ])
                predicted_level = np.cumsum(temp_series_for_inverse)[-1] # summing them and then taking the last value
            else:
                predicted_level = forecast_diff_value

            print(f"   Actual value (y_test.iloc[0]): {y_test.iloc[0]:.8f}")
            print(f"   Predicted level (after inverse transform): {predicted_level:.8f}") 
            actual_values.append(y_test.iloc[0])
            predicted_values.append(predicted_level)

        except Exception as e:
            warnings.warn(f"Error during ARIMA fitting or forecasting for split {i+1}: {e}. Skipping this split.")
            continue

    if not actual_values or not predicted_values:
        raise ValueError("No valid predictions were generated across any splits. This might be due to insufficient data or persistent model fitting errors.")

    mse = mean_squared_error(actual_values, predicted_values)
    print(f"\nFinal ARIMAX MSE: {mse}") 


    results_df = pd.DataFrame({"Actual_Vs_Predicted": [mse], "ARIMA_Order": [f"({best_p},{best_d},{best_q})"]})

    return results_df, actual_values, predicted_values, (best_p, best_d, best_q)
