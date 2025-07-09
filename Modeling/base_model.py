import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import streamlit as st

def run_naive_forecast(msft_df, target_col='Close', n_splits=100):
    """
    Runs a naive model with walk forward validation.

    Parameters:
        msft_df (pd.DataFrame): DataFrame containing the target_col.
        target_col (str): The name of the target column (e.g., 'Close').
        n_splits (int): Number of splits for TimeSeriesSplit cross-validation.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the Naive Mean Squared Error and Order.
            - list: List of actual values for the first step of each test split.
            - list: List of predicted values for the first step of each test split.
    """

    if len(msft_df) < n_splits + 1: 
        raise ValueError(f"Not enough data points ({len(msft_df)}) for {n_splits} splits. Consider reducing n_splits or providing more data.")


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
    results_df = pd.DataFrame({"Actual_Vs_Predicted": [mse]})
    return results_df, actual_values, naive_predictions
