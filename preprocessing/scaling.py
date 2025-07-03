# In your scaling.py file
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 


from preprocessing.imputation import impute_features

def handle_infinite_values(df):
    """
    Detects and replaces infinite values with NaN, then imputes missing values.
    This function now correctly handles mixed data types by applying checks only to numeric columns.

    Parameters:
        df (pd.DataFrame): DataFrame to sanitize and impute.

    Returns:
        pd.DataFrame: Cleaned DataFrame with infinite values handled and imputed.
    """
    df_copy = df.copy()

    # Identify numeric columns
    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    # Check for infinite values ONLY in numeric columns
    if not numeric_cols.empty and np.isinf(df_copy[numeric_cols]).any().any(): 
        print("Infinite values detected in numeric columns. Replacing with NaN and imputing...")
        # Replace infinite values with NaN in numeric columns only
        df_copy[numeric_cols] = df_copy[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        df_copy = impute_features(df_copy)
    else:
        print("No infinite values detected in numeric columns.")

    return df_copy



def scale_features(df, target_col=None, feature_range=(0, 1)):
    """
    Scales feature columns and optionally a target column using MinMaxScaler.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str or None): If provided, returns X_scaled and y_scaled separately.
        feature_range (tuple): Range for MinMax scaling.

    Returns:
        If target_col:
            (X_scaled, y_scaled, X_scaler, y_scaler)
        Else:
            scaled_df, scaler
    """
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_scaler = MinMaxScaler(feature_range=feature_range)
        y_scaler = MinMaxScaler(feature_range=feature_range)

        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

        return X_scaled, y_scaled, X_scaler, y_scaler

    else:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_array = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
        return scaled_df, scaler

