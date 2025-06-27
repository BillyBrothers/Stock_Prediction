import numpy as np
from sklearn.preprocessing import MinMaxScaler

def handle_infinite_values(df):
    """
    Detects and replaces infinite values with NaN, then imputes missing values.

    Parameters:
        df (pd.DataFrame): DataFrame to sanitize and impute.

    Returns:
        pd.DataFrame: Cleaned DataFrame with infinite values handled and imputed.
    """
    df = df.copy()
    
    if np.isinf(df.values).any():
        print("Infinite values detected. Replacing with NaN and imputing...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = impute_features(df)
    else:
        print("No infinite values detected.")
    
    return df


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

