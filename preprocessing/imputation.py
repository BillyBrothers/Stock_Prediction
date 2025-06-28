def trim_nans_by_window(df, window_sizes):
    """
    Drops initial rows based on the maximum rolling window size to ensure clean feature computation.

    Parameters:
        df (pd.DataFrame): Input DataFrame with rolling/lagged features.
        window_sizes (list of int): List of rolling window sizes used across features.

    Returns:
        pd.DataFrame: Trimmed DataFrame with leading NaNs removed.
    """
    max_window = max(window_sizes)
    print(f"Amount of NaNs before accounting for largest window size: {df.isna().sum().sum()}")
    trimmed_df = df.iloc[max_window:].copy()
    print(f"Trimmed first {max_window} rows to avoid NaNs from rolling features.")
    print(f"Remaining rows: {trimmed_df.shape[0]}")
    return trimmed_df

def impute_features(df):
    """
    Applies feature-specific imputation strategies to a DataFrame.

    Logic:
    - Skips 'Close' column (assumed target or baseline).
    - Forward fills values for lagged, rolling, volume-related, or log-derived features.
    - Fills technical indicators using expanding median, preserving signal stability.
    - Interpolates price action features for smoother transitions.
    - Applies general fallback imputation (linear + ffill + bfill) to any unclassified features.

    Parameters:
        df (pd.DataFrame): DataFrame with engineered feature columns.

    Returns:
        pd.DataFrame: Imputed DataFrame with missing values filled based on feature type.
    """
    df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        if col_lower == 'close':
            continue

        # Lagged, rolling, volatility, log, and volume features
        if any(key in col_lower for key in [
            "price_lag", "hourly_return_lag", "daily_return_lag", "weekly_return_lag_",
            "sma", "ema", "volume", "rolling_log_avg", "rolling_log_std", "volatility"
        ]):
            df[col] = df[col].ffill()

        # Technical indicators
        elif any(key in col_lower for key in [
            "atr", "rsi", "macd", "bollinger", "upper_bb", "lower_bb",
            "adx", "di", "%d", "%k", "obv"
        ]):
            df[col] = df[col].fillna(df[col].expanding().median())

        # Price structure features
        elif any(key in col_lower for key in [
            "low_range", "close_range", "range_pct", "rsi_x_volume"
        ]):
            df[col] = df[col].interpolate(method='linear')

        # General fallback
        else:
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

    return df
