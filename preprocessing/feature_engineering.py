import pandas as pd

def add_lag_prices(df: pd.DataFrame, columns=None, lag_periods=None):
    """
    Adds lagged price features for specified columns over defined periods.

    Parameters:
        df (pd.DataFrame): Input DataFrame with datetime index.
        columns (list of str): Price columns to lag. Defaults to OHLCV.
        lag_periods (list of int): Number of lag periods (e.g., in hours) to apply.

    Returns:
        pd.DataFrame: DataFrame with new lag features added.
    """
    df = df.copy()
    n_rows = len(df)

    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if lag_periods is None:
        lag_periods = [1, 2, 3, 4, 5, 6]

    # Auto-adjust lag periods to be valid
    valid_lags = [lag for lag in lag_periods if lag < n_rows]
    skipped_lags = [lag for lag in lag_periods if lag >= n_rows]

    if skipped_lags:
        print(f"⚠️ Skipped lag periods {skipped_lags} due to insufficient data (only {n_rows} rows).")

    for col in columns:
        for lag in valid_lags:
            df[f'{col}_lag_{lag}H'] = df[col].shift(lag)

    return df

def add_lagged_returns(df: pd.DataFrame, columns=None, frequency="hourly", lags=None) -> pd.DataFrame:
    """
    Adds lagged percentage returns for specified columns and frequencies,
    with auto-adjustment for lag values based on data length.

    Parameters:
        df (pd.DataFrame): DataFrame with price data.
        columns (list of str): Columns to compute returns on. Defaults to ['Close'].
        frequency (str): One of 'hourly', 'daily', or 'weekly'.
        lags (list of int): Lag steps to shift returns by.

    Returns:
        pd.DataFrame: DataFrame with lagged return features added.
    """
    df = df.copy()
    n_rows = len(df)

    freq_map = {
        "hourly": {"periods": 1, "label": "H", "default_lags": [1, 2, 3, 4, 5, 6]},
        "daily": {"periods": 24, "label": "D", "default_lags": [1, 2, 3, 4, 5]},
        "weekly": {"periods": 24 * 5, "label": "W", "default_lags": [1, 2, 3]},
    }

    if frequency not in freq_map:
        raise ValueError("Frequency must be 'hourly', 'daily', or 'weekly'")

    periods = freq_map[frequency]["periods"]
    label = freq_map[frequency]["label"]
    if columns is None:
        columns = ['Close']
    if lags is None:
        lags = freq_map[frequency]["default_lags"]

    valid_lags = [lag for lag in lags if lag < n_rows]
    skipped_lags = [lag for lag in lags if lag >= n_rows]

    if skipped_lags:
        print(f"⚠️ Skipped lag periods {skipped_lags} due to insufficient data (only {n_rows} rows).")

    for col in columns:
        return_col = f"{frequency.capitalize()}_{col}_return"
        df[return_col] = df[col].pct_change(periods=periods)

        for lag in valid_lags:
            df[f"{return_col}_lag_{lag}{label}"] = df[return_col].shift(lag)

    return df

import pandas as pd
import numpy as np

def add_moving_averages(
    df: pd.DataFrame,
    method: str = "SMA",
    windows: list = None,
    column: str = "Close"
) -> pd.DataFrame:
    """
    Adds moving average features to the DataFrame with auto-adjusted windows.

    Parameters:
        df (pd.DataFrame): Input stock price data.
        method (str): Type of moving average ("SMA", "EMA", "LOG").
        windows (list): List of window sizes. Defaults to [5, 7, 10, 21, 28, 35].
        column (str): Column on which to compute the moving averages.

    Returns:
        pd.DataFrame: DataFrame with added moving average features.
    """
    import numpy as np  # Make sure np is imported
    df = df.copy()
    n_rows = len(df)

    if windows is None:
        windows = [5, 7, 10, 21, 28, 35]

    valid_windows = [w for w in windows if w < n_rows]
    skipped = [w for w in windows if w >= n_rows]
    if skipped:
        print(f"⚠️ Skipped window sizes {skipped} due to insufficient data (only {n_rows} rows).")

    method = method.upper()
    if method == "SMA":
        for w in valid_windows:
            df[f"SMA_{w}"] = df[column].rolling(window=w).mean()

    elif method == "EMA":
        for w in valid_windows:
            df[f"EMA_{w}"] = df[column].ewm(span=w, adjust=False).mean()

    elif method == "LOG":
        log_col = f"Log_{column}"
        df[log_col] = np.log(df[column])
        for w in valid_windows:
            df[f"Rolling_Log_Avg_{w}"] = df[log_col].rolling(window=w).mean()
            df[f"Rolling_Log_Std_{w}"] = df[log_col].rolling(window=w).std()

    else:
        raise ValueError("Invalid method. Choose from 'SMA', 'EMA', or 'LOG'.")

    return df

def compute_rolling_stddev(
    df: pd.DataFrame,
    target_col: str = "Close",
    windows: list = None
) -> pd.DataFrame:
    """
    Computes rolling standard deviations for a list of window sizes,
    with validation based on available data length.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): Column to calculate volatility on. Defaults to 'Close'.
        windows (list): List of rolling window sizes. Defaults to [7].

    Returns:
        pd.DataFrame: DataFrame with added volatility columns.
    """
    df = df.copy()
    n_rows = len(df)

    if windows is None:
        windows = [7]

    valid_windows = [w for w in windows if w < n_rows]
    skipped_windows = [w for w in windows if w >= n_rows]

    if skipped_windows:
        print(f"⚠️ Skipped window sizes {skipped_windows} due to insufficient data (only {n_rows} rows).")

    for w in valid_windows:
        col_name = f"Volatility_StdDEV_{target_col}_{w}"
        df[col_name] = df[target_col].rolling(window=w).std()

    return df

def add_technical_indicators(df: pd.DataFrame, windows = None) -> pd.DataFrame:
    """
    Enhances a financial DataFrame with multiple technical indicators,
    automatically skipping invalid window sizes based on available data.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns:
        pd.DataFrame: Copy enriched with technical analysis features.
    """
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, ADXIndicator

    df = df.copy()
    n_rows = len(df)
    applied = []
    skipped = []

    if windows is None:
        windows = [5,7,10,14,21] # fallback default 

     # --- ATR ---
    for period in windows:
        if period < n_rows:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=period)
            df[f'ATR_{period}'] = atr.average_true_range()
            applied.append(f'ATR_{period}')
        else:
            skipped.append(f'ATR_{period}')

    # --- RSI (requires window=14) ---
    if 14 in windows and 14 < n_rows:
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI_14'] = rsi.rsi()
        df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
        df['RSI_x_Volume'] = df['RSI_14'] * df['Volume']
        applied.append("RSI_14")

    # --- MACD (requires 26, 12, 9) ---
    if all(w in windows and w < n_rows for w in [26, 12, 9]):
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        df['MACD_Prev'] = df['MACD'].shift(1).fillna(method='bfill')
        df['MACD_Signal_Prev'] = df['MACD_Signal'].shift(1).fillna(method='bfill')
        df['MACD_Cross_Up'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD_Prev'] <= df['MACD_Signal_Prev'])).astype(int)
        df['MACD_Cross_Down'] = ((df['MACD'] < df['MACD_Signal']) & (df['MACD_Prev'] >= df['MACD_Signal_Prev'])).astype(int)
        applied.extend(['MACD', 'MACD_Signal', 'MACD_Cross_Up', 'MACD_Cross_Down'])

    # --- Bollinger Bands (window=2) ---
    if 2 in windows and 2 < n_rows:
        bb = BollingerBands(close=df['Close'], window=2, window_dev=2)
        df['Bollinger_Lower'] = bb.bollinger_lband()
        df['Bollinger_Upper'] = bb.bollinger_hband()
        df['Bollinger_Middle'] = bb.bollinger_mavg()
        df['Bollinger_Bandwidth'] = bb.bollinger_wband() / 100
        df['Bollinger_PercentageB'] = bb.bollinger_pband()
        df['Price_Above_Upper_BB'] = (df['Close'] > df['Bollinger_Upper']).astype(int)
        df['Price_Below_Lower_BB'] = (df['Close'] < df['Bollinger_Lower']).astype(int)
        applied.append('Bollinger_Bands')

    # --- Stochastic Oscillator (requires 14 and 3) ---
    if all(w in windows and w < n_rows for w in [14, 3]):
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            df['%K'] = stoch.stoch()
            df['%D'] = stoch.stoch_signal()
            applied.extend(['Stochastic_%K', 'Stochastic_%D'])
        except Exception as e:
            skipped.append(f"Stochastic_Oscillator (Error: {e})")

    # --- ADX (window=14) ---
    
    if 14 in windows and 14 < n_rows:
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ADX_14'] = adx.adx().reindex(df.index)
            df['Positive_DI'] = adx.adx_pos().reindex(df.index)
            df['Negative_DI'] = adx.adx_neg().reindex(df.index)
            df['Trend_Strong_ADX'] = (df['ADX_14'].fillna(0) > 25).astype(int)
            applied.extend(['ADX_14', 'Positive_DI', 'Negative_DI', 'Trend_Strong_ADX'])
        except Exception as e:
            skipped.append(f"ADX (Error: {e})")

    # --- Summary ---
    print(f"✅ Applied indicators: {applied}")
    print(f"⚠️ Skipped due to data limits or config: {skipped}")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and encodes time-based features from the DataFrame index.
    Includes basic, cyclical, and one-hot encodings — with validation.

    Parameters:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.

    Returns:
        pd.DataFrame: Enriched DataFrame with time features added.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Validation
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("❌ The DataFrame index must be a pd.DatetimeIndex to extract time features.")

    n_rows = len(df)
    if n_rows == 0:
        print("⚠️ The DataFrame is empty. No time features were added.")
        return df

    # Basic time components
    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Week_of_Year'] = df.index.isocalendar().week.astype(int)

    # Cyclical encodings
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

    # One-hot encodings (preserve complete cycles)
    df = pd.get_dummies(df, columns=['Day_of_Week'], prefix='Day', drop_first=False)
    df = pd.get_dummies(df, columns=['Month'], prefix='Month', drop_first=False)

    return df

def add_volume_features(
    df: pd.DataFrame,
    volume_sma_windows: list = None,
    obv_ema_span: int = 9
) -> pd.DataFrame:
    """
    Adds volume-based technical indicators to a DataFrame with window validation.

    Features:
    - Volume SMAs over selected windows
    - Volume percentage change
    - On-Balance Volume (OBV) with EMA smoothing

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
        volume_sma_windows (list): List of window sizes for volume SMAs. Defaults to [7, 14, 21, 28, 35].
        obv_ema_span (int): Span for OBV EMA smoothing. Default is 9.

    Returns:
        pd.DataFrame: Enriched DataFrame with volume-derived features.
    """
    import pandas as pd
    from ta.volume import OnBalanceVolumeIndicator

    df = df.copy()
    n_rows = len(df)

    # Default windows
    if volume_sma_windows is None:
        volume_sma_windows = [7, 14, 21, 28, 35]

    valid_windows = [w for w in volume_sma_windows if w < n_rows]
    skipped_windows = [w for w in volume_sma_windows if w >= n_rows]

    if skipped_windows:
        print(f"⚠️ Skipped volume SMA windows {skipped_windows} due to insufficient data ({n_rows} rows).")

    # Volume SMAs
    for w in valid_windows:
        df[f'Volume_SMA_{w}H'] = df['Volume'].rolling(window=w).mean()

    # Volume percentage change
    df['Volume_Change'] = df['Volume'].pct_change()

    # OBV and its EMA
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'], fillna=False)
    df['OBV'] = obv.on_balance_volume()

    if obv_ema_span < n_rows:
        df[f'OBV_EMA_{obv_ema_span}H'] = df['OBV'].ewm(span=obv_ema_span, adjust=False).mean()
    else:
        print(f"⚠️ Skipped OBV EMA: span {obv_ema_span} exceeds available data ({n_rows} rows).")

    return df


def add_price_features(
    df: pd.DataFrame,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Adds price-derived candle structure features with optional normalization.

    Features:
    - High-Low absolute range
    - Open-Close absolute range
    - (Optional) Percentage versions of High-Low and Open-Close ranges

    Parameters:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Open', 'Close' columns.
        normalize (bool): Whether to include normalized (%) variants.

    Returns:
        pd.DataFrame: Enriched with price structure features.
    """
    df = df.copy()
    n_rows = len(df)

    # Validate minimum required structure
    required_cols = ['High', 'Low', 'Open', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")

    if n_rows < 2:
        print(f"⚠️ Only {n_rows} row(s) available — price features added, but percentage normalization may be unstable.")

    # Core candle ranges
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Open_Close_Range'] = df['Close'] - df['Open']

    if normalize:
        # Avoid division warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            df['High_Low_Range_Pct'] = ((df['High'] - df['Low']) / df['Close'].replace(0, np.nan)) * 100
            df['Open_Close_Range_Pct'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)) * 100

    return df

