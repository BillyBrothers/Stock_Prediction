import pandas as pd
import numpy as np

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
        "daily": {"periods": 7, "label": "D", "default_lags": [1, 2, 3, 4, 5]},
        "weekly": {"periods": 7 * 5, "label": "W", "default_lags": [1, 2, 3]},
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



def add_moving_averages(
    df: pd.DataFrame,
    method: str = "SMA",
    windows: list = None,
    column: str = "Close",
    lag_periods: list = None
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
    df = df.copy()
    n_rows = len(df)

    if windows is None:
        windows = [5, 7, 10, 21, 28, 35]
    if lag_periods is None:
        lag_periods = [1, 2, 3, 4, 5, 6, 7]

    valid_windows = [w for w in windows if w < n_rows]
    skipped = [w for w in windows if w >= n_rows]
    if skipped:
        print(f"⚠️ Skipped window sizes {skipped} due to insufficient data (only {n_rows} rows).")

    method = method.upper()
    ma_cols_created = []

    if method == "SMA":
        for w in valid_windows:
            base_ma_col = f"SMA{w}"
            df[base_ma_col] = df[column].rolling(window=w).mean()
            ma_cols_created.append(base_ma_col)

            for lag in lag_periods:
                df[f"{base_ma_col}_lag_{lag}"] = df[base_ma_col].shift(lag)

    elif method == "EMA":
        for w in valid_windows:
            base_ma_col = f"EMA{w}"
            df[base_ma_col] = df[column].rolling(window=w).mean()
            ma_cols_created.append(base_ma_col)
            
            for lag in lag_periods:
                df[f"{base_ma_col}_lag_{lag}"] = df[base_ma_col].shift(lag)


    elif method == "LOG":
        log_col = f"Log_{column}"
        df[log_col] = np.log(df[column])
        for w in valid_windows:
            base_log_avg_col = f"Rolling_Log_Avg_{w}"
            df[base_log_avg_col] = df[log_col].rolling(window=w).mean()
            ma_cols_created.append(base_log_avg_col)

            base_log_std_col = f"Rolling_Log_Std_{w}"
            df[base_log_std_col] = df[log_col].rolling(window=w).std()
            ma_cols_created.append(base_log_std_col)

            for lag in lag_periods:
                df[f"{base_log_avg_col}_lag_{lag}"] = df[base_log_avg_col].shift(lag)
                df[f"{base_log_std_col}_lag_{lag}"] = df[base_log_std_col].shift(lag)

    else:
        raise ValueError("Invalid method. Choose from 'SMA', 'EMA', or 'LOG'.")

    return df

def compute_rolling_stddev(
    df: pd.DataFrame,
    target_col: str = "Close",
    windows: list = None,
    lag_periods: list = None
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
    
    if lag_periods is None:
        lag_periods = [1,2,3,4,5,6,7]

    valid_windows = [w for w in windows if w < n_rows]
    skipped_windows = [w for w in windows if w >= n_rows]

    if skipped_windows:
        print(f"⚠️ Skipped window sizes {skipped_windows} due to insufficient data (only {n_rows} rows).")

    for w in valid_windows:
        base_stddev_col = f"Volatility_StdDev_{target_col}_{w}"
        df[base_stddev_col] = df[target_col].rolling(window=w).std()
        
        for lag in lag_periods:
            df[f"{base_stddev_col}_lag_{lag}"] = df[base_stddev_col].shift(lag)

    return df

def add_technical_indicators(
        df: pd.DataFrame, 
        windows: list = None,
        lag_periods: list = None
        ) -> pd.DataFrame:
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

    if lag_periods is None:
        lag_periods = [1]

     # --- ATR ---
    for period in windows:
        if period < n_rows:
            base_atr_col = f'ATR_{period}'
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=period)
            df[base_atr_col] = atr.average_true_range()
            applied.append(base_atr_col)

            for lag in lag_periods:
                df[f'{base_atr_col}_lag_{lag}'] = df[base_atr_col].shift(lag)

        else:
            skipped.append(f'ATR_{period}')

    # --- RSI (requires window=14) ---
    if 14 in windows and 14 < n_rows:
        base_rsi_col = 'RSI_14'
        rsi = RSIIndicator(close=df['Close'], window=14)
        df[base_rsi_col] = rsi.rsi()
        applied.append(base_rsi_col)
        for lag in lag_periods:
            df[f'{base_rsi_col}_lag_{lag}'] = df[base_rsi_col].shift(lag)

        
        base_rsi_overbought = 'RSI_Overbought'
        df[base_rsi_overbought] = (df[base_rsi_col] > 70).astype(int)
        applied.append(base_rsi_overbought)
        for lag in lag_periods:
            df[f'{base_rsi_overbought}_lag_{lag}'] = df[base_rsi_overbought].shift(lag)

        base_rsi_Oversold = 'RSI_Oversold'
        df[base_rsi_Oversold] = (df[base_rsi_col] < 30).astype(int)
        applied.append(base_rsi_Oversold)
        for lag in lag_periods:
            df[f'{base_rsi_Oversold}_lag_{lag}'] = df[base_rsi_Oversold].shift(lag)


        base_rsi_x_volume = 'RSI_x_Volume'
        df[base_rsi_x_volume] = df[base_rsi_col] * df['Volume']
        applied.append(base_rsi_x_volume)
        for lag in lag_periods:
            df[f'{base_rsi_x_volume}_lag_{lag}'] = df[base_rsi_x_volume].shift(lag)


    # --- MACD (requires 26, 12, 9) ---
    if all(w in windows and w < n_rows for w in [26, 12, 9]):
        base_macd_col = 'MACD'
        base_macd_signal_col = 'MACD_Signal'

        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df[base_macd_col] = macd.macd()
        df[base_macd_signal_col] = macd.macd_signal()
        applied.extend(base_macd_col, base_macd_signal_col)

        for lag in lag_periods:
            df[f'{base_macd_col}_lag_{lag}'] = df[base_macd_col].shift(lag)
            df[f'{base_macd_signal_col}_lag_{lag}'] = df[base_macd_signal_col].shift(lag)
        
        df['MACD_PREV'] = df[base_macd_col].shift(1).fillna(method = "ffill")
        df['MACD_Signal_Prev'] = df[base_macd_signal_col].shift(1).fillna(method='ffill')

        base_macd_cross_up = 'MACD_Cross_Up'
        df[base_macd_cross_up] = ((df[base_macd_col] > df[base_macd_signal_col]) & (df['MACD_Prev'] <= df['MACD_Signal_Prev'])).astype(int)
        applied.append(base_macd_cross_up)

        base_macd_cross_down = 'MACD_Cross_Down'
        df[base_macd_cross_down] = ((df[base_macd_col] < df[base_macd_signal_col]) & (df['MACD_Prev'] >= df['MACD_Signal_Prev'])).astype(int)
        applied.append(base_macd_cross_down)

        for lag in lag_periods:
            df[f'{base_macd_cross_up}_lag_{lag}'] = df[base_macd_cross_up].shift(lag)
            df[f'{base_macd_cross_down}_lag_{lag}'] = df[base_macd_cross_down].shift(lag)

        
    # --- Bollinger Bands (window=2) ---
    if 2 in windows and 2 < n_rows:
        bb = BollingerBands(close=df['Close'], window=2, window_dev=2)

        base_bb_lower = 'Bollinger_Lower'
        df[base_bb_lower] = bb.bollinger_lband()
        applied.append(base_bb_lower)
        for lag in lag_periods:
            df[f'{base_bb_lower}_lag_{lag}'] = df[base_bb_lower].shift(lag)

        base_bb_upper = 'Bollinger_Upper'
        df[base_bb_upper] = bb.bollinger_hband()
        applied.append(base_bb_upper)
        for lag in lag_periods:
            df[f'{base_bb_upper}_lag_{lag}'] = df[base_bb_upper].shift(lag)

        base_bb_middle = 'Bollinger_Middle'
        df[base_bb_middle] = bb.bollinger_mavg()
        applied.append(base_bb_middle)
        for lag in lag_periods:
            df[f'{base_bb_middle}_lag_{lag}'] = df[base_bb_middle].shift(lag)
        
        base_bb_bandwidth = 'Bollinger_Bandwidth'
        df[base_bb_bandwidth] = bb.bollinger_wband() / 100
        applied.append(base_bb_bandwidth)
        for lag in lag_periods:
            df[f'{base_bb_bandwidth}_lag_{lag}'] = df[base_bb_bandwidth].shift(lag)
    
        base_bb_percentageb = 'Bollinger_PercentageB'
        df[base_bb_percentageb] = bb.bollinger_pband()
        applied.append(base_bb_percentageb)
        for lag in lag_periods:
            df[f'{base_bb_percentageb}_lag_{lag}'] = df[base_bb_percentageb].shift(lag)

        base_price_above_upper_bb = 'Price_Above_Upper_BB'
        df[base_price_above_upper_bb] = (df['Close'] > df[base_bb_lower]).astype(int)
        applied.append(base_price_above_upper_bb)
        for lag in lag_periods:
            df[f'{base_price_above_upper_bb}_lag_{lag}'] = df[base_price_above_upper_bb].shift(lag)

        base_price_below_lower_bb = 'Price_Below_Lower_BB'
        df[base_price_below_lower_bb] = (df['Close'] < df[base_bb_lower]).astype(int)
        applied.append(base_price_below_lower_bb)
        for lag in lag_periods:
            df[f'{base_price_below_lower_bb}_lag_{lag}'] = df[base_price_below_lower_bb].shift(lag)

    # --- Stochastic Oscillator (requires 14 and 3) ---
    if all(w in windows and w < n_rows for w in [14, 3]):
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            base_k_col = "%K"
            df[base_k_col] = stoch.stoch()
            applied.append(base_k_col)
            for lag in lag_periods:
                df[f'{base_k_col}_lag_{lag}'] = df[base_k_col].shift(lag)

            base_d_col = "%D"
            df[base_d_col] = stoch.stoch.stoch_signal()
            applied.append(base_d_col)
            for lag in lag_periods:
                df[f'{base_d_col}_lag_{lag}'] = df[base_d_col].shift(lag)
            
        except Exception as e:
            skipped.append(f"Stochastic_Oscillator (Error: {e})")

    # --- ADX (window=14) ---
    
    if 14 in windows and 14 < n_rows:
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)

            base_adx_col = 'ADX_14'
            df[base_adx_col] = adx.adx().reindex(df.index)
            applied.append(base_adx_col)
            for lag in lag_periods:
                df[f'{base_adx_col}_lag_{lag}'] = df[base_adx_col].shift(lag)

            base_pos_di_col = 'Positive_DI'
            df[base_pos_di_col] = adx.adx_pos().reindex(df.index)
            applied.append(base_pos_di_col)
            for lag in lag_periods:
                df[f'{base_pos_di_col}_lag_{lag}'] = df[base_pos_di_col].shift(lag)

            base_neg_di_col = 'Negative_DI'
            df[base_neg_di_col] = adx.adx_neg().reindex(df.index)
            applied.append(base_neg_di_col)
            for lag in lag_periods:
                df[f'{base_neg_di_col}_lag_{lag}'] = df[base_neg_di_col].shift(lag)

            base_trend_strong_col = 'Trend_Strong_ADX'
            df[base_trend_strong_col] = (df[base_adx_col].fillna(0) > 25).astype(int)
            applied.append(base_trend_strong_col)
            for lag in lag_periods:
                df[f'{base_trend_strong_col}_lag_{lag}'] = df[base_trend_strong_col].shift(lag)

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

