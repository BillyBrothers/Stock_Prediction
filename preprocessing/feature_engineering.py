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
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if lag_periods is None:
        lag_periods = [1, 2, 3, 4, 5, 6]

    for col in columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}H'] = df[col].shift(lag)

    return df


def add_lagged_returns(df, columns=None, frequency="hourly", lags=None):
    """
    Adds lagged percentage returns for specified columns and frequencies.

    Parameters:
        df (pd.DataFrame): DataFrame with price data.
        columns (list of str): Columns to compute returns on. Defaults to ['Close'].
        frequency (str): One of 'hourly', 'daily', or 'weekly'.
        lags (list of int): Lag steps to shift returns by.

    Returns:
        pd.DataFrame: DataFrame with lagged return features.
    """
    df = df.copy()

    freq_map = {
        "hourly": {"periods": 1, "label": "H", "default_lags": [1, 2, 3, 4, 5, 6]},
        "daily": {"periods": 24, "label": "D", "default_lags": [1, 2, 3, 4, 5]},
        "weekly": {"periods": 24 * 5, "label": "W", "default_lags": [1, 2, 3]},
    }

    if frequency not in freq_map:
        raise ValueError("Frequency must be 'hourly', 'daily', or 'weekly'")

    periods = freq_map[frequency]["periods"]
    label = freq_map[frequency]["label"]
    if lags is None:
        lags = freq_map[frequency]["default_lags"]
    if columns is None:
        columns = ['Close']

    for col in columns:
        return_col = f"{frequency.capitalize()}_{col}_return"
        df[return_col] = df[col].pct_change(periods=periods)

        for lag in lags:
            df[f"{return_col}_lag_{lag}{label}"] = df[return_col].shift(lag)

    return df


import pandas as pd
import numpy as np

def add_moving_averages(
    df: pd.DataFrame,
    method: str = "SMA",
    windows: list = [5, 7, 10, 21, 28, 35],
    column: str = "Close"
):
    """
    Adds moving average features to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input stock price data.
        method (str): Type of moving average ("SMA", "EMA", "Log").
        windows (list): Window sizes to compute the averages.
        column (str): Column on which to compute the moving averages.

    Returns:
        pd.DataFrame: DataFrame with added moving average features.
    """
    df = df.copy()

    if method.upper() == "SMA":
        for w in windows:
            df[f"SMA_{w}"] = df[column].rolling(window=w).mean()

    elif method.upper() == "EMA":
        for w in windows:
            df[f"EMA_{w}"] = df[column].ewm(span=w, adjust=False).mean()

    elif method.upper() == "LOG":
        log_col = f"Log_{column}"
        df[log_col] = np.log(df[column])
        for w in windows:
            df[f"Rolling_Log_Avg_{w}"] = df[log_col].rolling(window=w).mean()
            df[f"Rolling_Log_Std_{w}"] = df[log_col].rolling(window=w).std()
    else:
        raise ValueError("Invalid method. Choose from 'SMA', 'EMA', or 'Log'.")

    return df



def compute_rolling_stddev(df, target_col, window=7):
    """
    Computes the rolling standard deviation on the specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the column to calculate volatility on.
        window (int): The size of the rolling window.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the new rolling std dev column.
    """
    df_copy = df.copy()
    col_name = f'Volatility_StdDEV_{target_col}_{window}'
    df_copy[col_name] = df_copy[target_col].rolling(window=window).std()
    return df_copy


import pandas as pd
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator

def add_technical_indicators(df):
    """
    Enhances a financial DataFrame with key technical analysis indicators.

    This function computes and appends multiple technical indicators commonly used 
    in quantitative trading and market analysis. It operates on columns named 
    'Open', 'High', 'Low', 'Close', and assumes they are present in the input DataFrame.

    Indicators added:
    - Average True Range (ATR) over multiple periods
    - Relative Strength Index (RSI), with overbought/oversold flags and RSI x Volume interaction
    - MACD with signal line, histogram, and crossover flags
    - Bollinger Bands with bandwidth and p-band
    - Stochastic Oscillator (%K, %D)
    - ADX with Positive/Negative DI and trend strength flag

    Parameters:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', 'Volume', 'Open'

    Returns:
        pd.DataFrame: A copy enriched with technical indicator features
    """
    df = df.copy()

    # ATR
    atr_periods = [7, 14, 21, 28, 35]
    for period in atr_periods:
        atr = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'].shift(1),
            window=period,
            fillna=False
        )
        df[f'ATR_{period}H'] = atr.average_true_range()

    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI_14'] = rsi.rsi()
    df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
    df['RSI_x_Volume'] = df['RSI_14'] * df['Volume']  

    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    df['MACD_Prev'] = df['MACD'].shift(1)
    df['MACD_Signal_Prev'] = df['MACD_Signal'].shift(1)
    df['MACD_Cross_Up'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD_Prev'] <= df['MACD_Signal_Prev'])).astype(int)
    df['MACD_Cross_Down'] = ((df['MACD'] < df['MACD_Signal']) & (df['MACD_Prev'] >= df['MACD_Signal_Prev'])).astype(int)

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=2, window_dev=2, fillna=False)
    df['Bollinger_Lower'] = bb.bollinger_lband()
    df['Bollinger_Middle'] = bb.bollinger_mavg()
    df['Bollinger_Upper'] = bb.bollinger_hband()
    df['Bollinger_Bandwidth_Raw'] = bb.bollinger_wband()
    df['Bollinger_PercentageB'] = bb.bollinger_pband()
    df['Bollinger_Bandwidth'] = df['Bollinger_Bandwidth_Raw'] / 100
    df['Price_Above_Upper_BB'] = (df['Close'] > df['Bollinger_Upper']).astype(int)
    df['Price_Below_Lower_BB'] = (df['Close'] < df['Bollinger_Lower']).astype(int)

    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3,
        fillna=False
    )
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()

    # ADX
    adx = ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        fillna=False
    )
    df['ADX_14'] = adx.adx()
    df['Positive_DI'] = adx.adx_pos()
    df['Negative_DI'] = adx.adx_neg()
    df['Trend_Strong_ADX'] = (df['ADX_14'] > 25).astype(int)

    return df



import numpy as np
import pandas as pd

def add_time_features(df):
    """
    Extracts and encodes time-based features from the DataFrame index.

    Adds:
    - Hour, Day of Week, Day of Month, Month, Year, and ISO Week of Year
    - Cyclical encodings for Hour and Day of Week using sine/cosine transformation
    - One-hot encodings for Day of Week and Month
    
    Parameters:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
    
    Returns:
        pd.DataFrame: Enriched DataFrame with time features added.
    """
    df = df.copy()
    
    # Basic time features
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

    # One-hot encodings (without dropping first to preserve full cycle)
    df = pd.get_dummies(df, columns=['Day_of_Week'], prefix='Day', drop_first=False)
    df = pd.get_dummies(df, columns=['Month'], prefix='Month', drop_first=False)

    return df

from ta.volume import OnBalanceVolumeIndicator

def add_volume_features(df, volume_sma_windows=[7, 14, 21, 28, 35], obv_ema_span=9):
    """
    Adds volume-based technical features to a DataFrame.

    Features:
    - Volume SMAs across user-defined windows.
    - Volume percentage change.
    - On-Balance Volume (OBV) and its exponential moving average.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
        volume_sma_windows (list): List of window sizes for volume SMAs.
        obv_ema_span (int): Span value for EMA smoothing of OBV.

    Returns:
        pd.DataFrame: DataFrame with additional volume-derived features.
    """
    df = df.copy()

    # Customizable Volume SMAs
    for window in volume_sma_windows:
        df[f'Volume_SMA_{window}H'] = df['Volume'].rolling(window=window).mean()

    # Volume percentage change
    df['Volume_Change'] = df['Volume'].pct_change()

    # On-Balance Volume and EMA
    from ta.volume import OnBalanceVolumeIndicator
    obv = OnBalanceVolumeIndicator(
        close=df['Close'],
        volume=df['Volume'],
        fillna=False
    )
    df['OBV'] = obv.on_balance_volume()
    df[f'OBV_EMA_{obv_ema_span}H'] = df['OBV'].ewm(span=obv_ema_span, adjust=False).mean()

    return df


def add_price_features(df, normalize=True):
    """
    Adds price-derived features based on candle structure.

    Features:
    - High-Low range (absolute)
    - Open-Close range (absolute)
    - (Optional) Normalized High-Low and Open-Close ranges as percentages

    Parameters:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Open', 'Close' columns.
        normalize (bool): If True, adds % versions of the ranges.

    Returns:
        pd.DataFrame: DataFrame with new price structure features added.
    """
    df = df.copy()
    
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Open_Close_Range'] = df['Close'] - df['Open']
    
    if normalize:
        df['High_Low_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
        df['Open_Close_Range_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100

    return df
